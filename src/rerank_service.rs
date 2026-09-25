// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Asking a real cross-encoder to rerank, over HTTP.
//!
//! # What was missing
//!
//! This crate had three rerankers and not one of them used a model. The best of
//! them, [`CrossEncoderReranker`](crate::reranker::CrossEncoderReranker), scores
//! Jaccard overlap of word sets by default — it reorders semantic results by
//! literal vocabulary, which is what the keyword search already did.
//!
//! Meanwhile `llama-server` — the same binary this crate already talks to as
//! [`AiProvider::LlamaCpp`](crate::AiProvider) — accepts
//! `--rerank` and serves `POST /v1/rerank`. So a genuine cross-encoder needed no
//! new Rust dependency and no inference code: it needed a client.
//!
//! # Bi-encoder, cross-encoder
//!
//! An embedding is a **bi-encoder**: query and document are encoded separately
//! and their vectors compared, so the document never saw the question. Cheap,
//! because documents are encoded once and stored — which is what makes search
//! over a large corpus possible at all.
//!
//! A **cross-encoder** feeds `query + document` through the model *together* and
//! returns one number. Seeing both at once is what lets it handle paraphrase,
//! negation and word order. It costs one forward pass **per candidate**, so it
//! cannot search; it can only reorder what search already found.
//!
//! Bi-encoder to retrieve, cross-encoder to rerank. They are two stages, not two
//! options.
//!
//! # The wire format
//!
//! llama.cpp, Jina, Cohere and TEI all speak approximately the same shape, which
//! is why one client covers them:
//!
//! ```text
//! POST /v1/rerank
//! {"model": "...", "query": "...", "documents": ["...", "..."], "top_n": 5}
//!
//! {"results": [{"index": 1, "relevance_score": 0.91},
//!              {"index": 0, "relevance_score": 0.22}]}
//! ```
//!
//! # What this refuses to do
//!
//! **It never falls back to a heuristic.** If the service is unreachable or
//! answers something unusable, [`RerankService::rerank`] returns an error and the
//! caller decides. Quietly scoring word overlap instead would produce a plausible
//! ordering with no way to tell it apart from a real one — which is the defect
//! this module exists to remove, not to relocate. See N55.
//!
//! **It validates every index.** A service that returns an index outside the
//! document list is a protocol error, not something to clamp. There is an open
//! llama.cpp issue (ggml-org/llama.cpp#16407) reporting wrong rerank output for
//! several models, so the response is treated as untrusted input.
//!
//! **It reports partial answers.** If the service scores fewer documents than it
//! was given, [`Reranking::unscored`] says how many, and those documents keep
//! their original relative order at the end of the list. Silently dropping them
//! is what `rag_methods::LlmReranker` did — ten documents in, five out, from a
//! method called `rerank`.

use std::collections::HashSet;
use std::time::Duration;

/// Why a reranking attempt produced nothing usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankError {
    /// The request never completed: connection refused, timeout, DNS, 5xx.
    Transport(String),
    /// A reply arrived and was not JSON, or not the expected shape.
    Unusable(String),
    /// The service scored a document that was not in the request.
    IndexOutOfRange {
        /// The index the service returned.
        got: usize,
        /// How many documents were sent.
        sent: usize,
    },
    /// Nothing to do: no documents were supplied.
    NoDocuments,
}

impl std::fmt::Display for RerankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "rerank service unreachable: {e}"),
            Self::Unusable(e) => write!(f, "rerank service replied with something unusable: {e}"),
            Self::IndexOutOfRange { got, sent } => write!(
                f,
                "rerank service scored document {got} but only {sent} were sent; \
                 treating the reply as untrusted rather than clamping it"
            ),
            Self::NoDocuments => write!(f, "rerank called with no documents"),
        }
    }
}

impl std::error::Error for RerankError {}

/// One scored document, identified by its position in the request.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankHit {
    /// Position in the `documents` slice that was sent.
    pub index: usize,
    /// Relevance as the model judged it. Scale is the service's, not ours —
    /// compare within one reranking, never across services.
    pub relevance_score: f64,
}

/// The outcome of one reranking, including what did not get scored.
#[derive(Debug, Clone, PartialEq)]
pub struct Reranking {
    /// Scored documents, best first.
    pub hits: Vec<RerankHit>,
    /// Indices the service did not score, in their original order.
    ///
    /// Non-empty is not an error — `top_n` legitimately produces it — but it is
    /// something a caller has to decide about rather than discover later.
    pub unscored: Vec<usize>,
    /// Which service answered, for reports and logs.
    pub service: String,
}

impl Reranking {
    /// How many documents were left unscored.
    pub fn unscored_count(&self) -> usize {
        self.unscored.len()
    }
}

/// Something that can rank documents against a query.
pub trait RerankService: Send + Sync {
    /// Rank `documents` by relevance to `query`.
    ///
    /// `top_n` asks the service to return only its best n. `None` means all.
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Reranking, RerankError>;

    /// Human-readable identity, for reports.
    fn name(&self) -> &str;

    /// Is a model producing these scores?
    ///
    /// Ask before presenting a score as a judgement about meaning. The whole
    /// reason this trait exists is that the crate's built-in reranker answers
    /// `false` here and was being described as a cross-encoder anyway.
    fn is_model_backed(&self) -> bool {
        true
    }
}

/// Configuration for a `/v1/rerank` endpoint.
#[derive(Debug, Clone)]
pub struct RerankEndpoint {
    /// Full URL, e.g. `http://127.0.0.1:8080/v1/rerank`.
    pub url: String,
    /// Model name to send. llama.cpp ignores it; hosted services need it.
    pub model: String,
    /// Bearer token, for hosted services.
    pub api_key: Option<String>,
    /// How long to wait. A reranker runs one forward pass per document, so this
    /// scales with the candidate count, not with the query.
    pub timeout: Duration,
}

impl RerankEndpoint {
    /// A local `llama-server` started with `--rerank`.
    ///
    /// Port 8080 is llama.cpp's own default, and the same one
    /// [`AiConfig`](crate::AiConfig) uses for `LlamaCpp`.
    pub fn local_llama_cpp() -> Self {
        Self {
            url: "http://127.0.0.1:8080/v1/rerank".to_string(),
            model: "reranker".to_string(),
            api_key: None,
            timeout: Duration::from_secs(60),
        }
    }

    /// An endpoint at an arbitrary URL.
    pub fn at(url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            model: model.into(),
            api_key: None,
            timeout: Duration::from_secs(60),
        }
    }

    /// Attach a bearer token.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// A client for the Jina-shaped `/v1/rerank` API.
///
/// Covers llama.cpp (`--rerank`), Jina, Cohere and TEI, which agree on the
/// request and response shape closely enough for one implementation.
#[derive(Clone)]
pub struct HttpReranker {
    endpoint: RerankEndpoint,
    name: String,
    /// Built once, because that is the only way
    /// [`RerankEndpoint::timeout`] actually applies to anything. A first draft
    /// of this type stored the timeout and called `ureq::post` directly, so the
    /// field documented a behaviour the code did not have — the exact defect
    /// class the rest of this module was written to remove.
    agent: ureq::Agent,
}

impl std::fmt::Debug for HttpReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ureq::Agent` is not Debug, and a derived impl would have to leak the
        // api_key anyway. Print what is safe to print.
        f.debug_struct("HttpReranker")
            .field("url", &self.endpoint.url)
            .field("model", &self.endpoint.model)
            .field("has_api_key", &self.endpoint.api_key.is_some())
            .field("timeout", &self.endpoint.timeout)
            .finish()
    }
}

impl HttpReranker {
    /// Point a client at an endpoint.
    pub fn new(endpoint: RerankEndpoint) -> Self {
        let name = format!("http:{}", endpoint.model);
        let agent = ureq::AgentBuilder::new().timeout(endpoint.timeout).build();
        Self {
            endpoint,
            name,
            agent,
        }
    }

    /// The endpoint this client talks to.
    pub fn endpoint(&self) -> &RerankEndpoint {
        &self.endpoint
    }

    /// Parse a reply that has already been turned into JSON.
    ///
    /// Separate from the request so the shape can be tested without a server —
    /// which matters, because the shape is the part that varies between
    /// services and the part an open llama.cpp bug says can be wrong.
    pub fn parse_reply(
        &self,
        body: &serde_json::Value,
        sent: usize,
    ) -> Result<Reranking, RerankError> {
        let results = body
            .get("results")
            .and_then(|r| r.as_array())
            .ok_or_else(|| {
                RerankError::Unusable(format!(
                    "no `results` array in the reply; got keys {:?}",
                    body.as_object()
                        .map(|o| o.keys().cloned().collect::<Vec<_>>())
                        .unwrap_or_default()
                ))
            })?;

        let mut hits = Vec::with_capacity(results.len());
        for item in results {
            let index =
                item.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                    RerankError::Unusable("a result had no integer `index`".into())
                })? as usize;
            if index >= sent {
                return Err(RerankError::IndexOutOfRange { got: index, sent });
            }
            // Cohere and llama.cpp use `relevance_score`; some builds use `score`.
            let relevance_score = item
                .get("relevance_score")
                .or_else(|| item.get("score"))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    RerankError::Unusable(
                        "a result had neither `relevance_score` nor `score` as a number".into(),
                    )
                })?;
            hits.push(RerankHit {
                index,
                relevance_score,
            });
        }

        // The service is expected to sort, but nothing guarantees it — and a
        // reranker whose output is not sorted is indistinguishable from one
        // whose scores are wrong. Sort here so the contract holds regardless.
        hits.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let scored: HashSet<usize> = hits.iter().map(|h| h.index).collect();
        let unscored: Vec<usize> = (0..sent).filter(|i| !scored.contains(i)).collect();

        Ok(Reranking {
            hits,
            unscored,
            service: self.name.clone(),
        })
    }
}

impl RerankService for HttpReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Reranking, RerankError> {
        if documents.is_empty() {
            return Err(RerankError::NoDocuments);
        }

        let mut body = serde_json::json!({
            "model": self.endpoint.model,
            "query": query,
            "documents": documents,
        });
        if let Some(n) = top_n {
            body["top_n"] = serde_json::json!(n);
        }

        let mut request = self
            .agent
            .post(&self.endpoint.url)
            .set("Content-Type", "application/json");
        if let Some(ref key) = self.endpoint.api_key {
            request = request.set("Authorization", &format!("Bearer {}", key));
        }

        let response = request
            .send_json(body)
            .map_err(|e| RerankError::Transport(e.to_string()))?;

        let parsed: serde_json::Value = response
            .into_json()
            .map_err(|e| RerankError::Unusable(e.to_string()))?;

        self.parse_reply(&parsed, documents.len())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> HttpReranker {
        HttpReranker::new(RerankEndpoint::local_llama_cpp())
    }

    #[test]
    fn it_parses_the_jina_shape() {
        let body = serde_json::json!({
            "results": [
                {"index": 2, "relevance_score": 0.10},
                {"index": 0, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.55}
            ]
        });
        let r = client().parse_reply(&body, 3).expect("well-formed");
        assert_eq!(
            r.hits.iter().map(|h| h.index).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "must come back sorted by score, best first"
        );
        assert!(r.unscored.is_empty());
    }

    #[test]
    fn it_sorts_even_when_the_service_does_not() {
        // A reranker whose output is not sorted is indistinguishable from one
        // whose scores are wrong, so the contract is enforced on our side.
        let body = serde_json::json!({
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9}
            ]
        });
        let r = client().parse_reply(&body, 2).expect("well-formed");
        assert_eq!(r.hits[0].index, 1);
    }

    #[test]
    fn it_accepts_score_as_well_as_relevance_score() {
        let body = serde_json::json!({"results": [{"index": 0, "score": 0.42}]});
        let r = client().parse_reply(&body, 1).expect("well-formed");
        assert!((r.hits[0].relevance_score - 0.42).abs() < 1e-12);
    }

    #[test]
    fn an_index_outside_the_request_is_an_error_not_a_clamp() {
        // ggml-org/llama.cpp#16407 reports wrong rerank output for several
        // models, so the reply is untrusted input.
        let body = serde_json::json!({"results": [{"index": 7, "relevance_score": 1.0}]});
        let err = client().parse_reply(&body, 3).expect_err("must reject");
        assert_eq!(err, RerankError::IndexOutOfRange { got: 7, sent: 3 });
    }

    #[test]
    fn documents_the_service_skipped_are_reported_not_dropped() {
        // top_n legitimately causes this. What must not happen is the caller
        // finding out by counting.
        let body = serde_json::json!({
            "results": [
                {"index": 3, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8}
            ]
        });
        let r = client().parse_reply(&body, 5).expect("well-formed");
        assert_eq!(r.unscored, vec![0, 2, 4], "in their original order");
        assert_eq!(r.unscored_count(), 3);
    }

    #[test]
    fn a_reply_without_results_says_what_it_did_contain() {
        // The error message has to be actionable: "unusable" alone sends
        // somebody to a packet capture.
        let body = serde_json::json!({"error": "model does not support reranking"});
        let err = client().parse_reply(&body, 2).expect_err("must reject");
        match err {
            RerankError::Unusable(msg) => {
                assert!(msg.contains("error"), "should name the keys it saw: {msg}")
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn a_result_missing_its_score_is_rejected_rather_than_defaulted() {
        // Defaulting to 0.0 would push the document to the bottom, which reads
        // exactly like a confident judgement of irrelevance.
        let body = serde_json::json!({"results": [{"index": 0}]});
        assert!(client().parse_reply(&body, 1).is_err());
    }

    #[test]
    fn an_unreachable_service_errors_and_does_not_quietly_score_anything() {
        // The promise this module is built on, tested rather than asserted in
        // prose: when the reranker is not there, the caller finds out. Falling
        // back to word overlap would return a plausible ordering with no way to
        // tell it from a real one — the N55 pattern.
        //
        // Port 9 is the discard port; nothing listens on it here. A short
        // timeout keeps this fast even if something does.
        let unreachable = HttpReranker::new(
            RerankEndpoint::at("http://127.0.0.1:9/v1/rerank", "nobody")
                .with_timeout(Duration::from_millis(400)),
        );
        let err = unreachable
            .rerank("query", &["a".to_string(), "b".to_string()], None)
            .expect_err("nothing is listening, so this cannot succeed");
        assert!(
            matches!(err, RerankError::Transport(_)),
            "an unreachable service is a transport failure, not a parse one: {err:?}"
        );
    }

    #[test]
    fn no_documents_is_its_own_error() {
        let err = client()
            .rerank("q", &[], None)
            .expect_err("nothing to rank");
        assert_eq!(err, RerankError::NoDocuments);
        // And it does not touch the network to find that out.
    }

    #[test]
    fn it_says_it_is_model_backed_unlike_the_built_in_scorer() {
        let c = client();
        assert!(c.is_model_backed());
        assert!(c.name().contains("reranker"));
    }

    #[test]
    fn the_configured_timeout_actually_bounds_the_request() {
        // The first draft of this type stored `timeout` and then called
        // `ureq::post` directly, so the field described a behaviour the code did
        // not have. The first version of THIS test only checked that the struct
        // remembered the number — and a mutation removing `.timeout(...)` from
        // the agent survived it. A test whose name claims more than it verifies
        // is the same defect wearing a green tick, so it now measures the clock.
        use std::io::Read;
        use std::net::TcpListener;
        use std::time::Instant;

        // Not `bind(0)`: the OS hands back a port from the ephemeral range,
        // which is the same pool an outgoing connection draws its source port
        // from, and the two can collide. Scan a fixed range instead.
        let listener = (18200..18300)
            .find_map(|p| TcpListener::bind(("127.0.0.1", p)).ok())
            .expect("no free port in 18200..18300");
        let port = listener.local_addr().expect("the listener is bound").port();

        // Accept the connection, read the request, and then say nothing at all.
        // This is the case a connect-only timeout would miss: the socket opens
        // fine and the reply never comes.
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut sink = [0u8; 1024];
                let _ = stream.read(&mut sink);
                std::thread::sleep(Duration::from_secs(10));
            }
        });

        let client = HttpReranker::new(
            RerankEndpoint::at(format!("http://127.0.0.1:{port}/v1/rerank"), "silent")
                .with_timeout(Duration::from_millis(300)),
        );
        assert!(
            format!("{client:?}").contains("300"),
            "the timeout should be visible where somebody debugging would look"
        );

        let started = Instant::now();
        let err = client
            .rerank("q", &["a".to_string()], None)
            .expect_err("the server never replies, so this cannot succeed");
        let elapsed = started.elapsed();

        assert!(
            matches!(err, RerankError::Transport(_)),
            "a server that accepts and never answers is a transport failure: {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "the 300 ms timeout did not bound the request — it gave up after \
             {elapsed:?}, which means the value never reached the agent"
        );
    }

    #[test]
    fn debug_does_not_print_the_api_key() {
        let c = HttpReranker::new(
            RerankEndpoint::at("https://example.invalid/v1/rerank", "m")
                .with_api_key("super-secret-token"),
        );
        let shown = format!("{c:?}");
        assert!(
            !shown.contains("super-secret-token"),
            "a bearer token must not reach a log line: {shown}"
        );
        assert!(shown.contains("has_api_key: true"));
    }

    #[test]
    fn the_local_default_matches_llama_cpp_own_default_port() {
        let e = RerankEndpoint::local_llama_cpp();
        assert!(
            e.url.starts_with("http://127.0.0.1:8080"),
            "llama-server's default is 8080 and AiConfig already uses it: {}",
            e.url
        );
        assert!(e.api_key.is_none(), "a local engine needs no bearer token");
    }
}
