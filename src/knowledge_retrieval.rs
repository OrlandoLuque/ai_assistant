//! Lightweight in-memory knowledge retrieval (V196).
//!
//! Injecting a whole large `--knowledge` document into the prompt forces a
//! huge context window (V192) — expensive in VRAM and impossible on
//! small-window models. This module chunks an ad-hoc knowledge string and
//! returns only the passages most relevant to the query, so the prompt stays
//! small and the answer is still grounded. It is a cheap term-overlap ranker
//! (no embeddings, no persistent store, deterministic) — the always-available
//! complement to the feature-gated `rag` store.

/// Very small bilingual (ES/EN) stop-word set: tokens with no discriminative
/// value that would otherwise match every chunk.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "you", "your", "are", "was", "por",
    "para", "con", "que", "los", "las", "una", "uno", "del", "sus", "como", "cual", "cuanto",
    "cuánto", "cuesta", "año", "años", "solo", "sólo", "the", "responde", "importe", "exacto",
];

/// Split `text` into passages of roughly `target_chars` characters. Prefers
/// paragraph boundaries (blank lines); a paragraph longer than the target is
/// sliced on whitespace so no chunk is unbounded.
fn chunk(text: &str, target_chars: usize) -> Vec<String> {
    let target = target_chars.max(120);
    let mut chunks = Vec::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.len() <= target * 3 / 2 {
            chunks.push(para.to_string());
            continue;
        }
        // Oversized paragraph: pack whitespace-separated words up to target.
        let mut cur = String::new();
        for word in para.split_whitespace() {
            if !cur.is_empty() && cur.len() + 1 + word.len() > target {
                chunks.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(word);
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
    }
    chunks
}

/// Content terms of `query`: lowercased alphanumeric tokens of length >= 3
/// that are not stop-words.
fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.chars().count() >= 3 && !STOP_WORDS.contains(&t.as_str()))
        .collect()
}

/// Relevance score of `chunk` for the given query terms: summed occurrences of
/// each distinct term (a BM25-lite term-overlap), so a passage mentioning the
/// query's content words ranks above filler.
fn score(chunk_lower: &str, terms: &[String]) -> usize {
    let mut total = 0;
    for term in terms {
        total += chunk_lower.matches(term.as_str()).count();
    }
    total
}

/// Return the passages of `knowledge` most relevant to `query`, concatenated
/// in original document order, up to about `max_chars` characters.
///
/// Falls back to the leading `max_chars` when the query has no usable terms or
/// nothing matches, so the caller always gets *some* grounding. When the whole
/// document already fits in `max_chars` it is returned unchanged.
pub fn select_relevant(knowledge: &str, query: &str, max_chars: usize) -> String {
    if knowledge.len() <= max_chars {
        return knowledge.to_string();
    }
    // Chunk to a fraction of the budget so several passages can be combined.
    let chunk_target = (max_chars / 4).max(200);
    let chunks = chunk(knowledge, chunk_target);
    if chunks.is_empty() {
        return String::new();
    }

    let terms = query_terms(query);

    // Rank chunks (with original index for stable, in-order output).
    let mut ranked: Vec<(usize, usize, &String)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (score(&c.to_lowercase(), &terms), i, c))
        .collect();
    // Highest score first; ties keep earlier chunks.
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

    // If nothing matched, fall back to the leading budget of text.
    if terms.is_empty() || ranked.first().map(|r| r.0).unwrap_or(0) == 0 {
        return knowledge.chars().take(max_chars).collect();
    }

    // Greedily take top-scoring chunks until the budget is spent.
    let mut selected: Vec<(usize, &String)> = Vec::new();
    let mut used = 0usize;
    for (s, idx, chunk) in ranked {
        if s == 0 {
            break;
        }
        let cost = chunk.len() + 2; // + separator
        if used + cost > max_chars && !selected.is_empty() {
            break;
        }
        selected.push((idx, chunk));
        used += cost;
        if used >= max_chars {
            break;
        }
    }

    // Emit in original document order for readability.
    selected.sort_by_key(|(idx, _)| *idx);
    selected
        .into_iter()
        .map(|(_, c)| c.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Cosine similarity of two equal-length vectors (0 when either is empty/zero
/// or lengths differ).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Embed a batch of texts via Ollama's `/api/embed`. Returns `None` on any
/// failure (model missing, server down, malformed response) so callers can
/// fall back to lexical retrieval.
fn ollama_embed(ollama_url: &str, model: &str, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    let url = format!("{}/api/embed", ollama_url.trim_end_matches('/'));
    let body = serde_json::json!({ "model": model, "input": texts });
    let resp = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(30))
        .send_json(&body)
        .ok()?;
    let json: serde_json::Value = resp.into_json().ok()?;
    let arr = json.get("embeddings")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for e in arr {
        let v: Vec<f32> = e
            .as_array()?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        out.push(v);
    }
    if out.len() == texts.len() {
        Some(out)
    } else {
        None
    }
}

/// Semantic variant of [`select_relevant`]: ranks chunks by cosine similarity
/// of their Ollama embeddings to the query embedding, so paraphrased or
/// synonymous queries still match (lexical term-overlap would miss "¿de qué
/// vivo?" against "trabajo de arquitecta"). Returns `None` if embeddings are
/// unavailable — the caller should then fall back to [`select_relevant`].
pub fn select_relevant_semantic(
    knowledge: &str,
    query: &str,
    max_chars: usize,
    ollama_url: &str,
    model: &str,
) -> Option<String> {
    if knowledge.len() <= max_chars {
        return Some(knowledge.to_string());
    }
    let chunk_target = (max_chars / 4).max(200);
    let chunks = chunk(knowledge, chunk_target);
    if chunks.is_empty() {
        return Some(String::new());
    }
    // Embed the query and every chunk in one batch call.
    let mut texts: Vec<&str> = Vec::with_capacity(chunks.len() + 1);
    texts.push(query);
    texts.extend(chunks.iter().map(|c| c.as_str()));
    let embs = ollama_embed(ollama_url, model, &texts)?;
    let (q_emb, chunk_embs) = embs.split_first()?;

    let mut ranked: Vec<(f32, usize, &String)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (cosine(q_emb, &chunk_embs[i]), i, c))
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut selected: Vec<(usize, &String)> = Vec::new();
    let mut used = 0usize;
    for (_, idx, c) in ranked {
        let cost = c.len() + 2;
        if used + cost > max_chars && !selected.is_empty() {
            break;
        }
        selected.push((idx, c));
        used += cost;
        if used >= max_chars {
            break;
        }
    }
    selected.sort_by_key(|(i, _)| *i);
    Some(
        selected
            .into_iter()
            .map(|(_, c)| c.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    )
}

/// Retrieve relevant passages using **semantic** ranking when an
/// `embedding_model` is supplied and reachable (better for paraphrased
/// queries), falling back to the always-available **lexical**
/// [`select_relevant`] otherwise.
pub fn select_relevant_auto(
    knowledge: &str,
    query: &str,
    max_chars: usize,
    ollama_url: &str,
    embedding_model: Option<&str>,
) -> String {
    if let Some(model) = embedding_model {
        if let Some(sem) = select_relevant_semantic(knowledge, query, max_chars, ollama_url, model)
        {
            return sem;
        }
    }
    select_relevant(knowledge, query, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn auto_falls_back_to_lexical_without_model() {
        // No embedding model -> lexical path, still retrieves the buried line.
        let mut doc = String::new();
        for i in 0..200 {
            doc.push_str(&format!("Section {i}: filler notes.\n\n"));
        }
        doc.push_str("Startup (menos de 10 empleados): 490 EUR.\n\n");
        let out = select_relevant_auto(&doc, "precio Startup", 600, "http://127.0.0.1:11434", None);
        assert!(out.contains("490"));
    }

    #[test]
    fn returns_short_knowledge_unchanged() {
        let k = "Startup: 490 EUR.";
        assert_eq!(select_relevant(k, "startup price", 1000), k);
    }

    #[test]
    fn retrieves_the_relevant_buried_passage() {
        // A price line buried in a lot of unrelated filler.
        let mut doc = String::new();
        for i in 0..200 {
            doc.push_str(&format!(
                "Section {i}: polygon scales and grid sizes and epsilon comparison notes.\n\n"
            ));
        }
        doc.push_str("Licencia Startup (menos de 10 empleados): 490 EUR al año.\n\n");
        for i in 200..400 {
            doc.push_str(&format!(
                "Section {i}: more unrelated engine pipeline notes.\n\n"
            ));
        }
        assert!(doc.len() > 5000);

        let out = select_relevant(&doc, "¿cuánto cuesta la licencia para una Startup?", 800);
        // The relevant passage is retrieved and the output stays small.
        assert!(out.to_lowercase().contains("startup"));
        assert!(out.contains("490"));
        assert!(out.len() <= 900);
    }

    #[test]
    fn falls_back_to_leading_text_when_nothing_matches() {
        let doc = "AAAA ".repeat(500); // > max, no query terms present
        let out = select_relevant(&doc, "zzzzz qqqqq", 100);
        assert!(!out.is_empty());
        assert!(out.len() <= 100);
    }
}
