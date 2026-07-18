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

#[cfg(test)]
mod tests {
    use super::*;

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
