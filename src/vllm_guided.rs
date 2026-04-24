//! V103.1: vLLM structured output / guided decoding helpers.
//!
//! vLLM supports three guided-decoding modes on top of OpenAI-compatible
//! `/v1/chat/completions` and `/v1/completions` bodies:
//!
//! - `guided_json`  — the response must be a JSON value conforming to a JSON
//!                    schema (passed as a `serde_json::Value`).
//! - `guided_regex` — the response must match a regular expression.
//! - `guided_choice` — the response must be one of a fixed set of strings.
//!
//! Only one mode can be active per request; callers that set more than one
//! will have the server reject the request. This module does not enforce
//! that — it's a thin builder that just injects the fields.

use serde::{Deserialize, Serialize};

/// Options for vLLM guided decoding. `None` on every field means
/// "unconstrained" (normal generation).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLlmGuidedOptions {
    /// JSON schema the response must conform to. Passed as a raw
    /// `serde_json::Value` rather than a typed struct so callers can
    /// supply arbitrary schemas.
    pub guided_json: Option<serde_json::Value>,
    /// Regular expression the response must match.
    pub guided_regex: Option<String>,
    /// Exact list of allowed response strings.
    pub guided_choice: Option<Vec<String>>,
}

impl VLlmGuidedOptions {
    /// Returns true if any guided-decoding option is set.
    pub fn is_active(&self) -> bool {
        self.guided_json.is_some() || self.guided_regex.is_some() || self.guided_choice.is_some()
    }
}

/// Merge guided-decoding options into an OpenAI-style JSON request body.
///
/// `body` is mutated in place. Fields that are `None` on `opts` are left
/// alone; set fields are written into the top level of the body object.
/// Does nothing if `body` is not a JSON object.
pub fn apply_guided(body: &mut serde_json::Value, opts: &VLlmGuidedOptions) {
    let Some(map) = body.as_object_mut() else {
        return;
    };
    if let Some(schema) = &opts.guided_json {
        map.insert("guided_json".to_string(), schema.clone());
    }
    if let Some(regex) = &opts.guided_regex {
        map.insert(
            "guided_regex".to_string(),
            serde_json::Value::String(regex.clone()),
        );
    }
    if let Some(choices) = &opts.guided_choice {
        map.insert(
            "guided_choice".to_string(),
            serde_json::Value::Array(
                choices
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn is_active_false_when_all_none() {
        let opts = VLlmGuidedOptions::default();
        assert!(!opts.is_active());
    }

    #[test]
    fn is_active_true_when_any_set() {
        let opts = VLlmGuidedOptions {
            guided_regex: Some("^[A-Z]+$".into()),
            ..Default::default()
        };
        assert!(opts.is_active());
    }

    #[test]
    fn apply_noop_when_opts_empty() {
        let mut body = json!({"model": "qwen", "messages": []});
        let before = body.clone();
        apply_guided(&mut body, &VLlmGuidedOptions::default());
        assert_eq!(body, before);
    }

    #[test]
    fn apply_injects_json_schema() {
        let mut body = json!({"model": "qwen"});
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });
        let opts = VLlmGuidedOptions {
            guided_json: Some(schema.clone()),
            ..Default::default()
        };
        apply_guided(&mut body, &opts);
        assert_eq!(body["guided_json"], schema);
    }

    #[test]
    fn apply_injects_regex_and_choice() {
        let mut body = json!({});
        let opts = VLlmGuidedOptions {
            guided_regex: Some(r"^\d{3}$".into()),
            guided_choice: Some(vec!["yes".into(), "no".into()]),
            ..Default::default()
        };
        apply_guided(&mut body, &opts);
        assert_eq!(body["guided_regex"], "^\\d{3}$");
        assert_eq!(body["guided_choice"], json!(["yes", "no"]));
    }

    #[test]
    fn apply_noop_on_non_object_body() {
        let mut body = json!([1, 2, 3]);
        let opts = VLlmGuidedOptions {
            guided_regex: Some("x".into()),
            ..Default::default()
        };
        apply_guided(&mut body, &opts);
        // Body is untouched because it wasn't an object.
        assert_eq!(body, json!([1, 2, 3]));
    }

    #[test]
    fn apply_overwrites_existing_fields() {
        let mut body = json!({"guided_regex": "old"});
        let opts = VLlmGuidedOptions {
            guided_regex: Some("new".into()),
            ..Default::default()
        };
        apply_guided(&mut body, &opts);
        assert_eq!(body["guided_regex"], "new");
    }
}
