//! V103.1: vLLM Prometheus `/metrics` scraper.
//!
//! vLLM exposes a standard Prometheus text-format endpoint at `/metrics`.
//! The full surface is large (80+ metrics); we extract the handful that
//! actually matter for runtime decisions: queue depth, how many requests
//! are running, KV-cache utilisation, and cumulative token counters.
//!
//! This is a zero-dependency text parser — no prometheus client crate. A
//! line like `vllm:num_requests_running{model_name="..."} 3.0` becomes the
//! `running_requests` field. Labels are ignored (we aggregate across models).

use serde::{Deserialize, Serialize};

/// Runtime metrics extracted from vLLM's `/metrics` endpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VLlmMetrics {
    /// Requests currently being generated (not queued).
    pub running_requests: Option<f64>,
    /// Requests waiting for a slot (queue depth). High + stable suggests
    /// the server is saturated and a larger TP size or bigger GPU would
    /// help.
    pub waiting_requests: Option<f64>,
    /// GPU KV-cache utilisation as a fraction 0.0–1.0. Persistent values
    /// ≥0.9 mean the server is close to evicting prefixes.
    pub gpu_cache_usage: Option<f64>,
    /// Cumulative prompt tokens processed since launch.
    pub prompt_tokens_total: Option<f64>,
    /// Cumulative generated tokens since launch.
    pub generation_tokens_total: Option<f64>,
}

impl VLlmMetrics {
    /// Coarse "is the server saturated right now?" signal. True when
    /// either waiting ≥4 OR cache usage ≥0.9.
    pub fn saturated(&self) -> bool {
        self.waiting_requests.unwrap_or(0.0) >= 4.0 || self.gpu_cache_usage.unwrap_or(0.0) >= 0.9
    }
}

/// Fetch `/metrics` from a running vLLM server and parse the subset we
/// care about. Non-2xx / connection errors surface as `Err`.
pub fn scrape_vllm_metrics(base_url: &str) -> Result<VLlmMetrics, String> {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));
    let body = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(3))
        .call()
        .map_err(|e| format!("GET {}: {}", url, e))?
        .into_string()
        .map_err(|e| format!("read body from {}: {}", url, e))?;
    Ok(parse_vllm_metrics(&body))
}

/// Parse a Prometheus text-format body into a [`VLlmMetrics`].
///
/// Sums values across label sets — vLLM publishes per-model metrics and we
/// aggregate across them. Comment lines (`# …`) are ignored.
pub fn parse_vllm_metrics(body: &str) -> VLlmMetrics {
    let mut m = VLlmMetrics::default();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (name, value) = match split_metric_line(line) {
            Some(tup) => tup,
            None => continue,
        };

        match name {
            "vllm:num_requests_running" => accumulate(&mut m.running_requests, value),
            "vllm:num_requests_waiting" => accumulate(&mut m.waiting_requests, value),
            "vllm:gpu_cache_usage_perc" | "vllm:gpu_cache_usage_percentage" => {
                // vLLM reports 0.0–1.0 despite the "perc" name on some versions.
                accumulate_max(&mut m.gpu_cache_usage, value);
            }
            "vllm:prompt_tokens_total" => accumulate(&mut m.prompt_tokens_total, value),
            "vllm:generation_tokens_total" => accumulate(&mut m.generation_tokens_total, value),
            _ => {}
        }
    }
    m
}

/// Split `metric_name{labels} 123.45` into `(name, value)`, ignoring
/// labels. Returns `None` if the line doesn't parse.
fn split_metric_line(line: &str) -> Option<(&str, f64)> {
    // Strip labels: `name{...} value` → `name value`
    let (name_part, rest) = if let Some(brace) = line.find('{') {
        let close = line.find('}')?;
        let name = &line[..brace];
        let after = line[close + 1..].trim_start();
        (name, after)
    } else {
        let space = line.find(char::is_whitespace)?;
        (&line[..space], line[space..].trim_start())
    };
    let value_str = rest.split_whitespace().next()?;
    let value: f64 = value_str.parse().ok()?;
    Some((name_part.trim(), value))
}

fn accumulate(slot: &mut Option<f64>, v: f64) {
    *slot = Some(slot.unwrap_or(0.0) + v);
}

fn accumulate_max(slot: &mut Option<f64>, v: f64) {
    *slot = Some(slot.map(|cur| cur.max(v)).unwrap_or(v));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_empty_body() {
        let m = parse_vllm_metrics("");
        assert!(m.running_requests.is_none());
        assert!(m.waiting_requests.is_none());
    }

    #[test]
    fn ignores_comments_and_unknown_metrics() {
        let body = "# HELP vllm:num_requests_running Requests\n\
                    # TYPE vllm:num_requests_running gauge\n\
                    unrelated_metric 42\n";
        let m = parse_vllm_metrics(body);
        assert!(m.running_requests.is_none());
    }

    #[test]
    fn parses_simple_value() {
        let body = "vllm:num_requests_running 3.0\n";
        let m = parse_vllm_metrics(body);
        assert_eq!(m.running_requests, Some(3.0));
    }

    #[test]
    fn parses_value_with_labels() {
        let body = "vllm:num_requests_waiting{model_name=\"qwen/q7\",extra=\"1\"} 7\n";
        let m = parse_vllm_metrics(body);
        assert_eq!(m.waiting_requests, Some(7.0));
    }

    #[test]
    fn aggregates_across_label_sets() {
        let body = "vllm:num_requests_running{model=\"a\"} 2\n\
                    vllm:num_requests_running{model=\"b\"} 3\n";
        let m = parse_vllm_metrics(body);
        assert_eq!(m.running_requests, Some(5.0));
    }

    #[test]
    fn gpu_cache_usage_takes_max_not_sum() {
        let body = "vllm:gpu_cache_usage_perc{model=\"a\"} 0.4\n\
                    vllm:gpu_cache_usage_perc{model=\"b\"} 0.7\n";
        let m = parse_vllm_metrics(body);
        assert_eq!(m.gpu_cache_usage, Some(0.7));
    }

    #[test]
    fn saturated_when_waiting_queue_high() {
        let m = VLlmMetrics {
            waiting_requests: Some(5.0),
            ..Default::default()
        };
        assert!(m.saturated());
    }

    #[test]
    fn saturated_when_kv_cache_near_full() {
        let m = VLlmMetrics {
            gpu_cache_usage: Some(0.95),
            ..Default::default()
        };
        assert!(m.saturated());
    }

    #[test]
    fn not_saturated_under_light_load() {
        let m = VLlmMetrics {
            running_requests: Some(2.0),
            waiting_requests: Some(0.0),
            gpu_cache_usage: Some(0.3),
            ..Default::default()
        };
        assert!(!m.saturated());
    }

    #[test]
    fn scrape_fails_cleanly_on_unreachable_host() {
        let res = scrape_vllm_metrics("http://127.0.0.1:1");
        assert!(res.is_err());
    }
}
