//! Webhook notifications
//!
//! Send event notifications via webhooks.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Webhook event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebhookEvent {
    MessageReceived,
    MessageSent,
    ErrorOccurred,
    RateLimitHit,
    ModelChanged,
    SessionStarted,
    SessionEnded,
    Custom,
}

/// Webhook configuration
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub secret: Option<String>,
    pub timeout: Duration,
    pub retry_count: u32,
    pub retry_delay: Duration,
    pub headers: HashMap<String, String>,
}

impl WebhookConfig {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            events: vec![],
            secret: None,
            timeout: Duration::from_secs(10),
            retry_count: 3,
            retry_delay: Duration::from_secs(1),
            headers: HashMap::new(),
        }
    }

    pub fn with_events(mut self, events: Vec<WebhookEvent>) -> Self {
        self.events = events;
        self
    }

    pub fn with_secret(mut self, secret: &str) -> Self {
        self.secret = Some(secret.to_string());
        self
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }
}

/// Webhook payload
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    pub event: WebhookEvent,
    pub timestamp: u64,
    pub data: serde_json::Value,
}

impl WebhookPayload {
    pub fn new(event: WebhookEvent, data: serde_json::Value) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            event,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            data,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "event": format!("{:?}", self.event),
            "timestamp": self.timestamp,
            "data": self.data,
        }).to_string()
    }
}

/// Webhook delivery result
#[derive(Debug, Clone)]
pub struct DeliveryResult {
    pub success: bool,
    pub status_code: Option<u16>,
    pub attempts: u32,
    pub duration: Duration,
    pub error: Option<String>,
}

/// Webhook manager
pub struct WebhookManager {
    webhooks: Vec<WebhookConfig>,
    delivery_history: Vec<(Instant, DeliveryResult)>,
    max_history: usize,
}

impl WebhookManager {
    pub fn new() -> Self {
        Self {
            webhooks: Vec::new(),
            delivery_history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn register(&mut self, config: WebhookConfig) {
        self.webhooks.push(config);
    }

    pub fn unregister(&mut self, url: &str) {
        self.webhooks.retain(|w| w.url != url);
    }

    /// Send webhook for an event
    pub fn send(&mut self, payload: WebhookPayload) -> Vec<DeliveryResult> {
        // Clone webhooks to avoid borrow issues
        let webhooks_to_call: Vec<WebhookConfig> = self.webhooks.iter()
            .filter(|w| w.events.is_empty() || w.events.contains(&payload.event))
            .cloned()
            .collect();

        let mut results = Vec::new();
        for webhook in &webhooks_to_call {
            let result = self.deliver(webhook, &payload);
            self.record_delivery(result.clone());
            results.push(result);
        }

        results
    }

    fn deliver(&self, webhook: &WebhookConfig, payload: &WebhookPayload) -> DeliveryResult {
        let start = Instant::now();
        let json = payload.to_json();

        for attempt in 1..=webhook.retry_count {
            let mut request = ureq::post(&webhook.url)
                .timeout(webhook.timeout)
                .set("Content-Type", "application/json");

            for (key, value) in &webhook.headers {
                request = request.set(key, value);
            }

            if let Some(secret) = &webhook.secret {
                // Simple signature
                let sig = format!("{:x}", json.len().wrapping_mul(secret.len()));
                request = request.set("X-Webhook-Signature", &sig);
            }

            match request.send_string(&json) {
                Ok(response) => {
                    return DeliveryResult {
                        success: response.status() < 400,
                        status_code: Some(response.status()),
                        attempts: attempt,
                        duration: start.elapsed(),
                        error: None,
                    };
                }
                Err(e) => {
                    if attempt == webhook.retry_count {
                        return DeliveryResult {
                            success: false,
                            status_code: None,
                            attempts: attempt,
                            duration: start.elapsed(),
                            error: Some(e.to_string()),
                        };
                    }
                    std::thread::sleep(webhook.retry_delay);
                }
            }
        }

        DeliveryResult {
            success: false,
            status_code: None,
            attempts: webhook.retry_count,
            duration: start.elapsed(),
            error: Some("Max retries exceeded".to_string()),
        }
    }

    fn record_delivery(&mut self, result: DeliveryResult) {
        if self.delivery_history.len() >= self.max_history {
            self.delivery_history.remove(0);
        }
        self.delivery_history.push((Instant::now(), result));
    }

    pub fn stats(&self) -> WebhookStats {
        let total = self.delivery_history.len();
        let successful = self.delivery_history.iter().filter(|(_, r)| r.success).count();

        WebhookStats {
            total_deliveries: total,
            successful_deliveries: successful,
            failed_deliveries: total - successful,
            success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            registered_webhooks: self.webhooks.len(),
        }
    }
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Webhook statistics
#[derive(Debug, Clone)]
pub struct WebhookStats {
    pub total_deliveries: usize,
    pub successful_deliveries: usize,
    pub failed_deliveries: usize,
    pub success_rate: f64,
    pub registered_webhooks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload() {
        let payload = WebhookPayload::new(
            WebhookEvent::MessageReceived,
            serde_json::json!({"message": "test"}),
        );

        let json = payload.to_json();
        assert!(json.contains("MessageReceived"));
    }

    #[test]
    fn test_manager() {
        let mut manager = WebhookManager::new();

        manager.register(WebhookConfig::new("http://example.com/webhook")
            .with_events(vec![WebhookEvent::MessageReceived]));

        assert_eq!(manager.webhooks.len(), 1);
    }
}
