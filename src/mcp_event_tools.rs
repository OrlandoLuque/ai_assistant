//! MCP tools for managing event subscriptions (Universal Event System).
//!
//! 5 tools: subscribe, unsubscribe, list rules, get notifications, dismiss.

use crate::event_source::{
    EventAction, EventFilter, EventRule, EventSourceConfig, EventSourceManager,
};
use crate::mcp_protocol::server::McpServer;
use crate::mcp_protocol::types::{McpTool, McpToolAnnotation};
use std::sync::{Arc, Mutex};

fn new_uuid() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (d.as_nanos() & 0xFFFF_FFFF) as u32,
        ((d.as_nanos() >> 32) & 0xFFFF) as u16,
        ((d.as_nanos() >> 48) & 0x0FFF) as u16,
        (0x8000 | ((d.as_nanos() >> 60) & 0x3FFF)) as u16,
        (d.as_nanos() >> 74) ^ (d.subsec_nanos() as u128),
    )
}

/// Register all 5 event management MCP tools.
pub fn register_event_tools(server: &mut McpServer, manager: Arc<Mutex<EventSourceManager>>) {
    // --- event_subscribe ---
    {
        let mgr = manager.clone();
        server.register_tool(
            McpTool::new(
                "event_subscribe",
                "Subscribe to an event source. Types: webhook, rss, scraper, calendar, mqtt, websocket, rest_poll, email. Configure action (prompt_llm, notify, both), filters, cooldown, and prompt template.",
            )
            .with_property("name", "string", "Rule name (required)", true)
            .with_property("source_type", "string", "Source type: webhook, rss, scraper, calendar, mqtt, websocket, rest_poll, email (required)", true)
            .with_property("url", "string", "URL for the source (feed URL, scrape URL, API URL, etc.)", false)
            .with_property("action", "string", "Action: prompt_llm, notify, both (default: notify)", false)
            .with_property("prompt_template", "string", "Template for LLM prompt (uses {{title}}, {{body}}, {{url}}, {{data.field}})", false)
            .with_property("cooldown_secs", "integer", "Min seconds between events (default: 60)", false)
            .with_property("filters", "array", "Array of filter objects: {type, value}", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Subscribe to Events".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(false),
                open_world_hint: Some(true),
            }),
            move |args| {
                let name = args.get("name").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: name")?;
                let source_type = args.get("source_type").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: source_type")?;
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");

                let action = match args.get("action").and_then(|v| v.as_str()) {
                    Some("prompt_llm") => EventAction::PromptLlm,
                    Some("both") => EventAction::Both,
                    _ => EventAction::Notify,
                };

                let prompt_template = args.get("prompt_template").and_then(|v| v.as_str()).map(|s| s.to_string());
                let cooldown = args.get("cooldown_secs").and_then(|v| v.as_u64()).unwrap_or(60);

                // Parse filters
                let filters = parse_filters(args.get("filters"));

                // Build source config based on type
                let source = match source_type {
                    "webhook" => EventSourceConfig::Webhook { secret: None },
                    "rss" => {
                        if url.is_empty() { return Err("RSS requires 'url' parameter".into()); }
                        EventSourceConfig::Rss { feed_url: url.into() }
                    }
                    "scraper" => {
                        if url.is_empty() { return Err("Scraper requires 'url' parameter".into()); }
                        let selector = args.get("selector").and_then(|v| v.as_str()).map(|s| s.to_string());
                        EventSourceConfig::Scraper { url: url.into(), selector, watch_field: None }
                    }
                    "calendar" => {
                        if url.is_empty() { return Err("Calendar requires 'url' (iCal URL) parameter".into()); }
                        let reminder = args.get("reminder_minutes").and_then(|v| v.as_u64()).unwrap_or(15) as u32;
                        EventSourceConfig::Calendar { ical_url: url.into(), reminder_minutes: reminder }
                    }
                    "mqtt" => {
                        let topic = args.get("topic").and_then(|v| v.as_str()).unwrap_or(url);
                        if topic.is_empty() { return Err("MQTT requires 'topic' or 'url' parameter".into()); }
                        EventSourceConfig::Mqtt { topic: topic.into(), broker_url: None }
                    }
                    "websocket" => {
                        if url.is_empty() { return Err("WebSocket requires 'url' parameter".into()); }
                        EventSourceConfig::WebSocket { url: url.into() }
                    }
                    "rest_poll" => {
                        if url.is_empty() { return Err("REST poll requires 'url' parameter".into()); }
                        let method = args.get("method").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let watch_path = args.get("watch_path").and_then(|v| v.as_str()).map(|s| s.to_string());
                        EventSourceConfig::RestPoll { url: url.into(), method, headers: None, watch_path }
                    }
                    "email" => {
                        let server = args.get("imap_server").and_then(|v| v.as_str())
                            .ok_or("Email requires 'imap_server' parameter")?;
                        let port = args.get("imap_port").and_then(|v| v.as_u64()).unwrap_or(993) as u16;
                        let username = args.get("username").and_then(|v| v.as_str())
                            .ok_or("Email requires 'username' parameter")?;
                        let password_key = args.get("password_key").and_then(|v| v.as_str())
                            .ok_or("Email requires 'password_key' (credential resolver key)")?;
                        let from_filter = args.get("from_filter").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let subject_filter = args.get("subject_filter").and_then(|v| v.as_str()).map(|s| s.to_string());
                        EventSourceConfig::Email {
                            imap_server: server.into(), imap_port: port,
                            username: username.into(), password_key: password_key.into(),
                            from_filter, subject_filter,
                        }
                    }
                    _ => return Err(format!("Unknown source type '{}'. Valid: webhook, rss, scraper, calendar, mqtt, websocket, rest_poll, email", source_type)),
                };

                let rule = EventRule {
                    id: new_uuid(),
                    name: name.into(),
                    source,
                    action,
                    prompt_template,
                    filters,
                    cooldown_secs: cooldown,
                    schedule: args.get("schedule").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    enabled: true,
                    created_by: None,
                };

                let rule_id = rule.id.clone();
                let mut guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.add_rule(rule)?;

                Ok(serde_json::json!({
                    "rule_id": rule_id,
                    "name": name,
                    "source_type": source_type,
                    "action": format!("{:?}", action),
                    "created": true,
                }))
            },
        );
    }

    // --- event_unsubscribe ---
    {
        let mgr = manager.clone();
        server.register_tool(
            McpTool::new("event_unsubscribe", "Remove an event subscription by rule ID.")
                .with_property("rule_id", "string", "Rule ID to remove (required)", true)
                .with_annotations(McpToolAnnotation {
                    title: Some("Unsubscribe".into()),
                    read_only_hint: Some(false),
                    destructive_hint: Some(true),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |args| {
                let rule_id = args.get("rule_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: rule_id")?;
                let mut guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                let removed = guard.remove_rule(rule_id);
                Ok(serde_json::json!({ "removed": removed, "rule_id": rule_id }))
            },
        );
    }

    // --- event_list_rules ---
    {
        let mgr = manager.clone();
        server.register_tool(
            McpTool::new("event_list_rules", "List all active event subscriptions.")
                .with_annotations(McpToolAnnotation {
                    title: Some("List Event Rules".into()),
                    read_only_hint: Some(true),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |_args| {
                let guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                let rules: Vec<serde_json::Value> = guard.list_rules().iter().map(|r| {
                    serde_json::json!({
                        "id": r.id,
                        "name": r.name,
                        "action": format!("{:?}", r.action),
                        "enabled": r.enabled,
                        "cooldown_secs": r.cooldown_secs,
                    })
                }).collect();
                Ok(serde_json::json!({ "rules": rules, "count": rules.len() }))
            },
        );
    }

    // --- event_notifications ---
    {
        let mgr = manager.clone();
        server.register_tool(
            McpTool::new("event_notifications", "Get pending event notifications.")
                .with_annotations(McpToolAnnotation {
                    title: Some("Event Notifications".into()),
                    read_only_hint: Some(true),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |_args| {
                let guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                let notifs = guard.notifications();
                let items: Vec<serde_json::Value> = notifs.iter().map(|n| {
                    serde_json::json!({
                        "id": n.id,
                        "title": n.title,
                        "body": n.body,
                        "source": n.source_name,
                        "timestamp": n.timestamp,
                        "url": n.url,
                    })
                }).collect();
                Ok(serde_json::json!({ "notifications": items, "count": items.len() }))
            },
        );
    }

    // --- event_dismiss ---
    {
        let mgr = manager.clone();
        server.register_tool(
            McpTool::new("event_dismiss", "Dismiss event notifications. Pass event_id to dismiss one, or omit to dismiss all.")
                .with_property("event_id", "string", "Event ID to dismiss (omit to dismiss all)", false)
                .with_annotations(McpToolAnnotation {
                    title: Some("Dismiss Notifications".into()),
                    read_only_hint: Some(false),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |args| {
                let guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                if let Some(event_id) = args.get("event_id").and_then(|v| v.as_str()) {
                    guard.dismiss_notification(event_id);
                    Ok(serde_json::json!({ "dismissed": event_id }))
                } else {
                    guard.dismiss_notifications();
                    Ok(serde_json::json!({ "dismissed": "all" }))
                }
            },
        );
    }
}

/// Parse filter objects from JSON array.
fn parse_filters(filters_val: Option<&serde_json::Value>) -> Vec<EventFilter> {
    let arr = match filters_val.and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };

    let mut filters = Vec::new();
    for item in arr {
        let filter_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match filter_type {
            "title_contains" => {
                if let Some(val) = item.get("value").and_then(|v| v.as_str()) {
                    filters.push(EventFilter::TitleContains(val.into()));
                }
            }
            "body_contains" => {
                if let Some(val) = item.get("value").and_then(|v| v.as_str()) {
                    filters.push(EventFilter::BodyContains(val.into()));
                }
            }
            "price_below" => {
                if let Some(val) = item.get("value").and_then(|v| v.as_f64()) {
                    filters.push(EventFilter::PriceBelow(val));
                }
            }
            "price_above" => {
                if let Some(val) = item.get("value").and_then(|v| v.as_f64()) {
                    filters.push(EventFilter::PriceAbove(val));
                }
            }
            "data_equals" => {
                if let (Some(path), Some(value)) = (
                    item.get("path").and_then(|v| v.as_str()),
                    item.get("value"),
                ) {
                    filters.push(EventFilter::DataFieldEquals {
                        path: path.into(),
                        value: value.clone(),
                    });
                }
            }
            _ => {} // Unknown filter type — skip
        }
    }
    filters
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filters_empty() {
        let filters = parse_filters(None);
        assert!(filters.is_empty());
    }

    #[test]
    fn test_parse_filters_title_contains() {
        let json = serde_json::json!([
            {"type": "title_contains", "value": "urgent"},
            {"type": "price_below", "value": 50.0},
        ]);
        let filters = parse_filters(Some(&json));
        assert_eq!(filters.len(), 2);
    }

    #[test]
    fn test_parse_filters_unknown_type() {
        let json = serde_json::json!([
            {"type": "unknown_filter", "value": "test"},
        ]);
        let filters = parse_filters(Some(&json));
        assert!(filters.is_empty()); // Unknown types are skipped
    }
}
