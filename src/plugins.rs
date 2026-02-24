//! Plugin system for extensible functionality
//!
//! This module provides a plugin architecture for extending the AI assistant
//! with custom providers, processors, and hooks.
//!
//! # Features
//!
//! - **Provider plugins**: Add custom AI providers
//! - **Processor plugins**: Transform messages before/after
//! - **Hook plugins**: React to events
//! - **Dynamic loading**: Load plugins at runtime
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::plugins::{Plugin, PluginManager, PluginContext};
//!
//! struct MyPlugin;
//!
//! impl Plugin for MyPlugin {
//!     fn name(&self) -> &str { "my-plugin" }
//!     fn version(&self) -> &str { "1.0.0" }
//!     fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> { Ok(()) }
//! }
//!
//! let mut manager = PluginManager::new();
//! manager.register(Box::new(MyPlugin));
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Plugin trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Get plugin description
    fn description(&self) -> &str {
        ""
    }

    /// Called when plugin is loaded
    fn on_load(&mut self, ctx: &PluginContext) -> Result<(), String>;

    /// Called when plugin is unloaded
    fn on_unload(&mut self) {}

    /// Called before a message is sent
    fn on_before_send(&self, _message: &str) -> Option<String> {
        None
    }

    /// Called after a response is received
    fn on_after_receive(&self, _response: &str) -> Option<String> {
        None
    }

    /// Get plugin capabilities
    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![]
    }

    /// Get plugin configuration schema
    fn config_schema(&self) -> Option<serde_json::Value> {
        None
    }

    /// Update plugin configuration
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), String> {
        Ok(())
    }

    /// Called before a server request is processed. Returns None to continue, Some(response) to short-circuit.
    fn on_request(
        &self,
        _method: &str,
        _path: &str,
        _headers: &[(String, String)],
    ) -> Option<(String, String)> {
        None
    }

    /// Called after a server request is processed with the status code.
    fn on_response(&self, _method: &str, _path: &str, _status: &str, _duration_ms: f64) {}

    /// Called when a server event occurs (e.g., "startup", "shutdown", "error").
    fn on_event(&self, _event: &str, _data: &str) {}

    /// Get plugin as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get mutable plugin as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Plugin capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PluginCapability {
    /// Can provide AI responses
    Provider,
    /// Can process messages
    Processor,
    /// Can react to events
    EventHandler,
    /// Can modify configuration
    ConfigModifier,
    /// Can provide embeddings
    Embeddings,
    /// Can store data
    Storage,
    /// Custom capability
    Custom(String),
}

/// Context passed to plugins
pub struct PluginContext {
    /// Plugin data directory
    pub data_dir: Option<std::path::PathBuf>,
    /// Shared state
    pub state: Arc<RwLock<HashMap<String, Box<dyn Any + Send + Sync>>>>,
    /// Configuration
    pub config: HashMap<String, serde_json::Value>,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new() -> Self {
        Self {
            data_dir: None,
            state: Arc::new(RwLock::new(HashMap::new())),
            config: HashMap::new(),
        }
    }

    /// Set data directory
    pub fn with_data_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.data_dir = Some(path.into());
        self
    }

    /// Get shared state
    pub fn get_state<T: 'static + Clone + Send + Sync>(&self, key: &str) -> Option<T> {
        let state = self.state.read().ok()?;
        state.get(key)?.downcast_ref::<T>().cloned()
    }

    /// Set shared state
    pub fn set_state<T: 'static + Send + Sync>(&self, key: impl Into<String>, value: T) {
        if let Ok(mut state) = self.state.write() {
            state.insert(key.into(), Box::new(value));
        }
    }
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin capabilities
    pub capabilities: Vec<PluginCapability>,
    /// Whether plugin is enabled
    pub enabled: bool,
    /// Load order priority
    pub priority: i32,
}

/// Plugin manager for registering and managing plugins
pub struct PluginManager {
    plugins: Vec<(PluginInfo, Box<dyn Plugin>)>,
    context: PluginContext,
    #[allow(dead_code)]
    event_handlers: HashMap<String, Vec<usize>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            context: PluginContext::new(),
            event_handlers: HashMap::new(),
        }
    }

    /// Create with custom context
    pub fn with_context(context: PluginContext) -> Self {
        Self {
            plugins: Vec::new(),
            context,
            event_handlers: HashMap::new(),
        }
    }

    /// Register a plugin
    pub fn register(&mut self, mut plugin: Box<dyn Plugin>) -> Result<(), String> {
        let info = PluginInfo {
            name: plugin.name().to_string(),
            version: plugin.version().to_string(),
            description: plugin.description().to_string(),
            capabilities: plugin.capabilities(),
            enabled: true,
            priority: 0,
        };

        // Check for duplicates
        if self.plugins.iter().any(|(i, _)| i.name == info.name) {
            return Err(format!("Plugin '{}' is already registered", info.name));
        }

        // Load plugin
        plugin.on_load(&self.context)?;

        self.plugins.push((info, plugin));

        // Sort by priority
        self.plugins.sort_by_key(|(i, _)| -i.priority);

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&mut self, name: &str) -> bool {
        if let Some(idx) = self.plugins.iter().position(|(i, _)| i.name == name) {
            let (_, mut plugin) = self.plugins.remove(idx);
            plugin.on_unload();
            true
        } else {
            false
        }
    }

    /// Get plugin by name
    pub fn get(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins
            .iter()
            .find(|(i, _)| i.name == name)
            .map(|(_, p)| p.as_ref())
    }

    /// Get mutable plugin by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn Plugin + 'static)> {
        for (info, plugin) in self.plugins.iter_mut() {
            if info.name == name {
                return Some(plugin.as_mut());
            }
        }
        None
    }

    /// Enable a plugin
    pub fn enable(&mut self, name: &str) -> bool {
        if let Some((info, _)) = self.plugins.iter_mut().find(|(i, _)| i.name == name) {
            info.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a plugin
    pub fn disable(&mut self, name: &str) -> bool {
        if let Some((info, _)) = self.plugins.iter_mut().find(|(i, _)| i.name == name) {
            info.enabled = false;
            true
        } else {
            false
        }
    }

    /// List all plugins
    pub fn list(&self) -> Vec<PluginInfo> {
        self.plugins.iter().map(|(i, _)| i.clone()).collect()
    }

    /// List enabled plugins
    pub fn enabled(&self) -> Vec<&PluginInfo> {
        self.plugins
            .iter()
            .filter(|(i, _)| i.enabled)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get plugins with a specific capability
    pub fn with_capability(&self, cap: PluginCapability) -> Vec<&dyn Plugin> {
        self.plugins
            .iter()
            .filter(|(i, _)| i.enabled && i.capabilities.contains(&cap))
            .map(|(_, p)| p.as_ref())
            .collect()
    }

    /// Process message through all enabled processor plugins
    pub fn process_message(&self, message: &str) -> String {
        let mut current = message.to_string();

        for (info, plugin) in &self.plugins {
            if info.enabled && info.capabilities.contains(&PluginCapability::Processor) {
                if let Some(modified) = plugin.on_before_send(&current) {
                    current = modified;
                }
            }
        }

        current
    }

    /// Process response through all enabled processor plugins
    pub fn process_response(&self, response: &str) -> String {
        let mut current = response.to_string();

        for (info, plugin) in &self.plugins {
            if info.enabled && info.capabilities.contains(&PluginCapability::Processor) {
                if let Some(modified) = plugin.on_after_receive(&current) {
                    current = modified;
                }
            }
        }

        current
    }

    /// Emit an event to all handlers
    pub fn emit(&self, event: &str, data: &serde_json::Value) {
        for (info, plugin) in &self.plugins {
            if info.enabled && info.capabilities.contains(&PluginCapability::EventHandler) {
                // Event handlers would need a separate trait method
                let _ = (event, data, plugin);
            }
        }
    }

    /// Get plugin context
    pub fn context(&self) -> &PluginContext {
        &self.context
    }

    /// Get mutable plugin context
    pub fn context_mut(&mut self) -> &mut PluginContext {
        &mut self.context
    }

    /// Dispatch on_request to all enabled plugins. Returns first short-circuit response if any.
    pub fn dispatch_on_request(
        &self,
        method: &str,
        path: &str,
        headers: &[(String, String)],
    ) -> Option<(String, String)> {
        for (info, plugin) in &self.plugins {
            if info.enabled {
                if let Some(response) = plugin.on_request(method, path, headers) {
                    return Some(response);
                }
            }
        }
        None
    }

    /// Dispatch on_response to all enabled plugins.
    pub fn dispatch_on_response(
        &self,
        method: &str,
        path: &str,
        status: &str,
        duration_ms: f64,
    ) {
        for (info, plugin) in &self.plugins {
            if info.enabled {
                plugin.on_response(method, path, status, duration_ms);
            }
        }
    }

    /// Dispatch on_event to all enabled plugins.
    pub fn dispatch_on_event(&self, event: &str, data: &str) {
        for (info, plugin) in &self.plugins {
            if info.enabled {
                plugin.on_event(event, data);
            }
        }
    }

    /// Configure a plugin
    pub fn configure(&mut self, name: &str, config: serde_json::Value) -> Result<(), String> {
        if let Some((_, plugin)) = self.plugins.iter_mut().find(|(i, _)| i.name == name) {
            plugin.configure(config)
        } else {
            Err(format!("Plugin '{}' not found", name))
        }
    }

    /// Set plugin priority
    pub fn set_priority(&mut self, name: &str, priority: i32) -> bool {
        if let Some((info, _)) = self.plugins.iter_mut().find(|(i, _)| i.name == name) {
            info.priority = priority;
            self.plugins.sort_by_key(|(i, _)| -i.priority);
            true
        } else {
            false
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple message processor plugin
pub struct MessageProcessorPlugin {
    name: String,
    before_send: Option<Box<dyn Fn(&str) -> Option<String> + Send + Sync>>,
    after_receive: Option<Box<dyn Fn(&str) -> Option<String> + Send + Sync>>,
}

impl MessageProcessorPlugin {
    /// Create a new processor plugin
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            before_send: None,
            after_receive: None,
        }
    }

    /// Set before send handler
    pub fn on_before_send<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        self.before_send = Some(Box::new(f));
        self
    }

    /// Set after receive handler
    pub fn on_after_receive<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        self.after_receive = Some(Box::new(f));
        self
    }
}

impl Plugin for MessageProcessorPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::Processor]
    }

    fn on_before_send(&self, message: &str) -> Option<String> {
        self.before_send.as_ref().and_then(|f| f(message))
    }

    fn on_after_receive(&self, response: &str) -> Option<String> {
        self.after_receive.as_ref().and_then(|f| f(response))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A logging plugin for debugging
pub struct LoggingPlugin {
    enabled: bool,
    log_messages: bool,
    log_responses: bool,
}

impl LoggingPlugin {
    pub fn new() -> Self {
        Self {
            enabled: true,
            log_messages: true,
            log_responses: true,
        }
    }
}

impl Default for LoggingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for LoggingPlugin {
    fn name(&self) -> &str {
        "logging"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Logs messages and responses for debugging"
    }

    fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::Processor, PluginCapability::EventHandler]
    }

    fn on_before_send(&self, message: &str) -> Option<String> {
        if self.enabled && self.log_messages {
            log::debug!("[LOG] Sending message ({} chars)", message.len());
        }
        None
    }

    fn on_after_receive(&self, response: &str) -> Option<String> {
        if self.enabled && self.log_responses {
            log::debug!("[LOG] Received response ({} chars)", response.len());
        }
        None
    }

    fn configure(&mut self, config: serde_json::Value) -> Result<(), String> {
        if let Some(enabled) = config.get("enabled").and_then(|v| v.as_bool()) {
            self.enabled = enabled;
        }
        if let Some(log_msg) = config.get("log_messages").and_then(|v| v.as_bool()) {
            self.log_messages = log_msg;
        }
        if let Some(log_resp) = config.get("log_responses").and_then(|v| v.as_bool()) {
            self.log_responses = log_resp;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A plugin that logs every server request/response using `log::info!`
pub struct RequestLoggingPlugin;

impl RequestLoggingPlugin {
    /// Create a new request logging plugin
    pub fn new() -> Self {
        Self
    }
}

impl Default for RequestLoggingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for RequestLoggingPlugin {
    fn name(&self) -> &str {
        "request-logging"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Logs every server request and response"
    }

    fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::EventHandler]
    }

    fn on_response(&self, method: &str, path: &str, status: &str, duration_ms: f64) {
        log::info!(
            "[RequestLoggingPlugin] {} {} -> {} ({:.1}ms)",
            method,
            path,
            status,
            duration_ms
        );
    }

    fn on_event(&self, event: &str, data: &str) {
        log::info!("[RequestLoggingPlugin] event={} data={}", event, data);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A plugin that blocks requests from IPs not in the allowlist
pub struct IpAllowlistPlugin {
    allowed_ips: Vec<String>,
}

impl IpAllowlistPlugin {
    /// Create a new IP allowlist plugin with the given allowed IPs.
    /// If the list is empty, all IPs are allowed.
    pub fn new(ips: Vec<String>) -> Self {
        Self { allowed_ips: ips }
    }
}

impl Plugin for IpAllowlistPlugin {
    fn name(&self) -> &str {
        "ip-allowlist"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Blocks requests from non-allowed IP addresses"
    }

    fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::EventHandler]
    }

    fn on_request(
        &self,
        _method: &str,
        _path: &str,
        headers: &[(String, String)],
    ) -> Option<(String, String)> {
        // If no IPs configured, allow all
        if self.allowed_ips.is_empty() {
            return None;
        }

        // Check x-forwarded-for and x-real-ip headers
        let client_ip = headers.iter().find_map(|(k, v)| {
            let lower = k.to_lowercase();
            if lower == "x-forwarded-for" || lower == "x-real-ip" {
                Some(v.as_str())
            } else {
                None
            }
        });

        match client_ip {
            Some(ip) if self.allowed_ips.iter().any(|allowed| allowed == ip) => None,
            Some(_) => Some((
                "403 Forbidden".to_string(),
                "{\"error\":\"IP address not allowed\"}".to_string(),
            )),
            // No IP header present and allowlist is non-empty: block
            None => Some((
                "403 Forbidden".to_string(),
                "{\"error\":\"IP address not allowed\"}".to_string(),
            )),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A plugin that collects request metrics (counts and latencies)
pub struct MetricsCollectorPlugin {
    request_count: AtomicU64,
    total_latency_ms: AtomicU64,
}

impl MetricsCollectorPlugin {
    /// Create a new metrics collector plugin
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
        }
    }

    /// Get the total number of requests observed
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// Get the average latency in milliseconds across all observed requests
    pub fn avg_latency_ms(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        total as f64 / count as f64
    }
}

impl Default for MetricsCollectorPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for MetricsCollectorPlugin {
    fn name(&self) -> &str {
        "metrics-collector"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Collects request counts and latencies"
    }

    fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::EventHandler]
    }

    fn on_response(&self, _method: &str, _path: &str, _status: &str, duration_ms: f64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(duration_ms as u64, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        loaded: bool,
    }

    impl TestPlugin {
        fn new() -> Self {
            Self { loaded: false }
        }
    }

    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn on_load(&mut self, _ctx: &PluginContext) -> Result<(), String> {
            self.loaded = true;
            Ok(())
        }

        fn on_unload(&mut self) {
            self.loaded = false;
        }

        fn capabilities(&self) -> Vec<PluginCapability> {
            vec![PluginCapability::Processor]
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_plugin_registration() {
        let mut manager = PluginManager::new();

        manager.register(Box::new(TestPlugin::new())).unwrap();

        assert_eq!(manager.list().len(), 1);
        assert!(manager.get("test").is_some());
    }

    #[test]
    fn test_plugin_enable_disable() {
        let mut manager = PluginManager::new();
        manager.register(Box::new(TestPlugin::new())).unwrap();

        assert!(manager.disable("test"));
        assert_eq!(manager.enabled().len(), 0);

        assert!(manager.enable("test"));
        assert_eq!(manager.enabled().len(), 1);
    }

    #[test]
    fn test_plugin_unregister() {
        let mut manager = PluginManager::new();
        manager.register(Box::new(TestPlugin::new())).unwrap();

        assert!(manager.unregister("test"));
        assert!(manager.get("test").is_none());
    }

    #[test]
    fn test_message_processor() {
        let mut manager = PluginManager::new();

        let processor =
            MessageProcessorPlugin::new("uppercase").on_before_send(|msg| Some(msg.to_uppercase()));

        manager.register(Box::new(processor)).unwrap();

        let result = manager.process_message("hello");
        assert_eq!(result, "HELLO");
    }

    #[test]
    fn test_duplicate_registration() {
        let mut manager = PluginManager::new();

        manager.register(Box::new(TestPlugin::new())).unwrap();
        let result = manager.register(Box::new(TestPlugin::new()));

        assert!(result.is_err());
    }

    #[test]
    fn test_plugin_context() {
        let ctx = PluginContext::new();

        ctx.set_state("key", "value".to_string());
        let value: Option<String> = ctx.get_state("key");

        assert_eq!(value, Some("value".to_string()));
    }

    // --- Phase 8: Server integration, event dispatch, built-in plugins (24 tests) ---

    #[test]
    fn test_request_logging_plugin_name() {
        let plugin = RequestLoggingPlugin::new();
        assert_eq!(plugin.name(), "request-logging");
        assert_eq!(plugin.version(), "1.0.0");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_request_logging_plugin_on_response() {
        let plugin = RequestLoggingPlugin::new();
        // Should not panic
        plugin.on_response("GET", "/api/health", "200 OK", 12.5);
        plugin.on_response("POST", "/api/chat", "500 Internal Server Error", 250.0);
    }

    #[test]
    fn test_ip_allowlist_plugin_allows_all_when_empty() {
        let plugin = IpAllowlistPlugin::new(vec![]);
        let headers = vec![
            ("x-forwarded-for".to_string(), "1.2.3.4".to_string()),
        ];
        assert!(plugin.on_request("GET", "/", &headers).is_none());
    }

    #[test]
    fn test_ip_allowlist_plugin_blocks_unknown() {
        let plugin = IpAllowlistPlugin::new(vec!["10.0.0.1".to_string()]);
        let headers = vec![
            ("x-forwarded-for".to_string(), "192.168.1.1".to_string()),
        ];
        let result = plugin.on_request("GET", "/", &headers);
        assert!(result.is_some());
        let (status, body) = result.unwrap();
        assert_eq!(status, "403 Forbidden");
        assert!(body.contains("not allowed"));
    }

    #[test]
    fn test_ip_allowlist_plugin_allows_known() {
        let plugin = IpAllowlistPlugin::new(vec!["10.0.0.1".to_string()]);
        let headers = vec![
            ("x-forwarded-for".to_string(), "10.0.0.1".to_string()),
        ];
        assert!(plugin.on_request("GET", "/", &headers).is_none());
    }

    #[test]
    fn test_ip_allowlist_plugin_name() {
        let plugin = IpAllowlistPlugin::new(vec![]);
        assert_eq!(plugin.name(), "ip-allowlist");
        assert_eq!(plugin.version(), "1.0.0");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_metrics_collector_initial_zero() {
        let plugin = MetricsCollectorPlugin::new();
        assert_eq!(plugin.request_count(), 0);
        assert_eq!(plugin.avg_latency_ms(), 0.0);
    }

    #[test]
    fn test_metrics_collector_counts_requests() {
        let plugin = MetricsCollectorPlugin::new();
        plugin.on_response("GET", "/a", "200 OK", 10.0);
        plugin.on_response("POST", "/b", "201 Created", 20.0);
        plugin.on_response("GET", "/c", "404 Not Found", 5.0);
        assert_eq!(plugin.request_count(), 3);
    }

    #[test]
    fn test_metrics_collector_avg_latency() {
        let plugin = MetricsCollectorPlugin::new();
        plugin.on_response("GET", "/a", "200 OK", 10.0);
        plugin.on_response("GET", "/b", "200 OK", 20.0);
        plugin.on_response("GET", "/c", "200 OK", 30.0);
        // total = 60ms (10+20+30 truncated to u64), count = 3 → avg = 20.0
        assert_eq!(plugin.request_count(), 3);
        let avg = plugin.avg_latency_ms();
        assert!((avg - 20.0).abs() < 1.0);
    }

    #[test]
    fn test_dispatch_on_request_no_plugins() {
        let manager = PluginManager::new();
        let result = manager.dispatch_on_request("GET", "/", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_dispatch_on_request_passes_through() {
        let mut manager = PluginManager::new();
        manager.register(Box::new(RequestLoggingPlugin::new())).unwrap();
        // RequestLoggingPlugin returns None from on_request (uses default)
        let result = manager.dispatch_on_request("GET", "/health", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_dispatch_on_request_short_circuits() {
        let mut manager = PluginManager::new();
        let allowlist = IpAllowlistPlugin::new(vec!["10.0.0.1".to_string()]);
        manager.register(Box::new(allowlist)).unwrap();
        manager.register(Box::new(RequestLoggingPlugin::new())).unwrap();

        let headers = vec![
            ("x-forwarded-for".to_string(), "evil.ip".to_string()),
        ];
        let result = manager.dispatch_on_request("GET", "/secret", &headers);
        assert!(result.is_some());
        let (status, _) = result.unwrap();
        assert_eq!(status, "403 Forbidden");
    }

    #[test]
    fn test_dispatch_on_response_all_plugins() {
        let mut manager = PluginManager::new();
        let metrics = MetricsCollectorPlugin::new();
        // Store a reference-counted pointer to check later
        let metrics_ptr = &metrics as *const MetricsCollectorPlugin;
        manager.register(Box::new(RequestLoggingPlugin::new())).unwrap();
        manager.register(Box::new(metrics)).unwrap();

        manager.dispatch_on_response("GET", "/api", "200 OK", 15.0);

        // Verify via downcast
        let plugin = manager.get("metrics-collector").unwrap();
        let mc = plugin.as_any().downcast_ref::<MetricsCollectorPlugin>().unwrap();
        assert_eq!(mc.request_count(), 1);
        let _ = metrics_ptr; // suppress unused warning
    }

    #[test]
    fn test_dispatch_on_event_all_plugins() {
        let mut manager = PluginManager::new();
        manager.register(Box::new(RequestLoggingPlugin::new())).unwrap();
        manager.register(Box::new(MetricsCollectorPlugin::new())).unwrap();
        // Should not panic
        manager.dispatch_on_event("startup", "server started on port 8080");
        manager.dispatch_on_event("shutdown", "graceful shutdown");
    }

    #[test]
    fn test_plugin_trait_default_on_request() {
        let plugin = TestPlugin::new();
        let result = plugin.on_request("GET", "/", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_plugin_trait_default_on_response() {
        let plugin = TestPlugin::new();
        // Should not panic — default impl is empty
        plugin.on_response("GET", "/", "200 OK", 1.0);
    }

    #[test]
    fn test_plugin_trait_default_on_event() {
        let plugin = TestPlugin::new();
        // Should not panic — default impl is empty
        plugin.on_event("test-event", "data");
    }

    #[test]
    fn test_builtin_plugins_implement_trait() {
        // Verify all 3 built-in plugins can be boxed as dyn Plugin
        let plugins: Vec<Box<dyn Plugin>> = vec![
            Box::new(RequestLoggingPlugin::new()),
            Box::new(IpAllowlistPlugin::new(vec![])),
            Box::new(MetricsCollectorPlugin::new()),
        ];
        assert_eq!(plugins.len(), 3);
        assert_eq!(plugins[0].name(), "request-logging");
        assert_eq!(plugins[1].name(), "ip-allowlist");
        assert_eq!(plugins[2].name(), "metrics-collector");
    }

    #[test]
    fn test_ip_allowlist_with_x_forwarded_for() {
        let plugin = IpAllowlistPlugin::new(vec!["192.168.1.100".to_string()]);
        let headers = vec![
            ("X-Forwarded-For".to_string(), "192.168.1.100".to_string()),
        ];
        assert!(plugin.on_request("GET", "/", &headers).is_none());
    }

    #[test]
    fn test_ip_allowlist_with_x_real_ip() {
        let plugin = IpAllowlistPlugin::new(vec!["172.16.0.5".to_string()]);
        let headers = vec![
            ("X-Real-IP".to_string(), "172.16.0.5".to_string()),
        ];
        assert!(plugin.on_request("GET", "/", &headers).is_none());
    }

    #[test]
    fn test_metrics_collector_thread_safe() {
        // MetricsCollectorPlugin must be Send + Sync since Plugin: Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricsCollectorPlugin>();
    }

    #[test]
    fn test_request_logging_plugin_event() {
        let plugin = RequestLoggingPlugin::new();
        // Should not panic
        plugin.on_event("startup", "listening on 0.0.0.0:8080");
        plugin.on_event("error", "connection refused");
        plugin.on_event("shutdown", "bye");
    }

    #[test]
    fn test_plugin_manager_with_all_builtins() {
        let mut manager = PluginManager::new();
        manager
            .register(Box::new(RequestLoggingPlugin::new()))
            .unwrap();
        manager
            .register(Box::new(IpAllowlistPlugin::new(vec!["127.0.0.1".to_string()])))
            .unwrap();
        manager
            .register(Box::new(MetricsCollectorPlugin::new()))
            .unwrap();

        assert_eq!(manager.list().len(), 3);
        assert!(manager.get("request-logging").is_some());
        assert!(manager.get("ip-allowlist").is_some());
        assert!(manager.get("metrics-collector").is_some());

        // Dispatch a full request cycle
        let headers = vec![
            ("x-forwarded-for".to_string(), "127.0.0.1".to_string()),
        ];
        let blocked = manager.dispatch_on_request("GET", "/test", &headers);
        assert!(blocked.is_none()); // 127.0.0.1 is allowed

        manager.dispatch_on_response("GET", "/test", "200 OK", 42.0);
        manager.dispatch_on_event("request_complete", "/test");

        let mc = manager
            .get("metrics-collector")
            .unwrap()
            .as_any()
            .downcast_ref::<MetricsCollectorPlugin>()
            .unwrap();
        assert_eq!(mc.request_count(), 1);
    }

    #[test]
    fn test_dispatch_order_matters() {
        // IpAllowlist should short-circuit BEFORE MetricsCollector sees the request
        let mut manager = PluginManager::new();
        let allowlist = IpAllowlistPlugin::new(vec!["10.0.0.1".to_string()]);
        manager.register(Box::new(allowlist)).unwrap();
        manager.register(Box::new(MetricsCollectorPlugin::new())).unwrap();

        let headers = vec![
            ("x-forwarded-for".to_string(), "bad-ip".to_string()),
        ];

        // on_request short-circuits at IpAllowlist
        let result = manager.dispatch_on_request("GET", "/", &headers);
        assert!(result.is_some());

        // Since we short-circuited, if the server logic doesn't call dispatch_on_response,
        // metrics should remain at zero
        let mc = manager
            .get("metrics-collector")
            .unwrap()
            .as_any()
            .downcast_ref::<MetricsCollectorPlugin>()
            .unwrap();
        assert_eq!(mc.request_count(), 0);
    }
}
