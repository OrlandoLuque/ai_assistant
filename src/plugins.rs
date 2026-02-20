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
}
