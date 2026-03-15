//! Pub/sub event system for inter-module communication.
//!
//! Provides [`EventBus`], [`EventHandler`], [`EventFilter`], and [`AiEvent`] for
//! typed lifecycle hooks and monitoring across all major `AiAssistant` operations.
//! Handlers receive events synchronously on the calling thread.
//!
//! Part of the core feature set.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ============================================================================
// Event Types
// ============================================================================

/// All events emitted by AiAssistant during its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AiEvent {
    // --- Response lifecycle ---
    /// User message was sent to the model
    MessageSent {
        content_length: usize,
        has_knowledge: bool,
    },
    /// A streaming chunk was received
    ResponseChunk { chunk_length: usize },
    /// The complete response was received
    ResponseComplete { response_length: usize },
    /// The response was cancelled by the user
    ResponseCancelled { partial_length: usize },
    /// An error occurred during generation
    ResponseError { error: String },

    // --- Provider events ---
    /// A provider attempt was started
    ProviderAttempt { provider: String, model: String },
    /// A provider failed
    ProviderFailed { provider: String, error: String },
    /// Fallback was triggered to an alternative provider
    FallbackTriggered {
        from_provider: String,
        to_provider: String,
    },

    // --- Session events ---
    /// A new session was created
    SessionCreated { session_id: String },
    /// A session was loaded
    SessionLoaded { session_id: String },
    /// A session was saved
    SessionSaved {
        session_id: String,
        message_count: usize,
    },
    /// A session was deleted
    SessionDeleted { session_id: String },

    // --- Context events ---
    /// Context usage reached the warning threshold
    ContextWarning { usage_percent: f32 },
    /// Context usage reached the critical threshold
    ContextCritical { usage_percent: f32 },
    /// Conversation compaction started
    CompactionStarted { message_count: usize },
    /// Conversation compaction completed
    CompactionCompleted { removed_count: usize },

    // --- Model events ---
    /// Model discovery started
    ModelsDiscoveryStarted,
    /// Models were discovered
    ModelsDiscovered { count: usize },
    /// A model was selected
    ModelSelected { name: String, provider: String },

    // --- RAG events ---
    /// A document was indexed into the knowledge base
    DocumentIndexed { source: String, chunks: usize },
    /// Knowledge was retrieved for a query
    KnowledgeRetrieved {
        query_length: usize,
        chunks_found: usize,
    },

    // --- Tool events ---
    /// A tool was executed
    ToolExecuted {
        name: String,
        success: bool,
        duration_ms: u64,
    },
}

impl AiEvent {
    /// Returns the event category as a static string.
    pub fn category(&self) -> &'static str {
        match self {
            AiEvent::MessageSent { .. }
            | AiEvent::ResponseChunk { .. }
            | AiEvent::ResponseComplete { .. }
            | AiEvent::ResponseCancelled { .. }
            | AiEvent::ResponseError { .. } => "response",

            AiEvent::ProviderAttempt { .. }
            | AiEvent::ProviderFailed { .. }
            | AiEvent::FallbackTriggered { .. } => "provider",

            AiEvent::SessionCreated { .. }
            | AiEvent::SessionLoaded { .. }
            | AiEvent::SessionSaved { .. }
            | AiEvent::SessionDeleted { .. } => "session",

            AiEvent::ContextWarning { .. }
            | AiEvent::ContextCritical { .. }
            | AiEvent::CompactionStarted { .. }
            | AiEvent::CompactionCompleted { .. } => "context",

            AiEvent::ModelsDiscoveryStarted
            | AiEvent::ModelsDiscovered { .. }
            | AiEvent::ModelSelected { .. } => "model",

            AiEvent::DocumentIndexed { .. } | AiEvent::KnowledgeRetrieved { .. } => "rag",

            AiEvent::ToolExecuted { .. } => "tool",
        }
    }

    /// Returns the event name as a static string.
    pub fn name(&self) -> &'static str {
        match self {
            AiEvent::MessageSent { .. } => "message_sent",
            AiEvent::ResponseChunk { .. } => "response_chunk",
            AiEvent::ResponseComplete { .. } => "response_complete",
            AiEvent::ResponseCancelled { .. } => "response_cancelled",
            AiEvent::ResponseError { .. } => "response_error",
            AiEvent::ProviderAttempt { .. } => "provider_attempt",
            AiEvent::ProviderFailed { .. } => "provider_failed",
            AiEvent::FallbackTriggered { .. } => "fallback_triggered",
            AiEvent::SessionCreated { .. } => "session_created",
            AiEvent::SessionLoaded { .. } => "session_loaded",
            AiEvent::SessionSaved { .. } => "session_saved",
            AiEvent::SessionDeleted { .. } => "session_deleted",
            AiEvent::ContextWarning { .. } => "context_warning",
            AiEvent::ContextCritical { .. } => "context_critical",
            AiEvent::CompactionStarted { .. } => "compaction_started",
            AiEvent::CompactionCompleted { .. } => "compaction_completed",
            AiEvent::ModelsDiscoveryStarted => "models_discovery_started",
            AiEvent::ModelsDiscovered { .. } => "models_discovered",
            AiEvent::ModelSelected { .. } => "model_selected",
            AiEvent::DocumentIndexed { .. } => "document_indexed",
            AiEvent::KnowledgeRetrieved { .. } => "knowledge_retrieved",
            AiEvent::ToolExecuted { .. } => "tool_executed",
        }
    }
}

// ============================================================================
// Event Handler Trait
// ============================================================================

/// Trait for receiving events from AiAssistant.
///
/// Implement this trait to react to lifecycle events. Handlers are called
/// synchronously on the thread that emits the event.
///
/// # Example
///
/// ```rust
/// use ai_assistant::events::{AiEvent, EventHandler};
///
/// struct MetricsHandler;
///
/// impl EventHandler for MetricsHandler {
///     fn on_event(&self, event: &AiEvent) {
///         match event {
///             AiEvent::ResponseComplete { response_length } => {
///                 println!("Response: {} chars", response_length);
///             }
///             AiEvent::ProviderFailed { provider, error } => {
///                 println!("Provider {} failed: {}", provider, error);
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
pub trait EventHandler: Send + Sync {
    /// Called when an event is emitted.
    fn on_event(&self, event: &AiEvent);
}

// ============================================================================
// Closure-based handler
// ============================================================================

/// An event handler backed by a closure. Created via `EventBus::on()`.
struct ClosureHandler<F: Fn(&AiEvent) + Send + Sync> {
    callback: F,
}

impl<F: Fn(&AiEvent) + Send + Sync> EventHandler for ClosureHandler<F> {
    fn on_event(&self, event: &AiEvent) {
        (self.callback)(event);
    }
}

// ============================================================================
// Category filter handler
// ============================================================================

/// An event handler that only forwards events of certain categories.
pub struct FilteredHandler {
    inner: Box<dyn EventHandler>,
    categories: Vec<&'static str>,
}

impl FilteredHandler {
    /// Create a handler that only receives events from the given categories.
    ///
    /// Valid categories: "response", "provider", "session", "context", "model", "rag", "tool".
    pub fn new(handler: Box<dyn EventHandler>, categories: Vec<&'static str>) -> Self {
        Self {
            inner: handler,
            categories,
        }
    }
}

impl EventHandler for FilteredHandler {
    fn on_event(&self, event: &AiEvent) {
        if self.categories.contains(&event.category()) {
            self.inner.on_event(event);
        }
    }
}

// ============================================================================
// Event Bus
// ============================================================================

/// Central event bus for AiAssistant. Thread-safe, supports multiple handlers.
///
/// # Example
///
/// ```rust
/// use ai_assistant::events::{AiEvent, EventBus, EventHandler};
///
/// let mut bus = EventBus::new();
///
/// // Register a closure handler
/// bus.on(|event| {
///     println!("[{}] {}", event.category(), event.name());
/// });
///
/// // Emit an event
/// bus.emit(AiEvent::ModelsDiscoveryStarted);
/// ```
pub struct EventBus {
    handlers: Vec<Arc<dyn EventHandler>>,
    /// Event history for replay/debugging (bounded)
    history: Arc<Mutex<Vec<TimestampedEvent>>>,
    /// Maximum number of events to keep in history (0 = disabled)
    max_history: usize,
}

/// An event with its emission timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedEvent {
    pub event: AiEvent,
    pub timestamp_ms: u64,
}

impl EventBus {
    /// Create a new event bus with no handlers and history disabled.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            history: Arc::new(Mutex::new(Vec::new())),
            max_history: 0,
        }
    }

    /// Create a new event bus with history tracking enabled.
    pub fn with_history(max_events: usize) -> Self {
        Self {
            handlers: Vec::new(),
            history: Arc::new(Mutex::new(Vec::with_capacity(max_events.min(1024)))),
            max_history: max_events,
        }
    }

    /// Register a trait-based event handler.
    pub fn add_handler(&mut self, handler: Arc<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    /// Register a closure as an event handler.
    pub fn on<F>(&mut self, callback: F)
    where
        F: Fn(&AiEvent) + Send + Sync + 'static,
    {
        self.handlers.push(Arc::new(ClosureHandler { callback }));
    }

    /// Register a handler that only receives events from specific categories.
    pub fn on_category(&mut self, categories: Vec<&'static str>, handler: Box<dyn EventHandler>) {
        self.handlers
            .push(Arc::new(FilteredHandler::new(handler, categories)));
    }

    /// Remove all handlers.
    pub fn clear_handlers(&mut self) {
        self.handlers.clear();
    }

    /// Returns the number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Emit an event to all registered handlers.
    ///
    /// Handlers are called synchronously in registration order.
    /// If a handler panics, the panic is caught and logged — other handlers
    /// still receive the event.
    pub fn emit(&self, event: AiEvent) {
        // Store in history if enabled
        if self.max_history > 0 {
            let mut history = self.history.lock().unwrap_or_else(|e| e.into_inner());
            if history.len() >= self.max_history {
                history.remove(0);
            }
            history.push(TimestampedEvent {
                event: event.clone(),
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            });
        }

        // Dispatch to all handlers
        for handler in &self.handlers {
            let handler = handler.clone();
            // Catch panics to prevent one handler from breaking others
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handler.on_event(&event);
            }));
            if let Err(_) = result {
                log::error!("Event handler panicked on event: {}", event.name());
            }
        }
    }

    /// Get a snapshot of the event history.
    pub fn history(&self) -> Vec<TimestampedEvent> {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clear the event history.
    pub fn clear_history(&self) {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Count events of a specific category in history.
    pub fn count_events(&self, category: &str) -> usize {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|e| e.event.category() == category)
            .count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in handlers
// ============================================================================

/// A logging event handler that outputs events via the `log` crate.
pub struct LoggingHandler {
    /// Minimum log level for events
    level: LogLevel,
}

/// Log level for the LoggingHandler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogLevel {
    /// Log all events
    Debug,
    /// Log important events (responses, errors, session changes)
    Info,
    /// Log only warnings and errors
    Warn,
}

impl LoggingHandler {
    /// Create a handler that logs at the given level.
    pub fn new(level: LogLevel) -> Self {
        Self { level }
    }
}

impl EventHandler for LoggingHandler {
    fn on_event(&self, event: &AiEvent) {
        match event {
            AiEvent::ResponseError { error } | AiEvent::ProviderFailed { error, .. } => {
                log::error!("[event:{}] {}", event.name(), error);
            }
            AiEvent::ContextCritical { usage_percent } => {
                log::warn!("[event:context_critical] usage: {:.1}%", usage_percent);
            }
            AiEvent::ContextWarning { usage_percent } => {
                if self.level != LogLevel::Warn {
                    log::warn!("[event:context_warning] usage: {:.1}%", usage_percent);
                }
            }
            AiEvent::FallbackTriggered {
                from_provider,
                to_provider,
            } => {
                log::warn!("[event:fallback] {} -> {}", from_provider, to_provider);
            }
            _ => {
                if self.level == LogLevel::Debug {
                    log::debug!("[event:{}] {:?}", event.name(), event);
                } else if self.level == LogLevel::Info {
                    match event {
                        AiEvent::ResponseChunk { .. } => {} // too noisy
                        _ => log::info!("[event:{}]", event.name()),
                    }
                }
            }
        }
    }
}

/// A collecting handler that stores events for later inspection.
/// Useful for testing.
pub struct CollectingHandler {
    events: Arc<Mutex<Vec<AiEvent>>>,
}

impl CollectingHandler {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get a clone of all collected events.
    pub fn events(&self) -> Vec<AiEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Count collected events.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if no events have been collected.
    pub fn is_empty(&self) -> bool {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Clear all collected events.
    pub fn clear(&self) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Get a shared reference for use with EventBus.
    pub fn shared(&self) -> Arc<Mutex<Vec<AiEvent>>> {
        self.events.clone()
    }
}

impl Default for CollectingHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl EventHandler for CollectingHandler {
    fn on_event(&self, event: &AiEvent) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }
}

// ============================================================================
// Utility: event timing
// ============================================================================

/// Helper to measure operation duration and emit a timed event.
pub struct EventTimer {
    start: Instant,
    name: String,
}

impl EventTimer {
    /// Start timing an operation.
    pub fn start(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Finish timing and return elapsed milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Create a ToolExecuted event with the elapsed time.
    pub fn tool_event(self, success: bool) -> AiEvent {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        AiEvent::ToolExecuted {
            name: self.name,
            success,
            duration_ms,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_category() {
        let event = AiEvent::MessageSent {
            content_length: 10,
            has_knowledge: false,
        };
        assert_eq!(event.category(), "response");

        let event = AiEvent::ProviderFailed {
            provider: "ollama".to_string(),
            error: "timeout".to_string(),
        };
        assert_eq!(event.category(), "provider");

        let event = AiEvent::SessionCreated {
            session_id: "s1".to_string(),
        };
        assert_eq!(event.category(), "session");
    }

    #[test]
    fn test_event_name() {
        let event = AiEvent::ModelsDiscoveryStarted;
        assert_eq!(event.name(), "models_discovery_started");

        let event = AiEvent::CompactionCompleted { removed_count: 5 };
        assert_eq!(event.name(), "compaction_completed");
    }

    #[test]
    fn test_event_bus_emit() {
        let mut bus = EventBus::new();
        let counter = Arc::new(Mutex::new(0usize));
        let counter_clone = counter.clone();

        bus.on(move |_event| {
            *counter_clone.lock().unwrap() += 1;
        });

        bus.emit(AiEvent::ModelsDiscoveryStarted);
        bus.emit(AiEvent::ModelsDiscovered { count: 5 });

        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn test_event_bus_multiple_handlers() {
        let mut bus = EventBus::new();
        let counter1 = Arc::new(Mutex::new(0usize));
        let counter2 = Arc::new(Mutex::new(0usize));
        let c1 = counter1.clone();
        let c2 = counter2.clone();

        bus.on(move |_| {
            *c1.lock().unwrap() += 1;
        });
        bus.on(move |_| {
            *c2.lock().unwrap() += 1;
        });

        bus.emit(AiEvent::ModelsDiscoveryStarted);

        assert_eq!(*counter1.lock().unwrap(), 1);
        assert_eq!(*counter2.lock().unwrap(), 1);
    }

    #[test]
    fn test_event_bus_history() {
        let bus = EventBus::with_history(3);

        bus.emit(AiEvent::ModelsDiscoveryStarted);
        bus.emit(AiEvent::ModelsDiscovered { count: 2 });
        bus.emit(AiEvent::ModelsDiscoveryStarted);
        bus.emit(AiEvent::ModelsDiscovered { count: 3 });

        let history = bus.history();
        // Max 3, oldest dropped
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].event.name(), "models_discovered");
    }

    #[test]
    fn test_event_bus_count_events() {
        let bus = EventBus::with_history(100);

        bus.emit(AiEvent::ModelsDiscoveryStarted);
        bus.emit(AiEvent::ModelsDiscovered { count: 2 });
        bus.emit(AiEvent::SessionCreated {
            session_id: "s1".to_string(),
        });

        assert_eq!(bus.count_events("model"), 2);
        assert_eq!(bus.count_events("session"), 1);
        assert_eq!(bus.count_events("provider"), 0);
    }

    #[test]
    fn test_collecting_handler() {
        let mut bus = EventBus::new();
        let collector = Arc::new(CollectingHandler::new());
        bus.add_handler(collector.clone());

        bus.emit(AiEvent::MessageSent {
            content_length: 50,
            has_knowledge: true,
        });
        bus.emit(AiEvent::ResponseComplete {
            response_length: 200,
        });

        assert_eq!(collector.len(), 2);
        let events = collector.events();
        assert_eq!(events[0].name(), "message_sent");
        assert_eq!(events[1].name(), "response_complete");
    }

    #[test]
    fn test_filtered_handler() {
        let mut bus = EventBus::new();
        let collector = Arc::new(CollectingHandler::new());
        let filtered = FilteredHandler::new(
            Box::new(CollectingHandler {
                events: collector.shared(),
            }),
            vec!["session"],
        );
        bus.add_handler(Arc::new(filtered));

        bus.emit(AiEvent::ModelsDiscoveryStarted); // model, filtered out
        bus.emit(AiEvent::SessionCreated {
            session_id: "s1".to_string(),
        }); // session, passes

        assert_eq!(collector.len(), 1);
        assert_eq!(collector.events()[0].name(), "session_created");
    }

    #[test]
    fn test_event_timer() {
        let timer = EventTimer::start("test_tool");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let event = timer.tool_event(true);

        match event {
            AiEvent::ToolExecuted {
                name,
                success,
                duration_ms,
            } => {
                assert_eq!(name, "test_tool");
                assert!(success);
                assert!(duration_ms >= 5); // at least 5ms
            }
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn test_handler_panic_doesnt_break_others() {
        let mut bus = EventBus::new();
        let counter = Arc::new(Mutex::new(0usize));
        let c = counter.clone();

        // Handler that panics
        bus.on(|_| {
            panic!("intentional test panic");
        });

        // Handler after the panicking one
        bus.on(move |_| {
            *c.lock().unwrap() += 1;
        });

        bus.emit(AiEvent::ModelsDiscoveryStarted);

        // Second handler still ran
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn test_clear_handlers() {
        let mut bus = EventBus::with_history(10);
        let counter = Arc::new(Mutex::new(0usize));
        let c = counter.clone();

        bus.on(move |_| {
            *c.lock().unwrap() += 1;
        });

        bus.emit(AiEvent::ModelsDiscoveryStarted);
        assert_eq!(*counter.lock().unwrap(), 1);

        bus.clear_handlers();
        assert_eq!(bus.handler_count(), 0);

        bus.emit(AiEvent::ModelsDiscoveryStarted);
        assert_eq!(*counter.lock().unwrap(), 1); // no change
    }

    #[test]
    fn test_event_serialization() {
        let event = AiEvent::ProviderFailed {
            provider: "ollama".to_string(),
            error: "timeout".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ollama"));
        assert!(json.contains("timeout"));

        let deserialized: AiEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name(), "provider_failed");
    }
}
