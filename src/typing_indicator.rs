//! Typing indicators
//!
//! Show typing/processing indicators during AI responses.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Typing indicator state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypingState {
    Idle,
    Typing,
    Processing,
    Generating,
    Finished,
    Error,
}

impl TypingState {
    pub fn display_text(&self) -> &'static str {
        match self {
            Self::Idle => "",
            Self::Typing => "AI is typing...",
            Self::Processing => "Processing...",
            Self::Generating => "Generating response...",
            Self::Finished => "",
            Self::Error => "Error occurred",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Typing | Self::Processing | Self::Generating)
    }
}

/// Typing indicator
#[derive(Debug, Clone)]
pub struct TypingIndicator {
    state: Arc<Mutex<TypingState>>,
    started_at: Arc<Mutex<Option<Instant>>>,
    message: Arc<Mutex<Option<String>>>,
}

impl TypingIndicator {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(TypingState::Idle)),
            started_at: Arc::new(Mutex::new(None)),
            message: Arc::new(Mutex::new(None)),
        }
    }

    pub fn start(&self, state: TypingState) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = state;
        *self.started_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    pub fn stop(&self) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = TypingState::Finished;
        *self.started_at.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    pub fn error(&self, message: Option<String>) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = TypingState::Error;
        *self.message.lock().unwrap_or_else(|e| e.into_inner()) = message;
    }

    pub fn state(&self) -> TypingState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn is_active(&self) -> bool {
        self.state().is_active()
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map(|t| t.elapsed())
    }

    pub fn display(&self) -> String {
        let state = self.state();
        let base = state.display_text();

        if let Some(elapsed) = self.elapsed() {
            if elapsed.as_secs() > 2 {
                format!("{} ({:.1}s)", base, elapsed.as_secs_f64())
            } else {
                base.to_string()
            }
        } else {
            base.to_string()
        }
    }
}

impl Default for TypingIndicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Animated typing indicator
pub struct AnimatedIndicator {
    frames: Vec<&'static str>,
    current: usize,
    interval: Duration,
    last_update: Instant,
}

impl AnimatedIndicator {
    pub fn dots() -> Self {
        Self {
            frames: vec![".", "..", "..."],
            current: 0,
            interval: Duration::from_millis(500),
            last_update: Instant::now(),
        }
    }

    pub fn spinner() -> Self {
        Self {
            frames: vec!["|", "/", "-", "\\"],
            current: 0,
            interval: Duration::from_millis(100),
            last_update: Instant::now(),
        }
    }

    pub fn braille() -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            current: 0,
            interval: Duration::from_millis(80),
            last_update: Instant::now(),
        }
    }

    pub fn tick(&mut self) -> &str {
        if self.last_update.elapsed() >= self.interval {
            self.current = (self.current + 1) % self.frames.len();
            self.last_update = Instant::now();
        }
        self.frames[self.current]
    }

    pub fn current(&self) -> &str {
        self.frames[self.current]
    }

    pub fn reset(&mut self) {
        self.current = 0;
        self.last_update = Instant::now();
    }
}

/// Progress indicator with percentage
pub struct ProgressIndicator {
    total: u64,
    current: u64,
    message: String,
    started_at: Instant,
}

impl ProgressIndicator {
    pub fn new(total: u64, message: &str) -> Self {
        Self {
            total,
            current: 0,
            message: message.to_string(),
            started_at: Instant::now(),
        }
    }

    pub fn increment(&mut self, amount: u64) {
        self.current = (self.current + amount).min(self.total);
    }

    pub fn set(&mut self, value: u64) {
        self.current = value.min(self.total);
    }

    pub fn percentage(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.current as f64 / self.total as f64) * 100.0
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn eta(&self) -> Option<Duration> {
        if self.current == 0 {
            return None;
        }

        let elapsed = self.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;
        let remaining = (self.total - self.current) as f64;

        Some(Duration::from_secs_f64(remaining / rate))
    }

    pub fn display(&self) -> String {
        let pct = self.percentage();
        let bar_width = 20;
        let filled = (pct / 100.0 * bar_width as f64) as usize;

        let bar: String = std::iter::repeat_n('█', filled)
            .chain(std::iter::repeat_n('░', bar_width - filled))
            .collect();

        format!("{} [{}] {:.1}%", self.message, bar, pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typing_indicator() {
        let indicator = TypingIndicator::new();

        assert_eq!(indicator.state(), TypingState::Idle);
        assert!(!indicator.is_active());

        indicator.start(TypingState::Typing);
        assert!(indicator.is_active());
        assert!(indicator.display().contains("typing"));

        indicator.stop();
        assert!(!indicator.is_active());
    }

    #[test]
    fn test_typing_state_display_text() {
        assert_eq!(TypingState::Idle.display_text(), "");
        assert!(TypingState::Typing.display_text().contains("typing"));
        assert!(TypingState::Processing.display_text().contains("Processing"));
        assert!(TypingState::Generating.display_text().contains("Generating"));
        assert_eq!(TypingState::Finished.display_text(), "");
        assert!(TypingState::Error.display_text().contains("Error"));
    }

    #[test]
    fn test_typing_state_is_active() {
        assert!(!TypingState::Idle.is_active());
        assert!(TypingState::Typing.is_active());
        assert!(TypingState::Processing.is_active());
        assert!(TypingState::Generating.is_active());
        assert!(!TypingState::Finished.is_active());
        assert!(!TypingState::Error.is_active());
    }

    #[test]
    fn test_indicator_error_state() {
        let indicator = TypingIndicator::new();
        indicator.error(Some("timeout".to_string()));
        assert_eq!(indicator.state(), TypingState::Error);
        assert!(!indicator.is_active());
    }

    #[test]
    fn test_indicator_elapsed() {
        let indicator = TypingIndicator::new();
        assert!(indicator.elapsed().is_none());
        indicator.start(TypingState::Processing);
        assert!(indicator.elapsed().is_some());
        indicator.stop();
        assert!(indicator.elapsed().is_none());
    }

    #[test]
    fn test_animated_dots() {
        let mut anim = AnimatedIndicator::dots();
        assert_eq!(anim.current(), ".");
        // tick won't advance because interval not elapsed
        let _ = anim.tick();
        anim.reset();
        assert_eq!(anim.current(), ".");
    }

    #[test]
    fn test_animated_spinner() {
        let anim = AnimatedIndicator::spinner();
        assert_eq!(anim.current(), "|");
    }

    #[test]
    fn test_animated_braille() {
        let anim = AnimatedIndicator::braille();
        assert_eq!(anim.current(), "⠋");
    }

    #[test]
    fn test_progress_display() {
        let mut progress = ProgressIndicator::new(200, "Downloading");
        progress.set(100);
        let display = progress.display();
        assert!(display.contains("Downloading"));
        assert!(display.contains("50.0%"));
        assert!(display.contains("█"));
    }

    #[test]
    fn test_progress_eta() {
        let progress = ProgressIndicator::new(100, "Test");
        assert!(progress.eta().is_none()); // no progress yet, no ETA
    }

    #[test]
    fn test_progress_zero_total() {
        let progress = ProgressIndicator::new(0, "Empty");
        assert_eq!(progress.percentage(), 100.0);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_progress() {
        let mut progress = ProgressIndicator::new(100, "Loading");

        assert_eq!(progress.percentage(), 0.0);

        progress.increment(50);
        assert_eq!(progress.percentage(), 50.0);

        progress.set(100);
        assert!(progress.is_complete());
    }
}
