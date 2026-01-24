//! Typing indicators
//!
//! Show typing/processing indicators during AI responses.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Typing indicator state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        *self.state.lock().unwrap() = state;
        *self.started_at.lock().unwrap() = Some(Instant::now());
    }

    pub fn stop(&self) {
        *self.state.lock().unwrap() = TypingState::Finished;
        *self.started_at.lock().unwrap() = None;
    }

    pub fn error(&self, message: Option<String>) {
        *self.state.lock().unwrap() = TypingState::Error;
        *self.message.lock().unwrap() = message;
    }

    pub fn state(&self) -> TypingState {
        *self.state.lock().unwrap()
    }

    pub fn is_active(&self) -> bool {
        self.state().is_active()
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.lock().unwrap().map(|t| t.elapsed())
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

        let bar: String = std::iter::repeat('█').take(filled)
            .chain(std::iter::repeat('░').take(bar_width - filled))
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
    fn test_progress() {
        let mut progress = ProgressIndicator::new(100, "Loading");

        assert_eq!(progress.percentage(), 0.0);

        progress.increment(50);
        assert_eq!(progress.percentage(), 50.0);

        progress.set(100);
        assert!(progress.is_complete());
    }
}
