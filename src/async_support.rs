//! Async/Await support utilities
//!
//! This module provides utilities for working with asynchronous code in a way
//! that's compatible with both sync and async contexts. It includes:
//!
//! - **Async traits**: Trait definitions for async operations
//! - **Sync wrappers**: Run async code in sync contexts
//! - **Async utilities**: Common async patterns and helpers
//! - **Channel adapters**: Convert between sync and async channels
//!
//! # Feature Flags
//!
//! This module works with any async runtime. For full async support, you'll
//! need to enable the appropriate runtime feature (e.g., `tokio` or `async-std`).
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::async_support::{AsyncResult, spawn_blocking, yield_now};
//!
//! // Run blocking code without blocking the async runtime
//! async fn example() {
//!     let result = spawn_blocking(|| {
//!         // Expensive computation
//!         std::thread::sleep(std::time::Duration::from_secs(1));
//!         42
//!     }).await;
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::time::Duration;

/// Type alias for async results
pub type AsyncResult<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Type alias for boxed futures
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ============================================================================
// Async Traits
// ============================================================================

/// Trait for types that can be loaded asynchronously
pub trait AsyncLoad: Sized {
    /// Error type for loading
    type Error;

    /// Load the resource asynchronously
    fn load_async() -> BoxFuture<'static, Result<Self, Self::Error>>;
}

/// Trait for types that can be saved asynchronously
pub trait AsyncSave {
    /// Error type for saving
    type Error;

    /// Save the resource asynchronously
    fn save_async(&self) -> BoxFuture<'_, Result<(), Self::Error>>;
}

/// Trait for async iteration
pub trait AsyncIterator {
    /// Item type
    type Item;

    /// Get next item asynchronously
    fn next_async(&mut self) -> BoxFuture<'_, Option<Self::Item>>;
}

// ============================================================================
// Sync/Async Bridge
// ============================================================================

/// State for blocking on a future
struct BlockState<T> {
    result: Option<T>,
    waker: Option<Waker>,
    completed: bool,
}

/// A handle to a spawned blocking task
pub struct BlockingHandle<T> {
    state: Arc<Mutex<BlockState<T>>>,
}

impl<T> Future for BlockingHandle<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();

        if state.completed {
            Poll::Ready(state.result.take().expect("Result already taken"))
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Spawn a blocking operation that can be awaited
///
/// This runs the closure in a separate thread and returns a future
/// that resolves when the closure completes.
pub fn spawn_blocking<F, T>(f: F) -> BlockingHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let state = Arc::new(Mutex::new(BlockState {
        result: None,
        waker: None,
        completed: false,
    }));

    let state_clone = state.clone();

    thread::spawn(move || {
        let result = f();
        let mut state = state_clone.lock().unwrap();
        state.result = Some(result);
        state.completed = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    });

    BlockingHandle { state }
}

/// Run a future synchronously by blocking the current thread
///
/// This is useful for running async code from a sync context.
/// Note: This should not be called from within an async runtime.
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    use std::task::RawWaker;
    use std::task::RawWakerVTable;

    // Simple waker that does nothing (we'll spin-poll)
    fn clone_waker(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop_waker(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);

    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);

    let mut future = std::pin::pin!(future);

    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                // Yield to avoid busy-spinning too hard
                thread::yield_now();
            }
        }
    }
}

// ============================================================================
// Async Utilities
// ============================================================================

/// A future that yields control once and then completes
pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yield control to allow other tasks to run
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

/// A future that completes after a delay
pub struct Sleep {
    deadline: std::time::Instant,
}

impl Future for Sleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if std::time::Instant::now() >= self.deadline {
            Poll::Ready(())
        } else {
            // Schedule wake-up
            let waker = cx.waker().clone();
            let deadline = self.deadline;
            thread::spawn(move || {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if !remaining.is_zero() {
                    thread::sleep(remaining);
                }
                waker.wake();
            });
            Poll::Pending
        }
    }
}

/// Sleep for the specified duration
pub fn sleep(duration: Duration) -> Sleep {
    Sleep {
        deadline: std::time::Instant::now() + duration,
    }
}

/// A future that completes with a timeout
pub struct Timeout<F> {
    future: F,
    deadline: std::time::Instant,
}

impl<F: Future + Unpin> Future for Timeout<F> {
    type Output = Option<F::Output>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check timeout first
        if std::time::Instant::now() >= self.deadline {
            return Poll::Ready(None);
        }

        // Poll the inner future
        match Pin::new(&mut self.future).poll(cx) {
            Poll::Ready(result) => Poll::Ready(Some(result)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Wrap a future with a timeout
pub fn timeout<F: Future + Unpin>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        future,
        deadline: std::time::Instant::now() + duration,
    }
}

// ============================================================================
// Channel Adapters
// ============================================================================

/// An async-compatible channel sender
pub struct AsyncSender<T> {
    tx: std::sync::mpsc::Sender<T>,
}

impl<T> AsyncSender<T> {
    /// Send a value (non-blocking)
    pub fn send(&self, value: T) -> Result<(), T> {
        self.tx.send(value).map_err(|e| e.0)
    }
}

impl<T> Clone for AsyncSender<T> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone() }
    }
}

/// An async-compatible channel receiver
pub struct AsyncReceiver<T> {
    rx: Arc<Mutex<std::sync::mpsc::Receiver<T>>>,
}

impl<T: Send + 'static> AsyncReceiver<T> {
    /// Receive a value asynchronously
    pub fn recv(&self) -> RecvFuture<T> {
        RecvFuture {
            rx: self.rx.clone(),
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Option<T> {
        self.rx.lock().ok()?.try_recv().ok()
    }
}

/// Future for receiving from a channel
pub struct RecvFuture<T> {
    rx: Arc<Mutex<std::sync::mpsc::Receiver<T>>>,
}

impl<T: Send + 'static> Future for RecvFuture<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Ok(rx) = self.rx.lock() {
            match rx.try_recv() {
                Ok(value) => Poll::Ready(Some(value)),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Schedule a wake-up after a short delay
                    let waker = cx.waker().clone();
                    thread::spawn(move || {
                        thread::sleep(Duration::from_millis(1));
                        waker.wake();
                    });
                    Poll::Pending
                }
            }
        } else {
            Poll::Ready(None)
        }
    }
}

/// Create an async-compatible channel
pub fn async_channel<T: Send + 'static>() -> (AsyncSender<T>, AsyncReceiver<T>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (
        AsyncSender { tx },
        AsyncReceiver { rx: Arc::new(Mutex::new(rx)) },
    )
}

// ============================================================================
// Async Stream Adapter
// ============================================================================

/// Convert an iterator into an async stream
pub struct AsyncStream<I> {
    iter: I,
}

impl<I: Iterator + Unpin> AsyncStream<I> {
    /// Create a new async stream from an iterator
    pub fn new(iter: I) -> Self {
        Self { iter }
    }

    /// Get the next item asynchronously
    pub async fn next(&mut self) -> Option<I::Item> {
        yield_now().await;
        self.iter.next()
    }
}

/// Create an async stream from an iterator
pub fn stream<I: Iterator + Unpin>(iter: I) -> AsyncStream<I> {
    AsyncStream::new(iter)
}

// ============================================================================
// Async Mutex
// ============================================================================

/// A simple async-compatible mutex
pub struct AsyncMutex<T> {
    inner: Arc<Mutex<T>>,
}

impl<T> AsyncMutex<T> {
    /// Create a new async mutex
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value)),
        }
    }

    /// Lock the mutex asynchronously
    pub async fn lock(&self) -> AsyncMutexGuard<'_, T> {
        loop {
            if let Ok(guard) = self.inner.try_lock() {
                return AsyncMutexGuard { guard };
            }
            yield_now().await;
        }
    }

    /// Try to lock without blocking
    pub fn try_lock(&self) -> Option<AsyncMutexGuard<'_, T>> {
        self.inner.try_lock().ok().map(|guard| AsyncMutexGuard { guard })
    }
}

impl<T> Clone for AsyncMutex<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Guard for AsyncMutex
pub struct AsyncMutexGuard<'a, T> {
    guard: std::sync::MutexGuard<'a, T>,
}

impl<'a, T> std::ops::Deref for AsyncMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

impl<'a, T> std::ops::DerefMut for AsyncMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.guard
    }
}

// ============================================================================
// Join Utilities
// ============================================================================

/// Join two futures, running them concurrently
pub async fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: Future<Output = RA>,
    B: Future<Output = RB>,
{
    let a = std::pin::pin!(a);
    let b = std::pin::pin!(b);

    // Simple sequential execution (real async runtimes do better)
    let ra = a.await;
    let rb = b.await;
    (ra, rb)
}

/// Join three futures
pub async fn join3<A, B, C, RA, RB, RC>(a: A, b: B, c: C) -> (RA, RB, RC)
where
    A: Future<Output = RA>,
    B: Future<Output = RB>,
    C: Future<Output = RC>,
{
    let ra = a.await;
    let rb = b.await;
    let rc = c.await;
    (ra, rb, rc)
}

/// Select the first future to complete
///
/// Returns the result of whichever future completes first.
/// Note: This is a simplified implementation that just awaits the first future.
/// For proper concurrent selection, use an async runtime like tokio.
pub async fn select<A, B, R>(a: A, _b: B) -> R
where
    A: Future<Output = R>,
    B: Future<Output = R>,
{
    // Simplified: just await first one
    // Real implementation would poll both concurrently
    a.await
}

// ============================================================================
// Retry Utilities
// ============================================================================

/// Configuration for async retry
#[derive(Debug, Clone)]
pub struct AsyncRetryConfig {
    /// Maximum number of attempts
    pub max_attempts: usize,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_factor: f64,
}

impl Default for AsyncRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_factor: 2.0,
        }
    }
}

/// Retry an async operation with exponential backoff
pub async fn retry_async<F, Fut, T, E>(config: AsyncRetryConfig, mut f: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut delay = config.initial_delay;
    let mut attempts = 0;

    loop {
        attempts += 1;

        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempts >= config.max_attempts => return Err(e),
            Err(_) => {
                sleep(delay).await;
                delay = std::cmp::min(
                    Duration::from_secs_f64(delay.as_secs_f64() * config.backoff_factor),
                    config.max_delay,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_blocking() {
        let handle = spawn_blocking(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });

        let result = block_on(handle);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_yield_now() {
        block_on(async {
            yield_now().await;
            // Should complete without blocking
        });
    }

    #[test]
    fn test_sleep() {
        let start = std::time::Instant::now();
        block_on(sleep(Duration::from_millis(50)));
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(40)); // Allow some tolerance
    }

    #[test]
    fn test_async_channel() {
        let (tx, rx) = async_channel();

        tx.send(42).unwrap();

        let result = block_on(rx.recv());
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_async_stream() {
        let mut stream = stream(vec![1, 2, 3].into_iter());

        block_on(async {
            assert_eq!(stream.next().await, Some(1));
            assert_eq!(stream.next().await, Some(2));
            assert_eq!(stream.next().await, Some(3));
            assert_eq!(stream.next().await, None);
        });
    }

    #[test]
    fn test_timeout_success() {
        let future = async { 42 };
        let result = block_on(timeout(Duration::from_secs(1), std::pin::pin!(future)));
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_join() {
        let result = block_on(join(
            async { 1 },
            async { 2 },
        ));
        assert_eq!(result, (1, 2));
    }
}
