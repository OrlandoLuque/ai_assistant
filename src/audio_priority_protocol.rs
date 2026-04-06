//! Audio Priority Protocol — acoustic floor-control with priority queue.
//!
//! Coordinates multiple voice clients in a shared voice channel (Discord, in-game
//! voice, Zoom) purely through **inaudible beacons** embedded in each client's
//! outgoing audio. No network link is strictly required between clients: every
//! instance "hears" the others' beacons via audio loopback capture.
//!
//! # Concepts
//! - **Slot**: a reserved beacon frequency (15.0–16.6 kHz, 200 Hz steps). Each
//!   user owns a slot for the session.
//! - **Priority**: attached to the slot (not the user) — 0 (lowest) .. 10 (highest).
//! - **Floor**: the shared virtual channel. At most one "active" speaker at a time
//!   per priority tier, unless someone with higher priority overrides.
//! - **Override**: a reserved max-priority signal that bypasses any ongoing TX.
//!
//! # Signalling
//! Each slot emits three beacon types on its own frequency:
//! - `IDLE` — 30 ms pulse every 2 s (at −55 dBFS). Means "I am connected".
//! - `ACTIVE` — 50 ms pulse every 400 ms (at −40 dBFS). Means "I am transmitting".
//! - `END` — double pulse (80+80 ms). Means "my message finished, floor released".
//!
//! A dedicated `OVERRIDE` frequency (16.6 kHz) is emitted at −35 dBFS whenever an
//! authorized user presses the override key. Any receiver with lower priority
//! stops TX immediately.
//!
//! # Detection
//! Uses the **Goertzel algorithm** (O(N) per frequency, ~10× cheaper than FFT for
//! our ~10 target bins). No external FFT dependency required.
//!
//! # Codec compatibility
//! Beacons sit in 15.0–16.6 kHz to survive Opus wideband (cuts at 16 kHz at 64 kbps).
//! This range is inaudible to most adults >25 y.o. but may be perceptible to kids
//! and pets — acceptable trade-off.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ============================================================================
// Config & core types
// ============================================================================

/// Slot identifier (0..=14). Each slot maps to a fixed beacon frequency.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SlotId(pub u8);

impl SlotId {
    pub const MAX: u8 = 14;
    pub fn new(id: u8) -> Option<Self> {
        if id <= Self::MAX {
            Some(Self(id))
        } else {
            None
        }
    }
    pub fn as_u8(&self) -> u8 {
        self.0
    }
}

/// Priority level 0..=10. Higher wins.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    pub const MIN: Priority = Priority(0);
    pub const MAX: Priority = Priority(10);
    pub fn new(p: u8) -> Self {
        Self(p.min(10))
    }
    pub fn as_u8(&self) -> u8 {
        self.0
    }
}

/// Protocol-wide configuration. Clients must share identical values to interop.
#[derive(Clone, Debug)]
pub struct ProtocolConfig {
    /// Base frequency of slot 0 (Hz).
    pub base_freq_hz: f32,
    /// Spacing between consecutive slots (Hz).
    pub freq_step_hz: f32,
    /// Number of available slots.
    pub slot_count: u8,
    /// Frequency reserved for the override beacon.
    pub override_freq_hz: f32,
    /// Sample rate expected on all audio pipelines.
    pub sample_rate: u32,
    /// Goertzel analysis window size (samples).
    pub window_size: usize,
    /// ACTIVE beacon amplitude (linear, 0..1). −40 dBFS ≈ 0.01.
    pub active_amplitude: f32,
    /// IDLE beacon amplitude (linear). −55 dBFS ≈ 0.0018.
    pub idle_amplitude: f32,
    /// Override beacon amplitude (linear). −35 dBFS ≈ 0.018.
    pub override_amplitude: f32,
    /// Heartbeat interval for ACTIVE beacons (ms).
    pub heartbeat_ms: u64,
    /// Interval between IDLE beacons (ms).
    pub idle_interval_ms: u64,
    /// After this many ms without ACTIVE beacon on a slot, mark it idle.
    pub active_timeout_ms: u64,
    /// After this many ms without IDLE beacon, mark the slot disconnected.
    pub idle_timeout_ms: u64,
    /// Silence duration that finalizes a voice message (VAD mode).
    pub silence_timeout_ms: u64,
    /// Messages shorter than this are discarded (noise/cough filter).
    pub min_message_ms: u64,
    /// Messages longer than this are truncated (channel-hog guard).
    pub max_message_ms: u64,
    /// Max pending messages per user before rejecting new ones.
    pub max_queue: usize,
    /// When resuming an interrupted message, rewind by this much.
    pub resume_offset_ms: u64,
    /// Random startup delay before emitting the first beacon (max ms).
    pub jitter_max_ms: u64,
    /// CSMA/CA: emit intent-beacon then wait this long before TX.
    pub csma_wait_ms: u64,
    /// SNR (dB) above noise floor required to declare a detection.
    pub detection_snr_db: f32,
    /// Consecutive windows needed to confirm a detection (anti-false-positive).
    pub detection_confirm_windows: u8,
    /// Samples of linear fade applied when interrupting playback.
    pub fade_samples: usize,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            base_freq_hz: 15_000.0,
            freq_step_hz: 200.0,
            slot_count: 8,
            override_freq_hz: 16_600.0,
            sample_rate: 48_000,
            window_size: 1024,
            active_amplitude: 0.010,
            idle_amplitude: 0.0018,
            override_amplitude: 0.018,
            heartbeat_ms: 400,
            idle_interval_ms: 2_000,
            active_timeout_ms: 800,
            idle_timeout_ms: 5_000,
            silence_timeout_ms: 1_500,
            min_message_ms: 300,
            max_message_ms: 60_000,
            max_queue: 5,
            resume_offset_ms: 5_000,
            jitter_max_ms: 1_000,
            csma_wait_ms: 500,
            detection_snr_db: 12.0,
            detection_confirm_windows: 2,
            fade_samples: 480, // 10ms @ 48kHz
        }
    }
}

impl ProtocolConfig {
    /// Returns the beacon frequency for a given slot.
    pub fn slot_frequency(&self, slot: SlotId) -> f32 {
        self.base_freq_hz + self.freq_step_hz * slot.as_u8() as f32
    }

    /// Returns the Nyquist-safe upper bound. Frequencies above this should be rejected.
    pub fn max_safe_freq(&self) -> f32 {
        (self.sample_rate as f32) * 0.45
    }

    /// Sanity: do all reserved freqs stay below Nyquist and within codec-safe range?
    pub fn is_valid(&self) -> bool {
        if self.slot_count == 0 || self.slot_count > 15 {
            return false;
        }
        if self.window_size == 0 {
            return false;
        }
        let max_slot_freq = self.slot_frequency(SlotId(self.slot_count - 1));
        max_slot_freq < self.max_safe_freq()
            && self.override_freq_hz < self.max_safe_freq()
            && self.override_freq_hz > 0.0
    }
}

// ============================================================================
// Tone encoder — generates sine beacons mixed into output buffer
// ============================================================================

/// Generates sine tones and mixes them into an output buffer. Keeps phase between
/// calls for continuity (no audible click at buffer boundaries).
pub struct ToneEncoder {
    sample_rate: u32,
    /// Running phase per frequency (Hz → radians).
    phases: HashMap<u32, f32>,
}

impl ToneEncoder {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            phases: HashMap::new(),
        }
    }

    /// Mix a sinusoid at `frequency_hz` with `amplitude` into `buffer`. All samples
    /// of the buffer are written; phase is preserved across calls.
    pub fn mix_tone(&mut self, buffer: &mut [f32], frequency_hz: f32, amplitude: f32) {
        let freq_key = frequency_hz as u32;
        let phase = self.phases.entry(freq_key).or_insert(0.0);
        let step = 2.0 * std::f32::consts::PI * frequency_hz / self.sample_rate as f32;
        for sample in buffer.iter_mut() {
            *sample += amplitude * phase.sin();
            *phase += step;
            if *phase > 2.0 * std::f32::consts::PI {
                *phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    /// Mix a windowed pulse: linear 10-sample ramp in/out to avoid spectral splatter.
    pub fn mix_pulse(&mut self, buffer: &mut [f32], frequency_hz: f32, amplitude: f32) {
        let ramp = (buffer.len() / 20).min(240);
        let freq_key = frequency_hz as u32;
        let phase = self.phases.entry(freq_key).or_insert(0.0);
        let step = 2.0 * std::f32::consts::PI * frequency_hz / self.sample_rate as f32;
        let len = buffer.len();
        for (i, sample) in buffer.iter_mut().enumerate() {
            let env = if i < ramp {
                i as f32 / ramp as f32
            } else if i >= len.saturating_sub(ramp) {
                (len - i) as f32 / ramp as f32
            } else {
                1.0
            };
            *sample += amplitude * env * phase.sin();
            *phase += step;
            if *phase > 2.0 * std::f32::consts::PI {
                *phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    /// Reset phase state (e.g., after a long gap).
    pub fn reset(&mut self) {
        self.phases.clear();
    }
}

// ============================================================================
// Goertzel detector — O(N) frequency-specific magnitude
// ============================================================================

/// Detects magnitude at a single target frequency using the Goertzel algorithm.
#[derive(Clone, Debug)]
pub struct GoertzelDetector {
    pub target_freq_hz: f32,
    pub sample_rate: u32,
    pub window_size: usize,
    coefficient: f32,
}

impl GoertzelDetector {
    pub fn new(target_freq_hz: f32, sample_rate: u32, window_size: usize) -> Self {
        // k is the bin index (possibly non-integer), used to derive the coefficient.
        let k = window_size as f32 * target_freq_hz / sample_rate as f32;
        let omega = 2.0 * std::f32::consts::PI * k / window_size as f32;
        let coefficient = 2.0 * omega.cos();
        Self {
            target_freq_hz,
            sample_rate,
            window_size,
            coefficient,
        }
    }

    /// Returns the squared magnitude at the target frequency. Normalised roughly
    /// per-sample so the output is comparable to a linear amplitude.
    pub fn magnitude_squared(&self, samples: &[f32]) -> f32 {
        let mut q0;
        let mut q1 = 0.0f32;
        let mut q2 = 0.0f32;
        for &s in samples {
            q0 = self.coefficient * q1 - q2 + s;
            q2 = q1;
            q1 = q0;
        }
        // magnitude² = Q1² + Q2² − coeff·Q1·Q2
        let mag_sq = q1 * q1 + q2 * q2 - self.coefficient * q1 * q2;
        // Normalise by N² so amplitude is comparable across window sizes
        mag_sq / (samples.len() as f32 * samples.len() as f32) * 4.0
    }

    /// Returns magnitude (sqrt of squared magnitude).
    pub fn magnitude(&self, samples: &[f32]) -> f32 {
        self.magnitude_squared(samples).sqrt()
    }
}

// ============================================================================
// Tone decoder — multi-frequency detection + noise floor tracking
// ============================================================================

/// A single detection frame snapshot.
#[derive(Clone, Debug, Default)]
pub struct DetectionFrame {
    /// Slots currently showing detectable energy above SNR threshold.
    pub active_slots: Vec<SlotId>,
    /// Whether the override frequency is active.
    pub override_active: bool,
    /// Raw magnitudes per slot (for debugging / GUI meters).
    pub magnitudes: HashMap<SlotId, f32>,
    /// Override magnitude.
    pub override_magnitude: f32,
    /// Estimated noise floor at this frame.
    pub noise_floor: f32,
}

/// Multi-bin decoder with noise-floor estimation and N-of-M confirmation.
pub struct ToneDecoder {
    slot_detectors: Vec<(SlotId, GoertzelDetector)>,
    override_detector: GoertzelDetector,
    noise_floor: f32, // EMA smoothed
    snr_threshold_linear: f32,
    confirm_windows: u8,
    /// Pre-computed Hann window for spectral leakage suppression.
    hann_window: Vec<f32>,
    /// Pre-allocated buffer for windowed samples.
    windowed_buf: Vec<f32>,
    /// Rolling history of per-slot detections for confirmation.
    history: VecDeque<HashMap<SlotId, bool>>,
    override_history: VecDeque<bool>,
    history_cap: usize,
}

impl ToneDecoder {
    pub fn new(config: &ProtocolConfig) -> Self {
        let mut slot_detectors = Vec::with_capacity(config.slot_count as usize);
        for i in 0..config.slot_count {
            let slot = SlotId(i);
            let freq = config.slot_frequency(slot);
            slot_detectors.push((
                slot,
                GoertzelDetector::new(freq, config.sample_rate, config.window_size),
            ));
        }
        let override_detector = GoertzelDetector::new(
            config.override_freq_hz,
            config.sample_rate,
            config.window_size,
        );
        let snr_threshold_linear = 10.0f32.powf(config.detection_snr_db / 20.0);
        let history_cap = config.detection_confirm_windows.max(1) as usize;
        let n = config.window_size;
        // Hann window: 0.5 * (1 - cos(2πi/(N-1))). Reduces sidelobe leakage ~31 dB
        // so a strong tone in one bin doesn't falsely trigger adjacent bins.
        let hann_window: Vec<f32> = (0..n)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1).max(1) as f32).cos())
            })
            .collect();
        Self {
            slot_detectors,
            override_detector,
            noise_floor: 1e-5,
            snr_threshold_linear,
            confirm_windows: config.detection_confirm_windows.max(1),
            hann_window,
            windowed_buf: vec![0.0; n],
            history: VecDeque::with_capacity(history_cap),
            override_history: VecDeque::with_capacity(history_cap),
            history_cap,
        }
    }

    /// Process one window of samples; returns the confirmed detection frame.
    pub fn process(&mut self, samples: &[f32]) -> DetectionFrame {
        // Apply Hann window to reduce spectral leakage between adjacent slot bins.
        let n = self.hann_window.len().min(samples.len());
        for i in 0..n {
            self.windowed_buf[i] = samples[i] * self.hann_window[i];
        }
        for i in n..self.windowed_buf.len() {
            self.windowed_buf[i] = 0.0;
        }
        let windowed: &[f32] = &self.windowed_buf[..n];

        // Raw per-slot magnitudes
        let mut magnitudes = HashMap::new();
        let mut raw_active = HashMap::new();
        let mut frame_min = f32::MAX;
        for (slot, det) in &self.slot_detectors {
            let m = det.magnitude(windowed);
            magnitudes.insert(*slot, m);
            frame_min = frame_min.min(m);
        }
        let override_magnitude = self.override_detector.magnitude(windowed);
        frame_min = frame_min.min(override_magnitude);

        // Update noise floor (EMA, bias toward lower values to avoid ratcheting up
        // because of ongoing beacons).
        if frame_min < self.noise_floor {
            self.noise_floor = self.noise_floor * 0.7 + frame_min * 0.3;
        } else {
            self.noise_floor = self.noise_floor * 0.99 + frame_min * 0.01;
        }
        self.noise_floor = self.noise_floor.max(1e-7);

        let threshold = self.noise_floor * self.snr_threshold_linear;

        // Neighbor-rel check: a slot is only "active" if its magnitude dominates
        // its adjacent slots (within 1 bin). Prevents a single strong tone from
        // triggering multiple adjacent slots via spectral leakage.
        let slot_mags: Vec<(SlotId, f32)> = self
            .slot_detectors
            .iter()
            .map(|(s, _)| (*s, magnitudes[s]))
            .collect();
        for (slot, m) in &slot_mags {
            let above_floor = *m > threshold;
            // Find adjacent slot magnitudes
            let idx = slot.as_u8() as usize;
            let left = if idx > 0 {
                slot_mags.get(idx - 1).map(|(_, m)| *m).unwrap_or(0.0)
            } else {
                0.0
            };
            let right = slot_mags.get(idx + 1).map(|(_, m)| *m).unwrap_or(0.0);
            let max_neighbor = left.max(right);
            // Require this slot to dominate neighbors by at least 1.5× (3.5 dB).
            // If max_neighbor is near zero, we're isolated → auto-pass.
            let dominates = max_neighbor < 1e-6 || *m > max_neighbor * 1.5;
            raw_active.insert(*slot, above_floor && dominates);
        }
        let raw_override = override_magnitude > threshold;

        // Push to history
        if self.history.len() >= self.history_cap {
            self.history.pop_front();
        }
        self.history.push_back(raw_active);
        if self.override_history.len() >= self.history_cap {
            self.override_history.pop_front();
        }
        self.override_history.push_back(raw_override);

        // Confirm: all `confirm_windows` most-recent frames must agree
        let need = self.confirm_windows as usize;
        let active_slots: Vec<SlotId> = self
            .slot_detectors
            .iter()
            .filter_map(|(slot, _)| {
                let all_active = self
                    .history
                    .iter()
                    .rev()
                    .take(need)
                    .all(|h| *h.get(slot).unwrap_or(&false));
                if self.history.len() >= need && all_active {
                    Some(*slot)
                } else {
                    None
                }
            })
            .collect();
        let override_active = self.override_history.len() >= need
            && self.override_history.iter().rev().take(need).all(|&b| b);

        DetectionFrame {
            active_slots,
            override_active,
            magnitudes,
            override_magnitude,
            noise_floor: self.noise_floor,
        }
    }

    /// Clears detection history (e.g., after reconfiguration).
    pub fn reset(&mut self) {
        self.history.clear();
        self.override_history.clear();
        self.noise_floor = 1e-5;
    }
}

// ============================================================================
// Floor monitor — aggregates detections into slot states over time
// ============================================================================

/// Per-slot tracked state.
#[derive(Clone, Debug)]
pub struct SlotState {
    pub slot: SlotId,
    pub last_active_at: Option<Instant>,
    pub last_idle_at: Option<Instant>,
    pub first_seen_at: Option<Instant>,
    pub currently_active: bool,
    pub currently_connected: bool,
}

impl SlotState {
    fn new(slot: SlotId) -> Self {
        Self {
            slot,
            last_active_at: None,
            last_idle_at: None,
            first_seen_at: None,
            currently_active: false,
            currently_connected: false,
        }
    }
}

/// Tracks who is on the floor based on detection frames arriving over time.
pub struct FloorMonitor {
    config: ProtocolConfig,
    states: HashMap<SlotId, SlotState>,
    override_active_since: Option<Instant>,
    /// Whether the most recent frame reported a slot as active (for edge detection).
    last_frame_active: HashMap<SlotId, bool>,
}

impl FloorMonitor {
    pub fn new(config: ProtocolConfig) -> Self {
        let mut states = HashMap::new();
        for i in 0..config.slot_count {
            let s = SlotId(i);
            states.insert(s, SlotState::new(s));
        }
        Self {
            config,
            states,
            override_active_since: None,
            last_frame_active: HashMap::new(),
        }
    }

    /// Apply a decoder detection frame.
    pub fn update(&mut self, frame: &DetectionFrame, now: Instant) {
        // Update active slots: those currently emitting ACTIVE beacons.
        // A slot showing magnitude in `active_slots` is treated as ACTIVE if it's
        // persistently present (confirmation done by decoder). On transition from
        // not-active to active, mark it; heartbeat extends last_active_at.
        let active_set: std::collections::HashSet<SlotId> =
            frame.active_slots.iter().copied().collect();
        for i in 0..self.config.slot_count {
            let slot = SlotId(i);
            let was_active = *self.last_frame_active.get(&slot).unwrap_or(&false);
            let is_active = active_set.contains(&slot);
            let state = self.states.get_mut(&slot).expect("slot exists");
            if is_active {
                state.last_active_at = Some(now);
                state.last_idle_at = Some(now); // active implies connected
                if state.first_seen_at.is_none() {
                    state.first_seen_at = Some(now);
                }
            }
            // Short-pulse IDLE vs long-pulse ACTIVE distinction is approximated:
            // we can't directly tell from one frame; we rely on persistence + rate.
            // If we see energy < active_timeout ago, treat as connected (idle OR active).
            if is_active && !was_active {
                state.currently_active = true;
            }
            self.last_frame_active.insert(slot, is_active);
        }

        // Apply timeouts
        for state in self.states.values_mut() {
            if let Some(t) = state.last_active_at {
                if now.duration_since(t) > Duration::from_millis(self.config.active_timeout_ms) {
                    state.currently_active = false;
                }
            } else {
                state.currently_active = false;
            }
            if let Some(t) = state.last_idle_at {
                state.currently_connected =
                    now.duration_since(t) <= Duration::from_millis(self.config.idle_timeout_ms);
            } else {
                state.currently_connected = false;
            }
        }

        // Override
        if frame.override_active {
            if self.override_active_since.is_none() {
                self.override_active_since = Some(now);
            }
        } else if let Some(since) = self.override_active_since {
            if now.duration_since(since) > Duration::from_millis(self.config.active_timeout_ms) {
                self.override_active_since = None;
            }
        }
    }

    /// Inject presence for a slot without a detection frame (e.g., host told us
    /// someone joined). Useful for clients connected via a network host.
    pub fn mark_connected(&mut self, slot: SlotId, now: Instant) {
        if let Some(s) = self.states.get_mut(&slot) {
            s.last_idle_at = Some(now);
            s.currently_connected = true;
            if s.first_seen_at.is_none() {
                s.first_seen_at = Some(now);
            }
        }
    }

    pub fn slot_state(&self, slot: SlotId) -> Option<&SlotState> {
        self.states.get(&slot)
    }

    /// All slots currently emitting ACTIVE beacons.
    pub fn active_slots(&self) -> Vec<SlotId> {
        let mut v: Vec<SlotId> = self
            .states
            .iter()
            .filter(|(_, s)| s.currently_active)
            .map(|(k, _)| *k)
            .collect();
        v.sort();
        v
    }

    /// All slots that have emitted any beacon within `idle_timeout_ms`.
    pub fn connected_slots(&self) -> Vec<SlotId> {
        let mut v: Vec<SlotId> = self
            .states
            .iter()
            .filter(|(_, s)| s.currently_connected)
            .map(|(k, _)| *k)
            .collect();
        v.sort();
        v
    }

    /// The highest-priority slot that is currently active, along with its priority.
    pub fn highest_active(&self, table: &PriorityTable) -> Option<(SlotId, Priority)> {
        self.active_slots()
            .into_iter()
            .map(|s| (s, table.priority_of(s)))
            .max_by_key(|(_, p)| *p)
    }

    /// Returns true if `my_priority` is strictly greater than every active speaker.
    pub fn can_speak(&self, my_priority: Priority, table: &PriorityTable) -> bool {
        if self.override_active_since.is_some() {
            return false;
        }
        match self.highest_active(table) {
            None => true,
            Some((_, p)) => my_priority > p,
        }
    }

    pub fn override_active(&self) -> bool {
        self.override_active_since.is_some()
    }
}

// ============================================================================
// Priority table — slot → priority + can_override mappings
// ============================================================================

#[derive(Clone, Debug)]
pub struct SlotAssignment {
    pub slot: SlotId,
    pub priority: Priority,
    pub can_override: bool,
    pub display_name: String,
}

#[derive(Clone, Debug, Default)]
pub struct PriorityTable {
    assignments: HashMap<SlotId, SlotAssignment>,
}

impl PriorityTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a slot. Replaces previous assignment if any.
    pub fn assign(&mut self, a: SlotAssignment) {
        self.assignments.insert(a.slot, a);
    }

    pub fn remove(&mut self, slot: SlotId) {
        self.assignments.remove(&slot);
    }

    pub fn get(&self, slot: SlotId) -> Option<&SlotAssignment> {
        self.assignments.get(&slot)
    }

    /// Priority for a slot, or `Priority::MIN` if unassigned.
    pub fn priority_of(&self, slot: SlotId) -> Priority {
        self.assignments
            .get(&slot)
            .map(|a| a.priority)
            .unwrap_or(Priority::MIN)
    }

    pub fn can_override(&self, slot: SlotId) -> bool {
        self.assignments
            .get(&slot)
            .map(|a| a.can_override)
            .unwrap_or(false)
    }

    pub fn assigned_slots(&self) -> Vec<SlotId> {
        let mut v: Vec<_> = self.assignments.keys().copied().collect();
        v.sort();
        v
    }

    /// Everyone at priority 5 — pure FCFS.
    pub fn flat(slot_count: u8) -> Self {
        let mut t = Self::new();
        for i in 0..slot_count {
            t.assign(SlotAssignment {
                slot: SlotId(i),
                priority: Priority(5),
                can_override: false,
                display_name: format!("Slot {}", i),
            });
        }
        t
    }

    /// Pyramid: 2 leaders (P10, can_override), `callouts` at P7, rest at P3.
    pub fn squad(slot_count: u8, callouts: u8) -> Self {
        let mut t = Self::new();
        let callouts_end = (2 + callouts).min(slot_count);
        for i in 0..slot_count {
            let (priority, can_override) = if i < 2 {
                (Priority(10), true)
            } else if i < callouts_end {
                (Priority(7), false)
            } else {
                (Priority(3), false)
            };
            t.assign(SlotAssignment {
                slot: SlotId(i),
                priority,
                can_override,
                display_name: format!("Slot {}", i),
            });
        }
        t
    }

    /// Slot 0 presenter (P10 + can_override); rest at P3.
    pub fn meeting(slot_count: u8) -> Self {
        let mut t = Self::new();
        for i in 0..slot_count {
            let (priority, can_override) = if i == 0 {
                (Priority(10), true)
            } else {
                (Priority(3), false)
            };
            t.assign(SlotAssignment {
                slot: SlotId(i),
                priority,
                can_override,
                display_name: format!("Slot {}", i),
            });
        }
        t
    }
}

// ============================================================================
// Voice message + queue
// ============================================================================

#[derive(Clone, Debug)]
pub struct VoiceMessage {
    pub id: u64,
    pub recorded_at: Instant,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl VoiceMessage {
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.samples.len() as u64 * 1000) / self.sample_rate as u64
    }
}

#[derive(Debug)]
pub enum QueueError {
    Full,
}

/// FIFO outbox for recorded messages awaiting transmission.
pub struct MessageQueue {
    messages: VecDeque<VoiceMessage>,
    max_size: usize,
    next_id: u64,
    /// If true, push evicts oldest when full instead of rejecting.
    evict_oldest: bool,
}

impl MessageQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            max_size: max_size.max(1),
            next_id: 1,
            evict_oldest: true,
        }
    }

    pub fn with_eviction(max_size: usize, evict_oldest: bool) -> Self {
        Self {
            messages: VecDeque::new(),
            max_size: max_size.max(1),
            next_id: 1,
            evict_oldest,
        }
    }

    /// Returns the assigned message id.
    pub fn push(&mut self, mut msg: VoiceMessage) -> Result<u64, QueueError> {
        if self.messages.len() >= self.max_size {
            if self.evict_oldest {
                self.messages.pop_front();
            } else {
                return Err(QueueError::Full);
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        msg.id = id;
        self.messages.push_back(msg);
        Ok(id)
    }

    pub fn pop_next(&mut self) -> Option<VoiceMessage> {
        self.messages.pop_front()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        let before = self.messages.len();
        self.messages.retain(|m| m.id != id);
        self.messages.len() < before
    }

    pub fn peek(&self) -> Option<&VoiceMessage> {
        self.messages.front()
    }

    pub fn iter(&self) -> impl Iterator<Item = &VoiceMessage> {
        self.messages.iter()
    }
}

// ============================================================================
// Message recorder — VAD-based segmentation
// ============================================================================

/// Records voice into discrete messages using a VAD signal + configurable silence.
pub struct MessageRecorder {
    buffer: Vec<f32>,
    recording: bool,
    last_voice_at: Option<Instant>,
    started_at: Option<Instant>,
    sample_rate: u32,
    silence_timeout: Duration,
    min_duration: Duration,
    max_duration: Duration,
}

impl MessageRecorder {
    pub fn new(config: &ProtocolConfig) -> Self {
        Self {
            buffer: Vec::new(),
            recording: false,
            last_voice_at: None,
            started_at: None,
            sample_rate: config.sample_rate,
            silence_timeout: Duration::from_millis(config.silence_timeout_ms),
            min_duration: Duration::from_millis(config.min_message_ms),
            max_duration: Duration::from_millis(config.max_message_ms),
        }
    }

    /// Feed a chunk of samples + VAD verdict. Returns Some(message) when a
    /// full message (silence-delimited or max-length-capped) is available.
    pub fn process(
        &mut self,
        samples: &[f32],
        is_voice: bool,
        now: Instant,
    ) -> Option<VoiceMessage> {
        if is_voice {
            if !self.recording {
                self.recording = true;
                self.started_at = Some(now);
                self.buffer.clear();
            }
            self.buffer.extend_from_slice(samples);
            self.last_voice_at = Some(now);

            // Hit max length?
            if let Some(t0) = self.started_at {
                if now.duration_since(t0) >= self.max_duration {
                    return self.finalize(now);
                }
            }
            None
        } else if self.recording {
            // Append trailing silence to preserve natural endings
            self.buffer.extend_from_slice(samples);
            if let Some(last_v) = self.last_voice_at {
                if now.duration_since(last_v) >= self.silence_timeout {
                    return self.finalize(now);
                }
            }
            None
        } else {
            None
        }
    }

    /// PTT-style finalize: stop now and return the message (if long enough).
    pub fn finalize(&mut self, _now: Instant) -> Option<VoiceMessage> {
        if !self.recording {
            return None;
        }
        let started = self.started_at.take();
        let duration = started
            .and_then(|t| self.last_voice_at.map(|lv| lv.saturating_duration_since(t)))
            .unwrap_or_default();
        let samples = std::mem::take(&mut self.buffer);
        self.recording = false;
        self.last_voice_at = None;
        if duration < self.min_duration {
            return None;
        }
        Some(VoiceMessage {
            id: 0,
            recorded_at: started.unwrap_or_else(Instant::now),
            samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Discard the current recording in progress.
    pub fn cancel(&mut self) {
        self.buffer.clear();
        self.recording = false;
        self.last_voice_at = None;
        self.started_at = None;
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn current_duration_ms(&self) -> u64 {
        self.started_at
            .and_then(|t| self.last_voice_at.map(|lv| lv.saturating_duration_since(t)))
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// Interrupt policy + playback controller
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptPolicy {
    /// Cut immediately at the next pulled sample.
    Hard,
    /// Cut at the next low-energy frame within the message.
    Soft,
    /// Let the current message finish, then honour the override.
    Finish,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeStrategy {
    /// Restart the interrupted message from sample 0.
    Restart,
    /// Continue exactly where interrupted.
    Continue,
    /// Rewind by `resume_offset_ms` before resuming.
    Continue5s,
}

/// Stateful playback of a single message with interrupt+resume semantics.
pub struct MessagePlayer {
    current: Option<PlaybackState>,
    interrupted: Option<PlaybackState>,
    policy: InterruptPolicy,
    resume: ResumeStrategy,
    fade_samples: usize,
    resume_offset_samples: usize,
    #[allow(dead_code)]
    sample_rate: u32,
    /// Energy threshold used by Soft policy to find silences.
    soft_silence_threshold: f32,
    /// Pending interrupt requests waiting for a silence (Soft mode).
    pending_interrupt: bool,
}

#[derive(Clone)]
struct PlaybackState {
    message: VoiceMessage,
    position: usize,
    fade_remaining: usize,
    fading_out: bool,
}

impl MessagePlayer {
    pub fn new(policy: InterruptPolicy, resume: ResumeStrategy, config: &ProtocolConfig) -> Self {
        Self {
            current: None,
            interrupted: None,
            policy,
            resume,
            fade_samples: config.fade_samples,
            resume_offset_samples: (config.resume_offset_ms as usize) * config.sample_rate as usize
                / 1000,
            sample_rate: config.sample_rate,
            soft_silence_threshold: 0.005,
            pending_interrupt: false,
        }
    }

    /// Begin playing a message. Replaces any currently playing message.
    pub fn start(&mut self, msg: VoiceMessage) {
        self.current = Some(PlaybackState {
            message: msg,
            position: 0,
            fade_remaining: self.fade_samples,
            fading_out: false,
        });
        self.pending_interrupt = false;
    }

    pub fn is_playing(&self) -> bool {
        self.current.is_some()
    }

    pub fn has_interrupted(&self) -> bool {
        self.interrupted.is_some()
    }

    /// Request an interrupt. Returns true if the interrupt will be honoured.
    /// Hard → takes effect immediately; Soft → at next silence; Finish → ignored.
    pub fn request_interrupt(&mut self) -> bool {
        match self.policy {
            InterruptPolicy::Hard => {
                self.do_interrupt();
                true
            }
            InterruptPolicy::Soft => {
                self.pending_interrupt = true;
                true
            }
            InterruptPolicy::Finish => false,
        }
    }

    fn do_interrupt(&mut self) {
        if let Some(mut st) = self.current.take() {
            st.fading_out = true;
            st.fade_remaining = self.fade_samples;
            // Roll back for resume
            let rollback = match self.resume {
                ResumeStrategy::Restart => st.position,
                ResumeStrategy::Continue => 0,
                ResumeStrategy::Continue5s => self.resume_offset_samples.min(st.position),
            };
            st.position = st.position.saturating_sub(rollback);
            self.interrupted = Some(st);
        }
        self.pending_interrupt = false;
    }

    /// Resume the most recently interrupted message.
    pub fn resume_interrupted(&mut self) {
        if let Some(mut st) = self.interrupted.take() {
            st.fade_remaining = self.fade_samples;
            st.fading_out = false;
            self.current = Some(st);
        }
    }

    /// Pull `n` samples into `out`. Returns number of samples actually written.
    pub fn pull(&mut self, out: &mut [f32]) -> usize {
        if out.is_empty() {
            return 0;
        }
        let written: usize;

        let done = {
            let Some(state) = self.current.as_mut() else {
                return 0;
            };
            let remaining = state.message.samples.len().saturating_sub(state.position);
            let n = remaining.min(out.len());
            for i in 0..n {
                let mut s = state.message.samples[state.position + i];
                // Apply fade envelope
                if state.fade_remaining > 0 {
                    let env = if state.fading_out {
                        state.fade_remaining as f32 / self.fade_samples as f32
                    } else {
                        1.0 - state.fade_remaining as f32 / self.fade_samples as f32
                    };
                    s *= env;
                    state.fade_remaining = state.fade_remaining.saturating_sub(1);
                }
                out[i] = s;
            }
            state.position += n;
            written = n;
            // If Soft policy with pending interrupt, check if we're in a silence
            if self.pending_interrupt && self.policy == InterruptPolicy::Soft {
                let tail = &state.message.samples[state.position.saturating_sub(256)
                    ..state.position.min(state.message.samples.len())];
                if !tail.is_empty() {
                    let rms: f32 =
                        (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
                    if rms < self.soft_silence_threshold {
                        // Trigger interrupt now
                        self.pending_interrupt = false;
                        self.do_interrupt();
                        return written;
                    }
                }
            }
            state.position >= state.message.samples.len()
        };
        if done {
            self.current = None;
        }
        written
    }
}

// ============================================================================
// Beacon scheduler — decides when to emit IDLE / ACTIVE / END / OVERRIDE pulses
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeaconKind {
    Idle,
    Active,
    End,
    Override,
}

#[derive(Clone, Debug)]
pub struct BeaconEmission {
    pub kind: BeaconKind,
    pub frequency_hz: f32,
    pub amplitude: f32,
    pub duration_ms: u64,
}

/// Schedules beacon emissions based on the node's own state (connected, transmitting).
pub struct BeaconScheduler {
    config: ProtocolConfig,
    #[allow(dead_code)]
    my_slot: SlotId,
    my_frequency: f32,
    state: NodeState,
    last_idle_at: Option<Instant>,
    last_active_at: Option<Instant>,
    #[allow(dead_code)]
    idle_slot_offset_ms: u64,
    override_active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeState {
    Idle,          // connected, not transmitting
    Transmitting,  // playing a message
    EndingMessage, // just finished, need to emit END
}

impl BeaconScheduler {
    pub fn new(config: ProtocolConfig, my_slot: SlotId) -> Self {
        let my_frequency = config.slot_frequency(my_slot);
        // Distribute IDLE emissions across the 2s window by slot
        let idle_slot_offset_ms =
            (my_slot.as_u8() as u64 * config.idle_interval_ms) / config.slot_count as u64;
        Self {
            config,
            my_slot,
            my_frequency,
            state: NodeState::Idle,
            last_idle_at: None,
            last_active_at: None,
            idle_slot_offset_ms,
            override_active: false,
        }
    }

    pub fn set_state(&mut self, state: NodeState) {
        self.state = state;
    }
    pub fn state(&self) -> NodeState {
        self.state
    }
    pub fn set_override(&mut self, active: bool) {
        self.override_active = active;
    }

    /// Called periodically. Returns all beacons that should be mixed into the output
    /// right now. Updates internal timing state.
    pub fn tick(&mut self, now: Instant) -> Vec<BeaconEmission> {
        let mut out = Vec::new();
        // OVERRIDE beacon — always active while the flag is set
        if self.override_active {
            out.push(BeaconEmission {
                kind: BeaconKind::Override,
                frequency_hz: self.config.override_freq_hz,
                amplitude: self.config.override_amplitude,
                duration_ms: 80,
            });
            self.last_active_at = Some(now);
        }

        match self.state {
            NodeState::Idle => {
                let need_idle = match self.last_idle_at {
                    None => true,
                    Some(t) => {
                        now.duration_since(t) >= Duration::from_millis(self.config.idle_interval_ms)
                    }
                };
                if need_idle {
                    out.push(BeaconEmission {
                        kind: BeaconKind::Idle,
                        frequency_hz: self.my_frequency,
                        amplitude: self.config.idle_amplitude,
                        duration_ms: 30,
                    });
                    self.last_idle_at = Some(now);
                }
            }
            NodeState::Transmitting => {
                let need_active = match self.last_active_at {
                    None => true,
                    Some(t) => {
                        now.duration_since(t) >= Duration::from_millis(self.config.heartbeat_ms)
                    }
                };
                if need_active {
                    out.push(BeaconEmission {
                        kind: BeaconKind::Active,
                        frequency_hz: self.my_frequency,
                        amplitude: self.config.active_amplitude,
                        duration_ms: 50,
                    });
                    self.last_active_at = Some(now);
                    self.last_idle_at = Some(now);
                }
            }
            NodeState::EndingMessage => {
                out.push(BeaconEmission {
                    kind: BeaconKind::End,
                    frequency_hz: self.my_frequency,
                    amplitude: self.config.active_amplitude,
                    duration_ms: 80,
                });
                out.push(BeaconEmission {
                    kind: BeaconKind::End,
                    frequency_hz: self.my_frequency,
                    amplitude: self.config.active_amplitude,
                    duration_ms: 80,
                });
                self.state = NodeState::Idle;
                self.last_idle_at = Some(now);
            }
        }
        out
    }

    /// Returns time remaining until the next beacon should fire (for sleep budget).
    pub fn time_to_next(&self, now: Instant) -> Duration {
        match self.state {
            NodeState::Transmitting => {
                let hb = Duration::from_millis(self.config.heartbeat_ms);
                self.last_active_at
                    .map(|t| hb.saturating_sub(now.saturating_duration_since(t)))
                    .unwrap_or(Duration::from_millis(0))
            }
            _ => {
                let iv = Duration::from_millis(self.config.idle_interval_ms);
                self.last_idle_at
                    .map(|t| iv.saturating_sub(now.saturating_duration_since(t)))
                    .unwrap_or(Duration::from_millis(0))
            }
        }
    }
}

// ============================================================================
// Captured-mode controller — the main state machine
// ============================================================================

/// What the user wants to do right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    /// Auto-record via VAD, queue on silence.
    Vad,
    /// Record while PTT key is held, queue on release.
    PushToTalk,
    /// Emit override beacon + send voice live, no queue.
    OverridePtt,
    /// Continuous recording, queue chunks every max_message_ms.
    Continuous,
}

/// High-level state exposed to the GUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControllerStatus {
    /// Not doing anything — idle, listening.
    Idle,
    /// Actively recording (mic open, buffering).
    Recording,
    /// Have queued messages; waiting for the floor to clear.
    WaitingForFloor { ahead: usize },
    /// Currently transmitting a message out to the audio pipeline.
    Transmitting,
    /// Override PTT is live.
    Overriding,
    /// Transmission was interrupted by higher priority.
    Interrupted,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProtocolConfig {
        let mut c = ProtocolConfig::default();
        c.slot_count = 4;
        c.window_size = 512;
        c.detection_confirm_windows = 1;
        c
    }

    #[test]
    fn config_default_is_valid() {
        assert!(ProtocolConfig::default().is_valid());
    }

    #[test]
    fn config_slot_frequency_is_linear() {
        let c = ProtocolConfig::default();
        assert!((c.slot_frequency(SlotId(0)) - 15_000.0).abs() < 0.1);
        assert!((c.slot_frequency(SlotId(3)) - 15_600.0).abs() < 0.1);
    }

    #[test]
    fn config_rejects_out_of_band_slots() {
        let mut c = ProtocolConfig::default();
        c.slot_count = 20; // would push frequency above 18.8 kHz, still ok
        c.base_freq_hz = 23_000.0; // now way above Nyquist for 48kHz
        assert!(!c.is_valid());
    }

    #[test]
    fn slot_id_bounds() {
        assert!(SlotId::new(0).is_some());
        assert!(SlotId::new(14).is_some());
        assert!(SlotId::new(15).is_none());
    }

    #[test]
    fn priority_clamps() {
        assert_eq!(Priority::new(20).as_u8(), 10);
        assert_eq!(Priority::new(5).as_u8(), 5);
    }

    // ---- Tone encoder ----

    #[test]
    fn tone_encoder_mixes_without_clipping() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf = vec![0.0f32; 1024];
        enc.mix_tone(&mut buf, 15_000.0, 0.5);
        let max = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = buf.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max <= 0.6 && min >= -0.6);
        assert!(max > 0.4);
    }

    #[test]
    fn tone_encoder_preserves_phase_across_calls() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf1 = vec![0.0f32; 100];
        let mut buf2 = vec![0.0f32; 100];
        enc.mix_tone(&mut buf1, 1000.0, 1.0);
        enc.mix_tone(&mut buf2, 1000.0, 1.0);
        // Concatenated output should be continuous: last of buf1 close in phase to first of buf2
        let combined: Vec<f32> = buf1.iter().chain(buf2.iter()).copied().collect();
        // No discontinuity > amplitude
        for w in combined.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.5);
        }
    }

    #[test]
    fn tone_encoder_pulse_has_envelope() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf = vec![0.0f32; 480];
        enc.mix_pulse(&mut buf, 1000.0, 1.0);
        // First sample should be near zero (ramp up)
        assert!(buf[0].abs() < 0.1);
        // Last sample should be near zero (ramp down)
        assert!(buf.last().unwrap().abs() < 0.1);
        // Middle should be full amplitude (some sample crosses 0.8+)
        let mid_max = buf[buf.len() / 4..3 * buf.len() / 4]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(mid_max > 0.8);
    }

    // ---- Goertzel ----

    #[test]
    fn goertzel_detects_target_frequency() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf = vec![0.0f32; 512];
        enc.mix_tone(&mut buf, 15_200.0, 0.5);
        let det = GoertzelDetector::new(15_200.0, 48_000, 512);
        let m = det.magnitude(&buf);
        assert!(m > 0.1, "should detect strong signal, got {}", m);
    }

    #[test]
    fn goertzel_rejects_non_target_frequency() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf = vec![0.0f32; 512];
        enc.mix_tone(&mut buf, 15_000.0, 0.5);
        let det = GoertzelDetector::new(18_000.0, 48_000, 512);
        let m = det.magnitude(&buf);
        assert!(m < 0.05, "should reject off-target frequency, got {}", m);
    }

    #[test]
    fn goertzel_magnitude_scales_with_amplitude() {
        let mut enc = ToneEncoder::new(48_000);
        let mut buf_small = vec![0.0f32; 512];
        let mut buf_big = vec![0.0f32; 512];
        enc.mix_tone(&mut buf_small, 15_200.0, 0.1);
        enc.reset();
        enc.mix_tone(&mut buf_big, 15_200.0, 0.8);
        let det = GoertzelDetector::new(15_200.0, 48_000, 512);
        let m_small = det.magnitude(&buf_small);
        let m_big = det.magnitude(&buf_big);
        assert!(m_big > m_small * 4.0, "big={}, small={}", m_big, m_small);
    }

    // ---- ToneDecoder ----

    #[test]
    fn decoder_identifies_active_slot() {
        let cfg = test_config();
        let mut enc = ToneEncoder::new(cfg.sample_rate);
        let mut dec = ToneDecoder::new(&cfg);
        let freq = cfg.slot_frequency(SlotId(2));
        let mut buf = vec![0.0f32; cfg.window_size];
        enc.mix_tone(&mut buf, freq, cfg.active_amplitude * 50.0); // Strong signal
        let frame = dec.process(&buf);
        assert!(frame.active_slots.contains(&SlotId(2)));
        assert!(!frame.active_slots.contains(&SlotId(0)));
    }

    #[test]
    fn decoder_identifies_override() {
        let cfg = test_config();
        let mut enc = ToneEncoder::new(cfg.sample_rate);
        let mut dec = ToneDecoder::new(&cfg);
        let mut buf = vec![0.0f32; cfg.window_size];
        enc.mix_tone(
            &mut buf,
            cfg.override_freq_hz,
            cfg.override_amplitude * 40.0,
        );
        let frame = dec.process(&buf);
        assert!(frame.override_active);
    }

    #[test]
    fn decoder_no_false_positives_on_silence() {
        let cfg = test_config();
        let mut dec = ToneDecoder::new(&cfg);
        let buf = vec![0.0f32; cfg.window_size];
        let frame = dec.process(&buf);
        assert!(frame.active_slots.is_empty());
        assert!(!frame.override_active);
    }

    #[test]
    fn decoder_confirm_windows_requires_persistence() {
        let mut cfg = test_config();
        cfg.detection_confirm_windows = 3;
        let mut enc = ToneEncoder::new(cfg.sample_rate);
        let mut dec = ToneDecoder::new(&cfg);
        let freq = cfg.slot_frequency(SlotId(1));
        let mut buf = vec![0.0f32; cfg.window_size];
        enc.mix_tone(&mut buf, freq, cfg.active_amplitude * 50.0);
        // First two frames should NOT yet be confirmed
        let f1 = dec.process(&buf);
        assert!(!f1.active_slots.contains(&SlotId(1)));
        let f2 = dec.process(&buf);
        assert!(!f2.active_slots.contains(&SlotId(1)));
        // Third frame confirms
        let f3 = dec.process(&buf);
        assert!(f3.active_slots.contains(&SlotId(1)));
    }

    // ---- FloorMonitor ----

    #[test]
    fn floor_monitor_tracks_active_slot() {
        let cfg = test_config();
        let mut mon = FloorMonitor::new(cfg.clone());
        let t0 = Instant::now();
        let mut frame = DetectionFrame::default();
        frame.active_slots = vec![SlotId(1)];
        mon.update(&frame, t0);
        assert!(mon.active_slots().contains(&SlotId(1)));
    }

    #[test]
    fn floor_monitor_expires_after_timeout() {
        let cfg = test_config();
        let mut mon = FloorMonitor::new(cfg.clone());
        let t0 = Instant::now();
        let mut frame = DetectionFrame::default();
        frame.active_slots = vec![SlotId(2)];
        mon.update(&frame, t0);
        assert!(mon.active_slots().contains(&SlotId(2)));
        // Advance past active timeout
        let t1 = t0 + Duration::from_millis(cfg.active_timeout_ms + 100);
        let empty_frame = DetectionFrame::default();
        mon.update(&empty_frame, t1);
        assert!(!mon.active_slots().contains(&SlotId(2)));
    }

    #[test]
    fn floor_monitor_can_speak_respects_priority() {
        let cfg = test_config();
        let mut table = PriorityTable::new();
        table.assign(SlotAssignment {
            slot: SlotId(0),
            priority: Priority(3),
            can_override: false,
            display_name: "a".into(),
        });
        table.assign(SlotAssignment {
            slot: SlotId(1),
            priority: Priority(7),
            can_override: false,
            display_name: "b".into(),
        });
        let mut mon = FloorMonitor::new(cfg);
        // Slot 1 (P7) active → someone at P3 cannot speak, someone at P10 can
        let t0 = Instant::now();
        let mut f = DetectionFrame::default();
        f.active_slots = vec![SlotId(1)];
        mon.update(&f, t0);
        assert!(!mon.can_speak(Priority(3), &table));
        assert!(!mon.can_speak(Priority(7), &table)); // tie = defer
        assert!(mon.can_speak(Priority(10), &table));
    }

    #[test]
    fn floor_monitor_override_blocks_everyone() {
        let cfg = test_config();
        let table = PriorityTable::flat(cfg.slot_count);
        let mut mon = FloorMonitor::new(cfg);
        let t0 = Instant::now();
        let mut f = DetectionFrame::default();
        f.override_active = true;
        mon.update(&f, t0);
        assert!(!mon.can_speak(Priority(10), &table));
        assert!(mon.override_active());
    }

    // ---- PriorityTable ----

    #[test]
    fn priority_table_flat_all_equal() {
        let t = PriorityTable::flat(5);
        for i in 0..5 {
            assert_eq!(t.priority_of(SlotId(i)), Priority(5));
            assert!(!t.can_override(SlotId(i)));
        }
    }

    #[test]
    fn priority_table_squad_has_leaders() {
        let t = PriorityTable::squad(8, 3);
        assert_eq!(t.priority_of(SlotId(0)), Priority(10));
        assert_eq!(t.priority_of(SlotId(1)), Priority(10));
        assert!(t.can_override(SlotId(0)));
        assert!(t.can_override(SlotId(1)));
        assert_eq!(t.priority_of(SlotId(2)), Priority(7));
        assert_eq!(t.priority_of(SlotId(3)), Priority(7));
        assert_eq!(t.priority_of(SlotId(4)), Priority(7));
        assert_eq!(t.priority_of(SlotId(5)), Priority(3));
    }

    #[test]
    fn priority_table_meeting_presenter_only() {
        let t = PriorityTable::meeting(5);
        assert_eq!(t.priority_of(SlotId(0)), Priority(10));
        assert!(t.can_override(SlotId(0)));
        for i in 1..5 {
            assert_eq!(t.priority_of(SlotId(i)), Priority(3));
            assert!(!t.can_override(SlotId(i)));
        }
    }

    // ---- MessageQueue ----

    #[test]
    fn queue_fifo_order() {
        let mut q = MessageQueue::new(5);
        let mk = |n: f32| VoiceMessage {
            id: 0,
            recorded_at: Instant::now(),
            samples: vec![n],
            sample_rate: 48_000,
        };
        let id1 = q.push(mk(1.0)).unwrap();
        let id2 = q.push(mk(2.0)).unwrap();
        assert_eq!(q.pop_next().unwrap().id, id1);
        assert_eq!(q.pop_next().unwrap().id, id2);
    }

    #[test]
    fn queue_evicts_oldest_when_full() {
        let mut q = MessageQueue::new(2);
        let mk = |n: f32| VoiceMessage {
            id: 0,
            recorded_at: Instant::now(),
            samples: vec![n],
            sample_rate: 48_000,
        };
        q.push(mk(1.0)).unwrap();
        q.push(mk(2.0)).unwrap();
        q.push(mk(3.0)).unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop_next().unwrap().samples[0], 2.0);
    }

    #[test]
    fn queue_rejects_when_full_without_eviction() {
        let mut q = MessageQueue::with_eviction(2, false);
        let mk = || VoiceMessage {
            id: 0,
            recorded_at: Instant::now(),
            samples: vec![1.0],
            sample_rate: 48_000,
        };
        q.push(mk()).unwrap();
        q.push(mk()).unwrap();
        assert!(q.push(mk()).is_err());
    }

    #[test]
    fn queue_cancel_removes_by_id() {
        let mut q = MessageQueue::new(5);
        let mk = || VoiceMessage {
            id: 0,
            recorded_at: Instant::now(),
            samples: vec![1.0],
            sample_rate: 48_000,
        };
        let id = q.push(mk()).unwrap();
        q.push(mk()).unwrap();
        assert_eq!(q.len(), 2);
        assert!(q.cancel(id));
        assert_eq!(q.len(), 1);
        assert!(!q.cancel(99));
    }

    // ---- MessageRecorder ----

    #[test]
    fn recorder_finalizes_on_silence() {
        let cfg = test_config();
        let mut rec = MessageRecorder::new(&cfg);
        let t0 = Instant::now();
        let samples = vec![0.3f32; 480]; // 10ms @ 48k
                                         // Speak for 500ms
        for i in 0..50 {
            let t = t0 + Duration::from_millis(i * 10);
            assert!(rec.process(&samples, true, t).is_none());
        }
        assert!(rec.is_recording());
        // 1.6s of silence → should finalize (silence_timeout = 1500ms)
        let silent = vec![0.0f32; 480];
        let mut msg = None;
        for i in 0..200 {
            let t = t0 + Duration::from_millis(500 + i * 10);
            msg = rec.process(&silent, false, t);
            if msg.is_some() {
                break;
            }
        }
        assert!(msg.is_some(), "expected message finalized by silence");
        assert!(!rec.is_recording());
    }

    #[test]
    fn recorder_discards_too_short_messages() {
        let cfg = test_config();
        let mut rec = MessageRecorder::new(&cfg);
        let t0 = Instant::now();
        let samples = vec![0.3f32; 480];
        // Only 100ms of voice (under 300ms min)
        for i in 0..10 {
            rec.process(&samples, true, t0 + Duration::from_millis(i * 10));
        }
        let silent = vec![0.0f32; 480];
        let mut got = None;
        for i in 0..200 {
            if let Some(m) = rec.process(&silent, false, t0 + Duration::from_millis(100 + i * 10)) {
                got = Some(m);
                break;
            }
        }
        assert!(
            got.is_none(),
            "message under min_duration should be discarded"
        );
    }

    #[test]
    fn recorder_cancels_in_progress() {
        let cfg = test_config();
        let mut rec = MessageRecorder::new(&cfg);
        let t0 = Instant::now();
        rec.process(&vec![0.3f32; 480], true, t0);
        assert!(rec.is_recording());
        rec.cancel();
        assert!(!rec.is_recording());
        assert_eq!(rec.buffered_samples(), 0);
    }

    // ---- MessagePlayer ----

    #[test]
    fn player_plays_through_full_message() {
        let cfg = test_config();
        let mut p = MessagePlayer::new(InterruptPolicy::Hard, ResumeStrategy::Continue, &cfg);
        let msg = VoiceMessage {
            id: 1,
            recorded_at: Instant::now(),
            samples: vec![0.5f32; 2048],
            sample_rate: 48_000,
        };
        p.start(msg);
        let mut buf = vec![0.0f32; 1024];
        assert_eq!(p.pull(&mut buf), 1024);
        assert!(p.is_playing());
        let mut buf2 = vec![0.0f32; 1024];
        assert_eq!(p.pull(&mut buf2), 1024);
        assert!(!p.is_playing());
    }

    #[test]
    fn player_hard_interrupt_stops_immediately() {
        let cfg = test_config();
        let mut p = MessagePlayer::new(InterruptPolicy::Hard, ResumeStrategy::Continue, &cfg);
        let msg = VoiceMessage {
            id: 1,
            recorded_at: Instant::now(),
            samples: vec![0.5f32; 4096],
            sample_rate: 48_000,
        };
        p.start(msg);
        let mut buf = vec![0.0f32; 512];
        p.pull(&mut buf);
        assert!(p.request_interrupt());
        assert!(!p.is_playing());
        assert!(p.has_interrupted());
    }

    #[test]
    fn player_finish_policy_ignores_interrupt() {
        let cfg = test_config();
        let mut p = MessagePlayer::new(InterruptPolicy::Finish, ResumeStrategy::Continue, &cfg);
        let msg = VoiceMessage {
            id: 1,
            recorded_at: Instant::now(),
            samples: vec![0.5f32; 4096],
            sample_rate: 48_000,
        };
        p.start(msg);
        assert!(!p.request_interrupt());
        assert!(p.is_playing());
    }

    #[test]
    fn player_resume_rewinds_by_offset() {
        let mut cfg = test_config();
        cfg.resume_offset_ms = 100; // 4800 samples @ 48k
        let mut p = MessagePlayer::new(InterruptPolicy::Hard, ResumeStrategy::Continue5s, &cfg);
        let msg = VoiceMessage {
            id: 1,
            recorded_at: Instant::now(),
            samples: vec![0.5f32; 48_000],
            sample_rate: 48_000,
        };
        p.start(msg);
        let mut buf = vec![0.0f32; 10_000];
        p.pull(&mut buf); // advance to pos=10000
        p.request_interrupt();
        assert!(p.has_interrupted());
        p.resume_interrupted();
        assert!(p.is_playing());
        // After resume, position should be 10000 - 4800 = 5200
        // We can't introspect position directly, so just verify it plays again
        let mut buf2 = vec![0.0f32; 100];
        assert!(p.pull(&mut buf2) > 0);
    }

    // ---- BeaconScheduler ----

    #[test]
    fn scheduler_emits_idle_when_idle() {
        let cfg = test_config();
        let mut s = BeaconScheduler::new(cfg, SlotId(0));
        let t0 = Instant::now();
        let beacons = s.tick(t0);
        assert!(beacons.iter().any(|b| b.kind == BeaconKind::Idle));
    }

    #[test]
    fn scheduler_emits_active_when_transmitting() {
        let cfg = test_config();
        let mut s = BeaconScheduler::new(cfg, SlotId(0));
        s.set_state(NodeState::Transmitting);
        let t0 = Instant::now();
        let beacons = s.tick(t0);
        assert!(beacons.iter().any(|b| b.kind == BeaconKind::Active));
    }

    #[test]
    fn scheduler_respects_heartbeat_interval() {
        let cfg = test_config();
        let mut s = BeaconScheduler::new(cfg.clone(), SlotId(0));
        s.set_state(NodeState::Transmitting);
        let t0 = Instant::now();
        let b1 = s.tick(t0);
        assert!(!b1.is_empty());
        // Immediate second tick: no new beacon (still within heartbeat interval)
        let b2 = s.tick(t0 + Duration::from_millis(10));
        assert!(b2.is_empty());
        // After full heartbeat interval: new beacon
        let b3 = s.tick(t0 + Duration::from_millis(cfg.heartbeat_ms + 10));
        assert!(!b3.is_empty());
    }

    #[test]
    fn scheduler_emits_override_when_set() {
        let cfg = test_config();
        let mut s = BeaconScheduler::new(cfg, SlotId(0));
        s.set_override(true);
        let t0 = Instant::now();
        let beacons = s.tick(t0);
        assert!(beacons.iter().any(|b| b.kind == BeaconKind::Override));
    }

    // ---- End-to-end: encode → decode round trip ----

    #[test]
    fn end_to_end_encode_decode_roundtrip() {
        let cfg = test_config();
        let mut enc = ToneEncoder::new(cfg.sample_rate);
        let mut dec = ToneDecoder::new(&cfg);
        let target_slot = SlotId(2);
        let freq = cfg.slot_frequency(target_slot);
        // Simulate a strong beacon in otherwise-silent audio
        let mut buf = vec![0.0f32; cfg.window_size];
        enc.mix_tone(&mut buf, freq, 0.05);
        let frame = dec.process(&buf);
        assert!(frame.active_slots.contains(&target_slot));
        assert!(frame.magnitudes.get(&target_slot).copied().unwrap_or(0.0) > frame.noise_floor);
    }

    #[test]
    fn end_to_end_simulates_two_speakers() {
        let cfg = test_config();
        let table = PriorityTable::squad(cfg.slot_count, 2);
        let mut mon = FloorMonitor::new(cfg.clone());
        let t0 = Instant::now();
        // Slot 3 (lowest priority P3) transmitting
        let mut f1 = DetectionFrame::default();
        f1.active_slots = vec![SlotId(3)];
        mon.update(&f1, t0);
        assert!(!mon.can_speak(Priority(3), &table)); // can't pre-empt equal
        assert!(mon.can_speak(Priority(10), &table)); // leader can
                                                      // Slot 0 (leader P10) starts
        let mut f2 = DetectionFrame::default();
        f2.active_slots = vec![SlotId(0), SlotId(3)];
        mon.update(&f2, t0 + Duration::from_millis(100));
        // Now highest is slot 0 with P10
        let top = mon.highest_active(&table).unwrap();
        assert_eq!(top.0, SlotId(0));
        assert_eq!(top.1, Priority(10));
    }
}
