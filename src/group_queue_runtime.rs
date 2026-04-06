//! GroupQueueRuntime — integrated driver for the acoustic priority protocol.
//!
//! Wraps the components from [`crate::audio_priority_protocol`] into a single
//! tickable runtime that fits inside an audio pipeline callback. Callers feed
//! it chunks of:
//!
//! - **Local mic samples** + VAD decision → builds outbound messages.
//! - **Loopback samples** (Discord / voice-chat output) → detects remote beacons.
//!
//! And pulls back:
//!
//! - **Output samples** — mixed player audio + beacon pulses, ready to feed the
//!   virtual mic / speaker output.
//!
//! This keeps the GUI / audio thread code simple: one struct, one tick, done.

use crate::audio_priority_protocol::*;
use std::time::{Duration, Instant};

/// Capture mode — how the user produces messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CaptureMode {
    /// Auto-record via VAD; message finalised on silence timeout.
    Vad,
    /// Record only while PTT key is held.
    PushToTalk,
    /// Live transmit, bypass queue (only allowed for authorized slots).
    OverridePtt,
    /// Continuous — chunks emitted every `max_message_ms`.
    Continuous,
}

/// High-level status exposed to the GUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeStatus {
    /// Group-queue disabled.
    Disabled,
    /// Enabled, no activity.
    Idle,
    /// Actively recording user voice.
    Recording,
    /// Have queued messages, waiting for the floor to clear.
    Waiting { queued: usize },
    /// Transmitting a queued message.
    Transmitting { queued: usize },
    /// Override PTT active — live transmission bypassing the queue.
    Overriding,
    /// Transmission interrupted by higher-priority speaker.
    Interrupted { queued: usize },
}

/// Runtime configuration — mirrors GUI controls. Safe to snapshot & mutate.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub enabled: bool,
    pub my_slot: SlotId,
    pub capture_mode: CaptureMode,
    pub interrupt_policy: InterruptPolicy,
    pub resume_strategy: ResumeStrategy,
    pub protocol: ProtocolConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            my_slot: SlotId(0),
            capture_mode: CaptureMode::Vad,
            interrupt_policy: InterruptPolicy::Soft,
            resume_strategy: ResumeStrategy::Continue5s,
            protocol: ProtocolConfig::default(),
        }
    }
}

/// Snapshot of peer state for GUI rendering.
#[derive(Clone, Debug)]
pub struct PeerSnapshot {
    pub slot: SlotId,
    pub priority: Priority,
    pub display_name: String,
    pub connected: bool,
    pub transmitting: bool,
}

/// All the runtime state the integrated protocol needs.
pub struct GroupQueueRuntime {
    config: RuntimeConfig,
    table: PriorityTable,
    scheduler: BeaconScheduler,
    encoder: ToneEncoder,
    decoder: ToneDecoder,
    monitor: FloorMonitor,
    recorder: MessageRecorder,
    queue: MessageQueue,
    player: MessagePlayer,

    status: RuntimeStatus,
    ptt_held: bool,
    override_held: bool,
    /// Instant the runtime started, used for startup jitter.
    started_at: Instant,
    /// Our random offset (0..jitter_max_ms) applied before first beacon.
    jitter_offset_ms: u64,
    /// Window of loopback samples being accumulated for the decoder.
    decoder_accum: Vec<f32>,
}

impl GroupQueueRuntime {
    pub fn new(config: RuntimeConfig, table: PriorityTable) -> Self {
        let encoder = ToneEncoder::new(config.protocol.sample_rate);
        let decoder = ToneDecoder::new(&config.protocol);
        let monitor = FloorMonitor::new(config.protocol.clone());
        let scheduler = BeaconScheduler::new(config.protocol.clone(), config.my_slot);
        let recorder = MessageRecorder::new(&config.protocol);
        let queue = MessageQueue::new(config.protocol.max_queue);
        let player = MessagePlayer::new(
            config.interrupt_policy,
            config.resume_strategy,
            &config.protocol,
        );
        // Jitter offset based on slot id (deterministic) so restarts don't
        // keep producing different values, but distinct slots avoid collisions.
        let jitter_offset_ms =
            (config.my_slot.as_u8() as u64 * 137) % config.protocol.jitter_max_ms.max(1);
        Self {
            config,
            table,
            scheduler,
            encoder,
            decoder,
            monitor,
            recorder,
            queue,
            player,
            status: RuntimeStatus::Disabled,
            ptt_held: false,
            override_held: false,
            started_at: Instant::now(),
            jitter_offset_ms,
            decoder_accum: Vec::new(),
        }
    }

    /// Replace config (slot, mode, policy, etc.). Rebuilds anything affected.
    pub fn reconfigure(&mut self, new_config: RuntimeConfig, table: PriorityTable) {
        let slot_changed = new_config.my_slot != self.config.my_slot;
        let protocol_changed = new_config.protocol.sample_rate != self.config.protocol.sample_rate
            || new_config.protocol.slot_count != self.config.protocol.slot_count
            || new_config.protocol.window_size != self.config.protocol.window_size
            || new_config.protocol.base_freq_hz != self.config.protocol.base_freq_hz;
        if protocol_changed {
            self.decoder = ToneDecoder::new(&new_config.protocol);
            self.monitor = FloorMonitor::new(new_config.protocol.clone());
            self.encoder.reset();
        }
        if slot_changed || protocol_changed {
            self.scheduler = BeaconScheduler::new(new_config.protocol.clone(), new_config.my_slot);
        }
        let silence_changed = new_config.protocol.silence_timeout_ms
            != self.config.protocol.silence_timeout_ms
            || new_config.protocol.min_message_ms != self.config.protocol.min_message_ms
            || new_config.protocol.max_message_ms != self.config.protocol.max_message_ms;
        if silence_changed || protocol_changed {
            self.recorder = MessageRecorder::new(&new_config.protocol);
        }
        let player_changed = new_config.interrupt_policy != self.config.interrupt_policy
            || new_config.resume_strategy != self.config.resume_strategy;
        if player_changed || protocol_changed {
            self.player = MessagePlayer::new(
                new_config.interrupt_policy,
                new_config.resume_strategy,
                &new_config.protocol,
            );
        }
        self.config = new_config;
        self.table = table;
        if !self.config.enabled {
            self.status = RuntimeStatus::Disabled;
        } else if self.status == RuntimeStatus::Disabled {
            self.status = RuntimeStatus::Idle;
        }
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
    pub fn status(&self) -> &RuntimeStatus {
        &self.status
    }
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// User pressed/released the PTT key.
    pub fn set_ptt(&mut self, held: bool) {
        self.ptt_held = held;
        if self.config.capture_mode == CaptureMode::PushToTalk
            && !held
            && self.recorder.is_recording()
        {
            // Finalize on release
            if let Some(msg) = self.recorder.finalize(Instant::now()) {
                let _ = self.queue.push(msg);
            }
        }
    }

    /// User pressed/released the Override PTT key.
    pub fn set_override(&mut self, held: bool) {
        // Only authorized slots can actually override
        let allowed = self.table.can_override(self.config.my_slot);
        self.override_held = held && allowed;
        self.scheduler.set_override(self.override_held);
        if self.override_held {
            self.status = RuntimeStatus::Overriding;
        } else if self.player.is_playing() {
            // Return to regular state after override released
            self.status = RuntimeStatus::Transmitting {
                queued: self.queue.len(),
            };
        } else {
            self.status = if self.queue.is_empty() {
                RuntimeStatus::Idle
            } else {
                RuntimeStatus::Waiting {
                    queued: self.queue.len(),
                }
            };
        }
    }

    /// Cancel the current message recording in progress.
    pub fn cancel_recording(&mut self) {
        self.recorder.cancel();
    }

    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }

    /// Drop the message currently in playback and return to waiting state.
    pub fn stop_playback(&mut self) {
        if self.player.is_playing() {
            self.player.request_interrupt();
        }
    }

    /// Snapshot of all peers for GUI display.
    pub fn peers_snapshot(&self) -> Vec<PeerSnapshot> {
        let mut out = Vec::new();
        for slot_u8 in 0..self.config.protocol.slot_count {
            let slot = SlotId(slot_u8);
            let assignment = self.table.get(slot);
            let state = self.monitor.slot_state(slot);
            out.push(PeerSnapshot {
                slot,
                priority: assignment.map(|a| a.priority).unwrap_or(Priority::MIN),
                display_name: assignment
                    .map(|a| a.display_name.clone())
                    .unwrap_or_else(|| format!("Slot {}", slot_u8)),
                connected: state.map(|s| s.currently_connected).unwrap_or(false),
                transmitting: state.map(|s| s.currently_active).unwrap_or(false),
            });
        }
        out
    }

    /// Feed loopback (incoming voice-chat) samples to the decoder. Beacons
    /// from other clients are detected here; our own slot is filtered out.
    pub fn process_loopback(&mut self, samples: &[f32], now: Instant) {
        if !self.config.enabled {
            return;
        }
        let win = self.config.protocol.window_size;
        self.decoder_accum.extend_from_slice(samples);
        while self.decoder_accum.len() >= win {
            let frame: Vec<f32> = self.decoder_accum.drain(..win).collect();
            let mut det = self.decoder.process(&frame);
            // Filter our own slot out of the detection (we hear ourselves via loopback)
            det.active_slots.retain(|s| *s != self.config.my_slot);
            self.monitor.update(&det, now);
        }
    }

    /// Feed mic samples + VAD verdict. Builds messages into the local queue.
    /// Returns a ready-to-play override buffer if we're actively Overriding.
    pub fn process_mic(
        &mut self,
        samples: &[f32],
        is_voice: bool,
        now: Instant,
    ) -> Option<Vec<f32>> {
        if !self.config.enabled {
            return None;
        }

        // Override PTT: bypass queue, return samples directly
        if self.override_held {
            return Some(samples.to_vec());
        }

        let should_record = match self.config.capture_mode {
            CaptureMode::Vad => is_voice,
            CaptureMode::PushToTalk => self.ptt_held,
            CaptureMode::Continuous => true,
            CaptureMode::OverridePtt => false, // handled above
        };

        if let Some(msg) = self.recorder.process(samples, should_record, now) {
            // Enforce rate limit via max_queue; push evicts oldest on full.
            let _ = self.queue.push(msg);
        }
        None
    }

    /// Main pipeline tick. Fills `output` (sample length) with:
    /// - player samples (if transmitting)
    /// - beacon pulses mixed in (IDLE/ACTIVE/END/OVERRIDE as scheduled)
    pub fn tick_output(&mut self, output: &mut [f32], now: Instant) {
        // Respect startup jitter: stay silent until jitter delay elapsed
        let wait = self
            .config
            .protocol
            .jitter_max_ms
            .saturating_sub(self.jitter_offset_ms);
        if now.duration_since(self.started_at) < Duration::from_millis(wait) {
            for s in output.iter_mut() {
                *s = 0.0;
            }
            return;
        }

        if !self.config.enabled {
            for s in output.iter_mut() {
                *s = 0.0;
            }
            return;
        }

        // 1) Handle playback / queue transitions
        self.advance_state(now);

        // 2) Player fills with message samples (or silence)
        for s in output.iter_mut() {
            *s = 0.0;
        }
        if self.player.is_playing() {
            self.player.pull(output);
        }

        // 3) Scheduler emits beacon pulses — mix into output
        let beacons = self.scheduler.tick(now);
        for beacon in beacons {
            let samples_for_pulse = ((beacon.duration_ms as usize)
                * (self.config.protocol.sample_rate as usize))
                / 1000;
            let n = samples_for_pulse.min(output.len());
            if n == 0 {
                continue;
            }
            let mut pulse_buf = vec![0.0f32; n];
            self.encoder
                .mix_pulse(&mut pulse_buf, beacon.frequency_hz, beacon.amplitude);
            for (i, &b) in pulse_buf.iter().enumerate() {
                output[i] += b;
            }
        }
    }

    /// Internal state transitions: handle floor changes, interruption, etc.
    fn advance_state(&mut self, _now: Instant) {
        if self.override_held {
            return;
        } // status is `Overriding`, nothing else to do

        let my_priority = self.table.priority_of(self.config.my_slot);
        let can = self.monitor.can_speak(my_priority, &self.table);

        // External override blocks everything
        if self.monitor.override_active() && !self.override_held {
            if self.player.is_playing() {
                self.player.request_interrupt();
                self.status = RuntimeStatus::Interrupted {
                    queued: self.queue.len(),
                };
                self.scheduler.set_state(NodeState::Idle);
            }
            return;
        }

        // If higher priority slot is active, interrupt our playback
        if self.player.is_playing() && !can {
            self.player.request_interrupt();
            self.status = RuntimeStatus::Interrupted {
                queued: self.queue.len(),
            };
            self.scheduler.set_state(NodeState::Idle);
            return;
        }

        // If player finished, emit END beacon then return to idle
        if !self.player.is_playing() && self.scheduler.state() == NodeState::Transmitting {
            self.scheduler.set_state(NodeState::EndingMessage);
        }

        // If not playing and queue has messages and floor clear, start next
        if !self.player.is_playing() && !self.queue.is_empty() && can {
            // First resume interrupted message if any
            if self.player.has_interrupted() {
                self.player.resume_interrupted();
            } else if let Some(msg) = self.queue.pop_next() {
                self.player.start(msg);
            }
            self.scheduler.set_state(NodeState::Transmitting);
            self.status = RuntimeStatus::Transmitting {
                queued: self.queue.len(),
            };
            return;
        }

        // Status updates when idle
        if !self.player.is_playing() {
            self.status = if self.recorder.is_recording() {
                RuntimeStatus::Recording
            } else if !self.queue.is_empty() {
                RuntimeStatus::Waiting {
                    queued: self.queue.len(),
                }
            } else {
                RuntimeStatus::Idle
            };
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> RuntimeConfig {
        let mut c = RuntimeConfig::default();
        c.enabled = true;
        c.protocol.slot_count = 4;
        c.protocol.window_size = 512;
        c.protocol.detection_confirm_windows = 1;
        c.protocol.min_message_ms = 50;
        c.protocol.silence_timeout_ms = 200;
        c.protocol.jitter_max_ms = 0; // disable jitter for deterministic tests
        c
    }

    #[test]
    fn runtime_new_starts_disabled_status_after_construction() {
        let cfg = RuntimeConfig::default();
        let rt = GroupQueueRuntime::new(cfg, PriorityTable::flat(8));
        assert_eq!(*rt.status(), RuntimeStatus::Disabled);
    }

    #[test]
    fn runtime_enabled_is_idle_initially() {
        let mut rt = GroupQueueRuntime::new(test_cfg(), PriorityTable::flat(4));
        let mut out = vec![0.0f32; 512];
        rt.tick_output(&mut out, Instant::now());
        assert_eq!(*rt.status(), RuntimeStatus::Idle);
    }

    #[test]
    fn runtime_vad_creates_messages() {
        let mut rt = GroupQueueRuntime::new(test_cfg(), PriorityTable::flat(4));
        let t0 = Instant::now();
        let voice = vec![0.3f32; 4800]; // 100ms @ 48k
                                        // 500ms of voice
        for i in 0..5 {
            let t = t0 + Duration::from_millis(i * 100);
            rt.process_mic(&voice, true, t);
        }
        // 300ms silence triggers finalize (silence_timeout=200ms)
        let silence = vec![0.0f32; 4800];
        for i in 0..3 {
            let t = t0 + Duration::from_millis(500 + i * 100);
            rt.process_mic(&silence, false, t);
        }
        assert!(
            rt.queue_len() >= 1,
            "queue should have a message after VAD finalize"
        );
    }

    #[test]
    fn runtime_ptt_records_while_held() {
        let mut cfg = test_cfg();
        cfg.capture_mode = CaptureMode::PushToTalk;
        let mut rt = GroupQueueRuntime::new(cfg, PriorityTable::flat(4));
        let t0 = Instant::now();
        let voice = vec![0.3f32; 4800];
        // PTT off, voice → no recording
        rt.process_mic(&voice, true, t0);
        assert_eq!(rt.queue_len(), 0);
        // PTT on, record
        rt.set_ptt(true);
        for i in 0..5 {
            rt.process_mic(&voice, true, t0 + Duration::from_millis(i * 100));
        }
        // PTT off, should finalize
        rt.set_ptt(false);
        assert!(rt.queue_len() >= 1);
    }

    #[test]
    fn runtime_override_held_bypasses_queue() {
        let mut cfg = test_cfg();
        cfg.capture_mode = CaptureMode::OverridePtt;
        cfg.my_slot = SlotId(0);
        let rt_cfg_my_slot = cfg.my_slot;
        let mut table = PriorityTable::new();
        table.assign(SlotAssignment {
            slot: rt_cfg_my_slot,
            priority: Priority(10),
            can_override: true,
            display_name: "Leader".into(),
        });
        let mut rt = GroupQueueRuntime::new(cfg, table);
        rt.set_override(true);
        assert_eq!(*rt.status(), RuntimeStatus::Overriding);
        let mic = vec![0.3f32; 100];
        let out = rt.process_mic(&mic, true, Instant::now());
        assert!(out.is_some());
        assert_eq!(out.unwrap().len(), 100);
    }

    #[test]
    fn runtime_override_blocked_for_non_authorized() {
        let mut cfg = test_cfg();
        cfg.my_slot = SlotId(2);
        let table = PriorityTable::flat(4); // all have can_override = false
        let mut rt = GroupQueueRuntime::new(cfg, table);
        rt.set_override(true);
        assert_ne!(*rt.status(), RuntimeStatus::Overriding);
    }

    #[test]
    fn runtime_peers_snapshot_shows_all_slots() {
        let rt = GroupQueueRuntime::new(test_cfg(), PriorityTable::squad(4, 1));
        let peers = rt.peers_snapshot();
        assert_eq!(peers.len(), 4);
        assert_eq!(peers[0].priority, Priority(10));
        assert_eq!(peers[1].priority, Priority(10));
        assert_eq!(peers[2].priority, Priority(7));
        assert_eq!(peers[3].priority, Priority(3));
    }

    #[test]
    fn runtime_reconfigure_applies_new_slot() {
        let mut cfg = test_cfg();
        let mut rt = GroupQueueRuntime::new(cfg.clone(), PriorityTable::flat(4));
        cfg.my_slot = SlotId(2);
        rt.reconfigure(cfg, PriorityTable::flat(4));
        assert_eq!(rt.config().my_slot, SlotId(2));
    }

    #[test]
    fn runtime_emits_beacons_in_tick_output() {
        let mut rt = GroupQueueRuntime::new(test_cfg(), PriorityTable::flat(4));
        let mut out = vec![0.0f32; 4800]; // 100ms
        rt.tick_output(&mut out, Instant::now());
        // Should have emitted an IDLE beacon → some non-zero sample
        let peak = out.iter().cloned().fold(0f32, f32::max);
        assert!(
            peak > 0.0,
            "idle beacon should produce output, peak={}",
            peak
        );
    }

    #[test]
    fn runtime_clear_queue_removes_messages() {
        let mut rt = GroupQueueRuntime::new(test_cfg(), PriorityTable::flat(4));
        let t0 = Instant::now();
        let voice = vec![0.3f32; 4800];
        for i in 0..5 {
            rt.process_mic(&voice, true, t0 + Duration::from_millis(i * 100));
        }
        let silence = vec![0.0f32; 4800];
        for i in 0..3 {
            rt.process_mic(&silence, false, t0 + Duration::from_millis(500 + i * 100));
        }
        let len_before = rt.queue_len();
        rt.clear_queue();
        assert!(len_before > 0);
        assert_eq!(rt.queue_len(), 0);
    }
}
