//! WASM runtime for Skill Forge (wasmtime-backed, fuel + memory limited).
//!
//! v1 scope (this module):
//! - Engine/store construction with fuel metering enabled.
//! - `ResourceLimiter` enforcing `WasmRunConfig::memory_bytes`.
//! - Pure-compute ABI: the guest exports `run(input_ptr, input_len) -> u64`
//!   where the return packs `(ptr << 32) | len` of the output bytes in
//!   guest linear memory. Host reads output, then drops the store. No WASI
//!   in v1 — skills that need I/O should use Declarative mode + ToolCall.
//! - Source Rust path + compile fingerprint are carried on `WasmArtifact`
//!   but **compilation** (rustc → wasm32-wasip1) is a caller responsibility;
//!   the crate executes pre-built artifacts. A compile driver is on the
//!   roadmap (see `skill_forge_compile.rs`, future work).
//!
//! This module is feature-gated behind `skill-forge` because wasmtime is a
//! heavy dependency. Without the feature, only the config types are
//! available (as a thin shim) so downstream code can reference them.

use super::registry::{SkillError, SkillId, SkillOutput, WasmArtifact};
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

/// Runtime limits for a WASM skill invocation.
#[derive(Debug, Clone)]
pub struct WasmRunConfig {
    /// Maximum fuel (≈ instructions). Default: 1e9.
    pub max_fuel: u64,
    /// Maximum linear-memory bytes. Default: 64 MiB.
    pub max_memory_bytes: usize,
    /// Wall-clock timeout. Default: 30 s.
    pub wall_timeout: Duration,
    /// Maximum size of the input blob passed to the guest (bytes).
    pub max_input_bytes: usize,
    /// Maximum size of the output blob produced by the guest (bytes).
    pub max_output_bytes: usize,
}

impl Default for WasmRunConfig {
    fn default() -> Self {
        Self {
            max_fuel: super::wasm_limits::DEFAULT_WASM_FUEL,
            max_memory_bytes: super::wasm_limits::DEFAULT_WASM_MEMORY_BYTES,
            wall_timeout: Duration::from_secs(super::wasm_limits::DEFAULT_WASM_TIMEOUT_SECS),
            max_input_bytes: 1024 * 1024,      // 1 MiB
            max_output_bytes: 4 * 1024 * 1024, // 4 MiB
        }
    }
}

/// Runtime that executes WASM skills under the configured limits.
#[cfg(feature = "skill-forge")]
pub struct WasmRuntime {
    engine: wasmtime::Engine,
    config: WasmRunConfig,
}

#[cfg(feature = "skill-forge")]
impl WasmRuntime {
    /// Construct a new runtime with the given limits.
    pub fn new(config: WasmRunConfig) -> Result<Self, SkillError> {
        let mut wcfg = wasmtime::Config::new();
        wcfg.consume_fuel(true);
        // Epoch interruption allows us to impose a wall-clock timeout without
        // pre-emption. We increment the epoch from a background thread.
        wcfg.epoch_interruption(true);
        wcfg.wasm_bulk_memory(true);
        let engine = wasmtime::Engine::new(&wcfg)
            .map_err(|e| SkillError::Io(format!("wasmtime engine: {e}")))?;
        Ok(Self { engine, config })
    }

    /// Execute a skill artifact. Returns `SkillOutput.value` containing the
    /// guest's output (parsed as JSON if valid; raw string otherwise).
    pub fn execute(
        &self,
        skill_id: &SkillId,
        artifact: &WasmArtifact,
        input_bytes: &[u8],
    ) -> Result<SkillOutput, SkillError> {
        if input_bytes.len() > self.config.max_input_bytes {
            return Err(SkillError::BadInput {
                skill: skill_id.clone(),
                message: format!(
                    "input too large: {} > {}",
                    input_bytes.len(),
                    self.config.max_input_bytes
                ),
            });
        }

        let module = wasmtime::Module::new(&self.engine, &artifact.bytes).map_err(|e| {
            SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: format!("module compile: {e}"),
            }
        })?;

        let limits = MemoryLimits {
            max_memory_bytes: self.config.max_memory_bytes,
        };

        let mut store = wasmtime::Store::new(&self.engine, limits);
        store.limiter(|s| s as &mut dyn wasmtime::ResourceLimiter);
        store
            .set_fuel(self.config.max_fuel)
            .map_err(|e| SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: format!("set_fuel: {e}"),
            })?;
        store.set_epoch_deadline(1);

        let linker = wasmtime::Linker::new(&self.engine);
        let instance =
            linker
                .instantiate(&mut store, &module)
                .map_err(|e| SkillError::ExecutionFailed {
                    skill: skill_id.clone(),
                    message: format!("instantiate: {e}"),
                })?;

        // Start wall-clock watchdog that increments the epoch at deadline.
        let engine_clone = self.engine.clone();
        let wall = self.config.wall_timeout;
        let watchdog = std::thread::spawn(move || {
            std::thread::sleep(wall);
            engine_clone.increment_epoch();
        });

        let start = Instant::now();
        let result = self.run_guest(&mut store, &instance, skill_id, input_bytes);
        let elapsed = start.elapsed();

        // Best-effort: signal watchdog thread to exit. We don't join here
        // because the thread may still be sleeping; detaching is fine — it
        // just bumps the epoch on a now-disposed engine, which is a no-op.
        drop(watchdog);

        let consumed = self.config.max_fuel - store.get_fuel().unwrap_or(0);
        let mut output = result?;
        output.fuel_consumed = consumed;
        output.wall_ms = elapsed.as_millis() as u64;
        Ok(output)
    }

    fn run_guest(
        &self,
        store: &mut wasmtime::Store<MemoryLimits>,
        instance: &wasmtime::Instance,
        skill_id: &SkillId,
        input_bytes: &[u8],
    ) -> Result<SkillOutput, SkillError> {
        let memory = instance.get_memory(&mut *store, "memory").ok_or_else(|| {
            SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: "guest does not export 'memory'".into(),
            }
        })?;
        let alloc = instance
            .get_typed_func::<u32, u32>(&mut *store, "alloc")
            .map_err(|e| SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: format!("guest missing 'alloc(u32) -> u32': {e}"),
            })?;
        let run = instance
            .get_typed_func::<(u32, u32), u64>(&mut *store, "run")
            .map_err(|e| SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: format!("guest missing 'run(u32,u32) -> u64': {e}"),
            })?;

        let ptr = alloc
            .call(&mut *store, input_bytes.len() as u32)
            .map_err(|e| map_trap(skill_id, e))?;
        let ptr_usize = ptr as usize;
        let mem_slice = memory.data_mut(&mut *store);
        if ptr_usize
            .checked_add(input_bytes.len())
            .map_or(true, |end| end > mem_slice.len())
        {
            return Err(SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: "alloc returned out-of-bounds pointer".into(),
            });
        }
        mem_slice[ptr_usize..ptr_usize + input_bytes.len()].copy_from_slice(input_bytes);

        let packed = run
            .call(&mut *store, (ptr, input_bytes.len() as u32))
            .map_err(|e| map_trap(skill_id, e))?;
        let out_ptr = (packed >> 32) as u32;
        let out_len = (packed & 0xFFFF_FFFF) as u32 as usize;

        if out_len > self.config.max_output_bytes {
            return Err(SkillError::ResourceExhausted {
                skill: skill_id.clone(),
                what: "output_bytes",
            });
        }

        let mem_view = memory.data(&*store);
        let op = out_ptr as usize;
        if op
            .checked_add(out_len)
            .map_or(true, |end| end > mem_view.len())
        {
            return Err(SkillError::ExecutionFailed {
                skill: skill_id.clone(),
                message: "run returned out-of-bounds output pointer".into(),
            });
        }
        let out_bytes = mem_view[op..op + out_len].to_vec();

        let value = match std::str::from_utf8(&out_bytes) {
            Ok(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(v) => v,
                Err(_) => serde_json::Value::String(s.to_string()),
            },
            Err(_) => serde_json::Value::String(format!("<non-utf8:{}:bytes>", out_bytes.len())),
        };

        Ok(SkillOutput {
            value,
            trace: Vec::new(),
            fuel_consumed: 0,
            wall_ms: 0,
        })
    }
}

#[cfg(feature = "skill-forge")]
struct MemoryLimits {
    max_memory_bytes: usize,
}

#[cfg(feature = "skill-forge")]
impl wasmtime::ResourceLimiter for MemoryLimits {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        if desired > self.max_memory_bytes {
            return Ok(false);
        }
        if let Some(max) = maximum {
            if desired > max {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        _desired: usize,
        _maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        Ok(true)
    }
}

#[cfg(feature = "skill-forge")]
fn map_trap(skill_id: &SkillId, err: anyhow::Error) -> SkillError {
    let msg = format!("{err}");
    if msg.contains("fuel") || msg.contains("consumed all fuel") {
        SkillError::ResourceExhausted {
            skill: skill_id.clone(),
            what: "fuel",
        }
    } else if msg.contains("epoch") || msg.contains("interrupt") {
        SkillError::ResourceExhausted {
            skill: skill_id.clone(),
            what: "wall_timeout",
        }
    } else if msg.contains("out of bounds") || msg.contains("memory") {
        SkillError::ResourceExhausted {
            skill: skill_id.clone(),
            what: "memory",
        }
    } else {
        SkillError::ExecutionFailed {
            skill: skill_id.clone(),
            message: msg,
        }
    }
}

// =============================================================================
// Tests — config-only (wasmtime integration tests are in tests/)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_reasonable() {
        let c = WasmRunConfig::default();
        assert!(c.max_fuel > 0);
        assert!(c.max_memory_bytes >= 1024 * 1024);
        assert!(c.wall_timeout.as_secs() > 0);
        assert!(c.max_input_bytes >= 64 * 1024);
        assert!(c.max_output_bytes >= c.max_input_bytes);
    }

    #[test]
    fn config_custom_limits() {
        let c = WasmRunConfig {
            max_fuel: 1000,
            max_memory_bytes: 4096,
            wall_timeout: Duration::from_millis(100),
            max_input_bytes: 512,
            max_output_bytes: 1024,
        };
        assert_eq!(c.max_fuel, 1000);
        assert_eq!(c.max_memory_bytes, 4096);
    }

    #[cfg(feature = "skill-forge")]
    #[test]
    fn runtime_construction_succeeds() {
        let rt = WasmRuntime::new(WasmRunConfig::default());
        assert!(rt.is_ok());
    }

    #[cfg(feature = "skill-forge")]
    #[test]
    fn input_too_large_rejected() {
        let rt = WasmRuntime::new(WasmRunConfig {
            max_input_bytes: 16,
            ..WasmRunConfig::default()
        })
        .expect("rt");
        // Artifact doesn't need to be valid — we reject before loading.
        let artifact = WasmArtifact {
            bytes: vec![0],
            blake3_hex: "x".into(),
            ed25519_sig_hex: "".into(),
            signed_by: "".into(),
            compile_fingerprint: "".into(),
            source_path: None,
        };
        let big_input = vec![0u8; 100];
        let err = rt
            .execute(&SkillId::new("t"), &artifact, &big_input)
            .unwrap_err();
        match err {
            SkillError::BadInput { .. } => {}
            other => panic!("expected BadInput, got {other:?}"),
        }
    }
}
