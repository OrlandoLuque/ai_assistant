// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Host hardware probe (V139).
//!
//! Reports CPU, RAM, GPU and OS information for the recommender (V140)
//! to decide which model variants fit. Probes are best-effort: a
//! missing driver or vendor CLI yields an empty subsection plus a
//! `tracing` warning, never an error from [`detect`].
//!
//! Gated behind `--features hardware-detection`. Sub-features:
//! `hardware-nvml` (NVIDIA, Win/Linux), `hardware-rocm` (AMD Linux),
//! `hardware-metal` (macOS). The base feature only pulls `sysinfo`,
//! which is cross-platform and dependency-light.
//!
//! ```no_run
//! # #[cfg(feature = "hardware-detection")] {
//! use ai_assistant::hardware_info::detect_cached;
//!
//! let hw = detect_cached();
//! println!("{} cores, {} GB RAM, {} GPU(s)",
//!     hw.cpu.logical_cores,
//!     hw.ram.total_bytes / 1_000_000_000,
//!     hw.gpus.len());
//! # }
//! ```

use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

/// Cached snapshot of the host hardware.
///
/// `source` distinguishes a real probe from a config-supplied override
/// (useful for tests and for users on locked-down hosts where probes
/// are blocked). Consumers that care about fidelity should check it
/// before trusting numbers like `vram_free_bytes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HardwareInfo {
    pub source: HardwareSource,
    pub cpu: CpuInfo,
    pub ram: RamInfo,
    pub gpus: Vec<GpuInfo>,
    pub os: OsInfo,
}

/// Provenance of a [`HardwareInfo`] snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HardwareSource {
    /// Result of probing the running host.
    Detected,
    /// Supplied by the user / caller, e.g. via config file.
    Declared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CpuInfo {
    pub vendor: String,
    pub brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub base_freq_mhz: Option<u32>,
    pub features: CpuFeatures,
}

/// Coarse-grained CPU feature flags useful to a model recommender.
/// Not exhaustive — extended in lock-step with backends that gain
/// per-instruction-set optimisations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CpuFeatures {
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
    pub f16c: bool,
    pub neon: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RamInfo {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_bytes: u64,
    /// Currently-free VRAM if the driver exposes it. `None` is not the
    /// same as zero — Metal and some ROCm builds simply do not report.
    pub vram_free_bytes: Option<u64>,
    /// NVIDIA compute capability, e.g. `"8.9"` for sm_89. `None` for
    /// non-NVIDIA GPUs.
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
    /// Backend names the GPU is known to be usable with — opaque
    /// strings rather than a coupled enum so the recommender can
    /// evolve its backend taxonomy without bumping this schema.
    pub backend_support: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OsInfo {
    pub family: String,
    pub name: String,
    pub version: String,
    pub kernel: String,
    pub arch: String,
}

/// Errors a probe can surface. Most failures are logged and folded
/// into empty subsections instead of returning `Err` — this enum
/// exists for the rare cases where there is genuinely nothing to
/// report (e.g. sysinfo init fails entirely).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HardwareError {
    #[error("sysinfo failed to initialise")]
    SysinfoInit,
    #[error("probe timeout: {0}")]
    Timeout(&'static str),
    #[error("io: {0}")]
    Io(String),
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[cfg(feature = "hardware-detection")]
static CACHE: OnceLock<Arc<HardwareInfo>> = OnceLock::new();

/// Probe the host hardware now, bypassing the cache.
///
/// Each subsection (CPU, RAM, GPUs, OS) is gathered independently — a
/// failing GPU probe still returns a valid `HardwareInfo` with the
/// other fields populated. Errors propagate only when *no* useful
/// information could be obtained.
#[cfg(feature = "hardware-detection")]
pub fn detect() -> Result<HardwareInfo, HardwareError> {
    let sys = sysinfo_probe::collect()?;
    let mut gpus = nvml_probe::collect_nvidia();
    gpus.extend(rocm_probe::collect_amd());
    gpus.extend(metal_probe::collect_apple());

    Ok(HardwareInfo {
        source: HardwareSource::Detected,
        cpu: sys.cpu,
        ram: sys.ram,
        gpus,
        os: sys.os,
    })
}

/// Cached variant — first call probes, subsequent calls return the
/// shared `Arc` without re-probing. On probe failure, returns a
/// `Declared` snapshot with empty subsections and `source =
/// Declared` so the caller can tell the difference.
#[cfg(feature = "hardware-detection")]
pub fn detect_cached() -> Arc<HardwareInfo> {
    CACHE
        .get_or_init(|| {
            let info = match detect() {
                Ok(info) => info,
                Err(e) => {
                    log::warn!("hardware probe failed: {e}; returning empty snapshot");
                    HardwareInfo {
                        source: HardwareSource::Declared,
                        cpu: CpuInfo {
                            vendor: String::new(),
                            brand: String::new(),
                            physical_cores: 0,
                            logical_cores: 0,
                            base_freq_mhz: None,
                            features: CpuFeatures::default(),
                        },
                        ram: RamInfo::default(),
                        gpus: Vec::new(),
                        os: OsInfo::default(),
                    }
                }
            };
            Arc::new(info)
        })
        .clone()
}

/// Inject a manually-declared snapshot. Useful for tests and for
/// hosts where probes are intentionally disabled. Returns `false` if
/// the cache is already populated — callers should set this *before*
/// the first `detect_cached` call.
#[cfg(feature = "hardware-detection")]
pub fn set_declared(info: HardwareInfo) -> bool {
    let mut declared = info;
    declared.source = HardwareSource::Declared;
    CACHE.set(Arc::new(declared)).is_ok()
}

// ---------------------------------------------------------------------------
// sysinfo probe — CPU, RAM, OS
// ---------------------------------------------------------------------------

#[cfg(feature = "hardware-detection")]
mod sysinfo_probe {
    use super::*;

    pub(super) struct SystemSnapshot {
        pub cpu: CpuInfo,
        pub ram: RamInfo,
        pub os: OsInfo,
    }

    pub(super) fn collect() -> Result<SystemSnapshot, HardwareError> {
        use sysinfo::System;

        let mut sys = System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();

        let logical_cores = sys.cpus().len();
        let physical_cores = sys.physical_core_count().unwrap_or(logical_cores.max(1));

        let (vendor, brand, base_freq_mhz) = sys
            .cpus()
            .first()
            .map(|c| {
                let freq = c.frequency();
                (
                    c.vendor_id().to_string(),
                    c.brand().to_string(),
                    if freq > 0 { Some(freq as u32) } else { None },
                )
            })
            .unwrap_or_default();

        let cpu = CpuInfo {
            vendor,
            brand,
            physical_cores,
            logical_cores,
            base_freq_mhz,
            features: detect_cpu_features(),
        };

        let ram = RamInfo {
            total_bytes: sys.total_memory(),
            free_bytes: sys.available_memory(),
        };

        let os = OsInfo {
            family: std::env::consts::FAMILY.to_string(),
            name: System::name().unwrap_or_else(|| "unknown".into()),
            version: System::os_version().unwrap_or_else(|| "unknown".into()),
            kernel: System::kernel_version().unwrap_or_else(|| "unknown".into()),
            arch: std::env::consts::ARCH.to_string(),
        };

        Ok(SystemSnapshot { cpu, ram, os })
    }

    fn detect_cpu_features() -> CpuFeatures {
        // `is_x86_feature_detected!` is compile-time gated to x86/x86_64;
        // `is_aarch64_feature_detected!` to aarch64. Anything else
        // returns the default (all false), which is correct — we don't
        // know what to look for on, say, riscv64.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            CpuFeatures {
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                f16c: is_x86_feature_detected!("f16c"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            CpuFeatures {
                avx: false,
                avx2: false,
                avx512: false,
                fma: false,
                f16c: false,
                // NEON is mandatory on aarch64 but we still check
                // through std::arch detection for symmetry.
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuFeatures::default()
        }
    }
}

// ---------------------------------------------------------------------------
// NVIDIA probe — nvml-wrapper, gated on `hardware-nvml`
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware-detection", feature = "hardware-nvml"))]
mod nvml_probe {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Time budget for the entire NVML probe. NVML can hang if the
    /// driver is wedged; we cap the call so a broken driver cannot
    /// take the whole binary down with it.
    const NVML_TIMEOUT: Duration = Duration::from_secs(3);

    pub(super) fn collect_nvidia() -> Vec<GpuInfo> {
        let (tx, rx) = mpsc::channel::<Vec<GpuInfo>>();
        std::thread::spawn(move || {
            let result = match probe_inner() {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("NVML probe failed: {e}");
                    Vec::new()
                }
            };
            // Receiver may already be dropped after timeout — don't panic.
            let _ = tx.send(result);
        });

        match rx.recv_timeout(NVML_TIMEOUT) {
            Ok(v) => v,
            Err(_) => {
                log::warn!("NVML probe timed out after {:?}", NVML_TIMEOUT);
                Vec::new()
            }
        }
    }

    fn probe_inner() -> Result<Vec<GpuInfo>, String> {
        let nvml = nvml_wrapper::Nvml::init().map_err(|e| e.to_string())?;
        let count = nvml.device_count().map_err(|e| e.to_string())?;
        let driver_version = nvml.sys_driver_version().ok();
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = match nvml.device_by_index(i) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("NVML device {i} unreadable: {e}");
                    continue;
                }
            };
            let name = dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}"));
            let mem = dev.memory_info().ok();
            let (vram_bytes, vram_free) = match mem {
                Some(m) => (m.total, Some(m.free)),
                None => (0, None),
            };
            let compute_capability = dev
                .cuda_compute_capability()
                .ok()
                .map(|c| format!("{}.{}", c.major, c.minor));
            out.push(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name,
                vram_bytes,
                vram_free_bytes: vram_free,
                compute_capability,
                driver_version: driver_version.clone(),
                backend_support: nvidia_backends(),
            });
        }
        Ok(out)
    }

    fn nvidia_backends() -> Vec<String> {
        vec![
            "cuda".into(),
            "llama_cpp_mainline".into(),
            "vllm".into(),
            "ollama".into(),
            "lm_studio".into(),
        ]
    }
}

#[cfg(all(feature = "hardware-detection", not(feature = "hardware-nvml")))]
mod nvml_probe {
    use super::GpuInfo;
    pub(super) fn collect_nvidia() -> Vec<GpuInfo> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// ROCm probe — shells out to `rocm-smi --showmeminfo vram --json`
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware-detection", feature = "hardware-rocm"))]
mod rocm_probe {
    use super::*;

    pub(super) fn collect_amd() -> Vec<GpuInfo> {
        let output = match std::process::Command::new("rocm-smi")
            .args(["--showproductname", "--showmeminfo", "vram", "--json"])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                log::warn!(
                    "rocm-smi exited with status {}; assuming no AMD GPU",
                    o.status
                );
                return Vec::new();
            }
            Err(e) => {
                log::debug!("rocm-smi unavailable ({e}); skipping AMD probe");
                return Vec::new();
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_rocm_smi(&stdout)
    }

    fn parse_rocm_smi(json: &str) -> Vec<GpuInfo> {
        let v: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("rocm-smi JSON parse failed: {e}");
                return Vec::new();
            }
        };
        let obj = match v.as_object() {
            Some(o) => o,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        for (key, fields) in obj {
            // rocm-smi keys look like "card0", "card1", …; ignore the
            // top-level "system" / "ROCk module" keys.
            if !key.starts_with("card") {
                continue;
            }
            let name = fields
                .get("Card series")
                .or_else(|| fields.get("Card model"))
                .and_then(|s| s.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("AMD GPU ({key})"));
            let vram_bytes = fields
                .get("VRAM Total Memory (B)")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let vram_free_bytes = fields
                .get("VRAM Total Used Memory (B)")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<u64>().ok())
                .map(|used| vram_bytes.saturating_sub(used));
            out.push(GpuInfo {
                vendor: GpuVendor::Amd,
                name,
                vram_bytes,
                vram_free_bytes,
                compute_capability: None,
                driver_version: None,
                backend_support: vec!["rocm".into(), "llama_cpp_mainline".into(), "vllm".into()],
            });
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_minimal_rocm_smi_json() {
            let json = r#"{
                "system": {"Driver version": "6.0.0"},
                "card0": {
                    "Card series": "AMD Radeon RX 7900 XTX",
                    "VRAM Total Memory (B)": "25753026560",
                    "VRAM Total Used Memory (B)": "1073741824"
                }
            }"#;
            let gpus = parse_rocm_smi(json);
            assert_eq!(gpus.len(), 1);
            assert_eq!(gpus[0].vendor, GpuVendor::Amd);
            assert_eq!(gpus[0].vram_bytes, 25_753_026_560);
            assert_eq!(
                gpus[0].vram_free_bytes,
                Some(25_753_026_560 - 1_073_741_824)
            );
        }

        #[test]
        fn ignores_non_card_keys() {
            let json = r#"{"system":{"x":"y"},"ROCk module":{"a":"b"}}"#;
            assert!(parse_rocm_smi(json).is_empty());
        }
    }
}

#[cfg(all(feature = "hardware-detection", not(feature = "hardware-rocm")))]
mod rocm_probe {
    use super::GpuInfo;
    pub(super) fn collect_amd() -> Vec<GpuInfo> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Apple Metal probe — shells out to `system_profiler SPDisplaysDataType -json`
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware-detection", feature = "hardware-metal"))]
mod metal_probe {
    use super::*;

    pub(super) fn collect_apple() -> Vec<GpuInfo> {
        if !cfg!(target_os = "macos") {
            return Vec::new();
        }
        let output = match std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                log::warn!(
                    "system_profiler exited with status {}; assuming no Apple GPU",
                    o.status
                );
                return Vec::new();
            }
            Err(e) => {
                log::debug!("system_profiler unavailable ({e}); skipping Metal probe");
                return Vec::new();
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_system_profiler(&stdout)
    }

    fn parse_system_profiler(json: &str) -> Vec<GpuInfo> {
        let v: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("system_profiler JSON parse failed: {e}");
                return Vec::new();
            }
        };
        let arr = match v.get("SPDisplaysDataType").and_then(|x| x.as_array()) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        for entry in arr {
            let name = entry
                .get("sppci_model")
                .and_then(|s| s.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| "Apple GPU".into());
            // VRAM on Apple Silicon is shared with system RAM and not
            // reported by system_profiler; leave as 0 with note that
            // the recommender should fall back to RAM-based capacity.
            let vram_bytes = entry
                .get("spdisplays_vram")
                .and_then(|s| s.as_str())
                .and_then(parse_vram_string)
                .unwrap_or(0);
            out.push(GpuInfo {
                vendor: GpuVendor::Apple,
                name,
                vram_bytes,
                vram_free_bytes: None,
                compute_capability: None,
                driver_version: None,
                backend_support: vec!["metal".into(), "llama_cpp_mainline".into(), "mlx".into()],
            });
        }
        out
    }

    fn parse_vram_string(s: &str) -> Option<u64> {
        // Strings like "8 GB" or "1536 MB".
        let s = s.trim();
        let (num, unit) = s.split_once(' ')?;
        let n: u64 = num.parse().ok()?;
        let mult = match unit.to_ascii_uppercase().as_str() {
            "GB" => 1_000_000_000,
            "MB" => 1_000_000,
            "KB" => 1_000,
            _ => return None,
        };
        Some(n * mult)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_vram_string() {
            assert_eq!(parse_vram_string("8 GB"), Some(8_000_000_000));
            assert_eq!(parse_vram_string("1536 MB"), Some(1_536_000_000));
            assert_eq!(parse_vram_string("nope"), None);
        }
    }
}

#[cfg(all(feature = "hardware-detection", not(feature = "hardware-metal")))]
mod metal_probe {
    use super::GpuInfo;
    pub(super) fn collect_apple() -> Vec<GpuInfo> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Reporting helper — pretty-printed table for the CLI
// ---------------------------------------------------------------------------

#[cfg(feature = "hardware-detection")]
impl HardwareInfo {
    /// Render a human-readable summary. Used by the `ai_setup hardware`
    /// subcommand and by `tracing` log output. Stable: callers can
    /// parse the format if they need to, though the recommender uses
    /// the structured fields directly.
    pub fn pretty_summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "Source: {:?}", self.source);
        let _ = writeln!(
            s,
            "CPU:    {} ({}C/{}T){}",
            if self.cpu.brand.is_empty() {
                "<unknown>"
            } else {
                &self.cpu.brand
            },
            self.cpu.physical_cores,
            self.cpu.logical_cores,
            cpu_feature_summary(&self.cpu.features),
        );
        let _ = writeln!(
            s,
            "RAM:    {} total, {} free",
            format_bytes(self.ram.total_bytes),
            format_bytes(self.ram.free_bytes),
        );
        if self.gpus.is_empty() {
            let _ = writeln!(s, "GPU:    (none detected)");
        } else {
            for (i, g) in self.gpus.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "GPU {i}:  {:?} {} — {} VRAM{}",
                    g.vendor,
                    g.name,
                    format_bytes(g.vram_bytes),
                    match g.vram_free_bytes {
                        Some(free) => format!(" ({} free)", format_bytes(free)),
                        None => String::new(),
                    },
                );
                if let Some(cc) = &g.compute_capability {
                    let _ = writeln!(s, "        Compute capability {cc}");
                }
                if let Some(dv) = &g.driver_version {
                    let _ = writeln!(s, "        Driver {dv}");
                }
            }
        }
        let _ = writeln!(
            s,
            "OS:     {} {} ({} {})",
            self.os.name, self.os.version, self.os.family, self.os.arch
        );
        s
    }
}

#[cfg(feature = "hardware-detection")]
fn cpu_feature_summary(f: &CpuFeatures) -> String {
    let mut flags: Vec<&str> = Vec::new();
    if f.avx512 {
        flags.push("avx512");
    } else if f.avx2 {
        flags.push("avx2");
    } else if f.avx {
        flags.push("avx");
    }
    if f.fma {
        flags.push("fma");
    }
    if f.neon {
        flags.push("neon");
    }
    if flags.is_empty() {
        String::new()
    } else {
        format!(", {}", flags.join(" "))
    }
}

#[cfg(feature = "hardware-detection")]
fn format_bytes(b: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;
    if b >= GB {
        format!("{:.1} GB", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{} MB", b / MB)
    } else {
        format!("{b} B")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "hardware-detection"))]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_populated_snapshot() {
        let info = detect().expect("sysinfo should always succeed on supported targets");
        assert_eq!(info.source, HardwareSource::Detected);
        assert!(info.cpu.logical_cores >= 1);
        assert!(info.ram.total_bytes > 0);
        assert!(!info.os.family.is_empty());
    }

    #[test]
    fn format_bytes_thresholds() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1_500_000), "1 MB");
        assert_eq!(format_bytes(2_500_000_000), "2.5 GB");
    }

    #[test]
    fn pretty_summary_renders_minimal_snapshot() {
        let info = HardwareInfo {
            source: HardwareSource::Declared,
            cpu: CpuInfo {
                vendor: "GenuineIntel".into(),
                brand: "i9-12900K".into(),
                physical_cores: 16,
                logical_cores: 24,
                base_freq_mhz: Some(3200),
                features: CpuFeatures {
                    avx2: true,
                    ..Default::default()
                },
            },
            ram: RamInfo {
                total_bytes: 64 * 1_000_000_000,
                free_bytes: 32 * 1_000_000_000,
            },
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "RTX 4090".into(),
                vram_bytes: 24 * 1_000_000_000,
                vram_free_bytes: Some(22 * 1_000_000_000),
                compute_capability: Some("8.9".into()),
                driver_version: Some("545.84".into()),
                backend_support: vec!["cuda".into()],
            }],
            os: OsInfo {
                family: "windows".into(),
                name: "Windows".into(),
                version: "10".into(),
                kernel: "10.0.19045".into(),
                arch: "x86_64".into(),
            },
        };
        let s = info.pretty_summary();
        assert!(s.contains("i9-12900K"));
        assert!(s.contains("RTX 4090"));
        assert!(s.contains("24.0 GB"));
        assert!(s.contains("Compute capability 8.9"));
        assert!(s.contains("avx2"));
    }

    #[test]
    fn cpu_features_default_all_false() {
        let f = CpuFeatures::default();
        assert!(!f.avx && !f.avx2 && !f.avx512 && !f.fma && !f.f16c && !f.neon);
    }

    #[test]
    fn gpu_vendor_serde_roundtrip() {
        let v = GpuVendor::Other("RISC-V Vector".into());
        let s = serde_json::to_string(&v).unwrap();
        let back: GpuVendor = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn hardware_info_serde_roundtrip() {
        let info = HardwareInfo {
            source: HardwareSource::Declared,
            cpu: CpuInfo {
                vendor: "x".into(),
                brand: "y".into(),
                physical_cores: 1,
                logical_cores: 1,
                base_freq_mhz: None,
                features: CpuFeatures::default(),
            },
            ram: RamInfo::default(),
            gpus: Vec::new(),
            os: OsInfo::default(),
        };
        let s = serde_json::to_string(&info).unwrap();
        let back: HardwareInfo = serde_json::from_str(&s).unwrap();
        assert_eq!(info, back);
    }
}
