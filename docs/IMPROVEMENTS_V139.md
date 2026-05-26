# V139 — Hardware detection (0.2.86)

**Status**: SHIPPED 2026-05-26 (0.2.86) — `hardware_info` module +
4 feature flags + `ai_setup hardware` CLI.
**Scope**: módulo nuevo `src/hardware_info.rs` que detecta CPU, RAM,
GPUs y aceleradores del host. Foundation para V140 (Butler recommender).
**No-goals**: usar los datos (eso es V140), tuning runtime (out of project).

## Por qué

El recommender de V140 necesita saber qué VRAM tienes para decidir si
el sweet spot Llama-3.1-70B-Q4_K_M (≈40 GB) te entra o si tiene que
caer a la variante 8B. Hoy no hay módulo que reporte hardware.

## API propuesta

```rust
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub ram: RamInfo,
    pub gpus: Vec<GpuInfo>,
    pub os: OsInfo,
}

pub struct CpuInfo {
    pub vendor: String,
    pub brand: String,                 // "AMD Ryzen 9 7950X3D"
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub base_freq_mhz: Option<u32>,
    pub features: CpuFeatures,         // avx2, avx512, neon, …
}

pub struct GpuInfo {
    pub vendor: GpuVendor,             // Nvidia, Amd, Intel, Apple
    pub name: String,                  // "NVIDIA GeForce RTX 4090"
    pub vram_bytes: u64,
    pub vram_free_bytes: Option<u64>,  // current free; None if unsupported
    pub compute_capability: Option<String>, // "8.9" para sm_89
    pub driver_version: Option<String>,
    pub backend_support: Vec<Backend>, // Cuda, Rocm, Metal, Vulkan, …
}

pub fn detect() -> Result<HardwareInfo, HardwareError>;
pub fn detect_cached() -> Arc<HardwareInfo>;   // memoiza tras primera llamada
```

## Dependencias por backend

| Backend | Crate | Plataforma | Sub-feature |
|---|---|---|---|
| Base (CPU, RAM, OS) | `sysinfo` | cross | `hardware-detection` (siempre) |
| NVIDIA GPU | `nvml-wrapper` | Win/Linux | `hardware-nvml` |
| AMD GPU | parsing `rocm-smi --showmeminfo --json` | Linux | `hardware-rocm` |
| Apple Metal | parsing `system_profiler SPDisplaysDataType` | macOS | `hardware-metal` |
| Intel iGPU/Arc | `intel-gpu-tools` (Linux) o WMI (Win) | mixto | `hardware-intel` (opcional, fase 2) |

`hardware-detection` por defecto activa todo lo gratis (sysinfo +
nvml si plataforma compatible). Los demás opt-in.

## Estrategia de detección

- **Fallback graceful**: si `nvml-wrapper` falla porque no hay driver,
  no es error — `gpus` queda vacío con un `warning` log y `detect()`
  sigue.
- **Caché**: `detect()` es lento (~500 ms con NVML). `detect_cached()`
  memoiza con `OnceLock<Arc<HardwareInfo>>`. Refresh manual disponible.
- **VRAM free**: NVML reporta; ROCm sometimes; Metal no de forma
  fiable. `None` cuando no se puede.
- **Sin runtime instrumentation**: no medimos uso en vivo, sólo
  capacidades. La medición continua es scope de `gpu_sharing` o
  futuro `gpu_monitor`.

## Tests

- Mock backends — trait `HardwareProbe` con impl `SysinfoProbe`,
  `NvmlProbe`, etc. Tests inyectan probes que devuelven JSON canned.
- Cross-platform: tests `#[cfg(target_os = "...")]` para los
  específicos.
- Graceful fallback: probe que panics → captured + warning, no
  propaga.

## CLI helper

`ai_setup` (binary ya existente, V70) añade subcomando `hardware` que
imprime `HardwareInfo` como tabla legible. Útil para diagnóstico:

```
$ ai_setup hardware
CPU:  AMD Ryzen 9 7950X3D (16C/32T, avx512)
RAM:  64 GB total, 38 GB free
GPU 0: NVIDIA GeForce RTX 4090 — 24 GB VRAM (22 GB free)
       Compute capability 8.9, CUDA 12.4, Vulkan supported
```

## Wiring posterior

- V140 (Butler recommender) — input principal del recommender.
- `ai_serve` exponer `/hardware` endpoint opcional (oculto tras
  feature flag).
- MCP tool nuevo `hardware_info` (V140 o V143).
- `ai_setup_gui` (V70) — tab "Hardware" mostrando la info.

## Riesgos / vectores de ataque

- **Driver crash** — `nvml-wrapper` puede colgar el proceso si el
  driver está roto. `tokio::spawn_blocking` + timeout para aislar.
- **Information leak** — hardware info expuesta vía HTTP filtra
  fingerprint del host. V143 audit determina si necesita
  autenticación.
- **Spoofing** — config puede sobrescribir `HardwareInfo` para tests;
  hay que distinguir "detectado" de "declarado" en la struct.

## Iteraciones del plan

- **iter 1**: sólo `sysinfo` (sin GPU). Rechazado: VRAM es la
  constraint principal, sin GPU el recommender es ciego.
- **iter 2**: añadidos NVML + ROCm + Metal con sub-features.
- **iter 3 (actual)**: graceful fallback obligatorio, caché OnceLock,
  CLI helper en ai_setup.

## Out of scope para V139

- Tuning runtime (cambiar `n_gpu_layers` en vivo según VRAM libre).
- Monitorización continua (eso es gpu_monitor, futuro módulo).
- Recomendaciones (V140).

## Verification (shipped 2026-05-26)

- `cargo build --lib` (default = full incl. `hardware-detection` +
  `hardware-nvml`): clean.
- `cargo build --lib --features "hardware-detection hardware-nvml
  hardware-rocm hardware-metal"`: clean.
- `cargo clippy --lib -- -D warnings`: clean.
- `cargo clippy --bin ai_setup --features full -- -D warnings`: clean.
- `cargo test --lib`: **6268 passed** (6262 → 6268, +6 in
  `hardware_info::tests::`). Pre-existing flaky
  `opentelemetry_integration::test_otlp_exporter_flush_with_mock`
  passed on isolated re-run — unrelated to V139.

## Decisiones de implementación (no en el plan original)

- **`backend_support` as `Vec<String>` (not `Vec<models_dev::Backend>`)**:
  decoupling the hardware module from the catalog taxonomy means V137
  can evolve its `Backend` enum (e.g. adding `Mlx`, `Tgi`) without
  forcing a hardware-probe rebuild. The recommender in V140 owns the
  mapping.
- **NVML isolation via `std::thread::spawn` + `mpsc::recv_timeout`**:
  the plan suggested `tokio::spawn_blocking`. Using plain `std::thread`
  keeps `hardware-detection` from pulling in `async-runtime` — a
  hard requirement so that consumers who only want hardware data
  don't have to take tokio.
- **No `detect_async` variant**: the synchronous probe is fast enough
  for the cached pattern (one-time ~500 ms hit). Wrapping it for
  callers who happen to be in an async context belongs in those
  callers, not here.
- **`set_declared` returns `bool`, not `Result`**: the only failure
  mode is "cache already populated", which is rarely actionable —
  callers either set early or they don't. The previous `Result<(),
  HardwareInfo>` returned a 248-byte error variant that clippy
  flagged.
- **Sysinfo pinned to 0.32**: 0.39 requires rustc 1.95; the project
  targets 1.90. Re-evaluate on next toolchain bump.
- **CPU features detected via `is_x86_feature_detected!` /
  `is_aarch64_feature_detected!`**: compile-time `cfg(target_arch)`
  branches plus runtime detection. RISC-V and other arches return
  `CpuFeatures::default()` — honest "we don't know" rather than a
  panic.
- **Apple Silicon VRAM**: `system_profiler` does not expose VRAM on
  unified-memory Macs. We record `0` with the documented caveat that
  the recommender should fall back to RAM-based capacity.
- **`HardwareInfo::pretty_summary()` is part of the public API**:
  small string formatter, but elevated to a method so callers (the
  CLI here, future GUI tabs) can call it directly without
  duplicating the format.
