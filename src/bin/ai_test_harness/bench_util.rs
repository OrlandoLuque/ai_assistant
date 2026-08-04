// Shared config for the live-model benchmark categories (real_e2e, code_gen_bench).
//
// Provider / model / endpoint come from the environment so the SAME task set can
// be pointed at Ollama, llama.cpp, or any local OpenAI-compatible server — which
// is the whole point of a benchmark: compare backends and models fairly.
//
//   AI_BENCH_PROVIDER  (default "ollama")
//        ollama | llamacpp | lmstudio | localai | vllm | textgenwebui | koboldcpp
//   AI_BENCH_MODEL     (default "llama3.2:3b")
//   AI_BENCH_URL       (optional) overrides the provider's default endpoint
//   AI_BENCH_TEMP      (default 0.5)  sampling temperature
//   AI_BENCH_SEED      (default 42)   sampling seed; "none" to randomise
//   AI_BENCH_REPEATS   (default 3)    runs per task; a single run is not a measurement
//   AI_BENCH_NUM_CTX   (optional)     context window; shrink it to fit a big model fully on GPU
//
// e.g. run against llama.cpp's server:
//   $env:AI_BENCH_PROVIDER="llamacpp"; $env:AI_BENCH_MODEL="qwen2.5-coder-7b"
//   $env:AI_BENCH_URL="http://127.0.0.1:8080"

use std::time::Duration;

use ai_assistant::{AiAssistant, AiProvider};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

pub(crate) fn bench_provider() -> AiProvider {
    match env_or("AI_BENCH_PROVIDER", "ollama")
        .to_lowercase()
        .as_str()
    {
        "llamacpp" | "llama.cpp" | "llama_cpp" => AiProvider::LlamaCpp,
        "lmstudio" | "lm-studio" => AiProvider::LMStudio,
        "localai" => AiProvider::LocalAI,
        "vllm" => AiProvider::VLLM,
        "textgenwebui" | "text-gen-webui" | "oobabooga" => AiProvider::TextGenWebUI,
        "koboldcpp" | "kobold" => AiProvider::KoboldCpp,
        _ => AiProvider::Ollama,
    }
}

pub(crate) fn bench_model() -> String {
    env_or("AI_BENCH_MODEL", "llama3.2:3b")
}

/// Sampling temperature (`AI_BENCH_TEMP`, default 0.5).
///
/// This used to default to 0.0, on the usual assumption that greedy decoding is
/// how you get reproducible runs. That was wrong twice over:
///
/// * Reproducibility comes from [`bench_seed`], not from the temperature. With a
///   fixed seed the output is byte-identical at any temperature (measured).
/// * Near-greedy sampling *aborts* the llama.cpp runner on some inputs
///   (`Assertion failed: found, llama-sampling.cpp`, Ollama 0.21.2). The request
///   dies with the process, the client reports a connection failure, and the
///   benchmark records it as a model failure. Measured on one task: crash at
///   0.0/0.1/0.2/0.3, clean answer in seconds at 0.5.
///
/// So the default sits above the crashing band, and determinism is bought with
/// the seed instead. Raise it deliberately when measuring best-of-N sampling,
/// where identical samples would defeat the point.
pub(crate) fn bench_temperature() -> f32 {
    std::env::var("AI_BENCH_TEMP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| (0.0..=2.0).contains(t))
        .unwrap_or(0.5)
}

/// Base sampling seed (`AI_BENCH_SEED`, default 42). See [`effective_seed`] for the
/// value actually sent, which varies per repeat.
///
/// Set `AI_BENCH_SEED=none` (or `off`) to let the backend randomise per request,
/// which is what you want when measuring best-of-N or sampling variance.
pub(crate) fn bench_seed() -> Option<u64> {
    match std::env::var("AI_BENCH_SEED") {
        Ok(v) if matches!(v.trim().to_lowercase().as_str(), "none" | "off" | "") => None,
        Ok(v) => v.trim().parse::<u64>().ok().or(Some(42)),
        Err(_) => Some(42),
    }
}

// ─── The seed must VARY across repeats, or the repeats are not samples ────────
//
// Measured 2026-08-04 on `ledger: an infallible API becomes fallible`, qwen2.5-coder:14b:
//
//   seed 42   -> 0/3        seed 7 -> 1/3        seed 1234 -> 3/3
//
// Pooled that is p ~ 0.44, and the entry that recorded "0/3, never solves it, a capability
// boundary" was reading one unlucky seed as a property of the model. Two separate mistakes
// met there:
//
//   * 0 of 3 does not mean p = 0. At p = 0.44 you draw 0/3 about 17 % of the time, and the
//     97.5 % upper bound on p after 0/3 is ~0.6 (rule of three). Three samples cannot
//     distinguish "never" from "sometimes".
//   * repeats at a FIXED seed do not sample the seed. Interleaving decorrelates KV-cache
//     state, which is real variance, but the seed dimension stayed frozen — and it turned
//     out to dominate: within seed 1234 the task is 3/3, within seed 42 it is 0/3.
//
// So the effective seed is `base + repeat index`. The sweep stays exactly reproducible
// (the sequence of seeds is deterministic given the base) while the repeats finally vary
// the thing that mattered most.
static REPEAT_INDEX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by the repeat loop before each pass; see [`effective_seed`].
pub(crate) fn set_repeat_index(i: u64) {
    REPEAT_INDEX.store(i, std::sync::atomic::Ordering::Relaxed);
}

/// The seed actually sent to the backend: the base seed offset by the repeat index, so
/// three repeats are three different seeds rather than the same one three times.
///
/// `None` (randomise) stays `None` — there is nothing to offset.
pub(crate) fn effective_seed() -> Option<u64> {
    let i = REPEAT_INDEX.load(std::sync::atomic::Ordering::Relaxed);
    bench_seed().map(|s| s.wrapping_add(i))
}

/// How many times to repeat each task (`AI_BENCH_REPEATS`, default 3).
///
/// A pinned seed is not enough to make a live-model run reproducible: llama.cpp
/// is not bitwise deterministic once server-side KV-cache reuse and batch
/// splitting vary, so a knife-edge pass/fail verdict flips between runs
/// (measured: 2 of 8 verdicts, with the total unchanged). One sample of a
/// stochastic process is not a measurement.
///
/// Repeating turns each task into a **pass rate**, which is both honest and more
/// informative than a boolean: "solves it every time" and "solves it sometimes"
/// stop looking identical, and the flapping becomes visible instead of silently
/// moving the total. Set 1 for a quick smoke ejecución.
pub(crate) fn bench_repeats() -> usize {
    std::env::var("AI_BENCH_REPEATS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, 10)
}

/// Explicit context window (`AI_BENCH_NUM_CTX`), else the library's auto-sizing.
///
/// Exists because the KV cache, not the weights, is what decides whether a model
/// fits on the card: Ollama reserves it for **four** parallel sequences, so a 9 GB
/// model at the default 8192 window asks for ~18 GB and silently loads part of
/// itself onto CPU. A CPU-offloaded model does not merely run slower — it runs
/// slow enough to hit request timeouts, which the benchmark then records as the
/// model failing the task. Shrinking the window is what buys a full-GPU load for
/// a bigger model; these tasks have prompts of ~2k tokens, so a smaller window
/// costs nothing here. Always record the value used alongside the result.
pub(crate) fn bench_num_ctx() -> Option<usize> {
    std::env::var("AI_BENCH_NUM_CTX")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// The endpoint URL the assistant will hit (explicit override, else the
/// provider default). Used both to configure the assistant and to probe
/// reachability.
fn bench_endpoint(provider: &AiProvider) -> String {
    if let Ok(u) = std::env::var("AI_BENCH_URL") {
        if !u.trim().is_empty() {
            return u;
        }
    }
    match provider {
        AiProvider::LlamaCpp => "http://127.0.0.1:8080".to_string(),
        AiProvider::LMStudio => "http://127.0.0.1:1234".to_string(),
        AiProvider::VLLM | AiProvider::LocalAI => "http://127.0.0.1:8000".to_string(),
        AiProvider::TextGenWebUI => "http://127.0.0.1:5000".to_string(),
        AiProvider::KoboldCpp => "http://127.0.0.1:5001".to_string(),
        _ => "http://127.0.0.1:11434".to_string(),
    }
}

/// Human-readable backend label for report headers.
///
/// Includes the sampling settings, because a result recorded in the lab notebook
/// without them cannot be reproduced or compared against a later ejecución.
pub(crate) fn bench_label() -> String {
    // The base seed AND the span the repeats will walk, because "seed=42" alone would
    // now be a half-truth: run 2 uses 43, run 3 uses 44.
    let seed = match bench_seed() {
        Some(s) if bench_repeats() > 1 => format!("{}..{}", s, s + bench_repeats() as u64 - 1),
        Some(s) => s.to_string(),
        None => "random".to_string(),
    };
    let ctx = match bench_num_ctx() {
        Some(n) => format!(" num_ctx={n}"),
        None => String::new(),
    };
    format!(
        "{}/{} temp={} seed={}{}",
        bench_provider().display_name(),
        bench_model(),
        bench_temperature(),
        seed,
        ctx
    )
}

/// A fresh assistant configured for the selected backend, seeded from
/// [`effective_seed`] so that each repeat is a different draw rather than the same
/// one over again.
pub(crate) fn bench_assistant() -> AiAssistant {
    let provider = bench_provider();
    let endpoint = bench_endpoint(&provider);
    let mut a = AiAssistant::new();
    a.config.selected_model = bench_model();
    a.config.temperature = bench_temperature();
    a.config.seed = effective_seed();
    a.config.ollama_num_ctx = bench_num_ctx();
    // Point the right URL field at the endpoint. Only an explicit AI_BENCH_URL
    // (or a non-Ollama provider) changes anything from the defaults.
    match &provider {
        AiProvider::Ollama => a.config.ollama_url = endpoint,
        // Every other supported local backend is OpenAI-compatible and, in this
        // codebase, routed via the llama.cpp URL field.
        _ => a.config.llamacpp_url = endpoint,
    }
    a.config.provider = provider;
    a
}

/// Ask Ollama which models are resident and how much of each sits in VRAM.
///
/// Returns `Some((model, gpu_percent))` for the first loaded model. Ollama fixes
/// the CPU/GPU layer split when a model LOADS, so a model loaded while VRAM was
/// busy stays partly on CPU even after memory frees up (`ollama stop <model>`
/// forces a reload). A partially offloaded model does not merely run slower — it
/// runs slow enough to hit request timeouts, which silently turns into "the model
/// failed the task". Three separate experiments in this benchmark were invalidated
/// that way before this check existed.
fn ollama_gpu_share() -> Option<(String, u32)> {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};

    let ep = bench_endpoint(&bench_provider());
    let hostport = ep
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let hostport = hostport.split('/').next().unwrap_or(hostport);
    let addr = hostport.to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    write!(
        stream,
        "GET /api/ps HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    )
    .ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;

    let json: serde_json::Value = serde_json::from_str(body.split("\r\n\r\n").nth(1)?).ok()?;
    let m = json.get("models")?.as_array()?.first()?;
    let name = m.get("name")?.as_str()?.to_string();
    let total = m.get("size")?.as_u64()?;
    let vram = m.get("size_vram")?.as_u64()?;
    if total == 0 {
        return None;
    }
    Some((name, ((vram as f64 / total as f64) * 100.0).round() as u32))
}

/// Print a loud warning when the resident model is not (almost) fully on GPU.
/// Call this at the start of any live-model category: a CPU-offloaded run produces
/// numbers that look like model failures and are not comparable with anything.
pub(crate) fn warn_if_cpu_offloaded() {
    if bench_provider() != AiProvider::Ollama {
        return; // the /api/ps probe is Ollama-specific
    }
    if let Some((model, gpu_pct)) = ollama_gpu_share() {
        if gpu_pct < 95 {
            println!(
                "  {} {} is only {}% on GPU — results will NOT be comparable.\n     \
                 Free VRAM (nvidia-smi shows who holds it), then `ollama stop {}` to force a \
                 reload: the CPU/GPU split is fixed at load time.",
                crate::yellow("WARNING"),
                model,
                gpu_pct,
                model
            );
        }
    }
}

// ─── Did the backend die MID-SWEEP? ──────────────────────────────────────────
//
// Skipping a category when there is no backend is deliberate: it is what lets the
// battery run on a machine with no GPU, and in CI. But the two states below are not
// the same thing, and until V274 the harness reported them identically:
//
//   * never reachable  → nothing was measurable here; skipping is the right answer.
//   * reachable, then not → the sweep is INVALID. Whatever came after the daemon died
//     was not measured, and every remaining category prints SKIP.
//
// Measured on 2026-08-04: Ollama degraded through a five-category sweep (5 of 30 runs
// ending in BACKEND CRASH) and then the process died. The next category printed
// `ALL 0 TESTS PASSED [1 skipped]` and the run exited 0. Every piece was individually
// correct and the summary was a lie.
static BACKEND_SEEN_UP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static BACKEND_DIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True when the backend answered earlier in this run and has since stopped answering.
pub(crate) fn backend_died_mid_sweep() -> bool {
    BACKEND_DIED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the configured backend's endpoint accepts a TCP connection (so the
/// category can skip gracefully when the server isn't running, e.g. in CI).
///
/// Also records the up→down transition; see [`backend_died_mid_sweep`].
pub(crate) fn backend_reachable() -> bool {
    record_probe(probe_backend())
}

/// The transition rule, separated from the socket so it can be tested.
fn record_probe(up: bool) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if up {
        BACKEND_SEEN_UP.store(true, Relaxed);
    } else if BACKEND_SEEN_UP.load(Relaxed) {
        BACKEND_DIED.store(true, Relaxed);
    }
    up
}

fn probe_backend() -> bool {
    use std::net::ToSocketAddrs;
    let ep = bench_endpoint(&bench_provider());
    let hostport = ep
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let hostport = hostport.split('/').next().unwrap_or(hostport);
    let addr = if hostport.contains(':') {
        hostport.to_string()
    } else {
        format!("{hostport}:80")
    };
    addr.to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next())
        .and_then(|a| std::net::TcpStream::connect_timeout(&a, Duration::from_secs(2)).ok())
        .is_some()
}

#[cfg(test)]
mod backend_liveness_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    fn reset() {
        BACKEND_SEEN_UP.store(false, Relaxed);
        BACKEND_DIED.store(false, Relaxed);
    }

    /// One test, not three: the state is global, and Rust runs tests in parallel, so
    /// separate cases would race each other through the same two atomics.
    #[test]
    fn only_an_up_then_down_transition_invalidates_the_sweep() {
        // Never reachable: legitimate skip (no GPU, CI). Not an invalid sweep.
        reset();
        assert!(!record_probe(false));
        assert!(!record_probe(false));
        assert!(
            !backend_died_mid_sweep(),
            "a backend that was never up cannot have died"
        );

        // Up throughout: nothing to report.
        reset();
        record_probe(true);
        record_probe(true);
        assert!(!backend_died_mid_sweep());

        // Up, then gone: everything measured after this point is missing, and the run
        // must say so rather than print ALL 0 TESTS PASSED.
        reset();
        record_probe(true);
        record_probe(false);
        assert!(backend_died_mid_sweep());

        // And it stays flagged even if the daemon comes back: the categories that
        // skipped in between were still not measured.
        record_probe(true);
        assert!(
            backend_died_mid_sweep(),
            "a recovery does not retroactively measure the skipped categories"
        );
    }
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    /// One test: the repeat index is global state, and Rust runs tests in parallel.
    #[test]
    fn the_effective_seed_walks_with_the_repeat_index() {
        std::env::set_var("AI_BENCH_SEED", "42");
        set_repeat_index(0);
        assert_eq!(effective_seed(), Some(42));
        set_repeat_index(1);
        assert_eq!(effective_seed(), Some(43));
        set_repeat_index(2);
        assert_eq!(
            effective_seed(),
            Some(44),
            "three repeats must be three seeds, not one seed three times"
        );

        // Deterministic: the same pass of the same sweep always draws the same seed,
        // so the run stays reproducible even though the repeats now vary.
        set_repeat_index(1);
        assert_eq!(effective_seed(), Some(43));

        // Randomised stays randomised; there is nothing to offset.
        std::env::set_var("AI_BENCH_SEED", "none");
        set_repeat_index(2);
        assert_eq!(effective_seed(), None);

        std::env::remove_var("AI_BENCH_SEED");
        set_repeat_index(0);
    }
}
