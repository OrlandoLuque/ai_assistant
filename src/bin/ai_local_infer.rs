// `ai_local_infer` — drive the in-process Backend trait from the CLI.
// Streams generated chunks to stdout and persists a per-generation
// SloRecord as JSONL under ./.ai_assistant/local_infer_logs/ by default.
//
// Verbs:
//   info                                Print backend availability + VRAM detection
//   generate [opts]                     Single prompt, stream to stdout, log SLO
//   bench    [opts] [--iters N]         Repeat generate N times, print summary
//
// Options for generate / bench:
//   --backend <stub|candle|llama-cpp>   Default: stub
//   --model   <path>                    Path to model (ignored for stub)
//   --prompt  <text>                    Prompt text. Default: "Hello, world."
//   --max-tokens <N>                    Default: 256
//   --temperature <T>                   Default: 0.7
//   --top-p <T>                         Default: 0.9
//   --ctx-size <N>                      Default: 4096
//   --n-gpu-layers <N>                  Default: 0
//   --no-clamp                          Disable VRAM auto-clamp
//   --log-dir <path>                    Default: ./.ai_assistant/local_infer_logs

#![cfg(feature = "local-inference")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ai_assistant::local_inference::{
    load, vram, Backend, BackendKind, GenParams, LocalInferenceConfig, SloRecord,
};

const DEFAULT_LOG_DIR: &str = ".ai_assistant/local_infer_logs";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "info" => cmd_info(),
        "generate" => cmd_generate(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "--help" | "-h" => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("ai_local_infer: unknown verb '{}'\n\n{}", other, usage());
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "ai_local_infer <verb> [args]\n\
     \n\
     Verbs:\n\
       info                              Backend availability + VRAM detection\n\
       generate [opts]                   Stream a single prompt; persist SLO record\n\
       bench    [opts] [--iters N]       Repeat generate N times; print summary\n\
       --help, -h                        Show this message\n\
     \n\
     Common options for generate / bench:\n\
       --backend <stub|candle|llama-cpp>  Default: stub\n\
       --model   <path>                   Required for non-stub backends\n\
       --prompt  <text>                   Default: 'Hello, world.'\n\
       --max-tokens <N>                   Default: 256\n\
       --temperature <T>                  Default: 0.7\n\
       --top-p <T>                        Default: 0.9\n\
       --ctx-size <N>                     Default: 4096\n\
       --n-gpu-layers <N>                 Default: 0\n\
       --no-clamp                         Disable VRAM auto-clamp\n\
       --log-dir <path>                   Default: ./.ai_assistant/local_infer_logs"
}

fn cmd_info() -> ExitCode {
    println!("ai_local_infer — backend availability");
    for kind in [
        BackendKind::Stub,
        BackendKind::Candle,
        BackendKind::LlamaCpp,
    ] {
        let status = match kind {
            BackendKind::Stub => "available (always)",
            BackendKind::Candle => {
                #[cfg(feature = "local-inference-candle")]
                {
                    "available (local-inference-candle)"
                }
                #[cfg(not(feature = "local-inference-candle"))]
                {
                    "not compiled in (#319 — local-inference-candle)"
                }
            }
            BackendKind::LlamaCpp => {
                #[cfg(feature = "local-inference-llama-cpp")]
                {
                    "available (local-inference-llama-cpp)"
                }
                #[cfg(not(feature = "local-inference-llama-cpp"))]
                {
                    "not compiled in (#314 — local-inference-llama-cpp)"
                }
            }
        };
        println!("  {:<12} {}", kind.name(), status);
    }
    println!();
    println!("VRAM detection (best-effort, NVIDIA only)");
    match vram::detect_nvidia_mib() {
        Some((total, free)) => {
            println!("  total: {} MiB", total);
            println!("  free:  {} MiB", free);
        }
        None => println!("  (no NVIDIA GPU detected, or nvidia-smi unavailable)"),
    }
    ExitCode::SUCCESS
}

struct GenArgs {
    backend: BackendKind,
    model: PathBuf,
    prompt: String,
    params: GenParams,
    ctx_size: u32,
    n_gpu_layers: u32,
    allow_clamp: bool,
    log_dir: PathBuf,
}

fn parse_gen_args(args: &[String]) -> Result<GenArgs, String> {
    let backend = match arg(args, "--backend").as_deref() {
        Some("stub") | None => BackendKind::Stub,
        Some("candle") => BackendKind::Candle,
        Some("llama-cpp") | Some("llama-cpp-2") => BackendKind::LlamaCpp,
        Some(other) => return Err(format!("unknown backend: {}", other)),
    };
    let model = arg(args, "--model").map(PathBuf::from).unwrap_or_default();
    let prompt = arg(args, "--prompt").unwrap_or_else(|| "Hello, world.".into());
    let max_tokens: u32 = parse_arg(args, "--max-tokens", 256)?;
    let temperature: f32 = parse_arg(args, "--temperature", 0.7)?;
    let top_p: f32 = parse_arg(args, "--top-p", 0.9)?;
    let ctx_size: u32 = parse_arg(args, "--ctx-size", 4096)?;
    let n_gpu_layers: u32 = parse_arg(args, "--n-gpu-layers", 0)?;
    let allow_clamp = !args.iter().any(|a| a == "--no-clamp");
    let log_dir = arg(args, "--log-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LOG_DIR));
    Ok(GenArgs {
        backend,
        model,
        prompt,
        params: GenParams {
            max_tokens,
            temperature,
            top_p,
            stop: Vec::new(),
        },
        ctx_size,
        n_gpu_layers,
        allow_clamp,
        log_dir,
    })
}

fn cmd_generate(args: &[String]) -> ExitCode {
    let g = match parse_gen_args(args) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ai_local_infer generate: {}\n\n{}", e, usage());
            return ExitCode::from(2);
        }
    };
    match run_one(&g) {
        Ok(rec) => {
            print_summary(&rec);
            if let Err(e) = persist_record(&g.log_dir, &rec) {
                eprintln!("warning: could not persist SLO record: {}", e);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ai_local_infer generate: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_bench(args: &[String]) -> ExitCode {
    let g = match parse_gen_args(args) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ai_local_infer bench: {}\n\n{}", e, usage());
            return ExitCode::from(2);
        }
    };
    let iters: u32 = match parse_arg(args, "--iters", 5u32) {
        Ok(n) if n > 0 => n,
        Ok(_) => {
            eprintln!("ai_local_infer bench: --iters must be > 0");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("ai_local_infer bench: {}", e);
            return ExitCode::from(2);
        }
    };
    let mut records: Vec<SloRecord> = Vec::with_capacity(iters as usize);
    for i in 0..iters {
        eprintln!("--- iter {}/{} ---", i + 1, iters);
        match run_one(&g) {
            Ok(rec) => {
                eprintln!(
                    "  total {}ms, first_chunk {}ms, {:.1} tok/s",
                    rec.total_ms, rec.first_chunk_ms, rec.tokens_per_sec
                );
                if let Err(e) = persist_record(&g.log_dir, &rec) {
                    eprintln!("  warning: could not persist SLO record: {}", e);
                }
                records.push(rec);
            }
            Err(e) => {
                eprintln!("  iter failed: {}", e);
                return ExitCode::from(1);
            }
        }
    }
    print_bench_summary(&records);
    ExitCode::SUCCESS
}

fn run_one(g: &GenArgs) -> Result<SloRecord, String> {
    let cfg = LocalInferenceConfig::builder(g.backend, g.model.clone())
        .ctx_size(g.ctx_size)
        .n_gpu_layers(g.n_gpu_layers)
        .allow_gpu_clamp(g.allow_clamp)
        .build();

    let load_start = Instant::now();
    let mut backend: Box<dyn Backend> = load(&cfg).map_err(|e| format!("load: {}", e))?;
    let load_ms = load_start.elapsed().as_millis() as u64;

    // No total-layer count is exposed by the trait; absent that, "used"
    // mirrors "requested". Real backends will refine this.
    let n_used = g.n_gpu_layers;

    let gen_start = Instant::now();
    let mut first_chunk_ms: Option<u64> = None;
    let mut on_chunk = |c: &str| {
        if first_chunk_ms.is_none() {
            first_chunk_ms = Some(gen_start.elapsed().as_millis() as u64);
        }
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(c.as_bytes());
        let _ = out.flush();
    };
    let stats = backend
        .generate(&g.prompt, &g.params, &mut on_chunk)
        .map_err(|e| format!("generate: {}", e))?;
    let total_ms = gen_start.elapsed().as_millis() as u64;
    println!();

    Ok(SloRecord {
        ts_unix_ms: SloRecord::now_ms(),
        backend: backend.kind().name().to_string(),
        model_path: g.model.display().to_string(),
        load_ms,
        first_chunk_ms: first_chunk_ms.unwrap_or(total_ms),
        total_ms,
        prompt_tokens: stats.prompt_tokens,
        generated_tokens: stats.generated_tokens,
        tokens_per_sec: stats.tokens_per_sec,
        n_gpu_layers_requested: g.n_gpu_layers,
        n_gpu_layers_used: n_used,
        peak_vram_mib: stats.peak_vram_mib,
    })
}

fn print_summary(r: &SloRecord) {
    println!();
    println!("--- SLO record ---");
    println!("  backend:           {}", r.backend);
    println!("  model:             {}", r.model_path);
    println!("  load_ms:           {}", r.load_ms);
    println!("  first_chunk_ms:    {}", r.first_chunk_ms);
    println!("  total_ms:          {}", r.total_ms);
    println!("  prompt_tokens:     {}", r.prompt_tokens);
    println!("  generated_tokens:  {}", r.generated_tokens);
    println!("  tokens_per_sec:    {:.1}", r.tokens_per_sec);
    println!(
        "  gpu_layers:        {} requested, {} used",
        r.n_gpu_layers_requested, r.n_gpu_layers_used
    );
    if let Some(v) = r.peak_vram_mib {
        println!("  peak_vram_mib:     {}", v);
    }
}

fn print_bench_summary(records: &[SloRecord]) {
    if records.is_empty() {
        return;
    }
    let n = records.len() as f64;
    let avg_load = records.iter().map(|r| r.load_ms).sum::<u64>() as f64 / n;
    let avg_first = records.iter().map(|r| r.first_chunk_ms).sum::<u64>() as f64 / n;
    let avg_total = records.iter().map(|r| r.total_ms).sum::<u64>() as f64 / n;
    let avg_tps = records.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n;
    let max_total = records.iter().map(|r| r.total_ms).max().unwrap_or(0);
    let min_total = records.iter().map(|r| r.total_ms).min().unwrap_or(0);
    println!();
    println!("--- bench summary ({} iters) ---", records.len());
    println!("  avg load_ms:        {:.1}", avg_load);
    println!("  avg first_chunk_ms: {:.1}", avg_first);
    println!(
        "  avg total_ms:       {:.1} (min {}, max {})",
        avg_total, min_total, max_total
    );
    println!("  avg tokens/sec:     {:.1}", avg_tps);
}

fn persist_record(dir: &Path, rec: &SloRecord) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("local_infer_{}_{}.jsonl", ts, std::process::id()));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(rec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(f, "{}", line)?;
    Ok(())
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T, String>
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match arg(args, flag) {
        None => Ok(default),
        Some(v) => v
            .parse::<T>()
            .map_err(|e| format!("invalid value for {}: {}", flag, e)),
    }
}
