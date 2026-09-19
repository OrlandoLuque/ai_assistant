//! Proof that the bridge works: a GGUF loaded in THIS process, answering
//! through the ordinary `LlmProvider` port — no server, no subprocess.
//!
//! ```text
//! cargo run --release --example local_chat --features local-inference-llama-cpp -- <ruta.gguf>
//! ```
//!
//! A word about speed before anyone reads a number here as a verdict on the
//! model: for `Q1_0` weights on x86 this path is currently far slower than a
//! current `llama-cli` with the same file, because the llama.cpp vendored by
//! `llama-cpp-sys-2` has no x86 kernel for that type and falls back to a scalar
//! dot product. That is N84, and it is a fact about our bindings, not about the
//! model or about your computer.
use ai_assistant::local_inference::{BackendKind, GenParams, LocalInferenceConfig};
use ai_assistant::{ChatMessage, LlmProvider, LocalInferenceProvider};

fn main() {
    let Some(model) = std::env::args().nth(1) else {
        eprintln!("uso: local_chat <ruta a un .gguf>");
        eprintln!("p.ej. cargo run --release --example local_chat \\");
        eprintln!("        --features local-inference-llama-cpp -- modelos/Bonsai-4B-Q1_0.gguf");
        std::process::exit(2);
    };

    // Optional second argument: pin the thread count, so "is more threads
    // actually faster on this machine?" is a question you can answer by
    // running it rather than by assuming.
    let mut builder = LocalInferenceConfig::builder(BackendKind::LlamaCpp, &model).ctx_size(2048);
    if let Some(t) = std::env::args().nth(2).and_then(|s| s.parse::<u32>().ok()) {
        builder = builder.n_threads(t);
    }
    let config = builder.build();

    let started = std::time::Instant::now();
    let provider = match LocalInferenceProvider::load(&config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("no se pudo cargar {model}: {e}");
            std::process::exit(1);
        }
    };
    println!("cargado en {:?} -- {}", started.elapsed(), provider.label());

    let provider = provider.with_params(GenParams {
        max_tokens: 48,
        temperature: 0.5,
        top_p: 0.9,
        stop: vec!["<|im_end|>".to_string()],
    });

    // Deliberately a question with one right answer, so the output says whether
    // the model really ran rather than just whether bytes came back.
    let convo = vec![ChatMessage::user(
        "What is 17 times 3? Answer with just the number.",
    )];
    let t = std::time::Instant::now();
    match LlmProvider::generate(&provider, &convo, "You are concise.") {
        Ok(reply) => {
            println!("respuesta ({:?}): {:?}", t.elapsed(), reply.trim());
            // Counted by the backend, not divided by `max_tokens` — the model
            // stops at its end-of-turn token long before the ceiling, so the
            // ceiling would give a rate nobody measured.
            match provider.last_stats() {
                // `tokens_per_sec` mixes prompt processing with generation
                // (N85), so the counts are printed and the rate is labelled
                // for what it is rather than passed off as a writing speed.
                Some(s) => println!(
                    "  {} tokens de prompt, {} generados, {:.2} tok/s (mezclado: N85)",
                    s.prompt_tokens, s.generated_tokens, s.tokens_per_sec
                ),
                None => println!("  (sin estadisticas: el backend no devolvio ninguna)"),
            }
        }
        Err(e) => {
            eprintln!("error generando: {e}");
            std::process::exit(1);
        }
    }
}
