use super::*;

// ─── Real end-to-end battery (live model) ─────────────────────────────────────
//
// Unlike the mostly-deterministic categories, these drive a REAL local model
// (Ollama / llama3.2:3b — the 6/6 QA sweet spot) through realistic usage:
// multi-turn conversation with memory, document-grounded Q&A (inline + a real
// PDF), and task requests (extraction, summarization). Assertions are LENIENT
// (contains key terms / non-empty) because model output is not bit-exact. The
// whole category skips when Ollama is not reachable (e.g. CI).

pub(crate) fn tests_real_e2e() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Real E2E (live model: conversation / documents / tasks)"
        ))
    );
    let mut results = Vec::new();

    if !crate::bench_util::backend_reachable() {
        println!(
            "  {} backend not reachable — skipping real E2E",
            yellow("SKIP")
        );
        results.push(TestResult {
            name: "Ollama availability".to_string(),
            passed: true,
            message: Some("Skipped — Ollama not running".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "real_e2e".to_string(),
            results,
        };
    }
    crate::bench_util::warn_if_cpu_offloaded();

    // 1) Multi-turn conversation: state a fact, distract, then recall it.
    results.push(run_test("real: multi-turn recall (name + topic)", || {
        let mut a = crate::bench_util::bench_assistant();
        a.generate_sync(
            "My name is Carlos and I work mainly in Rust.".to_string(),
            "",
        )
        .map_err(|e| e.to_string())?;
        a.generate_sync("What is the capital of France?".to_string(), "")
            .map_err(|e| e.to_string())?;
        let ans = a
            .generate_sync(
                "Remind me: what is my name and which language do I work in?".to_string(),
                "",
            )
            .map_err(|e| e.to_string())?;
        let low = ans.to_lowercase();
        if !low.contains("carlos") || !low.contains("rust") {
            return Err(format!("recall failed: {:.200}", ans));
        }
        Ok(())
    }));

    // 2) Inline document grounding: the fact lives only in the knowledge block.
    results.push(run_test("real: inline document grounded answer", || {
        let mut a = crate::bench_util::bench_assistant();
        let doc = "Product sheet: the Model Q laptop weighs 1.2 kg, has 18 hours of \
                   battery life, and costs 1490 euros.";
        let ans = a
            .generate_sync(
                "According to the product sheet, how many hours of battery life does it have?"
                    .to_string(),
                doc,
            )
            .map_err(|e| e.to_string())?;
        if !ans.contains("18") {
            return Err(format!("grounding failed (expected 18): {:.200}", ans));
        }
        Ok(())
    }));

    // 3) Real PDF grounding: retrieve from a fetched arXiv paper, then answer.
    results.push(run_test("real: PDF-grounded answer (arxiv)", || {
        let Some(pdf) = crate::features::fetch_real_pdf() else {
            println!("      (network unavailable — skipped PDF-grounded answer)");
            return Ok(());
        };
        let parser =
            ai_assistant::DocumentParser::new(ai_assistant::DocumentParserConfig::default());
        let doc = parser
            .parse_bytes(&pdf, ai_assistant::DocumentFormat::Pdf)
            .map_err(|e| e.to_string())?;
        let passage = ai_assistant::knowledge_retrieval::select_relevant(
            &doc.text,
            "what is multi-head attention",
            1800,
        );
        let mut a = crate::bench_util::bench_assistant();
        let ans = a
            .generate_sync(
                "Using the provided context, briefly: what is multi-head attention?".to_string(),
                &passage,
            )
            .map_err(|e| e.to_string())?;
        if !ans.to_lowercase().contains("attention") {
            return Err(format!("PDF grounding off-topic: {:.200}", ans));
        }
        Ok(())
    }));

    // 4) Extraction task.
    results.push(run_test("real: extraction task (email)", || {
        let mut a = crate::bench_util::bench_assistant();
        let ans = a
            .generate_sync(
                "Extract only the email address and output just the email, nothing else: \
                 Please reach support at help@acme.io before Friday."
                    .to_string(),
                "",
            )
            .map_err(|e| e.to_string())?;
        if !ans.contains("help@acme.io") {
            return Err(format!("extraction failed: {:.200}", ans));
        }
        Ok(())
    }));

    // 5) Summarization task: non-empty and on-topic.
    results.push(run_test("real: summarization on-topic", || {
        let mut a = crate::bench_util::bench_assistant();
        let para = "Rust is a systems programming language focused on safety and performance. \
                    Its ownership model prevents data races and memory errors at compile time, \
                    without a garbage collector.";
        let ans = a
            .generate_sync(format!("Summarize this in one short sentence: {para}"), "")
            .map_err(|e| e.to_string())?;
        let low = ans.to_lowercase();
        let on_topic = low.contains("rust")
            || low.contains("safety")
            || low.contains("memory")
            || low.contains("performance")
            || low.contains("ownership");
        if ans.trim().is_empty() || !on_topic {
            return Err(format!("summary empty/off-topic: {:.200}", ans));
        }
        Ok(())
    }));

    CategoryResult {
        name: "real_e2e".to_string(),
        results,
    }
}
