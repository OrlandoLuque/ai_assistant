use super::*;

// ─── Anti-Hallucination & Verification (V88) ──────────────────────────────────

#[cfg(feature = "eval")]
pub(crate) fn tests_anti_hallucination() -> CategoryResult {
    let results = vec![
        run_test("ah_config_defaults", || {
            let config = ai_assistant::anti_hallucination::AntiHallucinationConfig::default();
            assert!(!config.enabled);
            assert!((config.abstention_threshold - 0.3).abs() < f64::EPSILON);
            Ok(())
        }),
        run_test("ah_pipeline_create", || {
            let config = ai_assistant::anti_hallucination::AntiHallucinationConfig::default();
            let pipeline = ai_assistant::anti_hallucination::AntiHallucinationPipeline::new(config);
            assert!(!pipeline.config().enabled);
            Ok(())
        }),
        run_test("ah_strategy_variants", || {
            let strategies = [
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::Omit,
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::Mark,
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::Warn,
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::Footnote,
            ];
            assert!(strategies.len() >= 4);
            Ok(())
        }),
    ];

    CategoryResult {
        name: "anti-hallucination".to_string(),
        results,
    }
}

#[cfg(feature = "eval")]
pub(crate) fn tests_quality_gates() -> CategoryResult {
    let results = vec![
        run_test("qg_production_defaults", || {
            let runner = ai_assistant::quality_gates::QualityGateRunner::production_defaults();
            assert_eq!(runner.gates().len(), 3);
            Ok(())
        }),
        run_test("qg_strict", || {
            let runner = ai_assistant::quality_gates::QualityGateRunner::strict();
            assert_eq!(runner.gates().len(), 5);
            Ok(())
        }),
        run_test("qg_pass_high_scores", || {
            let runner = ai_assistant::quality_gates::QualityGateRunner::production_defaults();
            let scores = ai_assistant::quality_gates::QualityScores {
                faithfulness: Some(0.9),
                confidence: Some(0.8),
                grounding_ratio: Some(0.85),
                consistency_score: None,
                citation_coverage: None,
            };
            let result = runner.run(&scores);
            assert!(result.passed);
            Ok(())
        }),
        run_test("qg_badge_color", || {
            let scores = ai_assistant::quality_gates::QualityScores {
                faithfulness: Some(0.9),
                confidence: Some(0.8),
                grounding_ratio: None,
                consistency_score: None,
                citation_coverage: None,
            };
            let color = scores.badge_color();
            assert_eq!(color, "green");
            Ok(())
        }),
    ];

    CategoryResult {
        name: "quality-gates".to_string(),
        results,
    }
}

#[cfg(feature = "eval")]
pub(crate) fn tests_faithfulness() -> CategoryResult {
    let results = vec![
        run_test("faith_scorer_create", || {
            let config = ai_assistant::faithfulness::FaithfulnessConfig::default();
            let _scorer = ai_assistant::faithfulness::FaithfulnessScorer::new(config);
            Ok(())
        }),
        run_test("faith_nli_verdicts", || {
            assert_ne!(
                ai_assistant::faithfulness::NliVerdict::Entailed,
                ai_assistant::faithfulness::NliVerdict::Contradicted
            );
            Ok(())
        }),
    ];

    CategoryResult {
        name: "faithfulness".to_string(),
        results,
    }
}

#[cfg(feature = "eval")]
pub(crate) fn tests_verification() -> CategoryResult {
    let results = vec![
        run_test("cove_config_defaults", || {
            let config = ai_assistant::chain_of_verification::CoVeConfig::default();
            assert_eq!(config.max_claims_to_verify, 10);
            Ok(())
        }),
        run_test("cove_correction_modes", || {
            let modes = [
                ai_assistant::chain_of_verification::CorrectionMode::Replace,
                ai_assistant::chain_of_verification::CorrectionMode::Annotate,
                ai_assistant::chain_of_verification::CorrectionMode::Footnote,
            ];
            assert_eq!(modes.len(), 3);
            Ok(())
        }),
    ];

    CategoryResult {
        name: "verification".to_string(),
        results,
    }
}

#[cfg(feature = "research")]
pub(crate) fn tests_research() -> CategoryResult {
    let results = vec![
        run_test("research_arxiv_provider", || {
            let _provider = ai_assistant::academic_search::ArxivProvider::new();
            Ok(())
        }),
        run_test("research_bibtex_parse_empty", || {
            let entries = ai_assistant::bibtex::BibParser::parse("").map_err(|e| e.to_string())?;
            assert!(entries.is_empty());
            Ok(())
        }),
        run_test("research_paper_metadata", || {
            let _extractor = ai_assistant::paper_metadata::PaperMetadataExtractor::new();
            Ok(())
        }),
        run_test("research_mcp_tools", || {
            let registry = ai_assistant::mcp_research_tools::ResearchToolRegistry::new();
            assert_eq!(registry.tools().len(), 6);
            Ok(())
        }),
    ];

    CategoryResult {
        name: "research".to_string(),
        results,
    }
}
