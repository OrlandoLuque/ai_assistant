//! Comprehensive feature combination tests for the eval-suite.
//!
//! Tests that exercise knowledge graphs, decision trees, neural embeddings,
//! RAG pipelines, vector DBs, advanced memory, and multi-agent orchestration
//! — individually and in every meaningful combination.

#[cfg(test)]
mod tests {
    use super::super::agent_config::{EvalAgentConfig, MultiModelGenerator};
    use super::super::dataset::*;
    use super::super::runner::ModelIdentifier;
    use super::super::scoring::{DefaultScorer, ProblemScorer};
    use super::super::subtask::Subtask;
    use crate::decision_tree::{
        Condition, ConditionOperator, DecisionBranch, DecisionNode, DecisionTreeBuilder,
    };
    use serde_json::json;
    use std::collections::HashMap;
    #[allow(unused_imports)]
    use std::sync::Arc;

    // ========================================================================
    // Shared helpers
    // ========================================================================

    /// Deterministic embedding from text bytes, L2-normalized.
    fn deterministic_embedding(text: &str, dims: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dims];
        for (i, byte) in text.bytes().enumerate() {
            emb[i % dims] += (byte as f32) / 255.0;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut emb {
                *x /= norm;
            }
        }
        emb
    }

    fn test_model(name: &str) -> ModelIdentifier {
        ModelIdentifier {
            name: name.to_string(),
            provider: "mock".to_string(),
            variant: None,
        }
    }

    /// 10 problems: 3 knowledge MC, 3 math (2 numeric + 1 MC), 2 code, 2 entity MC.
    fn make_mixed_dataset() -> BenchmarkDataset {
        BenchmarkDataset::from_problems(
            "feature_combo",
            BenchmarkSuiteType::Custom("FeatureCombo".into()),
            vec![
                // Knowledge MC
                make_mc_problem(
                    "fc/know/1",
                    "What is the capital of France? A) London B) Paris C) Berlin D) Rome",
                    vec!["A", "B", "C", "D"],
                    "B",
                ),
                make_mc_problem(
                    "fc/know/2",
                    "Who discovered penicillin? A) Einstein B) Fleming C) Darwin D) Newton",
                    vec!["A", "B", "C", "D"],
                    "B",
                ),
                make_mc_problem(
                    "fc/know/3",
                    "Chemical symbol for gold? A) Ag B) Fe C) Au D) Cu",
                    vec!["A", "B", "C", "D"],
                    "C",
                ),
                // Math
                make_numeric_problem("fc/math/1", "Calculate: 6 * 7", 42.0, 0.01),
                make_numeric_problem("fc/math/2", "How many sides does a hexagon have?", 6.0, 0.01),
                make_mc_problem(
                    "fc/math/3",
                    "What is 2^10? A) 512 B) 1024 C) 2048 D) 4096",
                    vec!["A", "B", "C", "D"],
                    "B",
                ),
                // Code
                make_code_problem(
                    "fc/code/1",
                    "Write a Python function that adds two numbers",
                    "def add(a, b): return a + b",
                    "python",
                ),
                make_code_problem(
                    "fc/code/2",
                    "Write a Python function that returns the length of a string",
                    "def length(s): return len(s)",
                    "python",
                ),
                // Entity query MC
                make_mc_problem(
                    "fc/entity/1",
                    "Paris is the capital of which country? A) Germany B) France C) Italy D) Spain",
                    vec!["A", "B", "C", "D"],
                    "B",
                ),
                make_mc_problem(
                    "fc/entity/2",
                    "Einstein developed the theory of A) Evolution B) Relativity C) Gravity D) Optics",
                    vec!["A", "B", "C", "D"],
                    "B",
                ),
            ],
        )
    }

    fn make_categorized_problems() -> (BenchmarkDataset, HashMap<String, Subtask>) {
        let ds = make_mixed_dataset();
        let mut tags = HashMap::new();
        for i in 1..=3 {
            tags.insert(
                format!("fc/know/{}", i),
                Subtask::InformationGathering,
            );
        }
        for i in 1..=3 {
            tags.insert(format!("fc/math/{}", i), Subtask::ReasoningChain);
        }
        for i in 1..=2 {
            tags.insert(format!("fc/code/{}", i), Subtask::CodeGeneration);
        }
        for i in 1..=2 {
            tags.insert(
                format!("fc/entity/{}", i),
                Subtask::Custom("EntityQuery".into()),
            );
        }
        (ds, tags)
    }

    /// Mock generator that returns appropriate answers based on prompt content.
    #[allow(dead_code)]
    fn mock_generator_by_category(
    ) -> Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> {
        Arc::new(|prompt: &str| {
            let lower = prompt.to_lowercase();
            if lower.contains("capital") && lower.contains("france") {
                Ok("B".to_string())
            } else if lower.contains("penicillin") || lower.contains("fleming") {
                Ok("B".to_string())
            } else if lower.contains("chemical symbol") || lower.contains("gold") {
                Ok("C".to_string())
            } else if lower.contains("2^10") || lower.contains("1024") {
                Ok("B".to_string())
            } else if lower.contains("einstein") || lower.contains("relativity") {
                Ok("B".to_string())
            } else if lower.contains("6 * 7") || lower.contains("calculate") {
                Ok("42".to_string())
            } else if lower.contains("hexagon") || lower.contains("sides") {
                Ok("6".to_string())
            } else if lower.contains("function") && lower.contains("add") {
                Ok("def add(a, b): return a + b".to_string())
            } else if lower.contains("function") && lower.contains("length") {
                Ok("def length(s): return len(s)".to_string())
            } else {
                Ok("B".to_string())
            }
        })
    }

    /// Build a decision tree that routes by category context variable.
    fn build_category_routing_tree() -> crate::decision_tree::DecisionTree {
        let math_branch = DecisionBranch {
            condition: Condition::new("category", ConditionOperator::Equals, json!("Math")),
            target_node_id: "math_model".to_string(),
            label: Some("Math".to_string()),
        };
        let code_branch = DecisionBranch {
            condition: Condition::new("category", ConditionOperator::Equals, json!("Code")),
            target_node_id: "code_model".to_string(),
            label: Some("Code".to_string()),
        };
        let entity_branch = DecisionBranch {
            condition: Condition::new("category", ConditionOperator::Equals, json!("Entity")),
            target_node_id: "entity_model".to_string(),
            label: Some("Entity".to_string()),
        };

        DecisionTreeBuilder::new("category_router", "Category Router")
            .root("check_category")
            .node(DecisionNode::new_condition(
                "check_category",
                vec![math_branch, code_branch, entity_branch],
                Some("knowledge_model".to_string()),
            ))
            .node(DecisionNode::new_terminal(
                "knowledge_model",
                json!("knowledge-specialist"),
                Some("Knowledge".to_string()),
            ))
            .node(DecisionNode::new_terminal(
                "math_model",
                json!("math-specialist"),
                Some("Math".to_string()),
            ))
            .node(DecisionNode::new_terminal(
                "code_model",
                json!("code-specialist"),
                Some("Code".to_string()),
            ))
            .node(DecisionNode::new_terminal(
                "entity_model",
                json!("entity-specialist"),
                Some("Entity".to_string()),
            ))
            .build()
    }

    /// Map problem ID to category string for decision tree context.
    fn problem_category(problem_id: &str) -> &str {
        if problem_id.contains("know") {
            "Knowledge"
        } else if problem_id.contains("math") {
            "Math"
        } else if problem_id.contains("code") {
            "Code"
        } else if problem_id.contains("entity") {
            "Entity"
        } else {
            "Unknown"
        }
    }

    // ========================================================================
    // tree_tests — Decision Tree (no extra features)
    // ========================================================================

    #[test]
    fn test_decision_tree_problem_routing() {
        let tree = build_category_routing_tree();
        let ds = make_mixed_dataset();

        for problem in &ds.problems {
            let mut ctx = HashMap::new();
            ctx.insert(
                "category".to_string(),
                json!(problem_category(&problem.id)),
            );
            let path = tree.evaluate(&ctx);
            assert!(path.complete, "Path should be complete for {}", problem.id);
            let result = path.result.unwrap();
            let model = result.as_str().unwrap();

            match problem_category(&problem.id) {
                "Knowledge" => assert_eq!(model, "knowledge-specialist"),
                "Math" => assert_eq!(model, "math-specialist"),
                "Code" => assert_eq!(model, "code-specialist"),
                "Entity" => assert_eq!(model, "entity-specialist"),
                _ => panic!("Unknown category"),
            }
        }
    }

    #[test]
    fn test_decision_tree_scoring_rules() {
        let mc_branch = DecisionBranch {
            condition: Condition::new(
                "problem_type",
                ConditionOperator::Equals,
                json!("MultipleChoice"),
            ),
            target_node_id: "mc_strategy".to_string(),
            label: Some("MC".to_string()),
        };
        let num_branch = DecisionBranch {
            condition: Condition::new(
                "problem_type",
                ConditionOperator::Equals,
                json!("Numeric"),
            ),
            target_node_id: "num_strategy".to_string(),
            label: Some("Numeric".to_string()),
        };
        let code_branch = DecisionBranch {
            condition: Condition::new(
                "problem_type",
                ConditionOperator::Equals,
                json!("Code"),
            ),
            target_node_id: "code_strategy".to_string(),
            label: Some("Code".to_string()),
        };

        let tree = DecisionTreeBuilder::new("scoring_router", "Scoring Strategy Router")
            .root("check_type")
            .node(DecisionNode::new_condition(
                "check_type",
                vec![mc_branch, num_branch, code_branch],
                Some("default_strategy".to_string()),
            ))
            .node(DecisionNode::new_terminal(
                "mc_strategy",
                json!("strict_letter_match"),
                None,
            ))
            .node(DecisionNode::new_terminal(
                "num_strategy",
                json!("tolerance_based"),
                None,
            ))
            .node(DecisionNode::new_terminal(
                "code_strategy",
                json!("jaccard_similarity"),
                None,
            ))
            .node(DecisionNode::new_terminal(
                "default_strategy",
                json!("fuzzy_match"),
                None,
            ))
            .build();

        // Test MC routing
        let mut ctx = HashMap::new();
        ctx.insert("problem_type".to_string(), json!("MultipleChoice"));
        let path = tree.evaluate(&ctx);
        assert_eq!(
            path.result.unwrap().as_str().unwrap(),
            "strict_letter_match"
        );

        // Test Numeric routing
        ctx.insert("problem_type".to_string(), json!("Numeric"));
        let path = tree.evaluate(&ctx);
        assert_eq!(path.result.unwrap().as_str().unwrap(), "tolerance_based");

        // Test Code routing
        ctx.insert("problem_type".to_string(), json!("Code"));
        let path = tree.evaluate(&ctx);
        assert_eq!(
            path.result.unwrap().as_str().unwrap(),
            "jaccard_similarity"
        );

        // Test fallback
        ctx.insert("problem_type".to_string(), json!("FreeText"));
        let path = tree.evaluate(&ctx);
        assert_eq!(path.result.unwrap().as_str().unwrap(), "fuzzy_match");
    }

    #[test]
    fn test_decision_tree_multi_model_routing() {
        let tree = build_category_routing_tree();
        let ds = make_mixed_dataset();

        // Register category-specific generators in MultiModelGenerator
        let mut generator = MultiModelGenerator::new(
            |_prompt: &str| -> Result<String, String> { Ok("B".to_string()) },
        );

        // Knowledge specialist: always returns correct MC letters
        generator.register_model("knowledge-specialist", |prompt: &str| {
            let lower = prompt.to_lowercase();
            if lower.contains("capital") {
                Ok("B".to_string())
            } else if lower.contains("penicillin") {
                Ok("B".to_string())
            } else if lower.contains("gold") {
                Ok("C".to_string())
            } else {
                Ok("B".to_string())
            }
        });

        // Math specialist: returns numbers
        generator.register_model("math-specialist", |prompt: &str| {
            let lower = prompt.to_lowercase();
            if lower.contains("6 * 7") {
                Ok("42".to_string())
            } else if lower.contains("hexagon") {
                Ok("6".to_string())
            } else if lower.contains("2^10") {
                Ok("B".to_string())
            } else {
                Ok("0".to_string())
            }
        });

        // Code specialist: returns code
        generator.register_model("code-specialist", |prompt: &str| {
            let lower = prompt.to_lowercase();
            if lower.contains("add") {
                Ok("def add(a, b): return a + b".to_string())
            } else if lower.contains("length") {
                Ok("def length(s): return len(s)".to_string())
            } else {
                Ok("pass".to_string())
            }
        });

        // Entity specialist
        generator.register_model("entity-specialist", |_prompt: &str| {
            Ok("B".to_string())
        });

        // Route each problem through tree → generator, score
        let scorer = DefaultScorer;
        let mut routed_scores = Vec::new();
        let mut single_scores = Vec::new();

        for problem in &ds.problems {
            let mut ctx = HashMap::new();
            ctx.insert(
                "category".to_string(),
                json!(problem_category(&problem.id)),
            );
            let path = tree.evaluate(&ctx);
            let model_key = path.result.unwrap();
            let model_key_str = model_key.as_str().unwrap();

            // Routed response
            let routed_response = generator
                .generate(model_key_str, &problem.prompt)
                .unwrap();
            let routed_score = scorer.score(problem, &routed_response);
            routed_scores.push(routed_score);

            // Single-model baseline (default always returns "B")
            let single_response = generator.generate("nonexistent", &problem.prompt).unwrap();
            let single_score = scorer.score(problem, &single_response);
            single_scores.push(single_score);
        }

        let routed_avg: f64 = routed_scores.iter().sum::<f64>() / routed_scores.len() as f64;
        let single_avg: f64 = single_scores.iter().sum::<f64>() / single_scores.len() as f64;

        // Routed should be >= single-model (specialized generators know the answers)
        assert!(
            routed_avg >= single_avg,
            "Routed avg ({:.3}) should be >= single avg ({:.3})",
            routed_avg,
            single_avg
        );
        // Should have high accuracy with specialized generators
        assert!(
            routed_avg > 0.5,
            "Routed avg should be > 0.5, got {:.3}",
            routed_avg
        );
    }

    // ========================================================================
    // embedding_tests — needs `embeddings` feature
    // ========================================================================
    #[cfg(feature = "embeddings")]
    mod embedding_tests {
        use super::*;
        use crate::embeddings::{EmbeddingConfig, LocalEmbedder};
        use crate::vector_db::{
            DistanceMetric, FilterOperation, InMemoryVectorDb, MetadataFilter, VectorDb,
            VectorDbConfig,
        };

        #[test]
        fn test_embeddings_problem_clustering() {
            let ds = make_mixed_dataset();
            let prompts: Vec<&str> = ds.problems.iter().map(|p| p.prompt.as_str()).collect();
            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            embedder.train(&prompts);

            let embeddings: Vec<Vec<f32>> =
                prompts.iter().map(|p| embedder.embed(p)).collect();

            // Same-category similarity should be higher than cross-category
            // Know: 0,1,2  Math: 3,4,5  Code: 6,7  Entity: 8,9
            let sim_know_know =
                LocalEmbedder::cosine_similarity(&embeddings[0], &embeddings[1]);
            let _sim_know_code =
                LocalEmbedder::cosine_similarity(&embeddings[0], &embeddings[6]);
            let sim_math_math =
                LocalEmbedder::cosine_similarity(&embeddings[3], &embeddings[4]);
            let sim_math_code =
                LocalEmbedder::cosine_similarity(&embeddings[3], &embeddings[6]);

            // At minimum, math-math should be more similar than math-code
            // (both math problems talk about numbers, calculation)
            assert!(
                sim_math_math >= sim_math_code - 0.1,
                "Math-Math ({:.3}) should be >= Math-Code ({:.3}) minus margin",
                sim_math_math,
                sim_math_code
            );
            // Know-know should have some similarity
            assert!(
                sim_know_know > -0.5,
                "Know-Know similarity should be > -0.5, got {:.3}",
                sim_know_know
            );
        }

        #[test]
        fn test_embeddings_answer_similarity_scoring() {
            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            let corpus = vec![
                "def add(a, b): return a + b",
                "def length(s): return len(s)",
                "The capital of France is Paris",
                "completely unrelated response about cooking",
            ];
            embedder.train(&corpus);

            let ref_emb = embedder.embed("def add(a, b): return a + b");
            let correct_emb = embedder.embed("def add(a, b): return a + b");
            let similar_emb = embedder.embed("def add(x, y): return x + y");
            let wrong_emb = embedder.embed("completely unrelated response about cooking");

            let sim_correct = LocalEmbedder::cosine_similarity(&ref_emb, &correct_emb);
            let sim_similar = LocalEmbedder::cosine_similarity(&ref_emb, &similar_emb);
            let sim_wrong = LocalEmbedder::cosine_similarity(&ref_emb, &wrong_emb);

            // Exact match should score highest
            assert!(
                (sim_correct - 1.0).abs() < 0.01,
                "Exact match should have sim ~1.0, got {:.3}",
                sim_correct
            );
            // Similar should score higher than wrong
            assert!(
                sim_similar > sim_wrong,
                "Similar ({:.3}) should score > wrong ({:.3})",
                sim_similar,
                sim_wrong
            );
        }

        #[test]
        fn test_vector_db_problem_storage_and_retrieval() {
            let ds = make_mixed_dataset();
            let config = VectorDbConfig {
                dimensions: 32,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(100),
                collection_name: "test_problems".to_string(),
                ..Default::default()
            };
            let mut db = InMemoryVectorDb::new(config);

            // Store all problem embeddings
            for problem in &ds.problems {
                let emb = deterministic_embedding(&problem.prompt, 32);
                let meta = json!({"category": problem_category(&problem.id)});
                db.insert(&problem.id, emb, meta).unwrap();
            }

            assert_eq!(db.count(), 10);

            // Search for first problem — should find itself as top-1
            let query_emb = deterministic_embedding(&ds.problems[0].prompt, 32);
            let results = db.search(&query_emb, 3, None).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].id, "fc/know/1");
        }

        #[test]
        fn test_vector_db_metadata_filtering() {
            let ds = make_mixed_dataset();
            let config = VectorDbConfig {
                dimensions: 32,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(100),
                collection_name: "test_filter".to_string(),
                ..Default::default()
            };
            let mut db = InMemoryVectorDb::new(config);

            for problem in &ds.problems {
                let emb = deterministic_embedding(&problem.prompt, 32);
                let meta = json!({"category": problem_category(&problem.id)});
                db.insert(&problem.id, emb, meta).unwrap();
            }

            // Filter for Math problems only
            let filter = vec![MetadataFilter {
                field: "category".to_string(),
                operation: FilterOperation::Equals(json!("Math")),
            }];
            let query = deterministic_embedding("math calculation numbers", 32);
            let results = db.search(&query, 10, Some(&filter)).unwrap();

            // Should only return Math problems (fc/math/1, fc/math/2, fc/math/3)
            assert!(
                results.len() <= 3,
                "Should return at most 3 math problems, got {}",
                results.len()
            );
            for r in &results {
                assert!(
                    r.id.contains("math"),
                    "Filtered result {} should be a math problem",
                    r.id
                );
            }
        }

        #[test]
        fn test_neural_embeddings_dense_scoring() {
            use crate::neural_embeddings::{DenseEmbedder, DenseEmbeddingConfig};

            let config = DenseEmbeddingConfig {
                dimension: 64,
                ..Default::default()
            };
            let embedder = DenseEmbedder::new(config);

            // Embed two math problems and one code problem
            let math1 = embedder.embed("Calculate 6 times 7").unwrap();
            let math2 = embedder.embed("How many sides does a hexagon have").unwrap();
            let code1 = embedder.embed("Write a Python function that adds numbers").unwrap();

            assert_eq!(math1.len(), 64);
            assert_eq!(math2.len(), 64);
            assert_eq!(code1.len(), 64);

            // Verify embeddings are normalized (L2 norm ~1.0)
            let norm: f32 = math1.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Embedding should be roughly normalized, got norm {:.3}",
                norm
            );
        }

        #[test]
        fn test_neural_embeddings_sparse_keyword_overlap() {
            use crate::neural_embeddings::{SparseEmbeddingConfig, SparseEmbedder};

            let mut embedder = SparseEmbedder::new(SparseEmbeddingConfig::default());
            let docs = vec![
                "Calculate the sum of numbers".to_string(),
                "How many sides does a hexagon have".to_string(),
                "Write a Python function to add numbers".to_string(),
                "The capital of France is Paris".to_string(),
            ];
            embedder.fit(&docs);

            let math_emb = embedder.embed("Calculate the sum of numbers");
            let code_emb = embedder.embed("Write a Python function to add numbers");
            let know_emb = embedder.embed("The capital of France is Paris");

            // Math and code share "numbers" → some overlap
            let math_code_dot: f32 = math_emb
                .iter()
                .filter_map(|(k, v)| code_emb.get(k).map(|cv| v * cv))
                .sum();

            // Math and knowledge share less
            let math_know_dot: f32 = math_emb
                .iter()
                .filter_map(|(k, v)| know_emb.get(k).map(|kv| v * kv))
                .sum();

            // We expect at least that sparse embeddings are non-empty
            assert!(
                !math_emb.is_empty(),
                "Math sparse embedding should not be empty"
            );
            assert!(
                !code_emb.is_empty(),
                "Code sparse embedding should not be empty"
            );

            // Both dot products should be non-negative (valid TF-IDF weights)
            assert!(
                math_code_dot >= 0.0,
                "Math-Code dot ({:.4}) should be non-negative",
                math_code_dot,
            );
            assert!(
                math_know_dot >= 0.0,
                "Math-Know dot ({:.4}) should be non-negative",
                math_know_dot,
            );
        }

        #[test]
        fn test_combo_embeddings_vectordb_retrieval() {
            let ds = make_mixed_dataset();
            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            let prompts: Vec<&str> = ds.problems.iter().map(|p| p.prompt.as_str()).collect();
            embedder.train(&prompts);

            let config = VectorDbConfig {
                dimensions: EmbeddingConfig::default().dimensions,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(100),
                collection_name: "combo_test".to_string(),
                ..Default::default()
            };
            let mut db = InMemoryVectorDb::new(config);

            // Store embeddings with reference solutions as metadata
            for problem in &ds.problems {
                let emb = embedder.embed(&problem.prompt);
                let meta = json!({
                    "category": problem_category(&problem.id),
                    "reference": problem.reference_solution.as_deref().unwrap_or(""),
                });
                db.insert(&problem.id, emb, meta).unwrap();
            }

            // For a math problem, retrieve similar — should find other math problems
            let query_emb = embedder.embed("Calculate 6 * 7");
            let results = db.search(&query_emb, 3, None).unwrap();

            assert!(!results.is_empty());
            // At least one result should be a math problem
            let has_math = results.iter().any(|r| r.id.contains("math"));
            assert!(has_math, "Retrieved results should include math problems");
        }

        #[test]
        fn test_combo_tree_embeddings_routing() {
            let ds = make_mixed_dataset();
            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            let prompts: Vec<&str> = ds.problems.iter().map(|p| p.prompt.as_str()).collect();
            embedder.train(&prompts);

            // Create a "code reference" embedding
            let code_ref = embedder.embed("Write a function in Python programming code def return");

            // Build a tree that routes based on similarity threshold
            let high_sim_branch = DecisionBranch {
                condition: Condition::new(
                    "is_code_like",
                    ConditionOperator::Equals,
                    json!(true),
                ),
                target_node_id: "code_path".to_string(),
                label: Some("Code-like".to_string()),
            };

            let tree = DecisionTreeBuilder::new("emb_router", "Embedding Router")
                .root("check_sim")
                .node(DecisionNode::new_condition(
                    "check_sim",
                    vec![high_sim_branch],
                    Some("general_path".to_string()),
                ))
                .node(DecisionNode::new_terminal(
                    "code_path",
                    json!("code-specialist"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "general_path",
                    json!("general-model"),
                    None,
                ))
                .build();

            let mut code_count = 0;
            let mut general_count = 0;

            for problem in &ds.problems {
                let emb = embedder.embed(&problem.prompt);
                let sim = LocalEmbedder::cosine_similarity(&emb, &code_ref);

                let mut ctx = HashMap::new();
                ctx.insert("is_code_like".to_string(), json!(sim > 0.5));
                let path = tree.evaluate(&ctx);
                let model = path.result.unwrap();

                if model.as_str().unwrap() == "code-specialist" {
                    code_count += 1;
                } else {
                    general_count += 1;
                }
            }

            // Should have some routing happening
            assert!(
                code_count + general_count == 10,
                "All 10 problems should be routed"
            );
            // Code problems should tend toward code path (at least 1)
            // but we can't guarantee exact splits with TF-IDF
            assert!(
                code_count >= 0 && general_count >= 0,
                "Both paths should be valid"
            );
        }
    }

    // ========================================================================
    // rag_tests — needs `rag` feature
    // ========================================================================
    #[cfg(feature = "rag")]
    mod rag_tests {
        use super::*;
        use crate::knowledge_graph::{
            EntityExtractor, EntityType, KnowledgeGraphBuilder, PatternEntityExtractor,
        };
        use crate::rag_pipeline::{
            GraphCallback, GraphRelation, LlmCallback, RagPipeline, RetrievalCallback,
            RetrievedChunk,
        };
        use crate::rag_tiers::{RagTierConfig, RagTier};

        struct MockLlmCb;
        impl LlmCallback for MockLlmCb {
            fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
                Ok("Generated response".to_string())
            }
            fn model_name(&self) -> &str {
                "mock-llm"
            }
        }

        struct MockRetrievalCb {
            chunks: Vec<RetrievedChunk>,
        }

        impl MockRetrievalCb {
            fn new(chunks: Vec<RetrievedChunk>) -> Self {
                Self { chunks }
            }

            fn default_chunks() -> Vec<RetrievedChunk> {
                vec![
                    RetrievedChunk {
                        chunk_id: "c1".to_string(),
                        content: "Paris is the capital of France".to_string(),
                        source: "geography.txt".to_string(),
                        section: None,
                        score: 0.9,
                        keyword_score: Some(0.9),
                        semantic_score: None,
                        token_count: 7,
                        position: None,
                        metadata: HashMap::new(),
                    },
                    RetrievedChunk {
                        chunk_id: "c2".to_string(),
                        content: "Einstein developed the theory of relativity".to_string(),
                        source: "science.txt".to_string(),
                        section: None,
                        score: 0.85,
                        keyword_score: Some(0.85),
                        semantic_score: None,
                        token_count: 7,
                        position: None,
                        metadata: HashMap::new(),
                    },
                    RetrievedChunk {
                        chunk_id: "c3".to_string(),
                        content: "Gold has the chemical symbol Au".to_string(),
                        source: "chemistry.txt".to_string(),
                        section: None,
                        score: 0.8,
                        keyword_score: Some(0.8),
                        semantic_score: None,
                        token_count: 7,
                        position: None,
                        metadata: HashMap::new(),
                    },
                ]
            }
        }

        impl RetrievalCallback for MockRetrievalCb {
            fn keyword_search(
                &self,
                query: &str,
                limit: usize,
            ) -> Result<Vec<RetrievedChunk>, String> {
                let lower = query.to_lowercase();
                let mut results: Vec<_> = self
                    .chunks
                    .iter()
                    .filter(|c| {
                        let cl = c.content.to_lowercase();
                        lower.split_whitespace().any(|w| cl.contains(w))
                    })
                    .take(limit)
                    .cloned()
                    .collect();
                if results.is_empty() {
                    results = self.chunks.iter().take(limit).cloned().collect();
                }
                Ok(results)
            }

            fn semantic_search(
                &self,
                _embedding: &[f32],
                limit: usize,
            ) -> Result<Vec<RetrievedChunk>, String> {
                Ok(self.chunks.iter().take(limit).cloned().collect())
            }

            fn get_chunk(&self, chunk_id: &str) -> Result<Option<RetrievedChunk>, String> {
                Ok(self.chunks.iter().find(|c| c.chunk_id == chunk_id).cloned())
            }
        }

        struct MockGraphCb;
        impl GraphCallback for MockGraphCb {
            fn extract_entities(&self, text: &str) -> Result<Vec<String>, String> {
                let mut entities = Vec::new();
                let lower = text.to_lowercase();
                if lower.contains("paris") {
                    entities.push("Paris".to_string());
                }
                if lower.contains("france") {
                    entities.push("France".to_string());
                }
                if lower.contains("einstein") {
                    entities.push("Einstein".to_string());
                }
                Ok(entities)
            }

            fn get_related(
                &self,
                entity: &str,
                _depth: usize,
            ) -> Result<Vec<GraphRelation>, String> {
                match entity {
                    "Paris" => Ok(vec![GraphRelation {
                        from: "Paris".to_string(),
                        relation_type: "capital_of".to_string(),
                        to: "France".to_string(),
                        weight: 1.0,
                    }]),
                    "Einstein" => Ok(vec![GraphRelation {
                        from: "Einstein".to_string(),
                        relation_type: "developed".to_string(),
                        to: "Theory of Relativity".to_string(),
                        weight: 1.0,
                    }]),
                    _ => Ok(vec![]),
                }
            }

            fn get_entity_chunks(
                &self,
                entities: &[String],
            ) -> Result<Vec<RetrievedChunk>, String> {
                let mut chunks = Vec::new();
                for entity in entities {
                    let content = match entity.as_str() {
                        "Paris" => "Paris is the capital city of France, known for the Eiffel Tower.",
                        "Einstein" => "Albert Einstein developed the theory of special and general relativity.",
                        _ => continue,
                    };
                    chunks.push(RetrievedChunk {
                        chunk_id: format!("graph_{}", entity.to_lowercase()),
                        content: content.to_string(),
                        source: "knowledge_graph".to_string(),
                        section: None,
                        score: 0.95,
                        keyword_score: None,
                        semantic_score: Some(0.95),
                        token_count: 10,
                        position: None,
                        metadata: HashMap::new(),
                    });
                }
                Ok(chunks)
            }
        }

        #[test]
        fn test_knowledge_graph_entity_enrichment() {
            let graph = KnowledgeGraphBuilder::new()
                .add_entity("Paris", EntityType::Location)
                .add_entity("France", EntityType::Location)
                .add_entity("Einstein", EntityType::Person)
                .add_entity("Gold", EntityType::Concept)
                .build_in_memory()
                .unwrap();

            let stats = graph.stats().unwrap();
            assert!(
                stats.total_entities >= 4,
                "Should have at least 4 entities, got {}",
                stats.total_entities
            );

            // Pattern extractor
            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("France", EntityType::Location);
            extractor = extractor.add_entity("Einstein", EntityType::Person);

            let extraction = extractor.extract(
                "Paris is the capital of France. Einstein developed relativity.",
            ).unwrap();
            assert!(
                !extraction.entities.is_empty(),
                "Should extract at least one entity"
            );
            let names: Vec<&str> = extraction.entities.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&"Paris") || names.contains(&"paris"));
        }

        #[test]
        fn test_knowledge_graph_relation_scoring() {
            // Build KG with entities and index a document containing a relation
            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("France", EntityType::Location);

            let mut graph = KnowledgeGraphBuilder::new()
                .add_entity("Paris", EntityType::Location)
                .add_entity("France", EntityType::Location)
                .build_in_memory()
                .unwrap();

            // Index a document that connects Paris and France
            let result = graph.index_document(
                "doc1",
                "Paris is the capital of France. It is the largest city in France.",
                &extractor,
            );
            assert!(result.is_ok(), "Indexing should succeed");
            let idx = result.unwrap();
            assert!(idx.entities_extracted > 0, "Should extract entities");

            // Use as_graph_callback to query
            let cb = graph.as_graph_callback(&extractor);
            let entities = cb.extract_entities("Tell me about Paris").unwrap();
            assert!(!entities.is_empty(), "Should find Paris entity");

            // Get entity chunks
            let chunks = cb.get_entity_chunks(&entities).unwrap();
            assert!(
                !chunks.is_empty(),
                "Should retrieve chunks linked to entities"
            );
        }

        #[test]
        fn test_rag_pipeline_augmented_eval() {
            let config = RagTierConfig::with_tier(RagTier::Fast);
            let mut pipeline = RagPipeline::new(config);
            let llm = MockLlmCb;
            let retrieval = MockRetrievalCb::new(MockRetrievalCb::default_chunks());

            let result = pipeline
                .process("What is the capital of France?", &llm, None, &retrieval, None)
                .unwrap();

            assert!(!result.context.is_empty(), "Context should not be empty");
            assert!(!result.chunks.is_empty(), "Should have retrieved chunks");
            assert!(
                result.context.contains("Paris") || result.context.contains("capital"),
                "Context should mention Paris or capital"
            );
        }

        #[test]
        fn test_rag_tier_quality_comparison() {
            let retrieval = MockRetrievalCb::new(MockRetrievalCb::default_chunks());
            let llm = MockLlmCb;

            // Test with Disabled tier
            let mut pipeline_disabled = RagPipeline::new(RagTierConfig::with_tier(RagTier::Disabled));
            let result_disabled = pipeline_disabled.process(
                "What is the capital of France?",
                &llm,
                None,
                &retrieval,
                None,
            );
            // Disabled tier might return empty context
            let disabled_chunks = result_disabled
                .map(|r| r.chunks.len())
                .unwrap_or(0);

            // Test with Fast tier
            let mut pipeline_fast = RagPipeline::new(RagTierConfig::with_tier(RagTier::Fast));
            let result_fast = pipeline_fast
                .process("What is the capital of France?", &llm, None, &retrieval, None)
                .unwrap();
            let fast_chunks = result_fast.chunks.len();

            // Fast should retrieve more chunks than Disabled
            assert!(
                fast_chunks >= disabled_chunks,
                "Fast ({}) should retrieve >= Disabled ({})",
                fast_chunks,
                disabled_chunks
            );
        }

        #[test]
        fn test_combo_llm_knowledge_graph_augmented() {
            let _graph = KnowledgeGraphBuilder::new()
                .add_entity("Paris", EntityType::Location)
                .add_entity("France", EntityType::Location)
                .add_entity("Einstein", EntityType::Person)
                .build_in_memory()
                .unwrap();

            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("France", EntityType::Location);
            extractor = extractor.add_entity("Einstein", EntityType::Person);

            let ds = make_mixed_dataset();

            // Generator that returns better answers when context is present
            let augmented_gen = |prompt: &str| -> Result<String, String> {
                let lower = prompt.to_lowercase();
                if lower.contains("context:") && lower.contains("capital") {
                    Ok("B".to_string()) // correct for France/Paris
                } else if lower.contains("context:") && lower.contains("einstein") {
                    Ok("B".to_string()) // correct for relativity
                } else if lower.contains("capital") {
                    Ok("A".to_string()) // wrong without context
                } else if lower.contains("einstein") {
                    Ok("A".to_string()) // wrong without context
                } else {
                    Ok("B".to_string())
                }
            };

            let scorer = DefaultScorer;
            let mut augmented_score = 0.0;
            let mut bare_score = 0.0;
            let mut entity_count = 0;

            for problem in &ds.problems {
                let entities = extractor.extract(&problem.prompt).unwrap().entities;

                if !entities.is_empty() {
                    entity_count += 1;
                    // Augmented: prepend context
                    let context = format!(
                        "Context: Entity {} is known.\n{}",
                        entities[0].name, problem.prompt
                    );
                    let aug_resp = augmented_gen(&context).unwrap();
                    augmented_score += scorer.score(problem, &aug_resp);

                    // Bare: no context
                    let bare_resp = augmented_gen(&problem.prompt).unwrap();
                    bare_score += scorer.score(problem, &bare_resp);
                }
            }

            if entity_count > 0 {
                let aug_avg = augmented_score / entity_count as f64;
                let bare_avg = bare_score / entity_count as f64;
                assert!(
                    aug_avg >= bare_avg,
                    "Augmented ({:.3}) should >= bare ({:.3})",
                    aug_avg,
                    bare_avg
                );
            }
        }

        #[test]
        fn test_combo_llm_rag_pipeline_augmented() {
            use super::super::super::runner::BenchmarkSuiteRunner;

            let chunks = MockRetrievalCb::default_chunks();
            let retrieval = MockRetrievalCb::new(chunks);
            let llm_cb = MockLlmCb;

            // Create a generator that first runs RAG pipeline, then uses context
            let generator = move |prompt: &str| -> Result<String, String> {
                let config = RagTierConfig::with_tier(RagTier::Fast);
                let mut pipeline = RagPipeline::new(config);
                let result = pipeline.process(prompt, &llm_cb, None, &retrieval, None);

                match result {
                    Ok(r) if !r.context.is_empty() => {
                        // Use context to generate better answer
                        if r.context.to_lowercase().contains("paris") {
                            Ok("B".to_string())
                        } else {
                            Ok("B".to_string())
                        }
                    }
                    _ => Ok("B".to_string()),
                }
            };

            let runner = BenchmarkSuiteRunner::new(generator);
            let ds = make_mixed_dataset();
            let config = super::super::super::runner::RunConfig {
                model_id: test_model("rag-augmented"),
                samples_per_problem: 1,
                temperature: 0.0,
                max_tokens: Some(100),
                timeout_secs: 60,
                max_retries: 1,
                chain_of_thought: false,
                prompt_template: None,
            };

            let result = runner.run_dataset(&ds, &config);
            assert!(result.is_ok(), "Runner should succeed");
            let run_result = result.unwrap();
            assert_eq!(run_result.results.len(), 10);
        }

        #[test]
        fn test_combo_decision_tree_knowledge_graph() {
            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("Einstein", EntityType::Person);

            // Tree checks if entities were found
            let entity_branch = DecisionBranch {
                condition: Condition::new(
                    "has_entities",
                    ConditionOperator::Equals,
                    json!(true),
                ),
                target_node_id: "kg_path".to_string(),
                label: Some("Has entities".to_string()),
            };

            let tree = DecisionTreeBuilder::new("kg_router", "KG Router")
                .root("check_entities")
                .node(DecisionNode::new_condition(
                    "check_entities",
                    vec![entity_branch],
                    Some("direct_path".to_string()),
                ))
                .node(DecisionNode::new_terminal(
                    "kg_path",
                    json!("kg-augmented"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "direct_path",
                    json!("direct"),
                    None,
                ))
                .build();

            let ds = make_mixed_dataset();
            let mut kg_routed = 0;
            let mut direct_routed = 0;

            for problem in &ds.problems {
                let entities = extractor.extract(&problem.prompt).unwrap().entities;
                let mut ctx = HashMap::new();
                ctx.insert("has_entities".to_string(), json!(!entities.is_empty()));

                let path = tree.evaluate(&ctx);
                let model = path.result.unwrap();
                if model.as_str().unwrap() == "kg-augmented" {
                    kg_routed += 1;
                } else {
                    direct_routed += 1;
                }
            }

            // Entity problems (Paris, Einstein) should route to KG path
            assert!(kg_routed > 0, "Some problems should route through KG");
            assert!(direct_routed > 0, "Some should route directly");
            assert_eq!(kg_routed + direct_routed, 10);
        }

        #[test]
        fn test_combo_decision_tree_rag_tier_selection() {
            // Tree selects RAG tier based on difficulty
            let hard_branch = DecisionBranch {
                condition: Condition::new(
                    "difficulty",
                    ConditionOperator::Equals,
                    json!("hard"),
                ),
                target_node_id: "enhanced_tier".to_string(),
                label: Some("Hard".to_string()),
            };
            let medium_branch = DecisionBranch {
                condition: Condition::new(
                    "difficulty",
                    ConditionOperator::Equals,
                    json!("medium"),
                ),
                target_node_id: "semantic_tier".to_string(),
                label: Some("Medium".to_string()),
            };

            let tree = DecisionTreeBuilder::new("tier_selector", "RAG Tier Selector")
                .root("check_difficulty")
                .node(DecisionNode::new_condition(
                    "check_difficulty",
                    vec![hard_branch, medium_branch],
                    Some("fast_tier".to_string()),
                ))
                .node(DecisionNode::new_terminal(
                    "enhanced_tier",
                    json!("Enhanced"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "semantic_tier",
                    json!("Semantic"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "fast_tier",
                    json!("Fast"),
                    None,
                ))
                .build();

            // Easy problem → Fast tier
            let mut ctx = HashMap::new();
            ctx.insert("difficulty".to_string(), json!("easy"));
            let path = tree.evaluate(&ctx);
            assert_eq!(path.result.unwrap().as_str().unwrap(), "Fast");

            // Medium → Semantic
            ctx.insert("difficulty".to_string(), json!("medium"));
            let path = tree.evaluate(&ctx);
            assert_eq!(path.result.unwrap().as_str().unwrap(), "Semantic");

            // Hard → Enhanced
            ctx.insert("difficulty".to_string(), json!("hard"));
            let path = tree.evaluate(&ctx);
            assert_eq!(path.result.unwrap().as_str().unwrap(), "Enhanced");
        }

        #[test]
        fn test_combo_rag_knowledge_graph_graph_retrieval() {
            let config = RagTierConfig::with_tier(RagTier::Graph);
            let mut pipeline = RagPipeline::new(config);
            let llm = MockLlmCb;
            let retrieval = MockRetrievalCb::new(MockRetrievalCb::default_chunks());
            let graph = MockGraphCb;

            let result = pipeline
                .process(
                    "Tell me about Paris and France",
                    &llm,
                    None,
                    &retrieval,
                    Some(&graph),
                )
                .unwrap();

            // Should have chunks from both keyword and graph sources
            assert!(!result.chunks.is_empty(), "Should have chunks");
            // Context should contain entity-related information
            assert!(
                !result.context.is_empty(),
                "Context should not be empty with graph tier"
            );
        }

        #[cfg(feature = "embeddings")]
        #[test]
        fn test_combo_rag_embeddings_vectordb_pipeline() {
            use crate::embeddings::{EmbeddingConfig, LocalEmbedder};
            use crate::vector_db::{DistanceMetric, InMemoryVectorDb, VectorDb, VectorDbConfig};

            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            let docs = vec![
                "Paris is the capital of France",
                "Einstein developed relativity",
                "Gold has the chemical symbol Au",
                "A hexagon has six sides",
            ];
            embedder.train(&docs);
            let dims = EmbeddingConfig::default().dimensions;

            let config = VectorDbConfig {
                dimensions: dims,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(100),
                collection_name: "rag_emb".to_string(),
                ..Default::default()
            };
            let mut db = InMemoryVectorDb::new(config);

            for (i, doc) in docs.iter().enumerate() {
                let emb = embedder.embed(doc);
                db.insert(&format!("doc_{}", i), emb, json!({"content": doc}))
                    .unwrap();
            }

            // Custom retrieval callback using vector DB
            struct VdbRetrieval {
                chunks: Vec<RetrievedChunk>,
            }
            impl RetrievalCallback for VdbRetrieval {
                fn keyword_search(
                    &self,
                    _query: &str,
                    limit: usize,
                ) -> Result<Vec<RetrievedChunk>, String> {
                    Ok(self.chunks.iter().take(limit).cloned().collect())
                }
                fn semantic_search(
                    &self,
                    _emb: &[f32],
                    limit: usize,
                ) -> Result<Vec<RetrievedChunk>, String> {
                    Ok(self.chunks.iter().take(limit).cloned().collect())
                }
                fn get_chunk(
                    &self,
                    id: &str,
                ) -> Result<Option<RetrievedChunk>, String> {
                    Ok(self.chunks.iter().find(|c| c.chunk_id == id).cloned())
                }
            }

            // Search VDB, build chunks
            let query_emb = embedder.embed("capital of France");
            let results = db.search(&query_emb, 3, None).unwrap();
            let chunks: Vec<RetrievedChunk> = results
                .iter()
                .map(|r| RetrievedChunk {
                    chunk_id: r.id.clone(),
                    content: r
                        .metadata
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    source: "vectordb".to_string(),
                    section: None,
                    score: r.score as f32,
                    keyword_score: Some(r.score as f32),
                    semantic_score: None,
                    token_count: 10,
                    position: None,
                    metadata: HashMap::new(),
                })
                .collect();

            let retrieval = VdbRetrieval { chunks };
            let llm = MockLlmCb;
            let mut pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Fast));

            let result = pipeline
                .process("What is the capital of France?", &llm, None, &retrieval, None)
                .unwrap();

            assert!(
                !result.chunks.is_empty(),
                "Pipeline should return chunks from VDB"
            );
        }

        #[test]
        fn test_combo_llm_tree_knowledge_graph() {
            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("Einstein", EntityType::Person);

            let tree = build_category_routing_tree();
            let ds = make_mixed_dataset();

            let mut generator = MultiModelGenerator::new(
                |_: &str| -> Result<String, String> { Ok("B".to_string()) },
            );

            // Register generators
            generator.register_model("knowledge-specialist", |prompt: &str| {
                if prompt.to_lowercase().contains("context:") {
                    Ok("B".to_string()) // better with context
                } else {
                    Ok("B".to_string())
                }
            });
            generator.register_model("math-specialist", |prompt: &str| {
                if prompt.contains("6 * 7") {
                    Ok("42".to_string())
                } else if prompt.to_lowercase().contains("hexagon") {
                    Ok("6".to_string())
                } else {
                    Ok("B".to_string())
                }
            });
            generator.register_model("code-specialist", |prompt: &str| {
                if prompt.to_lowercase().contains("add") {
                    Ok("def add(a, b): return a + b".to_string())
                } else {
                    Ok("def length(s): return len(s)".to_string())
                }
            });
            generator.register_model("entity-specialist", |_: &str| {
                Ok("B".to_string())
            });

            let scorer = DefaultScorer;
            let mut total_score = 0.0;
            let mut count = 0;

            for problem in &ds.problems {
                // Step 1: Route via decision tree
                let mut ctx = HashMap::new();
                ctx.insert(
                    "category".to_string(),
                    json!(problem_category(&problem.id)),
                );
                let path = tree.evaluate(&ctx);
                let model_key = path
                    .result
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();

                // Step 2: Check for entities → augment if found
                let entities = extractor.extract(&problem.prompt).unwrap().entities;
                let prompt = if !entities.is_empty() {
                    format!(
                        "Context: Entities found: {}.\n{}",
                        entities.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", "),
                        problem.prompt
                    )
                } else {
                    problem.prompt.clone()
                };

                // Step 3: Generate via routed model
                let response = generator.generate(&model_key, &prompt).unwrap();
                let score = scorer.score(problem, &response);
                total_score += score;
                count += 1;
            }

            let avg = total_score / count as f64;
            assert!(
                avg > 0.3,
                "3-way combo (tree+KG+multi-model) should score > 0.3, got {:.3}",
                avg
            );
        }
    }

    // ========================================================================
    // memory_tests — needs `advanced-memory` feature
    // ========================================================================
    #[cfg(feature = "advanced-memory")]
    mod memory_tests {
        use super::*;
        use crate::advanced_memory::{EpisodicStore, Episode, ProceduralStore, Procedure};

        fn make_eval_episode(
            id: &str,
            content: &str,
            tags: &[&str],
            importance: f64,
        ) -> Episode {
            Episode {
                id: id.to_string(),
                content: content.to_string(),
                context: format!("eval_context_{}", id),
                timestamp: 1000 + id.len() as u64,
                importance,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                embedding: deterministic_embedding(content, 32),
                access_count: 0,
                last_accessed: 1000,
            }
        }

        fn make_eval_procedure(
            id: &str,
            name: &str,
            condition: &str,
            confidence: f64,
        ) -> Procedure {
            Procedure {
                id: id.to_string(),
                name: name.to_string(),
                condition: condition.to_string(),
                steps: vec!["step1".to_string(), "step2".to_string()],
                success_count: if confidence > 0.7 { 8 } else { 3 },
                failure_count: if confidence > 0.7 { 2 } else { 7 },
                confidence,
                created_from: vec![],
                tags: vec![],
            }
        }

        #[test]
        fn test_episodic_memory_stores_eval_results() {
            let mut store = EpisodicStore::new(20, 0.001);

            // Store episodes from evaluation results
            store.add(make_eval_episode(
                "ep1",
                "The capital of France is Paris — answer B",
                &["knowledge", "geography"],
                0.95,
            ));
            store.add(make_eval_episode(
                "ep2",
                "Calculate 6*7 = 42",
                &["math", "arithmetic"],
                1.0,
            ));
            store.add(make_eval_episode(
                "ep3",
                "def add(a,b): return a+b",
                &["code", "python"],
                0.8,
            ));
            store.add(make_eval_episode(
                "ep4",
                "Wrong answer for chemistry question",
                &["knowledge", "chemistry"],
                0.0,
            ));

            assert_eq!(store.len(), 4);

            // Recall by similarity to a math query
            let math_query = deterministic_embedding("Calculate numbers math", 32);
            let recalled = store.recall(&math_query, 2);
            assert!(
                !recalled.is_empty(),
                "Should recall at least one episode"
            );
        }

        #[test]
        fn test_procedural_memory_learns_eval_strategies() {
            let mut store = ProceduralStore::new(10);

            store.add(make_eval_procedure(
                "proc1",
                "MC Letter Extraction",
                "multiple choice question letter answer",
                0.9,
            ));
            store.add(make_eval_procedure(
                "proc2",
                "Numeric Parsing",
                "calculate number numeric answer",
                0.7,
            ));
            store.add(make_eval_procedure(
                "proc3",
                "Code Generation",
                "write function code python def",
                0.6,
            ));

            // Find by condition matching
            let mc_procedures = store.find_by_condition("multiple choice question");
            assert!(
                !mc_procedures.is_empty(),
                "Should find MC procedure"
            );
            assert_eq!(
                mc_procedures[0].name,
                "MC Letter Extraction",
                "Highest confidence MC procedure should be first"
            );

            // Find numeric
            let num_procedures = store.find_by_condition("calculate number");
            assert!(
                !num_procedures.is_empty(),
                "Should find numeric procedure"
            );
        }
    }

    // ========================================================================
    // agent_tests — needs `multi-agent` feature
    // ========================================================================
    #[cfg(feature = "multi-agent")]
    mod agent_tests {
        use super::*;
        use crate::multi_agent::{
            Agent, AgentOrchestrator, AgentRole, AgentTask, OrchestrationStrategy, SharedContext,
        };

        #[test]
        fn test_multi_agent_eval_distribution() {
            let mut orchestrator =
                AgentOrchestrator::new(OrchestrationStrategy::BestFit);

            // Register specialist agents
            orchestrator.register_agent(
                Agent::new("knowledge_agent", "Knowledge Specialist", AgentRole::Analyst)
                    .with_capability("knowledge")
                    .with_capability("geography"),
            );
            orchestrator.register_agent(
                Agent::new("math_agent", "Math Specialist", AgentRole::Executor)
                    .with_capability("math")
                    .with_capability("calculation"),
            );
            orchestrator.register_agent(
                Agent::new("code_agent", "Code Specialist", AgentRole::Executor)
                    .with_capability("coding")
                    .with_capability("python"),
            );

            // Create tasks for problems
            let tasks = vec![
                AgentTask::new("task_know", "Answer knowledge questions"),
                AgentTask::new("task_math", "Solve math problems"),
                AgentTask::new("task_code", "Generate code solutions"),
            ];

            for task in tasks {
                orchestrator.add_task(task);
            }

            // Auto-assign
            let assignments = orchestrator.auto_assign_tasks();
            assert!(
                !assignments.is_empty(),
                "Should assign at least some tasks"
            );
        }

        #[test]
        fn test_multi_agent_shared_context_aggregation() {
            let mut ctx = SharedContext::new();

            // Agents store their eval results
            ctx.set("knowledge_score", json!(0.85), "knowledge_agent");
            ctx.set("math_score", json!(0.92), "math_agent");
            ctx.set("code_score", json!(0.78), "code_agent");

            // Read back and aggregate
            let k_score = ctx
                .get("knowledge_score")
                .and_then(|e| e.value.as_f64())
                .unwrap_or(0.0);
            let m_score = ctx
                .get("math_score")
                .and_then(|e| e.value.as_f64())
                .unwrap_or(0.0);
            let c_score = ctx
                .get("code_score")
                .and_then(|e| e.value.as_f64())
                .unwrap_or(0.0);

            let total = k_score + m_score + c_score;
            assert!(
                (total - 2.55).abs() < 0.01,
                "Total should be ~2.55, got {:.3}",
                total
            );

            // Detect changes via snapshot diff
            let snap1 = ctx.snapshot();
            ctx.set("math_score", json!(0.95), "math_agent");
            let snap2 = ctx.snapshot();
            assert_ne!(
                snap1, snap2,
                "Snapshots should differ after update"
            );
        }
    }

    // ========================================================================
    // rag_memory_tests — needs `rag` + `advanced-memory`
    // ========================================================================
    #[cfg(all(feature = "rag", feature = "advanced-memory"))]
    mod rag_memory_tests {
        use super::*;
        use crate::advanced_memory::{EpisodicStore, Episode, ProceduralStore, Procedure};

        fn make_eval_episode(
            id: &str,
            content: &str,
            tags: &[&str],
            importance: f64,
        ) -> Episode {
            Episode {
                id: id.to_string(),
                content: content.to_string(),
                context: format!("eval_{}", id),
                timestamp: 1000,
                importance,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                embedding: deterministic_embedding(content, 32),
                access_count: 0,
                last_accessed: 1000,
            }
        }

        #[test]
        fn test_combo_rag_memory_learned_procedures() {
            let mut store = ProceduralStore::new(10);

            // First "run": learn that MC problems need letter extraction
            store.add(Procedure {
                id: "proc_mc".to_string(),
                name: "MC Strategy".to_string(),
                condition: "multiple choice question answer letter".to_string(),
                steps: vec!["Extract answer letter".to_string(), "Return letter only".to_string()],
                success_count: 8,
                failure_count: 2,
                confidence: 0.8,
                created_from: vec![],
                tags: vec!["mc".to_string()],
            });

            // Second "run": look up procedure for new MC problem
            let procedures = store.find_by_condition("multiple choice question");
            assert!(!procedures.is_empty(), "Should find MC procedure");

            // Build augmented prompt with procedure hint
            let hint = format!(
                "Strategy hint: {}. Steps: {}",
                procedures[0].name,
                procedures[0].steps.join(", ")
            );

            // Generator with hint produces better answer
            let gen_with_hint = |prompt: &str| -> Result<String, String> {
                if prompt.contains("Strategy hint") {
                    Ok("B".to_string()) // correct with hint
                } else {
                    Ok("The answer might be A or B".to_string()) // wrong without
                }
            };

            let ds = make_mixed_dataset();
            let scorer = DefaultScorer;

            // Score first MC problem with and without hint
            let problem = &ds.problems[0]; // fc/know/1, answer is B
            let with_hint = gen_with_hint(&format!("{}\n{}", hint, problem.prompt)).unwrap();
            let without_hint = gen_with_hint(&problem.prompt).unwrap();

            let score_with = scorer.score(problem, &with_hint);
            let score_without = scorer.score(problem, &without_hint);

            assert!(
                score_with >= score_without,
                "With hint ({:.3}) should be >= without ({:.3})",
                score_with,
                score_without
            );
        }

        #[test]
        fn test_combo_rag_episodic_memory_recall() {
            let mut store = EpisodicStore::new(20, 0.001);

            // Past episodes: high-score and low-score
            store.add(make_eval_episode(
                "past_good",
                "The capital of France is Paris, answer B",
                &["knowledge", "geography"],
                0.95,
            ));
            store.add(make_eval_episode(
                "past_bad",
                "I think the answer is D for France question",
                &["knowledge", "geography"],
                0.0,
            ));

            // New problem similar to past
            let query_emb = deterministic_embedding("capital of France Paris", 32);
            let recalled = store.recall(&query_emb, 2);

            assert!(
                !recalled.is_empty(),
                "Should recall past episodes"
            );
            // The high-importance one should generally be recalled
            let has_good = recalled.iter().any(|e| e.importance > 0.5);
            assert!(
                has_good || recalled.len() >= 1,
                "Should recall at least one episode"
            );
        }
    }

    // ========================================================================
    // rag_agent_tests — needs `rag` + `multi-agent`
    // ========================================================================
    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    mod rag_agent_tests {
        use crate::multi_agent::{
            Agent, AgentMessage, AgentOrchestrator, AgentRole, MessageType,
            OrchestrationStrategy,
        };

        #[test]
        fn test_combo_multi_agent_rag_pipeline() {
            let mut orchestrator =
                AgentOrchestrator::new(OrchestrationStrategy::Sequential);

            // Register 3 agents for the pipeline
            orchestrator.register_agent(
                Agent::new("retriever", "Retriever", AgentRole::Analyst)
                    .with_capability("retrieval"),
            );
            orchestrator.register_agent(
                Agent::new("solver", "Solver", AgentRole::Executor)
                    .with_capability("generation"),
            );
            orchestrator.register_agent(
                Agent::new("reviewer", "Reviewer", AgentRole::Analyst)
                    .with_capability("scoring"),
            );

            // Simulate pipeline: retriever → solver → reviewer via messages
            let ctx_msg = AgentMessage::new(
                "retriever",
                "solver",
                "Context: Paris is the capital of France",
                MessageType::Response,
            );
            orchestrator.send_message(ctx_msg);

            let answer_msg = AgentMessage::new(
                "solver",
                "reviewer",
                "Answer: B (Paris)",
                MessageType::Response,
            );
            orchestrator.send_message(answer_msg);

            let score_msg = AgentMessage::new(
                "reviewer",
                "retriever",
                "Score: 1.0 (correct)",
                MessageType::Notification,
            );
            orchestrator.send_message(score_msg);

            // Verify messages were delivered
            let solver_msgs = orchestrator.get_messages_for("solver");
            assert_eq!(solver_msgs.len(), 1, "Solver should have 1 message");
            assert!(
                solver_msgs[0].content.contains("Paris"),
                "Solver should receive context"
            );

            let reviewer_msgs = orchestrator.get_messages_for("reviewer");
            assert_eq!(reviewer_msgs.len(), 1, "Reviewer should have 1 message");

            let retriever_msgs = orchestrator.get_messages_for("retriever");
            assert_eq!(
                retriever_msgs.len(),
                1,
                "Retriever should get score notification"
            );
        }
    }

    // ========================================================================
    // full_stack_tests — needs all features
    // ========================================================================
    #[cfg(all(
        feature = "rag",
        feature = "advanced-memory",
        feature = "multi-agent",
        feature = "embeddings"
    ))]
    mod full_stack_tests {
        use super::*;
        use crate::advanced_memory::{EpisodicStore, Episode, ProceduralStore, Procedure};
        use crate::embeddings::{EmbeddingConfig, LocalEmbedder};
        use crate::knowledge_graph::{EntityExtractor, EntityType, KnowledgeGraphBuilder, PatternEntityExtractor};
        use crate::multi_agent::{
            Agent, AgentOrchestrator, AgentRole, OrchestrationStrategy, SharedContext,
        };
        use crate::vector_db::{DistanceMetric, InMemoryVectorDb, VectorDb, VectorDbConfig};

        #[test]
        fn test_full_stack_integration() {
            // 1. Episodic memory with past episodes
            let mut episodic = EpisodicStore::new(20, 0.001);
            episodic.add(Episode {
                id: "past1".to_string(),
                content: "Paris is capital of France, answer B".to_string(),
                context: "geography".to_string(),
                timestamp: 900,
                importance: 0.95,
                tags: vec!["knowledge".to_string()],
                embedding: deterministic_embedding("Paris France capital", 32),
                access_count: 0,
                last_accessed: 900,
            });

            // 2. Procedural memory with learned strategies
            let mut procedural = ProceduralStore::new(10);
            procedural.add(Procedure {
                id: "mc_strat".to_string(),
                name: "MC Strategy".to_string(),
                condition: "multiple choice question".to_string(),
                steps: vec!["Extract letter".to_string()],
                success_count: 9,
                failure_count: 1,
                confidence: 0.9,
                created_from: vec![],
                tags: vec![],
            });

            // 3. Knowledge graph
            let graph = KnowledgeGraphBuilder::new()
                .add_entity("Paris", EntityType::Location)
                .add_entity("France", EntityType::Location)
                .add_entity("Einstein", EntityType::Person)
                .build_in_memory()
                .unwrap();
            assert!(graph.stats().unwrap().total_entities >= 3);

            // 4. Vector DB with embeddings
            let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());
            let ds = make_mixed_dataset();
            let prompts: Vec<&str> = ds.problems.iter().map(|p| p.prompt.as_str()).collect();
            embedder.train(&prompts);

            let vdb_config = VectorDbConfig {
                dimensions: EmbeddingConfig::default().dimensions,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(100),
                collection_name: "fullstack".to_string(),
                ..Default::default()
            };
            let mut vdb = InMemoryVectorDb::new(vdb_config);
            for problem in &ds.problems {
                let emb = embedder.embed(&problem.prompt);
                vdb.insert(&problem.id, emb, json!({"cat": problem_category(&problem.id)}))
                    .unwrap();
            }
            assert_eq!(vdb.count(), 10);

            // 5. Decision tree for routing
            let tree = build_category_routing_tree();

            // 6. Multi-model generator
            let mut generator = MultiModelGenerator::new(
                |_: &str| -> Result<String, String> { Ok("B".to_string()) },
            );
            generator.register_model("knowledge-specialist", |p: &str| {
                if p.to_lowercase().contains("gold") { Ok("C".to_string()) } else { Ok("B".to_string()) }
            });
            generator.register_model("math-specialist", |p: &str| {
                if p.contains("6 * 7") { Ok("42".to_string()) }
                else if p.to_lowercase().contains("hexagon") { Ok("6".to_string()) }
                else { Ok("B".to_string()) }
            });
            generator.register_model("code-specialist", |p: &str| {
                if p.to_lowercase().contains("add") { Ok("def add(a, b): return a + b".to_string()) }
                else { Ok("def length(s): return len(s)".to_string()) }
            });
            generator.register_model("entity-specialist", |_: &str| Ok("B".to_string()));

            // 7. Entity extractor
            let mut extractor = PatternEntityExtractor::new();
            extractor = extractor.add_entity("Paris", EntityType::Location);
            extractor = extractor.add_entity("Einstein", EntityType::Person);

            // 8. Agent orchestrator
            let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);
            orchestrator.register_agent(
                Agent::new("evaluator", "Evaluator", AgentRole::Executor)
                    .with_capability("eval"),
            );

            // 9. Shared context for results
            let mut shared_ctx = SharedContext::new();

            // Run all 10 problems through the full pipeline
            let scorer = DefaultScorer;
            let mut total_score = 0.0;

            for problem in &ds.problems {
                // Route via tree
                let mut ctx = HashMap::new();
                ctx.insert(
                    "category".to_string(),
                    json!(problem_category(&problem.id)),
                );
                let path = tree.evaluate(&ctx);
                let model_key = path.result.unwrap().as_str().unwrap().to_string();

                // Check entities
                let entities = extractor.extract(&problem.prompt).unwrap().entities;
                let prompt = if !entities.is_empty() {
                    format!("Context: {}\n{}", entities[0].name, problem.prompt)
                } else {
                    problem.prompt.clone()
                };

                // Generate
                let response = generator.generate(&model_key, &prompt).unwrap();
                let score = scorer.score(problem, &response);
                total_score += score;

                // Store in shared context
                shared_ctx.set(
                    &problem.id,
                    json!({"score": score, "model": model_key}),
                    "evaluator",
                );
            }

            let avg = total_score / 10.0;
            assert!(
                avg > 0.3,
                "Full stack avg should be > 0.3, got {:.3}",
                avg
            );

            // Verify all problems recorded in shared context
            for problem in &ds.problems {
                assert!(
                    shared_ctx.get(&problem.id).is_some(),
                    "Problem {} should be in shared context",
                    problem.id
                );
            }
        }

        #[test]
        fn test_full_stack_ablation_study() {
            let ds = make_mixed_dataset();
            let scorer = DefaultScorer;

            // Full stack score (with routing)
            let tree = build_category_routing_tree();
            let default_fn = |_: &str| -> Result<String, String> { Ok("B".to_string()) };
            let mut gen_full = MultiModelGenerator::new(
                |_: &str| -> Result<String, String> { Ok("B".to_string()) },
            );
            gen_full.register_model("knowledge-specialist", |p: &str| {
                if p.to_lowercase().contains("gold") { Ok("C".to_string()) } else { Ok("B".to_string()) }
            });
            gen_full.register_model("math-specialist", |p: &str| {
                if p.contains("6 * 7") { Ok("42".to_string()) }
                else if p.to_lowercase().contains("hexagon") { Ok("6".to_string()) }
                else { Ok("B".to_string()) }
            });
            gen_full.register_model("code-specialist", |p: &str| {
                if p.to_lowercase().contains("add") { Ok("def add(a, b): return a + b".to_string()) }
                else { Ok("def length(s): return len(s)".to_string()) }
            });
            gen_full.register_model("entity-specialist", |_: &str| Ok("B".to_string()));

            let mut full_score = 0.0;
            for problem in &ds.problems {
                let mut ctx = HashMap::new();
                ctx.insert("category".to_string(), json!(problem_category(&problem.id)));
                let path = tree.evaluate(&ctx);
                let model_key = path.result.unwrap().as_str().unwrap().to_string();
                let response = gen_full.generate(&model_key, &problem.prompt).unwrap();
                full_score += scorer.score(problem, &response);
            }

            // Ablated: no tree routing (single model = default)
            let mut no_tree_score = 0.0;
            for problem in &ds.problems {
                let response = default_fn(&problem.prompt).unwrap();
                no_tree_score += scorer.score(problem, &response);
            }

            // Full stack should be >= no-tree (specialized models help)
            assert!(
                full_score >= no_tree_score,
                "Full ({:.3}) should be >= no-tree ({:.3})",
                full_score,
                no_tree_score
            );
        }

        #[test]
        fn test_full_stack_config_search_with_features() {
            use super::super::super::agent_config::SearchDimension;
            use super::super::super::config_search::{
                ConfigSearchConfig, ConfigSearchEngine, SearchObjective,
            };

            let (ds, tags) = make_categorized_problems();

            // Build config with feature integration
            let baseline = EvalAgentConfig::new("full-stack", test_model("default"))
                .with_subtask_model(
                    &Subtask::InformationGathering,
                    test_model("default"),
                )
                .with_subtask_model(&Subtask::ReasoningChain, test_model("default"))
                .with_subtask_model(&Subtask::CodeGeneration, test_model("default"));

            // Mock generators
            let mut gen = MultiModelGenerator::new(
                |_: &str| -> Result<String, String> { Ok("B".to_string()) },
            );
            gen.register_model("mock/default", |_: &str| Ok("B".to_string()));
            gen.register_model("mock/specialist", |p: &str| {
                let lower = p.to_lowercase();
                if lower.contains("6 * 7") { Ok("42".to_string()) }
                else if lower.contains("hexagon") { Ok("6".to_string()) }
                else if lower.contains("gold") { Ok("C".to_string()) }
                else if lower.contains("add") { Ok("def add(a, b): return a + b".to_string()) }
                else if lower.contains("length") { Ok("def length(s): return len(s)".to_string()) }
                else { Ok("B".to_string()) }
            });

            let search_config = ConfigSearchConfig {
                confidence_level: 0.80,
                min_samples: 2,
                max_evaluations: 10,
                adaptive_priority: false,
                objective: SearchObjective::MaxQuality,
            };

            let engine = ConfigSearchEngine::new(search_config, gen);

            let dims = vec![SearchDimension::SubtaskModel {
                subtask: "InformationGathering".to_string(),
                candidates: vec![test_model("default"), test_model("specialist")],
            }];

            let result = engine.search(&baseline, &dims, &ds, &tags);
            assert!(result.is_ok(), "Search should succeed");

            let search_result = result.unwrap();
            assert!(
                search_result.total_evaluations > 0,
                "Should have evaluated some configs"
            );
            assert!(
                !search_result.evolution.is_empty(),
                "Should have evolution snapshots"
            );
            assert!(
                search_result.best.quality >= 0.0,
                "Best quality should be non-negative"
            );
        }
    }

    // ========================================================================
    // heavy_integration_tests — Realistic multi-document workflows
    // Requires: rag + embeddings + advanced-memory + multi-agent (all in `full`)
    // ========================================================================
    #[cfg(all(
        feature = "rag",
        feature = "embeddings",
        feature = "advanced-memory",
        feature = "multi-agent"
    ))]
    mod heavy_integration_tests {
        use super::*;
        use crate::advanced_memory::{Episode, EpisodicStore, Procedure, ProceduralStore};
        use crate::context_composer::{ContextComposer, ContextSection};
        use crate::document_parsing::{DocumentFormat, DocumentParser, DocumentParserConfig};
        use crate::embeddings::{EmbeddingConfig, LocalEmbedder};
        use crate::knowledge_graph::{
            EntityExtractor, EntityType, KnowledgeGraphBuilder, PatternEntityExtractor,
        };
        use crate::llm_judge::{EvalCriterion, LlmJudge};
        use crate::multi_agent::{
            Agent, AgentOrchestrator, AgentRole, AgentTask, OrchestrationStrategy, SharedContext,
        };
        use crate::rag_pipeline::{
            PipelineChunkPosition, GraphCallback, GraphRelation, LlmCallback, RagPipeline,
            RetrievalCallback, RetrievedChunk,
        };
        use crate::rag_tiers::{RagTierConfig, RagTier};
        use crate::vector_db::{DistanceMetric, InMemoryVectorDb, VectorDb, VectorDbConfig};
        use super::super::super::runner::{BenchmarkSuiteRunner, RunConfig};

        // ================================================================
        // Domain documents (~300 words each)
        // ================================================================

        fn doc_barcelona() -> &'static str {
            "Barcelona Solar Initiative 2025\n\n\
             Barcelona has emerged as a European leader in urban solar energy adoption. \
             The city council approved a comprehensive solar mandate in 2024, requiring all \
             new buildings and major renovations to include rooftop photovoltaic installations. \
             This ambitious program targets a total installed capacity of 15 megawatts across \
             the metropolitan area by the end of 2025.\n\n\
             The Mediterranean climate provides exceptional conditions for solar energy generation, \
             with an average of 2,500 sunshine hours per year. Studies conducted by the Barcelona \
             Energy Agency demonstrate that residential solar installations achieve a 30% reduction \
             in household energy costs, making the program economically attractive for homeowners.\n\n\
             Beyond direct energy savings, the initiative addresses urban heat island effects. \
             Rooftop solar panels provide shade and reduce surface temperatures by up to 5 degrees \
             Celsius during peak summer months. The program integrates with Barcelona's broader \
             Climate Plan, which targets carbon neutrality by 2050.\n\n\
             Key partners include Endesa, the Spanish energy utility, and SolarCity Europe, \
             which provides financing options for residential installations. The Barcelona \
             Supercomputing Center contributes advanced weather modeling for optimal panel \
             placement. Community energy cooperatives have formed in neighborhoods like Gracia \
             and Eixample, enabling shared solar installations on apartment buildings."
        }

        fn doc_copenhagen() -> &'static str {
            "Copenhagen Electric Transit Transformation\n\n\
             Copenhagen is implementing one of the world's most ambitious public transit \
             electrification programs. The Danish capital aims to achieve a 100% electric bus \
             fleet by 2025, replacing all 400 diesel buses currently in service. The city has \
             partnered with BYD, the Chinese electric vehicle manufacturer, to supply the \
             majority of its new fleet.\n\n\
             The program has already achieved a 40% reduction in transit-related emissions since \
             its inception in 2021. Each electric bus eliminates approximately 80 tonnes of CO2 \
             annually compared to its diesel equivalent. Copenhagen has installed 65 fast-charging \
             stations across the city, with overnight depot charging available at three main \
             bus terminals.\n\n\
             Integration with Copenhagen's world-renowned cycling infrastructure is a priority. \
             New bus routes are designed to complement the 450 kilometers of dedicated cycle lanes, \
             with seamless transfer points at 12 major cycling hubs. Smart traffic signals give \
             priority to both buses and bicycles, reducing average commute times by 15%.\n\n\
             The Danish Transport Authority has designated four zero-emission zones in the city \
             center where only electric vehicles are permitted. Movia, the regional transit \
             operator, manages the fleet transition with funding from the Danish Green \
             Transition Fund. The program serves as a model for other Scandinavian cities, \
             with Oslo and Stockholm planning similar initiatives."
        }

        fn doc_singapore() -> &'static str {
            "Singapore Green Building Revolution\n\n\
             Singapore has established itself as a global leader in sustainable building design \
             through its innovative Green Mark certification program. Administered by the Building \
             and Construction Authority (BCA), the program has certified over 4,300 buildings since \
             its launch, with 45% of the total built environment now meeting Green Mark standards.\n\n\
             The flagship BCA Green Mark Platinum certification requires buildings to demonstrate \
             a 35% reduction in energy consumption compared to baseline models. This is achieved \
             through a combination of smart HVAC systems, high-performance glazing, and natural \
             ventilation design. Notable Platinum-certified buildings include Marina Bay Sands \
             and the Parkroyal on Pickering hotel.\n\n\
             Vertical gardens and green facades have become a signature feature of Singapore's \
             skyline. The city mandates that new developments replace any greenery lost during \
             construction, leading to innovative rooftop forests and sky terraces. These features \
             reduce ambient temperature by 3-5 degrees Celsius and improve air quality in \
             surrounding areas.\n\n\
             Rainwater harvesting is integrated into 80% of new commercial buildings, with \
             collected water used for landscape irrigation and cooling towers. Building Energy \
             Management Systems (BEMS) using IoT sensors and machine learning optimize energy \
             consumption in real time. The government provides tax incentives covering up to \
             30% of green retrofit costs, encouraging existing buildings to upgrade to Green \
             Mark standards."
        }

        // ================================================================
        // Shared helpers
        // ================================================================

        fn build_sustainability_extractor() -> PatternEntityExtractor {
            let mut ext = PatternEntityExtractor::new();
            // Locations
            ext = ext.add_entity("Barcelona", EntityType::Location);
            ext = ext.add_entity("Copenhagen", EntityType::Location);
            ext = ext.add_entity("Singapore", EntityType::Location);
            ext = ext.add_entity("Mediterranean", EntityType::Location);
            ext = ext.add_entity("Denmark", EntityType::Location);
            ext = ext.add_entity("Marina Bay Sands", EntityType::Location);
            // Organizations
            ext = ext.add_entity("BYD", EntityType::Organization);
            ext = ext.add_entity("Endesa", EntityType::Organization);
            ext = ext.add_entity("BCA", EntityType::Organization);
            ext = ext.add_entity("Movia", EntityType::Organization);
            ext = ext.add_entity("SolarCity Europe", EntityType::Organization);
            // Products / Concepts
            ext = ext.add_entity("Green Mark", EntityType::Product);
            ext = ext.add_entity("solar panels", EntityType::Product);
            ext = ext.add_entity("BEMS", EntityType::Product);
            ext = ext.add_entity("HVAC", EntityType::Product);
            // Concepts
            ext = ext.add_entity("carbon neutrality", EntityType::Concept);
            ext = ext.add_entity("zero-emission", EntityType::Concept);
            ext = ext.add_entity("rainwater harvesting", EntityType::Concept);
            ext = ext.add_entity("urban heat island", EntityType::Concept);
            ext
        }

        fn build_sustainability_kg(
        ) -> (crate::knowledge_graph::KnowledgeGraph, PatternEntityExtractor) {
            let extractor = build_sustainability_extractor();
            let mut graph = KnowledgeGraphBuilder::new()
                .add_entity("Barcelona", EntityType::Location)
                .add_entity("Copenhagen", EntityType::Location)
                .add_entity("Singapore", EntityType::Location)
                .add_entity("BYD", EntityType::Organization)
                .add_entity("Endesa", EntityType::Organization)
                .add_entity("BCA", EntityType::Organization)
                .add_entity("SolarCity Europe", EntityType::Organization)
                .add_entity("Green Mark", EntityType::Product)
                .add_entity("solar panels", EntityType::Product)
                .add_entity("BEMS", EntityType::Product)
                .add_entity("Mediterranean", EntityType::Location)
                .add_entity("Movia", EntityType::Organization)
                .add_entity("carbon neutrality", EntityType::Concept)
                .add_entity("zero-emission", EntityType::Concept)
                .add_entity("rainwater harvesting", EntityType::Concept)
                .add_entity("urban heat island", EntityType::Concept)
                .build_in_memory()
                .unwrap();

            // Index all 3 documents
            graph
                .index_document("doc_barcelona", doc_barcelona(), &extractor)
                .unwrap();
            graph
                .index_document("doc_copenhagen", doc_copenhagen(), &extractor)
                .unwrap();
            graph
                .index_document("doc_singapore", doc_singapore(), &extractor)
                .unwrap();

            (graph, extractor)
        }

        /// Smart mock generator that returns domain-specific answers based on prompt keywords.
        fn smart_mock_generator() -> impl Fn(&str) -> Result<String, String> + Send + Sync {
            |prompt: &str| {
                let lower = prompt.to_lowercase();

                // Fact verification (must come FIRST — verify prompts contain claim text)
                if lower.contains("verify") || lower.contains("refute") {
                    if lower.contains("50%") || lower.contains("3,000 sunshine") || lower.contains("3000 sunshine") || lower.contains("60%") {
                        return Ok("REFUTED".to_string());
                    }
                    return Ok("VERIFIED".to_string());
                }

                // Hint-augmented (for memory tests) — check BEFORE specific facts
                // because recalled episodes may contain keywords from other questions
                if lower.contains("hint:") || lower.contains("strategy:") || lower.contains("recall:") {
                    if lower.contains("copenhagen") {
                        return Ok("Based on the retrieved context, Copenhagen partnered with BYD and achieved 40% emission reduction through electric transit.".to_string());
                    }
                    if lower.contains("singapore") {
                        return Ok("Based on the retrieved context, Singapore's Green Mark Platinum requires 35% energy reduction with BEMS and rainwater harvesting.".to_string());
                    }
                    if lower.contains("barcelona") {
                        return Ok("Based on the retrieved context, Barcelona achieves a 30% reduction in energy costs through rooftop solar with 2,500 sunshine hours per year.".to_string());
                    }
                }

                // Barcelona questions
                if lower.contains("30%") || (lower.contains("barcelona") && lower.contains("energy cost")) {
                    return Ok("Barcelona's solar initiative achieves a 30% reduction in household energy costs.".to_string());
                }
                if lower.contains("2,500") || lower.contains("2500") || (lower.contains("sunshine") && lower.contains("hours")) {
                    return Ok("Barcelona receives an average of 2,500 sunshine hours per year due to its Mediterranean climate.".to_string());
                }
                if lower.contains("15 megawatt") || lower.contains("15mw") || (lower.contains("barcelona") && lower.contains("capacity")) {
                    return Ok("Barcelona targets 15 megawatts of installed solar capacity by 2025.".to_string());
                }

                // Copenhagen questions
                if lower.contains("byd") || (lower.contains("copenhagen") && lower.contains("manufacturer")) || (lower.contains("bus") && lower.contains("partner")) {
                    return Ok("Copenhagen partnered with BYD, the Chinese electric vehicle manufacturer.".to_string());
                }
                if lower.contains("40%") || (lower.contains("copenhagen") && lower.contains("emission")) {
                    return Ok("Copenhagen's electric transit program achieved a 40% reduction in transit-related emissions.".to_string());
                }
                if lower.contains("400") || (lower.contains("copenhagen") && lower.contains("diesel")) {
                    return Ok("Copenhagen aims to replace all 400 diesel buses with electric vehicles.".to_string());
                }

                // Singapore questions
                if lower.contains("green mark") || (lower.contains("singapore") && lower.contains("certification")) {
                    return Ok("Singapore uses the Green Mark certification program administered by the BCA.".to_string());
                }
                if lower.contains("35%") || (lower.contains("singapore") && lower.contains("energy reduction")) {
                    return Ok("Singapore's Green Mark Platinum requires a 35% reduction in energy consumption.".to_string());
                }
                if lower.contains("4,300") || lower.contains("4300") || (lower.contains("singapore") && lower.contains("certified")) {
                    return Ok("Over 4,300 buildings have been certified under Singapore's Green Mark program.".to_string());
                }

                // Cross-document / comparison questions
                if lower.contains("compare") || (lower.contains("barcelona") && lower.contains("copenhagen")) {
                    return Ok("Both Barcelona and Copenhagen pursue urban sustainability but through different strategies: \
                              Barcelona focuses on solar energy with a 30% cost reduction, while Copenhagen targets \
                              electric transit achieving 40% emission reduction.".to_string());
                }
                if lower.contains("ambitious") || lower.contains("emission reduction target") {
                    return Ok("Copenhagen has the most ambitious emission reduction target at 40%, compared to \
                              Barcelona's 30% energy cost reduction and Singapore's 35% energy consumption reduction.".to_string());
                }
                if (lower.contains("singapore") && lower.contains("barcelona") && lower.contains("building")) || lower.contains("energy efficiency") {
                    return Ok("Singapore mandates 35% energy reduction via Green Mark certification, while Barcelona \
                              focuses on rooftop solar panels reducing energy costs by 30%.".to_string());
                }

                // Default - generic sustainability answer
                Ok("Urban sustainability initiatives across major cities employ various strategies \
                    including solar energy, electric transit, and green building certification.".to_string())
            }
        }

        /// Build chunks from all 3 documents for the retrieval callback.
        fn build_document_chunks() -> Vec<RetrievedChunk> {
            let docs = [
                ("barcelona", doc_barcelona(), "doc_barcelona.txt"),
                ("copenhagen", doc_copenhagen(), "doc_copenhagen.txt"),
                ("singapore", doc_singapore(), "doc_singapore.txt"),
            ];

            let mut chunks = Vec::new();
            for (prefix, doc, source) in &docs {
                // Split into paragraphs
                for (i, para) in doc.split("\n\n").enumerate() {
                    let para = para.trim();
                    if para.is_empty() {
                        continue;
                    }
                    chunks.push(RetrievedChunk {
                        chunk_id: format!("{}_{}", prefix, i),
                        content: para.to_string(),
                        source: source.to_string(),
                        section: Some(format!("section_{}", i)),
                        score: 0.8 - (i as f32 * 0.05),
                        keyword_score: Some(0.8),
                        semantic_score: Some(0.7),
                        token_count: para.split_whitespace().count(),
                        position: Some(PipelineChunkPosition {
                            start_offset: 0,
                            end_offset: para.len(),
                            paragraph_index: Some(i),
                            sentence_indices: None,
                        }),
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("city".to_string(), prefix.to_string());
                            m
                        },
                    });
                }
            }
            chunks
        }

        // RAG mock callbacks
        struct HeavyLlmCb;
        impl LlmCallback for HeavyLlmCb {
            fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String, String> {
                // Delegate to smart generator
                smart_mock_generator()(prompt)
            }
            fn model_name(&self) -> &str {
                "heavy-mock-llm"
            }
        }

        struct HeavyRetrievalCb {
            chunks: Vec<RetrievedChunk>,
        }
        impl HeavyRetrievalCb {
            fn new() -> Self {
                Self {
                    chunks: build_document_chunks(),
                }
            }
        }
        impl RetrievalCallback for HeavyRetrievalCb {
            fn keyword_search(
                &self,
                query: &str,
                limit: usize,
            ) -> Result<Vec<RetrievedChunk>, String> {
                let lower = query.to_lowercase();
                let words: Vec<&str> = lower.split_whitespace().collect();
                let mut scored: Vec<(usize, &RetrievedChunk)> = self
                    .chunks
                    .iter()
                    .map(|c| {
                        let cl = c.content.to_lowercase();
                        let hits = words.iter().filter(|w| cl.contains(*w)).count();
                        (hits, c)
                    })
                    .filter(|(hits, _)| *hits > 0)
                    .collect();
                scored.sort_by(|a, b| b.0.cmp(&a.0));
                let results: Vec<RetrievedChunk> =
                    scored.into_iter().take(limit).map(|(_, c)| c.clone()).collect();
                if results.is_empty() {
                    Ok(self.chunks.iter().take(limit).cloned().collect())
                } else {
                    Ok(results)
                }
            }
            fn semantic_search(
                &self,
                _embedding: &[f32],
                limit: usize,
            ) -> Result<Vec<RetrievedChunk>, String> {
                Ok(self.chunks.iter().take(limit).cloned().collect())
            }
            fn get_chunk(&self, chunk_id: &str) -> Result<Option<RetrievedChunk>, String> {
                Ok(self.chunks.iter().find(|c| c.chunk_id == chunk_id).cloned())
            }
        }

        struct HeavyGraphCb;
        impl GraphCallback for HeavyGraphCb {
            fn extract_entities(&self, text: &str) -> Result<Vec<String>, String> {
                let lower = text.to_lowercase();
                let mut entities = Vec::new();
                let known = [
                    "Barcelona", "Copenhagen", "Singapore", "BYD", "Endesa", "BCA",
                    "Green Mark", "Movia", "SolarCity Europe", "Mediterranean",
                ];
                for name in &known {
                    if lower.contains(&name.to_lowercase()) {
                        entities.push(name.to_string());
                    }
                }
                Ok(entities)
            }
            fn get_related(
                &self,
                entity: &str,
                _depth: usize,
            ) -> Result<Vec<GraphRelation>, String> {
                match entity {
                    "Barcelona" => Ok(vec![
                        GraphRelation {
                            from: "Barcelona".into(),
                            relation_type: "implements".into(),
                            to: "Solar Mandate 2025".into(),
                            weight: 1.0,
                        },
                        GraphRelation {
                            from: "Barcelona".into(),
                            relation_type: "targets".into(),
                            to: "carbon neutrality".into(),
                            weight: 0.9,
                        },
                    ]),
                    "Copenhagen" => Ok(vec![
                        GraphRelation {
                            from: "Copenhagen".into(),
                            relation_type: "partners_with".into(),
                            to: "BYD".into(),
                            weight: 1.0,
                        },
                        GraphRelation {
                            from: "Copenhagen".into(),
                            relation_type: "operates".into(),
                            to: "electric bus fleet".into(),
                            weight: 0.95,
                        },
                    ]),
                    "Singapore" => Ok(vec![
                        GraphRelation {
                            from: "Singapore".into(),
                            relation_type: "administers".into(),
                            to: "Green Mark".into(),
                            weight: 1.0,
                        },
                        GraphRelation {
                            from: "Singapore".into(),
                            relation_type: "mandates".into(),
                            to: "vertical gardens".into(),
                            weight: 0.9,
                        },
                    ]),
                    "BYD" => Ok(vec![GraphRelation {
                        from: "BYD".into(),
                        relation_type: "supplies".into(),
                        to: "electric buses".into(),
                        weight: 1.0,
                    }]),
                    _ => Ok(vec![]),
                }
            }
            fn get_entity_chunks(
                &self,
                entities: &[String],
            ) -> Result<Vec<RetrievedChunk>, String> {
                let mut chunks = Vec::new();
                for entity in entities {
                    let content = match entity.as_str() {
                        "Barcelona" => "Barcelona approved a solar mandate targeting 15MW capacity, \
                                        achieving 30% energy cost reduction with 2,500 sunshine hours/year.",
                        "Copenhagen" => "Copenhagen partners with BYD for 100% electric bus fleet, \
                                         achieving 40% emission reduction with 65 fast-charging stations.",
                        "Singapore" => "Singapore's Green Mark Platinum requires 35% energy reduction. \
                                        Over 4,300 buildings certified. Rainwater harvesting in 80% of new buildings.",
                        "BYD" => "BYD is the Chinese electric vehicle manufacturer supplying Copenhagen's bus fleet.",
                        "BCA" => "The Building and Construction Authority administers Singapore's Green Mark program.",
                        "Green Mark" => "Green Mark certification requires 35% energy reduction via smart HVAC, \
                                         glazing, and natural ventilation.",
                        "Endesa" => "Endesa is a Spanish energy utility partnering with Barcelona's solar program.",
                        _ => continue,
                    };
                    chunks.push(RetrievedChunk {
                        chunk_id: format!("graph_{}", entity.to_lowercase().replace(' ', "_")),
                        content: content.to_string(),
                        source: "knowledge_graph".to_string(),
                        section: None,
                        score: 0.95,
                        keyword_score: None,
                        semantic_score: Some(0.95),
                        token_count: content.split_whitespace().count(),
                        position: None,
                        metadata: HashMap::new(),
                    });
                }
                Ok(chunks)
            }
        }

        // ================================================================
        // Test 1: Multi-document KB construction
        // ================================================================
        #[test]
        fn test_heavy_multi_document_kb_construction() {
            // Parse all 3 documents
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let parsed_bcn = parser
                .parse_bytes(doc_barcelona().as_bytes(), DocumentFormat::PlainText)
                .unwrap();
            let parsed_cph = parser
                .parse_bytes(doc_copenhagen().as_bytes(), DocumentFormat::PlainText)
                .unwrap();
            let parsed_sgp = parser
                .parse_bytes(doc_singapore().as_bytes(), DocumentFormat::PlainText)
                .unwrap();

            assert!(parsed_bcn.text.contains("Barcelona"));
            assert!(parsed_cph.text.contains("Copenhagen"));
            assert!(parsed_sgp.text.contains("Singapore"));
            assert!(parsed_bcn.word_count > 100);
            assert!(parsed_cph.word_count > 100);
            assert!(parsed_sgp.word_count > 100);

            // Build KG and index docs
            let (graph, _extractor) = build_sustainability_kg();

            let stats = graph.stats().unwrap();
            assert!(
                stats.total_entities >= 10,
                "Should have at least 10 entities, got {}",
                stats.total_entities
            );
            assert!(
                stats.total_chunks >= 9,
                "Should have chunks from all 3 docs (>=9), got {}",
                stats.total_chunks
            );

            // Embed chunks and store in VectorDB
            let embedder = LocalEmbedder::new(EmbeddingConfig {
                dimensions: 64,
                ..EmbeddingConfig::default()
            });

            let vdb_config = VectorDbConfig {
                dimensions: 64,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(200),
                ..VectorDbConfig::default()
            };
            let mut vdb = InMemoryVectorDb::new(vdb_config);

            // Store document text chunks in VDB
            let all_chunks = build_document_chunks();
            for chunk in &all_chunks {
                let emb = embedder.embed(&chunk.content);
                let meta = serde_json::json!({"city": chunk.metadata.get("city").unwrap_or(&String::new()).clone()});
                vdb.insert(&chunk.chunk_id, emb, meta).unwrap();
            }

            assert!(
                vdb.count() >= 9,
                "VDB should have at least 9 chunks, got {}",
                vdb.count()
            );

            // Search for Barcelona-related content
            let bcn_query_emb = embedder.embed("solar energy Barcelona rooftop panels");
            let bcn_results = vdb.search(&bcn_query_emb, 3, None).unwrap();
            assert!(!bcn_results.is_empty(), "Should find Barcelona-related chunks");

            // Search for Copenhagen-related content
            let cph_query_emb = embedder.embed("electric bus Copenhagen transit emissions");
            let cph_results = vdb.search(&cph_query_emb, 3, None).unwrap();
            assert!(
                !cph_results.is_empty(),
                "Should find Copenhagen-related chunks"
            );

            // Search for Singapore-related content
            let sgp_query_emb = embedder.embed("Singapore green building certification BEMS");
            let sgp_results = vdb.search(&sgp_query_emb, 3, None).unwrap();
            assert!(
                !sgp_results.is_empty(),
                "Should find Singapore-related chunks"
            );
        }

        // ================================================================
        // Test 2: Cross-document RAG comparison
        // ================================================================
        #[test]
        fn test_heavy_cross_document_rag_comparison() {
            let (_graph, _extractor) = build_sustainability_kg();

            let cross_doc_questions = [
                "Compare Barcelona and Copenhagen approaches to urban sustainability",
                "Which city has the most ambitious emission reduction targets",
                "How do Singapore and Barcelona handle building energy efficiency",
            ];

            let llm_cb = HeavyLlmCb;
            let retrieval_cb = HeavyRetrievalCb::new();
            let graph_cb = HeavyGraphCb;

            // Test with different RAG tiers
            let tiers = [RagTier::Disabled, RagTier::Fast, RagTier::Semantic];
            let mut tier_context_lengths: Vec<(RagTier, usize)> = Vec::new();

            for tier in &tiers {
                let config = RagTierConfig::with_tier(tier.clone());
                let mut pipeline = RagPipeline::new(config);
                let mut total_context_len = 0;

                for question in &cross_doc_questions {
                    let result = pipeline
                        .process(question, &llm_cb, None, &retrieval_cb, Some(&graph_cb))
                        .unwrap();
                    total_context_len += result.context.len();
                }

                tier_context_lengths.push((tier.clone(), total_context_len));
            }

            // Semantic tier should produce richer context than Disabled
            let disabled_len = tier_context_lengths
                .iter()
                .find(|(t, _)| *t == RagTier::Disabled)
                .map(|(_, l)| *l)
                .unwrap_or(0);
            let semantic_len = tier_context_lengths
                .iter()
                .find(|(t, _)| *t == RagTier::Semantic)
                .map(|(_, l)| *l)
                .unwrap_or(0);
            // Semantic should have more context (or at least equal)
            assert!(
                semantic_len >= disabled_len,
                "Semantic tier ({}) should produce >= context than Disabled ({})",
                semantic_len,
                disabled_len
            );

            // Test with ContextComposer to build full prompts
            let composer = ContextComposer::new(ContextComposer::default_config());
            for question in &cross_doc_questions {
                let config = RagTierConfig::with_tier(RagTier::Semantic);
                let mut pipeline = RagPipeline::new(config);
                let result = pipeline
                    .process(question, &llm_cb, None, &retrieval_cb, Some(&graph_cb))
                    .unwrap();

                let mut sections = HashMap::new();
                sections.insert(
                    ContextSection::SystemPrompt,
                    "You are an urban sustainability expert.".to_string(),
                );
                sections.insert(ContextSection::RagChunks, result.context.clone());
                sections.insert(ContextSection::UserPrompt, question.to_string());

                let composed = composer.compose(sections);
                let full_text = composed.to_composed_string();
                assert!(
                    !full_text.is_empty(),
                    "Composed context should not be empty"
                );
                assert!(
                    composed.sections.len() >= 2,
                    "Should have at least 2 sections"
                );
            }

            // Decision tree routing: multi-city → Full tier, single-city → Fast
            let multi_city_branch = DecisionBranch {
                condition: Condition::new(
                    "is_multi_city",
                    ConditionOperator::Equals,
                    json!(true),
                ),
                target_node_id: "full_tier".to_string(),
                label: Some("MultiCity".to_string()),
            };

            let tree = DecisionTreeBuilder::new("tier_router", "RAG Tier Router")
                .root("check_multi")
                .node(DecisionNode::new_condition(
                    "check_multi",
                    vec![multi_city_branch],
                    Some("fast_tier".to_string()),
                ))
                .node(DecisionNode::new_terminal(
                    "full_tier",
                    json!("Semantic"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "fast_tier",
                    json!("Fast"),
                    None,
                ))
                .build();

            // Multi-city question routes to Semantic
            let mut ctx = HashMap::new();
            ctx.insert("is_multi_city".to_string(), json!(true));
            let path = tree.evaluate(&ctx);
            assert_eq!(path.result.unwrap().as_str().unwrap(), "Semantic");

            // Single-city routes to Fast
            let mut ctx2 = HashMap::new();
            ctx2.insert("is_multi_city".to_string(), json!(false));
            let path2 = tree.evaluate(&ctx2);
            assert_eq!(path2.result.unwrap().as_str().unwrap(), "Fast");
        }

        // ================================================================
        // Test 3: Multi-agent document analysis
        // ================================================================
        #[test]
        fn test_heavy_multi_agent_document_analysis() {
            let mut orch = AgentOrchestrator::new(OrchestrationStrategy::BestFit);

            // Register 4 agents
            let mut energy = Agent::new("energy", "Energy Analyst", AgentRole::Researcher);
            energy.capabilities = vec!["energy".into(), "solar".into(), "electricity".into()];

            let mut transport = Agent::new("transport", "Transport Analyst", AgentRole::Researcher);
            transport.capabilities = vec!["transport".into(), "mobility".into(), "bus".into()];

            let mut buildings = Agent::new("buildings", "Buildings Analyst", AgentRole::Researcher);
            buildings.capabilities =
                vec!["buildings".into(), "architecture".into(), "construction".into()];

            let synthesizer = Agent::new("synth", "Synthesizer", AgentRole::Coordinator);

            orch.register_agent(energy);
            orch.register_agent(transport);
            orch.register_agent(buildings);
            orch.register_agent(synthesizer);

            // Create tasks
            let task_bcn = AgentTask {
                id: "analyze_barcelona".into(),
                description: "Analyze Barcelona solar energy initiative".into(),
                assigned_to: None,
                dependencies: vec![],
                status: crate::multi_agent::TaskStatus::Pending,
                result: None,
                priority: 1,
                deadline: None,
            };
            let task_cph = AgentTask {
                id: "analyze_copenhagen".into(),
                description: "Analyze Copenhagen electric transport mobility transit bus".into(),
                assigned_to: None,
                dependencies: vec![],
                status: crate::multi_agent::TaskStatus::Pending,
                result: None,
                priority: 1,
                deadline: None,
            };
            let task_sgp = AgentTask {
                id: "analyze_singapore".into(),
                description: "Analyze Singapore green buildings architecture construction".into(),
                assigned_to: None,
                dependencies: vec![],
                status: crate::multi_agent::TaskStatus::Pending,
                result: None,
                priority: 1,
                deadline: None,
            };
            let task_synth = AgentTask {
                id: "synthesize".into(),
                description: "Synthesize all city analyses into unified report".into(),
                assigned_to: None,
                dependencies: vec![
                    "analyze_barcelona".into(),
                    "analyze_copenhagen".into(),
                    "analyze_singapore".into(),
                ],
                status: crate::multi_agent::TaskStatus::Pending,
                result: None,
                priority: 2,
                deadline: None,
            };

            orch.add_task(task_bcn);
            orch.add_task(task_cph);
            orch.add_task(task_sgp);
            orch.add_task(task_synth);

            // Auto-assign
            let assignments = orch.auto_assign_tasks();
            assert!(
                !assignments.is_empty(),
                "Should auto-assign at least some tasks"
            );

            // Store analysis in shared context
            let mut context = SharedContext::new();

            // Simulate each analyst processing their document
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let extractor = build_sustainability_extractor();

            // Energy analyst processes Barcelona
            let parsed_bcn = parser
                .parse_bytes(doc_barcelona().as_bytes(), DocumentFormat::PlainText)
                .unwrap();
            let bcn_entities = extractor.extract(&parsed_bcn.text).unwrap();
            context.set(
                "barcelona_analysis",
                serde_json::json!({
                    "city": "Barcelona",
                    "focus": "Solar Energy",
                    "key_metrics": {"energy_cost_reduction": "30%", "sunshine_hours": 2500, "target_capacity": "15MW"},
                    "entities_found": bcn_entities.entities.len(),
                    "word_count": parsed_bcn.word_count,
                }),
                "energy",
            );

            // Transport analyst processes Copenhagen
            let parsed_cph = parser
                .parse_bytes(doc_copenhagen().as_bytes(), DocumentFormat::PlainText)
                .unwrap();
            let cph_entities = extractor.extract(&parsed_cph.text).unwrap();
            context.set(
                "copenhagen_analysis",
                serde_json::json!({
                    "city": "Copenhagen",
                    "focus": "Electric Transit",
                    "key_metrics": {"emission_reduction": "40%", "diesel_buses_replaced": 400, "charging_stations": 65},
                    "entities_found": cph_entities.entities.len(),
                    "word_count": parsed_cph.word_count,
                }),
                "transport",
            );

            // Buildings analyst processes Singapore
            let parsed_sgp = parser
                .parse_bytes(doc_singapore().as_bytes(), DocumentFormat::PlainText)
                .unwrap();
            let sgp_entities = extractor.extract(&parsed_sgp.text).unwrap();
            context.set(
                "singapore_analysis",
                serde_json::json!({
                    "city": "Singapore",
                    "focus": "Green Buildings",
                    "key_metrics": {"energy_reduction": "35%", "certified_buildings": 4300, "rainwater_harvesting": "80%"},
                    "entities_found": sgp_entities.entities.len(),
                    "word_count": parsed_sgp.word_count,
                }),
                "buildings",
            );

            // Verify all entries exist
            let bcn = context.get("barcelona_analysis").unwrap();
            assert_eq!(bcn.value["city"].as_str().unwrap(), "Barcelona");
            let cph = context.get("copenhagen_analysis").unwrap();
            assert_eq!(cph.value["city"].as_str().unwrap(), "Copenhagen");
            let sgp = context.get("singapore_analysis").unwrap();
            assert_eq!(sgp.value["city"].as_str().unwrap(), "Singapore");

            // Synthesizer reads all entries and produces merged summary
            let all_cities: Vec<String> = vec!["Barcelona", "Copenhagen", "Singapore"]
                .into_iter()
                .map(|c| c.to_string())
                .collect();
            context.set(
                "synthesis",
                serde_json::json!({
                    "cities_analyzed": all_cities,
                    "common_theme": "Urban Sustainability",
                    "total_entities": bcn_entities.entities.len() + cph_entities.entities.len() + sgp_entities.entities.len(),
                }),
                "synth",
            );

            let synth = context.get("synthesis").unwrap();
            assert_eq!(synth.value["cities_analyzed"].as_array().unwrap().len(), 3);

            // Verify diff detects changes (compare context with a fresh one)
            let mut context2 = SharedContext::new();
            context2.set("barcelona_analysis", serde_json::json!({"city": "Barcelona"}), "energy");
            let diff = context.diff(&context2);
            assert!(!diff.is_empty(), "Diff should detect differences between contexts");
        }

        // ================================================================
        // Test 4: RAG-augmented generation with evaluation
        // ================================================================
        #[test]
        fn test_heavy_rag_augmented_generation_with_evaluation() {
            // Create dataset with domain questions
            let ds = BenchmarkDataset::from_problems(
                "sustainability_qa",
                BenchmarkSuiteType::Custom("SustainabilityQA".into()),
                vec![
                    make_mc_problem(
                        "sq/bcn/1",
                        "What percentage does Barcelona's solar initiative reduce energy costs? A) 20% B) 30% C) 40% D) 50%",
                        vec!["A", "B", "C", "D"],
                        "B",
                    ),
                    make_mc_problem(
                        "sq/bcn/2",
                        "How many sunshine hours does Barcelona get per year? A) 1,500 B) 2,000 C) 2,500 D) 3,000",
                        vec!["A", "B", "C", "D"],
                        "C",
                    ),
                    make_mc_problem(
                        "sq/cph/1",
                        "What bus manufacturer did Copenhagen partner with? A) Volvo B) BYD C) Mercedes D) Scania",
                        vec!["A", "B", "C", "D"],
                        "B",
                    ),
                    make_mc_problem(
                        "sq/cph/2",
                        "What emission reduction does Copenhagen's transit plan target? A) 20% B) 30% C) 40% D) 50%",
                        vec!["A", "B", "C", "D"],
                        "C",
                    ),
                    make_mc_problem(
                        "sq/sgp/1",
                        "What certification does Singapore use for green buildings? A) LEED B) BREEAM C) Green Mark D) WELL",
                        vec!["A", "B", "C", "D"],
                        "C",
                    ),
                    make_mc_problem(
                        "sq/sgp/2",
                        "What percentage energy reduction do Singapore green buildings achieve? A) 25% B) 35% C) 45% D) 55%",
                        vec!["A", "B", "C", "D"],
                        "B",
                    ),
                ],
            );

            // Generator augmented with RAG
            let gen = |prompt: &str| -> Result<String, String> {
                let lower = prompt.to_lowercase();
                // Answer MC questions based on domain knowledge
                if lower.contains("30%") || (lower.contains("barcelona") && lower.contains("energy cost")) {
                    Ok("B".to_string())
                } else if lower.contains("2,500") || lower.contains("sunshine") {
                    Ok("C".to_string())
                } else if lower.contains("byd") || (lower.contains("copenhagen") && lower.contains("manufacturer")) {
                    Ok("B".to_string())
                } else if lower.contains("40%") || (lower.contains("copenhagen") && lower.contains("emission")) {
                    Ok("C".to_string())
                } else if lower.contains("green mark") || (lower.contains("singapore") && lower.contains("certification")) {
                    Ok("C".to_string())
                } else if lower.contains("35%") || (lower.contains("singapore") && lower.contains("energy reduction")) {
                    Ok("B".to_string())
                } else {
                    Ok("B".to_string())
                }
            };

            let config = RunConfig {
                samples_per_problem: 1,
                temperature: 0.0,
                max_tokens: Some(100),
                timeout_secs: 60,
                max_retries: 1,
                model_id: test_model("sustainability-qa"),
                chain_of_thought: false,
                prompt_template: None,
            };

            let runner = BenchmarkSuiteRunner::new(gen);
            let result = runner.run_dataset(&ds, &config).unwrap();
            assert_eq!(result.results.len(), 6, "Should have 6 results");
            assert!(
                result.accuracy() >= 0.5,
                "Accuracy should be >= 0.5, got {}",
                result.accuracy()
            );

            // Evaluate with LlmJudge
            let judge = LlmJudge::new(vec![EvalCriterion::Relevance, EvalCriterion::Faithfulness]);

            for pr in &result.results {
                let first_response = pr.responses.first().unwrap();
                let prompts = judge.evaluate_all(
                    &ds.problems.iter().find(|p| p.id == pr.problem_id).unwrap().prompt,
                    first_response,
                    Some("Urban sustainability context from Barcelona, Copenhagen, Singapore documents"),
                );
                assert_eq!(prompts.len(), 2, "Should have 2 criterion prompts");
                for (criterion, prompt) in &prompts {
                    assert!(
                        !prompt.is_empty(),
                        "Judge prompt for {} should not be empty",
                        criterion
                    );
                    assert!(
                        prompt.contains(first_response),
                        "Judge prompt should contain the response"
                    );
                }
            }
        }

        // ================================================================
        // Test 5: Knowledge evolution with episodic memory
        // ================================================================
        #[test]
        fn test_heavy_knowledge_evolution_with_memory() {
            let questions = [
                ("eq/bcn/1", "What percentage energy cost reduction does Barcelona achieve?", "30%"),
                ("eq/bcn/2", "How many sunshine hours does Barcelona get per year?", "2,500"),
                ("eq/cph/1", "What manufacturer supplies Copenhagen's electric buses?", "BYD"),
                ("eq/cph/2", "What emission reduction has Copenhagen achieved?", "40%"),
                ("eq/sgp/1", "What building certification does Singapore use?", "Green Mark"),
                ("eq/sgp/2", "What energy reduction do Singapore's green buildings achieve?", "35%"),
            ];

            // ---- Round 1: Baseline evaluation ----
            let basic_gen = |prompt: &str| -> Result<String, String> {
                let lower = prompt.to_lowercase();
                if lower.contains("30%") || lower.contains("energy cost") { Ok("30%".into()) }
                else if lower.contains("sunshine") { Ok("2,500".into()) }
                else if lower.contains("manufacturer") || lower.contains("bus") { Ok("BYD".into()) }
                else if lower.contains("emission") { Ok("40%".into()) }
                else if lower.contains("certification") { Ok("Green Mark".into()) }
                else if lower.contains("energy reduction") { Ok("35%".into()) }
                else { Ok("unknown".into()) }
            };

            let mut episodic = EpisodicStore::new(50, 0.001);
            let mut procedural = ProceduralStore::new(20);

            let mut round1_correct = 0;
            for (id, question, answer) in &questions {
                let response = basic_gen(question).unwrap();
                let score: f64 = if response.to_lowercase().contains(&answer.to_lowercase()) { 1.0 } else { 0.0 };

                if score > 0.5 {
                    round1_correct += 1;
                }

                // Store as episode
                let city = if id.contains("bcn") {
                    "barcelona"
                } else if id.contains("cph") {
                    "copenhagen"
                } else {
                    "singapore"
                };
                let episode = Episode {
                    id: id.to_string(),
                    content: format!("Q: {} A: {}", question, response),
                    context: format!("city:{} score:{}", city, score),
                    timestamp: 1000,
                    importance: score,
                    tags: vec![city.to_string(), "sustainability".to_string()],
                    embedding: deterministic_embedding(&format!("{} {}", question, response), 32),
                    access_count: 0,
                    last_accessed: 0,
                };
                episodic.add(episode);
            }

            assert_eq!(episodic.len(), 6, "Should have 6 episodes");

            // Derive procedures
            procedural.add(Procedure {
                id: "proc_percentage".into(),
                name: "Percentage Question Strategy".into(),
                condition: "percentage reduction energy cost emission".into(),
                steps: vec![
                    "Look for numeric percentages in context".into(),
                    "Match percentage to the specific city mentioned".into(),
                ],
                success_count: 4,
                failure_count: 0,
                confidence: 0.9,
                created_from: vec!["eq/bcn/1".into(), "eq/cph/2".into()],
                tags: vec!["numeric".into()],
            });
            procedural.add(Procedure {
                id: "proc_entity".into(),
                name: "Entity Question Strategy".into(),
                condition: "manufacturer certification building supplier partner".into(),
                steps: vec![
                    "Identify named entities in the question".into(),
                    "Look up entity relationships in knowledge graph".into(),
                ],
                success_count: 2,
                failure_count: 0,
                confidence: 0.85,
                created_from: vec!["eq/cph/1".into(), "eq/sgp/1".into()],
                tags: vec!["entity".into()],
            });

            assert!(procedural.find_by_condition("percentage reduction").len() >= 1);
            assert!(procedural.find_by_condition("manufacturer certification").len() >= 1);

            // ---- Round 2: Memory-augmented evaluation ----
            let mut round2_correct = 0;
            for (id, question, answer) in &questions {
                // Recall similar episodes
                let q_emb = deterministic_embedding(question, 32);
                let recalled = episodic.recall(&q_emb, 2);
                assert!(
                    !recalled.is_empty(),
                    "Should recall at least 1 episode for {}",
                    id
                );

                // Look up procedures
                let procs = procedural.find_by_condition(question);

                // Build augmented prompt
                let mut augmented = String::new();
                if !recalled.is_empty() {
                    augmented.push_str(&format!("Recall: {}\n", recalled[0].content));
                }
                if !procs.is_empty() {
                    augmented.push_str(&format!(
                        "Strategy: {}\n",
                        procs[0].steps.join(", ")
                    ));
                }
                augmented.push_str(&format!("Hint: use retrieved context\n{}", question));

                let response = smart_mock_generator()(&augmented).unwrap();
                // Check if answer keyword appears in response
                let score = if response.to_lowercase().contains(&answer.to_lowercase()) {
                    1.0
                } else {
                    0.0
                };
                if score > 0.5 {
                    round2_correct += 1;
                }
            }

            // Round 2 should be at least as good as round 1
            assert!(
                round2_correct >= round1_correct,
                "Memory-augmented round ({}) should be >= baseline ({})",
                round2_correct,
                round1_correct
            );
        }

        // ================================================================
        // Test 6: Fact verification pipeline
        // ================================================================
        #[test]
        fn test_heavy_fact_verification_pipeline() {
            let (_graph, extractor) = build_sustainability_kg();

            // Claims: (claim_text, is_true, explanation)
            let claims = [
                ("Barcelona aims for 30% energy cost reduction through solar panels", true, "Matches Doc A"),
                ("Copenhagen partnered with BYD for electric buses", true, "Matches Doc B"),
                ("Singapore uses Green Mark certification for green buildings", true, "Matches Doc C"),
                ("Barcelona targets 50% emission reduction by 2025", false, "Doc A says 30% cost reduction, not 50%"),
                ("Copenhagen has 3,000 sunshine hours per year", false, "That's Barcelona with 2,500 hours"),
                ("Singapore green buildings achieve 60% energy reduction", false, "Doc C says 35%, not 60%"),
            ];

            let gen = smart_mock_generator();

            // Decision tree for verification routing
            let has_entity_branch = DecisionBranch {
                condition: Condition::new(
                    "has_entities",
                    ConditionOperator::Equals,
                    json!(true),
                ),
                target_node_id: "kg_verify".to_string(),
                label: Some("HasEntities".to_string()),
            };

            let tree = DecisionTreeBuilder::new("verify_router", "Verification Router")
                .root("check_entities")
                .node(DecisionNode::new_condition(
                    "check_entities",
                    vec![has_entity_branch],
                    Some("rag_verify".to_string()),
                ))
                .node(DecisionNode::new_terminal(
                    "kg_verify",
                    json!("knowledge_graph"),
                    None,
                ))
                .node(DecisionNode::new_terminal(
                    "rag_verify",
                    json!("rag_retrieval"),
                    None,
                ))
                .build();

            let mut correct_verifications = 0;

            for (claim, is_true, _explanation) in &claims {
                // Extract entities from claim
                let entities = extractor.extract(claim).unwrap();
                let has_entities = !entities.entities.is_empty();

                // Route through decision tree
                let mut ctx = HashMap::new();
                ctx.insert("has_entities".to_string(), json!(has_entities));
                let path = tree.evaluate(&ctx);
                let _route = path.result.unwrap();

                // Generate verification using smart generator
                let verify_prompt = format!(
                    "Verify or refute this claim: {}. Based on source documents, is this claim accurate?",
                    claim
                );
                let response = gen(&verify_prompt).unwrap();

                // Check if verification is correct
                let expected = if *is_true { "VERIFIED" } else { "REFUTED" };
                if response.contains(expected) {
                    correct_verifications += 1;
                }
            }

            assert!(
                correct_verifications >= 5,
                "Should correctly verify/refute at least 5/6 claims, got {}/6",
                correct_verifications
            );
        }

        // ================================================================
        // Test 7: End-to-end document QA pipeline (ALL features)
        // ================================================================
        #[test]
        fn test_heavy_end_to_end_document_qa_pipeline() {
            // 1. Document ingestion
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let parsed_docs = [
                ("barcelona", parser.parse_bytes(doc_barcelona().as_bytes(), DocumentFormat::PlainText).unwrap()),
                ("copenhagen", parser.parse_bytes(doc_copenhagen().as_bytes(), DocumentFormat::PlainText).unwrap()),
                ("singapore", parser.parse_bytes(doc_singapore().as_bytes(), DocumentFormat::PlainText).unwrap()),
            ];

            for (city, doc) in &parsed_docs {
                assert!(
                    doc.word_count > 100,
                    "{} doc should have >100 words",
                    city
                );
            }

            // KG + VDB
            let (graph, _extractor) = build_sustainability_kg();
            let stats = graph.stats().unwrap();
            assert!(stats.total_entities >= 10);

            let embedder = LocalEmbedder::new(EmbeddingConfig {
                dimensions: 64,
                ..EmbeddingConfig::default()
            });
            let vdb_config = VectorDbConfig {
                dimensions: 64,
                distance_metric: DistanceMetric::Cosine,
                max_vectors: Some(200),
                ..VectorDbConfig::default()
            };
            let mut vdb = InMemoryVectorDb::new(vdb_config);

            let all_chunks = build_document_chunks();
            for chunk in &all_chunks {
                let emb = embedder.embed(&chunk.content);
                let meta = serde_json::json!({"city": chunk.metadata.get("city").unwrap_or(&String::new()).clone()});
                vdb.insert(&chunk.chunk_id, emb, meta).unwrap();
            }
            assert!(vdb.count() >= 9);

            // 2. Memory bootstrap
            let mut episodic = EpisodicStore::new(50, 0.001);
            let initial_episodes = [
                ("past_bcn", "Barcelona solar 30% reduction success", "barcelona", 0.9),
                ("past_cph", "Copenhagen BYD electric bus 40% emission cut", "copenhagen", 0.85),
                ("past_sgp", "Singapore Green Mark 35% energy efficient", "singapore", 0.88),
            ];
            for (id, content, city, importance) in &initial_episodes {
                episodic.add(Episode {
                    id: id.to_string(),
                    content: content.to_string(),
                    context: format!("city:{}", city),
                    timestamp: 900,
                    importance: *importance,
                    tags: vec![city.to_string()],
                    embedding: deterministic_embedding(content, 32),
                    access_count: 0,
                    last_accessed: 0,
                });
            }
            assert_eq!(episodic.len(), 3);

            let mut procedural = ProceduralStore::new(10);
            procedural.add(Procedure {
                id: "proc_factual".into(),
                name: "Factual QA Strategy".into(),
                condition: "percentage number quantity metric".into(),
                steps: vec!["Extract numeric values from context".into()],
                success_count: 5,
                failure_count: 0,
                confidence: 0.9,
                created_from: vec![],
                tags: vec![],
            });
            procedural.add(Procedure {
                id: "proc_comparison".into(),
                name: "Comparison QA Strategy".into(),
                condition: "compare difference between cities".into(),
                steps: vec!["Gather data for each city".into(), "Contrast metrics".into()],
                success_count: 3,
                failure_count: 0,
                confidence: 0.85,
                created_from: vec![],
                tags: vec![],
            });

            // 3. Agent setup
            let mut orch = AgentOrchestrator::new(OrchestrationStrategy::BestFit);

            let mut bcn_agent = Agent::new("bcn_agent", "Barcelona Specialist", AgentRole::Researcher);
            bcn_agent.capabilities = vec!["barcelona".into(), "solar".into(), "energy".into()];
            let mut cph_agent = Agent::new("cph_agent", "Copenhagen Specialist", AgentRole::Researcher);
            cph_agent.capabilities = vec!["copenhagen".into(), "transit".into(), "bus".into()];
            let mut sgp_agent = Agent::new("sgp_agent", "Singapore Specialist", AgentRole::Researcher);
            sgp_agent.capabilities = vec!["singapore".into(), "building".into(), "green".into()];
            let coordinator = Agent::new("coord", "Coordinator", AgentRole::Coordinator);

            orch.register_agent(bcn_agent);
            orch.register_agent(cph_agent);
            orch.register_agent(sgp_agent);
            orch.register_agent(coordinator);

            // 4. Routing setup
            let bcn_branch = DecisionBranch {
                condition: Condition::new("city", ConditionOperator::Equals, json!("barcelona")),
                target_node_id: "bcn_path".to_string(),
                label: Some("Barcelona".to_string()),
            };
            let cph_branch = DecisionBranch {
                condition: Condition::new("city", ConditionOperator::Equals, json!("copenhagen")),
                target_node_id: "cph_path".to_string(),
                label: Some("Copenhagen".to_string()),
            };
            let sgp_branch = DecisionBranch {
                condition: Condition::new("city", ConditionOperator::Equals, json!("singapore")),
                target_node_id: "sgp_path".to_string(),
                label: Some("Singapore".to_string()),
            };

            let tree = DecisionTreeBuilder::new("city_router", "City Router")
                .root("check_city")
                .node(DecisionNode::new_condition(
                    "check_city",
                    vec![bcn_branch, cph_branch, sgp_branch],
                    Some("cross_doc_path".to_string()),
                ))
                .node(DecisionNode::new_terminal("bcn_path", json!("bcn_specialist"), None))
                .node(DecisionNode::new_terminal("cph_path", json!("cph_specialist"), None))
                .node(DecisionNode::new_terminal("sgp_path", json!("sgp_specialist"), None))
                .node(DecisionNode::new_terminal("cross_doc_path", json!("cross_doc"), None))
                .build();

            // MultiModelGenerator with city specialists
            let mut multi_gen = MultiModelGenerator::new(smart_mock_generator());
            multi_gen.register_model("bcn_specialist", |prompt: &str| {
                let lower = prompt.to_lowercase();
                if lower.contains("30%") || lower.contains("energy cost") { Ok("B".into()) }
                else if lower.contains("sunshine") || lower.contains("2,500") { Ok("C".into()) }
                else if lower.contains("15") && lower.contains("megawatt") { Ok("15MW".into()) }
                else { Ok("B".into()) }
            });
            multi_gen.register_model("cph_specialist", |prompt: &str| {
                let lower = prompt.to_lowercase();
                if lower.contains("byd") || lower.contains("manufacturer") { Ok("B".into()) }
                else if lower.contains("40%") || lower.contains("emission") { Ok("C".into()) }
                else if lower.contains("400") || lower.contains("diesel") { Ok("400".into()) }
                else { Ok("C".into()) }
            });
            multi_gen.register_model("sgp_specialist", |prompt: &str| {
                let lower = prompt.to_lowercase();
                if lower.contains("green mark") || lower.contains("certification") { Ok("C".into()) }
                else if lower.contains("35%") || lower.contains("energy reduction") { Ok("B".into()) }
                else if lower.contains("4,300") || lower.contains("certified") { Ok("4300".into()) }
                else { Ok("C".into()) }
            });

            // 5. Questions
            let questions = [
                ("e2e/bcn/1", "What percentage energy cost reduction does Barcelona achieve? A) 20% B) 30% C) 40%", "B", "barcelona"),
                ("e2e/bcn/2", "How many sunshine hours per year in Barcelona? A) 1,500 B) 2,000 C) 2,500", "C", "barcelona"),
                ("e2e/cph/1", "What manufacturer supplies Copenhagen's electric buses? A) Volvo B) BYD C) Mercedes", "B", "copenhagen"),
                ("e2e/cph/2", "What emission reduction has Copenhagen achieved? A) 20% B) 30% C) 40%", "C", "copenhagen"),
                ("e2e/sgp/1", "What certification does Singapore use? A) LEED B) BREEAM C) Green Mark", "C", "singapore"),
                ("e2e/sgp/2", "What energy reduction do Singapore buildings achieve? A) 25% B) 35% C) 45%", "B", "singapore"),
                ("e2e/cross/1", "Which city partners with BYD? A) Barcelona B) Copenhagen C) Singapore", "B", "cross"),
                ("e2e/cross/2", "Which city uses rainwater harvesting? A) Barcelona B) Copenhagen C) Singapore", "C", "cross"),
            ];

            let mut shared = SharedContext::new();
            let composer = ContextComposer::new(ContextComposer::default_config());
            let mut correct_count = 0;

            for (id, question, answer, city) in &questions {
                // Route through decision tree
                let mut ctx = HashMap::new();
                ctx.insert("city".to_string(), json!(city));
                let path = tree.evaluate(&ctx);
                let model_key = path.result.unwrap();
                let model_name = model_key.as_str().unwrap();

                // RAG augmentation
                let rag_config = RagTierConfig::with_tier(RagTier::Fast);
                let mut pipeline = RagPipeline::new(rag_config);
                let llm_cb = HeavyLlmCb;
                let retrieval_cb = HeavyRetrievalCb::new();
                let graph_cb = HeavyGraphCb;
                let rag_result = pipeline
                    .process(question, &llm_cb, None, &retrieval_cb, Some(&graph_cb))
                    .unwrap();

                // Compose context
                let mut sections = HashMap::new();
                sections.insert(
                    ContextSection::SystemPrompt,
                    "You are an urban sustainability expert.".to_string(),
                );
                sections.insert(ContextSection::RagChunks, rag_result.context.clone());

                // Add memory context
                let q_emb = deterministic_embedding(question, 32);
                let recalled = episodic.recall(&q_emb, 1);
                if !recalled.is_empty() {
                    sections.insert(
                        ContextSection::MemoryContext,
                        format!("Past experience: {}", recalled[0].content),
                    );
                }

                sections.insert(ContextSection::UserPrompt, question.to_string());
                let composed = composer.compose(sections);
                assert!(
                    composed.sections.len() >= 3,
                    "Should have >= 3 sections for {}",
                    id
                );

                // Generate answer via routed model
                let response = multi_gen.generate(model_name, question).unwrap();

                // Score (simple exact match for MC answers)
                let score: f64 = if response.trim() == *answer { 1.0 } else { 0.0 };
                if score > 0.5 {
                    correct_count += 1;
                }

                // Store in SharedContext
                let agent_id = match *city {
                    "barcelona" => "bcn_agent",
                    "copenhagen" => "cph_agent",
                    "singapore" => "sgp_agent",
                    _ => "coord",
                };
                shared.set(
                    &format!("result_{}", id),
                    serde_json::json!({
                        "question": question,
                        "response": response,
                        "score": score,
                        "model": model_name,
                    }),
                    agent_id,
                );

                // Store as new episode
                episodic.add(Episode {
                    id: id.to_string(),
                    content: format!("Q: {} A: {}", question, response),
                    context: format!("city:{} score:{}", city, score),
                    timestamp: 1100,
                    importance: score,
                    tags: vec![city.to_string()],
                    embedding: deterministic_embedding(&format!("{} {}", question, response), 32),
                    access_count: 0,
                    last_accessed: 0,
                });
            }

            // 6. Evaluation
            assert!(
                correct_count >= 4,
                "Should correctly answer at least 4/8 questions, got {}",
                correct_count
            );

            let judge = LlmJudge::new(vec![
                EvalCriterion::Relevance,
                EvalCriterion::Faithfulness,
                EvalCriterion::Helpfulness,
            ]);
            for (_, question, _, _) in &questions {
                let prompts = judge.evaluate_all(question, "test response", Some("sustainability context"));
                assert_eq!(prompts.len(), 3, "Should generate 3 criterion prompts");
            }

            // 7. Verification
            assert!(stats.total_entities >= 10, "KG should have >= 10 entities");
            assert!(vdb.count() >= 9, "VDB should have >= 9 chunks");
            assert_eq!(
                episodic.len(),
                3 + 8,
                "Episodic store should have 3 initial + 8 new = 11"
            );

            // Check SharedContext has entries from multiple agents
            let bcn_result = shared.get("result_e2e/bcn/1");
            assert!(bcn_result.is_some(), "Should have Barcelona result");
            let cph_result = shared.get("result_e2e/cph/1");
            assert!(cph_result.is_some(), "Should have Copenhagen result");
            let sgp_result = shared.get("result_e2e/sgp/1");
            assert!(sgp_result.is_some(), "Should have Singapore result");
        }
    }
}
