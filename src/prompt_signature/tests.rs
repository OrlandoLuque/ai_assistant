use super::*;
use std::collections::HashMap;

    // --- FieldType tests ---

    #[test]
    fn test_field_type_variants() {
        let types = vec![
            FieldType::Text,
            FieldType::Number,
            FieldType::Boolean,
            FieldType::List,
            FieldType::Json,
        ];
        assert_eq!(types.len(), 5);
        assert_eq!(FieldType::Text, FieldType::Text);
        assert_ne!(FieldType::Text, FieldType::Number);
    }

    #[test]
    fn test_field_type_display() {
        assert_eq!(FieldType::Text.to_string(), "text");
        assert_eq!(FieldType::Number.to_string(), "number");
        assert_eq!(FieldType::Boolean.to_string(), "boolean");
        assert_eq!(FieldType::List.to_string(), "list");
        assert_eq!(FieldType::Json.to_string(), "json");
    }

    #[test]
    fn test_field_type_serialization() {
        let ft = FieldType::Json;
        let json = serde_json::to_string(&ft).expect("serialize");
        let back: FieldType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, ft);
    }

    // --- SignatureField tests ---

    #[test]
    fn test_signature_field_creation() {
        let field = SignatureField::new("question", "The question to answer");
        assert_eq!(field.name, "question");
        assert_eq!(field.description, "The question to answer");
        assert_eq!(field.field_type, FieldType::Text);
        assert!(field.required);
        assert!(field.prefix.is_none());
    }

    #[test]
    fn test_signature_field_builder() {
        let field = SignatureField::new("count", "Number of items")
            .with_type(FieldType::Number)
            .optional()
            .with_prefix("Item Count");
        assert_eq!(field.field_type, FieldType::Number);
        assert!(!field.required);
        assert_eq!(field.prefix.as_deref(), Some("Item Count"));
    }

    #[test]
    fn test_signature_field_clone() {
        let field = SignatureField::new("test", "desc").with_prefix("pfx");
        let cloned = field.clone();
        assert_eq!(cloned.name, "test");
        assert_eq!(cloned.prefix.as_deref(), Some("pfx"));
    }

    #[test]
    fn test_signature_field_serialization() {
        let field = SignatureField::new("query", "Search query")
            .with_type(FieldType::Text)
            .with_prefix("Q");
        let json = serde_json::to_string(&field).expect("serialize");
        let back: SignatureField = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "query");
        assert_eq!(back.prefix.as_deref(), Some("Q"));
    }

    // --- Signature builder tests ---

    #[test]
    fn test_signature_builder() {
        let sig = Signature::new("qa", "Answer questions based on context")
            .add_input(SignatureField::new("context", "The context passage"))
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"))
            .with_instructions("Be concise");

        assert_eq!(sig.name, "qa");
        assert_eq!(sig.inputs.len(), 2);
        assert_eq!(sig.outputs.len(), 1);
        assert_eq!(sig.instructions.as_deref(), Some("Be concise"));
    }

    #[test]
    fn test_signature_empty() {
        let sig = Signature::new("empty", "An empty signature");
        assert!(sig.inputs.is_empty());
        assert!(sig.outputs.is_empty());
        assert!(sig.instructions.is_none());
    }

    #[test]
    fn test_signature_clone() {
        let sig = Signature::new("test", "Test sig")
            .add_input(SignatureField::new("in1", "input"))
            .with_instructions("inst");
        let cloned = sig.clone();
        assert_eq!(cloned.name, "test");
        assert_eq!(cloned.inputs.len(), 1);
        assert_eq!(cloned.instructions.as_deref(), Some("inst"));
    }

    #[test]
    fn test_signature_serialization() {
        let sig = Signature::new("s", "desc")
            .add_input(SignatureField::new("i", "input"))
            .add_output(SignatureField::new("o", "output"));
        let json = serde_json::to_string(&sig).expect("serialize");
        let back: Signature = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "s");
        assert_eq!(back.inputs.len(), 1);
        assert_eq!(back.outputs.len(), 1);
    }

    // --- Signature compile tests ---

    #[test]
    fn test_signature_compile() {
        let sig = Signature::new("qa", "Answer questions")
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"));

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("Answer questions"));
        assert!(compiled.system_prompt.contains("question"));
        assert!(compiled.system_prompt.contains("answer"));
        assert!(compiled.user_template.contains("{question}"));
    }

    #[test]
    fn test_signature_compile_with_instructions() {
        let sig = Signature::new("qa", "Answer questions")
            .add_input(SignatureField::new("q", "question"))
            .add_output(SignatureField::new("a", "answer"))
            .with_instructions("Think step by step");

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("Think step by step"));
    }

    #[test]
    fn test_signature_compile_with_examples() {
        let sig = Signature::new("qa", "QA")
            .add_input(SignatureField::new("q", "question"))
            .add_output(SignatureField::new("a", "answer"));

        let examples = vec![PromptExample {
            inputs: HashMap::from([("q".to_string(), "What is 2+2?".to_string())]),
            outputs: HashMap::from([("a".to_string(), "4".to_string())]),
        }];

        let compiled = sig.compile_with_examples(&examples);
        assert_eq!(compiled.examples.len(), 1);
    }

    #[test]
    fn test_signature_compile_field_types() {
        let sig = Signature::new("typed", "Typed signature")
            .add_input(
                SignatureField::new("count", "a number")
                    .with_type(FieldType::Number),
            )
            .add_output(
                SignatureField::new("result", "a boolean")
                    .with_type(FieldType::Boolean),
            );

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("number"));
        assert!(compiled.system_prompt.contains("boolean"));
    }

    #[test]
    fn test_signature_compile_optional_fields() {
        let sig = Signature::new("opt", "Optional fields test")
            .add_input(SignatureField::new("required_in", "required").optional())
            .add_output(SignatureField::new("opt_out", "optional output").optional());

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("optional"));
    }

    #[test]
    fn test_signature_compile_with_prefixes() {
        let sig = Signature::new("pfx", "Prefix test")
            .add_input(SignatureField::new("q", "question").with_prefix("Question"))
            .add_output(SignatureField::new("a", "answer").with_prefix("Answer"));

        let compiled = sig.compile();
        assert!(compiled.user_template.contains("Question:"));
        assert!(compiled.user_template.contains("Answer:"));
    }

    // --- Validate inputs ---

    #[test]
    fn test_validate_inputs_success() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name"));
        let inputs = HashMap::from([("name".to_string(), "Alice".to_string())]);
        assert!(sig.validate_inputs(&inputs).is_ok());
    }

    #[test]
    fn test_validate_inputs_missing_required() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name"));
        let inputs: HashMap<String, String> = HashMap::new();
        let result = sig.validate_inputs(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_inputs_optional_missing_ok() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name").optional());
        let inputs: HashMap<String, String> = HashMap::new();
        assert!(sig.validate_inputs(&inputs).is_ok());
    }

    // --- CompiledPrompt tests ---

    #[test]
    fn test_compiled_prompt_render() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Q: {question}\nA: ".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([("question".to_string(), "What is Rust?".to_string())]);
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("What is Rust?"));
        assert!(!rendered.contains("{question}"));
    }

    #[test]
    fn test_compiled_prompt_render_multiple_fields() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Context: {ctx}\nQuestion: {q}".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([
            ("ctx".to_string(), "Rust is a language.".to_string()),
            ("q".to_string(), "What is Rust?".to_string()),
        ]);
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("Rust is a language."));
        assert!(rendered.contains("What is Rust?"));
    }

    #[test]
    fn test_compiled_prompt_build_full_no_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "You are helpful.".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([("q".to_string(), "Hi".to_string())]);
        let full = compiled.build_full_prompt(&inputs);
        assert!(full.contains("You are helpful."));
        assert!(full.contains("Q: Hi"));
        assert!(!full.contains("Examples"));
    }

    #[test]
    fn test_compiled_prompt_build_full_with_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![PromptExample {
                inputs: HashMap::from([("q".to_string(), "2+2?".to_string())]),
                outputs: HashMap::from([("a".to_string(), "4".to_string())]),
            }],
        };
        let inputs = HashMap::from([("q".to_string(), "3+3?".to_string())]);
        let full = compiled.build_full_prompt(&inputs);
        assert!(full.contains("Example 1:"));
        assert!(full.contains("2+2?"));
        assert!(full.contains("Q: 3+3?"));
    }

    #[test]
    fn test_compiled_prompt_serialization() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "user {x}".to_string(),
            examples: vec![PromptExample {
                inputs: HashMap::from([("x".to_string(), "val".to_string())]),
                outputs: HashMap::from([("y".to_string(), "res".to_string())]),
            }],
        };
        let json = serde_json::to_string(&compiled).expect("serialize");
        let back: CompiledPrompt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.system_prompt, "sys");
        assert_eq!(back.examples.len(), 1);
    }

    // --- Metric tests ---

    #[test]
    fn test_exact_match_metric() {
        let metric = ExactMatch;
        assert_eq!(metric.name(), "exact_match");
        assert_eq!(metric.score("hello", "hello"), 1.0);
        assert_eq!(metric.score("Hello", "hello"), 1.0);
        assert_eq!(metric.score(" hello ", "hello"), 1.0);
        assert_eq!(metric.score("world", "hello"), 0.0);
    }

    #[test]
    fn test_exact_match_empty() {
        let metric = ExactMatch;
        assert_eq!(metric.score("", ""), 1.0);
        assert_eq!(metric.score("a", ""), 0.0);
        assert_eq!(metric.score("", "a"), 0.0);
    }

    #[test]
    fn test_f1_score_metric() {
        let metric = F1Score;
        assert_eq!(metric.name(), "f1_score");

        // Perfect match
        assert_eq!(metric.score("the cat sat", "the cat sat"), 1.0);

        // Partial match
        let s = metric.score("the cat", "the cat sat on mat");
        assert!(s > 0.0 && s < 1.0, "partial match should give 0 < score < 1, got {}", s);

        // No match
        assert_eq!(metric.score("xyz", "abc"), 0.0);
    }

    #[test]
    fn test_f1_score_empty() {
        let metric = F1Score;
        assert_eq!(metric.score("", ""), 1.0);
        assert_eq!(metric.score("word", ""), 0.0);
        assert_eq!(metric.score("", "word"), 0.0);
    }

    #[test]
    fn test_f1_score_case_insensitive() {
        let metric = F1Score;
        assert_eq!(metric.score("The Cat", "the cat"), 1.0);
    }

    #[test]
    fn test_f1_score_duplicate_tokens() {
        let metric = F1Score;
        // "a a" vs "a b": 1 match out of pred=2, exp=2 => P=0.5, R=0.5, F1=0.5
        let s = metric.score("a a", "a b");
        assert!((s - 0.5).abs() < 1e-9, "expected 0.5, got {}", s);
    }

    #[test]
    fn test_contains_answer_metric() {
        let metric = ContainsAnswer;
        assert_eq!(metric.name(), "contains_answer");
        assert_eq!(
            metric.score("The answer is 42 of course", "42"),
            1.0
        );
        assert_eq!(metric.score("No match here", "42"), 0.0);
    }

    #[test]
    fn test_contains_answer_case_insensitive() {
        let metric = ContainsAnswer;
        assert_eq!(metric.score("HELLO world", "hello"), 1.0);
    }

    #[test]
    fn test_contains_answer_empty() {
        let metric = ContainsAnswer;
        // Empty expected is always contained
        assert_eq!(metric.score("anything", ""), 1.0);
        assert_eq!(metric.score("", "notempty"), 0.0);
    }

    // --- EvaluationBudget tests ---

    #[test]
    fn test_evaluation_budget_new() {
        let budget = EvaluationBudget::new(10, 5);
        assert_eq!(budget.max_trials, 10);
        assert_eq!(budget.max_examples, 5);
        assert!(budget.timeout_seconds.is_none());
        assert_eq!(budget.remaining(), 10);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_budget_with_timeout() {
        let budget = EvaluationBudget::new(5, 3).with_timeout(60);
        assert_eq!(budget.timeout_seconds, Some(60));
    }

    #[test]
    fn test_budget_try_use() {
        let mut budget = EvaluationBudget::new(3, 5);
        assert!(budget.try_use());
        assert_eq!(budget.remaining(), 2);
        assert_eq!(budget.used(), 1);
        assert!(budget.try_use());
        assert!(budget.try_use());
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn test_budget_exhausted() {
        let mut budget = EvaluationBudget::new(2, 5);
        assert!(budget.try_use());
        assert!(budget.try_use());
        assert!(!budget.try_use()); // exhausted
        assert_eq!(budget.remaining(), 0);
        assert_eq!(budget.used(), 2);
    }

    #[test]
    fn test_budget_reset() {
        let mut budget = EvaluationBudget::new(3, 5);
        budget.try_use();
        budget.try_use();
        assert_eq!(budget.remaining(), 1);
        budget.reset();
        assert_eq!(budget.remaining(), 3);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_budget_zero_trials() {
        let mut budget = EvaluationBudget::new(0, 5);
        assert!(!budget.try_use());
        assert_eq!(budget.remaining(), 0);
    }

    // --- TrainingExample tests ---

    #[test]
    fn test_training_example_roundtrip() {
        let ex = TrainingExample {
            inputs: HashMap::from([("q".to_string(), "What?".to_string())]),
            expected_outputs: HashMap::from([("a".to_string(), "Answer".to_string())]),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: TrainingExample = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.inputs.get("q").map(|s| s.as_str()), Some("What?"));
        assert_eq!(
            back.expected_outputs.get("a").map(|s| s.as_str()),
            Some("Answer")
        );
    }

    #[test]
    fn test_training_example_empty_fields() {
        let ex = TrainingExample {
            inputs: HashMap::new(),
            expected_outputs: HashMap::new(),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: TrainingExample = serde_json::from_str(&json).expect("deserialize");
        assert!(back.inputs.is_empty());
        assert!(back.expected_outputs.is_empty());
    }

    // --- OptimizationResult tests ---

    #[test]
    fn test_optimization_result_serialization() {
        let result = OptimizationResult {
            best_prompt: CompiledPrompt {
                system_prompt: "sys".to_string(),
                user_template: "usr".to_string(),
                examples: vec![],
            },
            best_score: 0.95,
            trials_run: 10,
            scores_history: vec![0.5, 0.7, 0.95],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let back: OptimizationResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.best_score, 0.95);
        assert_eq!(back.trials_run, 10);
        assert_eq!(back.scores_history.len(), 3);
    }

    // --- Helper: create a standard QA signature + examples ---

    fn make_qa_signature() -> Signature {
        Signature::new("qa", "Answer questions based on context")
            .add_input(SignatureField::new("context", "The context"))
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"))
    }

    fn make_training_examples() -> Vec<TrainingExample> {
        vec![
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "Rust is a systems programming language.".to_string()),
                    ("question".to_string(), "What is Rust?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "A systems programming language".to_string()),
                ]),
            },
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "Python is interpreted.".to_string()),
                    ("question".to_string(), "Is Python compiled?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "No, Python is interpreted".to_string()),
                ]),
            },
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "The sky is blue.".to_string()),
                    ("question".to_string(), "What color is the sky?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "Blue".to_string()),
                ]),
            },
        ]
    }

    // --- BootstrapFewShot tests ---

    #[test]
    fn test_bootstrap_few_shot_basic() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("should succeed");
        assert!(opt.trials_run > 0);
        assert!(!opt.scores_history.is_empty());
    }

    #[test]
    fn test_bootstrap_selects_best() {
        let optimizer = BootstrapFewShot::new(3, Box::new(ContainsAnswer));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("bootstrap optimize in selects_best");
        // Best score should be the maximum in history
        let max_score = result
            .scores_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-9,
            "best_score {} should equal max in history {}",
            result.best_score,
            max_score
        );
    }

    #[test]
    fn test_bootstrap_empty_examples_error() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &[], &mut budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_single_example() {
        let optimizer = BootstrapFewShot::new(1, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = vec![make_training_examples().remove(0)];
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bootstrap_respects_budget() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(2, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("bootstrap optimize in respects_budget");
        assert!(result.trials_run <= 2);
        assert_eq!(budget.remaining(), 0);
    }

    // --- GridSearchOptimizer tests ---

    #[test]
    fn test_grid_search_basic() {
        let variants = vec![
            "Be concise.".to_string(),
            "Think step by step.".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("grid_search_basic optimize result");
        assert!(opt.trials_run > 0);
    }

    #[test]
    fn test_grid_search_selects_best() {
        let variants = vec![
            "Short".to_string(),
            "Think step by step and provide a detailed answer.".to_string(),
            "Be precise.".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ContainsAnswer));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("grid_search optimize in selects_best");
        let max_score = result
            .scores_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-9,
            "best_score should equal max in history"
        );
    }

    #[test]
    fn test_grid_search_empty_variants_error() {
        let optimizer = GridSearchOptimizer::new(vec![], Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_search_respects_budget() {
        let variants = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(2, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("grid_search optimize in respects_budget");
        assert!(result.trials_run <= 2);
    }

    #[test]
    fn test_grid_search_no_examples() {
        let variants = vec!["A".to_string()];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &[], &mut budget).expect("grid_search optimize with no examples");
        assert_eq!(result.trials_run, 1);
    }

    // --- RandomSearchOptimizer tests ---

    #[test]
    fn test_random_search_basic() {
        let optimizer = RandomSearchOptimizer::new(5, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("random_search_basic optimize result");
        assert!(opt.trials_run > 0);
        assert_eq!(opt.scores_history.len(), opt.trials_run);
    }

    #[test]
    fn test_random_search_respects_budget() {
        let optimizer = RandomSearchOptimizer::new(100, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("random_search optimize in respects_budget");
        assert!(result.trials_run <= 3);
    }

    #[test]
    fn test_random_search_mutations_differ() {
        // Different seeds should produce different mutations
        let m0 = RandomSearchOptimizer::mutate_instruction("", 0);
        let m1 = RandomSearchOptimizer::mutate_instruction("", 1);
        assert_ne!(m0, m1);
    }

    #[test]
    fn test_random_search_mutation_with_base() {
        let mutated = RandomSearchOptimizer::mutate_instruction("Base instruction.", 0);
        assert!(mutated.starts_with("Base instruction."));
        assert!(mutated.len() > "Base instruction.".len());
    }

    #[test]
    fn test_random_search_with_instructions() {
        let sig = make_qa_signature().with_instructions("Be helpful.");
        let optimizer = RandomSearchOptimizer::new(3, Box::new(ContainsAnswer));
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("random_search optimize with instructions");
        assert!(result.best_prompt.system_prompt.len() > 0);
    }

    // --- BayesianOptimizer tests ---

    #[test]
    fn test_bayesian_optimizer_basic() {
        let optimizer = BayesianOptimizer::new(5, 1.0, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("bayesian_optimizer_basic optimize result");
        assert!(opt.trials_run > 0);
    }

    #[test]
    fn test_bayesian_optimizer_respects_budget() {
        let optimizer = BayesianOptimizer::new(100, 1.0, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("bayesian optimize in respects_budget");
        assert!(result.trials_run <= 3);
    }

    #[test]
    fn test_bayesian_rbf_kernel() {
        // Same point => kernel = 1.0
        let k = BayesianOptimizer::rbf_kernel(0.0, 0.0, 1.0);
        assert!((k - 1.0).abs() < 1e-9);

        // Distant points => kernel close to 0
        let k = BayesianOptimizer::rbf_kernel(0.0, 100.0, 1.0);
        assert!(k < 1e-6);

        // Symmetric
        let k1 = BayesianOptimizer::rbf_kernel(1.0, 2.0, 1.0);
        let k2 = BayesianOptimizer::rbf_kernel(2.0, 1.0, 1.0);
        assert!((k1 - k2).abs() < 1e-12);
    }

    #[test]
    fn test_bayesian_ucb() {
        let optimizer = BayesianOptimizer::new(1, 2.0, Box::new(ExactMatch));
        let ucb = optimizer.ucb(0.5, 0.3);
        assert!((ucb - 1.1).abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_gp_predict_empty() {
        let (mean, std_dev) = BayesianOptimizer::gp_predict(&[], 0.5, 1.0, 0.01);
        assert_eq!(mean, 0.0);
        assert_eq!(std_dev, 1.0);
    }

    #[test]
    fn test_bayesian_gp_predict_single_obs() {
        let obs = vec![(0.0, 1.0)];
        let (mean, std_dev) = BayesianOptimizer::gp_predict(&obs, 0.0, 1.0, 0.01);
        // At the observed point, mean should be close to the observed value
        assert!((mean - 1.0).abs() < 0.1, "mean at observed point should be near 1.0, got {}", mean);
        assert!(std_dev < 0.5, "std_dev at observed point should be small, got {}", std_dev);
    }

    #[test]
    fn test_bayesian_solve_linear_system() {
        // 2x = 4 => x = 2
        let a = vec![vec![2.0]];
        let b = vec![4.0];
        let x = BayesianOptimizer::solve_linear_system(&a, &b);
        assert!((x[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_solve_2x2() {
        // x + y = 3, 2x + y = 5 => x=2, y=1
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
        let b = vec![3.0, 5.0];
        let x = BayesianOptimizer::solve_linear_system(&a, &b);
        assert!((x[0] - 2.0).abs() < 1e-9, "x[0] should be 2.0, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-9, "x[1] should be 1.0, got {}", x[1]);
    }

    #[test]
    fn test_bayesian_solve_empty() {
        let x = BayesianOptimizer::solve_linear_system(&[], &[]);
        assert!(x.is_empty());
    }

    #[test]
    fn test_bayesian_exploration_weight() {
        // Higher exploration weight should prefer unexplored points
        let high_exp = BayesianOptimizer::new(1, 10.0, Box::new(ExactMatch));
        let low_exp = BayesianOptimizer::new(1, 0.1, Box::new(ExactMatch));

        let ucb_high = high_exp.ucb(0.5, 1.0);
        let ucb_low = low_exp.ucb(0.5, 1.0);
        assert!(ucb_high > ucb_low);
    }

    // --- SelfReflector tests ---

    #[test]
    fn test_self_reflector_generates_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature(); // no instructions
        let compiled = sig.compile();
        let examples = make_training_examples();

        let rules = reflector.reflect(&sig, &compiled, &examples);
        assert!(!rules.is_empty(), "reflector should generate at least one rule");
    }

    #[test]
    fn test_self_reflector_no_instructions_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test"); // no instructions
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        let has_no_instructions = rules
            .iter()
            .any(|r| r.condition == "no_instructions");
        assert!(has_no_instructions, "should detect missing instructions");
    }

    #[test]
    fn test_self_reflector_no_examples_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("be helpful");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        let has_no_examples = rules
            .iter()
            .any(|r| r.condition == "no_examples");
        assert!(has_no_examples, "should detect no examples");
    }

    #[test]
    fn test_self_reflector_few_examples_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("be helpful");
        let compiled = sig.compile();
        let examples = vec![make_training_examples().remove(0)]; // just 1

        let rules = reflector.reflect(&sig, &compiled, &examples);
        let has_few_examples = rules
            .iter()
            .any(|r| r.condition == "few_examples");
        assert!(has_few_examples, "should detect few examples");
    }

    #[test]
    fn test_self_reflector_max_iterations_limits_rules() {
        let reflector = SelfReflector::new(2, 0.05);
        let sig = Signature::new("test", "Test"); // will generate many rules
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        assert!(rules.len() <= 2, "should be limited to max_iterations=2");
    }

    #[test]
    fn test_self_reflector_incomplete_input_coverage() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("inst");
        let compiled = sig.compile();

        // Example missing the "context" input
        let incomplete = vec![TrainingExample {
            inputs: HashMap::from([
                ("question".to_string(), "What?".to_string()),
            ]),
            expected_outputs: HashMap::from([
                ("answer".to_string(), "Something".to_string()),
            ]),
        }];

        let rules = reflector.reflect(&sig, &compiled, &incomplete);
        let has_coverage_rule = rules
            .iter()
            .any(|r| r.condition.starts_with("incomplete_input_coverage:"));
        assert!(has_coverage_rule, "should detect incomplete input coverage");
    }

    #[test]
    fn test_self_reflector_incomplete_output_coverage() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("inst");
        let compiled = sig.compile();

        // Example missing the "answer" output
        let incomplete = vec![TrainingExample {
            inputs: HashMap::from([
                ("context".to_string(), "ctx".to_string()),
                ("question".to_string(), "q".to_string()),
            ]),
            expected_outputs: HashMap::new(),
        }];

        let rules = reflector.reflect(&sig, &compiled, &incomplete);
        let has_coverage_rule = rules
            .iter()
            .any(|r| r.condition.starts_with("incomplete_output_coverage:"));
        assert!(has_coverage_rule, "should detect incomplete output coverage");
    }

    #[test]
    fn test_self_reflector_typed_output_guidance() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("typed", "Typed outputs")
            .add_output(
                SignatureField::new("count", "a count")
                    .with_type(FieldType::Number),
            )
            .with_instructions("inst");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &make_training_examples());
        // The compiled prompt does contain "number" in field descriptions,
        // so this may or may not trigger depending on compilation output.
        // Just verify reflect doesn't panic and returns valid rules.
        for rule in &rules {
            assert!(!rule.condition.is_empty());
            assert!(!rule.action.is_empty());
        }
    }

    #[test]
    fn test_self_reflector_ambiguous_types() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("ambig", "Ambiguous outputs")
            .add_output(SignatureField::new("a", "first text"))
            .add_output(SignatureField::new("b", "second text"))
            .with_instructions("inst");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &make_training_examples());
        let has_ambig = rules
            .iter()
            .any(|r| r.condition.starts_with("ambiguous_output_types:"));
        assert!(has_ambig, "should detect ambiguous output types");
    }

    #[test]
    fn test_self_reflector_apply_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test"); // no instructions

        let rules = vec![
            ImprovementRule {
                condition: "no_instructions".to_string(),
                action: "Add instructions".to_string(),
                applied: false,
            },
        ];

        let improved = reflector.apply_rules(&sig, &rules);
        assert!(improved.instructions.is_some());
        assert!(
            improved.instructions.as_deref().unwrap_or("").contains("Follow the field descriptions"),
            "should add default instructions"
        );
    }

    #[test]
    fn test_self_reflector_apply_type_guidance_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test")
            .add_output(
                SignatureField::new("count", "a count")
                    .with_type(FieldType::Number),
            )
            .with_instructions("Existing.");

        let rules = vec![
            ImprovementRule {
                condition: "missing_type_guidance:count".to_string(),
                action: "Add format guidance for count".to_string(),
                applied: false,
            },
        ];

        let improved = reflector.apply_rules(&sig, &rules);
        let inst = improved.instructions.expect("should have instructions");
        assert!(inst.contains("number"), "should mention number format");
    }

    #[test]
    fn test_self_reflector_apply_empty_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature();
        let improved = reflector.apply_rules(&sig, &[]);
        // With no rules, the signature should be unchanged
        assert_eq!(improved.name, sig.name);
        assert_eq!(improved.instructions, sig.instructions);
    }

    // --- ImprovementRule tests ---

    #[test]
    fn test_improvement_rule_serialization() {
        let rule = ImprovementRule {
            condition: "test_cond".to_string(),
            action: "test_action".to_string(),
            applied: false,
        };
        let json = serde_json::to_string(&rule).expect("serialize");
        let back: ImprovementRule = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.condition, "test_cond");
        assert_eq!(back.action, "test_action");
        assert!(!back.applied);
    }

    #[test]
    fn test_improvement_rule_applied_flag() {
        let mut rule = ImprovementRule {
            condition: "c".to_string(),
            action: "a".to_string(),
            applied: false,
        };
        assert!(!rule.applied);
        rule.applied = true;
        assert!(rule.applied);
    }

    // --- PromptExample tests ---

    #[test]
    fn test_prompt_example_serialization() {
        let ex = PromptExample {
            inputs: HashMap::from([("k".to_string(), "v".to_string())]),
            outputs: HashMap::from([("o".to_string(), "r".to_string())]),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: PromptExample = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.inputs.get("k").map(|s| s.as_str()), Some("v"));
    }

    // --- Integration-style tests ---

    #[test]
    fn test_end_to_end_compile_and_render() {
        let sig = Signature::new("summarize", "Summarize text")
            .add_input(SignatureField::new("text", "The text to summarize"))
            .add_output(SignatureField::new("summary", "A concise summary"))
            .with_instructions("Keep it under 50 words.");

        let compiled = sig.compile();
        let inputs = HashMap::from([(
            "text".to_string(),
            "Rust is a multi-paradigm, general-purpose programming language.".to_string(),
        )]);

        sig.validate_inputs(&inputs).expect("validation ok");
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("Rust is a multi-paradigm"));
    }

    #[test]
    fn test_end_to_end_optimize_and_reflect() {
        let sig = make_qa_signature();
        let examples = make_training_examples();

        // Optimize
        let optimizer = BootstrapFewShot::new(2, Box::new(ContainsAnswer));
        let mut budget = EvaluationBudget::new(5, 5);
        let opt_result = optimizer.optimize(&sig, &examples, &mut budget).expect("optimize in end_to_end_optimize_and_reflect");

        // Reflect on the result
        let reflector = SelfReflector::new(10, 0.05);
        let rules = reflector.reflect(&sig, &opt_result.best_prompt, &examples);

        // Apply improvements
        let improved_sig = reflector.apply_rules(&sig, &rules);
        let improved_compiled = improved_sig.compile();
        assert!(improved_compiled.system_prompt.len() > 0);
    }

    #[test]
    fn test_multiple_optimizers_comparison() {
        let sig = make_qa_signature();
        let examples = make_training_examples();

        // Bootstrap
        let bs = BootstrapFewShot::new(2, Box::new(ContainsAnswer));
        let mut budget1 = EvaluationBudget::new(3, 5);
        let r1 = bs.optimize(&sig, &examples, &mut budget1).expect("bootstrap in optimizers_comparison");

        // Grid search
        let gs = GridSearchOptimizer::new(
            vec!["Be precise.".to_string(), "Think carefully.".to_string()],
            Box::new(ContainsAnswer),
        );
        let mut budget2 = EvaluationBudget::new(3, 5);
        let r2 = gs.optimize(&sig, &examples, &mut budget2).expect("grid_search in optimizers_comparison");

        // Random search
        let rs = RandomSearchOptimizer::new(3, Box::new(ContainsAnswer));
        let mut budget3 = EvaluationBudget::new(3, 5);
        let r3 = rs.optimize(&sig, &examples, &mut budget3).expect("random_search in optimizers_comparison");

        // All should produce valid results
        assert!(r1.trials_run > 0);
        assert!(r2.trials_run > 0);
        assert!(r3.trials_run > 0);
    }

    #[test]
    fn test_signature_many_fields() {
        let sig = Signature::new("multi", "Multi-field signature")
            .add_input(SignatureField::new("a", "field a"))
            .add_input(SignatureField::new("b", "field b"))
            .add_input(SignatureField::new("c", "field c").optional())
            .add_output(SignatureField::new("x", "output x"))
            .add_output(SignatureField::new("y", "output y"))
            .add_output(
                SignatureField::new("z", "output z")
                    .with_type(FieldType::Json)
                    .optional(),
            );

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("field a"));
        assert!(compiled.system_prompt.contains("output z"));
        assert!(compiled.user_template.contains("{a}"));
        assert!(compiled.user_template.contains("{b}"));
    }

    #[test]
    fn test_field_type_equality() {
        assert_eq!(FieldType::Text, FieldType::Text);
        assert_eq!(FieldType::Number, FieldType::Number);
        assert_eq!(FieldType::Boolean, FieldType::Boolean);
        assert_eq!(FieldType::List, FieldType::List);
        assert_eq!(FieldType::Json, FieldType::Json);
        assert_ne!(FieldType::Text, FieldType::Json);
        assert_ne!(FieldType::Number, FieldType::Boolean);
    }

    #[test]
    fn test_all_field_types_in_signature() {
        let sig = Signature::new("all_types", "All types test")
            .add_input(SignatureField::new("t", "text").with_type(FieldType::Text))
            .add_input(SignatureField::new("n", "number").with_type(FieldType::Number))
            .add_input(SignatureField::new("b", "bool").with_type(FieldType::Boolean))
            .add_input(SignatureField::new("l", "list").with_type(FieldType::List))
            .add_input(SignatureField::new("j", "json").with_type(FieldType::Json));

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("text"));
        assert!(compiled.system_prompt.contains("number"));
        assert!(compiled.system_prompt.contains("boolean"));
        assert!(compiled.system_prompt.contains("list"));
        assert!(compiled.system_prompt.contains("json"));
    }

    // ========================================================================
    // GEPA (Genetic Pareto Optimizer) tests
    // ========================================================================

    #[test]
    fn test_pareto_dominance_a_dominated_by_b() {
        let a = [0.5, 0.5];
        let b = [0.8, 0.9];
        assert!(ParetoFront::is_dominated(&a, &b));
    }

    #[test]
    fn test_pareto_dominance_not_dominated_when_equal() {
        let a = [0.5, 0.5];
        let b = [0.5, 0.5];
        // Equal on all objectives => not dominated (need strictly better on at least one)
        assert!(!ParetoFront::is_dominated(&a, &b));
    }

    #[test]
    fn test_pareto_dominance_not_dominated_when_mixed() {
        let a = [0.5, 0.9];
        let b = [0.8, 0.3];
        // b is better on first, worse on second => no domination
        assert!(!ParetoFront::is_dominated(&a, &b));
        assert!(!ParetoFront::is_dominated(&b, &a));
    }

    #[test]
    fn test_pareto_dominance_strictly_better_one_dim() {
        let a = [0.5, 0.5];
        let b = [0.5, 0.6]; // same on first, strictly better on second
        assert!(ParetoFront::is_dominated(&a, &b));
    }

    #[test]
    fn test_pareto_dominance_empty_scores() {
        assert!(!ParetoFront::is_dominated(&[], &[]));
    }

    #[test]
    fn test_pareto_dominance_mismatched_lengths() {
        let a = [0.5];
        let b = [0.5, 0.6];
        assert!(!ParetoFront::is_dominated(&a, &b));
    }

    #[test]
    fn test_pareto_non_dominated_sorting_basic() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };

        let mut solutions = vec![
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![1.0, 0.8],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.8, 1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.3, 0.3],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        ParetoFront::compute(&mut solutions);

        // First two are non-dominated (Pareto front 0)
        assert_eq!(solutions[0].rank, 0);
        assert_eq!(solutions[1].rank, 0);
        // Third is dominated by both (rank 1): [0.3,0.3] < [1.0,0.8] and [0.3,0.3] < [0.8,1.0]
        assert_eq!(solutions[2].rank, 1);
    }

    #[test]
    fn test_pareto_crowding_distance_boundary_points() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };

        let mut solutions = vec![
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![1.0, 0.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.5, 0.5],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.0, 1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        ParetoFront::compute(&mut solutions);

        // Boundary points should have infinite crowding distance
        assert!(solutions[0].crowding_distance.is_infinite());
        assert!(solutions[2].crowding_distance.is_infinite());
        // Middle point should have finite crowding distance
        assert!(solutions[1].crowding_distance.is_finite());
        assert!(solutions[1].crowding_distance > 0.0);
    }

    #[test]
    fn test_pareto_front_get_front() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };

        let mut solutions = vec![
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![1.0, 0.8],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.8, 1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.3, 0.3],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        ParetoFront::compute(&mut solutions);

        let front = ParetoFront { solutions };
        let rank0 = front.get_front(0);
        let rank1 = front.get_front(1);

        // [1.0,0.8] and [0.8,1.0] are non-dominated; [0.3,0.3] is dominated
        assert_eq!(rank0.len(), 2);
        assert_eq!(rank1.len(), 1);
        assert_eq!(front.get_front(5).len(), 0);
    }

    #[test]
    fn test_pareto_compute_empty() {
        let mut solutions: Vec<ParetoSolution> = Vec::new();
        ParetoFront::compute(&mut solutions);
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_pareto_compute_single_solution() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };
        let mut solutions = vec![ParetoSolution {
            compiled,
            scores: vec![0.5, 0.5],
            rank: 99,
            crowding_distance: 0.0,
        }];
        ParetoFront::compute(&mut solutions);
        assert_eq!(solutions[0].rank, 0);
        assert!(solutions[0].crowding_distance.is_infinite());
    }

    #[test]
    fn test_gepa_config_default() {
        let config = GEPAConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.generations, 10);
        assert!((config.mutation_rate - 0.1).abs() < 1e-9);
        assert!((config.crossover_rate - 0.7).abs() < 1e-9);
        assert_eq!(config.elitism_count, 2);
        assert_eq!(config.tournament_size, 3);
    }

    #[test]
    fn test_gepa_population_initialization() {
        let config = GEPAConfig {
            population_size: 5,
            ..GEPAConfig::default()
        };
        let optimizer = GEPAOptimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();

        let pop = optimizer.initialize_population(&sig, &examples);
        assert_eq!(pop.len(), 5);
        // Each compiled prompt should have a non-empty system prompt
        for p in &pop {
            assert!(!p.system_prompt.is_empty());
        }
    }

    #[test]
    fn test_gepa_tournament_selection() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };
        let solutions = vec![
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.5],
                rank: 1,
                crowding_distance: 0.5,
            },
            ParetoSolution {
                compiled: compiled.clone(),
                scores: vec![0.8],
                rank: 0,
                crowding_distance: 1.0,
            },
        ];

        let config = GEPAConfig {
            tournament_size: 2,
            population_size: 2,
            ..GEPAConfig::default()
        };
        let optimizer = GEPAOptimizer::new(config);
        let selected = optimizer.select_parent(&solutions, 0);
        // Should prefer rank 0 over rank 1
        assert_eq!(selected.rank, 0);
    }

    #[test]
    fn test_gepa_crossover_combines_demos() {
        let parent_a = CompiledPrompt {
            system_prompt: "Longer system prompt for parent A".to_string(),
            user_template: "usr".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "A1".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "R1".to_string())]),
                },
            ],
        };
        let parent_b = CompiledPrompt {
            system_prompt: "Short".to_string(),
            user_template: "usr".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "B1".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "R2".to_string())]),
                },
            ],
        };

        let child = GEPAOptimizer::crossover(&parent_a, &parent_b, 42);
        // Child should have examples from one or both parents
        assert!(!child.examples.is_empty());
        // Should pick longer system prompt
        assert!(child.system_prompt.contains("parent A"));
    }

    #[test]
    fn test_gepa_mutation_changes_prompt() {
        let compiled = CompiledPrompt {
            system_prompt: "Original system prompt".to_string(),
            user_template: "usr".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "Q1".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "A1".to_string())]),
                },
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "Q2".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "A2".to_string())]),
                },
            ],
        };

        // Mutation type 3 (seed % 4 == 3) perturbs instruction text
        let mutated = GEPAOptimizer::mutate(&compiled, 3);
        assert_ne!(mutated.system_prompt, compiled.system_prompt);
    }

    #[test]
    fn test_gepa_mutation_swap_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "first".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "A1".to_string())]),
                },
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "second".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "A2".to_string())]),
                },
            ],
        };

        // seed % 4 == 0 => swap
        let mutated = GEPAOptimizer::mutate(&compiled, 4);
        // After swap, order should differ (or not, depending on indices)
        assert_eq!(mutated.examples.len(), 2);
    }

    #[test]
    fn test_gepa_optimize_basic() {
        let config = GEPAConfig {
            population_size: 5,
            generations: 2,
            mutation_rate: 0.3,
            crossover_rate: 0.5,
            elitism_count: 1,
            tournament_size: 2,
        };
        let optimizer = GEPAOptimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric1 = ExactMatch;
        let metric2 = ContainsAnswer;
        let metrics: Vec<&dyn EvalMetric> = vec![&metric1, &metric2];
        let mut budget = EvaluationBudget::new(20, 10);

        let result = optimizer.optimize(&sig, &examples, &metrics, &mut budget);
        assert!(result.is_ok());
        let front = result.expect("GEPA optimize in gepa_optimize_basic");
        assert!(!front.solutions.is_empty());
        // All solutions should have been scored
        for sol in &front.solutions {
            assert_eq!(sol.scores.len(), 2);
        }
    }

    #[test]
    fn test_gepa_optimize_no_metrics_error() {
        let optimizer = GEPAOptimizer::new(GEPAConfig::default());
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metrics: Vec<&dyn EvalMetric> = vec![];
        let mut budget = EvaluationBudget::new(10, 5);

        let result = optimizer.optimize(&sig, &examples, &metrics, &mut budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_gepa_optimize_empty_examples() {
        let config = GEPAConfig {
            population_size: 3,
            generations: 1,
            ..GEPAConfig::default()
        };
        let optimizer = GEPAOptimizer::new(config);
        let sig = make_qa_signature();
        let metric = ExactMatch;
        let metrics: Vec<&dyn EvalMetric> = vec![&metric];
        let mut budget = EvaluationBudget::new(10, 5);

        let result = optimizer.optimize(&sig, &[], &metrics, &mut budget);
        assert!(result.is_ok());
    }

    // ========================================================================
    // MIPROv2 tests
    // ========================================================================

    #[test]
    fn test_miprov2_config_default() {
        let config = MIPROv2Config::default();
        assert_eq!(config.max_bootstrapped_demos, 8);
        assert_eq!(config.max_labeled_demos, 4);
        assert_eq!(config.num_instruction_candidates, 5);
        assert_eq!(config.num_trials, 10);
        assert_eq!(config.search_strategy, DiscreteSearchStrategy::Random);
    }

    #[test]
    fn test_instruction_proposer_generates_candidates() {
        let sig = make_qa_signature();
        let demos = vec![PromptExample {
            inputs: HashMap::from([("context".to_string(), "Some context text".to_string())]),
            outputs: HashMap::from([("answer".to_string(), "An answer".to_string())]),
        }];

        let candidates = InstructionProposer::propose(&sig, &demos, 5);
        assert_eq!(candidates.len(), 5);
        for c in &candidates {
            assert!(!c.is_empty());
            // Each candidate should reference the task
            assert!(
                c.contains("Answer questions") || c.contains("answer questions"),
                "candidate should reference task: {}",
                c
            );
        }
    }

    #[test]
    fn test_instruction_proposer_no_demos() {
        let sig = make_qa_signature();
        let candidates = InstructionProposer::propose(&sig, &[], 3);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_instruction_proposer_includes_field_names() {
        let sig = make_qa_signature();
        let candidates = InstructionProposer::propose(&sig, &[], 2);
        for c in &candidates {
            assert!(c.contains("context") || c.contains("question") || c.contains("answer"),
                "candidate should mention field names: {}", c);
        }
    }

    #[test]
    fn test_miprov2_bootstrap_stage() {
        let config = MIPROv2Config::default();
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ContainsAnswer;

        let demos = optimizer.bootstrap_demos(&sig, &examples, &metric);
        // Should have some demos (bootstrapped + labeled)
        assert!(!demos.is_empty());
    }

    #[test]
    fn test_miprov2_full_pipeline_random() {
        let config = MIPROv2Config {
            num_instruction_candidates: 3,
            num_trials: 5,
            search_strategy: DiscreteSearchStrategy::Random,
            ..MIPROv2Config::default()
        };
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ContainsAnswer;
        let mut budget = EvaluationBudget::new(20, 10);

        let result = optimizer.optimize(&sig, &examples, &metric, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("MIPROv2 optimize in pipeline_random");
        assert!(opt.trials_run > 0);
        assert!(!opt.scores_history.is_empty());
    }

    #[test]
    fn test_miprov2_full_pipeline_exhaustive() {
        let config = MIPROv2Config {
            num_instruction_candidates: 2,
            num_trials: 10,
            search_strategy: DiscreteSearchStrategy::Exhaustive,
            max_bootstrapped_demos: 2,
            max_labeled_demos: 2,
        };
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ExactMatch;
        let mut budget = EvaluationBudget::new(50, 10);

        let result = optimizer.optimize(&sig, &examples, &metric, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("MIPROv2 optimize in pipeline_exhaustive");
        assert!(opt.trials_run > 0);
    }

    #[test]
    fn test_miprov2_full_pipeline_bayesian() {
        let config = MIPROv2Config {
            num_instruction_candidates: 3,
            num_trials: 4,
            search_strategy: DiscreteSearchStrategy::Bayesian,
            ..MIPROv2Config::default()
        };
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ContainsAnswer;
        let mut budget = EvaluationBudget::new(10, 5);

        let result = optimizer.optimize(&sig, &examples, &metric, &mut budget);
        assert!(result.is_ok());
    }

    #[test]
    fn test_miprov2_selects_best() {
        let config = MIPROv2Config {
            num_instruction_candidates: 3,
            num_trials: 6,
            search_strategy: DiscreteSearchStrategy::Random,
            ..MIPROv2Config::default()
        };
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ContainsAnswer;
        let mut budget = EvaluationBudget::new(20, 10);

        let opt = optimizer.optimize(&sig, &examples, &metric, &mut budget).expect("MIPROv2 optimize in selects_best");
        let max_score = opt
            .scores_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (opt.best_score - max_score).abs() < 1e-9,
            "best_score {} should equal max in history {}",
            opt.best_score,
            max_score
        );
    }

    #[test]
    fn test_miprov2_respects_budget() {
        let config = MIPROv2Config {
            num_instruction_candidates: 5,
            num_trials: 100,
            search_strategy: DiscreteSearchStrategy::Random,
            ..MIPROv2Config::default()
        };
        let optimizer = MIPROv2Optimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let metric = ExactMatch;
        let mut budget = EvaluationBudget::new(3, 5);

        let opt = optimizer.optimize(&sig, &examples, &metric, &mut budget).expect("MIPROv2 optimize in respects_budget");
        assert!(opt.trials_run <= 3);
    }

    // ========================================================================
    // Prompt Assertions & Constraints tests
    // ========================================================================

    #[test]
    fn test_length_assertion_pass() {
        let assertion = LengthAssertion {
            min_chars: Some(5),
            max_chars: Some(100),
            min_tokens: None,
            max_tokens: None,
        };
        assert_eq!(assertion.check("Hello World"), AssertionResult::Pass);
        assert_eq!(assertion.name(), "length_assertion");
    }

    #[test]
    fn test_length_assertion_fail_too_short() {
        let assertion = LengthAssertion {
            min_chars: Some(20),
            max_chars: None,
            min_tokens: None,
            max_tokens: None,
        };
        match assertion.check("Hi") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("below minimum"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_length_assertion_fail_too_long() {
        let assertion = LengthAssertion {
            min_chars: None,
            max_chars: Some(5),
            min_tokens: None,
            max_tokens: None,
        };
        match assertion.check("This is way too long") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("exceeds maximum"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_length_assertion_tokens() {
        let assertion = LengthAssertion {
            min_chars: None,
            max_chars: None,
            min_tokens: Some(2),
            max_tokens: Some(5),
        };
        assert_eq!(assertion.check("two words"), AssertionResult::Pass);

        match assertion.check("one") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("below minimum"));
            }
            other => panic!("Expected Fail for too few tokens, got {:?}", other),
        }

        match assertion.check("one two three four five six seven") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("exceeds maximum"));
            }
            other => panic!("Expected Fail for too many tokens, got {:?}", other),
        }
    }

    #[test]
    fn test_format_assertion_pass() {
        let assertion = FormatAssertion {
            pattern: "result:".to_string(),
        };
        assert_eq!(assertion.check("The result: 42"), AssertionResult::Pass);
        assert_eq!(assertion.name(), "format_assertion");
    }

    #[test]
    fn test_format_assertion_fail() {
        let assertion = FormatAssertion {
            pattern: "JSON:".to_string(),
        };
        match assertion.check("This has no json marker") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("does not match pattern"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_contains_assertion_pass_case_sensitive() {
        let assertion = ContainsAssertion {
            required_keywords: vec!["Rust".to_string(), "language".to_string()],
            case_sensitive: true,
        };
        assert_eq!(
            assertion.check("Rust is a programming language"),
            AssertionResult::Pass
        );
        assert_eq!(assertion.name(), "contains_assertion");
    }

    #[test]
    fn test_contains_assertion_fail_case_sensitive() {
        let assertion = ContainsAssertion {
            required_keywords: vec!["Rust".to_string()],
            case_sensitive: true,
        };
        match assertion.check("rust is lowercase") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("missing required keyword"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_contains_assertion_pass_case_insensitive() {
        let assertion = ContainsAssertion {
            required_keywords: vec!["Rust".to_string()],
            case_sensitive: false,
        };
        assert_eq!(
            assertion.check("rust is great"),
            AssertionResult::Pass
        );
    }

    #[test]
    fn test_contains_assertion_fail_missing_keyword() {
        let assertion = ContainsAssertion {
            required_keywords: vec!["python".to_string(), "java".to_string()],
            case_sensitive: false,
        };
        match assertion.check("python is nice") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("java"));
            }
            other => panic!("Expected Fail for missing java, got {:?}", other),
        }
    }

    #[test]
    fn test_json_schema_assertion_valid() {
        let assertion = JsonSchemaAssertion;
        assert_eq!(
            assertion.check(r#"{"key": "value", "num": 42}"#),
            AssertionResult::Pass
        );
        assert_eq!(assertion.name(), "json_schema_assertion");
    }

    #[test]
    fn test_json_schema_assertion_invalid() {
        let assertion = JsonSchemaAssertion;
        match assertion.check("not valid json {{{") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("not valid JSON"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_json_schema_assertion_valid_array() {
        let assertion = JsonSchemaAssertion;
        assert_eq!(assertion.check("[1, 2, 3]"), AssertionResult::Pass);
    }

    #[test]
    fn test_json_schema_assertion_valid_primitive() {
        let assertion = JsonSchemaAssertion;
        assert_eq!(assertion.check("42"), AssertionResult::Pass);
        assert_eq!(assertion.check("true"), AssertionResult::Pass);
        assert_eq!(assertion.check("\"hello\""), AssertionResult::Pass);
    }

    #[test]
    fn test_custom_assertion() {
        let assertion = CustomAssertion::new(
            "starts_with_hello",
            Box::new(|output: &str| {
                if output.starts_with("Hello") {
                    AssertionResult::Pass
                } else {
                    AssertionResult::Fail {
                        reason: "Does not start with Hello".to_string(),
                    }
                }
            }),
        );
        assert_eq!(assertion.name(), "starts_with_hello");
        assert_eq!(assertion.check("Hello world"), AssertionResult::Pass);
        match assertion.check("Goodbye world") {
            AssertionResult::Fail { reason } => {
                assert!(reason.contains("Hello"));
            }
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_custom_assertion_warn() {
        let assertion = CustomAssertion::new(
            "warn_if_short",
            Box::new(|output: &str| {
                if output.len() < 5 {
                    AssertionResult::Warn {
                        reason: "Output is suspiciously short".to_string(),
                    }
                } else {
                    AssertionResult::Pass
                }
            }),
        );
        match assertion.check("Hi") {
            AssertionResult::Warn { reason } => {
                assert!(reason.contains("short"));
            }
            other => panic!("Expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn test_asserted_signature_check_output_all_pass() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1),
            max_chars: Some(100),
            min_tokens: None,
            max_tokens: None,
        }));
        asserted.add_assertion(Box::new(ContainsAssertion {
            required_keywords: vec!["result".to_string()],
            case_sensitive: false,
        }));

        let results = asserted.check_output("The result is 42");
        assert_eq!(results.len(), 2);
        for (_, r) in &results {
            assert_eq!(*r, AssertionResult::Pass);
        }
    }

    #[test]
    fn test_asserted_signature_check_output_mixed() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1),
            max_chars: Some(100),
            min_tokens: None,
            max_tokens: None,
        }));
        asserted.add_assertion(Box::new(ContainsAssertion {
            required_keywords: vec!["missing_keyword".to_string()],
            case_sensitive: false,
        }));

        let results = asserted.check_output("Hello world");
        // First assertion passes, second fails
        assert_eq!(results[0].1, AssertionResult::Pass);
        match &results[1].1 {
            AssertionResult::Fail { .. } => {}
            other => panic!("Expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn test_assertion_penalty_all_pass() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1),
            max_chars: None,
            min_tokens: None,
            max_tokens: None,
        }));

        let penalty = asserted.assertion_penalty("Hello");
        assert!((penalty - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_assertion_penalty_all_fail() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1000),
            max_chars: None,
            min_tokens: None,
            max_tokens: None,
        }));

        let penalty = asserted.assertion_penalty("Hi");
        assert!((penalty - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_assertion_penalty_mixed() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        // This one passes
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1),
            max_chars: None,
            min_tokens: None,
            max_tokens: None,
        }));
        // This one fails
        asserted.add_assertion(Box::new(LengthAssertion {
            min_chars: Some(1000),
            max_chars: None,
            min_tokens: None,
            max_tokens: None,
        }));

        let penalty = asserted.assertion_penalty("Hello");
        // 1 pass (0) + 1 fail (1) = 0.5
        assert!((penalty - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_assertion_penalty_with_warn() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(CustomAssertion::new(
            "warn",
            Box::new(|_| AssertionResult::Warn {
                reason: "warning".to_string(),
            }),
        )));

        let penalty = asserted.assertion_penalty("anything");
        // 1 warn = 0.5 weight
        assert!((penalty - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_assertion_penalty_empty_assertions() {
        let sig = make_qa_signature();
        let asserted = AssertedSignature::new(sig);
        let penalty = asserted.assertion_penalty("anything");
        assert!((penalty - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_asserted_signature_names_in_results() {
        let sig = make_qa_signature();
        let mut asserted = AssertedSignature::new(sig);
        asserted.add_assertion(Box::new(JsonSchemaAssertion));
        asserted.add_assertion(Box::new(FormatAssertion {
            pattern: "test".to_string(),
        }));

        let results = asserted.check_output(r#"{"test": true}"#);
        assert_eq!(results[0].0, "json_schema_assertion");
        assert_eq!(results[1].0, "format_assertion");
    }

    // ========================================================================
    // LM Adapters tests
    // ========================================================================

    #[test]
    fn test_chat_adapter_basic_formatting() {
        let compiled = CompiledPrompt {
            system_prompt: "You are helpful.".to_string(),
            user_template: "Question: {q}".to_string(),
            examples: vec![],
        };
        let adapter = ChatAdapter;
        let formatted = adapter.format_for_provider(&compiled, "openai");

        assert!(formatted.system_message.is_some());
        assert_eq!(
            formatted.system_message.as_deref(),
            Some("You are helpful.")
        );
        // Should have system + user query = 2 messages
        assert_eq!(formatted.messages.len(), 2);
        assert_eq!(formatted.messages[0].role, "system");
        assert_eq!(formatted.messages[1].role, "user");
        assert!(formatted.raw_prompt.is_none());
    }

    #[test]
    fn test_chat_adapter_with_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "Sys".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "2+2?".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "4".to_string())]),
                },
            ],
        };
        let adapter = ChatAdapter;
        let formatted = adapter.format_for_provider(&compiled, "anthropic");

        // system + user demo + assistant demo + user query = 4 messages
        assert_eq!(formatted.messages.len(), 4);
        assert_eq!(formatted.messages[0].role, "system");
        assert_eq!(formatted.messages[1].role, "user");
        assert_eq!(formatted.messages[2].role, "assistant");
        assert_eq!(formatted.messages[3].role, "user");
    }

    #[test]
    fn test_completion_adapter_formatting() {
        let compiled = CompiledPrompt {
            system_prompt: "Task description".to_string(),
            user_template: "Input: {x}".to_string(),
            examples: vec![],
        };
        let adapter = CompletionAdapter;
        let formatted = adapter.format_for_provider(&compiled, "ollama");

        assert!(formatted.raw_prompt.is_some());
        let raw = formatted.raw_prompt.as_deref().unwrap();
        assert!(raw.contains("Task description"));
        assert!(raw.contains("---Input:---"));
        assert!(raw.contains("---Output:---"));
        assert!(formatted.messages.is_empty());
        assert!(formatted.system_message.is_none());
    }

    #[test]
    fn test_completion_adapter_with_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "Sys".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "Hi".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "Hello".to_string())]),
                },
            ],
        };
        let adapter = CompletionAdapter;
        let formatted = adapter.format_for_provider(&compiled, "lmstudio");

        let raw = formatted.raw_prompt.as_deref().unwrap();
        assert!(raw.contains("Hi"));
        assert!(raw.contains("Hello"));
    }

    #[test]
    fn test_function_calling_adapter_formatting() {
        let compiled = CompiledPrompt {
            system_prompt: "Structured output task".to_string(),
            user_template: "Input: {x}".to_string(),
            examples: vec![],
        };
        let adapter = FunctionCallingAdapter;
        let formatted = adapter.format_for_provider(&compiled, "openai");

        assert!(formatted.system_message.is_some());
        let sys = formatted.system_message.as_deref().unwrap();
        assert!(sys.contains("respond"));
        assert!(sys.contains("function"));
        // system + user query = 2 messages
        assert_eq!(formatted.messages.len(), 2);
    }

    #[test]
    fn test_function_calling_adapter_with_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "Task".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![
                PromptExample {
                    inputs: HashMap::from([("q".to_string(), "test".to_string())]),
                    outputs: HashMap::from([("a".to_string(), "result".to_string())]),
                },
            ],
        };
        let adapter = FunctionCallingAdapter;
        let formatted = adapter.format_for_provider(&compiled, "openai");

        // system + user demo + assistant fn_call + user query = 4 messages
        assert_eq!(formatted.messages.len(), 4);
        // The assistant message should contain respond(...)
        assert!(formatted.messages[2].content.contains("respond("));
    }

    #[test]
    fn test_adapter_router_basic_routing() {
        let mut router = AdapterRouter::new();
        router.register("openai", Box::new(ChatAdapter));
        router.register("ollama", Box::new(CompletionAdapter));

        assert!(router.route("openai-gpt4").is_some());
        assert!(router.route("Ollama-Local").is_some());
    }

    #[test]
    fn test_adapter_router_unknown_provider() {
        let mut router = AdapterRouter::new();
        router.register("openai", Box::new(ChatAdapter));

        assert!(router.route("unknown-provider").is_none());
    }

    #[test]
    fn test_adapter_router_case_insensitive() {
        let mut router = AdapterRouter::new();
        router.register("anthropic", Box::new(ChatAdapter));

        assert!(router.route("ANTHROPIC").is_some());
        assert!(router.route("Anthropic-Claude").is_some());
    }

    #[test]
    fn test_adapter_router_empty() {
        let router = AdapterRouter::new();
        assert!(router.route("anything").is_none());
    }

    #[test]
    fn test_adapter_router_first_match_wins() {
        let mut router = AdapterRouter::new();
        router.register("open", Box::new(ChatAdapter));
        router.register("openai", Box::new(CompletionAdapter));

        // "open" matches first
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "usr".to_string(),
            examples: vec![],
        };
        let adapter = router.route("openai").unwrap();
        let formatted = adapter.format_for_provider(&compiled, "openai");
        // ChatAdapter sets system_message, CompletionAdapter does not
        assert!(formatted.system_message.is_some());
    }

    #[test]
    fn test_adapter_router_format_end_to_end() {
        let mut router = AdapterRouter::new();
        router.register("openai", Box::new(ChatAdapter));
        router.register("ollama", Box::new(CompletionAdapter));
        router.register("fn-", Box::new(FunctionCallingAdapter));

        let compiled = CompiledPrompt {
            system_prompt: "Task".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![PromptExample {
                inputs: HashMap::from([("q".to_string(), "test".to_string())]),
                outputs: HashMap::from([("a".to_string(), "result".to_string())]),
            }],
        };

        // Test each adapter through the router
        let chat = router.route("openai-gpt4").unwrap();
        let chat_fmt = chat.format_for_provider(&compiled, "openai-gpt4");
        assert!(chat_fmt.system_message.is_some());
        assert!(!chat_fmt.messages.is_empty());

        let completion = router.route("ollama-llama").unwrap();
        let comp_fmt = completion.format_for_provider(&compiled, "ollama-llama");
        assert!(comp_fmt.raw_prompt.is_some());

        let fn_call = router.route("fn-caller").unwrap();
        let fn_fmt = fn_call.format_for_provider(&compiled, "fn-caller");
        assert!(fn_fmt.system_message.as_deref().unwrap().contains("respond"));
    }

    #[test]
    fn test_formatted_prompt_message_roles() {
        let msg = FormattedMessage {
            role: "system".to_string(),
            content: "Hello".to_string(),
        };
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_formatted_prompt_clone() {
        let prompt = FormattedPrompt {
            system_message: Some("sys".to_string()),
            messages: vec![FormattedMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            raw_prompt: None,
        };
        let cloned = prompt.clone();
        assert_eq!(cloned.system_message, prompt.system_message);
        assert_eq!(cloned.messages.len(), 1);
    }

    // ========================================================================
    // 4.1 SIMBA Optimizer tests
    // ========================================================================

    #[test]
    fn test_cooling_schedule_linear_temperature_decreases() {
        let schedule = CoolingSchedule::Linear {
            initial_temp: 1.0,
            min_temp: 0.01,
        };
        let t0 = schedule.temperature(0, 100);
        let t50 = schedule.temperature(50, 100);
        let t100 = schedule.temperature(100, 100);

        assert!((t0 - 1.0).abs() < 1e-9);
        assert!(t50 < t0);
        assert!(t100 < t50);
        assert!((t100 - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_cooling_schedule_exponential_temperature_decreases() {
        let schedule = CoolingSchedule::Exponential {
            initial_temp: 1.0,
            decay_rate: 0.95,
        };
        let t0 = schedule.temperature(0, 100);
        let t10 = schedule.temperature(10, 100);
        let t50 = schedule.temperature(50, 100);

        assert!((t0 - 1.0).abs() < 1e-9);
        assert!(t10 < t0);
        assert!(t50 < t10);
        // Verify exponential formula: 1.0 * 0.95^10
        let expected_t10 = 0.95_f64.powi(10);
        assert!((t10 - expected_t10).abs() < 1e-9);
    }

    #[test]
    fn test_cooling_schedule_adaptive_temperature() {
        let schedule = CoolingSchedule::Adaptive {
            initial_temp: 1.0,
            improvement_factor: 0.9,
        };
        let t0 = schedule.temperature(0, 100);
        let t10 = schedule.temperature(10, 100);

        assert!((t0 - 1.0).abs() < 1e-9);
        // Adaptive: 1.0 * 0.9^(10 * 0.1) = 1.0 * 0.9^1.0 = 0.9
        assert!((t10 - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_simba_config_default_values() {
        let config = SimbaConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.generations, 50);
        assert!((config.mutation_rate - 0.3).abs() < 1e-9);
        assert_eq!(config.tournament_size, 3);
        assert_eq!(config.elite_count, 2);
        assert_eq!(config.mini_batch_size, 5);
        // Verify default cooling schedule is Linear
        match &config.cooling_schedule {
            CoolingSchedule::Linear {
                initial_temp,
                min_temp,
            } => {
                assert!((initial_temp - 1.0).abs() < 1e-9);
                assert!((min_temp - 0.01).abs() < 1e-9);
            }
            _ => panic!("Expected Linear cooling schedule as default"),
        }
    }

    #[test]
    fn test_tournament_selector_select_keeps_elites() {
        let selector = TournamentSelector::new(3, 2);
        let population = vec![
            ("variant_a".to_string(), 0.9),
            ("variant_b".to_string(), 0.1),
            ("variant_c".to_string(), 0.5),
            ("variant_d".to_string(), 0.3),
            ("variant_e".to_string(), 0.8),
        ];

        let selected = selector.select(&population);

        assert_eq!(selected.len(), population.len());
        // The top 2 by score are "variant_a" (0.9) and "variant_e" (0.8)
        assert_eq!(selected[0], "variant_a");
        assert_eq!(selected[1], "variant_e");
    }

    #[test]
    fn test_tournament_selector_select_returns_correct_size() {
        let selector = TournamentSelector::new(2, 1);
        let population = vec![
            ("a".to_string(), 0.5),
            ("b".to_string(), 0.6),
            ("c".to_string(), 0.7),
        ];

        let selected = selector.select(&population);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_tournament_selector_with_elite_count_zero() {
        let selector = TournamentSelector::new(2, 0);
        let population = vec![
            ("a".to_string(), 0.5),
            ("b".to_string(), 0.6),
            ("c".to_string(), 0.7),
        ];

        let selected = selector.select(&population);
        assert_eq!(selected.len(), 3);
        // With 0 elites, all selections are from tournament
    }

    #[test]
    fn test_simba_optimizer_new() {
        let config = SimbaConfig::default();
        let optimizer = SimbaOptimizer::new(config);
        assert_eq!(optimizer.config().population_size, 20);
        assert_eq!(optimizer.config().generations, 50);
        match &optimizer.mutation_strategy {
            MutationStrategy::RandomPerturbation { strength } => {
                assert!((strength - 0.3).abs() < 1e-9);
            }
            _ => panic!("Expected RandomPerturbation as default strategy"),
        }
    }

    #[test]
    fn test_simba_optimizer_optimize_runs_and_returns_result() {
        let mut config = SimbaConfig::default();
        config.population_size = 5;
        config.generations = 3;
        config.mini_batch_size = 2;

        let optimizer = SimbaOptimizer::new(config);

        let sig = make_qa_signature();
        let examples = make_training_examples();

        let result = optimizer.optimize(&sig, &examples, &ContainsAnswer);

        assert!(result.best_score >= 0.0);
        assert!(!result.scores_history.is_empty());
        assert!(result.trials_run > 0);
    }

    #[test]
    fn test_simba_optimizer_with_exponential_cooling() {
        let mut config = SimbaConfig::default();
        config.population_size = 4;
        config.generations = 2;
        config.mini_batch_size = 2;
        config.cooling_schedule = CoolingSchedule::Exponential {
            initial_temp: 2.0,
            decay_rate: 0.9,
        };

        let optimizer = SimbaOptimizer::new(config);
        let sig = make_qa_signature();
        let examples = make_training_examples();

        let result = optimizer.optimize(&sig, &examples, &ContainsAnswer);
        assert!(result.best_score >= 0.0);
        assert!(!result.scores_history.is_empty());
    }

    #[test]
    fn test_mutation_strategy_variants() {
        // Test that all variants can be constructed and serialized
        let perturbation = MutationStrategy::RandomPerturbation { strength: 0.5 };
        let json = serde_json::to_string(&perturbation).expect("serialize");
        assert!(json.contains("RandomPerturbation"));

        let crossover = MutationStrategy::Crossover {
            crossover_rate: 0.7,
        };
        let json = serde_json::to_string(&crossover).expect("serialize");
        assert!(json.contains("Crossover"));

        let llm = MutationStrategy::LlmGuided {
            prompt_template: "improve: {instruction}".to_string(),
        };
        let json = serde_json::to_string(&llm).expect("serialize");
        assert!(json.contains("LlmGuided"));
    }

    #[test]
    fn test_combined_mutation_strategy() {
        let combined = MutationStrategy::Combined {
            strategies: vec![
                MutationStrategy::RandomPerturbation { strength: 0.3 },
                MutationStrategy::Crossover {
                    crossover_rate: 0.5,
                },
            ],
            weights: vec![0.7, 0.3],
        };
        let json = serde_json::to_string(&combined).expect("serialize");
        assert!(json.contains("Combined"));

        // Test with_strategy
        let mut config = SimbaConfig::default();
        config.population_size = 4;
        config.generations = 2;
        config.mini_batch_size = 2;
        let optimizer = SimbaOptimizer::with_strategy(config, combined);
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let result = optimizer.optimize(&sig, &examples, &ContainsAnswer);
        assert!(result.best_score >= 0.0);
    }

    // ========================================================================
    // 4.2 Reasoning Trace Capture tests
    // ========================================================================

    #[test]
    fn test_reasoning_step_construction() {
        let step = ReasoningStep {
            thought: "The capital of France is Paris".to_string(),
            conclusion: Some("Paris".to_string()),
            evidence: vec!["Known fact".to_string()],
            confidence: 0.95,
            token_count: 7,
        };
        assert_eq!(step.thought, "The capital of France is Paris");
        assert_eq!(step.conclusion, Some("Paris".to_string()));
        assert_eq!(step.evidence.len(), 1);
        assert!((step.confidence - 0.95).abs() < 1e-9);
        assert_eq!(step.token_count, 7);
    }

    #[test]
    fn test_reasoning_trace_new_add_step_step_count_avg_confidence() {
        let mut trace = ReasoningTrace::new("sig_1".to_string(), "hash_abc".to_string());
        assert_eq!(trace.step_count(), 0);
        assert!((trace.avg_confidence() - 0.0).abs() < 1e-9);

        trace.add_step(ReasoningStep {
            thought: "Step A".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.8,
            token_count: 5,
        });
        trace.add_step(ReasoningStep {
            thought: "Step B".to_string(),
            conclusion: Some("Result".to_string()),
            evidence: Vec::new(),
            confidence: 0.6,
            token_count: 3,
        });

        assert_eq!(trace.step_count(), 2);
        assert!((trace.avg_confidence() - 0.7).abs() < 1e-9);
        assert_eq!(trace.total_tokens(), 8);
        assert_eq!(trace.signature_id, "sig_1");
        assert_eq!(trace.input_hash, "hash_abc");
    }

    #[test]
    fn test_reasoning_trace_final_conclusion() {
        let mut trace = ReasoningTrace::new("sig".to_string(), "hash".to_string());

        // Empty trace has no conclusion
        assert!(trace.final_conclusion().is_none());

        trace.add_step(ReasoningStep {
            thought: "Thinking...".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.5,
            token_count: 1,
        });
        assert!(trace.final_conclusion().is_none());

        trace.add_step(ReasoningStep {
            thought: "Concluding...".to_string(),
            conclusion: Some("Final answer".to_string()),
            evidence: Vec::new(),
            confidence: 0.9,
            token_count: 2,
        });
        assert_eq!(trace.final_conclusion(), Some("Final answer"));
    }

    #[test]
    fn test_trace_extractor_with_defaults() {
        let extractor = TraceExtractor::with_defaults();
        assert!(!extractor.thought_markers.is_empty());
        assert!(!extractor.conclusion_markers.is_empty());
        assert!(extractor.thought_markers.contains(&"<thinking>".to_string()));
        assert!(extractor.conclusion_markers.contains(&"Therefore".to_string()));
    }

    #[test]
    fn test_trace_extractor_extract_with_thought_markers() {
        let extractor = TraceExtractor::with_defaults();
        let text = "Let me think about this. The question is about geography. \
                     Step 1: France is in Europe. \
                     Step 2: The capital is Paris. \
                     Therefore the answer is Paris.";
        let trace = extractor.extract(text);
        assert!(trace.step_count() >= 2);
        // The last step should have a conclusion containing "Paris"
        let conclusion = trace.final_conclusion();
        assert!(conclusion.is_some());
    }

    #[test]
    fn test_trace_extractor_extract_with_no_markers_returns_single_step() {
        let extractor = TraceExtractor::new();
        let text = "The answer is 42.";
        let trace = extractor.extract(text);
        assert_eq!(trace.step_count(), 1);
        assert_eq!(trace.steps[0].thought, "The answer is 42.");
    }

    #[test]
    fn test_trace_store_store_and_get() {
        let mut store = TraceStore::new(10);

        let mut trace = ReasoningTrace::new("sig_a".to_string(), "h1".to_string());
        trace.add_step(ReasoningStep {
            thought: "t1".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.7,
            token_count: 2,
        });
        store.store(trace);

        let retrieved = store.get("sig_a");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);

        assert!(store.get("sig_b").is_none());
    }

    #[test]
    fn test_trace_store_get_best_filters_by_confidence() {
        let mut store = TraceStore::new(10);

        // Low confidence trace
        let mut t1 = ReasoningTrace::new("sig".to_string(), "h1".to_string());
        t1.add_step(ReasoningStep {
            thought: "low".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.3,
            token_count: 1,
        });
        store.store(t1);

        // High confidence trace
        let mut t2 = ReasoningTrace::new("sig".to_string(), "h2".to_string());
        t2.add_step(ReasoningStep {
            thought: "high".to_string(),
            conclusion: Some("good".to_string()),
            evidence: Vec::new(),
            confidence: 0.9,
            token_count: 1,
        });
        store.store(t2);

        let best = store.get_best("sig", 0.5);
        assert_eq!(best.len(), 1);
        assert!((best[0].avg_confidence() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_trace_store_trace_count_and_signature_count() {
        let mut store = TraceStore::new(10);

        let t1 = ReasoningTrace::new("sig_a".to_string(), "h1".to_string());
        let t2 = ReasoningTrace::new("sig_a".to_string(), "h2".to_string());
        let t3 = ReasoningTrace::new("sig_b".to_string(), "h3".to_string());

        store.store(t1);
        store.store(t2);
        store.store(t3);

        assert_eq!(store.trace_count(), 3);
        assert_eq!(store.signature_count(), 2);
    }

    #[test]
    fn test_trace_store_clear() {
        let mut store = TraceStore::new(10);
        store.store(ReasoningTrace::new("sig".to_string(), "h".to_string()));
        assert_eq!(store.trace_count(), 1);

        store.clear();
        assert_eq!(store.trace_count(), 0);
        assert_eq!(store.signature_count(), 0);
    }

    #[test]
    fn test_trace_analyzer_avg_steps() {
        let analyzer = TraceAnalyzer::new();

        let mut t1 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t1.add_step(ReasoningStep {
            thought: "a".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.5,
            token_count: 1,
        });
        t1.add_step(ReasoningStep {
            thought: "b".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.5,
            token_count: 1,
        });

        let mut t2 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t2.add_step(ReasoningStep {
            thought: "c".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.5,
            token_count: 1,
        });

        let avg = analyzer.avg_steps(&[t1, t2]);
        assert!((avg - 1.5).abs() < 1e-9); // (2 + 1) / 2 = 1.5
    }

    #[test]
    fn test_trace_analyzer_avg_confidence() {
        let analyzer = TraceAnalyzer::new();

        let mut t1 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t1.add_step(ReasoningStep {
            thought: "a".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.8,
            token_count: 1,
        });

        let mut t2 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t2.add_step(ReasoningStep {
            thought: "b".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.6,
            token_count: 1,
        });

        let avg = analyzer.avg_confidence(&[t1, t2]);
        assert!((avg - 0.7).abs() < 1e-9); // (0.8 + 0.6) / 2
    }

    #[test]
    fn test_trace_analyzer_success_rate() {
        let analyzer = TraceAnalyzer::new();

        let mut t1 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t1.add_step(ReasoningStep {
            thought: "a".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.9,
            token_count: 1,
        });

        let mut t2 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t2.add_step(ReasoningStep {
            thought: "b".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.3,
            token_count: 1,
        });

        let mut t3 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t3.add_step(ReasoningStep {
            thought: "c".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.7,
            token_count: 1,
        });

        let rate = analyzer.success_rate(&[t1, t2, t3], 0.5);
        // t1 (0.9) and t3 (0.7) pass, t2 (0.3) fails => 2/3
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_trace_analyzer_common_patterns() {
        let analyzer = TraceAnalyzer::new();

        let mut t1 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t1.add_step(ReasoningStep {
            thought: "the answer involves reasoning carefully about facts".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.8,
            token_count: 7,
        });

        let mut t2 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t2.add_step(ReasoningStep {
            thought: "we need careful reasoning about the problem".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.7,
            token_count: 7,
        });

        let mut t3 = ReasoningTrace::new("s".to_string(), "h".to_string());
        t3.add_step(ReasoningStep {
            thought: "reasoning about this topic requires analysis".to_string(),
            conclusion: None,
            evidence: Vec::new(),
            confidence: 0.6,
            token_count: 6,
        });

        let patterns = analyzer.common_patterns(&[t1, t2, t3]);
        // "reasoning" and "about" appear in all 3 traces (>50% of 3)
        assert!(patterns.contains(&"reasoning".to_string()));
        assert!(patterns.contains(&"about".to_string()));
    }

    // ========================================================================
    // 4.3 LLM-as-Judge Automated Grading tests
    // ========================================================================

    #[test]
    fn test_judge_criterion_construction() {
        let criterion = JudgeCriterion {
            name: "relevance".to_string(),
            description: "How relevant is the answer".to_string(),
            weight: 0.5,
            scale: (0.0, 1.0),
        };
        assert_eq!(criterion.name, "relevance");
        assert_eq!(criterion.description, "How relevant is the answer");
        assert!((criterion.weight - 0.5).abs() < 1e-9);
        assert!((criterion.scale.0 - 0.0).abs() < 1e-9);
        assert!((criterion.scale.1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_judge_rubric_new_add_criterion_validate() {
        let mut rubric = JudgeRubric::new("Evaluate the answer quality".to_string());
        assert_eq!(rubric.criterion_count(), 0);

        rubric.add_criterion(JudgeCriterion {
            name: "accuracy".to_string(),
            description: "Is the answer correct".to_string(),
            weight: 1.0,
            scale: (0.0, 1.0),
        });

        assert_eq!(rubric.criterion_count(), 1);
        assert!(rubric.validate().is_ok());
    }

    #[test]
    fn test_judge_rubric_validate_fails_on_empty_criteria() {
        let rubric = JudgeRubric::new("Empty rubric".to_string());
        let result = rubric.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one criterion"));
    }

    #[test]
    fn test_judge_rubric_validate_fails_on_non_positive_weight() {
        let mut rubric = JudgeRubric::new("Bad weight".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "bad".to_string(),
            description: "zero weight".to_string(),
            weight: 0.0,
            scale: (0.0, 1.0),
        });
        let result = rubric.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-positive weight"));
    }

    #[test]
    fn test_judge_rubric_total_weight() {
        let mut rubric = JudgeRubric::new("test".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "a".to_string(),
            description: "first".to_string(),
            weight: 0.6,
            scale: (0.0, 1.0),
        });
        rubric.add_criterion(JudgeCriterion {
            name: "b".to_string(),
            description: "second".to_string(),
            weight: 0.4,
            scale: (0.0, 1.0),
        });
        assert!((rubric.total_weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_judge_rubric_with_multiple_criteria() {
        let mut rubric = JudgeRubric::new("Multi-criterion".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "relevance".to_string(),
            description: "How relevant".to_string(),
            weight: 0.4,
            scale: (0.0, 1.0),
        });
        rubric.add_criterion(JudgeCriterion {
            name: "fluency".to_string(),
            description: "How fluent".to_string(),
            weight: 0.3,
            scale: (0.0, 5.0),
        });
        rubric.add_criterion(JudgeCriterion {
            name: "accuracy".to_string(),
            description: "How accurate".to_string(),
            weight: 0.3,
            scale: (0.0, 10.0),
        });

        assert_eq!(rubric.criterion_count(), 3);
        assert!((rubric.total_weight() - 1.0).abs() < 1e-9);
        assert!(rubric.validate().is_ok());
    }

    #[test]
    fn test_prompt_judge_result_construction() {
        let result = PromptJudgeResult {
            overall_score: 0.85,
            per_criterion: vec![CriterionScore {
                criterion_name: "accuracy".to_string(),
                score: 0.9,
                reasoning: "Very accurate".to_string(),
            }],
            reasoning: "Overall good quality".to_string(),
            confidence: 0.9,
        };
        assert!((result.overall_score - 0.85).abs() < 1e-9);
        assert_eq!(result.per_criterion.len(), 1);
        assert!((result.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_criterion_score_construction() {
        let score = CriterionScore {
            criterion_name: "fluency".to_string(),
            score: 4.5,
            reasoning: "Very fluent output".to_string(),
        };
        assert_eq!(score.criterion_name, "fluency");
        assert!((score.score - 4.5).abs() < 1e-9);
        assert_eq!(score.reasoning, "Very fluent output");
    }

    #[test]
    fn test_judge_config_construction() {
        let mut rubric = JudgeRubric::new("test".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "q".to_string(),
            description: "quality".to_string(),
            weight: 1.0,
            scale: (0.0, 1.0),
        });
        let config = JudgeConfig {
            rubric,
            few_shot_examples: Vec::new(),
            temperature: 0.0,
        };
        assert_eq!(config.few_shot_examples.len(), 0);
        assert!((config.temperature - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_judge_example_construction() {
        let ex = JudgeExample {
            input: "What is 2+2?".to_string(),
            output: "4".to_string(),
            expected_score: 1.0,
            reasoning: "Correct arithmetic".to_string(),
        };
        assert_eq!(ex.input, "What is 2+2?");
        assert_eq!(ex.output, "4");
        assert!((ex.expected_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_judge_metric_evaluate_returns_result() {
        let mut rubric = JudgeRubric::new("Evaluate answer".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "correctness".to_string(),
            description: "Is it correct".to_string(),
            weight: 1.0,
            scale: (0.0, 1.0),
        });

        let config = JudgeConfig {
            rubric,
            few_shot_examples: Vec::new(),
            temperature: 0.0,
        };

        let judge = JudgeMetric::new(
            config,
            Box::new(|_input: &str, output: &str, _rubric: &JudgeRubric| -> f64 {
                if output.contains("Paris") {
                    0.9
                } else {
                    0.2
                }
            }),
        );

        let result = judge.evaluate("What is the capital of France?", "Paris");
        assert!((result.overall_score - 0.9).abs() < 1e-9);
        assert_eq!(result.per_criterion.len(), 1);
        assert!((result.confidence - 0.8).abs() < 1e-9);

        let result2 = judge.evaluate("What is the capital of France?", "London");
        assert!((result2.overall_score - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_judge_metric_eval_metric_impl_score() {
        let mut rubric = JudgeRubric::new("test".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "match".to_string(),
            description: "does it match".to_string(),
            weight: 1.0,
            scale: (0.0, 1.0),
        });

        let config = JudgeConfig {
            rubric,
            few_shot_examples: Vec::new(),
            temperature: 0.0,
        };

        let judge = JudgeMetric::new(
            config,
            Box::new(|predicted: &str, expected: &str, _rubric: &JudgeRubric| -> f64 {
                if predicted == expected {
                    1.0
                } else {
                    0.0
                }
            }),
        );

        // Test EvalMetric trait impl
        assert_eq!(judge.name(), "judge_metric");
        assert!((judge.score("hello", "hello") - 1.0).abs() < 1e-9);
        assert!((judge.score("hello", "world") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_calibration_set_new_add_len() {
        let mut cal = CalibrationSet::new();
        assert!(cal.is_empty());
        assert_eq!(cal.len(), 0);

        cal.add(CalibrationExample {
            input: "q1".to_string(),
            output: "a1".to_string(),
            human_score: 0.8,
        });
        cal.add(CalibrationExample {
            input: "q2".to_string(),
            output: "a2".to_string(),
            human_score: 0.6,
        });

        assert!(!cal.is_empty());
        assert_eq!(cal.len(), 2);
        assert_eq!(cal.examples().len(), 2);
    }

    #[test]
    fn test_calibration_example_construction() {
        let ex = CalibrationExample {
            input: "test input".to_string(),
            output: "test output".to_string(),
            human_score: 0.75,
        };
        assert_eq!(ex.input, "test input");
        assert_eq!(ex.output, "test output");
        assert!((ex.human_score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_calibration_set_calibration_error_computation() {
        let mut rubric = JudgeRubric::new("test".to_string());
        rubric.add_criterion(JudgeCriterion {
            name: "q".to_string(),
            description: "quality".to_string(),
            weight: 1.0,
            scale: (0.0, 1.0),
        });

        let config = JudgeConfig {
            rubric,
            few_shot_examples: Vec::new(),
            temperature: 0.0,
        };

        // Judge always returns 0.7
        let judge = JudgeMetric::new(
            config,
            Box::new(|_: &str, _: &str, _: &JudgeRubric| -> f64 { 0.7 }),
        );

        let mut cal = CalibrationSet::new();
        cal.add(CalibrationExample {
            input: "q1".to_string(),
            output: "a1".to_string(),
            human_score: 0.8, // error = |0.7 - 0.8| = 0.1
        });
        cal.add(CalibrationExample {
            input: "q2".to_string(),
            output: "a2".to_string(),
            human_score: 0.5, // error = |0.7 - 0.5| = 0.2
        });

        let error = cal.calibration_error(&judge);
        // Mean absolute error = (0.1 + 0.2) / 2 = 0.15
        assert!((error - 0.15).abs() < 1e-9);
    }
