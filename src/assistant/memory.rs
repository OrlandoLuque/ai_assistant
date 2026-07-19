use super::*;

impl AiAssistant {
    // === Memory Integration ===

    /// Enable the memory system with the given configuration.
    pub fn enable_memory(&mut self, config: MemoryConfig) {
        self.memory_manager = Some(MemoryManager::new(config));
    }

    /// Disable the memory system and discard stored memories.
    pub fn disable_memory(&mut self) {
        self.memory_manager = None;
    }

    /// Attach an LLM fact extractor to the memory manager (enabling memory first
    /// if needed). The extractor may be backed by a **different, more capable**
    /// model/endpoint than the chat model — e.g.
    /// `SelfChatEnhancer::new(stronger_config)` pointing at a stronger model on
    /// another machine — so a cheap local chat model is "rescued" by an
    /// occasional capable extraction pass. The deterministic heuristic stays
    /// authoritative, so a slow/absent/wrong extractor can only add facts, never
    /// corrupt a known one.
    pub fn set_fact_extractor(&mut self, llm: Box<dyn crate::llm_enhance::LlmEnhancer>) {
        if self.memory_manager.is_none() {
            self.enable_memory(MemoryConfig::default());
        }
        if let Some(mm) = self.memory_manager.as_mut() {
            mm.set_fact_extractor(llm);
        }
    }

    /// Whether the memory system is enabled.
    pub fn has_memory(&self) -> bool {
        self.memory_manager.is_some()
    }

    /// Access the memory manager (read-only).
    pub fn memory_manager(&self) -> Option<&MemoryManager> {
        self.memory_manager.as_ref()
    }

    /// Access the memory manager (mutable).
    pub fn memory_manager_mut(&mut self) -> Option<&mut MemoryManager> {
        self.memory_manager.as_mut()
    }

    /// Build context using the Adaptive Context Budget Allocator.
    ///
    /// Collects items from all available sources (RAG, Memory, Procedural,
    /// References), assigns scores, and packs into the available token budget
    /// using score-based greedy allocation.
    ///
    /// Falls back to simple concatenation if allocator would add no benefit
    /// (e.g., total context is small enough to fit without allocation).
    pub fn build_allocated_context(
        &mut self,
        user_message: &str,
        knowledge_context: &str,
        intent: Option<&crate::intent::IntentResult>,
    ) -> String {
        use crate::context_budget::{ContextBudgetAllocator, ContextItem, ContextSourceType};

        // Clone config to avoid borrow conflicts with &mut self methods below
        let cfg = self.context_budget_config.clone();
        let mut items: Vec<ContextItem> = Vec::new();

        // Use model-aware token counting for all budget calculations
        let model = self.config.selected_model.clone();
        let count =
            |text: &str| -> usize { crate::context::estimate_tokens_for_model(text, &model) };

        // 1. RAG/knowledge context (passed in from caller or build_rag_context)
        if !knowledge_context.is_empty() {
            let tokens = count(knowledge_context);
            let score = cfg.effective_score(cfg.rag_base_score, ContextSourceType::Rag, intent);
            items.push(
                ContextItem::new(knowledge_context, tokens, score, ContextSourceType::Rag)
                    .with_label("knowledge_context"),
            );
        }

        // 2. Memory context
        {
            let query = self
                .conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let memory_ctx = self.build_memory_context(&query, cfg.memory_max_tokens);
            if !memory_ctx.is_empty() {
                let tokens = count(&memory_ctx);
                let score =
                    cfg.effective_score(cfg.memory_base_score, ContextSourceType::Memory, intent);
                items.push(
                    ContextItem::new(memory_ctx, tokens, score, ContextSourceType::Memory)
                        .with_label("memory"),
                );
            }
        }

        // 3. Procedural context
        #[cfg(feature = "advanced-memory")]
        {
            let proc_ctx = self.build_procedural_context(
                user_message,
                cfg.procedural_max_items,
                cfg.procedural_max_tokens,
            );
            if !proc_ctx.is_empty() {
                let tokens = count(&proc_ctx);
                let score = cfg.effective_score(
                    cfg.procedural_base_score,
                    ContextSourceType::Procedural,
                    intent,
                );
                items.push(
                    ContextItem::new(proc_ctx, tokens, score, ContextSourceType::Procedural)
                        .with_label("procedural"),
                );
            }
        }

        // 4. Resolved references
        let resolved_refs = self.reference_resolver.resolve_reference(user_message);
        if let Some(ref refs) = resolved_refs {
            let tokens = count(refs);
            let score = cfg.effective_score(
                cfg.reference_base_score,
                ContextSourceType::Reference,
                intent,
            );
            items.push(
                ContextItem::new(refs.clone(), tokens, score, ContextSourceType::Reference)
                    .with_label("references"),
            );
        }

        // 5. Knowledge graph context (separate from RAG to allow independent scoring)
        #[cfg(feature = "multi-agent")]
        {
            let graph_ctx = self.build_graph_context_string(user_message);
            if !graph_ctx.is_empty() {
                let tokens = count(&graph_ctx);
                let score =
                    cfg.effective_score(cfg.graph_base_score, ContextSourceType::Graph, intent);
                items.push(
                    ContextItem::new(graph_ctx, tokens, score, ContextSourceType::Graph)
                        .with_label("knowledge_graph"),
                );
            }
        }

        // If nothing to allocate, return empty
        if items.is_empty() {
            return String::new();
        }

        // Calculate available budget with model-aware counting + dynamic reserve
        let model_ctx = self.detected_context_size.unwrap_or_else(|| {
            crate::context::get_model_context_size_cached(&self.config.selected_model, |_| None)
        });
        let system_tokens = count(&self.system_prompt_base);
        let conversation_tokens: usize = self.conversation.iter().map(|m| count(&m.content)).sum();
        let user_tokens = count(user_message);
        let precision = self.token_precision();
        let response_reserve = (model_ctx as f64 * precision.reserve_factor())
            .max(cfg.min_response_reserve as f64) as usize;

        let budget = ContextBudgetAllocator::available_budget(
            model_ctx,
            system_tokens,
            conversation_tokens,
            user_tokens,
            response_reserve,
        );

        // Select overflow strategy: bandit (if learning) or configured
        let (strategy, bandit_arm) = if let Some(ref bandit) = self.strategy_bandit {
            let arm = bandit.select().to_string();
            let compressor_model = match &cfg.overflow_strategy {
                crate::context_budget::OverflowStrategy::LlmCompression {
                    compressor_model,
                    ..
                } => Some(compressor_model.as_str()),
                crate::context_budget::OverflowStrategy::Hybrid {
                    compressor_model, ..
                } => Some(compressor_model.as_str()),
                _ => None,
            };
            let strat =
                crate::context_budget::StrategyBandit::arm_to_strategy(&arm, compressor_model);
            (strat, Some(arm))
        } else {
            (cfg.overflow_strategy.clone(), None)
        };

        let allocator = ContextBudgetAllocator::new(strategy);
        let result = allocator.build_from_items(items, budget);

        // Update bandit reward based on utilization
        if let Some(arm) = bandit_arm {
            if let Some(ref mut bandit) = self.strategy_bandit {
                bandit.update(&arm, result.utilization() as f64);
            }
        }

        crate::diag_debug!(
            "[context-budget] allocated: {} tokens used / {} budget ({:.0}% utilization), {} included, {} dropped, scoring={:?}",
            result.tokens_used, result.budget, result.utilization() * 100.0,
            result.included.len(), result.dropped.len(), cfg.scoring_mode
        );

        result.context
    }

    /// Classify intent for context budget scoring (returns None if Static mode).
    pub(crate) fn classify_intent_for_budget(
        &self,
        user_message: &str,
    ) -> Option<crate::intent::IntentResult> {
        use crate::context_budget::ScoringMode;
        match &self.context_budget_config.scoring_mode {
            ScoringMode::Static => None,
            _ => Some(crate::intent::IntentClassifier::new().classify(user_message)),
        }
    }

    /// Build memory-based context for a query (empty string if memory disabled).
    pub fn build_memory_context(&mut self, query: &str, max_tokens: usize) -> String {
        crate::diag_debug!(
            "[memory-context] build_memory_context: max_tokens={}, memory_enabled={}",
            max_tokens,
            self.memory_manager.is_some()
        );
        match self.memory_manager.as_mut() {
            Some(mm) => {
                let result = mm.build_context(query, max_tokens);
                crate::diag_debug!("[memory-context] result: {} chars", result.len());
                crate::diag_trace!("[memory-context] content={:.500}", result);
                result
            }
            None => String::new(),
        }
    }

    /// Build graph context string from the knowledge graph for a query.
    ///
    /// Extracts matching entities and their relations from the multi-layer
    /// knowledge graph. Returns an empty string if no graph is available or
    /// no entities match.
    #[cfg(feature = "multi-agent")]
    pub fn build_graph_context_string(&self, query: &str) -> String {
        let graph = match self.graph.as_ref() {
            Some(g) => g,
            None => return String::new(),
        };

        let session_id = self.current_session.as_ref().map(|s| s.id.as_str());

        // Extract entity names from the query for graph lookup
        let query_words: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();

        // Query unified view for matching entities
        let unified = graph.query_unified(session_id);
        let matching: Vec<&crate::multi_layer_graph::UnifiedEntity> = unified
            .entities
            .iter()
            .filter(|e| {
                let name_lower = e.name.to_lowercase();
                query_words.iter().any(|w| name_lower.contains(w))
            })
            .collect();

        if matching.is_empty() {
            return String::new();
        }

        let mut result = String::from("--- GRAPH CONTEXT ---\n");
        for entity in matching.iter().take(10) {
            result.push_str(&format!(
                "- {} ({}): confidence {:.0}%",
                entity.name,
                entity.entity_type,
                entity.confidence * 100.0
            ));
            for (key, val) in &entity.merged_attributes {
                result.push_str(&format!(", {}={}", key, val));
            }
            result.push('\n');
        }

        // Add relevant relations
        let matching_names: Vec<&str> = matching.iter().map(|e| e.name.as_str()).collect();
        for rel in &unified.relations {
            if matching_names.contains(&rel.source.as_str())
                || matching_names.contains(&rel.target.as_str())
            {
                result.push_str(&format!(
                    "- {} --[{}]--> {}\n",
                    rel.source, rel.relation_type, rel.target
                ));
            }
        }
        result.push_str("--- END GRAPH ---\n");

        crate::diag_debug!(
            "[graph-context] enrichment: {} entities, {} relations",
            matching.len(),
            unified.relations.len()
        );

        result
    }

    // === Procedural Memory Integration ===

    /// Enable procedural memory with the given capacity.
    ///
    /// Once enabled, matching procedures are automatically injected into the
    /// system prompt as `--- WORKFLOW GUIDELINES ---` when their condition
    /// keywords match the user's message.
    #[cfg(feature = "advanced-memory")]
    pub fn enable_procedural_memory(&mut self, max_procedures: usize) {
        self.procedural_store = Some(crate::advanced_memory::ProceduralStore::new(max_procedures));
        self.procedure_evolver = Some(crate::advanced_memory::ProcedureEvolver::new(
            crate::advanced_memory::EvolutionConfig::default(),
        ));
    }

    /// Disable procedural memory and discard all procedures.
    #[cfg(feature = "advanced-memory")]
    pub fn disable_procedural_memory(&mut self) {
        self.procedural_store = None;
        self.procedure_evolver = None;
        self.active_procedure_ids.clear();
    }

    /// Whether procedural memory is enabled.
    #[cfg(feature = "advanced-memory")]
    pub fn has_procedural_memory(&self) -> bool {
        self.procedural_store.is_some()
    }

    /// Add a procedure to the procedural store.
    #[cfg(feature = "advanced-memory")]
    pub fn add_procedure(&mut self, procedure: crate::advanced_memory::Procedure) {
        if let Some(ref mut store) = self.procedural_store {
            store.add(procedure);
        }
    }

    /// List all procedures (read-only slice).
    #[cfg(feature = "advanced-memory")]
    pub fn list_procedures(&self) -> &[crate::advanced_memory::Procedure] {
        match &self.procedural_store {
            Some(store) => store.all(),
            None => &[],
        }
    }

    /// Remove a procedure by ID. Returns the removed procedure if found.
    #[cfg(feature = "advanced-memory")]
    pub fn remove_procedure(&mut self, id: &str) -> Option<crate::advanced_memory::Procedure> {
        if let Some(ref mut store) = self.procedural_store {
            store.remove(id)
        } else {
            None
        }
    }

    /// Find procedures matching a query string.
    #[cfg(feature = "advanced-memory")]
    pub fn find_procedures(&self, query: &str) -> Vec<&crate::advanced_memory::Procedure> {
        match &self.procedural_store {
            Some(store) => store.find_relevant(query, 0.3, 0.1, 5),
            None => Vec::new(),
        }
    }

    /// Record explicit outcome feedback for a procedure.
    #[cfg(feature = "advanced-memory")]
    pub fn record_procedure_outcome(
        &mut self,
        procedure_id: &str,
        success: bool,
    ) -> Result<(), crate::error::AiError> {
        if let Some(ref mut store) = self.procedural_store {
            store.update_outcome(procedure_id, success)
        } else {
            Ok(())
        }
    }

    /// Save the procedural store to a file. Returns error if not enabled.
    #[cfg(feature = "advanced-memory")]
    pub fn save_procedures(&self, path: &std::path::Path) -> Result<(), String> {
        match &self.procedural_store {
            Some(store) => store.save_to_file(path).map(|_| ()),
            None => Err("Procedural memory not enabled".to_string()),
        }
    }

    /// Load procedures from a file. Enables procedural memory if not already enabled.
    #[cfg(feature = "advanced-memory")]
    pub fn load_procedures(
        &mut self,
        path: &std::path::Path,
        max_procedures: usize,
    ) -> Result<(), String> {
        let store = crate::advanced_memory::ProceduralStore::load_from_file(path, max_procedures)?;
        self.procedural_store = Some(store);
        if self.procedure_evolver.is_none() {
            self.procedure_evolver = Some(crate::advanced_memory::ProcedureEvolver::new(
                crate::advanced_memory::EvolutionConfig::default(),
            ));
        }
        Ok(())
    }

    /// Access the procedural store (read-only).
    #[cfg(feature = "advanced-memory")]
    pub fn procedural_store(&self) -> Option<&crate::advanced_memory::ProceduralStore> {
        self.procedural_store.as_ref()
    }

    /// Load default builtin procedures (skip any whose ID already exists).
    #[cfg(feature = "advanced-memory")]
    pub fn load_default_procedures(&mut self) {
        if self.procedural_store.is_none() {
            self.enable_procedural_memory(50);
        }
        if let Some(ref mut store) = self.procedural_store {
            store.load_defaults();
        }
    }

    /// Create a versioned export of all procedures.
    #[cfg(feature = "advanced-memory")]
    pub fn export_procedures(&self) -> crate::advanced_memory::ProcedureExport {
        match &self.procedural_store {
            Some(store) => store.export("user"),
            None => crate::advanced_memory::ProcedureExport::default(),
        }
    }

    /// Export procedures to a JSON file.
    #[cfg(feature = "advanced-memory")]
    pub fn export_procedures_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        match &self.procedural_store {
            Some(store) => store.export_to_file(path, "user"),
            None => Err("Procedural memory not enabled".to_string()),
        }
    }

    /// Import procedures from a versioned export.
    #[cfg(feature = "advanced-memory")]
    pub fn import_procedures(
        &mut self,
        export: &crate::advanced_memory::ProcedureExport,
        options: &crate::advanced_memory::ProcedureImportOptions,
    ) -> crate::advanced_memory::ProcedureImportResult {
        if self.procedural_store.is_none() {
            self.enable_procedural_memory(50);
        }
        match self.procedural_store.as_mut() {
            Some(store) => store.import(export, options),
            None => crate::advanced_memory::ProcedureImportResult::default(),
        }
    }

    /// Import procedures from a JSON file.
    #[cfg(feature = "advanced-memory")]
    pub fn import_procedures_from_file(
        &mut self,
        path: &std::path::Path,
        options: &crate::advanced_memory::ProcedureImportOptions,
    ) -> Result<crate::advanced_memory::ProcedureImportResult, String> {
        if self.procedural_store.is_none() {
            self.enable_procedural_memory(50);
        }
        match self.procedural_store.as_mut() {
            Some(store) => store.import_from_file(path, options),
            None => Err("Procedural memory not enabled".to_string()),
        }
    }

    /// Build a `--- WORKFLOW GUIDELINES ---` section from procedures matching the
    /// user message. Returns an empty string if no procedures match or if
    /// procedural memory is disabled.
    #[cfg(feature = "advanced-memory")]
    pub(crate) fn build_procedural_context(
        &mut self,
        user_message: &str,
        max_procedures: usize,
        max_tokens: usize,
    ) -> String {
        let store = match &self.procedural_store {
            Some(s) => s,
            None => return String::new(),
        };

        let matches = store.find_relevant(user_message, 0.3, 0.1, max_procedures);
        if matches.is_empty() {
            return String::new();
        }

        let mut result = String::from(
            "--- WORKFLOW GUIDELINES ---\n\
             The following workflow procedures are relevant. Follow these steps where applicable.\n",
        );

        let mut token_count = crate::context::estimate_tokens(&result);
        let mut used_ids = Vec::new();

        for proc in &matches {
            // Format this procedure
            let mut section = format!(
                "\n## {} (confidence: {:.0}%)\n",
                proc.name,
                proc.confidence * 100.0
            );
            for (i, step) in proc.steps.iter().enumerate() {
                section.push_str(&format!("{}. {}\n", i + 1, step));
            }

            let section_tokens = crate::context::estimate_tokens(&section);
            if token_count + section_tokens > max_tokens {
                break;
            }

            result.push_str(&section);
            token_count += section_tokens;
            used_ids.push(proc.id.clone());
        }

        if used_ids.is_empty() {
            return String::new();
        }

        result.push_str("--- END WORKFLOW GUIDELINES ---");
        self.active_procedure_ids = used_ids;
        result
    }
}
