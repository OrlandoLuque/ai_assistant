//! MCP tool registration for runtime routing management.

use super::*;

// =============================================================================
// MCP TOOLS FOR ROUTING MANAGEMENT (Section I)
// =============================================================================

use std::sync::{Arc, Mutex};

/// Register MCP tools for runtime routing management on the given server.
///
/// Tools registered:
/// - `routing.get_stats` — Bandit statistics (arms, pulls, rewards per task type)
/// - `routing.add_arm` — Add a model arm (global or per-task)
/// - `routing.remove_arm` — Remove an arm from the bandit
/// - `routing.warm_start` — Set priors for an arm (influence bandit towards/away from a model)
/// - `routing.record_outcome` — Manually record a feedback outcome
/// - `routing.add_rule` — Add a rule to the NFA and recompile DFA
/// - `routing.force_resynthesize` — Force NFA→DFA re-synthesis from bandit data
/// - `routing.export` — Export full pipeline state as JSON
/// - `routing.import` — Import pipeline state from JSON
/// - `routing.get_config` — Get current pipeline configuration
pub fn register_routing_tools(
    server: &mut crate::mcp_protocol::McpServer,
    pipeline: Arc<Mutex<RoutingPipeline>>,
) {
    use crate::mcp_protocol::McpTool;

    // --- routing.get_stats ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new(
            "routing.get_stats",
            "Get bandit routing statistics: arms, pulls, rewards per task type",
        )
        .with_property(
            "task_type",
            "string",
            "Optional task type to filter (omit for global)",
            false,
        ),
        move |args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let bandit = pipeline.bandit();

            let task_type = args.get("task_type").and_then(|v| v.as_str());

            let arms: Vec<serde_json::Value> = bandit
                .all_arms_vec(task_type)
                .iter()
                .map(|arm| {
                    let mean = if arm.pull_count > 0 {
                        arm.total_reward / arm.pull_count as f64
                    } else {
                        0.0
                    };
                    serde_json::json!({
                        "id": arm.id,
                        "pull_count": arm.pull_count,
                        "total_reward": arm.total_reward,
                        "mean_reward": mean,
                        "alpha": arm.params.alpha,
                        "beta": arm.params.beta,
                    })
                })
                .collect();

            let task_types = bandit.task_types();

            Ok(serde_json::json!({
                "total_pulls": bandit.total_pulls(),
                "task_types": task_types,
                "arms": arms,
                "has_dfa": pipeline.active_dfa().is_some(),
                "synthesis_count": pipeline.synthesis_count(),
            }))
        },
    );

    // --- routing.add_arm ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.add_arm", "Add a model arm to the bandit router")
            .with_property("arm_id", "string", "Model identifier to add", true)
            .with_property(
                "task_type",
                "string",
                "Task type (omit for global arm)",
                false,
            )
            .with_property("alpha", "number", "Initial alpha prior (optional)", false)
            .with_property("beta", "number", "Initial beta prior (optional)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args
                .get("arm_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;

            let task_type = args.get("task_type").and_then(|v| v.as_str());
            let alpha = args.get("alpha").and_then(|v| v.as_f64());
            let beta = args.get("beta").and_then(|v| v.as_f64());

            match task_type {
                Some(tt) => pipeline.add_arm_for_task(tt, arm_id),
                None => pipeline.add_arm(arm_id),
            }

            // Apply warm-start if priors provided
            if let (Some(a), Some(b)) = (alpha, beta) {
                match task_type {
                    Some(tt) => pipeline.bandit_mut().warm_start_for_task(tt, arm_id, a, b),
                    None => pipeline.bandit_mut().warm_start(arm_id, a, b),
                }
            }

            Ok(serde_json::json!({
                "status": "ok",
                "arm_id": arm_id,
                "task_type": task_type,
            }))
        },
    );

    // --- routing.remove_arm ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new(
            "routing.remove_arm",
            "Remove a model arm from the bandit router",
        )
        .with_property("arm_id", "string", "Model identifier to remove", true)
        .with_property(
            "task_type",
            "string",
            "Task type (omit to remove from global)",
            false,
        ),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args
                .get("arm_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;

            let task_type = args.get("task_type").and_then(|v| v.as_str());

            let removed = pipeline.bandit_mut().remove_arm(arm_id, task_type);

            Ok(serde_json::json!({
                "status": if removed { "removed" } else { "not_found" },
                "arm_id": arm_id,
            }))
        },
    );

    // --- routing.warm_start ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new(
            "routing.warm_start",
            "Set priors for a bandit arm to influence routing",
        )
        .with_property("arm_id", "string", "Model identifier", true)
        .with_property(
            "alpha",
            "number",
            "Alpha (success) prior — higher means more preferred",
            true,
        )
        .with_property(
            "beta",
            "number",
            "Beta (failure) prior — higher means less preferred",
            true,
        )
        .with_property("task_type", "string", "Task type (omit for global)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args
                .get("arm_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let alpha = args
                .get("alpha")
                .and_then(|v| v.as_f64())
                .ok_or("Missing required parameter: alpha")?;
            let beta = args
                .get("beta")
                .and_then(|v| v.as_f64())
                .ok_or("Missing required parameter: beta")?;

            match args.get("task_type").and_then(|v| v.as_str()) {
                Some(tt) => pipeline
                    .bandit_mut()
                    .warm_start_for_task(tt, arm_id, alpha, beta),
                None => pipeline.bandit_mut().warm_start(arm_id, alpha, beta),
            }

            Ok(serde_json::json!({
                "status": "ok",
                "arm_id": arm_id,
                "alpha": alpha,
                "beta": beta,
            }))
        },
    );

    // --- routing.record_outcome ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new(
            "routing.record_outcome",
            "Record a feedback outcome for a model arm",
        )
        .with_property("arm_id", "string", "Model that was used", true)
        .with_property(
            "success",
            "boolean",
            "Whether the response was successful",
            true,
        )
        .with_property(
            "quality",
            "number",
            "Quality score 0.0-1.0 (optional, more precise than success)",
            false,
        )
        .with_property("task_type", "string", "Task type (optional)", false)
        .with_property(
            "latency_ms",
            "number",
            "Response latency in ms (optional)",
            false,
        )
        .with_property("cost", "number", "Cost of the call (optional)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args
                .get("arm_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let success = args
                .get("success")
                .and_then(|v| v.as_bool())
                .ok_or("Missing required parameter: success")?;

            let feedback = ArmFeedback {
                arm_id: arm_id.to_string(),
                success,
                quality: args.get("quality").and_then(|v| v.as_f64()),
                latency_ms: args.get("latency_ms").and_then(|v| v.as_u64()),
                cost: args.get("cost").and_then(|v| v.as_f64()),
                task_type: args
                    .get("task_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            };

            pipeline.record_outcome(&feedback);
            let resynthesized = pipeline.maybe_resynthesize();

            Ok(serde_json::json!({
                "status": "ok",
                "resynthesized": resynthesized,
            }))
        },
    );

    // --- routing.add_rule ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.add_rule", "Add a routing rule and recompile DFA")
            .with_property(
                "domain",
                "string",
                "Domain to match (e.g. 'code', 'math')",
                false,
            )
            .with_property(
                "min_complexity",
                "number",
                "Minimum complexity % 0-100 (optional)",
                false,
            )
            .with_property(
                "max_complexity",
                "number",
                "Maximum complexity % 0-100 (optional)",
                false,
            )
            .with_property(
                "has_code",
                "boolean",
                "Match queries with code (optional)",
                false,
            )
            .with_property("arm_id", "string", "Model to route to", true)
            .with_property("priority", "number", "Rule priority (higher wins)", true),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args
                .get("arm_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let priority = args
                .get("priority")
                .and_then(|v| v.as_u64())
                .ok_or("Missing required parameter: priority")? as u32;

            // Build conditions from provided parameters
            let mut conditions: Vec<NfaSymbol> = Vec::new();

            if let Some(domain) = args.get("domain").and_then(|v| v.as_str()) {
                conditions.push(NfaSymbol::Domain(domain.to_string()));
            }

            let min_c = args
                .get("min_complexity")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let max_c = args
                .get("max_complexity")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            if min_c.is_some() || max_c.is_some() {
                conditions.push(NfaSymbol::ComplexityRange {
                    low_pct: min_c.unwrap_or(0),
                    high_pct: max_c.unwrap_or(100),
                });
            }

            if let Some(has_code) = args.get("has_code").and_then(|v| v.as_bool()) {
                conditions.push(NfaSymbol::BoolFeature {
                    name: "has_code".to_string(),
                    value: has_code,
                });
            }

            // Get or create source NFA, add the new rule, recompile
            let mut nfa = pipeline
                .source_nfa()
                .cloned()
                .unwrap_or_else(NfaRouter::new);

            // If NFA is empty, create a start state
            if nfa.state_count() == 0 {
                nfa.add_state("start", None, 0);
            }

            let start = 0; // start state is always 0
            if conditions.is_empty() {
                // No conditions: direct accepting from start via Any
                let accept = nfa.add_state(&format!("rule_{}", arm_id), Some(arm_id), priority);
                nfa.add_transition(start, NfaSymbol::Any, accept);
            } else {
                let mut prev = start;
                for (i, cond) in conditions.iter().enumerate() {
                    let is_last = i == conditions.len() - 1;
                    if is_last {
                        let accept =
                            nfa.add_state(&format!("rule_{}", arm_id), Some(arm_id), priority);
                        nfa.add_transition(prev, cond.clone(), accept);
                    } else {
                        let inter = nfa.add_state(&format!("rule_{}_{}", arm_id, i), None, 0);
                        nfa.add_transition(prev, cond.clone(), inter);
                        prev = inter;
                    }
                }
            }

            // Recompile DFA
            match NfaDfaCompiler::compile(&nfa) {
                Ok(mut dfa) => {
                    dfa.minimize();
                    pipeline.set_nfa_and_dfa(nfa, dfa);
                    Ok(serde_json::json!({
                        "status": "ok",
                        "arm_id": arm_id,
                        "priority": priority,
                        "conditions_count": conditions.len(),
                    }))
                }
                Err(e) => Err(format!("DFA compilation failed: {}", e)),
            }
        },
    );

    // --- routing.force_resynthesize ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new(
            "routing.force_resynthesize",
            "Force re-synthesis of NFA/DFA from bandit data",
        ),
        move |_args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            pipeline.force_resynthesize().map_err(|e| e.to_string())?;

            Ok(serde_json::json!({
                "status": "ok",
                "synthesis_count": pipeline.synthesis_count(),
                "has_dfa": pipeline.active_dfa().is_some(),
            }))
        },
    );

    // --- routing.export ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.export", "Export full pipeline state as JSON"),
        move |_args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let json = pipeline.to_json().map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "status": "ok",
                "pipeline_json": json,
            }))
        },
    );

    // --- routing.import ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.import", "Import pipeline state from JSON").with_property(
            "pipeline_json",
            "string",
            "Pipeline JSON (from routing.export)",
            true,
        ),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let json = args
                .get("pipeline_json")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: pipeline_json")?;

            let restored = RoutingPipeline::from_json(json).map_err(|e| e.to_string())?;
            *pipeline = restored;

            Ok(serde_json::json!({
                "status": "ok",
                "has_dfa": pipeline.active_dfa().is_some(),
                "synthesis_count": pipeline.synthesis_count(),
            }))
        },
    );

    // --- routing.get_config ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.get_config", "Get current pipeline configuration"),
        move |_args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let snapshot = pipeline.export_snapshot();

            Ok(serde_json::json!({
                "synthesis_interval": snapshot.config.synthesis_interval,
                "min_pulls_for_synthesis": snapshot.config.min_pulls_for_synthesis,
                "quality_threshold": snapshot.config.quality_threshold,
                "auto_minimize": snapshot.config.auto_minimize,
                "synthesis_count": snapshot.synthesis_count,
                "has_dfa": pipeline.active_dfa().is_some(),
                "has_nfa": pipeline.source_nfa().is_some(),
                "bandit_arms_count": pipeline.bandit().all_arms(None).len(),
                "bandit_task_types": pipeline.bandit().task_types(),
            }))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // Section I: remove_arm + MCP routing tools tests
    // ================================================================

    #[test]
    fn test_remove_arm_global() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("m1");
        bandit.add_arm("m2");
        bandit.add_arm("m3");
        assert_eq!(bandit.all_arms(None).len(), 3);

        assert!(bandit.remove_arm("m2", None));
        assert_eq!(bandit.all_arms(None).len(), 2);
        assert!(bandit.all_arms(None).iter().all(|a| a.id != "m2"));
    }

    #[test]
    fn test_remove_arm_task_specific() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("code", "m1");
        bandit.add_arm_for_task("code", "m2");
        bandit.add_arm_for_task("math", "m3");

        assert!(bandit.remove_arm("m1", Some("code")));
        assert_eq!(bandit.all_arms_vec(Some("code")).len(), 1);
        assert_eq!(bandit.all_arms_vec(Some("code"))[0].id, "m2");
        // math unaffected
        assert_eq!(bandit.all_arms_vec(Some("math")).len(), 1);
    }

    #[test]
    fn test_remove_arm_not_found() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("m1");
        assert!(!bandit.remove_arm("m999", None));
        assert!(!bandit.remove_arm("m1", Some("nonexistent")));
    }

    /// Helper: invoke an MCP tool via handle_request and parse the JSON result.
    fn mcp_call(
        server: &crate::mcp_protocol::McpServer,
        tool_name: &str,
        args: serde_json::Value,
    ) -> serde_json::Value {
        use crate::mcp_protocol::McpRequest;
        let request = McpRequest::new("tools/call")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "name": tool_name,
                "arguments": args,
            }));
        let response = server.handle_request(request);
        assert!(response.error.is_none(), "MCP error: {:?}", response.error);
        let result = response.result.expect("no result");
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .expect("no content");
        let text = content[0]
            .get("text")
            .and_then(|t| t.as_str())
            .expect("no text");
        serde_json::from_str(text).expect("invalid JSON in response")
    }

    #[test]
    fn test_mcp_routing_get_stats() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.get_stats", serde_json::json!({}));
        assert_eq!(result.get("total_pulls").and_then(|v| v.as_u64()), Some(0));
        let arms = result.get("arms").and_then(|v| v.as_array()).unwrap();
        assert_eq!(arms.len(), 2);
    }

    #[test]
    fn test_mcp_routing_add_arm() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(
            &server,
            "routing.add_arm",
            serde_json::json!({
                "arm_id": "m_new",
                "alpha": 5.0,
                "beta": 1.0,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        // Verify arm was added
        let pipeline = shared.lock().unwrap();
        let arms = pipeline.bandit().all_arms(None);
        assert_eq!(arms.len(), 2);
        let new_arm = arms.iter().find(|a| a.id == "m_new").unwrap();
        assert!((new_arm.params.alpha - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mcp_routing_remove_arm() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2", "m3"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(
            &server,
            "routing.remove_arm",
            serde_json::json!({
                "arm_id": "m2",
            }),
        );
        assert_eq!(
            result.get("status").and_then(|v| v.as_str()),
            Some("removed")
        );

        let pipeline = shared.lock().unwrap();
        assert_eq!(pipeline.bandit().all_arms(None).len(), 2);
    }

    #[test]
    fn test_mcp_routing_warm_start() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(
            &server,
            "routing.warm_start",
            serde_json::json!({
                "arm_id": "m1",
                "alpha": 20.0,
                "beta": 2.0,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        let pipeline = shared.lock().unwrap();
        let arm = pipeline
            .bandit()
            .all_arms(None)
            .iter()
            .find(|a| a.id == "m1")
            .unwrap()
            .clone();
        assert!((arm.params.alpha - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_mcp_routing_record_outcome() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(
            &server,
            "routing.record_outcome",
            serde_json::json!({
                "arm_id": "m1",
                "success": true,
                "quality": 0.9,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_add_rule_and_recompile() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(
            &server,
            "routing.add_rule",
            serde_json::json!({
                "domain": "code",
                "arm_id": "m1",
                "priority": 10,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        // Should now have a DFA
        let pipeline = shared.lock().unwrap();
        assert!(pipeline.active_dfa().is_some());
        assert!(pipeline.source_nfa().is_some());
    }

    #[test]
    fn test_mcp_routing_add_rule_with_complexity() {
        let pipeline = RoutingPipeline::for_models(&["fast", "smart"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(
            &server,
            "routing.add_rule",
            serde_json::json!({
                "domain": "code",
                "min_complexity": 70,
                "max_complexity": 100,
                "arm_id": "smart",
                "priority": 10,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(
            result.get("conditions_count").and_then(|v| v.as_u64()),
            Some(2)
        );
    }

    #[test]
    fn test_mcp_routing_force_resynthesize() {
        let config = PipelineConfig {
            min_pulls_for_synthesis: 1,
            quality_threshold: 0.0,
            ..PipelineConfig::default()
        };
        let mut pipeline = RoutingPipeline::for_models(&["m1", "m2"], config);
        pipeline.add_arm("m1");
        pipeline.record_outcome(&ArmFeedback {
            arm_id: "m1".into(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: Some("code".into()),
        });
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.force_resynthesize", serde_json::json!({}));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_export_import_roundtrip() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        // Export
        let exported = mcp_call(&server, "routing.export", serde_json::json!({}));
        let json = exported
            .get("pipeline_json")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(!json.is_empty());

        // Import into same pipeline
        let result = mcp_call(
            &server,
            "routing.import",
            serde_json::json!({
                "pipeline_json": json,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_get_config() {
        let config = PipelineConfig {
            synthesis_interval: 42,
            ..PipelineConfig::default()
        };
        let pipeline = RoutingPipeline::for_models(&["m1"], config);
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.get_config", serde_json::json!({}));
        assert_eq!(
            result.get("synthesis_interval").and_then(|v| v.as_u64()),
            Some(42)
        );
        assert_eq!(
            result.get("bandit_arms_count").and_then(|v| v.as_u64()),
            Some(1)
        );
    }

    #[test]
    fn test_mcp_routing_add_arm_for_task() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(
            &server,
            "routing.add_arm",
            serde_json::json!({
                "arm_id": "code_specialist",
                "task_type": "code",
                "alpha": 10.0,
                "beta": 1.0,
            }),
        );
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(
            result.get("task_type").and_then(|v| v.as_str()),
            Some("code")
        );

        let pipeline = shared.lock().unwrap();
        assert!(pipeline.bandit().task_types().contains(&"code"));
    }

    #[test]
    fn test_mcp_routing_full_workflow() {
        // Full MCP workflow: add arm → add rule → record outcome → get stats
        let pipeline = RoutingPipeline::for_models(&["base"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        // 1. Add a specialist arm
        mcp_call(
            &server,
            "routing.add_arm",
            serde_json::json!({
                "arm_id": "code_expert",
                "alpha": 8.0,
                "beta": 1.0,
            }),
        );

        // 2. Add a routing rule for it
        mcp_call(
            &server,
            "routing.add_rule",
            serde_json::json!({
                "domain": "code",
                "min_complexity": 50,
                "arm_id": "code_expert",
                "priority": 20,
            }),
        );

        // 3. Record some outcomes
        mcp_call(
            &server,
            "routing.record_outcome",
            serde_json::json!({
                "arm_id": "code_expert",
                "success": true,
                "quality": 0.95,
            }),
        );

        // 4. Check stats
        let stats = mcp_call(&server, "routing.get_stats", serde_json::json!({}));
        let arms = stats.get("arms").and_then(|v| v.as_array()).unwrap();
        assert!(arms.len() >= 2); // base + code_expert
        assert!(stats.get("has_dfa").and_then(|v| v.as_bool()).unwrap());

        // 5. Export and verify non-empty
        let exported = mcp_call(&server, "routing.export", serde_json::json!({}));
        assert!(
            exported
                .get("pipeline_json")
                .and_then(|v| v.as_str())
                .unwrap()
                .len()
                > 10
        );
    }
}
