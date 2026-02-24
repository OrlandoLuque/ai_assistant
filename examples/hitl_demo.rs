//! Human-in-the-Loop (HITL) demo.
//!
//! Run with: cargo run --example hitl_demo --features hitl
//!
//! Demonstrates tool approval gates, confidence escalation,
//! policy engine, and approval workflows.

use ai_assistant::{
    ApprovalRequest, ApprovalDecision, HitlApprovalGate, ImpactLevel,
    AutoApproveGate, AutoDenyGate, CallbackApprovalGate,
    EscalationPolicy, EscalationThreshold, EscalationTrigger, EscalationAction,
    EscalationEvaluator, PolicyEngine, ApprovalPolicy, PolicyRule, PolicyCondition,
    PolicyAction,
};

fn main() {
    println!("=== Human-in-the-Loop (HITL) Demo ===\n");

    // 1. Create approval requests for different tools
    let read_req = ApprovalRequest::new(
        "req-001", "read_file",
        std::collections::HashMap::new(),
        "agent-1", "Reading configuration file",
        ImpactLevel::Low,
    );

    let delete_req = ApprovalRequest::new(
        "req-002", "delete_file",
        [("path".to_string(), serde_json::json!("/data/temp.log"))].into(),
        "agent-1", "Deleting temporary log file",
        ImpactLevel::High,
    );

    println!("Request 1: {} (impact: {:?})", read_req.tool_name, read_req.estimated_impact);
    println!("Request 2: {} (impact: {:?})", delete_req.tool_name, delete_req.estimated_impact);

    // 2. Test different approval gates
    let auto_approve = AutoApproveGate;
    let auto_deny = AutoDenyGate::new("All operations blocked during maintenance");

    println!("\n--- Auto-Approve Gate ---");
    println!("  read_file: {:?}", auto_approve.request_approval(&read_req).unwrap());
    println!("  delete_file: {:?}", auto_approve.request_approval(&delete_req).unwrap());

    println!("\n--- Auto-Deny Gate ---");
    println!("  read_file: {:?}", auto_deny.request_approval(&read_req).unwrap());

    // 3. Callback gate: approve low-impact, deny high-impact
    let smart_gate = CallbackApprovalGate::new("smart-gate", |req| {
        if req.estimated_impact.rank() >= ImpactLevel::High.rank() {
            ApprovalDecision::Deny {
                reason: "High-impact operations require human review".to_string(),
            }
        } else {
            ApprovalDecision::Approve
        }
    });

    println!("\n--- Smart Gate (callback) ---");
    println!("  read_file: {:?}", smart_gate.request_approval(&read_req).unwrap());
    println!("  delete_file: {:?}", smart_gate.request_approval(&delete_req).unwrap());

    // 4. Escalation policy with confidence thresholds
    let policy = EscalationPolicy {
        thresholds: vec![
            EscalationThreshold {
                trigger: EscalationTrigger::ConfidenceBelow(0.5),
                action: EscalationAction::Abort {
                    reason: "Confidence too low".to_string(),
                },
            },
            EscalationThreshold {
                trigger: EscalationTrigger::ConfidenceBelow(0.8),
                action: EscalationAction::RequestApproval,
            },
            EscalationThreshold {
                trigger: EscalationTrigger::ConsecutiveErrors(3),
                action: EscalationAction::PauseAndNotify,
            },
        ],
        default_action: EscalationAction::Continue,
    };

    let mut evaluator = EscalationEvaluator::new(policy);
    println!("\n--- Escalation Evaluator ---");
    println!("  High confidence (0.95): {:?}", evaluator.evaluate(0.95));
    println!("  Medium confidence (0.7): {:?}", evaluator.evaluate(0.7));
    println!("  Low confidence (0.3): {:?}", evaluator.evaluate(0.3));

    evaluator.record_error();
    evaluator.record_error();
    evaluator.record_error();
    println!("  After 3 errors, check_all(0.95): {:?}", evaluator.check_all_triggers(0.95));

    // 5. Policy engine with named rules
    let mut engine = PolicyEngine::new();
    engine.add_policy(ApprovalPolicy {
        name: "production-safety".to_string(),
        rules: vec![
            PolicyRule {
                name: "block-delete".to_string(),
                condition: PolicyCondition::ToolNameMatch("delete_file".to_string()),
                action: PolicyAction::RequireHumanApproval,
                priority: 10,
            },
            PolicyRule {
                name: "block-critical".to_string(),
                condition: PolicyCondition::ImpactAtLeast(ImpactLevel::Critical),
                action: PolicyAction::AutoDeny {
                    reason: "Critical operations blocked".to_string(),
                },
                priority: 20,
            },
        ],
        default_action: PolicyAction::AutoApprove,
    });

    println!("\n--- Policy Engine ---");
    println!("  read_file -> {:?}", engine.evaluate(&read_req));
    println!("  delete_file -> {:?}", engine.evaluate(&delete_req));

    println!("\n=== Done ===");
}
