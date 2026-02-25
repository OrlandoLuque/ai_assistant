//! Agent DevTools demo — debugging, profiling, and execution replay.
//!
//! Demonstrates the agent debugging subsystem: recording events,
//! setting breakpoints, profiling performance, inspecting state,
//! and replaying an execution trace.
//!
//! Run with: cargo run --example devtools_demo --features "devtools"

use std::collections::HashMap;

use ai_assistant::{
    AgentDebugger, Breakpoint, DebugEvent, DebugEventType, DevToolsConfig, ExecutionRecorder,
    ExecutionReplay, PerformanceProfiler, ProfileSummary, StateInspector,
    StepProfile,
};

fn main() {
    println!("=== Agent DevTools Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. DevToolsConfig
    // -----------------------------------------------------------------------
    let config = DevToolsConfig {
        enable_recording: true,
        enable_profiling: true,
        max_recording_steps: 1000,
        breakpoints: vec![
            Breakpoint::OnError,
            Breakpoint::OnCostAbove { threshold: 5.0 },
        ],
    };
    println!("DevToolsConfig: recording={}, profiling={}, max_steps={}",
        config.enable_recording, config.enable_profiling, config.max_recording_steps);
    println!("Breakpoints configured: {}\n", config.breakpoints.len());

    // -----------------------------------------------------------------------
    // 2. ExecutionRecorder — record a sequence of events
    // -----------------------------------------------------------------------
    println!("--- ExecutionRecorder ---");
    let mut recorder = ExecutionRecorder::new("agent-demo", 500);
    recorder.start();
    println!("Recording: {}", recorder.is_recording());

    // Simulate a few agent steps
    let events = vec![
        {
            let mut e = DebugEvent::now(DebugEventType::PlanningStart, 0);
            e.data.insert("goal".to_string(), "Summarize document".to_string());
            e
        },
        DebugEvent::now(DebugEventType::PlanningEnd, 0),
        {
            let mut e = DebugEvent::now(DebugEventType::ToolCallStart, 1);
            e.tool_name = Some("search".to_string());
            e
        },
        {
            let mut e = DebugEvent::now(DebugEventType::ToolCallEnd, 1);
            e.tool_name = Some("search".to_string());
            e
        },
        DebugEvent::now(DebugEventType::LlmQueryStart, 2),
        {
            let mut e = DebugEvent::now(DebugEventType::LlmQueryEnd, 2);
            e.confidence = Some(0.92);
            e.cost = Some(0.003);
            e
        },
        DebugEvent::now(DebugEventType::StepComplete, 2),
    ];

    for event in events {
        let _ = recorder.record(event);
    }
    recorder.stop();

    println!("Recorded {} events for agent '{}'", recorder.event_count(), recorder.agent_id());

    // Filter by type
    let tool_events = recorder.events_by_type(&DebugEventType::ToolCallStart);
    println!("Tool call starts: {}", tool_events.len());

    // Range query
    let step_2_events = recorder.events_in_range(2, 2);
    println!("Events at step 2: {}", step_2_events.len());

    // -----------------------------------------------------------------------
    // 3. ExecutionReplay — step through recorded events
    // -----------------------------------------------------------------------
    println!("\n--- ExecutionReplay ---");
    let mut replay = ExecutionReplay::new(&recorder);
    println!("Replay for agent '{}', {} total events", replay.agent_id(), replay.total_events());

    while let Some(event) = replay.next() {
        println!(
            "  Step {} | {:?} | tool={:?} | conf={:?}",
            event.step_number, event.event_type, event.tool_name, event.confidence,
        );
    }
    println!("Replay complete: {} (progress: {:.0}%)", replay.is_complete(), replay.progress() * 100.0);

    // Reset and skip to a specific step
    replay.reset();
    if let Some(event) = replay.skip_to_step(2) {
        println!("Skipped to step 2: {:?}", event.event_type);
    }

    // -----------------------------------------------------------------------
    // 4. PerformanceProfiler — collect per-step metrics
    // -----------------------------------------------------------------------
    println!("\n--- PerformanceProfiler ---");
    let mut profiler = PerformanceProfiler::new();
    profiler.start();

    let _ = profiler.record_step(StepProfile {
        step_number: 0,
        action_name: "planning".to_string(),
        duration_ms: 120,
        token_count: 700,
        cost: 0.001,
        memory_delta_bytes: 1024,
    });
    let _ = profiler.record_step(StepProfile {
        step_number: 1,
        action_name: "search_tool".to_string(),
        duration_ms: 350,
        token_count: 2000,
        cost: 0.004,
        memory_delta_bytes: 4096,
    });
    let _ = profiler.record_step(StepProfile {
        step_number: 2,
        action_name: "llm_response".to_string(),
        duration_ms: 200,
        token_count: 450,
        cost: 0.0008,
        memory_delta_bytes: 512,
    });
    profiler.stop();

    let summary: ProfileSummary = profiler.summary();
    println!("Total steps: {}", summary.total_steps);
    println!("Total duration: {} ms", summary.total_duration_ms);
    println!("Total tokens: {}", summary.total_tokens);
    println!("Total cost: ${:.4}", summary.total_cost);
    println!("Avg step duration: {:.1} ms", summary.avg_duration_ms);
    println!("Slowest step name: {:?}", summary.slowest_step_name);
    println!("Most expensive step name: {:?}", summary.most_expensive_step_name);

    if let Some(slowest) = profiler.slowest_step() {
        println!("Slowest step: #{} '{}' ({} ms)", slowest.step_number, slowest.action_name, slowest.duration_ms);
    }
    if let Some(expensive) = profiler.most_expensive_step() {
        println!("Most expensive step: #{} '{}' (${:.4})", expensive.step_number, expensive.action_name, expensive.cost);
    }

    // -----------------------------------------------------------------------
    // 5. StateInspector — capture and diff state snapshots
    // -----------------------------------------------------------------------
    println!("\n--- StateInspector ---");
    let mut inspector = StateInspector::new(100);

    // capture(step_number, label, data)
    let mut state_0 = HashMap::new();
    state_0.insert("goal".to_string(), serde_json::json!("Summarize document"));
    state_0.insert("progress".to_string(), serde_json::json!(0));
    inspector.capture(0, "Initial state", state_0);

    let mut state_1 = HashMap::new();
    state_1.insert("goal".to_string(), serde_json::json!("Summarize document"));
    state_1.insert("progress".to_string(), serde_json::json!(50));
    state_1.insert("search_results".to_string(), serde_json::json!(3));
    inspector.capture(1, "After search tool call", state_1);

    println!("Snapshots captured: {}", inspector.snapshot_count());
    if let Some(latest) = inspector.latest() {
        println!("Latest snapshot at step {}: {} keys, label='{}'",
            latest.step_number, latest.state_data.len(), latest.label);
    }

    if let Some(diff) = inspector.diff(0, 1) {
        println!("Diff between step {} and {}:", diff.step_a, diff.step_b);
        println!("  Added keys: {:?}", diff.added_keys);
        println!("  Removed keys: {:?}", diff.removed_keys);
        println!("  Changed keys: {:?}", diff.changed_keys);
    }

    // -----------------------------------------------------------------------
    // 6. AgentDebugger — unified facade
    // -----------------------------------------------------------------------
    println!("\n--- AgentDebugger (unified facade) ---");
    let mut debugger = AgentDebugger::new("demo-agent", DevToolsConfig {
        enable_recording: true,
        enable_profiling: true,
        max_recording_steps: 500,
        breakpoints: vec![
            Breakpoint::BeforeToolCall { tool_name: "dangerous_tool".to_string() },
            Breakpoint::AfterStep { step_number: 5 },
            Breakpoint::OnConfidenceBelow { threshold: 0.5 },
            Breakpoint::AtStepCount { count: 10 },
        ],
    });
    debugger.start();

    // Process an event that triggers the "BeforeToolCall" breakpoint
    let mut tool_event = DebugEvent::now(DebugEventType::ToolCallStart, 3);
    tool_event.tool_name = Some("dangerous_tool".to_string());
    let hits = debugger.process_event(tool_event);
    println!("Breakpoints hit: {}", hits.len());
    for bp in hits {
        println!("  Hit: {:?}", bp);
    }

    debugger.stop();

    println!("Total recorded events: {}", debugger.recorder().event_count());
    println!("Total hit breakpoints: {}", debugger.hit_breakpoints().len());

    // Create a replay from the debugger
    let replay2 = debugger.create_replay();
    println!("Replay has {} events", replay2.total_events());

    println!("\nDevTools demo complete.");
}
