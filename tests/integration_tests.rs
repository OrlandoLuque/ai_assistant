//! Integration tests for AI Assistant new modules
//!
//! These tests verify that the new modules work correctly.

// === Multi-Agent Integration Tests ===

mod multi_agent_integration_tests {
    use ai_assistant::agent_memory::*;
    use ai_assistant::multi_agent::*;

    #[test]
    fn test_multi_agent_workflow_collaboration() {
        // Create orchestrator
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::BestFit);

        // Register specialized agents
        orchestrator.register_agent(
            Agent::new("coordinator", "Coordinator", AgentRole::Coordinator)
                .with_capability("coordinate")
                .with_capability("manage"),
        );

        orchestrator.register_agent(
            Agent::new("researcher", "Researcher", AgentRole::Researcher)
                .with_capability("search")
                .with_capability("gather")
                .with_capability("investigate"),
        );

        orchestrator.register_agent(
            Agent::new("analyst", "Analyst", AgentRole::Analyst)
                .with_capability("analyze")
                .with_capability("process")
                .with_capability("evaluate"),
        );

        orchestrator.register_agent(
            Agent::new("writer", "Writer", AgentRole::Writer)
                .with_capability("write")
                .with_capability("draft")
                .with_capability("compose"),
        );

        orchestrator.register_agent(
            Agent::new("reviewer", "Reviewer", AgentRole::Reviewer)
                .with_capability("review")
                .with_capability("validate")
                .with_capability("check"),
        );

        // Create a pipeline of tasks with dependencies
        orchestrator.add_task(
            AgentTask::new("research", "Search and gather information on AI trends")
                .with_priority(10),
        );

        orchestrator.add_task(
            AgentTask::new("analyze", "Analyze the gathered data and extract insights")
                .with_priority(9)
                .depends_on("research"),
        );

        orchestrator.add_task(
            AgentTask::new("write", "Write a comprehensive report based on analysis")
                .with_priority(8)
                .depends_on("analyze"),
        );

        orchestrator.add_task(
            AgentTask::new("review", "Review and validate the final report")
                .with_priority(7)
                .depends_on("write"),
        );

        // Execute the workflow
        let mut completed_tasks = 0;

        // Phase 1: Research
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "research");
        assert_eq!(assignments[0].1, "researcher");

        orchestrator
            .complete_task("research", "Found 50 relevant articles on AI trends")
            .unwrap();
        completed_tasks += 1;

        // Phase 2: Analysis
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "analyze");
        assert_eq!(assignments[0].1, "analyst");

        orchestrator
            .complete_task(
                "analyze",
                "Identified 5 major trends: LLMs, multimodal, agents, safety, efficiency",
            )
            .unwrap();
        completed_tasks += 1;

        // Phase 3: Writing
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "write");
        assert_eq!(assignments[0].1, "writer");

        orchestrator
            .complete_task("write", "20-page report drafted with executive summary")
            .unwrap();
        completed_tasks += 1;

        // Phase 4: Review
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "review");
        assert_eq!(assignments[0].1, "reviewer");

        orchestrator
            .complete_task("review", "Report validated and approved for publication")
            .unwrap();
        completed_tasks += 1;

        // Verify final state
        let status = orchestrator.get_status();
        assert_eq!(status.completed_tasks, 4);
        assert_eq!(status.pending_tasks, 0);
        assert_eq!(status.failed_tasks, 0);
        assert_eq!(completed_tasks, 4);
    }

    #[test]
    fn test_multi_agent_with_shared_memory() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);
        let mut memory = SharedMemory::new();

        // Register agents
        orchestrator.register_agent(Agent::new("agent1", "Agent 1", AgentRole::Researcher));
        orchestrator.register_agent(Agent::new("agent2", "Agent 2", AgentRole::Analyst));

        // Agent 1 stores research data
        let entry1 = MemoryEntry::new(
            "research_data",
            "Important research findings about AI",
            MemoryType::Result,
            "agent1",
        )
        .share_with("agent2");

        let data_id = memory.store(entry1);

        // Agent 2 can access the shared data
        let retrieved = memory.get(&data_id, "agent2");
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().value.contains("research findings"));

        // Agents communicate via messages
        let msg = AgentMessage::new(
            "agent1",
            "agent2",
            "Research complete, please begin analysis",
            MessageType::Handoff,
        );
        orchestrator.send_message(msg);

        let messages = orchestrator.get_messages_for("agent2");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_type, MessageType::Handoff);
    }

    #[test]
    fn test_parallel_task_execution() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);

        // Register multiple workers
        for i in 0..5 {
            orchestrator.register_agent(Agent::new(
                &format!("worker_{}", i),
                &format!("Worker {}", i),
                AgentRole::Executor,
            ));
        }

        // Add independent tasks (no dependencies)
        for i in 0..5 {
            orchestrator.add_task(AgentTask::new(
                &format!("task_{}", i),
                &format!("Independent task {}", i),
            ));
        }

        // All tasks should be assignable at once
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 5);

        // Complete all tasks
        for i in 0..5 {
            orchestrator
                .complete_task(&format!("task_{}", i), "Done")
                .unwrap();
        }

        let status = orchestrator.get_status();
        assert_eq!(status.completed_tasks, 5);
        assert_eq!(status.idle_agents, 5);
    }

    #[test]
    fn test_agent_failure_recovery() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("primary", "Primary", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("backup", "Backup", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Important task"));

        // Assign to primary
        orchestrator.assign_task("task1", "primary").unwrap();

        // Primary fails
        orchestrator
            .fail_task("task1", "Primary agent encountered error")
            .unwrap();

        // Check that primary is marked as failed
        let primary = orchestrator.get_agent("primary").unwrap();
        assert_eq!(primary.status, AgentStatus::Failed);

        // Task is marked as failed
        let task = orchestrator.get_task("task1").unwrap();
        assert_eq!(task.status, TaskStatus::Failed);

        // In real scenario, we'd create a new task for backup to handle
        orchestrator.add_task(AgentTask::new("task1_retry", "Important task (retry)"));
        orchestrator.assign_task("task1_retry", "backup").unwrap();
        orchestrator
            .complete_task("task1_retry", "Completed by backup")
            .unwrap();

        let backup = orchestrator.get_agent("backup").unwrap();
        assert_eq!(backup.status, AgentStatus::Idle);
    }

    #[test]
    fn test_complex_dependency_graph() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("worker", "Worker", AgentRole::Executor));

        // Create a diamond dependency pattern:
        //       A
        //      / \
        //     B   C
        //      \ /
        //       D
        orchestrator.add_task(AgentTask::new("A", "Task A"));
        orchestrator.add_task(AgentTask::new("B", "Task B").depends_on("A"));
        orchestrator.add_task(AgentTask::new("C", "Task C").depends_on("A"));
        orchestrator.add_task(
            AgentTask::new("D", "Task D")
                .depends_on("B")
                .depends_on("C"),
        );

        // Only A is executable initially
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "A");

        orchestrator.complete_task("A", "Done").unwrap();

        // Now B and C are executable
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1); // Only one worker
        let first = assignments[0].0.clone();
        assert!(first == "B" || first == "C");

        orchestrator.complete_task(&first, "Done").unwrap();

        // Complete the other
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        let second = assignments[0].0.clone();
        orchestrator.complete_task(&second, "Done").unwrap();

        // Now D is executable
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, "D");

        orchestrator.complete_task("D", "Done").unwrap();

        let status = orchestrator.get_status();
        assert_eq!(status.completed_tasks, 4);
    }

    #[test]
    fn test_memory_lifecycle_across_agents() {
        let mut memory = SharedMemory::new();

        // Agent creates temporary data with TTL
        let temp_entry = MemoryEntry::new(
            "temp_cache",
            "Cached computation result",
            MemoryType::Temporary,
            "compute_agent",
        )
        .with_ttl(std::time::Duration::from_secs(3600))
        .share_with("consumer_agent");

        let temp_id = memory.store(temp_entry);

        // Store permanent result
        let result_entry = MemoryEntry::new(
            "final_result",
            "The final computed result",
            MemoryType::Result,
            "compute_agent",
        )
        .with_metadata("computation_time", "150ms")
        .with_metadata("accuracy", "99.5%");

        let result_id = memory.store_global(result_entry);

        // Consumer can access both
        assert!(memory.get(&temp_id, "consumer_agent").is_some());
        assert!(memory.get(&result_id, "consumer_agent").is_some());

        // Random agent can only access global
        assert!(memory.get(&temp_id, "random_agent").is_none());
        assert!(memory.get(&result_id, "random_agent").is_some());

        // Check stats
        let stats = memory.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.global_entries, 1);
    }
}

// === Chaos Engineering Tests ===

mod chaos_engineering_tests {
    use ai_assistant::agent_memory::*;
    use ai_assistant::multi_agent::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::thread;
    use std::time::Duration;

    /// Test behavior under random agent failures
    #[test]
    fn chaos_random_agent_failures() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);
        let failure_count = Arc::new(AtomicUsize::new(0));

        // Register many agents
        for i in 0..10 {
            orchestrator.register_agent(Agent::new(
                &format!("agent_{}", i),
                &format!("Agent {}", i),
                AgentRole::Executor,
            ));
        }

        // Add many tasks
        for i in 0..20 {
            orchestrator.add_task(AgentTask::new(
                &format!("task_{}", i),
                &format!("Task {}", i),
            ));
        }

        // Simulate chaotic execution with random failures
        let mut completed = 0;
        let mut failed = 0;

        for _ in 0..50 {
            let assignments = orchestrator.auto_assign_tasks();
            if assignments.is_empty() {
                break;
            }

            for (task_id, _agent_id) in assignments {
                // Simulate 20% failure rate
                if rand_simple() % 5 == 0 {
                    orchestrator.fail_task(&task_id, "Random failure").ok();
                    failed += 1;
                    failure_count.fetch_add(1, Ordering::SeqCst);
                } else {
                    orchestrator.complete_task(&task_id, "Success").ok();
                    completed += 1;
                }
            }
        }

        let status = orchestrator.get_status();
        assert_eq!(
            status.completed_tasks + status.failed_tasks,
            completed + failed
        );
    }

    /// Test concurrent memory access under stress
    #[test]
    fn chaos_concurrent_memory_stress() {
        let memory = ThreadSafeMemory::new();
        let ops_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads doing various memory operations
        for thread_id in 0..10 {
            let mem = memory.clone();
            let ops = ops_count.clone();

            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let agent = format!("agent_{}", thread_id);
                    let key = format!("key_{}_{}", thread_id, i);

                    // Store
                    let entry = MemoryEntry::new(
                        &key,
                        &format!("value_{}_{}", thread_id, i),
                        MemoryType::Fact,
                        &agent,
                    );
                    let id = mem.store(entry);
                    ops.fetch_add(1, Ordering::SeqCst);

                    // Read
                    mem.get(&id, &agent);
                    ops.fetch_add(1, Ordering::SeqCst);

                    // Update (sometimes)
                    if i % 3 == 0 {
                        let _ = mem.update(&id, &format!("updated_{}", i), &agent);
                        ops.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked during chaos test");
        }

        // Verify operations completed
        let total_ops = ops_count.load(Ordering::SeqCst);
        assert!(
            total_ops >= 2000,
            "Expected at least 2000 ops, got {}",
            total_ops
        );
    }

    /// Test system behavior with many rapid state transitions
    #[test]
    fn chaos_rapid_state_transitions() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);

        // Register many agents to handle the workload and absorb failures
        for i in 0..100 {
            orchestrator.register_agent(Agent::new(
                &format!("agent_{}", i),
                &format!("Agent {}", i),
                AgentRole::Executor,
            ));
        }

        // Add all tasks first
        for i in 0..100 {
            let task_id = format!("rapid_task_{}", i);
            orchestrator.add_task(AgentTask::new(&task_id, &format!("Rapid task {}", i)));
        }

        // Process all tasks - since we have 100 agents and 100 tasks with RoundRobin,
        // all tasks should be assigned in one batch
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 100, "All 100 tasks should be assigned");

        // Complete/fail all tasks
        let mut completed_count = 0;
        let mut failed_count = 0;
        for (i, (task_id, _agent_id)) in assignments.into_iter().enumerate() {
            // Alternate between success and failure
            if i % 2 == 0 {
                orchestrator.complete_task(&task_id, "Done").unwrap();
                completed_count += 1;
            } else {
                orchestrator.fail_task(&task_id, "Failed").unwrap();
                failed_count += 1;
            }
        }

        let status = orchestrator.get_status();
        assert_eq!(status.total_tasks, 100);
        assert_eq!(status.completed_tasks, completed_count);
        assert_eq!(status.failed_tasks, failed_count);
        assert_eq!(status.completed_tasks + status.failed_tasks, 100);
    }

    /// Test behavior with massive task dependencies
    #[test]
    fn chaos_massive_dependency_chain() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("worker", "Worker", AgentRole::Executor));

        // Create a long chain of 50 dependent tasks
        orchestrator.add_task(AgentTask::new("chain_0", "Start of chain"));

        for i in 1..50 {
            orchestrator.add_task(
                AgentTask::new(&format!("chain_{}", i), &format!("Chain task {}", i))
                    .depends_on(&format!("chain_{}", i - 1)),
            );
        }

        // Execute the entire chain
        for i in 0..50 {
            let assignments = orchestrator.auto_assign_tasks();
            assert_eq!(
                assignments.len(),
                1,
                "Expected exactly 1 task at iteration {}",
                i
            );
            assert_eq!(assignments[0].0, format!("chain_{}", i));

            orchestrator
                .complete_task(&format!("chain_{}", i), "Done")
                .unwrap();
        }

        let status = orchestrator.get_status();
        assert_eq!(status.completed_tasks, 50);
    }

    /// Test memory under TTL expiration pressure
    #[test]
    fn chaos_memory_ttl_expiration_race() {
        let mut memory = SharedMemory::new();

        // Create entries with very short TTL
        for i in 0..50 {
            let entry = MemoryEntry::new(
                &format!("expiring_{}", i),
                &format!("value_{}", i),
                MemoryType::Temporary,
                "agent",
            )
            .with_ttl(Duration::from_millis(1));

            memory.store(entry);
        }

        // Create some permanent entries
        for i in 0..10 {
            let entry = MemoryEntry::new(
                &format!("permanent_{}", i),
                &format!("value_{}", i),
                MemoryType::Fact,
                "agent",
            );
            memory.store(entry);
        }

        // Wait for expiration
        thread::sleep(Duration::from_millis(50));

        // Cleanup
        memory.cleanup_expired();

        let stats = memory.stats();
        assert_eq!(
            stats.total_entries, 10,
            "Only permanent entries should remain"
        );
    }

    /// Test orchestrator with agents being registered/unregistered rapidly
    #[test]
    fn chaos_agent_churn() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);

        // Churn agents while tasks are being processed
        for cycle in 0..10 {
            // Register agents
            for i in 0..5 {
                orchestrator.register_agent(Agent::new(
                    &format!("cycle_{}_agent_{}", cycle, i),
                    "Worker",
                    AgentRole::Executor,
                ));
            }

            // Add and process tasks
            for i in 0..3 {
                let task_id = format!("cycle_{}_task_{}", cycle, i);
                orchestrator.add_task(AgentTask::new(&task_id, "Task"));

                let assignments = orchestrator.auto_assign_tasks();
                for (t_id, _) in assignments {
                    orchestrator.complete_task(&t_id, "Done").ok();
                }
            }

            // Unregister some agents
            for i in 0..3 {
                orchestrator.unregister_agent(&format!("cycle_{}_agent_{}", cycle, i));
            }
        }

        let status = orchestrator.get_status();
        assert!(
            status.completed_tasks > 0,
            "Some tasks should have completed"
        );
    }

    /// Test message flooding between agents
    #[test]
    fn chaos_message_flood() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        // Register agents
        for i in 0..5 {
            orchestrator.register_agent(Agent::new(
                &format!("agent_{}", i),
                &format!("Agent {}", i),
                AgentRole::Executor,
            ));
        }

        // Flood with messages
        for i in 0..1000 {
            let from = format!("agent_{}", i % 5);
            let to = format!("agent_{}", (i + 1) % 5);

            let msg = AgentMessage::new(
                &from,
                &to,
                &format!("Message {}", i),
                match i % 6 {
                    0 => MessageType::Request,
                    1 => MessageType::Response,
                    2 => MessageType::Notification,
                    3 => MessageType::Error,
                    4 => MessageType::Handoff,
                    _ => MessageType::Status,
                },
            );

            orchestrator.send_message(msg);
        }

        // Verify messages are stored and retrievable
        let mut total_messages = 0;
        for i in 0..5 {
            let messages = orchestrator.get_messages_for(&format!("agent_{}", i));
            total_messages += messages.len();
        }

        assert_eq!(total_messages, 1000);
    }

    /// Test memory search under heavy load
    #[test]
    fn chaos_memory_search_stress() {
        let mut memory = SharedMemory::new();

        // Create many entries with varied content
        let keywords = [
            "rust",
            "python",
            "java",
            "golang",
            "typescript",
            "cpp",
            "csharp",
        ];

        for i in 0..500 {
            let keyword = keywords[i % keywords.len()];
            let entry = MemoryEntry::new(
                &format!("entry_{}", i),
                &format!(
                    "This entry is about {} programming language number {}",
                    keyword, i
                ),
                MemoryType::Fact,
                "agent",
            );
            memory.store(entry);
        }

        // Perform many searches
        for keyword in &keywords {
            let results = memory.search(keyword, "agent");
            assert!(!results.is_empty(), "Should find results for '{}'", keyword);
        }

        // Search for something that doesn't exist
        let results = memory.search("nonexistent_keyword_xyz", "agent");
        assert!(results.is_empty());
    }

    /// Test best-fit assignment under adversarial conditions
    #[test]
    fn chaos_best_fit_adversarial() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::BestFit);

        // Create agents with overlapping capabilities
        orchestrator.register_agent(
            Agent::new("generalist", "Generalist", AgentRole::Executor)
                .with_capability("a")
                .with_capability("b")
                .with_capability("c"),
        );

        orchestrator.register_agent(
            Agent::new("specialist_a", "Specialist A", AgentRole::Executor).with_capability("a"),
        );

        orchestrator.register_agent(
            Agent::new("specialist_ab", "Specialist AB", AgentRole::Executor)
                .with_capability("a")
                .with_capability("b"),
        );

        // Task that matches 'a' should go to most capable available agent
        orchestrator.add_task(AgentTask::new("task_a", "Task requiring A capability"));

        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
        // The generalist has 'a' in capabilities, but the task description doesn't contain 'a'
        // so BestFit looks at the description matching capabilities

        // Test with description that clearly matches
        orchestrator.add_task(AgentTask::new("task_b", "This task needs b capability"));

        // Complete previous task first
        orchestrator.complete_task(&assignments[0].0, "Done").ok();

        let assignments = orchestrator.auto_assign_tasks();
        if !assignments.is_empty() {
            // Should prefer agent with 'b' capability
            let assigned = &assignments[0].1;
            let agent = orchestrator.get_agent(assigned).unwrap();
            assert!(
                agent.capabilities.contains(&"b".to_string()),
                "Should assign to agent with 'b' capability"
            );
        }
    }

    /// Test behavior with tasks that have circular-like dependencies (should still work)
    #[test]
    fn chaos_dependency_validation() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("worker", "Worker", AgentRole::Executor));

        // Create tasks with complex but valid dependencies
        orchestrator.add_task(AgentTask::new("root", "Root task"));

        // Fan out
        for i in 0..5 {
            orchestrator.add_task(
                AgentTask::new(&format!("branch_{}", i), &format!("Branch {}", i))
                    .depends_on("root"),
            );
        }

        // Fan in
        let mut final_task = AgentTask::new("final", "Final task");
        for i in 0..5 {
            final_task = final_task.depends_on(&format!("branch_{}", i));
        }
        orchestrator.add_task(final_task);

        // Execute
        // First, only root is available
        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments[0].0, "root");
        orchestrator.complete_task("root", "Done").unwrap();

        // Complete all branches
        for _ in 0..5 {
            let assignments = orchestrator.auto_assign_tasks();
            if !assignments.is_empty() {
                orchestrator
                    .complete_task(&assignments[0].0, "Done")
                    .unwrap();
            }
        }

        // Final task should now be available
        let assignments = orchestrator.auto_assign_tasks();
        if !assignments.is_empty() {
            assert_eq!(assignments[0].0, "final");
        }
    }

    /// Test system resilience to rapid operations
    #[test]
    fn chaos_operation_burst() {
        let memory = ThreadSafeMemory::new();
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new(
            "burst_agent",
            "Burst Agent",
            AgentRole::Executor,
        ));

        // Burst of operations
        let start = std::time::Instant::now();

        for i in 0..1000 {
            // Memory operations
            let entry = MemoryEntry::new(
                &format!("burst_key_{}", i),
                &format!("burst_value_{}", i),
                MemoryType::Fact,
                "burst_agent",
            );
            let id = memory.store(entry);
            memory.get(&id, "burst_agent");

            // Orchestrator operations
            orchestrator.add_task(AgentTask::new(&format!("burst_task_{}", i), "Burst task"));
        }

        let elapsed = start.elapsed();

        // Should complete within reasonable time (< 5 seconds)
        assert!(
            elapsed.as_secs() < 5,
            "Operations took too long: {:?}",
            elapsed
        );
    }

    // Simple pseudo-random number generator for tests (no external dependency)
    fn rand_simple() -> usize {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as usize;
        nanos.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31)
    }
}

// === Error Module Tests ===

mod error_tests {
    use ai_assistant::error::*;

    #[test]
    fn test_error_chain() {
        let config_err = ConfigError::MissingValue {
            field: "api_key".to_string(),
            description: "Required for authentication".to_string(),
        };
        let ai_err: AiError = config_err.into();

        assert_eq!(ai_err.code(), "CONFIG");
        assert!(ai_err.suggestion().is_some());
        assert!(!ai_err.is_recoverable());
    }

    #[test]
    fn test_provider_error_recoverable() {
        let err = AiError::rate_limited(100, 60);
        assert!(err.is_recoverable());

        let err = AiError::model_not_found("ollama", "nonexistent");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_suggestions() {
        let err = AiError::context_exceeded(4096, 5000);
        assert!(err.suggestion().unwrap().contains("RAG"));

        let err = AiError::provider_unavailable("Ollama", "http://localhost:11434");
        assert!(err.suggestion().unwrap().contains("running"));
    }

    #[test]
    fn test_rag_errors() {
        let err = AiError::append_only_violation("delete", "guide.md");
        assert!(err.to_string().contains("append-only"));
        assert!(err.suggestion().unwrap().contains("set_append_only_mode"));
    }
}

// === Progress Module Tests ===

mod progress_tests {
    use ai_assistant::progress::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;

    #[test]
    fn test_progress_percentage() {
        let p = Progress::new("test", 25, 100);
        assert_eq!(p.percentage(), 25);

        let p = Progress::new("test", 0, 0);
        assert_eq!(p.percentage(), 0);

        let p = Progress::complete("test");
        assert_eq!(p.percentage(), 100);
    }

    #[test]
    fn test_progress_reporter() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let callback: ProgressCallback = Box::new(move |_p| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let mut reporter = ProgressReporter::new(Some(callback));
        reporter.set_min_interval(Duration::from_millis(0));
        reporter.start("test_op", 5);

        for i in 1..=5 {
            reporter.update(i, format!("Step {}", i));
        }

        reporter.complete("Done!");

        assert!(count.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_progress_time_estimates() {
        let mut p = Progress::new("test", 50, 100);
        p.elapsed_ms = 5000; // 5 seconds for 50 items
        p.estimate_remaining();

        assert!(p.remaining_ms.is_some());
        let remaining = p.remaining_ms.unwrap();
        assert!(remaining > 4000 && remaining < 6000);
    }

    #[test]
    fn test_progress_aggregator() {
        let agg = ProgressAggregator::new("batch", 5, None);

        agg.record_success();
        agg.record_success();
        agg.record_success();
        agg.record_failure();
        agg.record_failure();

        let progress = agg.get_progress();
        assert!(progress.is_complete);
        assert!(progress.is_error);
        assert_eq!(progress.current, 5);
        assert!(progress.message.contains("3"));
        assert!(progress.message.contains("2 failed"));
    }

    #[test]
    fn test_remaining_time_format() {
        let mut p = Progress::new("test", 50, 100);
        p.remaining_ms = Some(65000); // 65 seconds
        assert_eq!(p.remaining_human(), Some("1m 5s".to_string()));

        p.remaining_ms = Some(3665000); // 1h 1m 5s
        assert_eq!(p.remaining_human(), Some("1h 1m".to_string()));
    }
}

// === Config File Module Tests ===

mod config_file_tests {
    use ai_assistant::config_file::*;
    use ai_assistant::AiProvider;

    #[test]
    fn test_parse_toml_basic() {
        let toml = r#"
[provider]
type = "ollama"
model = "llama2"

[generation]
temperature = 0.8
max_history = 30
"#;
        let config = ConfigFile::parse(toml, ConfigFormat::Toml).unwrap();
        assert_eq!(config.provider.provider_type, "ollama");
        assert_eq!(config.provider.model, "llama2");
        assert_eq!(config.generation.temperature, 0.8);
        assert_eq!(config.generation.max_history, 30);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "provider": {
                "type": "lmstudio",
                "model": "mistral-7b"
            },
            "generation": {
                "temperature": 0.5
            },
            "rag": {
                "knowledge_enabled": true,
                "knowledge_tokens": 3000
            }
        }"#;

        let config = ConfigFile::parse(json, ConfigFormat::Json).unwrap();
        assert_eq!(config.provider.provider_type, "lmstudio");
        assert_eq!(config.provider.model, "mistral-7b");
        assert_eq!(config.generation.temperature, 0.5);
        assert!(config.rag.knowledge_enabled);
        assert_eq!(config.rag.knowledge_tokens, 3000);
    }

    #[test]
    fn test_to_ai_config() {
        let mut config = ConfigFile::default();
        config.provider.provider_type = "ollama".to_string();
        config.provider.model = "phi3".to_string();
        config.generation.temperature = 0.9;
        config.generation.max_history = 50;

        let ai_config = config.to_ai_config();
        assert!(matches!(ai_config.provider, AiProvider::Ollama));
        assert_eq!(ai_config.selected_model, "phi3");
        assert_eq!(ai_config.temperature, 0.9);
        assert_eq!(ai_config.max_history_messages, 50);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ConfigFormat::from_content("{\"key\": \"value\"}"),
            ConfigFormat::Json
        );
        assert_eq!(
            ConfigFormat::from_content("[section]\nkey = \"value\""),
            ConfigFormat::Toml
        );
        assert_eq!(
            ConfigFormat::from_content("key = \"value\""),
            ConfigFormat::Toml
        );
    }

    #[test]
    fn test_validation() {
        let mut config = ConfigFile::default();
        config.generation.temperature = 0.7;
        assert!(config.validate().is_ok());

        config.generation.temperature = 3.0; // Invalid
        assert!(config.validate().is_err());
    }
}

// === Memory Management Tests ===

mod memory_management_tests {
    use ai_assistant::memory_management::*;

    #[test]
    fn test_bounded_cache_lru_eviction() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(3, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Access "a" to make it recently used
        cache.get(&"a".to_string());

        // Insert "d", should evict "b" (least recently used after "a" was touched)
        cache.insert("d".to_string(), 4);

        assert!(cache.contains(&"a".to_string()));
        assert!(!cache.contains(&"b".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_bounded_cache_fifo_eviction() {
        let mut cache: BoundedCache<i32, &str> = BoundedCache::new(3, EvictionPolicy::Fifo);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.insert(4, "four");

        assert!(!cache.contains(&1)); // First in, first out
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_bounded_cache_stats() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(10, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        cache.get(&"a".to_string()); // Hit
        cache.get(&"c".to_string()); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.entries, 2);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bounded_vec() {
        let mut vec: BoundedVec<i32> = BoundedVec::new(5);

        for i in 1..=10 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.eviction_count(), 5);
        assert_eq!(vec.get(0), Some(&6)); // Oldest kept is 6
        assert_eq!(vec.get(4), Some(&10)); // Newest is 10
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::with_limit(1000);

        tracker.register("cache", Some(500));
        tracker.register("embeddings", Some(500));

        tracker.update("cache", 200);
        tracker.update("embeddings", 300);

        assert_eq!(tracker.total_usage(), 500);
        assert_eq!(tracker.pressure(), MemoryPressure::Normal);

        // Push into warning zone
        tracker.update("cache", 450);
        tracker.update("embeddings", 450);

        assert_eq!(tracker.total_usage(), 900);
        assert_eq!(tracker.pressure(), MemoryPressure::Warning);

        // Push into critical zone
        tracker.update("cache", 600);

        assert_eq!(tracker.pressure(), MemoryPressure::Critical);
    }

    #[test]
    fn test_memory_report() {
        let mut tracker = MemoryTracker::with_limit(1000);

        tracker.register("cache", None);
        tracker.register("data", None);

        tracker.update("cache", 200);
        tracker.update("data", 300);

        let report = tracker.report();

        assert_eq!(report.total_bytes, 500);
        assert!(report.limit_bytes == Some(1000));
        assert_eq!(report.components.len(), 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
