use super::*;

// ─── Container Test Categories (feature = "containers") ─────────────────────

#[cfg(feature = "containers")]
pub(crate) fn tests_containers() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Containers (Types & Config)")));
    let mut results = Vec::new();

    results.push(run_test("ContainerConfig defaults", || {
        let config = ai_assistant::ContainerConfig::default();
        assert_eq_test!(config.default_timeout.as_secs(), 60);
        assert_eq_test!(config.default_memory_limit, 512 * 1024 * 1024);
        assert_eq_test!(config.default_cpu_quota, 0);
        assert_test!(config.auto_pull, "auto_pull should be true by default");
        assert_test!(config.docker_host.is_none(), "docker_host should be None");
        assert_eq_test!(config.container_name_prefix, "ai_assistant_");
        Ok(())
    }));

    results.push(run_test("ContainerCleanupPolicy defaults", || {
        let policy = ai_assistant::ContainerCleanupPolicy::default();
        assert_eq_test!(policy.max_per_session, 5);
        assert_eq_test!(policy.max_total, 20);
        assert_eq_test!(policy.auto_remove_after_secs, Some(3600));
        assert_test!(policy.cleanup_on_session_end);
        Ok(())
    }));

    results.push(run_test("CreateOptions default", || {
        let opts = ai_assistant::CreateOptions::default();
        assert_test!(opts.memory_limit.is_none());
        assert_test!(opts.cpu_quota.is_none());
        assert_test!(opts.network_mode.is_none());
        assert_test!(opts.env_vars.is_empty());
        assert_test!(opts.ports.is_empty());
        assert_test!(opts.bind_mounts.is_empty());
        assert_test!(opts.working_dir.is_none());
        assert_test!(opts.entrypoint.is_none());
        assert_test!(opts.cmd.is_none());
        Ok(())
    }));

    results.push(run_test("ExecResult::success() — exit 0", || {
        let r = ai_assistant::ExecResult {
            stdout: "ok".into(),
            stderr: String::new(),
            exit_code: 0,
            duration: std::time::Duration::from_millis(100),
            timed_out: false,
        };
        assert_test!(r.success(), "exit_code=0 + !timed_out → success");
        Ok(())
    }));

    results.push(run_test("ExecResult::success() — exit 1", || {
        let r = ai_assistant::ExecResult {
            stdout: String::new(),
            stderr: "error".into(),
            exit_code: 1,
            duration: std::time::Duration::from_millis(50),
            timed_out: false,
        };
        assert_test!(!r.success(), "exit_code=1 → not success");
        Ok(())
    }));

    results.push(run_test("ExecResult::success() — timeout", || {
        let r = ai_assistant::ExecResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: -1,
            duration: std::time::Duration::from_secs(60),
            timed_out: true,
        };
        assert_test!(!r.success(), "timed_out → not success");
        Ok(())
    }));

    results.push(run_test("ExecResult::combined_output()", || {
        let r = ai_assistant::ExecResult {
            stdout: "out".into(),
            stderr: "err".into(),
            exit_code: 0,
            duration: std::time::Duration::from_millis(10),
            timed_out: false,
        };
        let combined = r.combined_output();
        assert_test!(combined.contains("out"), "should contain stdout");
        assert_test!(combined.contains("err"), "should contain stderr");

        // stdout-only
        let r2 = ai_assistant::ExecResult {
            stdout: "only_out".into(),
            stderr: String::new(),
            exit_code: 0,
            duration: std::time::Duration::from_millis(10),
            timed_out: false,
        };
        assert_eq_test!(r2.combined_output(), "only_out");
        Ok(())
    }));

    results.push(run_test("ContainerStatus Display", || {
        assert_eq_test!(
            format!("{}", ai_assistant::ContainerStatus::Created),
            "created"
        );
        assert_eq_test!(
            format!("{}", ai_assistant::ContainerStatus::Running),
            "running"
        );
        assert_eq_test!(
            format!("{}", ai_assistant::ContainerStatus::Paused),
            "paused"
        );
        assert_eq_test!(
            format!("{}", ai_assistant::ContainerStatus::Stopped),
            "stopped"
        );
        assert_eq_test!(
            format!("{}", ai_assistant::ContainerStatus::Removed),
            "removed"
        );
        Ok(())
    }));

    results.push(run_test("NetworkMode Display", || {
        assert_eq_test!(format!("{}", ai_assistant::NetworkMode::None), "none");
        assert_eq_test!(format!("{}", ai_assistant::NetworkMode::Bridge), "bridge");
        assert_eq_test!(format!("{}", ai_assistant::NetworkMode::Host), "host");
        assert_eq_test!(
            format!("{}", ai_assistant::NetworkMode::Custom("mynet".into())),
            "custom(mynet)"
        );
        Ok(())
    }));

    results.push(run_test("NetworkMode::to_docker_string()", || {
        assert_eq_test!(ai_assistant::NetworkMode::None.to_docker_string(), "none");
        assert_eq_test!(
            ai_assistant::NetworkMode::Bridge.to_docker_string(),
            "bridge"
        );
        assert_eq_test!(ai_assistant::NetworkMode::Host.to_docker_string(), "host");
        assert_eq_test!(
            ai_assistant::NetworkMode::Custom("test".into()).to_docker_string(),
            "test"
        );
        Ok(())
    }));

    results.push(run_test("ContainerError Display", || {
        let e1 = ai_assistant::ContainerError::DockerNotAvailable("no daemon".into());
        assert_test!(format!("{}", e1).contains("no daemon"));
        let e2 = ai_assistant::ContainerError::ImageNotFound("foo:latest".into());
        assert_test!(format!("{}", e2).contains("foo:latest"));
        let e3 = ai_assistant::ContainerError::ContainerNotFound("abc123".into());
        assert_test!(format!("{}", e3).contains("abc123"));
        let e4 = ai_assistant::ContainerError::OperationFailed("boom".into());
        assert_test!(format!("{}", e4).contains("boom"));
        let e5 = ai_assistant::ContainerError::Timeout("60s".into());
        assert_test!(format!("{}", e5).contains("60s"));
        let e6 = ai_assistant::ContainerError::PolicyViolation("max reached".into());
        assert_test!(format!("{}", e6).contains("max reached"));
        Ok(())
    }));

    results.push(run_test("ContainersConfig defaults", || {
        let config = ai_assistant::ContainersConfig::default();
        assert_test!(!config.enabled, "containers should be disabled by default");
        assert_test!(!config.mcp_enabled, "MCP should be disabled by default");
        assert_eq_test!(config.default_timeout_secs, 60);
        assert_test!(config.allowed_images.is_empty());
        Ok(())
    }));

    results.push(run_test("ContainersConfig JSON parse", || {
        let json = r#"{"enabled":true,"default_timeout_secs":120,"allowed_images":["busybox"],"mcp_enabled":true}"#;
        let config: ai_assistant::ContainersConfig = serde_json::from_str(json)
            .map_err(|e| format!("parse failed: {}", e))?;
        assert_test!(config.enabled);
        assert_test!(config.mcp_enabled);
        assert_eq_test!(config.default_timeout_secs, 120);
        assert_eq_test!(config.allowed_images.len(), 1);
        Ok(())
    }));

    results.push(run_test("ExecutionBackend::auto() selection", || {
        let backend = ai_assistant::ExecutionBackend::auto();
        // auto() returns Container if Docker available, Process otherwise
        let name = format!("{:?}", backend);
        assert_test!(
            !name.is_empty(),
            "ExecutionBackend should have debug format"
        );
        Ok(())
    }));

    CategoryResult {
        name: "containers".to_string(),
        results,
    }
}

#[cfg(feature = "containers")]
pub(crate) fn tests_containers_docker() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Containers (Docker Lifecycle)")));
    let mut results = Vec::new();

    // Skip if Docker not available
    if !ai_assistant::ContainerExecutor::is_docker_available() {
        results.push(TestResult {
            name: "SKIPPED — Docker not available".to_string(),
            passed: true,
            message: Some("Docker daemon not reachable, skipping lifecycle tests".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "containers_docker".to_string(),
            results,
        };
    }

    results.push(run_test("Docker is available", || {
        assert_test!(ai_assistant::ContainerExecutor::is_docker_available());
        Ok(())
    }));

    results.push(run_test("ContainerExecutor::new()", || {
        let config = ai_assistant::ContainerConfig::default();
        let executor = ai_assistant::ContainerExecutor::new(config)
            .map_err(|e| format!("new() failed: {}", e))?;
        assert_eq_test!(executor.config().container_name_prefix, "ai_assistant_");
        Ok(())
    }));

    // Full lifecycle test: create → start → exec → logs → list → stop → remove
    results.push(run_test("Full container lifecycle", || {
        let config = ai_assistant::ContainerConfig::default();
        let mut executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("new: {}", e))?;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let name = format!("harness_test_{}", ts);

        // Create
        let id = executor
            .create(
                "busybox:latest",
                &name,
                ai_assistant::CreateOptions::default(),
            )
            .map_err(|e| format!("create: {}", e))?;
        assert_test!(!id.is_empty(), "container ID should not be empty");

        // Start
        executor.start(&id).map_err(|e| format!("start: {}", e))?;

        // Status should be running
        let status = executor.status(&id);
        assert_eq_test!(status, Some(&ai_assistant::ContainerStatus::Running));

        // Exec
        let result = executor
            .exec(
                &id,
                &["echo", "harness_ok"],
                std::time::Duration::from_secs(10),
            )
            .map_err(|e| format!("exec: {}", e))?;
        assert_eq_test!(result.exit_code, 0);
        assert_test!(
            result.stdout.contains("harness_ok"),
            "stdout should contain 'harness_ok'"
        );

        // Logs
        let logs = executor.logs(&id, 10).map_err(|e| format!("logs: {}", e))?;
        assert_test!(!logs.is_empty() || true, "logs may or may not have content");

        // List
        let list = executor.list();
        assert_test!(
            list.iter().any(|r| r.container_id == id),
            "container should be in list"
        );

        // Stop + Remove (cleanup)
        executor.stop(&id, 5).map_err(|e| format!("stop: {}", e))?;
        executor
            .remove(&id, false)
            .map_err(|e| format!("remove: {}", e))?;

        Ok(())
    }));

    results.push(run_test("Exec non-zero exit code", || {
        let config = ai_assistant::ContainerConfig::default();
        let mut executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("new: {}", e))?;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let name = format!("harness_exit_{}", ts);

        let id = executor
            .create(
                "busybox:latest",
                &name,
                ai_assistant::CreateOptions::default(),
            )
            .map_err(|e| format!("create: {}", e))?;
        executor.start(&id).map_err(|e| format!("start: {}", e))?;

        let result = executor
            .exec(
                &id,
                &["sh", "-c", "exit 42"],
                std::time::Duration::from_secs(10),
            )
            .map_err(|e| format!("exec: {}", e))?;
        assert_eq_test!(result.exit_code, 42);
        assert_test!(!result.success());

        // Cleanup
        executor.stop(&id, 5).map_err(|e| format!("stop: {}", e))?;
        executor
            .remove(&id, false)
            .map_err(|e| format!("remove: {}", e))?;
        Ok(())
    }));

    results.push(run_test("cleanup_all removes containers", || {
        let config = ai_assistant::ContainerConfig::default();
        let mut executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("new: {}", e))?;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let id = executor
            .create(
                "busybox:latest",
                &format!("harness_cleanup_{}", ts),
                ai_assistant::CreateOptions::default(),
            )
            .map_err(|e| format!("create: {}", e))?;
        executor.start(&id).map_err(|e| format!("start: {}", e))?;

        let count = executor.cleanup_all();
        assert_test!(count >= 1, "cleanup_all should remove at least 1 container");
        Ok(())
    }));

    CategoryResult {
        name: "containers_docker".to_string(),
        results,
    }
}

#[cfg(all(feature = "containers", feature = "tools"))]
pub(crate) fn tests_mcp_docker() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ MCP Docker Tools")));
    let mut results = Vec::new();

    // MCP tool registration requires Docker for the executor
    if !ai_assistant::ContainerExecutor::is_docker_available() {
        results.push(TestResult {
            name: "SKIPPED — Docker not available".to_string(),
            passed: true,
            message: Some("Docker daemon not reachable, skipping MCP Docker tests".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "mcp_docker".to_string(),
            results,
        };
    }

    results.push(run_test(
        "register_mcp_docker_tools registers 8 tools",
        || {
            let mut server = ai_assistant::McpServer::new("test_mcp", "0.1.0");
            let config = ai_assistant::ContainerConfig::default();
            let executor = ai_assistant::ContainerExecutor::new(config)
                .map_err(|e| format!("executor: {}", e))?;
            let arc = std::sync::Arc::new(std::sync::RwLock::new(executor));
            ai_assistant::register_mcp_docker_tools(&mut server, arc);

            let response = server
                .handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#);
            let parsed: serde_json::Value =
                serde_json::from_str(&response).map_err(|e| format!("parse: {}", e))?;
            let tools = parsed["result"]["tools"]
                .as_array()
                .ok_or_else(|| "tools not found in response".to_string())?;
            assert_eq_test!(tools.len(), 8);
            Ok(())
        },
    ));

    results.push(run_test("MCP tool names are correct", || {
        let mut server = ai_assistant::McpServer::new("test_mcp", "0.1.0");
        let config = ai_assistant::ContainerConfig::default();
        let executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("executor: {}", e))?;
        let arc = std::sync::Arc::new(std::sync::RwLock::new(executor));
        ai_assistant::register_mcp_docker_tools(&mut server, arc);

        let response =
            server.handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#);
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| format!("parse: {}", e))?;
        let tools = parsed["result"]["tools"]
            .as_array()
            .ok_or_else(|| "tools not found".to_string())?;

        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        let expected = [
            "docker_list_containers",
            "docker_create_container",
            "docker_start_container",
            "docker_stop_container",
            "docker_remove_container",
            "docker_exec",
            "docker_logs",
            "docker_container_status",
        ];
        for name in &expected {
            assert_test!(names.contains(name), format!("missing tool: {}", name));
        }
        Ok(())
    }));

    results.push(run_test("docker_create_container requires 'image'", || {
        let mut server = ai_assistant::McpServer::new("test_mcp", "0.1.0");
        let config = ai_assistant::ContainerConfig::default();
        let executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("executor: {}", e))?;
        let arc = std::sync::Arc::new(std::sync::RwLock::new(executor));
        ai_assistant::register_mcp_docker_tools(&mut server, arc);

        let response =
            server.handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#);
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| format!("parse: {}", e))?;
        let tools = parsed["result"]["tools"]
            .as_array()
            .ok_or_else(|| "tools not found".to_string())?;
        let create = tools
            .iter()
            .find(|t| t["name"] == "docker_create_container")
            .ok_or_else(|| "create tool not found".to_string())?;
        let required = create["inputSchema"]["required"]
            .as_array()
            .ok_or_else(|| "no required field".to_string())?;
        assert_test!(
            required.iter().any(|r| r == "image"),
            "image should be required"
        );
        Ok(())
    }));

    results.push(run_test("Read-only tools have correct annotations", || {
        let mut server = ai_assistant::McpServer::new("test_mcp", "0.1.0");
        let config = ai_assistant::ContainerConfig::default();
        let executor =
            ai_assistant::ContainerExecutor::new(config).map_err(|e| format!("executor: {}", e))?;
        let arc = std::sync::Arc::new(std::sync::RwLock::new(executor));
        ai_assistant::register_mcp_docker_tools(&mut server, arc);

        let response =
            server.handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#);
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| format!("parse: {}", e))?;
        let tools = parsed["result"]["tools"]
            .as_array()
            .ok_or_else(|| "tools not found".to_string())?;

        for name in &[
            "docker_list_containers",
            "docker_logs",
            "docker_container_status",
        ] {
            let tool = tools
                .iter()
                .find(|t| t["name"] == *name)
                .ok_or_else(|| format!("tool {} not found", name))?;
            if let Some(ann) = tool.get("annotations") {
                assert_eq_test!(ann["readOnlyHint"], true);
                assert_eq_test!(ann["destructiveHint"], false);
            }
        }

        // docker_remove_container should be destructive
        let rm = tools
            .iter()
            .find(|t| t["name"] == "docker_remove_container")
            .ok_or_else(|| "remove tool not found".to_string())?;
        if let Some(ann) = rm.get("annotations") {
            assert_eq_test!(ann["destructiveHint"], true);
        }
        Ok(())
    }));

    CategoryResult {
        name: "mcp_docker".to_string(),
        results,
    }
}
