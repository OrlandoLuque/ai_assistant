//! Docker integration tests for the container execution engine.
//!
//! These tests require a running Docker daemon. If Docker is not available,
//! every test gracefully skips via `skip_if_no_docker!()`.
//!
//! Run: `cargo test --features containers --test docker_integration_tests`
//! Heavy: `AI_TEST_HEAVY_IMAGES=1 cargo test --features containers --test docker_integration_tests`
//!
//! Windows: Docker Desktop must be running in Linux containers mode.

#![cfg(feature = "containers")]

use std::sync::{Arc, RwLock};
use std::time::Duration;

use ai_assistant::container_executor::{
    ContainerConfig, ContainerError, ContainerExecutor, ContainerStatus, CreateOptions,
    NetworkMode,
};
use ai_assistant::container_sandbox::{ContainerSandbox, ContainerSandboxConfig, ExecutionBackend};
use ai_assistant::code_sandbox::Language;
use ai_assistant::shared_folder::SharedFolder;

// =============================================================================
// Helpers
// =============================================================================

/// Skip test if Docker is not available.
macro_rules! skip_if_no_docker {
    () => {
        if !ContainerExecutor::is_docker_available() {
            eprintln!("SKIPPED: Docker daemon not available");
            return;
        }
    };
}

/// Skip test if Docker is not available or heavy images are not enabled.
macro_rules! skip_if_no_heavy_images {
    () => {
        skip_if_no_docker!();
        if std::env::var("AI_TEST_HEAVY_IMAGES").is_err() {
            eprintln!("SKIPPED: Set AI_TEST_HEAVY_IMAGES=1 for heavy image tests");
            return;
        }
    };
}

/// RAII cleanup guard. Owns the executor and force-removes tracked containers on drop.
struct TestExecutor {
    executor: ContainerExecutor,
    tracked_ids: Vec<String>,
}

impl TestExecutor {
    fn new() -> Result<Self, ContainerError> {
        let mut config = ContainerConfig::default();
        // Increase max_total for tests that create multiple containers.
        config.cleanup_policy.max_total = 50;
        Ok(Self {
            executor: ContainerExecutor::new(config)?,
            tracked_ids: Vec::new(),
        })
    }

    fn create(&mut self, image: &str, name: &str, opts: CreateOptions) -> Result<String, ContainerError> {
        let id = self.executor.create(image, name, opts)?;
        self.tracked_ids.push(id.clone());
        Ok(id)
    }
}

impl Drop for TestExecutor {
    fn drop(&mut self) {
        for id in &self.tracked_ids {
            let _ = self.executor.stop(id, 2);
            let _ = self.executor.remove(id, true);
        }
    }
}

/// Generate a unique container name to avoid collisions in parallel test runs.
fn unique_name(prefix: &str) -> String {
    format!(
        "dit_{}_{}",
        prefix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

// =============================================================================
// Executor tests
// =============================================================================

#[test]
fn test_docker_is_available() {
    // This test verifies the static is_docker_available check works.
    // If Docker is not running, the rest of the suite simply skips.
    let available = ContainerExecutor::is_docker_available();
    eprintln!("Docker available: {}", available);
    // No assertion — this test always passes. Its purpose is diagnostic output.
}

#[test]
fn test_docker_full_lifecycle() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("lifecycle");

    // Create with a long-lived command so the container stays running.
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");

    // Status: Created
    assert_eq!(
        te.executor.status(&id),
        Some(&ContainerStatus::Created),
        "should be Created after create"
    );

    // Start
    te.executor.start(&id).expect("start");
    assert_eq!(
        te.executor.status(&id),
        Some(&ContainerStatus::Running),
        "should be Running after start"
    );

    // Exec
    let result = te
        .executor
        .exec(&id, &["echo", "hello_world"], Duration::from_secs(10))
        .expect("exec");
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("hello_world"));
    assert!(!result.timed_out);
    assert!(result.success());

    // Stop
    te.executor.stop(&id, 5).expect("stop");
    assert_eq!(
        te.executor.status(&id),
        Some(&ContainerStatus::Stopped),
        "should be Stopped after stop"
    );

    // Remove
    te.executor.remove(&id, false).expect("remove");
    assert_eq!(
        te.executor.status(&id),
        Some(&ContainerStatus::Removed),
        "should be Removed after remove"
    );
}

#[test]
fn test_docker_exec_stdout_stderr() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("stdio");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["sh", "-c", "echo stdout_msg; echo stderr_msg >&2"],
            Duration::from_secs(10),
        )
        .expect("exec");

    assert_eq!(result.exit_code, 0);
    assert!(
        result.stdout.contains("stdout_msg"),
        "stdout should contain stdout_msg, got: {}",
        result.stdout
    );
    assert!(
        result.stderr.contains("stderr_msg"),
        "stderr should contain stderr_msg, got: {}",
        result.stderr
    );
}

#[test]
fn test_docker_exec_nonzero_exit() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("exit_code");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["sh", "-c", "exit 42"],
            Duration::from_secs(10),
        )
        .expect("exec");

    assert_eq!(result.exit_code, 42);
    assert!(!result.success());
    assert!(!result.timed_out);
}

#[test]
fn test_docker_exec_timeout() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("timeout");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["sleep", "60"],
            Duration::from_secs(2),
        )
        .expect("exec");

    assert!(result.timed_out, "should have timed out");
    assert_eq!(result.exit_code, -1);
    assert!(!result.success());
}

#[test]
fn test_docker_copy_roundtrip() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("copy");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    // Write a temp file on the host
    let tmp_dir = std::env::temp_dir().join(format!("ai_test_copy_{}", unique_name("f")));
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let src_file = tmp_dir.join("test_input.txt");
    let content = b"Hello from the host filesystem!";
    std::fs::write(&src_file, content).expect("write src");

    // Copy to container
    te.executor
        .copy_to(&id, &src_file, "/tmp/")
        .expect("copy_to");

    // Verify the file is inside the container
    let check = te
        .executor
        .exec(
            &id,
            &["cat", "/tmp/test_input.txt"],
            Duration::from_secs(5),
        )
        .expect("cat");
    assert!(
        check.stdout.contains("Hello from the host filesystem!"),
        "container file content mismatch: {}",
        check.stdout
    );

    // Copy back from container
    let dest_file = tmp_dir.join("test_output.txt");
    te.executor
        .copy_from(&id, "/tmp/test_input.txt", &dest_file)
        .expect("copy_from");

    let read_back = std::fs::read(&dest_file).expect("read dest");
    assert_eq!(read_back, content);

    // Cleanup temp files
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_docker_env_vars() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("env");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.env_vars = vec![
        ("MY_TEST_VAR".into(), "hello_docker_123".into()),
        ("ANOTHER_VAR".into(), "world".into()),
    ];
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["sh", "-c", "echo $MY_TEST_VAR"],
            Duration::from_secs(5),
        )
        .expect("exec");

    assert!(
        result.stdout.contains("hello_docker_123"),
        "env var not found in output: {}",
        result.stdout
    );
}

#[test]
fn test_docker_logs_capture() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("logs");
    let mut opts = CreateOptions::default();
    // Produce output then sleep so container stays running.
    opts.cmd = Some(vec![
        "sh".into(),
        "-c".into(),
        "echo log_line_alpha; echo log_line_beta; sleep 300".into(),
    ]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    // Wait a moment for the initial output to be produced.
    std::thread::sleep(Duration::from_secs(2));

    let logs = te.executor.logs(&id, 10).expect("logs");
    assert!(
        logs.contains("log_line_alpha"),
        "logs should contain log_line_alpha, got: {}",
        logs
    );
    assert!(
        logs.contains("log_line_beta"),
        "logs should contain log_line_beta, got: {}",
        logs
    );
}

#[test]
fn test_docker_list_and_status() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");

    let name1 = unique_name("list1");
    let mut opts1 = CreateOptions::default();
    opts1.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id1 = te.create("busybox:latest", &name1, opts1).expect("create 1");

    let name2 = unique_name("list2");
    let mut opts2 = CreateOptions::default();
    opts2.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id2 = te.create("busybox:latest", &name2, opts2).expect("create 2");

    // Start only the first container
    te.executor.start(&id1).expect("start 1");

    let records = te.executor.list();
    assert!(records.len() >= 2, "should have at least 2 containers");

    assert_eq!(te.executor.status(&id1), Some(&ContainerStatus::Running));
    assert_eq!(te.executor.status(&id2), Some(&ContainerStatus::Created));
}

#[test]
fn test_docker_cleanup_all() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");

    let name1 = unique_name("clean1");
    let mut opts1 = CreateOptions::default();
    opts1.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id1 = te.create("busybox:latest", &name1, opts1).expect("create 1");
    te.executor.start(&id1).expect("start 1");

    let name2 = unique_name("clean2");
    let mut opts2 = CreateOptions::default();
    opts2.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id2 = te.create("busybox:latest", &name2, opts2).expect("create 2");
    te.executor.start(&id2).expect("start 2");

    let removed = te.executor.cleanup_all();
    assert_eq!(removed, 2, "cleanup_all should remove both containers");

    assert_eq!(te.executor.status(&id1), Some(&ContainerStatus::Removed));
    assert_eq!(te.executor.status(&id2), Some(&ContainerStatus::Removed));
}

// =============================================================================
// Sandbox tests
// =============================================================================

#[test]
fn test_sandbox_bash_execution() {
    skip_if_no_docker!();

    let mut config = ContainerSandboxConfig::default();
    // Use busybox for bash (override default ubuntu:24.04 for faster pull).
    // busybox has /bin/sh, and Language::Bash uses "bash" as interpreter.
    // So we keep ubuntu:24.04 which actually has bash.
    config.reuse_containers = false;

    let mut sandbox = ContainerSandbox::new(config).expect("sandbox");
    let result = sandbox.execute(&Language::Bash, "echo sandbox_output_42");

    assert_eq!(result.exit_code, 0, "exit code should be 0: stderr={}", result.stderr);
    assert!(
        result.stdout.contains("sandbox_output_42"),
        "stdout should contain output: {}",
        result.stdout
    );
}

#[test]
fn test_sandbox_warm_reuse() {
    skip_if_no_docker!();

    let mut config = ContainerSandboxConfig::default();
    config.reuse_containers = true;

    let mut sandbox = ContainerSandbox::new(config).expect("sandbox");

    // First execution
    let result1 = sandbox.execute(&Language::Bash, "echo first_run");
    assert_eq!(result1.exit_code, 0, "first run failed: {}", result1.stderr);
    assert!(result1.stdout.contains("first_run"));

    // Second execution — should reuse the same container
    let result2 = sandbox.execute(&Language::Bash, "echo second_run");
    assert_eq!(result2.exit_code, 0, "second run failed: {}", result2.stderr);
    assert!(result2.stdout.contains("second_run"));
}

#[test]
fn test_sandbox_no_reuse() {
    skip_if_no_docker!();

    let mut config = ContainerSandboxConfig::default();
    config.reuse_containers = false;

    let mut sandbox = ContainerSandbox::new(config).expect("sandbox");
    let result = sandbox.execute(&Language::Bash, "echo no_reuse_test");

    assert_eq!(result.exit_code, 0, "execution failed: {}", result.stderr);
    assert!(result.stdout.contains("no_reuse_test"));
    // Container should be cleaned up after execution (no warm containers remain).
}

#[test]
fn test_execution_backend_auto() {
    skip_if_no_docker!();

    let backend = ExecutionBackend::auto();
    assert!(
        backend.is_container(),
        "auto() should select container when Docker is available"
    );
    assert_eq!(backend.backend_name(), "container");
}

// =============================================================================
// Shared folder tests
// =============================================================================

#[test]
fn test_shared_folder_host_to_container_via_mount() {
    skip_if_no_docker!();

    let folder = SharedFolder::temp().expect("temp folder");
    folder
        .put_file("test_mount.txt", b"hello from host via mount")
        .expect("put_file");

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("mount_h2c");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.bind_mounts = vec![(
        folder.host_path().to_string_lossy().into_owned(),
        folder.container_path().to_string(),
    )];
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["cat", &format!("{}/test_mount.txt", folder.container_path())],
            Duration::from_secs(5),
        )
        .expect("exec cat");

    assert!(
        result.stdout.contains("hello from host via mount"),
        "container could not read host file via bind mount: {}",
        result.stdout
    );
}

#[test]
fn test_shared_folder_container_to_host() {
    skip_if_no_docker!();

    let folder = SharedFolder::temp().expect("temp folder");

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("mount_c2h");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.bind_mounts = vec![(
        folder.host_path().to_string_lossy().into_owned(),
        folder.container_path().to_string(),
    )];
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    // Write a file from inside the container
    let write_cmd = format!(
        "echo 'written by container' > {}/container_output.txt",
        folder.container_path()
    );
    let result = te
        .executor
        .exec(
            &id,
            &["sh", "-c", &write_cmd],
            Duration::from_secs(5),
        )
        .expect("exec write");
    assert_eq!(result.exit_code, 0);

    // Read it back from the host
    let data = folder.get_file("container_output.txt").expect("get_file");
    let content = String::from_utf8_lossy(&data);
    assert!(
        content.contains("written by container"),
        "host should see container-written file: {}",
        content
    );
}

#[test]
fn test_shared_folder_subdirectory_mount() {
    skip_if_no_docker!();

    let folder = SharedFolder::temp().expect("temp folder");
    folder
        .put_file("sub/dir/nested.txt", b"nested content")
        .expect("put nested");

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("mount_sub");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.bind_mounts = vec![(
        folder.host_path().to_string_lossy().into_owned(),
        folder.container_path().to_string(),
    )];
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(
            &id,
            &["cat", &format!("{}/sub/dir/nested.txt", folder.container_path())],
            Duration::from_secs(5),
        )
        .expect("exec cat nested");

    assert!(
        result.stdout.contains("nested content"),
        "nested file not readable via mount: {}",
        result.stdout
    );
}

// =============================================================================
// Network and resource limit tests
// =============================================================================

#[test]
fn test_docker_network_none_blocks_access() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("netblock");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.network_mode = Some(NetworkMode::None);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    // Try to ping an external address — should fail with network disabled.
    let result = te
        .executor
        .exec(
            &id,
            &["ping", "-c", "1", "-W", "2", "8.8.8.8"],
            Duration::from_secs(5),
        )
        .expect("exec ping");

    assert_ne!(
        result.exit_code, 0,
        "ping should fail with NetworkMode::None, got exit_code={}",
        result.exit_code
    );
}

#[test]
fn test_docker_memory_limit_applied() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("memlimit");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    opts.memory_limit = Some(128 * 1024 * 1024); // 128 MB
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");

    // Verify the container runs successfully with the memory limit.
    // Reading cgroup files to verify the exact limit (cgroups v1 or v2).
    let result = te
        .executor
        .exec(
            &id,
            &[
                "sh", "-c",
                "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown",
            ],
            Duration::from_secs(5),
        )
        .expect("exec cgroup check");

    assert_eq!(result.exit_code, 0);
    // 128 MB = 134217728 bytes
    let output = result.stdout.trim();
    if output != "unknown" {
        let limit: u64 = output.parse().unwrap_or(0);
        assert!(
            limit > 0 && limit <= 134217728 + 4096, // allow small alignment overhead
            "memory limit should be ~128MB, got: {} (raw: {})",
            limit,
            output
        );
    }
}

// =============================================================================
// Error tests
// =============================================================================

#[test]
fn test_docker_exec_on_stopped_container() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("exec_stopped");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);
    let id = te.create("busybox:latest", &name, opts).expect("create");
    te.executor.start(&id).expect("start");
    te.executor.stop(&id, 5).expect("stop");

    // Exec on a stopped container should fail.
    let result = te
        .executor
        .exec(&id, &["echo", "should_fail"], Duration::from_secs(5));

    assert!(
        result.is_err(),
        "exec on stopped container should return Err"
    );
}

#[test]
fn test_docker_remove_nonexistent() {
    skip_if_no_docker!();

    let mut te = TestExecutor::new().expect("executor");
    let result = te
        .executor
        .remove("nonexistent_container_id_12345abcdef", false);

    assert!(result.is_err(), "removing nonexistent container should fail");
}

// =============================================================================
// Heavy tests (require AI_TEST_HEAVY_IMAGES=1)
// =============================================================================

#[test]
fn test_sandbox_python() {
    skip_if_no_heavy_images!();

    let config = ContainerSandboxConfig::default();
    let mut sandbox = ContainerSandbox::new(config).expect("sandbox");
    let result = sandbox.execute(&Language::Python, "print(2 + 2)");

    assert_eq!(result.exit_code, 0, "Python execution failed: {}", result.stderr);
    assert!(
        result.stdout.contains("4"),
        "Python output should contain 4: {}",
        result.stdout
    );
}

#[test]
fn test_document_pipeline_conversion() {
    skip_if_no_heavy_images!();

    use ai_assistant::document_pipeline::{
        DocumentPipelineConfig, DocumentPipeline, DocumentRequest, OutputFormat,
    };

    let config = DocumentPipelineConfig::default();
    let executor = ContainerExecutor::new(ContainerConfig::default()).expect("executor");
    let folder = SharedFolder::temp().expect("temp folder");
    let mut pipeline = DocumentPipeline::new(
        config,
        Arc::new(RwLock::new(executor)),
        folder,
    );

    let request = DocumentRequest::new("# Hello\n\nThis is a **test**.", OutputFormat::Html);
    let result = pipeline.create(&request);

    assert!(
        result.is_ok(),
        "document conversion failed: {:?}",
        result.err()
    );
    let doc = result.unwrap();
    assert!(doc.size_bytes > 0, "output should not be empty");
}

#[test]
fn test_docker_auto_pull_specific_tag() {
    skip_if_no_heavy_images!();

    let mut te = TestExecutor::new().expect("executor");
    let name = unique_name("pull_tag");
    let mut opts = CreateOptions::default();
    opts.cmd = Some(vec!["sleep".into(), "300".into()]);

    // Use a specific tag that may not be cached locally.
    let id = te
        .create("busybox:1.36.1", &name, opts)
        .expect("create with specific tag (should auto-pull)");
    te.executor.start(&id).expect("start");

    let result = te
        .executor
        .exec(&id, &["echo", "pulled_ok"], Duration::from_secs(10))
        .expect("exec");

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("pulled_ok"));
}
