<#
.SYNOPSIS
    Build and package all 20 binaries for a GitHub Release.

.DESCRIPTION
    Compiles every binary declared in Cargo.toml in --release mode with its
    required-features, packages them into a zip with README.txt and SHA256
    checksums.

.PARAMETER Version
    Version string for the zip file name (default: read from Cargo.toml)

.PARAMETER SkipBuild
    Skip compilation and just package existing binaries from target/release

.EXAMPLE
    .\build_release.ps1
    .\build_release.ps1 -Version "0.2.0"
    .\build_release.ps1 -SkipBuild
#>

param(
    [string]$Version = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Step { param($n, $msg) Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host " +  $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host " !  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host " -  $msg" -ForegroundColor Red }

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

$binaries = @(
    @{ Name = "ai_cli";                 Features = "full,diagnostic-logging";  Desc = "Power-user CLI (scan, query, config, diagnostics)" },
    @{ Name = "ai_assistant_cli";       Features = "full,butler";              Desc = "Interactive Chat CLI with auto-detection" },
    @{ Name = "ai_assistant_server";    Features = "full";                     Desc = "HTTP API server (OpenAI-compatible)" },
    @{ Name = "ai_assistant_standalone";Features = "full,server-axum";         Desc = "Self-contained server + embedded assistant" },
    @{ Name = "ai_cluster_node";        Features = "full,server-cluster";      Desc = "Distributed cluster node (QUIC mesh)" },
    @{ Name = "ai_proxy";               Features = "server-axum,security";     Desc = "Hardened LLM gateway (rate limit, PII, budget, audit)" },
    @{ Name = "ai_gui";                 Features = "gui";                      Desc = "Desktop GUI (WIP)" },
    @{ Name = "ai_gui-pro";             Features = "gui-pro";                  Desc = "Advanced desktop GUI (egui widgets, tabs, search)" },
    @{ Name = "ai_setup";               Features = "full";                     Desc = "Interactive setup wizard (provider + model picker)" },
    @{ Name = "ai_setup_gui";           Features = "gui";                      Desc = "Setup wizard, graphical variant" },
    @{ Name = "ai_optimize";            Features = "full";                     Desc = "ML config optimiser (auto-tune for your hardware)" },
    @{ Name = "ai_gpu_share";           Features = "full,gpu-sharing";         Desc = "GPU-sharing network client (SETI-style LLM inference)" },
    @{ Name = "ai_jobs";                Features = "scheduler";                Desc = "Cron-like job daemon (delegated / embedded modes)" },
    @{ Name = "ai_logs";                Features = "distributed-network";      Desc = "Distributed log aggregator (read + tail)" },
    @{ Name = "ai_logs_gui";            Features = "gui-logs";                 Desc = "Log viewer GUI" },
    @{ Name = "ai_virtual_mic";         Features = "audio-io";                 Desc = "Virtual microphone output (loop-back for meetings)" },
    @{ Name = "ai_virtual_mic_host";    Features = "audio";                    Desc = "Virtual mic host (multi-client audio routing)" },
    @{ Name = "ai_virtual_cam";         Features = "video-io";                 Desc = "Virtual camera output (avatar / screen share)" },
    @{ Name = "ai_test_harness";        Features = "full,browser";             Desc = "Browser-driven E2E test harness" },
    @{ Name = "kpkg_tool";              Features = "rag";                      Desc = "Knowledge package manager (create, inspect, extract)" }
)

# Read version from Cargo.toml if not specified
if (-not $Version) {
    $cargoToml = Get-Content "Cargo.toml" -Raw
    if ($cargoToml -match 'version\s*=\s*"([^"]+)"') {
        $Version = $matches[1]
    } else {
        Write-Fail "Could not read version from Cargo.toml"
        exit 1
    }
}

$releaseDir = "target\release"
$outputDir  = "target\release-package"
$zipName    = "ai_assistant-v${Version}-windows-x86_64"
$zipPath    = "$outputDir\${zipName}.zip"

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  ai_assistant Release Builder v$Version" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# --------------------------------------------------------------------------- #
# Step 1: Build each binary
# --------------------------------------------------------------------------- #

if (-not $SkipBuild) {
    $step = 0
    foreach ($bin in $binaries) {
        $step++
        Write-Step $step "Building $($bin.Name) (features: $($bin.Features))..."

        # Temporarily allow stderr (cargo writes warnings there)
        $ErrorActionPreference = "Continue"
        & cargo build --release --bin $bin.Name --features $bin.Features 2>&1 | Write-Host
        $buildExitCode = $LASTEXITCODE
        $ErrorActionPreference = "Stop"

        if ($buildExitCode -ne 0) {
            Write-Fail "Failed to build $($bin.Name)"
            exit 1
        }

        $exePath = "$releaseDir\$($bin.Name).exe"
        if (Test-Path $exePath) {
            $size = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
            Write-Success ("$($bin.Name).exe (" + $size + " MB)")
        } else {
            Write-Fail "Binary not found: $exePath"
            exit 1
        }
    }
} else {
    Write-Warn "Skipping build (--SkipBuild), using existing binaries in $releaseDir"
}

# --------------------------------------------------------------------------- #
# Step 2: Create package directory
# --------------------------------------------------------------------------- #

$nextStep = if ($SkipBuild) { 1 } else { $binaries.Count + 1 }
Write-Step $nextStep "Creating package directory..."

$packageDir = "$outputDir\$zipName"
if (Test-Path $packageDir) {
    Remove-Item -Recurse -Force $packageDir
}
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
Write-Success "Created $packageDir"

# --------------------------------------------------------------------------- #
# Step 3: Copy binaries
# --------------------------------------------------------------------------- #

$nextStep++
Write-Step $nextStep "Copying binaries..."

foreach ($bin in $binaries) {
    $src = "$releaseDir\$($bin.Name).exe"
    if (-not (Test-Path $src)) {
        Write-Fail "Missing: $src - did you forget to build?"
        exit 1
    }
    Copy-Item $src $packageDir
    Write-Success "$($bin.Name).exe"
}

# --------------------------------------------------------------------------- #
# Step 4: Generate README.txt
# --------------------------------------------------------------------------- #

$nextStep++
Write-Step $nextStep "Generating README.txt..."

$readmeTxt = @'
ai_assistant v__VERSION__ - Pre-built Windows Binaries
====================================================

Prerequisites
-------------
- Install Ollama (https://ollama.com) and pull a model:
    ollama pull llama3.2:1b
- Or have LM Studio running with a loaded model.
- Windows may show a firewall dialog - click "Allow access" for private networks.

Included Binaries
-----------------

  ai_cli.exe - Power-User CLI
    Multi-command tool for scanning providers, querying models, managing
    configuration, and running diagnostics with verbose logging.
    Usage:
      ai_cli.exe scan                         Detect local LLM providers
      ai_cli.exe query -p "Hello" -P ollama   Send a one-shot query
      ai_cli.exe models -P ollama             List available models
      ai_cli.exe config                       Show current configuration
    Diagnostics (compile-time gated):
      ai_cli.exe -v  scan                     Info-level logging
      ai_cli.exe -vv scan                     Debug-level logging
      ai_cli.exe -vvv scan                    Trace-level logging (full)
      ai_cli.exe --log-file diag.log scan     Write logs to file

  ai_assistant_cli.exe - Interactive Chat CLI
    The fastest way to start chatting. Auto-detects local LLM providers.
    Usage:
      ai_assistant_cli.exe
      ai_assistant_cli.exe --provider ollama --model llama3.2:1b
    REPL commands: /help /models /model /history /clear /save /load /exit

  ai_gui.exe - Desktop GUI (Work in Progress)
    Graphical interface with chat, model scanning, and .kpkg file support.
    Usage:
      ai_gui.exe
    Note: Functional but still under active development.

  ai_assistant_server.exe - HTTP API Server
    Runs an HTTP server with REST API endpoints. OpenAI-compatible.
    Can be used as a backend from Python, Node.js, curl, or any HTTP client.
    Usage:
      ai_assistant_server.exe
      ai_assistant_server.exe --port 8090 --api-key mysecret
    Endpoints:
      GET  /health                   Health check
      GET  /models                   List available models
      POST /chat                     Send a message (JSON body)
      POST /chat/stream              SSE streaming responses
      POST /v1/chat/completions      OpenAI-compatible endpoint
      GET  /config                   View current config
      GET  /metrics                  Prometheus metrics
      GET  /openapi.json             OpenAPI 3.0 spec
    Works as a drop-in replacement for OpenAI in tools like Continue.dev, Cursor, etc.

  kpkg_tool.exe - Knowledge Package Manager
    Create, inspect, and extract encrypted knowledge packages.
    Usage:
      kpkg_tool.exe create --input ./docs --output knowledge.kpkg --name "My Docs"
      kpkg_tool.exe list knowledge.kpkg
      kpkg_tool.exe inspect knowledge.kpkg
      kpkg_tool.exe extract --input knowledge.kpkg --output ./extracted

  ai_cluster_node.exe - Distributed Cluster Node
    Run multiple nodes that synchronize via QUIC mesh networking with CRDTs.
    For advanced users who want distributed AI deployments.
    Usage:
      ai_cluster_node.exe --node-id node1 --port 8091 --quic-port 9001
      ai_cluster_node.exe --node-id node2 --port 8092 --quic-port 9002 --bootstrap-peers 192.168.1.10:9001

Full Documentation
------------------
Getting Started guide:  https://github.com/OrlandoLuque/ai_assistant/blob/master/docs/GETTING_STARTED.md
API Reference:          https://github.com/OrlandoLuque/ai_assistant/blob/master/docs/API_REFERENCE.md
Example clients:        https://github.com/OrlandoLuque/ai_assistant/tree/master/examples/clients
crates.io:              https://crates.io/crates/ai_assistant_core

License: PolyForm Noncommercial 1.0.0
For commercial licensing, contact: orlando.luque@gmail.com
'@

$readmeTxt = $readmeTxt.Replace("__VERSION__", $Version)
$readmeTxt | Out-File "$packageDir\README.txt" -Encoding UTF8
Write-Success "README.txt"

# --------------------------------------------------------------------------- #
# Step 5: Generate SHA256 checksums
# --------------------------------------------------------------------------- #

$nextStep++
Write-Step $nextStep "Generating SHA256 checksums..."

$checksumLines = @()
foreach ($bin in $binaries) {
    $exePath = "$packageDir\$($bin.Name).exe"
    $hash = (Get-FileHash $exePath -Algorithm SHA256).Hash
    $checksumLines += "$hash  $($bin.Name).exe"
    Write-Host "   $($bin.Name).exe: $hash" -ForegroundColor DarkGray
}
$checksumLines -join "`n" | Out-File "$packageDir\SHA256SUMS.txt" -Encoding ASCII -NoNewline
Write-Success "SHA256SUMS.txt"

# --------------------------------------------------------------------------- #
# Step 6: Create zip
# --------------------------------------------------------------------------- #

$nextStep++
Write-Step $nextStep "Creating zip archive..."

if (Test-Path $zipPath) {
    Remove-Item $zipPath
}

Compress-Archive -Path "$packageDir\*" -DestinationPath $zipPath -CompressionLevel Optimal
$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Success ("${zipName}.zip (" + $zipSize + " MB)")

# Also generate a checksum for the zip itself
$zipHash = (Get-FileHash $zipPath -Algorithm SHA256).Hash
$zipHash | Out-File "$outputDir\${zipName}.zip.sha256" -Encoding ASCII -NoNewline
Write-Success "Zip SHA256: $zipHash"

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Release Package Summary" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

foreach ($bin in $binaries) {
    $exePath = "$packageDir\$($bin.Name).exe"
    $size = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
    Write-Host ("  " + $bin.Name + ".exe  (" + $size + " MB)  - " + $bin.Desc) -ForegroundColor White
}

Write-Host ""
Write-Host "  Zip:      $zipPath" -ForegroundColor White
Write-Host "  Size:     $zipSize MB" -ForegroundColor White
Write-Host "  Checksum: $outputDir\${zipName}.zip.sha256" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Upload ${zipName}.zip to GitHub Releases" -ForegroundColor DarkGray
Write-Host "  2. Paste the SHA256 hash in the release notes" -ForegroundColor DarkGray
Write-Host ""
Write-Success "Release package ready!"
