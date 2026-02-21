# Plan de Mejoras para ai_assistant — v3

> Documento generado el 2026-02-21.
> Basado en completitud de v1 (15/15) y v2 (22/22) con 2510+ tests, 0 failures.
> 215+ source files, ~35k+ LoC.
>
> **Plan anterior** (v2): 22 mejoras, **TODAS completadas**.
> Ver `docs/IMPROVEMENTS.md` para historial completo.

---

## Contexto

Tras v1 (providers, embeddings, MCP, documents, guardrails) y v2 (async parity,
vector DBs, evaluation, testing), ai_assistant tiene cobertura amplia en LLM
tooling. Las **brechas restantes** frente a la industria son:

1. **Code execution sin aislamiento real** — El sandbox actual (`code_sandbox.rs`,
   688 lineas) ejecuta codigo en subprocesos del host con deteccion de patrones.
   No hay aislamiento de procesos/filesystem/red. Competidores como OpenAI Code
   Interpreter y Anthropic Computer Use usan contenedores.

2. **Sin generacion de documentos** — No hay forma de crear PDFs, DOCX, hojas
   de calculo o presentaciones. Solo hay parsing (lectura). Los usuarios necesitan
   que el agente _produzca_ documentos, no solo los lea.

3. **Sin audio/speech** — Es la unica capacidad multimodal que falta. LangChain
   tiene whisper+TTS, Semantic Kernel tiene Azure Speech, Vercel AI SDK tiene
   speech providers. Es la brecha competitiva mas visible.

4. **CI/CD incompleto** — Solo 4 jobs basicos. Falta: matrix testing de feature
   flags, cobertura de codigo, release automation, benchmarks en CI.

---

## Decisiones de Diseno (confirmadas)

| Decision | Eleccion |
|---|---|
| **Docker client** | `bollard` crate (async, tokio, well-maintained) |
| **Feature flag** | `containers` — NOT in `full` (requires Docker + heavy dep) |
| **Shared folders** | Docker bind mounts, with optional CloudStorage sync |
| **Document creation** | Container-based: pandoc/LibreOffice Docker images |
| **Audio** | `audio` feature with `SpeechProvider` trait, multiple backends |
| **CI coverage** | `cargo-llvm-cov` (more reliable than tarpaulin on complex codebases) |

---

## Fase 1 — Container Execution Engine (PRIORITY)

### 1.1 ContainerExecutor — Core Docker API Client

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/container_executor.rs` (1935 lineas, 39 tests, behind `containers` feature flag)

Implementa `ContainerExecutor` usando `bollard` crate para Docker API:
- `ContainerConfig`: docker_host, memory_limit, cpu_quota, network_mode, cleanup_policy
- `ContainerExecutor`: create, start, stop, remove, exec, logs, list, copy_to, copy_from
- `ContainerCleanupPolicy`: max_per_session (5), max_total (20), auto_remove_after_secs
- `ContainerRecord`: container_id, name, image, status, ports, bind_mounts
- Pattern: tokio runtime + block_on para sync API (igual que LanceVectorDb)
- Labels: `ai.assistant.managed=true`, `ai.assistant.session=X`

---

### 1.2 ContainerSandbox — Isolated Code Execution via Docker

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/container_sandbox.rs` (590 lineas, 18 tests)

- Drop-in replacement para `CodeSandbox` con aislamiento real
- Misma interfaz `ExecutionResult` para compatibilidad
- Images por defecto: `python:3.12-slim`, `node:20-slim`, `ubuntu:24.04`
- `ExecutionBackend` enum: `Container(ContainerSandbox)` | `Process(CodeSandbox)`
- Auto-detection: usa Container si Docker disponible, fallback a Process

---

### 1.3 SharedFolder — Bind Mount + Cloud Sync

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

Nuevo archivo: `src/shared_folder.rs` (822 lineas, 21 tests)

- Directorio local bind-mounted en containers en `/workspace`
- `sync_to_cloud(storage)` / `sync_from_cloud(storage)` via `CloudStorage` trait existente
- Carpetas temporales auto-eliminadas en Drop
- Size limit enforcement
- Content type detection automatica

---

### 1.4 Container Tool Registration

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

Nuevo archivo: `src/container_tools.rs` (1412 lineas, 29 tests)

8 tools registrados en ToolRegistry (patron de os_tools.rs):
- `container_create` (Medium risk), `container_start/stop` (Low),
  `container_remove` (Medium), `container_exec` (Medium),
  `container_logs/list` (Safe), `container_run_code` (Medium)
- Permission model: owner-agent auto-approve, sharing needs permission

---

### 1.5 Feature Flag Wiring

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

- Cargo.toml: `containers = ["dep:bollard", "dep:tokio", "dep:futures"]`
- lib.rs: module declarations + re-exports bajo `#[cfg(feature = "containers")]`
- Dependencia: `bollard = { version = "0.18", optional = true }`

---

## Fase 2 — Document Creation Pipeline (PRIORITY)

### 2.1 DocumentPipeline — Container-Based Document Generation

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/document_pipeline.rs` (1140 lineas, 20 tests)

- `OutputFormat`: PDF, DOCX, PPTX, XLSX, ODT, HTML, LaTeX, EPUB, PNG, SVG
- `DocumentRequest`: content + source_format + output_format + metadata
- `DocumentPipeline`: crea documentos en container con pandoc/LibreOffice
- Docker image: `pandoc/extra:latest` (pandoc + LaTeX)
- `sync_to_cloud()` para enviar documentos a S3/Google Drive
- Workflow: agent escribe Markdown/HTML -> pipeline convierte en container -> output en SharedFolder

---

### 2.2 Document Tool Registration

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

Integracion en container_tools.rs:
- `create_document`: markdown/html + formato -> file path
- `convert_document`: input file + target format
- `list_documents`: listar archivos en shared folder
- `sync_documents`: sincronizar shared folder a cloud storage

---

## Fase 3 — Audio/Speech

### 3.1 SpeechProvider Trait + Types

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

Nuevo archivo: `src/speech.rs` (2245 lineas, 76 tests)

- `SpeechProvider` trait: `transcribe()` (STT) + `synthesize()` (TTS)
- `AudioFormat`: WAV, MP3, OGG, FLAC, PCM, Opus, AAC
- `TranscriptionResult`: text, language, duration, segments, confidence
- `SynthesisResult`: audio bytes, format, duration, sample_rate
- `SynthesisOptions`: voice, format, speed, sample_rate

---

### 3.2 OpenAI Speech Provider

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/speech.rs`:
- STT: `POST /v1/audio/transcriptions` (Whisper)
- TTS: `POST /v1/audio/speech` (6 voices: alloy, echo, fable, onyx, nova, shimmer)
- `with_base_url()` builder para servidores locales whisper.cpp-compatible
- Multipart form upload para STT, JSON body para TTS
- Usa `ureq` (HTTP client existente)

---

### 3.3 Google Cloud Speech Provider

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio | **Estado**: HECHO

En `src/speech.rs`:
- STT: `POST speech.googleapis.com/v1/speech:recognize`
- TTS: `POST texttospeech.googleapis.com/v1/text:synthesize`
- Base64 audio encoding/decoding (implementacion inline sin deps)

---

### 3.4 Speech Tool Registration + Factory

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

- `create_speech_provider(name)` factory (6 providers: openai, google, piper, coqui, whisper, local)
- Tools: `transcribe_audio`, `synthesize_speech`
- Integracion con AiAssistant

---

### 3.5 Local Speech Providers (Whisper, Piper, Coqui)

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En `src/speech.rs` (ahora 2245 lineas, 76 tests):
- `WhisperLocalProvider`: STT offline via whisper-rs/whisper.cpp (feature `whisper-local`)
- `PiperTtsProvider`: TTS local via Piper HTTP server (localhost:5000)
- `CoquiTtsProvider`: TTS local via Coqui TTS server (localhost:5002)
- `LocalSpeechProvider`: composite STT+TTS, combina cualquier par de providers
- `OpenAISpeechProvider::with_base_url()`: builder para apuntar a servidores locales
- Factory actualizada: 6 providers (openai, google, piper, coqui, whisper, local)

---

### 3.6 Butler Speech Detection

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/butler.rs` (ahora 1808 lineas, 30 tests):
- `WhisperDetector`: detecta whisper.cpp server, WHISPER_MODEL_PATH, modelos en ~/.cache/whisper/
- `PiperDetector`: detecta Piper TTS server (PIPER_URL o localhost:5000)
- `CoquiDetector`: detecta Coqui TTS server (COQUI_URL o localhost:5002)
- `suggest_speech_config()`: recomienda mejor combo STT+TTS segun lo detectado
- 3 nuevas capabilities en EnvironmentReport: whisper_stt, piper_tts, coqui_tts

---

### 3.7 Whisper-local Feature Flag

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

- Cargo.toml: `whisper-local = ["dep:whisper-rs", "audio"]` (NOT in `full`)
- Dependencia: `whisper-rs = { version = "0.15", optional = true }`
- lib.rs: `WhisperLocalProvider` re-export bajo `#[cfg(feature = "whisper-local")]`

---

## Fase 4 — CI/CD Maturity

### 4.1 Feature Matrix Testing

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

Matrix strategy en `.github/workflows/ci.yml`:
- Test cada feature flag independiente (19 combinaciones)
- Detecta incompatibilidades entre feature flags

---

### 4.2 Code Coverage

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

- `cargo-llvm-cov` en CI
- Upload a Codecov
- Badge en README.md

---

### 4.3 Release Automation

**Prioridad**: BAJA | **Esfuerzo**: M | **Impacto**: Medio (LP) | **Estado**: HECHO

- Triggered by tag push (`v*`)
- Full test suite + release build
- GitHub Release con auto-generated release notes (softprops/action-gh-release@v2)

---

### 4.4 Benchmark CI

**Prioridad**: BAJA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

- Criterion benchmarks en CI
- Deteccion de regresiones (alert threshold 200%)

---

## Fase 5 — Polish e Integracion

### 5.1 AiAssistant Convenience Methods

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

9 metodos en AiAssistant:
- Containers: `create_container_executor()`, `create_container_executor_with_config()`, `run_code_isolated()`, `create_shared_folder()`
- Documents: `create_document_pipeline()`, `create_document()`
- Speech: `transcribe()`, `synthesize()`, `suggest_speech_providers()`

---

### 5.2 Documentation Updates

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio | **Estado**: HECHO

- GUIDE.md: 3 secciones nuevas (86-88): Container Execution, Document Creation Pipeline, Speech STT/TTS
- CONCEPTS.md: 4 secciones nuevas (54-57): Container Isolation, Shared Folders, Speech Pipeline, Whisper Local
- feature_matrix.html: v11 — 2 grupos nuevos (Containers & Documents, Audio & Speech), 3 feature flags, 10 filas

---

### 5.3 Examples

**Prioridad**: BAJA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

- `examples/speech_demo.rs` — 165 lineas (--features audio)
- `examples/container_sandbox.rs` — ~130 lineas (--features containers)
- `examples/document_creation.rs` — ~130 lineas (--features containers)

---

## Resumen de Prioridades

| # | Mejora | Fase | Prioridad | Esfuerzo | Impacto | Estado |
|---|--------|------|-----------|----------|---------|--------|
| 1.1 | ContainerExecutor core | 1 | CRITICA | L | Muy Alto | **HECHO** |
| 1.2 | ContainerSandbox (isolated exec) | 1 | CRITICA | M | Muy Alto | **HECHO** |
| 1.3 | SharedFolder abstraction | 1 | ALTA | S | Alto | **HECHO** |
| 1.4 | Container tool registration | 1 | ALTA | M | Alto | **HECHO** |
| 1.5 | Feature flag wiring | 1 | ALTA | S | Alto | **HECHO** |
| 2.1 | DocumentPipeline | 2 | ALTA | L | Muy Alto | **HECHO** |
| 2.2 | Document tool registration | 2 | ALTA | S | Alto | **HECHO** |
| 3.1 | SpeechProvider trait | 3 | MEDIA-ALTA | M | Alto | **HECHO** |
| 3.2 | OpenAI Speech provider | 3 | MEDIA-ALTA | M | Alto | **HECHO** |
| 3.3 | Google Speech provider | 3 | MEDIA | M | Medio | **HECHO** |
| 3.4 | Speech tool registration | 3 | MEDIA | S | Medio | **HECHO** |
| 3.5 | Local speech (Whisper/Piper/Coqui) | 3 | ALTA | M | Muy Alto | **HECHO** |
| 3.6 | Butler speech detection | 3 | ALTA | S | Alto | **HECHO** |
| 3.7 | Whisper-local feature flag | 3 | ALTA | S | Alto | **HECHO** |
| 4.1 | Feature matrix CI | 4 | MEDIA | M | Alto | **HECHO** |
| 4.2 | Code coverage | 4 | MEDIA | S | Medio | **HECHO** |
| 4.3 | Release automation | 4 | BAJA | M | Medio (LP) | **HECHO** |
| 4.4 | Benchmark CI | 4 | BAJA | S | Medio | **HECHO** |
| 5.1 | AiAssistant convenience methods | 5 | MEDIA | S | Medio | **HECHO** |
| 5.2 | Documentation updates | 5 | MEDIA | M | Medio | **HECHO** |
| 5.3 | Examples | 5 | BAJA | S | Medio | **HECHO** |

**Leyenda**: S = Small (1-2 dias), M = Medium (3-5 dias), L = Large (1-2 semanas), LP = largo plazo

---

## Orden de Ejecucion

```
Fase 1 (containers):
  1.1 -> 1.2 -> 1.3 -> 1.5 -> 1.4
  (executor primero, sandbox, shared folder, wiring, tools)

Fase 2 (documents):
  2.1 -> 2.2
  (pipeline usa executor + shared folder de Fase 1)

Fase 3 (audio) — PARALELO con Fases 1+2:
  3.1 -> 3.2 -> 3.3 -> 3.4
  (trait primero, OpenAI, Google, tools)

Fase 4 (CI/CD) — PARALELO con Fases 1+2+3:
  4.1 -> 4.2 -> 4.4 -> 4.3
  (feature matrix, coverage, benchmarks, release automation)

Fase 5 (polish) — DESPUES de Fases 1+2+3:
  5.1 -> 5.2 -> 5.3
  (convenience methods, docs, examples)
```

---

## Dependencias Entre Fases

```
Fase 1 (containers) <-- requerida por --> Fase 2 (documents usa containers)
Fase 3 (audio)      <-- independiente --> puede hacerse en paralelo con Fases 1+2
Fase 4 (CI/CD)      <-- independiente --> puede hacerse en paralelo
Fase 5 (polish)     <-- depende de   --> Fases 1, 2, 3
```

---

## Cambios en Cargo.toml

```toml
# New features (NOT in `full` — opt-in):
containers = ["dep:bollard", "dep:tokio", "dep:futures"]
audio = []
whisper-local = ["dep:whisper-rs", "audio"]

# New dependencies:
bollard = { version = "0.18", optional = true, default-features = false, features = ["ssl", "time"] }
whisper-rs = { version = "0.15", optional = true }
```
