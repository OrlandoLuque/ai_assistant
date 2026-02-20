# Diseno del Sistema Agentico Real

> Documento de diseno para convertir `ai_assistant` de un framework de patrones
> en un agente de ejecucion real capaz de tocar codigo, operar en disco,
> controlar navegadores y ejecutar comandos — con un sistema de permisos
> inteligente que evalua peligrosidad y sugiere alternativas seguras.
>
> Fecha: 2026-02-15

---

## Indice

1. [Estado Actual vs Objetivo](#1-estado-actual-vs-objetivo)
2. [Loop Agentico Real](#2-loop-agentico-real)
3. [Sistema de Tools del OS](#3-sistema-de-tools-del-os)
4. [Sistema de Permisos y Aprobacion](#4-sistema-de-permisos-y-aprobacion)
5. [Evaluacion de Peligrosidad](#5-evaluacion-de-peligrosidad)
6. [Generacion de Alternativas Seguras](#6-generacion-de-alternativas-seguras)
7. [Sandbox y Aislamiento](#7-sandbox-y-aislamiento)
8. [Browser Automation](#8-browser-automation)
9. [MCP (Model Context Protocol)](#9-mcp-model-context-protocol)
10. [Proyectos Open Source Analizados](#10-proyectos-open-source-analizados)
11. [Ideas Extraidas de Cada Proyecto](#11-ideas-extraidas-de-cada-proyecto)
12. [Plan de Implementacion](#12-plan-de-implementacion)
13. [Feedback Real: Que Tan Bien Funcionan](#13-feedback-real-que-tan-bien-funcionan)
14. [Estrategias de Iteracion (reemplazo de max_iterations)](#14-estrategias-de-iteracion)
15. [Politicas de Privacidad por Proveedor](#15-politicas-de-privacidad-por-proveedor)
16. [Sistema de Undo/Rollback](#16-sistema-de-undorollback)
17. [Herramientas de Red Avanzadas](#17-herramientas-de-red-avanzadas)
18. [Sistema de Plugins Extensible (VCS y otros)](#18-sistema-de-plugins-extensible)
19. [Introspeccion del Sistema (Diagnostics Tool)](#19-introspeccion-del-sistema)
20. [Gestion de Contenedores y Prerequisitos](#20-gestion-de-contenedores-y-prerequisitos)
21. [Deteccion de Trabajo Incompleto y Registro de Requisitos](#21-deteccion-de-trabajo-incompleto)
22. [Catalogo de Interfaces a Servicios Externos](#22-catalogo-de-interfaces-a-servicios-externos)
23. [Funcionalidades No Incluidas en el Plan Actual](#23-funcionalidades-no-incluidas)
24. [Estrategia Hibrida Local/Cloud](#24-estrategia-hibrida-localcloud)
25. [Assembly-Line con Back-Communication](#25-assembly-line-con-back-communication)
26. [Ventajas Competitivas de ai_assistant](#26-ventajas-competitivas-de-ai_assistant)
27. [Multi-Task Planning](#27-multi-task-planning)
28. [Modos de Operacion](#28-modos-de-operacion)
29. [Butler: Auto-Configuracion Inteligente](#29-butler-auto-configuracion-inteligente)
30. [Deteccion de GPU y Recomendacion de Modelos](#30-deteccion-de-gpu-y-recomendacion-de-modelos)
31. [UX de Conversacion: Navegacion y Widgets](#31-ux-de-conversacion)
32. [Proteccion de Contexto Contra Compresion](#32-proteccion-de-contexto-contra-compresion)
33. [Seleccion Automatica de Agente/Modo](#33-seleccion-automatica-de-agentemodo)
34. [Import/Export Universal de Configuracion](#34-importexport-universal-de-configuracion)
35. [Escalabilidad de Busqueda Vectorial](#35-escalabilidad-de-busqueda-vectorial)
36. [Sistema Distribuido Rediseñado](#36-sistema-distribuido-rediseñado--alta-disponibilidad-y-tolerancia-a-fallos)
37. [Sistema Autonomo Completo — Implementacion](#37-sistema-autonomo-completo--implementacion)
38. [Infraestructura de Testing](#38-infraestructura-de-testing)
39. [Phase 9 — REPL/CLI, Neural Reranking, A/B Testing](#39-phase-9--replcli-neural-reranking-ab-testing)
40. [Phase 10 — Token Counting, Cost Tracking, Multi-Modal RAG](#40-phase-10--token-counting-cost-tracking-multi-modal-rag)

---

## 1. Estado Actual vs Objetivo

### Lo que tenemos

```
Usuario -> AgenticLoop -> [NO llama al LLM] -> respuesta simulada (stub)
```

- `agentic_loop.rs`: linea 268 pone `status = Finished` sin llamar al LLM
- `ReactAgent`: patron ReAct implementado pero sin integracion LLM real
- `unified_tools.rs`: excelente sistema de validacion y parsing multi-formato, pero solo tools in-memory (calculadora, datetime, string_length, validate_json)
- `ProviderPlugin.generate_with_tools()`: ignora los tools (`let _ = tools;`)
- `multi_agent.rs`: orquestador de estado, no ejecuta nada
- Ningun tool accede al sistema operativo

### Lo que necesitamos

```
Usuario -> LLM (con tools disponibles en el request)
   -> LLM responde con tool_calls
   -> Validar permisos (evaluar peligrosidad, pedir aprobacion si necesario)
   -> Ejecutar tools (filesystem, shell, browser, red)
   -> Resultado como mensaje Tool en la conversacion
   -> Volver a llamar al LLM
   -> Repetir hasta respuesta final
```

---

## 2. Loop Agentico Real

### Flujo Principal

```
                    +-----------+
                    | User Query|
                    +-----+-----+
                          |
                    +-----v-----+
              +---->| Call LLM   |<---------------------------+
              |     | (with tools|                            |
              |     | definitions|                            |
              |     +-----+-----+                            |
              |           |                                   |
              |     +-----v---------+                        |
              |     | Parse Response |                        |
              |     +-----+---------+                        |
              |           |                                   |
              |     +-----v----------+     +-----------+     |
              |     | Has tool_calls?|---->| No: return |     |
              |     +-----+----------+     | final text |     |
              |           | Yes            +-----------+     |
              |     +-----v-----------+                      |
              |     | For each call:  |                      |
              |     |                 |                      |
              |     | 1. Validate     |                      |
              |     | 2. Check perms  |                      |
              |     | 3. Execute      |                      |
              |     | 4. Collect result|                     |
              |     +-----+-----------+                      |
              |           |                                   |
              |     +-----v-----------+                      |
              |     | Add tool results|                      |
              |     | to conversation |----------------------+
              |     +-----------------+
              |
              +--- Max iterations safety net
```

### Componentes del Loop

1. **ToolAwareProvider**: nuevo trait que extiende la generacion para incluir tools
   - Envia `tools: [...]` en el request al LLM (OpenAI function_call, Anthropic tool_use, Ollama tools)
   - Parsea `tool_calls` del response usando `unified_tools::parse_tool_calls()`
   - Maneja `tool_choice`: auto, required, none

2. **AgentExecutor**: el motor principal
   - Mantiene la conversacion (mensajes user/assistant/tool)
   - Controla iteraciones (max_iterations como safety net)
   - Coordina: llamar LLM -> parsear -> permisos -> ejecutar -> feedback
   - Emite eventos via `EventBus` en cada paso
   - Soporta cancelacion via `CancellationToken`

3. **ToolResultMessage**: formato estandar para resultados
   - `tool_call_id: String` (correlacionar con la llamada original)
   - `name: String` (nombre del tool)
   - `content: String` (resultado o error)
   - `is_error: bool`

### Condiciones de Parada

- El LLM responde sin tool_calls (respuesta final)
- Se alcanza `max_iterations`
- El usuario cancela
- Un tool critico falla y no hay recuperacion
- El LLM emite un marcador explicito de "tarea completada"

---

## 3. Sistema de Tools del OS

### 3.1 Tools de Filesystem

| Tool | Parametros | Descripcion | Riesgo Base |
|------|-----------|-------------|-------------|
| `read_file` | `path: String` | Leer contenido de un archivo | BAJO |
| `write_file` | `path: String, content: String` | Crear/sobrescribir archivo | MEDIO |
| `edit_file` | `path: String, old_text: String, new_text: String` | Buscar y reemplazar en archivo | MEDIO |
| `list_directory` | `path: String, glob: Option<String>` | Listar archivos con patron opcional | BAJO |
| `search_files` | `pattern: String, path: String, file_glob: Option<String>` | Grep recursivo (contenido) | BAJO |
| `create_directory` | `path: String` | Crear directorio (recursivo) | BAJO |
| `delete_file` | `path: String` | Eliminar archivo | ALTO |
| `delete_directory` | `path: String, recursive: bool` | Eliminar directorio | CRITICO |
| `move_file` | `src: String, dst: String` | Mover/renombrar archivo | MEDIO |
| `copy_file` | `src: String, dst: String` | Copiar archivo | BAJO |
| `file_info` | `path: String` | Metadata (tamano, permisos, timestamps) | BAJO |

### 3.2 Tools de Ejecucion de Comandos

| Tool | Parametros | Descripcion | Riesgo Base |
|------|-----------|-------------|-------------|
| `run_command` | `command: String, args: Vec<String>, cwd: Option<String>, timeout_secs: u64` | Ejecutar proceso hijo | ALTO |
| `run_shell` | `script: String, shell: Option<String>, cwd: Option<String>, timeout_secs: u64` | Ejecutar script en shell | CRITICO |

### 3.3 Tools de Red

| Tool | Parametros | Descripcion | Riesgo Base |
|------|-----------|-------------|-------------|
| `http_get` | `url: String, headers: HashMap` | GET request | MEDIO |
| `http_post` | `url: String, body: String, headers: HashMap` | POST request | MEDIO |
| `download_file` | `url: String, dest_path: String` | Descargar archivo | MEDIO |

### 3.4 Tools de Codigo

| Tool | Parametros | Descripcion | Riesgo Base |
|------|-----------|-------------|-------------|
| `git_status` | `repo_path: String` | Estado del repo git | BAJO |
| `git_diff` | `repo_path: String, staged: bool` | Diff de cambios | BAJO |
| `git_commit` | `repo_path: String, message: String, files: Vec<String>` | Commit | MEDIO |
| `git_log` | `repo_path: String, count: usize` | Historial | BAJO |

### 3.5 Tools de Browser (ver seccion 8)

---

## 4. Sistema de Permisos y Aprobacion

### 4.1 Modelo de Permisos por Capas

```
+--------------------------------------------------+
| Capa 1: POLITICA GLOBAL (config file)            |
|  - Directorios permitidos / bloqueados            |
|  - Comandos permitidos / bloqueados               |
|  - Hosts de red permitidos / bloqueados           |
|  - Nivel de autonomia (paranoid/normal/trust)     |
+--------------------------------------------------+
          |
+--------------------------------------------------+
| Capa 2: EVALUACION DE PELIGROSIDAD (automatica)  |
|  - Analisis estatico del tool call                |
|  - Clasificacion: SAFE / REVIEW / DANGEROUS       |
|  - Score numerico 0-100                           |
+--------------------------------------------------+
          |
+--------------------------------------------------+
| Capa 3: DECISION DE APROBACION                    |
|  - SAFE + dentro de politica -> auto-aprobar      |
|  - REVIEW -> pedir confirmacion al usuario        |
|  - DANGEROUS -> bloquear + sugerir alternativas   |
|  - DANGEROUS + usuario insiste -> doble confirm.  |
+--------------------------------------------------+
          |
+--------------------------------------------------+
| Capa 4: AUDIT LOG (siempre)                       |
|  - Registrar TODA accion (aprobada o denegada)    |
|  - Timestamp, tool, args, resultado, quien aprobo |
+--------------------------------------------------+
```

### 4.2 Configuracion de Politica

```rust
/// Politica de permisos para el agente
struct AgentPolicy {
    /// Nivel de autonomia global
    autonomy_level: AutonomyLevel, // Paranoid, Normal, Autonomous

    /// Directorios donde el agente puede leer
    allowed_read_dirs: Vec<PathBuf>,
    /// Directorios donde el agente puede escribir
    allowed_write_dirs: Vec<PathBuf>,
    /// Directorios bloqueados (siempre denegados)
    blocked_dirs: Vec<PathBuf>,       // e.g. ~/.ssh, ~/.gnupg, /etc

    /// Comandos permitidos sin confirmacion
    allowed_commands: Vec<String>,    // e.g. ["cargo", "npm", "git", "ls"]
    /// Comandos bloqueados (siempre denegados)
    blocked_commands: Vec<String>,    // e.g. ["rm -rf", "sudo", "chmod 777"]
    /// Patron de comandos peligrosos (regex)
    dangerous_command_patterns: Vec<String>,

    /// Hosts de red permitidos
    allowed_hosts: Vec<String>,       // e.g. ["api.openai.com", "localhost"]
    /// Hosts bloqueados
    blocked_hosts: Vec<String>,

    /// Extensiones de archivo que nunca se tocan
    protected_extensions: Vec<String>, // e.g. [".env", ".pem", ".key", ".gpg"]

    /// Tamano maximo de archivo que se puede escribir (bytes)
    max_write_size: usize,

    /// Timeout maximo para comandos (segundos)
    max_command_timeout: u64,

    /// Reglas personalizadas
    custom_rules: Vec<CustomRule>,
}

enum AutonomyLevel {
    /// Pedir confirmacion para TODO excepto lectura
    Paranoid,
    /// Auto-aprobar operaciones seguras, pedir confirmacion para el resto
    Normal,
    /// Auto-aprobar todo lo que la politica permite, solo bloquear lo prohibido
    Autonomous,
}
```

### 4.3 Flujo de Aprobacion del Usuario

Cuando una accion requiere confirmacion:

```
+----------------------------------------------------------+
| PERMISO REQUERIDO                                        |
|                                                          |
| El agente quiere ejecutar:                               |
|   Tool: run_command                                      |
|   Comando: cargo test --features full                    |
|   Directorio: /home/user/project                         |
|                                                          |
| Peligrosidad: BAJA (score: 15/100)                       |
| Razon: comando conocido, directorio permitido            |
|                                                          |
| [1] Aprobar esta vez                                     |
| [2] Aprobar siempre para este comando                    |
| [3] Aprobar siempre en este directorio                   |
| [4] Denegar                                              |
| [5] Ver alternativas mas seguras                         |
+----------------------------------------------------------+
```

Para acciones peligrosas:

```
+----------------------------------------------------------+
| !! ACCION PELIGROSA !!                                   |
|                                                          |
| El agente quiere ejecutar:                               |
|   Tool: delete_directory                                 |
|   Path: /home/user/project/node_modules                  |
|   Recursivo: true                                        |
|                                                          |
| Peligrosidad: ALTA (score: 78/100)                       |
| Razones:                                                 |
|   - Eliminacion recursiva de directorio                  |
|   - 15,234 archivos afectados                            |
|   - 890 MB de datos                                      |
|                                                          |
| ALTERNATIVAS MAS SEGURAS:                                |
|   [A] Mover a papelera en vez de borrar                  |
|   [B] Crear backup antes de borrar                       |
|   [C] Listar contenido primero (dry-run)                 |
|                                                          |
| [1] Aprobar (PELIGROSO — requiere escribir "CONFIRMAR")  |
| [2] Usar alternativa A (mover a papelera)                |
| [3] Usar alternativa B (backup + borrar)                 |
| [4] Usar alternativa C (solo listar)                     |
| [5] Denegar                                              |
+----------------------------------------------------------+
```

### 4.4 Escalacion de Permisos ("Aceptar de ahora en adelante")

Cuando el usuario aprueba una accion, se le ofrecen opciones de escalacion granular:

```
OPCIONES DE APROBACION:

[1] Aprobar solo esta vez
[2] Aprobar este tool en este directorio (y subdirectorios) para esta sesion
    -> "Aceptar read_file en /proyecto/ de ahora en adelante"
[3] Aprobar esta categoria de tool para esta sesion
    -> "Aceptar todos los tools de lectura de ahora en adelante"
[4] Anadir path al whitelist permanente
    -> "Agregar /proyecto/docs/ a directorios permitidos"
    -> (modifica AgentPolicy.allowed_read_dirs o allowed_write_dirs)
[5] Acceso unico a path no whitelistado
    -> "Permitir leer /tmp/data.csv solo para esta operacion"
    -> (no modifica la politica, permiso efimero)
[6] Denegar
[7] Denegar y bloquear este path para esta sesion
```

#### Reglas de escalacion

```rust
struct SessionPermission {
    /// Tool o categoria de tool permitida
    tool_pattern: ToolPattern,  // Exact("read_file"), Category("filesystem_read"), All
    /// Path permitido (si aplica)
    path_scope: PathScope,      // Exact(path), Subtree(path), Any
    /// Duracion
    duration: PermissionDuration, // ThisOperation, ThisSession, Persistent
    /// Quien lo aprobo
    granted_by: String,         // "user", "policy"
    /// Timestamp
    granted_at: u64,
}

enum ToolPattern {
    Exact(String),              // "read_file"
    Category(String),           // "filesystem_read", "filesystem_write", "git", "shell"
    All,                        // todo (peligroso, solo en Autonomous mode)
}

enum PathScope {
    Exact(PathBuf),             // solo ese archivo/directorio
    Subtree(PathBuf),           // ese directorio y todo lo que contiene
    Any,                        // cualquier path (peligroso)
}

enum PermissionDuration {
    ThisOperation,              // solo para este tool call
    ThisSession,                // hasta que termine la sesion del agente
    Persistent,                 // se guarda en AgentPolicy (requiere confirmacion extra)
}
```

#### Categorias de tools para escalacion masiva

| Categoria | Tools incluidos | Riesgo de "aceptar siempre" |
|-----------|----------------|---------------------------|
| `filesystem_read` | read_file, list_directory, search_files, file_info | BAJO |
| `filesystem_write` | write_file, edit_file, create_directory, copy_file | MEDIO |
| `filesystem_delete` | delete_file, delete_directory, move_file | ALTO |
| `git_read` | git_status, git_diff, git_log | BAJO |
| `git_write` | git_commit | MEDIO |
| `shell` | run_command, run_shell | ALTO |
| `network` | http_get, http_post, download_file | MEDIO |
| `browser` | browser_navigate, browser_click, etc. | MEDIO |

### 4.5 Reglas de Sesion

- Las aprobaciones "siempre en esta sesion" solo duran la sesion actual del agente
- Las aprobaciones "persistentes" se guardan en AgentPolicy (archivo de config) y requieren confirmacion extra
- NUNCA se auto-escalan permisos entre sesiones sin aprobacion explicita
- El usuario puede revocar permisos en cualquier momento
- Existe un "boton de panico" que detiene toda ejecucion inmediatamente
- Al inicio de cada sesion, se muestra un resumen de los permisos persistentes activos

### 4.6 Trust Mode — Sigue Evaluando

**Incluso en trust mode (AutonomyLevel::Autonomous):**
- Se SIGUE calculando el score de peligrosidad de cada operacion
- Se SIGUE registrando todo en el audit log
- Se SIGUEN detectando patrones peligrosos (rm -rf /, fork bombs, etc.)
- Lo unico que cambia es que NO se pide confirmacion al usuario para acciones dentro de la politica
- Las acciones que coinciden con `blocked_commands` o `blocked_dirs` se SIGUEN bloqueando SIEMPRE
- Si se detecta un patron critico (score > 90), se pide confirmacion INCLUSO en trust mode
- Razon: trust mode significa "confio en el agente para decisiones normales", no "el agente puede destruir mi sistema"

---

## 5. Evaluacion de Peligrosidad

### 5.1 Clasificacion de Riesgo

```rust
enum RiskLevel {
    Safe,       // 0-20:  auto-aprobable segun politica
    Low,        // 21-40: auto-aprobable en modo Normal/Autonomous
    Medium,     // 41-60: requiere confirmacion en Paranoid/Normal
    High,       // 61-80: requiere confirmacion siempre
    Critical,   // 81-100: requiere doble confirmacion + alternativas
}

struct RiskAssessment {
    /// Score numerico 0-100
    score: u32,
    /// Nivel de riesgo
    level: RiskLevel,
    /// Razones del score
    reasons: Vec<String>,
    /// Alternativas sugeridas (si aplica)
    alternatives: Vec<SaferAlternative>,
    /// Es reversible la accion?
    reversible: bool,
    /// Archivos/recursos afectados
    affected_resources: Vec<String>,
    /// Estimacion de impacto
    impact_description: String,
}
```

### 5.2 Factores de Peligrosidad

Cada factor suma o resta puntos al score base del tool:

#### Factores del Tool

| Factor | Puntos | Ejemplo |
|--------|--------|---------|
| Score base del tool | +0 a +50 | `read_file` = 5, `run_shell` = 50 |
| Operacion destructiva (no reversible) | +20 | delete, overwrite |
| Operacion reversible | -10 | git commit (se puede revertir) |
| Escritura en disco | +10 | write_file, edit_file |
| Ejecucion de codigo | +25 | run_command, run_shell |
| Acceso a red | +15 | http_get, download_file |
| Operacion de solo lectura | -15 | read_file, list_directory |

#### Factores del Contexto

| Factor | Puntos | Ejemplo |
|--------|--------|---------|
| Path fuera de directorio del proyecto | +20 | `/etc/`, `~/.ssh/` |
| Path contiene datos sensibles | +30 | `.env`, `.pem`, `credentials` |
| Comando con sudo/runas | +40 | `sudo rm` |
| Comando con pipe/redireccion | +10 | `cat x | sh`, `> /dev/sda` |
| Comando desconocido (no en whitelist) | +15 | binario custom |
| Muchos archivos afectados (>100) | +10 | `rm -rf node_modules` |
| Archivo muy grande (>10MB) | +5 | escritura de archivo grande |
| URL no en whitelist | +10 | host desconocido |
| URL localhost/127.0.0.1 | -5 | es local, bajo riesgo |
| Operacion comun en dev (build/test/lint) | -15 | `cargo test`, `npm run build` |
| Patrones peligrosos en shell | +30 | `rm -rf /`, `:(){ :|:& };:`, `dd if=` |

### 5.3 Analisis Estatico de Comandos

Antes de ejecutar `run_command` o `run_shell`, analizar el comando:

```rust
struct CommandAnalysis {
    /// Binario principal
    binary: String,
    /// Es un shell builtin?
    is_shell_builtin: bool,
    /// Tiene pipes?
    has_pipes: bool,
    /// Tiene redireccion de salida?
    has_output_redirect: bool,
    /// Tiene redireccion destructiva (>)?
    has_destructive_redirect: bool,
    /// Usa sudo/doas/runas?
    uses_elevation: bool,
    /// Patrones peligrosos detectados
    dangerous_patterns: Vec<String>,
    /// Archivos que tocaria (si se puede inferir)
    inferred_files: Vec<PathBuf>,
    /// Se puede ejecutar en dry-run?
    supports_dry_run: bool,
    /// Flag de dry-run (si existe)
    dry_run_flag: Option<String>,
}
```

Patrones peligrosos a detectar:

```
rm -rf /          # borrado raiz
rm -rf ~          # borrado home
rm -rf *          # borrado glob
chmod 777         # permisos abiertos
chmod -R          # permisos recursivos
> /dev/sd*        # escritura en dispositivo
dd if=            # escritura directa en disco
mkfs              # formateo
:(){ :|:& };:    # fork bomb
wget|curl|sh     # download and execute
eval $(...)       # ejecucion dinamica
base64 -d | sh   # ofuscacion + ejecucion
nc -e             # reverse shell
/dev/tcp/         # conexion raw
```

### 5.4 Evaluacion Contextual

El evaluador de peligrosidad no solo mira el tool call aislado, sino el contexto:

- **Historial de la sesion**: si el agente ya leyo un archivo `.env`, y ahora quiere hacer `http_post`, es sospechoso (posible exfiltracion)
- **Patron de comportamiento**: muchas escrituras seguidas en archivos distintos puede indicar un problema
- **Coherencia con la tarea**: si el usuario pidio "arregla el bug en main.rs" y el agente quiere borrar `database.db`, algo va mal
- **Acumulacion de riesgo**: cada accion aprobada incrementa un "risk budget" de sesion; si se acumula mucho, empezar a pedir confirmacion incluso para acciones normalmente auto-aprobadas

---

## 6. Generacion de Alternativas Seguras

### 6.1 Tabla de Alternativas por Accion

| Accion Peligrosa | Alternativas Seguras |
|-------------------|---------------------|
| `delete_file(x)` | Mover a `.trash/` temporal; Crear backup primero; Listar archivo antes |
| `delete_directory(x, recursive=true)` | Mover a `.trash/`; Listar contenido primero (dry-run); Backup zip antes |
| `write_file(x)` si ya existe | Crear backup del original (`.bak`); Mostrar diff antes de escribir; Escribir en archivo temporal primero |
| `run_shell("rm -rf ...")` | Convertir a `delete_file` individual (mas controlado); Usar `--dry-run` si el comando lo soporta; Listar lo que se borraria primero |
| `run_command` desconocido | Ejecutar con `--help` primero para ver que hace; Ejecutar en directorio temporal aislado; Mostrar man page o documentacion |
| `edit_file` cambio grande | Mostrar preview del diff antes de aplicar; Hacer backup `.bak` antes; Aplicar cambio por partes (parcial) |
| `http_post` a host desconocido | Mostrar el body que se enviaria; Hacer primero un `http_get` para verificar el host; Usar mock/dry-run |
| `git_commit` | Mostrar `git diff --staged` antes; Confirmar mensaje de commit |
| Escribir archivo >1MB | Preguntar si esta seguro; Escribir en chunks verificando |
| Ejecutar script desconocido | Mostrar contenido del script primero; Ejecutar en sandbox |

### 6.2 Motor de Sugerencias

```rust
struct SaferAlternative {
    /// Descripcion legible para el usuario
    description: String,
    /// Tool calls alternativos que reemplazan la accion original
    replacement_calls: Vec<ToolCall>,
    /// Reduccion estimada del riesgo (0-100)
    risk_reduction: u32,
    /// Se pierde funcionalidad respecto al original?
    functionality_loss: String, // e.g. "Ninguna", "No se borra, solo se mueve"
}
```

El motor de sugerencias funciona asi:

1. Recibe un `ToolCall` con score de riesgo > umbral
2. Busca en la tabla de alternativas por tipo de tool
3. Genera `SaferAlternative` concretas con los parametros reales del call
4. Ordena por `risk_reduction` descendente
5. Presenta al usuario junto con la accion original

### 6.3 Dry-Run Automatico

Para ciertos tools, el sistema puede ejecutar automaticamente un "dry-run" antes de pedir confirmacion:

- `delete_file/directory` -> listar archivos afectados y tamano total
- `edit_file` -> calcular y mostrar diff
- `run_command` si tiene `--dry-run` -> ejecutar con esa flag
- `write_file` sobre existente -> mostrar diff entre original y nuevo

Esto da al usuario informacion REAL antes de decidir, no solo una descripcion.

### 6.4 Dry-Run Extendido: Verificar Ficheros No Afectados

Ademas de listar lo que SI se va a tocar, verificar lo que NO se va a tocar:

- **Antes de `delete_directory(path, recursive=true)`**: hacer `list_directory(path, recursive=true)` y comparar con lo que el agente cree que va a borrar. Si hay archivos que el agente no menciono, alertar: "Hay 3 archivos mas que no mencionaste. Quieres revisarlos?"
- **Antes de `edit_file` con patron glob**: verificar que no haya ficheros que coincidan con el patron pero no esten en la lista del agente
- **Antes de `run_command("rm ...")` con wildcards**: expandir el glob y mostrar TODOS los ficheros que coinciden, no solo los que el agente espera
- **Antes de operaciones batch**: si el agente dice "voy a editar 5 archivos", verificar si hay mas archivos que podrian necesitar el mismo cambio

```rust
struct DryRunResult {
    /// Archivos que SI seran afectados
    affected_files: Vec<PathBuf>,
    /// Archivos que podrian ser afectados pero NO estan en el plan del agente
    potentially_missed: Vec<PathBuf>,
    /// Tamano total que se va a tocar/borrar
    total_size_bytes: u64,
    /// Diff preview (si aplica)
    diff_preview: Option<String>,
    /// Warning si hay discrepancias
    warnings: Vec<String>,
}
```

---

## 7. Sandbox y Aislamiento

### 7.1 Niveles de Sandbox

```
Nivel 0: Sin sandbox (trust mode)
  - Todo se ejecuta directamente
  - Solo para desarrollo local de confianza

Nivel 1: Restriccion de paths (default)
  - Filesystem: solo leer/escribir dentro de allowed_dirs
  - Red: sin restricciones
  - Procesos: sin restricciones (pero con timeout)

Nivel 2: Sandbox de proceso (recomendado)
  - Filesystem: chroot-like o bubblewrap (Linux) / seatbelt (macOS)
  - Red: solo hosts permitidos (proxy)
  - Procesos: timeout + limites de memoria

Nivel 3: Contenedor (maximo aislamiento)
  - Docker/Podman con volumen montado
  - Network namespace propio
  - Ideal para run_shell con scripts desconocidos
```

### 7.2 Implementacion en Rust

Para el Nivel 1 (restriccion de paths), se puede implementar enteramente en Rust:

```rust
/// Verifica que un path esta dentro de los directorios permitidos
fn validate_path(path: &Path, policy: &AgentPolicy, write: bool) -> Result<(), PermissionError> {
    let canonical = path.canonicalize()?;

    // Verificar blocked dirs primero (deny wins)
    for blocked in &policy.blocked_dirs {
        if canonical.starts_with(blocked) {
            return Err(PermissionError::Blocked(path));
        }
    }

    // Verificar allowed dirs
    let allowed = if write { &policy.allowed_write_dirs } else { &policy.allowed_read_dirs };
    if !allowed.iter().any(|dir| canonical.starts_with(dir)) {
        return Err(PermissionError::OutsideAllowed(path));
    }

    // Verificar extensiones protegidas
    if write {
        if let Some(ext) = path.extension() {
            if policy.protected_extensions.contains(&format!(".{}", ext.to_string_lossy())) {
                return Err(PermissionError::ProtectedExtension(path));
            }
        }
    }

    Ok(())
}
```

Para el Nivel 2 en Linux, usar `bubblewrap` o el crate `sandbox` (similar a lo que hace Claude Code con `sandbox-runtime`).

Para el Nivel 3, usar `bollard` (crate Rust para Docker API) para lanzar contenedores efimeros.

### 7.3 Aislamiento de Red

Opciones:
- **Proxy transparente**: todo el trafico HTTP del agente pasa por un proxy local que filtra por host
- **Firewall de proceso**: en Linux, usar `seccomp-bpf` para restringir syscalls de red
- **DNS filtering**: resolver solo hosts permitidos, devolver NXDOMAIN para el resto

---

## 8. Browser Automation

### 8.1 Opciones de Implementacion

| Crate/Herramienta | Protocolo | Ventajas | Desventajas |
|-------------------|-----------|----------|-------------|
| `chromiumoxide` | CDP (Chrome DevTools Protocol) | Nativo Rust, async, rapido | Solo Chromium |
| `fantoccini` | WebDriver (Selenium) | Multi-browser | Requiere driver externo |
| `headless_chrome` | CDP | API simple | Menos mantenido |
| Playwright MCP | MCP protocol | Estandar, multi-browser | Requiere Node.js |
| browser-use (Python) | Playwright | Muy completo, AI-native | Python, no Rust |

**Recomendacion**: `chromiumoxide` para integracion nativa Rust, con fallback a Playwright MCP para funcionalidad avanzada.

### 8.1b Instalacion y Gestion del Browser Headless

El browser headless NO viene incluido en el binario. Requiere instalacion separada:

**Opciones de browser**:

| Browser | Tamano | Instalacion | Notas |
|---------|--------|-------------|-------|
| Chromium (via Playwright) | ~350 MB | `npx playwright install chromium` | Requiere Node.js |
| Chromium (descarga directa) | ~200 MB | Download + extract | Sin dependencia de Node |
| Chrome del sistema | 0 MB extra | Detectar path existente | Ya instalado por el usuario |
| Firefox del sistema | 0 MB extra | Detectar path existente | Limitaciones con CDP |

**Tools de gestion de browser**:

| Tool | Descripcion | Riesgo | Permisos |
|------|-------------|--------|----------|
| `browser_install` | Instalar browser headless (elige tipo) | ALTO | Pedir permiso (descarga ~200-350MB) |
| `browser_uninstall` | Desinstalar browser instalado por el agente | MEDIO | Auto si instalado por este sistema |
| `browser_status` | Ver si hay browser disponible, version, tamano en disco | BAJO | Auto |
| `browser_detect_system` | Buscar Chrome/Firefox/Edge ya instalados en el sistema | BAJO | Auto |
| `browser_update` | Actualizar browser instalado | MEDIO | Pedir permiso |

**Flujo cuando el agente necesita browser**:

```
Agente necesita tool de browser
     |
     v
browser_detect_system():
  Encontro Chrome en /usr/bin/google-chrome? -> Usarlo (0 MB extra)
  No encontro nada?
     |
     v
Proponer al usuario:
  "Necesito un browser para automatizacion web.
   [1] Instalar Chromium headless (~200 MB) — recomendado
   [2] Instalar via Playwright (~350 MB, requiere Node.js)
   [3] No instalar (browser tools deshabilitados)
   Nota: se registrara en el sistema y podras desinstalarlo despues."
     |
     v
Si instala -> Registrar en InstalledPackageRegistry (seccion 20.5)
```

Todo lo instalado se registra y puede ser desinstalado (ver seccion 20.5).

### 8.2 Tools de Browser

#### Navegacion y lectura

| Tool | Parametros | Descripcion | Riesgo |
|------|-----------|-------------|--------|
| `browser_navigate` | `url: String` | Navegar a URL | MEDIO |
| `browser_back` | - | Volver atras | BAJO |
| `browser_forward` | - | Ir adelante | BAJO |
| `browser_refresh` | - | Recargar pagina | BAJO |
| `browser_get_url` | - | URL actual | BAJO |
| `browser_get_title` | - | Titulo de la pagina | BAJO |
| `browser_get_text` | `selector: String` | Extraer texto de elemento | BAJO |
| `browser_get_html` | `selector: Option<String>` | Obtener HTML | BAJO |
| `browser_get_attribute` | `selector: String, attr: String` | Leer atributo de elemento | BAJO |
| `browser_screenshot` | `full_page: bool` | Captura de pantalla | BAJO |
| `browser_wait` | `selector: String, timeout_ms: u64` | Esperar elemento | BAJO |
| `browser_scroll` | `direction: String, amount: u32` | Scroll | BAJO |

#### Interaccion (inspirado en Selenium/Playwright test APIs)

| Tool | Parametros | Descripcion | Riesgo |
|------|-----------|-------------|--------|
| `browser_click` | `selector: String` | Click en elemento | MEDIO |
| `browser_double_click` | `selector: String` | Doble click | MEDIO |
| `browser_right_click` | `selector: String` | Click derecho | MEDIO |
| `browser_hover` | `selector: String` | Hover sobre elemento | BAJO |
| `browser_type` | `selector: String, text: String` | Escribir en input | MEDIO |
| `browser_clear` | `selector: String` | Limpiar campo | BAJO |
| `browser_press_key` | `key: String, modifiers: Vec<String>` | Pulsar tecla (Enter, Tab, Ctrl+A, etc.) | MEDIO |
| `browser_fill_form` | `fields: HashMap<String, String>` | Rellenar multiples campos | MEDIO |
| `browser_select_option` | `selector: String, value: String` | Seleccionar en dropdown | MEDIO |
| `browser_check` | `selector: String, checked: bool` | Marcar/desmarcar checkbox | MEDIO |
| `browser_upload_file` | `selector: String, file_path: String` | Upload de archivo | ALTO |
| `browser_drag_drop` | `from: String, to: String` | Drag and drop | MEDIO |

#### JavaScript y avanzado

| Tool | Parametros | Descripcion | Riesgo |
|------|-----------|-------------|--------|
| `browser_execute_js` | `script: String` | Ejecutar JavaScript | ALTO |
| `browser_evaluate` | `expression: String` | Evaluar expresion JS y devolver resultado | ALTO |
| `browser_wait_for_network` | `timeout_ms: u64` | Esperar a que no haya requests pendientes | BAJO |
| `browser_get_cookies` | - | Listar cookies | MEDIO |
| `browser_set_cookie` | `name, value, domain` | Establecer cookie | MEDIO |
| `browser_get_local_storage` | `key: Option<String>` | Leer localStorage | BAJO |

### 8.3 Multi-Sesion de Browser

```rust
struct BrowserSessionManager {
    /// Sesiones activas (cada una con su propio perfil temporal)
    sessions: HashMap<String, BrowserSession>,
    /// Sesion activa actualmente
    active_session: Option<String>,
}

struct BrowserSession {
    id: String,
    name: String,           // "admin", "user1", "anonymous"
    profile_dir: PathBuf,   // directorio temporal unico por sesion
    cookies: Vec<Cookie>,   // cookies separadas por sesion
    is_logged_in: bool,
    created_at: u64,
}
```

**Casos de uso**:
- Tener sesion "admin" y sesion "user" para comparar lo que ve cada rol
- Sesion "logada" y sesion "anonima" para verificar accesos
- Multiples cuentas de test en paralelo
- Cambiar entre sesiones: `browser_switch_session("admin")`
- Crear sesion: `browser_new_session("test_user")`
- Listar: `browser_list_sessions()`
- Cerrar: `browser_close_session("test_user")`

### 8.4 Privacidad en Browser

Cuando el browser va a enviar datos (formularios, posts):

1. **Detectar datos privados** en los campos (emails, nombres, telefonos, tarjetas)
2. **Pedir permiso** al usuario si se detectan datos sensibles
3. **Opcion de anonimizar**: reemplazar datos reales por datos ficticios antes de enviar
4. **Opcion de des-anonimizar**: cuando se recibe respuesta, recolocar los datos reales en el resultado local
5. Esta decision se puede configurar por dominio/host en la politica

### 8.5 Seguridad del Browser

- **Verificacion de URL antes de navegar**: consultar servicios online de reputacion (ver 8.6)
- **Nunca** rellenar formularios con credenciales reales sin confirmacion explicita
- **Nunca** ejecutar JavaScript arbitrario sin confirmacion
- Bloquear descargas automaticas
- El browser debe ejecutarse en modo headless con perfil temporal (no usar el perfil del usuario)
- Capturas de pantalla pueden contener datos sensibles — marcarlas como tal en el audit log
- Detectar si una pagina pide credenciales y alertar al usuario

### 8.6 Verificacion de Seguridad de URLs

Antes de navegar a una URL desconocida, opcionalmente consultar servicios de reputacion:

```rust
struct UrlSafetyConfig {
    /// Habilitar verificacion de URLs
    enabled: bool,
    /// Servicios a consultar (en orden)
    services: Vec<UrlSafetyService>,
    /// Que hacer segun el resultado
    on_safe: UrlAction,        // Navigate (default)
    on_unknown: UrlAction,     // AskUser (default)
    on_suspicious: UrlAction,  // AskUser (default)
    on_malicious: UrlAction,   // Block (default)
    /// Cache de resultados (evitar consultar cada vez)
    cache_ttl_secs: u64,
}

enum UrlSafetyService {
    GoogleSafeBrowsing { api_key: String },
    VirusTotal { api_key: String },
    /// Listas locales de dominios conocidos como seguros
    LocalWhitelist(Vec<String>),
    /// Listas locales de dominios conocidos como peligrosos
    LocalBlacklist(Vec<String>),
    /// Servicio custom (URL de API)
    Custom { endpoint: String },
}

enum UrlAction {
    Navigate,           // navegar directamente
    AskUser,            // pedir confirmacion con el informe
    Block,              // bloquear y notificar
    NavigateInSandbox,  // navegar pero en sesion aislada
}
```

---

## 9. MCP (Model Context Protocol)

### 9.1 Que es MCP

MCP es un protocolo estandar abierto (originalmente de Anthropic, ahora bajo la Linux Foundation) para conectar LLMs con herramientas externas. Define un formato JSON-RPC para:
- Descubrir tools disponibles en un servidor MCP
- Llamar tools con parametros tipados
- Recibir resultados estructurados

### 9.2 Relevancia para Nuestro Sistema

MCP nos permitiria:
- **Exponer nuestros tools como servidor MCP**: cualquier cliente MCP (Claude, ChatGPT, VS Code) podria usar nuestros tools
- **Consumir servidores MCP existentes**: reusar tools de la comunidad (filesystem, git, browser, bases de datos)
- **Estandarizar la interfaz de tools**: en vez de inventar nuestro propio protocolo

### 9.3 Servidores MCP Relevantes

- **Filesystem MCP Server**: lectura/escritura de archivos con control de acceso
- **Git MCP Server**: operaciones git
- **Fetch MCP Server**: HTTP requests
- **Memory MCP Server**: knowledge graph persistente
- **Playwright MCP Server**: browser automation
- **GitHub MCP Server**: issues, PRs, code search

### 9.4 Integracion Propuesta

Fase 1: Implementar un cliente MCP basico en Rust para consumir servidores existentes
Fase 2: Exponer nuestro `ToolRegistry` como servidor MCP
Fase 3: Soporte para descubrimiento dinamico de servidores MCP

---

## 10. Proyectos Open Source Analizados

### 10.1 Agentes de Codigo / Software Engineering

#### Claude Code (Anthropic)
- **URL**: https://github.com/anthropics/claude-code
- **Lenguaje**: TypeScript/Node.js
- **Descripcion**: Agente de codificacion CLI que lee tu codebase, edita archivos, y ejecuta comandos en la terminal
- **Stars**: ~40K+
- **Arquitectura clave**:
  - **Single-threaded master loop**: un loop simple y secuencial, sin orquestacion compleja. Prioriza debuggability y transparencia
  - **Permission system de 3 capas**: deny rules > ask rules > allow rules. La primera coincidencia gana
  - **Sandboxing OS-level**: usa `bubblewrap` (Linux) y `seatbelt` (macOS) para aislamiento de filesystem y red. Reduce prompts de permisos un 84%
  - **Sandbox-runtime**: proceso separado que aplica restricciones sin contenedor
  - **Skills system**: archivos markdown que ensenan al LLM nuevas capacidades sin consumir contexto
  - **Sub-agentes**: puede lanzar multiples agentes en paralelo, con un lead que coordina
- **Ideas aplicables**:
  - El modelo de permisos deny > ask > allow es elegante y robusto
  - El sandbox OS-level es mucho mas ligero que Docker y suficiente para la mayoria de casos
  - El concepto de "skills" como archivos declarativos es poderoso

#### Aider
- **URL**: https://github.com/Aider-AI/aider
- **Lenguaje**: Python
- **Descripcion**: AI pair programming en la terminal. Edita codigo directamente en tu repo con commits git automaticos
- **Stars**: ~30K+
- **Arquitectura clave**:
  - **Repo map**: genera un mapa de todo el codebase (funciones, clases, imports) para dar contexto al LLM sin enviar todo el codigo
  - **Edit formats**: multiples formatos de edicion (unified diff, whole file, search/replace) adaptados a cada modelo
  - **Git-native**: cada cambio es un commit git automatico, facil de revertir con `git undo`
  - **Multi-file editing**: puede editar multiples archivos en una sola operacion
  - **Voice support**: acepta comandos por voz
- **Ideas aplicables**:
  - El "repo map" es brillante — da contexto sin enviar megabytes de codigo
  - La estrategia de commit-por-cambio hace todo reversible
  - Los edit formats adaptados por modelo evitan errores de parsing

#### SWE-agent (Princeton)
- **URL**: https://github.com/SWE-agent/SWE-agent
- **Lenguaje**: Python
- **Descripcion**: Toma un issue de GitHub y lo intenta resolver automaticamente. NeurIPS 2024
- **Stars**: ~15K+
- **Arquitectura clave**:
  - **ACI (Agent-Computer Interface)**: interfaz personalizada entre el LLM y el OS. No usa herramientas del sistema directamente, sino comandos adaptados para LLMs
  - **mini-swe-agent**: version minima (~100 lineas) que logra >74% en SWE-bench
  - **Edit commands**: `edit`, `create`, `open` — comandos simplificados en vez de `vim` o `sed`
  - **Linting integrado**: verifica el codigo despues de cada edicion
- **Ideas aplicables**:
  - La idea del ACI es clave: los LLMs trabajan mejor con interfaces disenadas para ellos, no para humanos
  - El linting post-edicion es esencial para evitar codigo roto
  - mini-swe-agent demuestra que simple > complejo

#### OpenHands (ex-OpenDevin)
- **URL**: https://github.com/OpenHands/OpenHands
- **Lenguaje**: Python
- **Descripcion**: Plataforma de agentes de desarrollo de software que pueden hacer todo lo que un dev humano: editar codigo, ejecutar comandos, navegar web
- **Stars**: ~50K+
- **Arquitectura clave**:
  - **Event-sourced state model**: todo el estado del agente es una secuencia de eventos, con replay determinista
  - **SDK modular**: agentes, tools y workspaces como paquetes separados
  - **Opt-in sandboxing**: mismo agente corre local (prototipo) o en contenedor (produccion)
  - **Delegacion jerarquica**: agentes pueden delegar subtareas a otros agentes
  - **MCP integration**: soporte para Model Context Protocol
  - **MIT license**
- **Ideas aplicables**:
  - Event sourcing para estado del agente es muy potente para debugging y replay
  - El opt-in sandboxing es practico: no forzar contenedores en desarrollo
  - La delegacion jerarquica es util para tareas complejas

#### Devon
- **URL**: https://github.com/entropy-research/Devon
- **Lenguaje**: Python
- **Descripcion**: Pair programmer open-source con multi-file editing, exploracion de codigo, bug fixes
- **Stars**: ~3K+
- **Ideas aplicables**:
  - Enfoque en exploracion de codigo antes de editar (read-first)
  - UI separada (npx devon-ui) del backend

### 10.2 Frameworks de Agentes Multi-Proposito

#### AutoGPT
- **URL**: https://github.com/Significant-Gravitas/AutoGPT
- **Lenguaje**: Python
- **Descripcion**: Agente autonomo que descompone tareas complejas en subtareas y las ejecuta
- **Stars**: ~165K+
- **Arquitectura clave**:
  - **Task decomposition**: descompone objetivo alto nivel en subtareas
  - **Block SDK**: sistema de bloques con auto-registro para crear tools
  - **Memoria episodica y semantica**: recuerda acciones pasadas y su contexto
  - **Monitoring**: alertas de tasa de error por bloque
- **Ideas aplicables**:
  - La memoria episodica (recordar que funciono y que no) es muy util
  - El Block SDK con auto-registro es un buen patron para extensibilidad

#### CrewAI
- **URL**: https://github.com/crewAIInc/crewAI
- **Lenguaje**: Python
- **Descripcion**: Framework de multi-agentes con roles especializados que colaboran como un equipo
- **Stars**: ~25K+
- **Arquitectura clave**:
  - **Role-based agents**: cada agente tiene un rol (researcher, writer, reviewer)
  - **Crews + Flows**: "Crews" para colaboracion, "Flows" para orquestacion de pasos
  - **Shared memory**: memoria compartida entre agentes (short-term, long-term, entity, contextual)
  - **100+ tools integrados**: busqueda web, interaccion con websites, vector DBs
  - **Delegacion natural**: agentes pueden pedir ayuda a otros
- **Ideas aplicables**:
  - El sistema de memoria compartida entre agentes es sofisticado
  - Los "Flows" para workflows de larga duracion con state persistence

#### OpenAI Agents SDK (ex-Swarm)
- **URL**: https://github.com/openai/openai-agents-python
- **Lenguaje**: Python
- **Descripcion**: Framework ligero de OpenAI para workflows multi-agente. Evolucion de Swarm
- **Stars**: ~20K+
- **Arquitectura clave**:
  - **Minimalista**: pocas abstracciones, facil de entender
  - **Agent handoffs**: un agente puede transferir el control a otro
  - **Provider-agnostic**: soporta cualquier LLM
  - **Guardrails**: validacion de input/output integrada
- **Ideas aplicables**:
  - La simplicidad del diseno es admirable
  - Los "handoffs" entre agentes son mas naturales que la delegacion explicita

#### MetaGPT
- **URL**: https://github.com/FoundationAgents/MetaGPT
- **Lenguaje**: Python
- **Descripcion**: Simula una empresa de software con agentes especializados (PM, arquitecto, dev, QA)
- **Stars**: ~45K+
- **Arquitectura clave**:
  - **SOP (Standard Operating Procedures)**: los agentes siguen procedimientos definidos, no improvisacion
  - **Assembly-line paradigm**: tareas pasan de agente en agente como en cadena de montaje
  - **Document generation**: PRDs, diagramas de arquitectura, tasks, codigo
  - **AFlow**: automatizacion de workflows agenticos (ICLR 2025)
- **Ideas aplicables**:
  - Los SOPs son un buen mecanismo de control: definen QUE puede hacer cada agente
  - El paradigma de cadena de montaje es mas predecible que agentes libres

### 10.3 Sandboxing y Seguridad

#### E2B
- **URL**: https://github.com/e2b-dev/E2B
- **Lenguaje**: TypeScript/Python SDK
- **Descripcion**: Entornos sandbox en la nube para ejecucion segura de codigo generado por IA
- **Stars**: ~9K+
- **Arquitectura clave**:
  - **Firecracker microVMs**: cada sandbox es una micro-maquina virtual, no un contenedor
  - **Efimero**: sandboxes se crean y destruyen en milisegundos
  - **E2B Desktop**: sandbox con entorno grafico para browser automation
  - **Code Interpreter SDK**: ejecutar codigo Python/JS dentro del sandbox
  - **Usado por**: Hugging Face, Groq
- **Ideas aplicables**:
  - Firecracker microVMs son el gold standard de aislamiento
  - El concepto de sandbox efimero: crear, ejecutar, destruir
  - E2B Desktop demuestra que browser automation dentro de sandbox es viable

#### Microsoft TaskWeaver
- **URL**: https://github.com/microsoft/TaskWeaver
- **Lenguaje**: Python
- **Descripcion**: Framework code-first para tareas de data analytics con verificacion de codigo
- **Stars**: ~5K+
- **Arquitectura clave**:
  - **Code verification**: verifica el codigo generado ANTES de ejecutar
  - **Auto-fix**: detecta problemas y los corrige automaticamente
  - **Security sandbox**: ejecucion en sandbox opcional
  - **Session management**: datos separados por sesion/usuario
  - **Rich data structures**: soporta DataFrames, no solo texto
- **Ideas aplicables**:
  - La verificacion pre-ejecucion del codigo es una capa de seguridad importante
  - El auto-fix ahorra iteraciones del loop agentico

### 10.4 Browser Automation

#### browser-use
- **URL**: https://github.com/browser-use/browser-use
- **Lenguaje**: Python
- **Descripcion**: Hace sitios web accesibles para agentes IA. Automatiza tareas online
- **Stars**: ~55K+
- **Ideas aplicables**:
  - Diseno AI-first para browser automation
  - Compatible con multiples LLMs

#### Stagehand (Browserbase)
- **URL**: https://github.com/browserbase/stagehand
- **Lenguaje**: TypeScript
- **Descripcion**: Framework de browser automation que combina AI con codigo preciso
- **Stars**: ~15K+
- **Arquitectura clave**:
  - **Hibrido**: usa AI para navegacion desconocida, codigo para acciones precisas
  - **Auto-caching**: recuerda acciones previas, ejecuta sin LLM si la pagina no cambio
  - **Self-healing**: detecta cuando la pagina cambio y re-involucra al AI
  - **Preview mode**: muestra que haria antes de hacerlo
  - **Modular drivers**: Puppeteer, CDP directo, etc.
- **Ideas aplicables**:
  - El hibrido AI+codigo es la mejor estrategia: AI cuando no sabes, codigo cuando sabes
  - Auto-caching reduce costes de tokens drasticamente
  - Self-healing hace la automatizacion robusta

### 10.5 Tool Integration

#### Composio
- **URL**: https://github.com/ComposioHQ/composio
- **Lenguaje**: Python/TypeScript
- **Descripcion**: Plataforma de integracion de tools para agentes IA. 800+ toolkits
- **Stars**: ~15K+
- **Arquitectura clave**:
  - **Managed auth**: gestiona autenticacion (OAuth, API keys) por ti
  - **800+ toolkits**: GitHub, Slack, Gmail, Notion, Jira, etc.
  - **Framework-agnostic**: funciona con LangChain, CrewAI, AutoGPT, etc.
  - **MCP compatible**: expone tools como servidores MCP
- **Ideas aplicables**:
  - La gestion centralizada de auth para tools es crucial
  - El enfoque "catalogo de tools" es escalable

### 10.6 Frameworks en Rust

#### Rig
- **URL**: https://github.com/0xPlaygrounds/rig
- **Lenguaje**: Rust
- **Descripcion**: Framework para LLM apps en Rust. Modular, portable, compilable a WASM
- **Stars**: ~5K+
- **Arquitectura clave**:
  - **Abstracciones sobre providers**: OpenAI, Cohere, etc. unificados
  - **RAG integrado**: vector stores (MongoDB, in-memory)
  - **Pipelines agenticos**: pasos asistidos por LLM
  - **Custom tools API**: define tools con trait
  - **WASM support**: puede correr en browser
- **Ideas aplicables**:
  - La compilacion a WASM es unica — agentes en el browser
  - El enfoque modular con trait-based tools se alinea con nuestro diseno

#### AutoAgents (Rust)
- **URL**: https://github.com/liquidos-ai/AutoAgents
- **Lenguaje**: Rust
- **Descripcion**: Framework multi-agente en Rust para construir y coordinar agentes
- **Ideas aplicables**: Referencia de como otros implementan agentes en Rust

---

## 11. Ideas Extraidas de Cada Proyecto

### Ideas de Alta Prioridad (implementar primero)

| Idea | Origen | Descripcion |
|------|--------|-------------|
| **Single-threaded master loop** | Claude Code | Loop simple y secuencial. No sobredisenar la orquestacion |
| **Deny > Ask > Allow** | Claude Code | Modelo de permisos con deny-first. Robusto y predecible |
| **OS-level sandbox** | Claude Code | Bubblewrap/seatbelt en vez de Docker. Ligero y efectivo |
| **Repo map** | Aider | Mapa de funciones/clases del codebase para dar contexto sin enviar todo |
| **Commit-por-cambio** | Aider | Cada cambio = commit git. Todo reversible |
| **ACI (Agent-Computer Interface)** | SWE-agent | Interfaz simplificada para LLMs, no rehusar comandos humanos |
| **Lint post-edicion** | SWE-agent | Verificar codigo despues de cada edit |
| **Event sourcing** | OpenHands | Estado del agente como secuencia de eventos. Replay determinista |
| **Code verification** | TaskWeaver | Verificar codigo generado ANTES de ejecutar |
| **Dry-run automatico** | Propio | Ejecutar version segura antes de pedir confirmacion |

### Ideas de Media Prioridad

| Idea | Origen | Descripcion |
|------|--------|-------------|
| **Edit formats por modelo** | Aider | Cada LLM trabaja mejor con un formato de edicion especifico |
| **Memoria episodica** | AutoGPT | Recordar que tools funcionaron y cuales fallaron |
| **Role-based agents** | CrewAI | Agentes con roles definidos para tareas complejas |
| **Skills declarativas** | Claude Code | Archivos markdown que ensenan capacidades sin codigo |
| **Auto-caching** | Stagehand | Cachear acciones repetidas (browser, tools frecuentes) |
| **Self-healing** | Stagehand | Detectar cuando algo cambio y adaptar |
| **Shared memory** | CrewAI | Memoria compartida entre agentes |
| **MCP client** | OpenHands | Consumir tools de servidores MCP existentes |

### Ideas de Baja Prioridad (futuro)

| Idea | Origen | Descripcion |
|------|--------|-------------|
| **Firecracker microVMs** | E2B | Maximo aislamiento para sandboxing |
| **SOP (procedures)** | MetaGPT | Procedimientos formales para agentes |
| **WASM agents** | Rig | Compilar agentes a WASM para browser |
| **Managed auth** | Composio | Gestion centralizada de credenciales para tools |
| **Agent handoffs** | OpenAI SDK | Transferir control entre agentes naturalmente |
| **AFlow** | MetaGPT | Automatizacion de generacion de workflows |
| **E2B Desktop** | E2B | Sandbox con entorno grafico para browser |

---

## 12. Plan de Implementacion

### Fase 1: Loop Agentico Real (critico)
1. Implementar `ToolAwareProvider` — enviar tools al LLM, parsear tool_calls
2. Reescribir `agentic_loop.rs` con el loop real: LLM -> parse -> execute -> feedback
3. Conectar con `unified_tools::ToolRegistry` para ejecucion
4. Tests con mock LLM que devuelve tool_calls predefinidos

### Fase 2: Tools del OS (critico)
1. Implementar tools de filesystem (read, write, edit, list, search, delete)
2. Implementar `run_command` con timeout y captura de output
3. Implementar tools de git (status, diff, commit, log)
4. Tests unitarios para cada tool

### Fase 3: Sistema de Permisos (critico)
1. Implementar `AgentPolicy` y configuracion
2. Implementar `RiskAssessment` y evaluacion de peligrosidad
3. Implementar analisis estatico de comandos
4. Implementar generacion de alternativas seguras
5. Implementar UI de aprobacion (terminal)
6. Implementar audit log

### Fase 4: Sandbox (importante)
1. Implementar Nivel 1: restriccion de paths en Rust puro
2. Investigar integracion con bubblewrap (Linux) / seatbelt (macOS)
3. Implementar Nivel 2 opcionalmente

### Fase 5: Mejoras de Contexto (importante)
1. Implementar repo map (al estilo Aider)
2. Lint post-edicion
3. Commit-por-cambio automatico
4. Event sourcing para estado del agente

### Fase 6: Browser Automation (opcional)
1. Integrar `chromiumoxide` o Playwright MCP
2. Implementar tools de browser
3. Sandboxing de browser (perfil temporal, headless)

### Fase 7: MCP (opcional)
1. Implementar cliente MCP basico
2. Exponer ToolRegistry como servidor MCP
3. Descubrimiento de servidores

---

## 13. Feedback Real: Que Tan Bien Funcionan

### Resumen Ejecutivo

| Proyecto | Funciona? | Limitaciones Principales | Nivel de Madurez |
|----------|-----------|--------------------------|------------------|
| **Claude Code** | Muy bien | Coste alto ($20-200/mo), solo macOS oficialmente | Produccion |
| **Aider** | Muy bien | Requiere buen modelo, 40-60% mas barato que alternativas | Produccion |
| **SWE-agent** | Bien en benchmarks | ~74% SWE-bench verified, cae a 15-23% en tareas reales complejas | Investigacion |
| **OpenHands** | Parcial | Alpha inestable, 26% SWE-bench Lite, sufre con tareas ambiguas | Beta |
| **AutoGPT** | Limitado | Loops frecuentes, hallucinations, coste alto en tokens | Beta |
| **CrewAI** | Parcial | Context overflow crashea agentes, debugging doloroso, bugs | Beta |
| **Goose** | Bien | 8.8/10, 3000+ MCP servers, vulnerabilidades de prompt injection | Produccion |
| **browser-use** | Parcial | ~89% accuracy controlada, CAPTCHAs, login walls, hallucinations | Beta |
| **E2B** | Excelente | Solo sandbox (no agente), ~150ms cold start, usado por Fortune 500 | Produccion |

### Detalle por Proyecto

#### Claude Code — Funciona muy bien
- Sonnet 4.5 resuelve problemas de SWE un 49% mas preciso que GPT-4o
- Contexto de 400K+ tokens
- El sandbox reduce prompts de permisos un 84%
- **Debilidad**: coste elevado, ecosistema cerrado (requiere Claude)
- **Realidad**: es el agente de codigo mas capaz en produccion a dia de hoy

#### Aider — Funciona muy bien
- Multi-OS (Windows, macOS, Linux), multi-LLM (incluido modelos locales gratis via Ollama)
- 40-60% mas barato que Cursor con funcionalidad equivalente
- Git-native: cada cambio es un commit, todo reversible
- **Debilidad**: depende del modelo — con modelos malos, los resultados son malos
- **Realidad**: la mejor opcion para developers que quieren control total y bajo coste

#### SWE-agent — Bien en benchmarks, limitado en la realidad
- mini-swe-agent logra >74% en SWE-bench Verified (benchmark curado)
- PERO en SWE-Bench Pro (mas realista): los mejores modelos solo logran 23%
- En codebases privados no vistos: cae a 15-18%
- **Leccion clave**: los benchmarks no reflejan la complejidad real. El scaffolding del agente (prompts, tools, context management) importa tanto como el modelo
- **Realidad**: bueno para issues bien definidos, malo para tareas abiertas

#### OpenHands — Prometedor pero inestable
- Alpha stage, "work in progress"
- SWE-bench Lite: 26%, HumanEvalFix: 79%, WebArena: 15%
- Funciona bien con tareas acotadas (scripts claros, errores precisos, criterios de aceptacion)
- **Falla con**: trabajo ambiguo, tests inestables, cambios multi-repo, tareas de largo plazo
- **Realidad**: tratar como "junior engineer" — darle tickets bien definidos, revisar su PR

#### AutoGPT — Mucho hype, resultados limitados
- 165K stars pero reviews mixtas
- Problemas frecuentes: loops infinitos, tangentes, busquedas Google sin fin
- Hallucinations especialmente con datos web y temas nicho
- Coste: $50-500+/mes dependiendo de complejidad
- **Leccion clave**: los agentes autonomos sin Human-in-the-Loop no funcionan en produccion
- **Realidad**: util como "orchestrator semi-autonomo" con checkpoints humanos, no como agente autonomo

#### CrewAI — Buena idea, ejecucion problematica
- Context window overflow crashea agentes sin que el crew se entere
- El agente "manager" no coordina realmente — ejecuta tareas secuencialmente
- Unit testing imposible, debugging doloroso
- Telemetria sin consentimiento
- Bugs de alta severidad en v1.1.0
- **Donde funciona**: research-to-draft, data enrichment, pipelines con review gates explicitos
- **Realidad**: requiere mucha experimentacion para afinar. No es plug-and-play

#### Goose — Sorprendentemente bueno
- 27K+ stars, 350+ contributors, 100+ releases
- Equipos de Block: "completaron 3 semanas de trabajo en 1 semana"
- Reduccion de errores de deployment ~40% vs manual
- 3000+ MCP servers disponibles
- **Pero**: Block red-teamo su propio agente en 2026 y encontro vulnerabilidades de prompt injection
- **Realidad**: buen agente de codigo general, especialmente con MCP. Escrito en Rust

#### browser-use — Prometedor pero no fiable
- ~89% accuracy en tareas controladas
- Supera a OpenAI Operator en WebVoyager benchmarks
- **Problemas reales**: CAPTCHAs, login walls, selectores DOM inestables, hallucinations en citas
- Accesibilidad: muchos agentes no pueden realmente "ver" las paginas
- **Leccion clave**: el patron ganador en 2025 es hibrido: determinista donde sea posible, agentico donde sea util
- **Realidad**: util para tareas de scraping/navegacion simples, no fiable para workflows complejos

#### E2B — Excelente en lo que hace
- ~150ms para crear un sandbox (Firecracker microVMs)
- Usado por ~88% de Fortune 100
- Millones de sandboxes por semana
- Python, JS, TS, Ruby, C++
- **No es un agente**: es infraestructura de sandbox. Necesita un agente encima
- **Realidad**: el gold standard para ejecucion segura de codigo. Dificil de self-host

### Lecciones Generales del Ecosistema

1. **Los agentes autonomos no funcionan sin human-in-the-loop**: AutoGPT, CrewAI, y OpenHands demuestran que dejar al agente solo lleva a loops, hallucinations, y costes desbocados

2. **Simple > Complejo**: mini-swe-agent (100 lineas) iguala a SWE-agent completo. Claude Code usa un loop single-threaded. La orquestacion compleja de CrewAI/MetaGPT causa mas problemas de los que resuelve

3. **El scaffolding importa mas que el modelo**: como presentas los tools, como manejas el contexto, y como das feedback es mas importante que el modelo que uses

4. **Benchmarks ≠ realidad**: SWE-bench Verified (74%) vs SWE-bench Pro (23%) vs codebases privados (15%). Reduccion de 5x en rendimiento real

5. **Git-native es esencial**: Aider y Claude Code hacen commit de cada cambio. Esto hace todo reversible y auditable

6. **El sandbox mas practico es OS-level**: Claude Code con bubblewrap/seatbelt es mas ligero que Docker y reduce 84% de prompts de permisos

7. **MCP es el futuro de tools**: Goose con 3000+ MCP servers demuestra que el enfoque "catalogo abierto" escala mejor que tools hardcodeados

---

## 14. Estrategias de Iteracion

### 14.1 Problema con max_iterations fijo

`max_iterations` es un numero fijo que limita cuantas veces el agente puede llamar al LLM.
Esto es demasiado rigido:
- Para operaciones con servicios externos de pago, 3-5 iteraciones puede ser demasiado
- Para tareas locales de exploracion de codigo, 50 iteraciones puede ser poco
- No distingue entre iteraciones productivas y loops sin progreso

### 14.2 Reemplazo: IterationStrategy

```rust
enum IterationStrategy {
    /// Limite fijo clasico (fallback)
    FixedLimit { max: usize },

    /// Limite por presupuesto de tokens/coste
    BudgetLimit {
        max_input_tokens: u64,
        max_output_tokens: u64,
        max_cost_usd: f64,       // estimacion basada en pricing del proveedor
    },

    /// Deteccion de progreso: parar si no hay avance
    ProgressBased {
        /// Max iteraciones sin progreso antes de parar
        max_stale_cycles: usize, // e.g. 3
        /// Como medir progreso (ver abajo)
        progress_detector: ProgressDetector,
        /// Limite absoluto de seguridad
        absolute_max: usize,     // e.g. 100
    },

    /// Combinacion de estrategias (la primera que se active gana)
    Combined(Vec<IterationStrategy>),
}

enum ProgressDetector {
    /// El agente hizo algun tool call nuevo (no repetido)
    NewToolCalls,
    /// El agente produjo output significativamente diferente al ciclo anterior
    OutputDifference { min_diff_ratio: f32 }, // e.g. 0.2 = 20% diferente
    /// El agente completo al menos un sub-objetivo
    SubGoalCompletion,
    /// Combinacion: cualquiera cuenta como progreso
    Any(Vec<ProgressDetector>),
}
```

### 14.3 Politicas por Tipo de Operacion

```rust
struct IterationPolicies {
    /// Para llamadas a LLMs de pago (cloud providers)
    paid_llm_calls: IterationStrategy,    // BudgetLimit + FixedLimit(20)
    /// Para operaciones locales (filesystem, git)
    local_operations: IterationStrategy,  // ProgressBased(stale=3, max=100)
    /// Para browser automation
    browser_actions: IterationStrategy,   // FixedLimit(30) + BudgetLimit
    /// Para network requests
    network_requests: IterationStrategy,  // FixedLimit(50) rate-limited
    /// Default para todo lo demas
    default: IterationStrategy,
}
```

### 14.4 Deteccion de Loops

```rust
struct LoopDetector {
    /// Historial de tool calls recientes
    recent_calls: VecDeque<ToolCallSignature>,
    /// Numero de ciclos a mirar atras
    window_size: usize, // e.g. 5

    /// Detectar: mismos tool calls repetidos
    fn is_repeating(&self) -> bool;
    /// Detectar: output del LLM es muy similar al anterior
    fn is_stale(&self, current_output: &str, previous_output: &str) -> bool;
    /// Detectar: el agente esta deshaciendo y rehaciendo lo mismo
    fn is_oscillating(&self) -> bool;
}
```

Cuando se detecta un loop:
1. Notificar al usuario: "El agente parece estar en un loop (misma accion 3 veces)"
2. Opciones: [Continuar] [Parar] [Dar pista al agente] [Cambiar estrategia]
3. Si el usuario da una pista, se inyecta como mensaje "system" en la conversacion

### 14.5 Informe Detallado de Parada

Cuando el agente se detiene por CUALQUIER razon (limite, enroque, presupuesto, error,
cancelacion manual), se genera un informe estructurado:

```rust
struct StopReport {
    /// Razon de la parada
    reason: StopReason,
    /// Iteraciones completadas vs presupuesto estimado
    iterations_completed: usize,
    iterations_budget: Option<usize>,
    /// Tokens consumidos
    input_tokens: u64,
    output_tokens: u64,
    /// Coste estimado en USD
    estimated_cost_usd: f64,
    /// Tiempo transcurrido
    elapsed: Duration,
    /// Progreso por iteracion (resumen legible de que hizo cada una)
    iteration_log: Vec<IterationSummary>,
    /// Requisitos cumplidos y no cumplidos (del RequirementRegistry)
    requirements_met: Vec<(String, RequirementStatus)>,
    /// Archivos creados/modificados/eliminados
    files_changed: Vec<FileChange>,
    /// Si hubo enroque: detalle del error recurrente
    recurring_error: Option<String>,
    /// Opciones disponibles para el usuario
    options: Vec<UserOption>,
}

enum StopReason {
    /// Tarea completada exitosamente
    Completed,
    /// Limite fijo alcanzado
    FixedLimitReached { limit: usize },
    /// Presupuesto de tokens/coste agotado
    BudgetExhausted { budget_type: String, used: f64, limit: f64 },
    /// Progreso insuficiente (N ciclos sin avance)
    InsufficientProgress { stale_cycles: usize },
    /// Enroque detectado (repitiendo los mismos cambios)
    StalemateDetected { repeating_pattern: String },
    /// Ciclo entre requisitos detectado (CycleDetector)
    RequirementCycle { conflicting_requirements: Vec<String> },
    /// Error no recuperable
    UnrecoverableError { error: String },
    /// Cancelado por el usuario
    UserCancelled,
    /// Escalacion necesaria (modelo actual insuficiente)
    EscalationNeeded { current_model: String, suggested_model: String },
}

struct IterationSummary {
    number: usize,
    /// Resumen de una linea de lo que hizo
    action: String,
    /// Progreso? (true = avanzo, false = no avanzo)
    made_progress: bool,
    /// Tokens usados en esta iteracion
    tokens: u64,
}

enum UserOption {
    /// Continuar iterando (mismo modelo, misma config) — para avance marginal
    Continue { additional_iterations: usize },
    /// Continuar PERO con cambios (modelo, pista, requisitos, estrategia)
    ContinueWithChanges {
        new_model: Option<(String, String)>,  // (provider, model)
        hint: Option<String>,                 // pista al agente
        modify_requirements: Vec<RequirementChange>,
        new_strategy: Option<IterationStrategy>,
    },
    /// Escalar a modelo mas potente (atajo de ContinueWithChanges)
    Escalate { model: String, estimated_cost: f64 },
    /// Reintentar desde cero (rollback completo de esta tarea + empezar de nuevo)
    RetryFromScratch,
    /// Reintentar parcial (rollback hasta iteracion N, continuar desde ahi)
    RetryFromIteration { iteration: usize },
    /// Saltar esta tarea, continuar con las demas (si hay multi-task)
    SkipTask,
    /// Parar y guardar progreso parcial
    StopAndSave,
    /// Deshacer todos los cambios de esta tarea
    RollbackTask,
    /// Deshacer TODAS las tareas (rollback global)
    RollbackAll,
    /// Dar una pista/instruccion al agente
    ProvideHint,
    /// Cambiar estrategia de iteracion
    ChangeStrategy,
}

enum RequirementChange {
    /// Relajar un requisito (de Must a Nice, o eliminarlo)
    Relax { requirement_id: String },
    /// Modificar un requisito
    Modify { requirement_id: String, new_description: String },
    /// Anadir un nuevo requisito
    Add { description: String, priority: Priority },
}

enum RequirementStatus {
    Met,              // Cumplido y verificado
    PartiallyMet,     // Parcialmente cumplido (ej: 2 de 3 tests)
    NotMet,           // No cumplido
    Blocked(String),  // No cumplido por un error especifico
    Skipped,          // El usuario decidio relajar/saltar este requisito
}
```

**Ejemplo de informe renderizado** (con opciones ampliadas):

```
╔══════════════════════════════════════════════════════════════════╗
║  AGENTE DETENIDO — Progreso insuficiente (3 ciclos)             ║
║  Tarea: T2 de 3 — "Anadir autenticacion JWT"                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Iteraciones: 7 de ~15 estimadas                                ║
║  Tokens: 12,400 (in) + 3,200 (out) | Coste: ~$0.08             ║
║  Tiempo: 2m 34s                                                  ║
║                                                                  ║
║  Progreso por iteracion:                                         ║
║    #1 ✓ Leidos 4 archivos, entendida estructura                 ║
║    #2 ✓ Generado plan de 3 pasos                                ║
║    #3 ✓ Implementado paso 1 (nuevo endpoint)                    ║
║    #4 ✓ Implementado paso 2 (validacion input)                  ║
║    #5 ✗ Paso 3: compilacion fallo (trait Serialize)             ║
║    #6 ✗ Intento arreglar, mismo error                           ║
║    #7 ✗ Mismo error — ENROQUE                                   ║
║                                                                  ║
║  Error recurrente:                                               ║
║    error[E0277]: `MyType: Serialize` not satisfied               ║
║    --> src/models.rs:45:10                                       ║
║                                                                  ║
║  Requisitos: 4/7 cumplidos                                       ║
║    ✓ R1: Endpoint /api/users existe                             ║
║    ✓ R2: Validacion de input                                    ║
║    ✓ R3: Tests unitarios (3 tests)                              ║
║    ✓ R4: Sin hardcoding                                         ║
║    ✗ R5: Serializacion JSON [BLOQUEADO por error E0277]         ║
║    ✗ R6: Test de integracion [NO INICIADO]                      ║
║    ✗ R7: Documentacion [NO INICIADO]                            ║
║                                                                  ║
║  Archivos modificados: 3                                         ║
║    + src/models.rs (nuevo, 45 lineas)                           ║
║    ~ src/handlers.rs (12 lineas cambiadas)                      ║
║    ~ src/lib.rs (1 linea: mod models)                           ║
║                                                                  ║
║  Undo disponible: Si (3 operaciones reversibles)                 ║
║                                                                  ║
║  Estado de otras tareas:                                         ║
║    T1 ✓ "Corregir bug login" — completada                      ║
║    T2 ✗ "Anadir auth" — DETENIDA (esta tarea)                  ║
║    T3 ⏸ "Actualizar README" — depende de T2, en espera         ║
║                                                                  ║
║  Opciones:                                                       ║
║    [1] Continuar iterando (mismo modelo, +10 iteraciones)       ║
║    [2] Continuar con cambios:                                    ║
║        a) Escalar a Claude Opus (~$0.30 extra)                  ║
║        b) Dar pista al agente                                    ║
║        c) Modificar requisitos (relajar/cambiar/anadir)         ║
║    [3] Reintentar desde cero (rollback T2, empezar de nuevo)    ║
║    [4] Reintentar desde iteracion #4 (deshacer #5-#7)           ║
║    [5] Saltar T2, continuar con T3 sin auth                     ║
║    [6] Parar y guardar progreso parcial                         ║
║    [7] Deshacer T2 (rollback solo esta tarea)                   ║
║    [8] Deshacer TODO (rollback T1 + T2)                         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 15. Politicas de Privacidad por Proveedor

### 15.1 Problema

Diferentes proveedores de LLM tienen diferentes niveles de confianza:
- **Ollama local**: total confianza, datos nunca salen de la maquina
- **OpenAI/Anthropic cloud**: confianza razonable, pero los datos viajan por internet
- **Proveedor desconocido**: cero confianza, podrian guardar/analizar datos

### 15.2 Politica de Privacidad por Proveedor

```rust
struct ProviderPrivacyPolicy {
    /// Nombre del proveedor
    provider: AiProvider,
    /// Nivel de confianza
    trust_level: TrustLevel,
    /// Anonimizar datos antes de enviar al LLM?
    anonymize_outgoing: bool,
    /// Des-anonimizar datos en la respuesta del LLM?
    deanonymize_incoming: bool,
    /// Tipos de datos a anonimizar
    anonymize_types: Vec<SensitiveDataType>,
    /// El LLM debe ser consciente de que se anonimiza?
    llm_aware: LlmAwareness,
}

enum TrustLevel {
    Full,       // Ollama local, modelos propios
    High,       // OpenAI, Anthropic (con ZDR policies)
    Medium,     // Proveedores cloud conocidos
    Low,        // Proveedores desconocidos
    None,       // Sin confianza, anonimizar todo
}

enum SensitiveDataType {
    PersonalNames,
    Emails,
    PhoneNumbers,
    Addresses,
    CreditCards,
    SocialSecurityNumbers,
    ApiKeys,
    Passwords,
    InternalUrls,
    CompanyNames,
    ProjectNames,
    CustomPattern(String), // regex
}

enum LlmAwareness {
    /// El LLM sabe que los datos estan anonimizados y trabaja con placeholders
    /// Ventaja: puede razonar mejor ("el usuario John..." vs "el usuario [PERSON_1]...")
    /// Uso: cuando el LLM de confianza prepara el prompt que luego va a otro LLM
    Aware,

    /// El LLM NO sabe que los datos son ficticios
    /// Ventaja: no puede filtrar la existencia de datos reales
    /// Uso: cuando envias a un proveedor no confiable
    Unaware,

    /// Decide automaticamente segun contexto
    /// - Si el mismo LLM preparo el prompt: Aware
    /// - Si un LLM externo preparo el prompt: Unaware
    /// - Si es un LLM local: no anonimizar
    Auto,
}
```

### 15.3 Flujo de Anonimizacion

```
Datos originales          Anonimizado              Respuesta LLM           Des-anonimizado
+------------------+     +------------------+     +------------------+    +------------------+
| "Juan Garcia     | --> | "[PERSON_1]      | --> | "[PERSON_1] debe | -> | "Juan Garcia     |
|  trabaja en      |     |  trabaja en      |     |  actualizar su   |    |  debe actualizar |
|  Acme Corp con   |     |  [COMPANY_1] con |     |  email en        |    |  su email en     |
|  juan@acme.com"  |     |  [EMAIL_1]"      |     |  [COMPANY_1]"    |    |  Acme Corp"      |
+------------------+     +------------------+     +------------------+    +------------------+
```

### 15.4 Mapa de Anonimizacion

```rust
struct AnonymizationMap {
    /// Mapeo placeholder -> dato real
    mappings: HashMap<String, String>,
    /// Contador por tipo para generar placeholders unicos
    counters: HashMap<SensitiveDataType, usize>,
    /// Sesion a la que pertenece este mapa
    session_id: String,
}

impl AnonymizationMap {
    /// Anonimizar texto
    fn anonymize(&mut self, text: &str) -> String;
    /// Des-anonimizar texto
    fn deanonymize(&self, text: &str) -> String;
    /// Mismo placeholder para misma entidad (consistencia)
    fn get_or_create_placeholder(&mut self, data: &str, data_type: SensitiveDataType) -> String;
}
```

### 15.5 Caso Complejo: LLM Externo Prepara Prompt

Escenario: Un modelo externo (no confiable) genera un prompt que luego se ejecuta en nuestro agente local.

```
LLM externo (no confiable)    Nuestro sistema              LLM local (confiable)
+----------------------+      +---------------------+      +------------------+
| "Analiza los datos   | ---> | Recibimos prompt    | ---> | Ejecuta con      |
|  del usuario"        |      | Detectamos que pide |      | datos REALES     |
+----------------------+      | acceso a datos      |      | (no anonimizar,  |
                              | Verificamos permisos|      |  es local)       |
                              | Si es para reenviar |      +------------------+
                              | a otro LLM externo: |
                              | ANONIMIZAR primero  |
                              +---------------------+
```

Regla: si el resultado va a salir de la maquina, anonimizar. Si se queda local, no.

---

## 16. Sistema de Undo/Rollback

### 16.1 Registro de Operaciones

Cada operacion del agente se registra en un log de undo:

```rust
struct UndoLog {
    operations: Vec<UndoableOperation>,
    max_history: usize,
}

struct UndoableOperation {
    /// ID unico
    id: String,
    /// Tool que se ejecuto
    tool_name: String,
    /// Argumentos originales
    args: serde_json::Value,
    /// Timestamp
    executed_at: u64,
    /// Es reversible?
    reversible: bool,
    /// Datos para deshacer
    undo_data: Option<UndoData>,
    /// Ya se deshizo?
    undone: bool,
}

enum UndoData {
    /// Para write_file/edit_file: contenido original del archivo
    FileBackup { path: PathBuf, original_content: Vec<u8> },
    /// Para delete_file: archivo movido a trash
    FileInTrash { original_path: PathBuf, trash_path: PathBuf },
    /// Para delete_directory: directorio movido a trash
    DirInTrash { original_path: PathBuf, trash_path: PathBuf },
    /// Para create_directory: path creado
    CreatedPath(PathBuf),
    /// Para move_file: src y dst originales
    MovedFile { original_src: PathBuf, original_dst: PathBuf },
    /// Para git_commit: hash del commit
    GitCommit { repo: PathBuf, commit_hash: String },
    /// Para run_command: no reversible automaticamente
    CommandExecuted { command: String, output: String },
}
```

### 16.2 Safe Delete: Mover a Papelera en vez de Borrar

```rust
struct SafeDeleteConfig {
    /// Habilitar safe delete (mover a trash en vez de borrar)
    enabled: bool,
    /// Directorio de papelera
    trash_dir: PathBuf,     // default: .agent_trash/ en el proyecto
    /// Tiempo antes de limpiar la papelera automaticamente
    auto_cleanup_hours: Option<u64>,  // None = nunca limpiar auto
    /// Tamano maximo de la papelera antes de alertar
    max_trash_size_mb: u64,
}
```

**Flujo de safe delete**:
1. `delete_file("important.txt")` -> el sistema mueve a `.agent_trash/important.txt.1708012345`
2. El usuario puede ejecutar `undo_last()` o `undo(operation_id)` para restaurar
3. Periodicamente (o cuando se pida), se puede limpiar la papelera: `clean_trash(older_than_hours: 24)`
4. Notificacion cuando la papelera supera el tamano maximo

**Consideracion multi-OS**: en algunos sistemas, `rm` no es reversible. El safe delete lo hace reversible en TODOS los sistemas operativos porque nunca borra realmente, solo mueve.

### 16.3 UndoLog Per-Task (Multi-Task)

Cuando hay multiples tareas (seccion 27), cada tarea tiene su propio UndoLog aislado:

```rust
struct TaskUndoManager {
    /// UndoLog independiente por tarea
    task_logs: HashMap<String, UndoLog>,  // task_id -> UndoLog
    /// Indice inverso: archivo -> tareas que lo tocaron
    file_ownership: HashMap<PathBuf, Vec<String>>,
}

impl TaskUndoManager {
    /// Deshacer solo una tarea — verificar conflictos con otras primero
    fn rollback_task(&mut self, task_id: &str) -> RollbackResult;

    /// Deshacer desde una iteracion especifica de una tarea
    fn rollback_task_from_iteration(&mut self, task_id: &str, iteration: usize) -> RollbackResult;

    /// Deshacer todas las tareas (rollback global)
    fn rollback_all(&mut self) -> RollbackResult;
}

enum RollbackResult {
    /// Rollback exitoso sin conflictos
    Success { operations_undone: usize },
    /// Conflicto: otra tarea toco los mismos archivos
    Conflict {
        conflicting_tasks: Vec<TaskConflict>,
        /// Opciones para el usuario
        options: Vec<ConflictResolution>,
    },
    /// Operaciones no reversibles encontradas
    PartialSuccess {
        operations_undone: usize,
        irreversible: Vec<String>,
    },
}

struct TaskConflict {
    /// Tarea que tambien toco estos archivos
    other_task_id: String,
    other_task_description: String,
    /// Archivos en conflicto
    shared_files: Vec<PathBuf>,
    /// La otra tarea esta completada, en progreso, o parada?
    other_task_status: TaskStatus,
}

enum ConflictResolution {
    /// Deshacer ambas tareas (seguro)
    RollbackBoth,
    /// Deshacer solo la solicitada, intentar mantener cambios de la otra (arriesgado)
    RollbackOnlyRequested,
    /// No deshacer nada
    Cancel,
    /// Deshacer la solicitada y re-ejecutar la otra sobre el estado limpio
    RollbackAndRerun { rerun_task_id: String },
}
```

**Ejemplo**:

```
Tareas ejecutadas:
  T1 "Corregir bug" -> UndoLog: [edit src/auth.rs (linea 20)]
  T2 "Anadir JWT"   -> UndoLog: [create src/jwt.rs, edit src/auth.rs (linea 45), edit Cargo.toml]
  T3 "Actualizar README" -> UndoLog: [edit README.md]

El usuario pide "deshaz T2":
  1. file_ownership detecta: src/auth.rs tocado por T1 Y T2
  2. Analizar: T1 toco linea 20, T2 toco linea 45 -> regiones distintas
     -> Rollback seguro: deshacer cambios de T2 en auth.rs sin afectar T1
  3. Resultado: Success (3 operaciones deshechas)

El usuario pide "deshaz T1":
  1. file_ownership detecta: src/auth.rs tocado por T1 Y T2
  2. Analizar: T1 toco linea 20, T2 depende de que el bug este corregido
     -> Conflicto: si deshaces T1, T2 podria romperse
  3. Informar: "T2 (JWT) modifico src/auth.rs que T1 (bug fix) tambien toco.
     Si deshago T1, el bug vuelve y T2 podria no funcionar.
     [1] Deshacer T1 y T2 (seguro)
     [2] Deshacer solo T1, re-verificar T2
     [3] No deshacer nada"
```

### 16.4 Rollback Parcial (por Iteracion)

El usuario puede deshacer desde una iteracion especifica, no solo la tarea entera:

```
Tarea T2 con 7 iteraciones:
  #1: leyo archivos         (no hay cambios en disco)
  #2: genero plan           (no hay cambios en disco)
  #3: creo src/jwt.rs       ← UndoLog: create
  #4: edito src/auth.rs     ← UndoLog: edit (backup del original)
  #5: intento arreglar      ← UndoLog: edit (backup pre-#5)
  #6: mismo intento fallido ← UndoLog: edit (backup pre-#6)
  #7: mismo error           ← UndoLog: edit (backup pre-#7)

El usuario elige "Reintentar desde iteracion #4":
  -> Deshacer #7, #6, #5 (en orden inverso)
  -> El estado de src/auth.rs vuelve a como estaba despues de #4
  -> src/jwt.rs sigue existiendo (fue creado en #3, no se deshace)
  -> El agente continua desde el estado de #4 con el contexto de lo que fallo
```

### 16.5 Operaciones No Reversibles

Algunas operaciones no se pueden deshacer:
- `run_command`: el efecto del comando ya ocurrio (pero se guarda el output)
- `http_post`: el request ya se envio
- `browser_click`: la accion ya se ejecuto

Para estas, el sistema registra que son `reversible: false` y lo indica al usuario en el audit log.
Si un rollback incluye operaciones irreversibles, se informa:
```
"Rollback parcial: 5 de 7 operaciones deshechas.
 2 operaciones no son reversibles:
   - run_command('npm install jwt-lib') -> ya se instalo
   - http_post('https://api.example.com/register') -> ya se envio
 Los cambios en archivos SI se deshicieron."
```

### 16.6 Integracion con Todo el Sistema

El UndoLog se integra con:
- **Seccion 14.5** (StopReport): el informe muestra si hay undo disponible y cuantas operaciones
- **Seccion 21** (CompletionVerifier): si el verifier detecta problemas, puede sugerir rollback parcial
- **Seccion 24** (Hibrido): al escalar modelo, se mantiene el UndoLog (no se pierde)
- **Seccion 25** (Assembly-line): cada agente de la cadena tiene su UndoLog; si QA rechaza el codigo, se puede deshacer solo lo del Engineer
- **Seccion 27** (Multi-task): cada tarea tiene su UndoLog aislado (seccion 16.3)
- **Seccion 29** (Butler): el butler puede sugerir limpiar la papelera si crece mucho

---

## 17. Herramientas de Red Avanzadas

### 17.1 Problema: Sitios que Bloquean Bots/AI

Muchos sitios web bloquean requests de agentes IA (e.g. starcitizen.tools). Necesitamos metodos alternativos.

### 17.2 Configuracion de HTTP Client

```rust
struct HttpClientConfig {
    /// Estrategia de User-Agent
    user_agent_strategy: UserAgentStrategy,
    /// Headers adicionales por defecto
    default_headers: HashMap<String, String>,
    /// Usar proxy?
    proxy: Option<ProxyConfig>,
    /// Respetar robots.txt? (default: true, configurable)
    respect_robots_txt: bool,
    /// Rate limiting por host
    rate_limits: HashMap<String, RateLimit>,
    /// Reintentos con backoff
    retry_config: RetryConfig,
}

enum UserAgentStrategy {
    /// User-Agent por defecto del crate (transparente)
    Default,
    /// Simular navegador real
    BrowserEmulation(BrowserProfile),
    /// User-Agent custom
    Custom(String),
    /// Rotar entre varios User-Agents
    Rotating(Vec<String>),
}

enum BrowserProfile {
    Chrome,
    Firefox,
    Safari,
    /// Ultimo User-Agent conocido de Chrome en la plataforma actual
    LatestChrome,
}

struct ProxyConfig {
    url: String,
    auth: Option<(String, String)>,
    /// Tipo: HTTP, SOCKS5, etc.
    proxy_type: ProxyType,
}
```

### 17.3 Estrategias Anti-Bloqueo

| Estrategia | Descripcion | Configuracion |
|-----------|-------------|---------------|
| User-Agent rotation | Rotar entre UA de navegadores reales | `UserAgentStrategy::Rotating` |
| Headers de navegador | Enviar Accept, Accept-Language, etc. como un browser real | `default_headers` |
| Rate limiting | No hacer demasiadas requests al mismo host | `rate_limits` |
| Proxy | Usar proxy HTTP/SOCKS5 para cambiar IP | `proxy` |
| Browser fallback | Si HTTP falla, usar browser headless real | Tool `browser_navigate` como fallback |
| Cache agresivo | Cachear responses para no repetir requests | Cache con TTL configurable |

**Nota etica**: el usuario configura estas opciones bajo su responsabilidad. El sistema NO habilita bypass de seguridad por defecto. `respect_robots_txt` es `true` por defecto.

---

## 18. Sistema de Plugins Extensible

### 18.1 Problema: Soporte de VCS mas alla de Git

El sistema actual solo tiene tools de git. Pero algun usuario puede necesitar SVN, Mercurial, Perforce, etc.

### 18.2 VCS Plugin Trait

```rust
/// Trait para plugins de control de versiones
trait VcsPlugin: Send + Sync {
    /// Nombre del VCS (e.g. "git", "svn", "hg")
    fn name(&self) -> &str;

    /// Detectar si un directorio es un repo de este tipo
    fn detect(&self, path: &Path) -> bool;

    /// Operaciones estandar
    fn status(&self, repo_path: &Path) -> Result<VcsStatus>;
    fn diff(&self, repo_path: &Path, staged: bool) -> Result<String>;
    fn commit(&self, repo_path: &Path, message: &str, files: &[&str]) -> Result<String>;
    fn log(&self, repo_path: &Path, count: usize) -> Result<Vec<VcsLogEntry>>;
    fn revert(&self, repo_path: &Path, files: &[&str]) -> Result<()>;

    /// Operaciones opcionales (no todos los VCS las soportan)
    fn branch_list(&self, _repo_path: &Path) -> Result<Vec<String>> {
        Err(anyhow!("Not supported by this VCS"))
    }
    fn stash(&self, _repo_path: &Path) -> Result<()> {
        Err(anyhow!("Not supported by this VCS"))
    }

    /// Registrar tools en el ToolRegistry
    fn register_tools(&self, registry: &mut ToolRegistry);
}

/// Tipos estandar que todos los VCS plugins devuelven
struct VcsStatus {
    modified: Vec<String>,
    added: Vec<String>,
    deleted: Vec<String>,
    untracked: Vec<String>,
}

struct VcsLogEntry {
    id: String,         // hash, revision number, etc.
    author: String,
    date: String,
    message: String,
}
```

### 18.3 Viabilidad para Desarrolladores

Un desarrollador puede agregar soporte para SVN asi:

1. Crear un struct `SvnPlugin` que implemente `VcsPlugin`
2. Dentro de cada metodo, llamar al CLI de SVN (`svn status`, `svn diff`, etc.) via `std::process::Command`
3. Parsear el output al formato estandar (`VcsStatus`, `VcsLogEntry`)
4. Registrar el plugin: `agent.register_vcs_plugin(Box::new(SvnPlugin::new()))`

**Estimacion de esfuerzo**: 200-400 lineas de Rust por VCS. El 80% del trabajo es parsear output CLI.

### 18.4 Plugin Generico para Tools Custom

El mismo patron aplica para cualquier tipo de tool:

```rust
/// Trait para plugins de herramientas genericas
trait ToolPlugin: Send + Sync {
    /// Nombre del plugin
    fn name(&self) -> &str;
    /// Descripcion
    fn description(&self) -> &str;
    /// Verificar si las dependencias estan instaladas
    fn check_dependencies(&self) -> Result<()>;
    /// Registrar tools en el ToolRegistry
    fn register_tools(&self, registry: &mut ToolRegistry);
}
```

Ejemplos: plugin de Docker, plugin de Kubernetes, plugin de base de datos, etc.

---

## 19. Introspeccion del Sistema (Diagnostics Tool)

### 19.1 Proposito

Un tool/MCP que analiza el estado completo del sistema y genera un informe para:
- Desarrolladores que quieren entender que esta configurado
- La propia aplicacion para auto-diagnostico
- Un agente IA que necesita saber que tools tiene disponibles

### 19.2 Informe de Diagnostico

```rust
struct SystemDiagnostics {
    /// Features compiladas en este binario
    compiled_features: Vec<String>,       // e.g. ["rag", "tools", "async-runtime"]
    /// Features NO compiladas (para no hablar de ellas)
    missing_features: Vec<String>,        // e.g. ["egui-widgets"]

    /// Configuracion actual
    config_summary: ConfigSummary,
    /// Tools registrados y su estado
    tools_status: Vec<ToolStatus>,
    /// MCP servers conectados
    mcp_connections: Vec<McpConnectionStatus>,
    /// Providers de LLM configurados y su disponibilidad
    provider_status: Vec<ProviderStatus>,
    /// Agentes definidos
    agent_definitions: Vec<AgentInfo>,
    /// Agentes en ejecucion
    running_agents: Vec<RunningAgentInfo>,
    /// Contenedores/sandboxes activos
    active_containers: Vec<ContainerInfo>,
    /// Warnings y recomendaciones
    warnings: Vec<DiagnosticWarning>,
    /// Sugerencias de mejora
    suggestions: Vec<Suggestion>,
}

struct ToolStatus {
    name: String,
    category: String,
    enabled: bool,
    /// Dependencias externas necesarias (e.g. "git binary", "docker")
    dependencies_met: bool,
    missing_dependencies: Vec<String>,
    /// Feature flag requerida
    required_feature: Option<String>,
}

struct ProviderStatus {
    provider: AiProvider,
    configured: bool,
    reachable: bool,           // se pudo conectar?
    models_available: usize,
    has_api_key: bool,
    last_error: Option<String>,
}

struct DiagnosticWarning {
    severity: WarningSeverity,  // Info, Warning, Error
    category: String,           // "security", "performance", "config"
    message: String,
    suggestion: Option<String>,
}
```

### 19.3 Ejemplos de Warnings

```
[WARNING] security: AgentPolicy tiene allowed_write_dirs vacio — el agente no podra escribir archivos
[WARNING] performance: RAG esta habilitado pero no hay documentos indexados
[WARNING] config: Provider OpenAI configurado pero sin API key (ni en config ni en env OPENAI_API_KEY)
[INFO] tools: 5 tools registrados, 2 deshabilitados (browser_execute_js, run_shell)
[INFO] features: Feature 'async-runtime' compilada pero no en uso (ningun async provider configurado)
[SUGGESTION] performance: Considerar habilitar cache de embeddings (SharedEmbeddingCache) para mejorar RAG
[SUGGESTION] security: Considerar habilitar sandbox Nivel 2 para ejecucion de comandos
```

### 19.4 Requisito de Mantenimiento

**REGLA**: Cada vez que se agregue un nuevo modulo, tool, provider o feature al crate, se DEBE actualizar la funcion de diagnostico para incluirlo. Esto se puede reforzar con:

- Un test que verifica que todos los tools registrados aparecen en el diagnostico
- Un test que verifica que todos los features del Cargo.toml se reportan
- Documentacion en CONTRIBUTING.md recordandolo

### 19.5 Exposicion como MCP/Tool

El diagnostico se expone de 3 formas:
1. **Funcion Rust**: `system_diagnostics() -> SystemDiagnostics`
2. **Tool del agente**: `system_status` — el agente puede llamarlo para saber que tiene disponible
3. **MCP server** (futuro): endpoint `/diagnostics` para clientes externos

---

## 20. Gestion de Contenedores y Prerequisitos

### 20.1 Instalacion de Prerequisitos

```rust
struct PrerequisiteManager {
    /// SO detectado
    os: OperatingSystem, // Windows, Linux(distro), MacOS
    /// Package manager disponible
    package_manager: PackageManager, // apt, brew, choco, winget, pacman, dnf
}

impl PrerequisiteManager {
    /// Verificar si un prerequisito esta instalado
    fn is_installed(&self, name: &str) -> bool;
    /// Instalar un prerequisito (pide permiso al usuario)
    fn install(&self, name: &str) -> Result<()>;
    /// Listar prerequisitos necesarios para una feature
    fn required_for(&self, feature: &str) -> Vec<Prerequisite>;
}

struct Prerequisite {
    name: String,
    /// Comando de verificacion (e.g. "docker --version")
    check_command: String,
    /// Instrucciones de instalacion por OS
    install_commands: HashMap<OperatingSystem, String>,
    /// Es opcional? (el sistema funciona sin el, pero con menos features)
    optional: bool,
}
```

### 20.2 Gestion de Contenedores Docker

```rust
struct ContainerManager {
    /// Contenedores creados por este sistema
    containers: HashMap<String, ContainerRecord>,
    /// Politica de limpieza
    cleanup_policy: ContainerCleanupPolicy,
}

struct ContainerRecord {
    container_id: String,
    name: String,
    image: String,
    /// Agente/sesion que lo creo
    created_by_agent: Option<String>,
    created_by_session: Option<String>,
    created_at: u64,
    /// Estado actual
    status: ContainerStatus,
    /// Puertos mapeados
    ports: Vec<(u16, u16)>,
}

struct ContainerCleanupPolicy {
    /// Max contenedores por sesion
    max_per_session: usize,       // e.g. 5
    /// Max contenedores totales
    max_total: usize,             // e.g. 20
    /// Auto-eliminar contenedores parados despues de X horas
    auto_remove_stopped_hours: Option<u64>,
    /// Alertar cuando se supere este numero
    warn_threshold: usize,        // e.g. 10
}
```

### 20.3 Tools de Contenedores

| Tool | Descripcion | Riesgo | Permisos |
|------|-------------|--------|----------|
| `container_create` | Crear contenedor | MEDIO | Pedir permiso |
| `container_start` | Arrancar contenedor existente | BAJO | Auto si creado por el agente |
| `container_stop` | Parar contenedor | BAJO | Auto si creado por el agente |
| `container_remove` | Eliminar contenedor | MEDIO | Auto si creado por el agente |
| `container_exec` | Ejecutar comando dentro del contenedor | MEDIO | Auto si creado por el agente |
| `container_logs` | Ver logs del contenedor | BAJO | Auto |
| `container_list` | Listar contenedores | BAJO | Auto |
| `container_export` | Exportar contenedor (tar/image) | BAJO | Auto |
| `container_import` | Importar contenedor | MEDIO | Pedir permiso |
| `container_share` | Compartir contenedor entre sesiones | MEDIO | Pedir permiso |

**Regla de permisos**: Un contenedor creado por un agente puede ser usado por ESE agente sin pedir permiso adicional. Compartir con otro agente/sesion requiere permiso.

### 20.4 Gestion de Servicios (Ollama, etc.)

| Tool | Descripcion | Riesgo |
|------|-------------|--------|
| `service_status` | Ver si Ollama/LM Studio/etc. esta corriendo | BAJO |
| `service_start` | Arrancar servicio | MEDIO |
| `service_stop` | Parar servicio | MEDIO |
| `service_restart` | Reiniciar servicio | MEDIO |
| `service_install` | Instalar servicio (prerequisitos) | ALTO |
| `service_logs` | Ver logs del servicio | BAJO |

### 20.5 Registro de Paquetes Instalados (InstalledPackageRegistry)

TODO lo que el sistema instala en la maquina del usuario se registra.
Esto permite saber que se ha hecho, cuanto espacio ocupa, y desinstalarlo.

```rust
struct InstalledPackageRegistry {
    /// Directorio donde se guarda el registro
    registry_path: PathBuf,
    /// Paquetes instalados
    packages: Vec<InstalledPackage>,
}

struct InstalledPackage {
    /// Nombre unico del paquete
    name: String,
    /// Version instalada
    version: String,
    /// Tipo de paquete
    kind: PackageKind,
    /// Tamano en disco (bytes)
    size_bytes: u64,
    /// Paths en disco que ocupa (para desinstalacion)
    installed_paths: Vec<PathBuf>,
    /// Como se instalo
    install_method: InstallMethod,
    /// Quien pidio la instalacion (agente, sesion, usuario directo)
    installed_by: String,
    installed_at: u64,
    /// Es desinstalable automaticamente?
    can_uninstall: bool,
    /// Comando de desinstalacion (si existe)
    uninstall_command: Option<String>,
    /// Otros paquetes que dependen de este
    depended_on_by: Vec<String>,
}

enum PackageKind {
    /// Browser headless (Chromium, Firefox)
    BrowserEngine,
    /// Runtime (Node.js, Python, etc.)
    Runtime,
    /// Servicio (Ollama, Docker, etc.)
    Service,
    /// Herramienta CLI (git, cargo, etc.)
    CliTool,
    /// Libreria o driver (CUDA toolkit, etc.)
    Library,
    /// Modelo de IA (descargado por Ollama, etc.)
    AiModel,
    /// Otro
    Other,
}

enum InstallMethod {
    /// Via package manager del sistema (apt, brew, choco)
    SystemPackageManager { manager: String, package_name: String },
    /// Descarga directa + extraccion
    DirectDownload { url: String },
    /// Via otro tool (npx playwright install, ollama pull, etc.)
    ViaTool { tool: String, command: String },
    /// Compilacion desde fuente
    FromSource { repo: String },
}
```

**Tools del registro**:

| Tool | Descripcion | Riesgo | Permisos |
|------|-------------|--------|----------|
| `package_list` | Listar todo lo instalado por este sistema (nombre, tamano, fecha) | BAJO | Auto |
| `package_info` | Detalle de un paquete (paths, dependencias, metodo de install) | BAJO | Auto |
| `package_uninstall` | Desinstalar un paquete instalado por este sistema | ALTO | Pedir permiso |
| `package_disk_usage` | Espacio total en disco usado por paquetes del sistema | BAJO | Auto |
| `package_cleanup` | Sugerir paquetes que no se usan y podrian eliminarse | BAJO | Auto |

**Regla**: Solo se pueden desinstalar paquetes que fueron instalados POR ESTE SISTEMA
(marcados como `installed_by: "ai_assistant"`). Nunca desinstalar software que el usuario
instalo por su cuenta.

**Integracion con Diagnostics (seccion 19)**: El diagnostics tool lista los paquetes
instalados, su tamano total, y sugiere limpiar los que no se usan.

---

## 21. Deteccion de Trabajo Incompleto y Registro de Requisitos

### 21.1 Problema

Los LLMs frecuentemente dejan cosas a medias:
- TODOs en el codigo
- Funciones vacias o con placeholders
- Integraciones mencionadas pero no implementadas
- Tests que faltan
- Imports sin usar
- Archivos creados pero no registrados

### 21.2 Registro de Requisitos (RequirementRegistry)

Antes de que el agente empiece a trabajar, se construye un registro de requisitos
extraidos automaticamente de multiples fuentes. Este registro es el contrato que
el CompletionVerifier usara para verificar que la tarea esta REALMENTE completa.

```rust
struct RequirementRegistry {
    requirements: Vec<Requirement>,
    /// Historial de contradicciones detectadas y su resolucion
    contradiction_log: Vec<ContradictionRecord>,
}

struct Requirement {
    id: String,
    description: String,
    /// De donde sale este requisito
    source: RequirementSource,
    /// Como verificar que se cumple
    verification: Vec<VerifyMethod>,
    /// El usuario lo confirmo explicitamente?
    confirmed: bool,
    /// Prioridad
    priority: Priority, // Must, Should, Nice
    /// Timestamp de cuando se agrego
    created_at: u64,
    /// Si fue reemplazado por otro requisito (contradiccion resuelta)
    superseded_by: Option<String>,
}

enum RequirementSource {
    /// El usuario lo pidio en la conversacion
    UserRequest {
        session_id: String,
        /// El mensaje original del usuario (evidencia)
        original_text: String,
    },
    /// Esta documentado en un archivo del proyecto
    Documentation {
        file: PathBuf,
        /// Seccion o linea relevante
        section: String,
        /// Texto literal de la documentacion
        evidence: String,
    },
    /// Inferido del codigo existente (ej: "hay tests -> no romperlos")
    CodeConvention {
        /// Que se observo en el codigo
        evidence: String,
        /// Archivos donde se observo el patron
        files: Vec<PathBuf>,
    },
    /// De una sesion anterior del usuario
    PriorSession {
        session_id: String,
        summary: String,
    },
    /// "Sentido comun" de desarrollo de software (ver 21.3)
    CommonSense {
        category: CommonSenseCategory,
        reasoning: String,
    },
    /// Obtenido de fuentes externas (internet, knowledge graph)
    ExternalKnowledge {
        source_url: Option<String>,
        graph_node: Option<String>,
        evidence: String,
    },
    /// Inferido por el LLM (necesita confirmacion si es ambiguo)
    Inferred {
        reasoning: String,
        confidence: f32, // 0.0-1.0
        /// Solo preguntar si confidence < umbral
        needs_confirmation: bool,
    },
}

enum VerifyMethod {
    /// Buscar patron en archivos
    GrepFor { pattern: String, files: Vec<String> },
    /// Verificar que compila
    CompileCheck,
    /// Verificar que existe un test con este patron de nombre
    TestExists { test_name_pattern: String },
    /// Ejecutar comando y verificar exit code 0
    CustomCommand { command: String },
    /// Preguntarle al LLM "el codigo en X realmente cumple Y?"
    AskLlm { question: String },
    /// Verificar que una dependencia esta en Cargo.toml / package.json
    DependencyExists { name: String },
    /// Verificar que un servicio responde (ej: Ollama en :11434)
    ServiceResponds { url: String },
}

enum Priority { Must, Should, Nice }
```

### 21.3 Requisitos de "Sentido Comun"

Ademas de lo que el usuario pide explicitamente y lo que esta en la documentacion,
hay requisitos que cualquier desarrollador competente aplicaria. Estos se inyectan
automaticamente segun el contexto de la tarea:

```rust
enum CommonSenseCategory {
    /// Contrasenas y secretos siempre cifrados, nunca en plaintext
    SecuritySecrets,
    /// Claves privadas protegidas (permisos de archivo, no en repos)
    SecurityKeys,
    /// Input validation en fronteras del sistema (user input, APIs externas)
    InputValidation,
    /// Logs con timestamp en formato estandar (ISO 8601 o configurable)
    LogFormat,
    /// Dependencias de servicios: si necesitas X, verificar que X esta instalado
    ServiceDependencies,
    /// No hardcodear paths, URLs, credenciales
    NoHardcoding,
    /// Cerrar recursos (archivos, conexiones, handles)
    ResourceCleanup,
    /// Encoding correcto (UTF-8 por defecto)
    Encoding,
    /// Manejo de errores: no swallow silenciosamente
    ErrorHandling,
    /// Thread safety cuando hay concurrencia
    ThreadSafety,
}
```

**Ejemplo**: Si el agente va a implementar autenticacion:

```
Requisitos automaticos de sentido comun:
  [CommonSense::SecuritySecrets] Las contrasenas se guardan hasheadas (bcrypt/argon2), nunca plaintext
  [CommonSense::SecurityKeys]    Los JWT secrets no van en el codigo fuente
  [CommonSense::InputValidation] Se valida formato de email, longitud de contrasena
  [CommonSense::LogFormat]       Los logs de auth incluyen timestamp ISO 8601
  [CommonSense::NoHardcoding]    Token expiry time es configurable, no hardcoded
```

Estos requisitos se aplican automaticamente SIN preguntar al usuario (son universales).
Si alguno entra en conflicto con un requisito del usuario, se detecta y se pregunta (ver 21.5).

### 21.4 Enriquecimiento desde Fuentes Externas

Los requisitos de sentido comun se pueden enriquecer buscando en fuentes externas:

```
Tarea: "implementar autenticacion JWT"
     |
     v
Buscar en internet (si configurado):
  - "JWT security best practices 2026"
  - OWASP cheat sheet para JWT
     |
     v
Buscar en knowledge graph local (si existe):
  - Nodos relacionados con "JWT", "authentication"
  - Patrones de seguridad almacenados de sesiones previas
     |
     v
Nuevos requisitos descubiertos:
  [ExternalKnowledge] "Usar algoritmo RS256 en vez de HS256 para multi-servicio"
  [ExternalKnowledge] "Incluir claim 'jti' para revocacion de tokens"
  [ExternalKnowledge] "Refresh tokens: rotacion en cada uso (one-time use)"
     |
     v
Estos son Inferred con needs_confirmation=true
  -> Preguntar al usuario: "He encontrado estas best practices. Quieres aplicarlas?"
```

**Configuracion**: El enriquecimiento externo es opcional y configurable:

```rust
struct RequirementEnrichmentConfig {
    /// Buscar best practices en internet
    search_web: bool,
    /// Buscar en knowledge graph local
    search_knowledge_graph: bool,
    /// Buscar en sesiones previas del proyecto
    search_prior_sessions: bool,
    /// Categorias de sentido comun activas (todas por defecto)
    active_common_sense: Vec<CommonSenseCategory>,
    /// Umbral de confianza para auto-aplicar sin preguntar (>0.9 = no preguntar)
    auto_apply_confidence_threshold: f32,
}
```

### 21.5 Deteccion de Contradicciones

Cuando un requisito nuevo contradice a uno previo, hay que detectarlo y resolverlo:

```rust
struct ContradictionRecord {
    /// Los dos requisitos que se contradicen
    requirement_a: String, // id
    requirement_b: String, // id
    /// Descripcion del conflicto
    description: String,
    /// Como se resolvio
    resolution: ContradictionResolution,
    timestamp: u64,
}

enum ContradictionResolution {
    /// El usuario eligio cual tiene prioridad
    UserDecision { chosen: String, reason: String },
    /// El mas reciente gana (si ambos del usuario)
    NewerWins,
    /// Se fusionaron en un nuevo requisito
    Merged { new_requirement_id: String },
    /// Sin resolver (se preguntara al usuario)
    Pending,
}
```

**Ejemplos de contradicciones**:

```
CONFLICTO 1: Requisito directo vs requisito directo
  R1 [UserRequest sesion 5]:  "Los logs deben ser en formato JSON"
  R2 [UserRequest sesion 12]: "Los logs deben ser plain text legibles"
  -> Preguntar: "En la sesion 5 pediste logs JSON, ahora pides plain text. Cual prefieres?"

CONFLICTO 2: Requisito de usuario vs sentido comun
  R1 [UserRequest]:     "Guarda las contrasenas en un archivo de texto para debug"
  R2 [CommonSense]:     "Las contrasenas nunca en plaintext"
  -> Advertir: "Guardar contrasenas en texto plano contradice buenas practicas de
     seguridad. Si quieres continuar, lo hare con un warning en el codigo.
     Alternativa: guardar un hash y loguear solo los primeros 4 chars."

CONFLICTO 3: Documentacion vs codigo existente
  R1 [Documentation]:   README dice "usar REST API"
  R2 [CodeConvention]:  El codigo actual usa GraphQL
  -> Informar: "El README dice REST pero el codigo usa GraphQL. Cual sigo?"
```

### 21.6 Deteccion de Ciclos

Problema: el agente puede entrar en ciclos donde aplica un cambio, otro requisito lo deshace,
y el primero lo vuelve a aplicar. Esto puede pasar con ciclos de mas de 1 paso (A->B->C->A).

```rust
struct CycleDetector {
    /// Historial de cambios (hash del estado de archivos modificados)
    state_history: Vec<StateSnapshot>,
    /// Ventana de deteccion (cuantos estados atras mirar)
    window_size: usize, // default: 10
}

struct StateSnapshot {
    /// Hash de los archivos modificados
    file_hashes: HashMap<PathBuf, u64>,
    /// Que requisito se estaba cumpliendo
    requirement_id: String,
    /// Que cambios se hicieron
    changes_summary: String,
    timestamp: u64,
}

impl CycleDetector {
    /// Detectar si el estado actual ya se vio antes
    fn detect_cycle(&self, current: &StateSnapshot) -> Option<CycleInfo>;

    /// Detectar ciclos de N pasos (A->B->C->A)
    fn detect_multi_step_cycle(&self, window: usize) -> Option<CycleInfo>;
}

struct CycleInfo {
    /// Estados que forman el ciclo
    cycle_states: Vec<StateSnapshot>,
    /// Requisitos involucrados en el ciclo
    conflicting_requirements: Vec<String>,
    /// Numero de veces que se ha repetido el ciclo
    repetitions: usize,
}
```

**Flujo cuando se detecta un ciclo**:

```
Agente aplica cambio X por requisito R1
     |
     v
Agente aplica cambio Y por requisito R2 (deshace parte de X)
     |
     v
Agente aplica cambio Z por requisito R1 (similar a X)
     |
     v
CycleDetector: "Ciclo detectado entre R1 y R2 (2 repeticiones)"
     |
     v
PARAR al agente. Reportar al usuario:
  "He detectado un conflicto ciclico:
   - R1 pide: 'usar tabs para indentacion' (fuente: .editorconfig)
   - R2 pide: 'formatear con rustfmt' (fuente: CI pipeline)
   - rustfmt convierte tabs a espacios, lo que contradice R1
   Por favor, decide cual tiene prioridad."
     |
     v
La resolucion se registra en ContradictionRecord
para no volver a caer en el mismo ciclo
```

### 21.7 Checks Genericos (Post-Task)

Ademas de verificar los requisitos especificos, siempre se ejecutan estos checks genericos:

```rust
enum CompletionCheck {
    /// Buscar TODOs, FIXMEs, HACK, XXX en archivos modificados
    SearchTodos { patterns: Vec<String> },
    /// Buscar funciones vacias o con "unimplemented!()", "todo!()"
    EmptyFunctions,
    /// Buscar placeholders ("placeholder", "lorem ipsum", "xxx", "...")
    Placeholders { patterns: Vec<String> },
    /// Compilar y verificar que no hay errores
    CompileCheck { command: String },
    /// Ejecutar tests y verificar que pasan
    TestCheck { command: String },
    /// Verificar que no hay warnings nuevos
    WarningCheck { command: String },
    /// Verificar que archivos nuevos estan registrados (en lib.rs, mod.rs, etc.)
    ModuleRegistration,
    /// Verificar imports sin usar
    UnusedImports,
    /// Verificar que la documentacion mencionada existe
    DocReferences,
    /// Custom: ejecutar script de verificacion
    CustomScript { command: String },
}
```

### 21.8 Flujo Completo

```
1. Usuario pide una tarea
        |
        v
2. Construir RequirementRegistry:
   a. Parsear peticion del usuario -> RequirementSource::UserRequest
   b. Escanear documentacion del proyecto -> RequirementSource::Documentation
   c. Analizar codigo existente -> RequirementSource::CodeConvention
   d. Buscar en sesiones previas -> RequirementSource::PriorSession
   e. Aplicar sentido comun segun contexto -> RequirementSource::CommonSense
   f. (Opcional) Enriquecer desde internet/knowledge graph -> RequirementSource::ExternalKnowledge
   g. LLM infiere requisitos implicitos -> RequirementSource::Inferred
        |
        v
3. Detectar contradicciones entre requisitos nuevos y previos
   - Nuevos vs previos de sesiones anteriores
   - Nuevos vs documentacion
   - Nuevos vs sentido comun
   - Preguntar al usuario si hay conflictos
        |
        v
4. Presentar requisitos al usuario:
   "Voy a implementar X. Estos son los requisitos que voy a cumplir:
    [Must] R1: ... (pedido por ti)
    [Must] R2: ... (en DESIGN.md)
    [Must] R3: ... (sentido comun: cifrar passwords)
    [Should] R4: ... (inferido de tu codigo, confianza 85%)
    [Nice] R5: ... (best practice encontrada online)
    Tengo una duda: R4 dice X pero R2 dice Y. Cual prefieres?"
        |
        v
5. El agente trabaja (con CycleDetector activo)
        |
        v
6. Agente declara "tarea completada"
        |
        v
7. CompletionVerifier ejecuta:
   a. Verificar CADA requisito del registro contra su VerifyMethod
   b. Ejecutar checks genericos (CompileCheck, TestCheck, SearchTodos, etc.)
   c. Verificar checklist segun tipo de tarea
        |
        v
8. Problemas encontrados?
   ---No---> Realmente completada
   ---Si---> Reportar al agente con detalle:
     "Has dejado 3 TODOs y 1 funcion vacia.
      Requisito R3 (cifrar passwords) NO cumplido: grep 'plaintext' encontro match.
      2 warnings de compilacion nuevos.
      Por favor, completa estos pendientes."
        |
        v
9. Volver a paso 5 (con CycleDetector monitorizando)
```

### 21.9 Checklist por Tipo de Tarea

| Tipo de tarea | Checks obligatorios |
|--------------|-------------------|
| Nueva feature | CompileCheck + TestCheck + SearchTodos + EmptyFunctions + ModuleRegistration + RequirementVerify |
| Bug fix | CompileCheck + TestCheck + WarningCheck + RequirementVerify |
| Refactoring | CompileCheck + TestCheck + UnusedImports + RequirementVerify |
| Documentacion | DocReferences + Placeholders + RequirementVerify |
| Cualquiera | SearchTodos + RequirementVerify (siempre) |

---

## 22. Catalogo de Interfaces a Servicios Externos

Recopilacion de TODAS las interfaces/APIs a servicios externos encontradas en los proyectos analizados y en nuestro crate.

### 22.1 Proveedores LLM

| Servicio | Protocolo | Endpoint | Auth | Estado en nuestro crate |
|----------|-----------|----------|------|------------------------|
| **Ollama** | REST (nativo) | `http://localhost:11434` | Ninguna | IMPLEMENTADO |
| **LM Studio** | REST (OpenAI-compat) | `http://localhost:1234` | Ninguna | IMPLEMENTADO |
| **text-generation-webui** | REST (OpenAI-compat) | `http://localhost:5000` | Ninguna | IMPLEMENTADO |
| **Kobold.cpp** | REST (nativo) | `http://localhost:5001` | Ninguna | IMPLEMENTADO |
| **LocalAI** | REST (OpenAI-compat) | `http://localhost:8080` | Ninguna | IMPLEMENTADO |
| **OpenAI** | REST | `https://api.openai.com/v1` | Bearer token | IMPLEMENTADO |
| **Anthropic** | REST | `https://api.anthropic.com/v1` | x-api-key header | IMPLEMENTADO |
| **Google Gemini** | REST | `https://generativelanguage.googleapis.com` | API key | PENDIENTE |
| **Mistral** | REST (OpenAI-compat) | `https://api.mistral.ai/v1` | Bearer token | PENDIENTE (compatible) |
| **Groq** | REST (OpenAI-compat) | `https://api.groq.com/openai/v1` | Bearer token | PENDIENTE (compatible) |
| **Together AI** | REST (OpenAI-compat) | `https://api.together.xyz/v1` | Bearer token | PENDIENTE (compatible) |
| **OpenRouter** | REST (OpenAI-compat) | `https://openrouter.ai/api/v1` | Bearer token | PENDIENTE (compatible) |
| **Cohere** | REST | `https://api.cohere.ai/v1` | Bearer token | PENDIENTE |
| **DeepSeek** | REST (OpenAI-compat) | `https://api.deepseek.com/v1` | Bearer token | PENDIENTE (compatible) |
| **Perplexity** | REST (OpenAI-compat) | `https://api.perplexity.ai` | Bearer token | PENDIENTE (compatible) |
| **AWS Bedrock** | AWS SDK | Regional endpoint | IAM/SigV4 | PENDIENTE |
| **Azure OpenAI** | REST | `https://{resource}.openai.azure.com` | api-key header | PENDIENTE |
| **Fireworks AI** | REST (OpenAI-compat) | `https://api.fireworks.ai/inference/v1` | Bearer token | PENDIENTE (compatible) |

### 22.2 Servicios de Embedding

| Servicio | Endpoint | Estado |
|----------|----------|--------|
| Ollama embeddings | `POST /api/embeddings` | Interno (RAG) |
| OpenAI embeddings | `POST /v1/embeddings` | PENDIENTE |
| Cohere embed | `POST /v1/embed` | PENDIENTE |
| Local (TF-IDF) | In-memory | IMPLEMENTADO |

### 22.3 Plataformas de Mensajeria (de OpenClaw)

| Servicio | Protocolo | Estado |
|----------|-----------|--------|
| WhatsApp | Baileys/Web API | PENDIENTE |
| Telegram | Bot API | PENDIENTE |
| Slack | Bolt/Web API | PENDIENTE |
| Discord | Discord.js/Gateway | PENDIENTE |
| Matrix | Matrix SDK | PENDIENTE |
| WebChat | WebSocket | IMPLEMENTADO (server.rs) |

### 22.4 Tools Externos / Integraciones

| Servicio | Protocolo | Uso | Estado |
|----------|-----------|-----|--------|
| GitHub API | REST | Issues, PRs, code search | PENDIENTE |
| GitLab API | REST | Issues, MRs | PENDIENTE |
| Jira | REST | Tickets | PENDIENTE |
| Linear | GraphQL | Tasks | PENDIENTE |
| Google Safe Browsing | REST | URL safety | PENDIENTE |
| VirusTotal | REST | URL/file safety | PENDIENTE |
| Docker Engine | REST/Socket | Container management | PENDIENTE |

### 22.5 Nota: OpenAI-Compatible Significa Facil

Los proveedores marcados como "OpenAI-compat" usan el mismo formato de request/response que OpenAI (`/v1/chat/completions`). Nuestro crate ya soporta el formato OpenAI via `AiProvider::OpenAICompatible { base_url }`, asi que estos proveedores funcionan HOY cambiando solo la URL y el API key. Lo que falta es:
- Registro explicito en `AiProvider` enum (para auto-detect y UX)
- URLs y modelos por defecto
- Tests de integracion

---

## 23. Funcionalidades No Incluidas en el Plan Actual

Cosas que los proyectos analizados tienen y que NO estan en el plan de las secciones 1-12:

### Alta prioridad (deberian estar)

| Funcionalidad | Origen | Descripcion | Por que incluir |
|---------------|--------|-------------|----------------|
| **Repo map** | Aider | Mapa de funciones/clases del codebase para contexto | Reduce tokens, mejora precision |
| **Lint post-edicion** | SWE-agent | Verificar codigo despues de cada edit | Evita codigo roto |
| **Commit-por-cambio** | Aider | Cada cambio del agente = commit git automatico | Todo es reversible |
| **Deteccion de loops** | AutoGPT, CrewAI | Detectar y romper loops del agente | Ahorra tokens/coste |
| **Memoria episodica** | AutoGPT | Recordar que funciono y que fallo en sesiones anteriores | Mejora con el tiempo |
| **Estrategias de iteracion** | Propio | Reemplazar max_iterations con estrategias inteligentes | Seccion 14 |
| **Anonimizacion por proveedor** | Propio | Proteger datos segun confianza del proveedor | Seccion 15 |
| **Sistema de undo/rollback** | Claude Code | Safe delete, papelera, undo de operaciones | Seccion 16 |
| **HTTP anti-bloqueo** | Propio | User-Agent rotation, headers de browser | Seccion 17 |
| **Plugin system para VCS** | Propio | Extensibilidad para SVN, Hg, etc. | Seccion 18 |
| **Diagnostics tool** | Propio | Introspeccion del sistema, warnings, suggestions | Seccion 19 |
| **Gestion de contenedores** | E2B, Goose | Docker lifecycle, prerequisitos, compartir | Seccion 20 |
| **Verificacion post-tarea** | Propio | Detectar TODOs, funciones vacias, tests rotos | Seccion 21 |
| **Catalogo de interfaces** | Todos | Soporte para 20+ LLM providers | Seccion 22 |

### Media prioridad (utiles pero no criticas)

| Funcionalidad | Origen | Descripcion |
|---------------|--------|-------------|
| **Skills declarativas** | Claude Code | Archivos markdown que ensenan capacidades al agente |
| **Self-healing** | Stagehand | Detectar cuando algo cambio y adaptar |
| **Auto-caching de acciones** | Stagehand | Cachear acciones repetidas para no gastar tokens |
| **Edit formats por modelo** | Aider | Cada LLM trabaja mejor con un formato de edicion |
| **SOP (Standard Operating Procedures)** | MetaGPT | Procedimientos formales por agente |
| **Agent handoffs** | OpenAI SDK | Transferir control entre agentes naturalmente |
| **Voice input** | Aider | Comandos por voz |
| **Image context** | Aider | Enviar screenshots/imagenes como contexto |
| **Multi-repo support** | OpenHands | Trabajar en multiples repos a la vez |
| **Cost tracking real-time** | AutoGPT | Mostrar coste acumulado de tokens durante ejecucion |

### Baja prioridad (futuro lejano)

| Funcionalidad | Origen | Descripcion detallada |
|---------------|--------|----------------------|
| **Firecracker microVMs** | E2B | Micro-maquinas virtuales (la misma tecnologia de AWS Lambda). Arrancan en ~125ms con kernel propio. Aislamiento a nivel de kernel, imposible escapar al host. Mas seguro que Docker pero requiere Linux con KVM. Overkill para la mayoria de casos, ideal para ejecutar codigo no confiable de terceros |
| **WASM agents** | Rig | Compilar agentes a WebAssembly para que corran en el navegador sin backend. Caso de uso: chatbot 100% frontend. Limitacion: WASM no tiene acceso a filesystem ni red directa, asi que solo sirve para agentes conversacionales puros |
| **E2B Desktop** | E2B | Variante de E2B que da un escritorio virtual completo (pantalla, raton, teclado) dentro del sandbox. El agente puede abrir aplicaciones GUI (Excel, IDEs, etc.) e interactuar visualmente. Muy experimental, cold start ~5s |
| **Managed OAuth** | Composio | Composio es un servicio **en la nube (SaaS)**, NO una libreria local. Gestiona tokens OAuth para 800+ servicios (Gmail, Slack, GitHub, Jira, Salesforce...). El usuario autoriza una vez, Composio guarda y renueva tokens. Ventaja: no implementar OAuth con cada servicio. Desventaja: dependencia de un tercero para credenciales. **No existe equivalente self-hosted maduro** — alternativas parciales: Nango (open-source, self-hostable pero menor cobertura) o implementar OAuth directamente por servicio |
| **Assembly-line paradigm** | MetaGPT | Simula una empresa de software: Product Manager -> Architect -> Engineer -> QA, cada agente pasa su output al siguiente como cadena de montaje. Problema: errores se acumulan de agente en agente. **Ver seccion 25 para una version mejorada con back-communication** |
| **AFlow** | MetaGPT | Meta-programacion de agentes: usa busqueda Monte Carlo para generar automaticamente workflows de agentes. Le dices "quiero un agente que haga X" y AFlow prueba combinaciones de nodos (LLM calls, tools, decisiones) hasta encontrar un workflow que funcione bien en un benchmark. Muy experimental, resultados inconsistentes |

### Datos de Benchmarks y Costes por Proyecto

| Proyecto | Benchmark | Score | Coste Tipico | Leccion |
|----------|-----------|-------|-------------|---------|
| Claude Code | SWE-bench Verified | 79% (Opus 4.6) | $20-200/mo | El mejor, pero caro |
| Aider | - | No publica benchmarks | $0-50/mo (con Ollama: $0) | Mejor coste-beneficio |
| SWE-agent | SWE-bench Verified | 74% (mini) | Coste del LLM | Simple > complejo |
| SWE-agent | SWE-bench Pro (real) | 23% maximo | Coste del LLM | Benchmarks ≠ realidad |
| OpenHands | SWE-bench Lite | 26% | Coste del LLM | Inestable, alpha |
| AutoGPT | - | No estandarizado | $50-500+/mo | Loops queman tokens |
| CrewAI | - | No estandarizado | Variable | Context overflow = crash |
| Goose | - | 3 semanas en 1 | Coste del LLM | MCP escala |
| browser-use | WebVoyager | ~89% controlado | Coste del LLM | Hibrido > puro AI |
| E2B | - | ~150ms cold start | $0.05/sandbox-hora | Infraestructura, no agente |

**Lecciones de coste**:
- Los loops de AutoGPT pueden costar $500+/mes sin producir resultados utiles
- Aider con modelos locales (Ollama) = $0 de coste de API
- El anti-patron mas caro: agente autonomo sin limites en provider de pago
- El patron mas eficiente: estrategia hibrida (local para exploracion, cloud para generacion final)

---

## 24. Estrategia Hibrida Local/Cloud

### 24.1 Concepto

No usar el mismo modelo para todo. Usar modelos locales (gratis) para tareas de exploracion
y verificacion, y modelos cloud (potentes, de pago) solo cuando se necesita calidad maxima.
Esto reduce costes drasticamente sin perder calidad en el resultado final.

### 24.2 Fases de Ejecucion

```
Tarea del agente: "Refactoriza el modulo de autenticacion"
        |
        v
  Fase 1: EXPLORACION (modelo local — gratis)
    - Leer archivos del proyecto
    - Buscar funciones, dependencias, patrones
    - Entender la estructura
    - Generar un plan borrador
    - Hacer preguntas al usuario
    -> Coste: $0 (corre en la maquina del usuario)
        |
        v
  Fase 2: GENERACION (modelo cloud — potente, de pago)
    - Ya con el contexto minimo necesario (no todo el repo)
    - Generar el codigo final
    - Un solo call (o pocos) al modelo potente
    -> Coste: solo los tokens estrictamente necesarios
        |
        v
  Fase 3: VERIFICACION (modelo local — gratis)
    - Revisar el codigo generado
    - Ejecutar CompletionVerifier (seccion 21)
    - Compilar, tests, checks
    -> Coste: $0
        |
        v
  Fase 4: CORRECCION (decision automatica)
    - Si los errores son simples -> modelo local corrige (gratis)
    - Si los errores son complejos -> modelo cloud corrige (pago, pero necesario)
    - Si se detecta enroque/estancamiento -> escalar a modelo MAS avanzado
```

### 24.3 Configuracion

```rust
struct HybridStrategy {
    /// Modelo para exploracion (barato/local)
    exploration_provider: (AiProvider, String),
    /// Modelo para generacion (potente/cloud)
    generation_provider: (AiProvider, String),
    /// Modelo para verificacion (barato/local)
    verification_provider: (AiProvider, String),
    /// Cuando escalar de local a cloud
    escalation_trigger: EscalationTrigger,
    /// Modelo de escalacion extra (mas potente que generation, para desbloqueo)
    escalation_provider: Option<(AiProvider, String)>,
}

enum EscalationTrigger {
    /// Escalar cuando el modelo local no puede resolver
    OnLocalFailure,
    /// Escalar solo para la fase de generacion de codigo
    GenerationPhaseOnly,
    /// Escalar segun complejidad estimada de la tarea
    ComplexityBased { threshold: f32 },
    /// Escalar cuando se detecta enroque (stalemate)
    OnStalemate { max_attempts: usize },
    /// El usuario decide manualmente
    Manual,
}
```

### 24.4 Deteccion de Enroque y Escalacion

Integrado con CompletionVerifier y CycleDetector (seccion 21.6):

```
CompletionVerifier falla por 3ra vez con el mismo error
     |
     v
CycleDetector confirma: "El agente no esta progresando"
     |
     v
Escalacion automatica:
  Nivel 1: Reintentar con modelo local mas grande
    ej: qwen2.5-coder:7b -> qwen2.5-coder:32b
  Nivel 2: Cambiar a modelo cloud estandar
    ej: -> Claude Sonnet
  Nivel 3: Cambiar a modelo cloud premium
    ej: -> Claude Opus
  Nivel 4: Parar y pedir ayuda al usuario
    "No he podido resolver esto ni con Opus. El problema es: ..."
```

### 24.5 Recomendaciones de Modelos

**Modelos locales (exploracion/verificacion)**:

| Modelo | RAM necesaria | Velocidad | Calidad | Notas |
|--------|--------------|-----------|---------|-------|
| Qwen 2.5 Coder 7B | ~6 GB | Rapido | Buena para code | Mejor relacion calidad/tamanio para codigo |
| Llama 3.1 8B | ~6 GB | Rapido | Buena general | Buen razonamiento general |
| DeepSeek Coder V2 Lite | ~10 GB | Medio | Muy buena | Excelente para codigo, algo mas lento |
| Codestral 22B | ~16 GB | Medio | Excelente | Si tienes RAM, el mejor local para codigo |
| Phi-3 Mini 3.8B | ~3 GB | Muy rapido | Aceptable | Para maquinas con poca RAM |

**Servicios externos (generacion)**:

| Servicio | Modelo recomendado | Coste aprox. | Calidad | Notas |
|----------|-------------------|-------------|---------|-------|
| Anthropic | Claude Sonnet 4.5 | ~$3/M input, $15/M output | Excelente | Mejor relacion calidad/precio para codigo |
| Anthropic | Claude Opus 4.6 | ~$15/M input, $75/M output | Maxima | Solo para escalacion, muy caro |
| OpenAI | GPT-4o | ~$2.50/M input, $10/M output | Muy buena | Alternativa competitiva |
| DeepSeek | DeepSeek V3 | ~$0.27/M input, $1.10/M output | Buena | El mas barato con calidad decente |
| Google | Gemini 2.0 Flash | ~$0.10/M input, $0.40/M output | Buena | Extremadamente barato, ventana 1M tokens |
| Groq | Llama 3.3 70B | ~$0.59/M input, $0.79/M output | Buena | Inferencia ultra-rapida |

**Estrategia recomendada por defecto**:
- Exploracion: Qwen 2.5 Coder 7B (Ollama local)
- Generacion: Claude Sonnet 4.5 (Anthropic)
- Verificacion: Qwen 2.5 Coder 7B (Ollama local)
- Escalacion: Claude Opus 4.6 (solo si Sonnet falla)

**Coste estimado con hibrida vs sin hibrida** (tarea tipica de refactoring):
- Sin hibrida (todo cloud): ~$0.50-2.00 por tarea
- Con hibrida: ~$0.05-0.30 por tarea (80-90% de ahorro)

### 24.6 Integracion con el Sistema

La estrategia hibrida conecta con:
- **Seccion 14** (Iteration strategies): las fases de exploracion local pueden tener mas iteraciones (son gratis)
- **Seccion 15** (Privacy per provider): el modelo local NO necesita anonimizacion (los datos no salen de la maquina)
- **Seccion 21** (CompletionVerifier): la verificacion la hace el modelo local; si falla, escala
- **Task 1.1 del plan original** (FallbackChain): si el cloud falla, fallback a otro cloud, no a local
- **Seccion 19** (Diagnostics): el diagnostics tool reporta que modelos estan disponibles localmente y recomienda la configuracion optima

---

## 25. Assembly-Line con Back-Communication

### 25.1 Problema del Assembly-Line Original (MetaGPT)

El paradigma original es una cadena unidireccional:

```
Product Manager -> Architect -> Engineer -> QA
     (req)          (design)     (code)     (test)
```

Problema: si el QA encuentra un error de diseno, no puede decirle al Architect. Solo
puede decirle al Engineer "arregla esto", pero el Engineer no puede arreglar un error
de diseno. Los errores se acumulan y el resultado es peor que si un solo agente hiciera todo.

### 25.2 Solucion: Back-Communication

Permitir que cualquier agente pueda devolver el trabajo a un agente anterior en la cadena,
con una descripcion del problema encontrado:

```
Product Manager <-> Architect <-> Engineer <-> QA
     (req)          (design)       (code)      (test)
         \___________\___________\__________/
              cualquiera puede escalar hacia arriba
```

```rust
struct AssemblyLine {
    stages: Vec<AgentStage>,
    /// Historial de escalaciones para detectar loops
    escalation_history: Vec<Escalation>,
    /// Maximo de escalaciones antes de pedir ayuda al usuario
    max_escalations: usize, // default: 5
}

struct AgentStage {
    name: String,           // "Architect", "Engineer", "QA"
    agent_config: AgentConfig,
    /// Output que produjo (mutable, se actualiza si hay re-trabajo)
    output: Option<StageOutput>,
    /// Cuantas veces se ha re-ejecutado esta fase
    revision_count: usize,
}

struct Escalation {
    /// Quien encontro el problema
    from_stage: String,     // "QA"
    /// A quien se le devuelve el trabajo
    to_stage: String,       // "Architect"
    /// Descripcion del problema
    problem: String,        // "El diseno no contempla autenticacion multi-factor"
    /// Sugerencia de solucion
    suggestion: Option<String>,
    timestamp: u64,
}

struct StageOutput {
    content: String,
    /// Version del output (incrementa con cada revision)
    version: usize,
    /// Que cambio respecto a la version anterior
    changelog: Option<String>,
}
```

### 25.3 Flujo con Back-Communication

```
1. Product Manager genera requisitos v1
        |
        v
2. Architect genera diseno v1
        |
        v
3. Engineer genera codigo v1
        |
        v
4. QA ejecuta tests
   -> Encuentra: "falta manejo de errores en autenticacion"
   -> Evalua: Es error de codigo (Engineer puede arreglarlo)
        |
        v
5. Engineer genera codigo v2 (arregla manejo de errores)
        |
        v
6. QA ejecuta tests
   -> Encuentra: "el diseno no soporta MFA pero los requisitos lo piden"
   -> Evalua: Es error de diseno (Architect debe arreglarlo)
        |
        v
7. ESCALACION: QA -> Architect
   Problema: "Los requisitos piden MFA pero el diseno no lo contempla"
        |
        v
8. Architect genera diseno v2 (anade MFA)
   -> Notifica al Engineer: "He actualizado el diseno. Cambios: seccion auth con MFA"
        |
        v
9. Engineer genera codigo v3 (implementa MFA segun nuevo diseno)
        |
        v
10. QA ejecuta tests
    -> Todo pasa
    -> COMPLETADO
```

### 25.4 Deteccion de Conflictos en la Cadena

Si una escalacion causa otra escalacion en cadena (ping-pong entre fases), interviene el usuario:

```rust
impl AssemblyLine {
    fn detect_ping_pong(&self) -> Option<PingPongConflict> {
        // Si la misma fase ha sido escalada 3+ veces por el mismo problema
        // O si hay un ciclo A->B->A->B
        // -> PingPongConflict con resumen del problema
    }
}
```

```
Escalacion #1: QA -> Engineer ("falta validacion de input")
Escalacion #2: QA -> Engineer ("sigue faltando validacion")
Escalacion #3: QA -> Engineer ("la validacion no cubre caso X")
     |
     v
PingPongConflict detectado!
     |
     v
Pedir al usuario:
  "El QA y el Engineer llevan 3 intentos sin resolver la validacion de input.
   El QA pide: validacion para caso X, Y, Z
   El Engineer responde: caso X y Y cubiertos pero Z no esta en los requisitos
   -> Quieres anadir Z a los requisitos?
   -> O quieres que el QA lo ignore?
   -> O quieres intervenir manualmente?"
```

### 25.5 Cuando Usar Assembly-Line vs Agente Unico vs Paralelo

| Escenario | Recomendacion |
|-----------|--------------|
| Tarea simple (bug fix, refactor pequeno) | Agente unico. Assembly-line es overhead innecesario |
| Feature compleja con diseno no trivial | Assembly-line. La separacion de concerns mejora la calidad |
| Tarea que requiere conocimiento especializado | Assembly-line. Cada agente puede usar un prompt/modelo diferente |
| Prototipo rapido | Agente unico. Velocidad > calidad |
| Codigo que debe pasar auditoria | Assembly-line con QA riguroso |
| N archivos/modulos independientes | **Paralelo**. Fan-out, cada agente trabaja un modulo |
| Investigacion multi-fuente | **Paralelo**. N agentes buscan en fuentes distintas a la vez |
| Probar N enfoques para un problema | **Paralelo especulativo**. N agentes, se elige el mejor resultado |

### 25.6 Paralelismo en Assembly-Line

El assembly-line no tiene por que ser una cadena lineal. Puede ser un **DAG de stages**
donde las ramas sin dependencia mutua corren en paralelo.

#### 25.6.1 Tres Tipos de Paralelismo

```
TIPO 1: Fan-out dentro de un stage
─────────────────────────────────

  Architect (1 agente)
    genera diseno con 3 modulos independientes
        |          |          |
        v          v          v
  Engineer_A  Engineer_B  Engineer_C    <- 3 agentes en paralelo
   (modulo A)  (modulo B)  (modulo C)
        |          |          |
        └──────────┴──────────┘
                   |
                   v
              QA (1 agente)
         testea todo junto


TIPO 2: DAG de stages (no lineal)
─────────────────────────────────

  Requirements
    |          \
    v           v
  Architect   ResearchAgent     <- paralelo (diseno + investigacion)
    |           |
    v           |
  Engineer <────┘               <- espera ambos
    |
    v
  QA + DocWriter                <- paralelo (test + documentacion)


TIPO 3: Pipeline parallelism
─────────────────────────────

  Tiempo ->
  t1: [Engineer: modulo A]
  t2: [QA: modulo A]  [Engineer: modulo B]       <- solapado
  t3: [QA: modulo B]  [Engineer: modulo C]       <- solapado
  t4: [QA: modulo C]

  El QA empieza a testear modulos terminados mientras
  el Engineer sigue implementando los siguientes.
```

#### 25.6.2 Modelo de Ejecucion Paralela

```rust
/// Estrategia de ejecucion para un assembly-line
enum ExecutionStrategy {
    /// Cadena lineal clasica (A -> B -> C)
    Sequential,
    /// DAG: stages sin dependencia mutua corren en paralelo
    ParallelDAG,
    /// Fan-out: un stage produce N outputs, N agentes procesan en paralelo
    FanOut {
        /// Stage que produce los outputs
        fan_out_stage: String,
        /// Maximo de agentes paralelos
        max_parallel: usize,
    },
    /// Pipeline: solapar stages consecutivos cuando sea posible
    Pipeline,
    /// Combinacion de los anteriores (el sistema elige segun el DAG)
    Auto,
}

/// Configuracion de un stage con soporte paralelo
struct ParallelStageConfig {
    /// Stage base
    stage: AgentStage,
    /// Puede este stage ejecutar N instancias en paralelo?
    parallelizable: bool,
    /// Maximo de instancias paralelas (-1 = sin limite)
    max_instances: Option<usize>,
    /// Stages de los que depende (para el DAG)
    depends_on: Vec<String>,
    /// Puede empezar antes de que sus dependencias terminen completamente?
    /// (pipeline parallelism: empieza cuando hay output parcial)
    accepts_partial_input: bool,
}
```

#### 25.6.3 Fan-Out y Fan-In

```
Fan-out:
  Un stage produce una lista de items independientes.
  El sistema lanza N agentes, uno por item.

  Architect.output = Design {
      modules: [
          Module { name: "auth", spec: "..." },
          Module { name: "api",  spec: "..." },
          Module { name: "db",   spec: "..." },
      ]
  }
       |
       v
  FanOutController:
    1. Detecta que el output tiene items independientes (modules[])
    2. Crea N instancias del siguiente stage (Engineer)
    3. Cada instancia recibe solo su modulo
    4. Lanza en paralelo (hasta max_parallel)
       |            |            |
  Engineer_auth  Engineer_api  Engineer_db

Fan-in:
  Los N resultados se recolectan en uno solo para el siguiente stage.

  Engineer_auth.output = Code { files: [...] }
  Engineer_api.output  = Code { files: [...] }
  Engineer_db.output   = Code { files: [...] }
       |            |            |
       └────────────┴────────────┘
                    |
                    v
  FanInController:
    1. Espera a que todos terminen (o a que los suficientes terminen)
    2. Combina outputs: merged_code = auth + api + db
    3. Detecta conflictos entre outputs (mismos archivos?)
    4. Pasa resultado combinado al siguiente stage
                    |
                    v
               QA (1 agente)
         testea todo el codigo junto
```

```rust
struct FanOutController {
    /// Stage que produce la lista
    source_stage: String,
    /// Stage que se paraleliza
    target_stage: String,
    /// Como partir el output en items
    split_strategy: SplitStrategy,
    /// Politica de espera en fan-in
    fan_in_policy: FanInPolicy,
}

enum SplitStrategy {
    /// Partir por items en una lista (modules, files, endpoints...)
    ByListField { field_name: String },
    /// Partir por secciones del documento
    BySections,
    /// Partir manualmente (el LLM decide)
    LlmDecides,
}

enum FanInPolicy {
    /// Esperar a que TODOS terminen
    WaitAll,
    /// Esperar a que N terminen (descartar el resto)
    WaitN(usize),
    /// Esperar a que el primero termine (carrera, especulativo)
    WaitFirst,
    /// Esperar a que un % termine
    WaitPercent(f32),
}
```

#### 25.6.4 Ejecucion Especulativa

Para problemas donde no esta claro cual es el mejor enfoque, lanzar N agentes
con estrategias diferentes y elegir el mejor resultado:

```
Problema: "Optimiza la funcion parse_config()"

  Agente_A: enfoque funcional (iteradores, map/filter)
  Agente_B: enfoque imperativo (loops, variables mutables)
  Agente_C: enfoque con regex (si aplica)
       |            |            |
       v            v            v
  Resultado_A    Resultado_B   Resultado_C
       |            |            |
       └────────────┴────────────┘
                    |
                    v
  Evaluador (QA o LLM):
    - Ejecuta benchmarks (si hay)
    - Compara legibilidad, complejidad ciclomatica
    - Compara tamano del cambio
    -> Elige Resultado_B (mejor rendimiento + legible)
    -> Descarta A y C
```

```rust
struct SpeculativeExecution {
    /// Problema a resolver
    problem: String,
    /// Enfoques a probar
    approaches: Vec<SpeculativeApproach>,
    /// Como evaluar los resultados
    evaluator: EvaluationMethod,
    /// Maximo de enfoques paralelos
    max_parallel: usize,
}

struct SpeculativeApproach {
    name: String,
    /// Instrucciones especificas para este enfoque
    prompt_override: String,
    /// Modelo a usar (puede variar por enfoque)
    model: Option<String>,
}

enum EvaluationMethod {
    /// El LLM compara los resultados
    LlmComparison,
    /// Ejecutar tests/benchmarks y comparar metricas
    MetricsBased { metrics: Vec<String> },
    /// El usuario elige
    UserChoice,
    /// Combinacion: metricas + LLM + usuario como tiebreaker
    Combined,
}
```

#### 25.6.5 Limites y Riesgos del Paralelismo

| Riesgo | Mitigacion |
|--------|-----------|
| **Conflicto de archivos**: dos agentes editan el mismo archivo | `file_ownership` index (seccion 16.3). Antes de lanzar, verificar que no haya solapamiento. Si lo hay, serializar esas tareas |
| **Consumo de recursos**: N agentes = N x tokens/seg | `max_parallel` configurable. Default conservador (2-3). En local: limitado por VRAM. En cloud: limitado por presupuesto |
| **Coherencia**: N agentes toman decisiones incompatibles | Fan-in con verificacion de coherencia. Si hay contradiccion, resolver antes de continuar |
| **Complejidad de undo**: deshacer N cambios paralelos | Cada agente tiene su UndoLog aislado. El `TaskUndoManager` detecta conflictos entre logs (seccion 16.3) |
| **Deadlocks en DAG**: dependencias circulares | Validacion del DAG antes de lanzar. Si hay ciclo, error inmediato |
| **Coste descontrolado**: especulativo multiplica el coste x N | Presupuesto maximo para ejecucion especulativa. Cancelar agentes lentos si uno ya termino bien |

#### 25.6.6 Cuando Paralelizar

```
El sistema decide automaticamente si paralelizar:

1. Hay N items independientes?
   - Analizar el output del stage anterior
   - Si tiene una lista de modulos/archivos/endpoints -> fan-out candidato

2. Los items tocan archivos distintos?
   - Analisis estatico: "modulo auth" toca src/auth/*, "modulo api" toca src/api/*
   - Si no hay solapamiento -> paralelizar
   - Si hay solapamiento -> serializar (o particionar diferente)

3. Hay suficientes recursos?
   - Local: VRAM libre para N modelos? (o compartir modelo con N contextos)
   - Cloud: presupuesto permite N llamadas paralelas?

4. La politica lo permite?
   - ModePolicy.max_mode >= AssemblyLine
   - El usuario no ha deshabilitado paralelismo explicitamente

Si todas las respuestas son SI -> paralelizar
Si alguna es NO -> serializar (con log de por que)
```

---

## 26. Ventajas Competitivas de ai_assistant

Funcionalidades que `ai_assistant` tiene (o tiene disenadas) y que otros proyectos analizados
**NO tienen** a fecha de 2026-02-15. Esta seccion se actualiza periodicamente.

> **Politica de actualizacion**: Cada ~mes, re-analizar los proyectos listados.
> Si un proyecto incorpora algo que antes no tenia, NO borrar la entrada — marcarla
> con la fecha en que dejo de ser exclusiva.

### 26.1 Funcionalidades Exclusivas

Estados: IMPL = implementado en codigo | DISEÑADO = disenado en detalle (este doc) | PLANIFICADO = idea aceptada, sin diseno detallado aun

| # | Funcionalidad | Estado | Descripcion | Proyectos que NO la tienen | Fecha | Notas |
|---|---------------|--------|-------------|---------------------------|-------|-------|
| 1 | **RAG multi-tier** | IMPL | Pipeline RAG con TF-IDF, embeddings, knowledge graph y routing inteligente | OpenClaw (RAG basico), Aider (repo map) | 2026-02-15 | Nuestro multi-tier es mas sofisticado |
| 2 | **Multi-layer knowledge graph** | IMPL | Grafo de conocimiento con capas (proyecto, dominio, internet) y busqueda entre capas | Todos los analizados | 2026-02-15 | — |
| 3 | **Content versioning con CRDT** | IMPL | Versionado de contenido con conflict resolution para edicion concurrente | Todos los analizados | 2026-02-15 | — |
| 4 | **Bincode storage con auto-deteccion** | IMPL | Almacenamiento binario comprimido con deteccion automatica de formato (JSON legacy vs bincode) | Todos los analizados | 2026-02-15 | Todos usan JSON o SQLite |
| 5 | **Encrypted sessions (AES-256-GCM)** | IMPL | Sesiones cifradas en disco con clave del usuario | Todos los analizados | 2026-02-15 | — |
| 6 | **FallbackChain multi-provider** | IMPL | Cadena de fallback entre proveedores con cooldown y rotacion de API keys | Claude Code (solo 1 provider), Aider (solo 1), OpenClaw (tiene fallback similar) | 2026-02-15 | OpenClaw tiene algo similar |
| 7 | **RetryExecutor con clasificacion de errores** | IMPL | Retry inteligente que distingue errores transitorios de permanentes | Aider (retry basico), SWE-agent (no tiene) | 2026-02-15 | — |
| 8 | **ConversationCompactor** | IMPL | Compactacion de conversaciones sin llamada al LLM (ligera) | Claude Code (usa LLM para summarize), Aider (no tiene) | 2026-02-15 | — |
| 9 | **Journal sessions (append-only log)** | IMPL | Sesiones como event log append-only con compactacion | OpenClaw (tiene .jsonl similar), OpenHands (event sourcing similar) | 2026-02-15 | — |
| 10 | **Log redaction** | IMPL | Redaccion automatica de API keys, tokens, passwords en logs | OpenClaw (sanitiza magic strings), otros no | 2026-02-15 | — |
| 11 | **Adaptive thinking** | IMPL | Niveles de razonamiento adaptativo segun complejidad | Claude Code (extended thinking fijo), otros no | 2026-02-15 | — |
| 12 | **PII detection** | IMPL | Deteccion de informacion personal identificable | Ninguno de los analizados | 2026-02-15 | — |
| 13 | **Feed monitor** | IMPL | Monitorizacion de feeds RSS/Atom con deteccion de cambios | Ninguno de los analizados | 2026-02-15 | Nicho pero unico |
| 14 | **Danger evaluation score 0-100** | DISEÑADO | Evaluacion cuantitativa con 20+ factores, alternativas seguras, dry-run con verificacion de archivos no afectados | Claude Code (3 niveles deny/ask/allow), Aider (no tiene), SWE-agent (no tiene) | 2026-02-15 | — |
| 15 | **RequirementRegistry con sentido comun** | DISEÑADO | Registro de requisitos con inferencia de buenas practicas, enriquecimiento web/graph, contradicciones, ciclos | Todos los analizados | 2026-02-15 | Ninguno infiere requisitos de sentido comun ni detecta contradicciones |
| 16 | **CycleDetector multi-paso** | DISEÑADO | Deteccion de ciclos A->B->C->A en cambios del agente | Claude Code (loops simples), AutoGPT (loops simples) | 2026-02-15 | Los demas solo detectan repeticion directa |
| 17 | **Anonimizacion/deanonimizacion por proveedor** | DISEÑADO | Politicas de privacidad configurables por proveedor LLM con anonimizacion automatica | Todos los analizados, incluido OpenClaw | 2026-02-15 | — |
| 18 | **Estrategia hibrida local/cloud** | DISEÑADO | Fases (explorar/generar/verificar) con diferente modelo, escalacion automatica por enroque | Aider (multi-model sin fases), Claude Code (solo cloud) | 2026-02-15 | — |
| 19 | **Assembly-line con back-communication** | DISEÑADO | Cadena de agentes con escalacion bidireccional y deteccion de ping-pong | MetaGPT (unidireccional), CrewAI (sin control de ciclos) | 2026-02-15 | — |
| 20 | **IterationStrategy configurable** | DISEÑADO | Reemplazo de max_iterations con budget, progress detection, per-operation | Claude Code (token budget), Aider (fijo), AutoGPT (fijo) | 2026-02-15 | — |
| 21 | **Informe detallado de parada** | DISEÑADO | Report completo con progreso/iteracion, requisitos cumplidos, opciones al usuario | Ninguno da informe tan detallado | 2026-02-15 | — |
| 22 | **Plugin system para VCS** | DISEÑADO | Trait extensible para SVN, Hg, etc. | Claude Code, Aider, SWE-agent, OpenHands (todos solo git) | 2026-02-15 | — |
| 23 | **Safe delete con papelera cross-OS** | DISEÑADO | Mover a papelera en vez de borrar, con restauracion | Todos los analizados | 2026-02-15 | — |
| 24 | **Diagnostics/introspection tool** | DISEÑADO | MCP/tool que analiza config, tools, conexiones, feature flags compiladas | Claude Code (/doctor), OpenClaw (doctor), pero menos completos | 2026-02-15 | — |
| 25 | **Container management con politicas** | DISEÑADO | Docker lifecycle con max_per_session, auto-cleanup, export/import/share | E2B (sandboxes pero sin compartir), Goose (no tiene) | 2026-02-15 | — |
| 26 | **Modos de operacion dinamicos** | DISEÑADO | Chat/Asistente/Programacion/Assembly-line con escalacion/des-escalacion automatica | Claude Code (modo unico), Aider (modo unico) | 2026-02-15 | Ver seccion 28 |
| 27 | **Butler (auto-configuracion)** | DISEÑADO | Deteccion automatica de recursos, auto-configuracion optima, wizard inteligente | Aider (setup basico), OpenClaw (onboard wizard) | 2026-02-15 | Ver seccion 29 |
| 28 | **Multi-task planning con DAG** | DISEÑADO | Descomposicion de prompts multi-tarea, dependencias, cola con prioridades, interrupciones a mitad | Claude Code (secuencial), Aider (1 tarea a la vez) | 2026-02-15 | Ver seccion 27 |
| 29 | **Browser multi-sesion con privacy** | DISEÑADO | Multiples sesiones de browser, anonimizacion de formularios, URL safety check | browser-use (1 sesion), Stagehand (1 sesion) | 2026-02-15 | — |
| 30 | **HTTP anti-bloqueo configurable** | DISEÑADO | User-Agent rotation, headers de browser, proxy support para sitios que bloquean bots | Ninguno de los analizados | 2026-02-15 | — |
| 31 | **Permission escalation granular** | DISEÑADO | "Aceptar siempre en esta sesion", "anadir path al whitelist", "solo esta vez", por categorias | Claude Code (allow/deny global), Aider (no tiene permisos) | 2026-02-15 | — |
| 32 | **Enriquecimiento de requisitos desde web/graph** | PLANIFICADO | Buscar best practices en internet y knowledge graph para enriquecer requisitos | Todos los analizados | 2026-02-15 | Requiere acceso web configurado |
| 33 | **Repo map** | PLANIFICADO | Mapa de funciones/clases del codebase para contexto (inspirado en Aider) | Claude Code (no tiene), SWE-agent (no tiene) | 2026-02-15 | Aider ya lo tiene |
| 34 | **Memoria episodica entre sesiones** | PLANIFICADO | Recordar que funciono y que fallo en sesiones anteriores | Aider (no tiene), SWE-agent (no tiene) | 2026-02-15 | AutoGPT lo tiene parcialmente |
| 35 | **Commit-por-cambio automatico** | PLANIFICADO | Cada cambio del agente = commit git automatico | SWE-agent (no tiene), Claude Code (no automatico) | 2026-02-15 | Aider ya lo tiene |
| 36 | **Cost tracking real-time** | PLANIFICADO | Mostrar coste acumulado de tokens durante ejecucion | Aider (no tiene), SWE-agent (no tiene) | 2026-02-15 | AutoGPT lo tiene parcialmente |
| 37 | **InstalledPackageRegistry** | DISEÑADO | Registro de todo lo instalado por el sistema (browsers, modelos, tools), con tamano, desinstalacion, cleanup | Todos los analizados | 2026-02-15 | Ninguno registra ni permite desinstalar lo que instala |
| 38 | **GPU detection + model recommendation** | DISEÑADO | Detectar GPUs/VRAM cross-OS, recomendar modelos locales por tarea, enriquecer con IA | Ollama (detecta GPU pero no recomienda modelos por tarea), Aider (no tiene), Claude Code (no local) | 2026-02-15 | Ver seccion 30 |
| 39 | **Browser headless lifecycle** | DISEÑADO | Instalar, actualizar, desinstalar browser headless via tools, con deteccion de browsers del sistema | browser-use (asume instalado), Stagehand (asume instalado) | 2026-02-15 | — |
| 40 | **Per-task UndoLog con rollback parcial** | DISEÑADO | Undo aislado por tarea, rollback desde iteracion N, deteccion de conflictos entre tareas | Claude Code (undo global), Aider (no tiene undo) | 2026-02-15 | Ver secciones 16.3-16.4 |
| 41 | **Task continuation policy** | DISEÑADO | Si una tarea falla, evaluar si las demas pueden continuar (DAG + dependencias parciales) | Todos los analizados (paran todo o no tienen multi-task) | 2026-02-15 | Ver seccion 27.4 |
| 42 | **Auto-Ranker de modelos multi-fuente** | DISEÑADO | Pipeline automatico: recopilar de 12+ APIs, normalizar, categorizar por heuristicas, rankear con score compuesto por tarea, filtrar por VRAM/presupuesto, enriquecer con IA, instalar modelos recomendados | Ollama (no rankea), LM Studio (no rankea), Aider (no tiene), Claude Code (no local) | 2026-02-15 | Ver seccion 30.5 |
| 43 | **Navegacion flotante entre prompts** | DISEÑADO | Botones ▲/▼ flotantes + atajos de teclado para saltar entre prompts del usuario en conversaciones largas, con minimap e indice lateral. Requisito obligatorio para toda interfaz UI | Ninguno de los analizados tiene esto como requisito de diseno | 2026-02-15 | Ver seccion 31 |
| 44 | **Proteccion de contexto contra compresion** | DISEÑADO | Zonas protegidas (PinnedContent) que nunca se comprimen, resumen de requisitos siempre presente, verificacion post-compresion | Claude Code (tiene compresion pero no proteccion selectiva), resto no tiene | 2026-02-15 | Ver seccion 32 |
| 45 | **Clasificador automatico de prompts** | DISEÑADO | Clasificacion de intent (Question/Research/Coding/ComplexTask/Casual/SystemCommand) con mapeo automatico a modo de operacion y seleccion de modelo segun HybridStrategy | CrewAI (requiere config manual), resto no tiene clasificacion automatica | 2026-02-15 | Ver seccion 33 |
| 46 | **Import/Export universal de configuracion** | DISEÑADO | Todo lo configurable es exportable/importable: tiers, politicas, prompts, modelos, estrategias, pipelines. Soporte JSON/TOML/YAML/Bincode. Bundles completos o parciales. Perfiles intercambiables. Migracion de versiones de schema | Ninguno ofrece export/import completo; la mayoria solo config basica via .env o JSON simple | 2026-02-15 | Ver seccion 34 |
| 47 | **Paralelismo multi-eje en assembly-line** | DISEÑADO | Fan-out (N agentes por modulo), DAG de stages, pipeline parallelism (solapar stages), ejecucion especulativa (N enfoques, elegir mejor). Con deteccion de conflictos de archivos y undo paralelo | MetaGPT (secuencial solo), CrewAI (paralelo basico sin fan-out/especulativo), resto no tiene | 2026-02-15 | Ver seccion 25.6 |
| 48 | **VectorBackend trait con 5 tiers de escala** | IMPL | Abstraccion con auto-seleccion: Tier 0 in-memory -> Tier 2 LanceDB -> Tier 3 Qdrant. Migracion transparente entre tiers sin re-calcular embeddings (`migrate_vectors()`). Todos los tiers 0-3 gratuitos. LanceDB backend con tests completos | La mayoria usa un solo backend fijo (Chroma, Pinecone, etc.) sin escala gradual ni migracion | 2026-02-16 | Ver seccion 35 |
| 49 | **Sistema distribuido con replicacion y tolerancia a fallos** | IMPL | Consistent hashing con vnodes, Phi Accrual Failure Detector, replicacion configurable (min/max copies, quorum R+W>N, enforce_min_copies), anti-entropy sync via Merkle trees, mutual TLS + join tokens + HMAC challenge-response + constant-time comparison + secure RNG, networking QUIC via quinn, LAN discovery (UDP broadcast), peer exchange, reputation tracking + probation, max_connections enforcement, DistributedVectorDb wrapper. Feature flag: `distributed-network`. 113 tests | Ninguno de los analizados tiene sistema distribuido con replicacion + tolerancia a fallos + seguridad de nodos + auto-discovery integrado | 2026-02-16 | Ver seccion 36 |

### 26.2 Funcionalidades que Otros Tienen y Nosotros Aun No

Para referencia cruzada, las funcionalidades que otros tienen y nosotros tenemos pendientes
estan en la seccion 23 (Alta, Media y Baja prioridad).

### 26.3 Historico de Exclusividad

> Cuando un proyecto externo incorpore algo que antes era exclusivo nuestro,
> actualizar aqui con la fecha y el proyecto.

| Funcionalidad | Fecha exclusiva desde | Dejo de ser exclusiva | Proyecto que la incorporo |
|---------------|----------------------|----------------------|--------------------------|
| (ninguna aun) | — | — | — |

---

## 27. Multi-Task Planning

### 27.1 Descomposicion de Prompts Multi-Tarea

Cuando el usuario envia un prompt con varias tareas, el sistema las descompone:

```rust
struct TaskPlanner {
    /// Tareas extraidas del prompt
    tasks: Vec<PlannedTask>,
    /// Grafo de dependencias (DAG)
    dependencies: HashMap<String, Vec<String>>,
    /// Orden de ejecucion resultante
    execution_order: Vec<String>,
}

struct PlannedTask {
    id: String,
    description: String,
    /// Estimacion de complejidad (el LLM evalua)
    estimated_complexity: Complexity,
    /// Modo recomendado para esta tarea
    recommended_mode: OperationMode,
    /// Depende de estas tareas
    depends_on: Vec<String>,
    /// Estado actual
    status: TaskStatus,
}

enum Complexity { Trivial, Simple, Medium, Complex, VeryComplex }
enum TaskStatus { Pending, InProgress, Completed, Failed, Skipped }
```

### 27.2 Flujo de Planificacion

```
Prompt: "Anade autenticacion, corrige el bug del login, y actualiza el README"
     |
     v
Fase de PLANIFICACION (el LLM analiza):
  1. Descomponer en tareas atomicas:
     T1: Corregir bug del login
     T2: Anadir autenticacion
     T3: Actualizar README
  2. Detectar dependencias:
     T2 depende de T1? -> Preguntar si el bug es de auth o no
     T3 depende de T1 y T2? -> Si, documenta lo nuevo
  3. Construir DAG:
     T1 -> T2 -> T3
  4. Estimar modo por tarea:
     T1: Programacion (bug fix)
     T2: Programacion (feature)
     T3: Asistente (doc)
  5. Presentar plan:
     "He identificado 3 tareas:
      1. [Bug fix] Corregir login (primero, independiente)
      2. [Feature] Anadir autenticacion (despues del fix)
      3. [Doc] Actualizar README (al final, documenta todo)
      Orden: T1 -> T2 -> T3. OK?"
```

### 27.3 Gestion de Prompts a Mitad de Ejecucion

```rust
enum IncomingPromptType {
    /// No relacionado con la tarea actual — encolar
    Unrelated,
    /// Modifica la tarea actual — pausar, actualizar requisitos
    ModifiesCurrent,
    /// Urgente ("para", "cancela") — parada inmediata
    Urgent,
    /// Respuesta a pregunta del agente — integrar inmediatamente
    ResponseToQuestion,
    /// Prioridad alta — insertar en cola antes de lo pendiente
    HighPriority,
}

struct PromptQueue {
    /// Cola de prompts pendientes (por prioridad)
    queue: BinaryHeap<QueuedPrompt>,
    /// Tarea actualmente en ejecucion
    current_task: Option<PlannedTask>,
}
```

**Comportamiento por tipo**:

```
Caso 1: Prompt NO relacionado
  -> Encolar. Informar: "He recibido tu peticion. La procesare al terminar T2."
  -> Continuar tarea actual.

Caso 2: Prompt MODIFICA tarea actual
  -> "Ah, quiero OAuth, no JWT"
  -> Pausar agente.
  -> Actualizar RequirementRegistry (R2_JWT -> R2_OAuth).
  -> Detectar contradiccion con trabajo ya hecho.
  -> Opciones: [Deshacer lo de JWT y rehacer con OAuth] [Adaptar lo existente]

Caso 3: Urgente ("para todo")
  -> Parada inmediata.
  -> Generar StopReport (seccion 14.5).
  -> Opciones de rollback.

Caso 4: Respuesta a pregunta del agente
  -> Integrar inmediatamente en el RequirementRegistry.
  -> Reanudar agente sin encolar.

Caso 5: Alta prioridad ("antes de lo que estas haciendo, haz X")
  -> Pausar tarea actual (guardar estado).
  -> Ejecutar nueva tarea.
  -> Al terminar, reanudar la pausada.
```

### 27.4 Interrupcion Per-Task: Si T2 Falla, Que Pasa con T3?

Cuando una tarea se detiene, el sistema decide que hacer con el resto usando el DAG
de dependencias y una politica configurable:

```rust
enum TaskContinuationPolicy {
    /// Si una tarea falla, parar todo (conservador)
    StopAll,
    /// Continuar tareas independientes, pausar las que dependen de la fallida
    ContinueIndependent,
    /// Evaluar si las dependencias parciales son suficientes
    SmartContinue,
    /// Preguntar al usuario por cada tarea afectada
    AskPerTask,
}
```

**Caso A: T3 NO depende de T2** (politica: ContinueIndependent)

```
T1 ✓ completada
T2 ✗ detenida (auth fallo)     T3 ✓ puede continuar (README de T1)

-> T3 se ejecuta automaticamente.
-> Al final: "T1 y T3 completadas. T2 sigue pausada. Que hacemos con T2?"
```

**Caso B: T3 SI depende de T2** (politica: ContinueIndependent)

```
T1 ✓ completada
T2 ✗ detenida (auth fallo)     T3 ⏸ bloqueada (README necesita documentar auth)

-> T3 NO se ejecuta.
-> Informe: "T2 fallo. T3 depende de T2 y no puede continuar."
-> Opciones: [Reintentar T2] [Saltar T2, ejecutar T3 sin auth] [Parar todo]
```

**Caso C: T3 depende PARCIALMENTE** (politica: SmartContinue)

```
T2 hizo 4 de 7 requisitos antes de pararse.
T3 solo necesita que exista el endpoint (R1 de T2, ya cumplido).

-> Evaluar: los requisitos cumplidos de T2 cubren lo que T3 necesita?
-> Si si: T3 puede continuar con advertencia.
   "T3 puede ejecutarse con lo que T2 alcanzo a hacer (endpoint existe).
    Pero la auth no esta completa. Continuar T3? [Si/No]"
-> Si no: T3 bloqueada.
```

```rust
impl TaskPlanner {
    /// Evaluar si una tarea dependiente puede ejecutarse
    /// aunque su dependencia no haya terminado completamente
    fn can_proceed_with_partial_dependency(
        &self,
        blocked_task: &PlannedTask,
        failed_task: &PlannedTask,
        failed_task_report: &StopReport,
    ) -> PartialDependencyResult {
        // Analizar que requisitos de failed_task ya se cumplieron
        // y si blocked_task solo necesita esos
        // Si ambiguo -> AskUser
    }
}

enum PartialDependencyResult {
    CanProceed { warning: String },
    CannotProceed { reason: String },
    NeedsUserDecision { question: String },
}
```

### 27.5 Rollback Integrado por Tarea

Cada tarea tiene su UndoLog aislado (seccion 16.3). Cuando una tarea falla:

```
T1 ✓ completada   -> UndoLog_T1: [edit src/login.rs]
T2 ✗ detenida     -> UndoLog_T2: [create src/jwt.rs, edit src/auth.rs, edit Cargo.toml]
T3 ✓ completada   -> UndoLog_T3: [edit README.md]

Escenario 1: "Deshacer solo T2"
  -> Revertir UndoLog_T2 (jwt.rs, auth.rs, Cargo.toml)
  -> T1 y T3 intactas
  -> Verificar: T3 (README) menciona auth? Si si, advertir inconsistencia.

Escenario 2: "Reintentar T2 desde cero"
  -> Revertir UndoLog_T2
  -> Re-ejecutar T2 con cambios del usuario (ej: nueva pista)
  -> Al completar T2 v2, verificar si T3 necesita actualizarse

Escenario 3: "Reintentar T2 desde iteracion #4"
  -> Revertir solo iteraciones #5-#7 de UndoLog_T2
  -> Mantener src/jwt.rs (creado en #3) y primer edit de auth.rs (#4)
  -> Continuar desde estado de #4

Escenario 4: "Deshacer todo"
  -> Revertir UndoLog_T3, UndoLog_T2, UndoLog_T1 (en orden inverso)
  -> Estado limpio como antes de empezar
```

### 27.6 Paralelizacion de Tareas Independientes

Si dos tareas no tienen dependencias entre si, se pueden ejecutar en paralelo
(usando multi-agent si esta habilitado):

```
T1: Corregir bug del login     ] en paralelo
T2: Anadir endpoint de health  ] en paralelo
     \_______ ambas completas _______/
              |
              v
T3: Actualizar README (depende de T1 y T2)
```

Requisitos para paralelizar:
- Las tareas no tocan los mismos archivos (detectado por analisis estatico)
- El modo Assembly-line o Autonomo esta habilitado
- El usuario lo aprueba

**UndoLog en paralelo**: Si T1 y T2 se ejecutan en paralelo y ambas se quieren
deshacer, el `file_ownership` index de `TaskUndoManager` (seccion 16.3) detecta
si tocaron los mismos archivos y aplica rollback en orden seguro.

### 27.7 Resumen Visual del Estado Multi-Task

En cualquier momento, el usuario puede pedir el estado:

```
╔══════════════════════════════════════════════════════╗
║  Estado del Plan (3 tareas)                          ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  T1 ✓ "Corregir bug login"                          ║
║     └─ Completada en 3 iteraciones | $0.02 | 45s    ║
║     └─ Undo: 1 operacion reversible                 ║
║                                                      ║
║  T2 ✗ "Anadir autenticacion JWT"                    ║
║     └─ DETENIDA: enroque en iter #5-#7              ║
║     └─ 4/7 requisitos cumplidos                     ║
║     └─ Undo: 3 operaciones reversibles              ║
║     └─ Esperando decision del usuario               ║
║                                                      ║
║  T3 ⏸ "Actualizar README"                           ║
║     └─ Bloqueada: depende de T2 (R5 no cumplido)   ║
║     └─ Podria ejecutarse parcialmente (R1-R4 OK)   ║
║                                                      ║
║  Coste total: $0.10 | Tiempo: 3m 20s                ║
║  Archivos modificados: 4 | Undo total: 4 ops        ║
╚══════════════════════════════════════════════════════╝
```

---

## 28. Modos de Operacion

### 28.1 Modos Disponibles

| Modo | Tools habilitados | Modelo | Iteraciones | Uso tipico |
|------|-------------------|--------|-------------|------------|
| **Chat** | Ninguno (solo LLM) | Uno solo | 1 (sin loop) | Responder preguntas simples, conversacion |
| **Asistente** | Web search, browser, memory | Uno solo | Pocas (3-5) | Buscar info, responder con contexto, tareas cotidianas |
| **Programacion** | Filesystem, git, shell + todo de Asistente | Hibrido local/cloud | Moderadas (~15-30) | Escribir/editar codigo, refactoring |
| **Assembly-line** | Todo de Programacion | Multi-agente con fases | Muchas por agente | Tareas complejas que requieren diseno |
| **Autonomo** | Todo | Hibrido con escalacion | Sin limite fijo (budget) | Tareas largas con supervision minima |

### 28.2 Escalacion y Des-escalacion Dinamica

Los modos no son fijos. El sistema puede proponer cambiar de modo segun la situacion:

```rust
struct ModeManager {
    /// Modo actual
    current_mode: OperationMode,
    /// Politica de limites
    policy: ModePolicy,
    /// Historial de cambios de modo
    mode_history: Vec<ModeChange>,
}

#[derive(PartialOrd, Ord)]
enum OperationMode {
    Chat,           // 0 — mas barato
    Assistant,      // 1
    Programming,    // 2
    AssemblyLine,   // 3
    Autonomous,     // 4 — mas caro
}

struct ModePolicy {
    /// Modo minimo permitido (no bajar de aqui)
    min_mode: OperationMode,
    /// Modo maximo permitido (no subir de aqui)
    max_mode: OperationMode,
    /// Escalar automaticamente o preguntar al usuario?
    auto_escalate: bool,
    /// Des-escalar automaticamente cuando la tarea es simple?
    auto_deescalate: bool,
    /// Presupuesto maximo por sesion (limita modos caros)
    session_budget_usd: Option<f64>,
}
```

**Flujo de escalacion**:

```
Usuario empieza en modo Chat:
  "Que es un JWT?"
  -> Respuesta conversacional, 0 tools.

Usuario pide algo que necesita herramientas:
  "Buscame las ultimas vulnerabilidades de JWT"
  -> Detectar: necesita web search.
  -> Proponer: "Necesito buscar en internet. Escalo a modo Asistente? [Si/No]"
     (o auto-escalar si auto_escalate=true)

Usuario pide codigo:
  "Implementa autenticacion JWT en mi proyecto"
  -> Detectar: necesita filesystem, git, shell.
  -> Proponer: "Necesito acceso al codigo. Escalo a modo Programacion?"
  -> Preguntar: "Tu proyecto esta en /home/user/project?"

La tarea resulta muy compleja:
  -> Detectar: multiples fases (diseno + impl + QA), >5 archivos.
  -> Proponer: "Esta tarea es compleja. Quieres Assembly-line (mas riguroso)
     o seguir en Programacion (mas rapido)?"
```

**Flujo de des-escalacion**:

```
Agente en modo Programacion termina una tarea.
El siguiente prompt es: "Gracias. Que librerias de JWT recomiendas?"
  -> Detectar: pregunta conversacional, no necesita tools de codigo.
  -> Auto des-escalar a Chat (o Asistente si necesita buscar).
  -> No gastar tokens de modelo caro para una pregunta simple.
```

### 28.3 Ejemplos de Politicas

```rust
// "No quiero gastar dinero" — todo local
ModePolicy {
    min_mode: Chat,
    max_mode: Programming,
    auto_escalate: true,
    auto_deescalate: true,
    session_budget_usd: Some(0.0), // solo modelos locales
}

// "Quiero calidad maxima, no importa el coste"
ModePolicy {
    min_mode: Programming,
    max_mode: Autonomous,
    auto_escalate: true,
    auto_deescalate: false, // siempre usar el modo potente
    session_budget_usd: None, // sin limite
}

// "Equilibrio — nunca assembly-line, siempre preguntar"
ModePolicy {
    min_mode: Chat,
    max_mode: Programming,
    auto_escalate: false, // siempre preguntar
    auto_deescalate: true,
    session_budget_usd: Some(5.0),
}
```

### 28.4 Uso para Tareas No-Codigo

El sistema NO es solo para programacion. En modo Chat o Asistente sirve para:

- Responder preguntas generales (modo Chat — solo LLM)
- Buscar informacion en internet (modo Asistente — web search tool)
- Navegar webs y extraer datos (modo Asistente — browser tools)
- Analizar documentos (modo Asistente — file read + LLM)
- Responder con contexto de memoria (modo Asistente — RAG, knowledge graph)
- Monitorizar feeds y alertar (modo Asistente — feed monitor)
- Cualquier cosa que no requiera tocar codigo

La diferencia con un chatbot normal: tiene herramientas, memoria persistente entre sesiones,
knowledge graph, y puede actuar (buscar, navegar, guardar).

### 28.5 Deteccion del Working Directory

El agente necesita saber donde esta el codigo fuente. Prioridad de deteccion:

```
1. Si el usuario lo especifica: working_directory = "/home/user/my_project"
2. Deteccion automatica: buscar .git subiendo directorios desde cwd
3. Si no hay git: usar el directorio actual (cwd)
4. Si es sesion remota: el directorio se pasa como parametro de sesion
5. Si no hay directorio: modo Chat/Asistente unicamente (no se puede programar)
```

El agente trabaja dentro de ese directorio y subdirectorios (whitelist por defecto).
Tocar archivos fuera requiere permiso (seccion 4).

---

## 29. Butler: Auto-Configuracion Inteligente

### 29.1 Concepto

El "Butler" es un tool/MCP que, a partir de lo minimo aportado por el usuario,
detecta y completa toda la configuracion necesaria para que el sistema funcione
de forma optima. Programacion orientada a mayordomo.

### 29.2 Flujo del Butler

```
El usuario solo aporta:
  - "Tengo Ollama en localhost"
  - "API key de Anthropic: sk-ant-xxx"
  - "Mi proyecto esta en /home/user/project"
       |
       v
Butler detecta automaticamente:
  1. OS y package manager (Linux/apt, macOS/brew, Windows/choco)
  2. Modelos disponibles en Ollama -> ["qwen2.5-coder:7b", "llama3.1:8b"]
  3. Validar API key de Anthropic -> OK, acceso a Sonnet y Opus
  4. Detectar tipo de proyecto -> Rust (Cargo.toml encontrado)
  5. Detectar VCS -> git (con 5 ramas, branch actual: main)
  6. Verificar herramientas instaladas -> cargo OK, docker NO, node NO
  7. Verificar sandbox disponible -> Windows = sin bubblewrap/seatbelt
       |
       v
Butler configura automaticamente:
  8. HybridStrategy:
     - Exploracion: Ollama/qwen2.5-coder:7b (mejor local para codigo)
     - Generacion: Anthropic/claude-sonnet-4-5 (mejor relacion calidad/precio)
     - Verificacion: Ollama/qwen2.5-coder:7b
     - Escalacion: Anthropic/claude-opus-4-6
  9. Herramientas:
     - filesystem_read, filesystem_write: habilitadas para /home/user/project/**
     - git_read, git_write: habilitadas
     - shell: habilitada con sandbox=None (Windows)
     - browser: deshabilitada (no detectado playwright/chrome)
  10. Permisos:
     - Modo: Ask (por defecto, seguro)
     - Paths bloqueados: /, /etc, ~/.ssh, ~/.gnupg (defaults)
     - Working dir: /home/user/project
  11. Compaction:
     - Activada (Ollama tiene ventana de contexto pequena: 8K-32K)
  12. RequirementRegistry:
     - CommonSense: todas las categorias activas
     - Enriquecimiento web: desactivado por defecto (opt-in)
     - Enriquecimiento graph: activado si existe knowledge graph
       |
       v
Butler genera informe:
  "He configurado todo:
   ✓ 2 modelos locales detectados (Ollama)
   ✓ 1 proveedor cloud verificado (Anthropic)
   ✓ Estrategia hibrida: local para exploracion, cloud para generacion
   ✓ Proyecto Rust detectado en /home/user/project
   ✓ Git configurado (branch: main)
   ✓ 15 tools habilitadas (filesystem, git, shell)
   ✓ Modo de permisos: Ask (preguntar antes de cada accion)
   ⚠ Docker no instalado — contenedores no disponibles
   ⚠ Sandbox no disponible en Windows — recomiendo instalar Docker
   ⚠ Browser tools deshabilitadas — instalar playwright para habilitarlas
   Quieres ajustar algo?"
```

### 29.3 Estructura

```rust
struct Butler {
    /// Detectores de recursos
    detectors: Vec<Box<dyn ResourceDetector>>,
    /// Configuracion generada
    generated_config: Option<FullConfig>,
}

trait ResourceDetector: Send + Sync {
    /// Nombre del recurso que detecta
    fn name(&self) -> &str;
    /// Detectar si el recurso esta disponible
    fn detect(&self) -> DetectionResult;
}

enum DetectionResult {
    Found { details: String },
    NotFound { suggestion: String },
    Error { message: String },
}

struct FullConfig {
    hybrid_strategy: HybridStrategy,
    tools_enabled: Vec<String>,
    permissions: AgentPolicy,
    working_directory: PathBuf,
    compaction_enabled: bool,
    requirement_config: RequirementEnrichmentConfig,
    warnings: Vec<String>,
    suggestions: Vec<String>,
}
```

### 29.4 Detectores Incluidos

| Detector | Que detecta | Configuracion que genera |
|----------|-------------|------------------------|
| `OsDetector` | SO, arch, package manager | Sandbox type, install commands |
| `OllamaDetector` | Ollama running? Modelos disponibles | exploration_provider, verification_provider |
| `CloudApiDetector` | API keys validas? Modelos accesibles | generation_provider, escalation_provider |
| `ProjectDetector` | Tipo de proyecto (Rust, JS, Python, etc.) | CompileCheck command, TestCheck command |
| `VcsDetector` | Git/SVN/Hg? Branch actual, remotes | VCS tools, commit strategy |
| `DockerDetector` | Docker instalado? Corriendo? | Container tools availability |
| `BrowserDetector` | Playwright/Chrome/Firefox instalado? | Browser tools availability |
| `SandboxDetector` | Bubblewrap/seatbelt/Docker? | Sandbox level recommendation |
| `NetworkDetector` | Conexion a internet? Proxy configurado? | Network tools, proxy config |

### 29.5 Modo Interactivo vs Auto

```rust
enum ButlerMode {
    /// Detectar todo y configurar sin preguntar. Reportar al final.
    Auto,
    /// Paso a paso: preguntar cada decision al usuario.
    Interactive,
    /// Auto pero preguntar solo cuando hay ambiguedad.
    SmartAuto,
}
```

En modo `SmartAuto` (recomendado): el Butler configura lo que puede y solo pregunta
lo que no puede decidir solo:

```
"He detectado 2 modelos locales: qwen2.5-coder:7b y llama3.1:8b.
 Para tareas de codigo, qwen2.5-coder:7b es mejor.
 Para tareas generales, llama3.1:8b es mejor.
 -> Quieres que use qwen para codigo y llama para el resto?
 -> O prefieres usar solo uno de los dos para todo?"
```

### 29.6 Re-configuracion y Actualizacion

El Butler no es solo para el setup inicial. Se puede invocar en cualquier momento:

```
butler reconfigure         — Re-detectar todo y ajustar configuracion
butler diagnose            — Ver estado actual y warnings
butler suggest             — Sugerir mejoras (nuevo modelo disponible, etc.)
butler add-provider <key>  — Anadir nuevo proveedor y reconfigurar hibrido
butler remove-provider     — Quitar proveedor y reconfigurar
```

---

## 30. Deteccion de GPU y Recomendacion de Modelos

### 30.1 Proposito

Detectar automaticamente las GPUs disponibles en la maquina del usuario, su VRAM,
y recomendar que modelos locales puede ejecutar segun el tipo de tarea. Integrado
con el Butler (seccion 29) para auto-configurar la estrategia hibrida.

### 30.2 Deteccion de Hardware

```rust
struct GpuDetector;

struct GpuInfo {
    /// Nombre de la GPU
    name: String,
    /// Fabricante
    vendor: GpuVendor,
    /// VRAM total (bytes)
    vram_total: u64,
    /// VRAM libre (bytes, si se puede obtener)
    vram_free: Option<u64>,
    /// Compute capability (CUDA) o version Metal/ROCm
    compute_version: Option<String>,
    /// Driver version
    driver_version: Option<String>,
    /// Soporta inferencia de LLMs?
    llm_capable: bool,
}

enum GpuVendor {
    Nvidia,     // CUDA — mejor soporte para LLMs
    Amd,        // ROCm — soporte creciente
    Intel,      // oneAPI — soporte limitado
    Apple,      // Metal — buen soporte via llama.cpp/MLX
    Unknown,
}
```

**Deteccion cross-OS**:

| OS | NVIDIA | AMD | Intel | Apple |
|----|--------|-----|-------|-------|
| **Linux** | `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv` | `rocm-smi --showmeminfo vram` | `intel_gpu_top` / sysfs | N/A |
| **Windows** | `nvidia-smi` (mismo formato) | `rocm-smi` (si ROCm instalado) | `wmic path win32_VideoController` | N/A |
| **macOS** | N/A (raro) | N/A | N/A | `system_profiler SPDisplaysDataType` + `sysctl hw.memsize` (unified memory) |

**Fallback sin herramienta especifica**: si `nvidia-smi` no esta disponible,
intentar `lspci | grep -i vga` (Linux) o `wmic` (Windows) para al menos obtener
el nombre de la GPU y estimar VRAM por modelo conocido.

### 30.3 Tabla de Recomendacion de Modelos por VRAM

```rust
struct ModelRecommendation {
    model_name: String,
    /// VRAM minima necesaria para ejecutar (con quantizacion Q4)
    min_vram_gb: f32,
    /// VRAM recomendada (Q8 o FP16)
    recommended_vram_gb: f32,
    /// Para que tipo de tareas es bueno
    best_for: Vec<TaskCategory>,
    /// Velocidad relativa (tokens/s estimados con la VRAM minima)
    estimated_speed: SpeedTier,
    /// Calidad relativa para codigo
    code_quality: QualityTier,
    /// Provider local recomendado
    provider: String,  // "ollama", "llama.cpp", "vllm"
}

enum TaskCategory { CodeGeneration, CodeReview, GeneralChat, Summarization, Translation, Research }
enum SpeedTier { Slow, Medium, Fast, VeryFast }
enum QualityTier { Basic, Good, VeryGood, Excellent, TopTier }
```

**Tabla estatica base** (actualizable):

| VRAM disponible | Modelos recomendados | Uso principal |
|-----------------|---------------------|---------------|
| **< 4 GB** | Phi-3 Mini 3.8B (Q4), Qwen2.5 1.5B | Chat basico, tareas simples. Calidad limitada para codigo |
| **4-6 GB** | Qwen2.5 Coder 7B (Q4), Llama 3.1 8B (Q4), DeepSeek Coder V2 Lite 16B (Q3) | Codigo: Qwen. General: Llama. Mejor relacion calidad/VRAM |
| **6-8 GB** | Qwen2.5 Coder 7B (Q8), Llama 3.1 8B (Q6), Gemma 2 9B (Q4) | Codigo con mejor calidad. Buena velocidad |
| **8-12 GB** | CodeLlama 13B (Q4), Qwen2.5 14B (Q4), Phi-3 Medium 14B (Q4) | Salto significativo en calidad de codigo |
| **12-16 GB** | Codestral 22B (Q4), Qwen2.5 Coder 32B (Q3), Llama 3.1 70B (Q2) | Codestral excelente para codigo. 70B Q2 es arriesgado pero posible |
| **16-24 GB** | Codestral 22B (Q8), Qwen2.5 Coder 32B (Q4), DeepSeek Coder V2 (Q4) | Calidad semi-cloud en local. Recomendado como exploration_provider |
| **24-48 GB** | Qwen2.5 Coder 32B (Q8), Llama 3.1 70B (Q4), DeepSeek V3 (Q3) | Casi nivel cloud. Puede reemplazar generation_provider en hibrido |
| **48+ GB** | Llama 3.1 70B (Q8), DeepSeek V3 (Q6), Qwen2.5 72B (Q4) | Nivel cloud en local. Estrategia full-local viable |
| **Apple Silicon** | MLX: Qwen2.5 Coder 7B, Llama 3.1 8B | Unified memory. Rendimiento bueno pero VRAM compartida con sistema |

### 30.4 Enriquecimiento con Fuentes Externas

La tabla estatica puede quedarse desactualizada. Fuentes para actualizar:

**APIs consultables — Proveedores locales**:

| Fuente | Endpoint | Que aporta | Auth |
|--------|----------|------------|------|
| **LM Studio v1 API** | `GET http://localhost:1234/api/v1/models` | Lista TODOS los modelos del sistema: tipo (llm/embedding), publisher, architecture, quantizacion (metodo + bits), tamano en bytes, param count (7B, 13B...), max context length. Es la fuente local MAS rica en metadatos | Bearer token (opcional) |
| **Ollama** | `GET http://localhost:11434/api/tags` | Modelos instalados: nombre, tamano, familia, quantizacion. Solo modelos descargados, no catalogo completo | Ninguna |
| **Ollama Library (terceros)** | `GET https://ollamadb.dev/api/v1/models` | Catalogo COMPLETO de modelos disponibles en Ollama con filtrado y sorting. API no oficial pero util | Ninguna |
| **text-generation-webui** | `GET http://localhost:5000/v1/models` | Modelos cargados (OpenAI-compat). Poca metadata | Ninguna |
| **LocalAI** | `GET http://localhost:8080/v1/models` | Modelos instalados (OpenAI-compat) | Ninguna |

**APIs consultables — Proveedores cloud**:

| Fuente | Endpoint | Que aporta | Auth |
|--------|----------|------------|------|
| **OpenRouter** | `GET https://openrouter.ai/api/v1/models` | 400+ modelos: id, nombre, descripcion, context_length, max_completion_tokens, pricing (prompt/completion en USD/token). LA MEJOR fuente para comparar precios de cloud | Ninguna (publico) |
| **OpenAI** | `GET https://api.openai.com/v1/models` | Lista de modelos disponibles para la cuenta. Poca metadata (no precios) | Bearer token |
| **Anthropic** | No tiene endpoint de listado | Modelos documentados en docs. No hay API de catalogo | — |
| **Google AI** | `GET https://generativelanguage.googleapis.com/v1/models` | Lista modelos con descripcion, input/output token limits | API key |
| **Together AI** | `GET https://api.together.xyz/v1/models` | Modelos con precios, contexto, tipo (chat/code/embedding) | Bearer token |
| **Groq** | `GET https://api.groq.com/openai/v1/models` | Modelos disponibles. Inferencia ultra-rapida | Bearer token |
| **Mistral** | `GET https://api.mistral.ai/v1/models` | Modelos con capacidades y limites | Bearer token |

**APIs consultables — Catalogos y rankings**:

| Fuente | URL / Metodo | Que aporta | Auth |
|--------|-------------|------------|------|
| **HuggingFace Models API** | `GET https://huggingface.co/api/models?library=gguf&sort=downloads` | Model cards: tamano, quantizaciones GGUF, metadata del archivo (bits por peso, architecture). Filtrable por tags, libreria, tarea | Ninguna (publico) |
| **HuggingFace GGUF metadata** | Parsear cabecera del archivo .gguf | Metadata estandarizada: architecture, quantizacion, tamano exacto, context length nativo | N/A (local) |
| **Open LLM Leaderboard** | `https://huggingface.co/spaces/open-llm-leaderboard` | Rankings de calidad por benchmark (MMLU, HumanEval, etc.) | Ninguna |
| **LM Arena (Chatbot Arena)** | `https://lmarena.ai/` | Rankings basados en preferencias humanas (ELO scores) | Ninguna |
| **GPUStack GGUF Parser** | `https://gpustack.ai/gguf-parser/` | Herramienta online para estimar requisitos de VRAM de cualquier GGUF | Ninguna |

**Estrategia de consulta recomendada**:

```
1. Detectar proveedores locales instalados (LM Studio, Ollama, etc.)
2. Consultar sus APIs para saber que modelos tiene el usuario YA descargados
3. Detectar GPU y VRAM (seccion 30.2)
4. Si el usuario quiere mas opciones:
   a. Consultar HuggingFace GGUF para modelos descargables que quepan en su VRAM
   b. Consultar OpenRouter para opciones cloud con precios
   c. Consultar OllamaDB para catalogo completo de Ollama
5. Rankear por calidad (Open LLM Leaderboard / LM Arena)
6. (Opcional) Enriquecer con IA
```

**Nota sobre LM Studio**: Su API v1 es la mas rica en metadatos de todos los proveedores
locales — devuelve architecture, quantizacion exacta (metodo + bits por peso), param count,
y max context length. Si el usuario tiene LM Studio, es la fuente preferida para
inventariar modelos locales.

### 30.5 Auto-Ranker: Rankeo y Categorizacion Automatica

El sistema puede, sin intervencion humana, recopilar modelos de todas las fuentes,
normalizarlos, categorizarlos, rankearlos, y filtrarlos por el hardware del usuario.

#### 30.5.1 Pipeline Automatico

```
Paso 1: RECOPILAR
  fetch_lm_studio_models()  → Vec<RawModelInfo>   // si LM Studio instalado
  fetch_ollama_models()     → Vec<RawModelInfo>   // si Ollama instalado
  fetch_openrouter_catalog() → Vec<RawModelInfo>  // siempre (API publica)
  fetch_huggingface_gguf()  → Vec<RawModelInfo>   // top N por descargas
  fetch_leaderboard_scores() → HashMap<String, BenchmarkScores>
       |
       v
Paso 2: NORMALIZAR → Vec<ModelProfile>
       |
       v
Paso 3: CATEGORIZAR (automatico)
       |
       v
Paso 4: ESTIMAR VRAM (formula)
       |
       v
Paso 5: RANKEAR por tipo de tarea (score compuesto)
       |
       v
Paso 6: FILTRAR por hardware + presupuesto del usuario
       |
       v
Paso 7: (Opcional) ENRIQUECER con IA
       |
       v
Resultado: Vec<ModelRecommendation> ordenadas por score
```

#### 30.5.2 Estructuras

```rust
struct ModelAutoRanker {
    /// Fuentes de datos configuradas
    sources: Vec<Box<dyn ModelSource>>,
    /// Cache de perfiles (para no re-fetchear constantemente)
    cache: ModelCache,
    /// Pesos de scoring por tipo de tarea
    scoring_weights: HashMap<TaskCategory, ScoringWeights>,
}

/// Perfil unificado de un modelo (todas las fuentes normalizadas a esto)
struct ModelProfile {
    /// Identificador unico: "ollama:qwen2.5-coder:7b-q8" o "openrouter:anthropic/claude-sonnet-4-5"
    id: String,
    /// Nombre legible
    display_name: String,
    /// Familia del modelo
    family: String,            // "qwen2.5", "llama3.1", "claude", etc.
    /// Parametros (si conocido)
    params_billions: Option<f32>,
    /// Quantizacion (si local)
    quantization: Option<Quantization>,
    /// VRAM estimada necesaria (GB)
    estimated_vram_gb: Option<f32>,
    /// Es local o cloud?
    locality: ModelLocality,
    /// Categorias detectadas automaticamente
    categories: Vec<TaskCategory>,
    /// Scores de benchmarks (si disponibles)
    benchmark_scores: BenchmarkScores,
    /// Precio (solo cloud): USD por millon de tokens
    pricing: Option<ModelPricing>,
    /// Context window (tokens)
    context_length: Option<usize>,
    /// Fuente de donde se obtuvo la info
    source: String,
    /// Fecha de ultima actualizacion de esta info
    last_updated: u64,
}

struct Quantization {
    method: String,     // "Q4_K_M", "Q8_0", "FP16", etc.
    bits_per_weight: f32,
}

enum ModelLocality {
    Local { provider: String },           // "ollama", "lm_studio", "llama.cpp"
    Cloud { provider: String, api: String }, // "anthropic", "openrouter"
    Both,                                   // disponible en ambos
}

struct BenchmarkScores {
    humaneval: Option<f32>,      // Codigo (0-100)
    mbpp: Option<f32>,           // Codigo (0-100)
    mmlu: Option<f32>,           // Conocimiento general (0-100)
    arena_elo: Option<f32>,      // Preferencia humana (ELO)
    swebench: Option<f32>,       // Software engineering (0-100)
    math: Option<f32>,           // Matematicas (0-100)
    coding_overall: Option<f32>, // Score agregado de codigo
}

struct ModelPricing {
    input_per_mtok: f64,     // USD por millon de tokens de input
    output_per_mtok: f64,    // USD por millon de tokens de output
    free_tier: bool,         // tiene tier gratuito?
}
```

#### 30.5.3 Categorizacion Automatica

La categorizacion se hace por heuristicas sin LLM:

```rust
fn auto_categorize(profile: &ModelProfile) -> Vec<TaskCategory> {
    let mut cats = vec![];
    let name_lower = profile.display_name.to_lowercase();
    let family_lower = profile.family.to_lowercase();

    // Por nombre / familia
    if name_lower.contains("coder") || name_lower.contains("code")
       || family_lower.contains("codestral") || family_lower.contains("starcoder")
       || family_lower.contains("deepseek-coder") {
        cats.push(TaskCategory::CodeGeneration);
        cats.push(TaskCategory::CodeReview);
    }

    // Por benchmarks
    if let Some(he) = profile.benchmark_scores.humaneval {
        if he > 50.0 { cats.push(TaskCategory::CodeGeneration); }
    }
    if let Some(mmlu) = profile.benchmark_scores.mmlu {
        if mmlu > 60.0 { cats.push(TaskCategory::GeneralChat); }
        if mmlu > 70.0 { cats.push(TaskCategory::Research); }
    }

    // Por tags de HuggingFace (si disponibles)
    // "text-generation" -> GeneralChat
    // "text2text-generation" -> Summarization, Translation
    // "conversational" -> GeneralChat

    // Si no se detecto nada especifico, es general
    if cats.is_empty() {
        cats.push(TaskCategory::GeneralChat);
    }

    cats
}
```

#### 30.5.4 Estimacion de VRAM

Formula conocida para modelos GGUF:

```rust
fn estimate_vram_gb(params_billions: f32, bits_per_weight: f32) -> f32 {
    let model_size_gb = params_billions * bits_per_weight / 8.0;
    let kv_cache_overhead_gb = 1.5; // ~1-2 GB para KV cache, dependiendo del context
    model_size_gb + kv_cache_overhead_gb
}

// Ejemplos:
// Qwen 7B Q4:   7 * 4 / 8 + 1.5 = 5.0 GB
// Qwen 7B Q8:   7 * 8 / 8 + 1.5 = 8.5 GB
// Llama 70B Q4: 70 * 4 / 8 + 1.5 = 36.5 GB
// Phi-3 3.8B Q4: 3.8 * 4 / 8 + 1.5 = 3.4 GB
```

#### 30.5.5 Scoring Compuesto

Cada tipo de tarea tiene pesos diferentes:

```rust
struct ScoringWeights {
    benchmark_weight: f32,   // importancia de benchmarks
    speed_weight: f32,       // importancia de velocidad
    price_weight: f32,       // importancia de precio bajo
    context_weight: f32,     // importancia de context window grande
    vram_efficiency: f32,    // importancia de usar poca VRAM
}

// Pesos por defecto por tipo de tarea:
fn default_weights(task: TaskCategory) -> ScoringWeights {
    match task {
        CodeGeneration => ScoringWeights {
            benchmark: 0.40,  // humaneval/mbpp muy importante
            speed: 0.15,
            price: 0.15,
            context: 0.20,    // codigo necesita context grande
            vram_efficiency: 0.10,
        },
        CodeReview => ScoringWeights {
            benchmark: 0.30,
            speed: 0.25,      // review es iterativo, velocidad importa
            price: 0.15,
            context: 0.20,
            vram_efficiency: 0.10,
        },
        GeneralChat => ScoringWeights {
            benchmark: 0.25,  // mmlu/arena_elo
            speed: 0.30,      // respuesta rapida importante
            price: 0.25,      // chat es frecuente, coste acumula
            context: 0.10,
            vram_efficiency: 0.10,
        },
        Research => ScoringWeights {
            benchmark: 0.45,  // calidad es critica
            speed: 0.05,      // no importa si tarda
            price: 0.10,
            context: 0.30,    // research necesita mucho contexto
            vram_efficiency: 0.10,
        },
        // ... etc
    }
}
```

```rust
fn score_model(profile: &ModelProfile, task: TaskCategory, weights: &ScoringWeights) -> f32 {
    let benchmark_score = match task {
        CodeGeneration | CodeReview =>
            profile.benchmark_scores.humaneval.unwrap_or(0.0) * 0.6
            + profile.benchmark_scores.mbpp.unwrap_or(0.0) * 0.4,
        GeneralChat =>
            profile.benchmark_scores.mmlu.unwrap_or(0.0) * 0.5
            + profile.benchmark_scores.arena_elo.map(|e| (e - 800.0) / 5.0).unwrap_or(0.0) * 0.5,
        Research =>
            profile.benchmark_scores.mmlu.unwrap_or(0.0) * 0.4
            + profile.benchmark_scores.arena_elo.map(|e| (e - 800.0) / 5.0).unwrap_or(0.0) * 0.6,
        _ => profile.benchmark_scores.mmlu.unwrap_or(50.0),
    };

    let speed_score = match profile.locality {
        Local { .. } => 80.0,   // local siempre rapido (no latencia de red)
        Cloud { .. } => 60.0,   // cloud tiene latencia
        Both => 80.0,
    };

    let price_score = match &profile.pricing {
        None => 100.0,  // local = gratis = score maximo
        Some(p) if p.free_tier => 100.0,
        Some(p) => (100.0 - (p.input_per_mtok + p.output_per_mtok) * 2.0).max(0.0),
    };

    let context_score = profile.context_length
        .map(|c| ((c as f32) / 1_000_000.0 * 100.0).min(100.0))
        .unwrap_or(50.0);

    let vram_score = profile.estimated_vram_gb
        .map(|v| (100.0 - v * 3.0).max(0.0))  // menos VRAM = mejor score
        .unwrap_or(50.0);

    benchmark_score * weights.benchmark_weight
    + speed_score * weights.speed_weight
    + price_score * weights.price_weight
    + context_score * weights.context_weight
    + vram_score * weights.vram_efficiency
}
```

#### 30.5.6 Filtrado por Hardware y Presupuesto

```rust
struct UserConstraints {
    /// VRAM disponible (detectada por gpu_detect)
    available_vram_gb: Option<f32>,
    /// Presupuesto mensual maximo (USD)
    max_monthly_budget_usd: Option<f64>,
    /// Solo modelos locales? Solo cloud? Ambos?
    locality_filter: LocalityFilter,
    /// Categorias de interes
    task_categories: Vec<TaskCategory>,
    /// Numero maximo de recomendaciones
    max_results: usize,  // default: 10
}

enum LocalityFilter { LocalOnly, CloudOnly, Both }

fn filter_and_rank(
    profiles: &[ModelProfile],
    constraints: &UserConstraints,
    task: TaskCategory,
) -> Vec<ModelRecommendation> {
    profiles.iter()
        // Filtrar por VRAM
        .filter(|p| match (constraints.available_vram_gb, p.estimated_vram_gb) {
            (Some(avail), Some(needed)) => needed <= avail,
            _ => true,  // si no sabemos, incluir
        })
        // Filtrar por presupuesto
        .filter(|p| match (constraints.max_monthly_budget_usd, &p.pricing) {
            (Some(budget), Some(pricing)) => {
                // Estimar coste mensual: ~1M tokens/mes uso tipico
                let monthly_est = (pricing.input_per_mtok * 0.7 + pricing.output_per_mtok * 0.3);
                monthly_est <= budget
            }
            _ => true,
        })
        // Filtrar por localidad
        .filter(|p| match constraints.locality_filter {
            LocalOnly => matches!(p.locality, ModelLocality::Local { .. } | ModelLocality::Both),
            CloudOnly => matches!(p.locality, ModelLocality::Cloud { .. } | ModelLocality::Both),
            Both => true,
        })
        // Filtrar por categoria
        .filter(|p| p.categories.iter().any(|c| c == &task))
        // Scorear y ordenar
        .map(|p| {
            let score = score_model(p, task, &default_weights(task));
            ModelRecommendation { profile: p.clone(), score, task }
        })
        .sorted_by(|a, b| b.score.partial_cmp(&a.score).unwrap())
        .take(constraints.max_results)
        .collect()
}
```

#### 30.5.7 Enriquecimiento con IA (Opcional)

Despues del rankeo automatico, opcionalmente pedir al LLM que refine:

```
Input al LLM (modelo local, gratis):
  "Estos son los top 5 modelos para generacion de codigo Rust,
   filtrados para una RTX 3060 (12 GB VRAM):
   1. Qwen2.5 Coder 7B Q8 — score 82, HumanEval 75%, 8.5 GB VRAM
   2. CodeLlama 13B Q4 — score 78, HumanEval 62%, 10 GB VRAM
   3. DeepSeek Coder V2 Lite Q4 — score 75, HumanEval 70%, 11 GB VRAM
   4. Phi-3 Medium 14B Q4 — score 71, HumanEval 58%, 10 GB VRAM
   5. Gemma 2 9B Q4 — score 68, HumanEval 55%, 7 GB VRAM
   Hay algo que los benchmarks no capturen? Algun matiz importante
   para un proyecto Rust especificamente?"

Output del LLM:
  "Para Rust especificamente, Qwen2.5 Coder es la mejor opcion porque
   fue entrenado con mas codigo Rust que los demas. CodeLlama tiene
   sesgo hacia Python/JavaScript. DeepSeek Coder V2 es bueno para Rust
   pero el Q4 pierde calidad en sugerencias de tipos. Recomiendo:
   - Principal: Qwen2.5 Coder 7B Q8 (el #1)
   - Alternativa para mas complejidad: DeepSeek Coder V2 Lite Q4 (#3)"
```

#### 30.5.8 Cache y Actualizacion

```rust
struct ModelCache {
    profiles: Vec<ModelProfile>,
    last_full_refresh: u64,
    /// Re-fetchear todo cada 24 horas (los catalogos cambian poco)
    refresh_interval_hours: u64,  // default: 24
    /// Re-fetchear solo los scores cada semana
    scores_refresh_interval_hours: u64,  // default: 168 (7 dias)
}
```

El cache se guarda en disco (via internal_storage) para no re-fetchear en cada sesion.
El Butler puede forzar un refresh: `butler reconfigure --refresh-models`.

#### 30.5.9 Ejemplo de Output Completo

```
╔══════════════════════════════════════════════════════════════════╗
║  Recomendacion de Modelos — RTX 3060 (12 GB VRAM)              ║
║  Tarea: Generacion de codigo Rust                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MODELOS LOCALES (gratis):                                       ║
║  #1 ★ Qwen2.5 Coder 7B Q8      Score: 82  VRAM: 8.5 GB        ║
║       HumanEval: 75% | Context: 32K | Velocidad: rapida        ║
║       ✓ Ya instalado en Ollama                                  ║
║                                                                  ║
║  #2   CodeLlama 13B Q4           Score: 78  VRAM: 10 GB        ║
║       HumanEval: 62% | Context: 16K | Velocidad: media         ║
║       ↓ Disponible: ollama pull codellama:13b-q4                ║
║                                                                  ║
║  #3   DeepSeek Coder V2 Lite Q4  Score: 75  VRAM: 11 GB        ║
║       HumanEval: 70% | Context: 128K | Velocidad: media        ║
║       ↓ Disponible: ollama pull deepseek-coder-v2:lite          ║
║                                                                  ║
║  MODELOS CLOUD (de pago, para fase de generacion):              ║
║  #1 ★ Claude Sonnet 4.5         Score: 95  Precio: $18/Mtok    ║
║       HumanEval: 92% | Context: 200K                           ║
║       ✓ API key configurada                                    ║
║                                                                  ║
║  #2   DeepSeek V3                Score: 88  Precio: $1.4/Mtok  ║
║       HumanEval: 85% | Context: 128K                           ║
║       ↓ Configurar en: OpenRouter ($0 para tier gratuito)       ║
║                                                                  ║
║  ESTRATEGIA HIBRIDA RECOMENDADA:                                ║
║    Exploracion: Qwen2.5 Coder 7B Q8 (local, $0)               ║
║    Generacion:  Claude Sonnet 4.5 (cloud, ~$0.15/tarea)        ║
║    Verificacion: Qwen2.5 Coder 7B Q8 (local, $0)              ║
║    Coste estimado: ~$0.15/tarea (vs $1.50 todo-cloud)          ║
║                                                                  ║
║  [IA] Nota: Para Rust, Qwen2.5 Coder tiene mejor cobertura    ║
║  que CodeLlama (que tiene sesgo Python/JS).                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### 30.6 Tools de GPU y Modelos

**Tools de hardware**:

| Tool | Descripcion | Riesgo |
|------|-------------|--------|
| `gpu_detect` | Detectar GPUs, VRAM, drivers, compute capability | BAJO |
| `gpu_vram_status` | VRAM actual: total, libre, usada por Ollama/otro | BAJO |
| `gpu_benchmark` | Ejecutar benchmark rapido de inferencia con un modelo | MEDIO |

**Tools de catalogo y recomendacion**:

| Tool | Descripcion | Riesgo |
|------|-------------|--------|
| `model_list_local` | Listar modelos instalados localmente (Ollama, LM Studio, etc.) con tamano, quantizacion | BAJO |
| `model_list_available` | Listar modelos descargables que caben en la VRAM del usuario | BAJO |
| `model_recommend` | Rankeo automatico: recomendar modelos por tarea, filtrado por hardware y presupuesto | BAJO |
| `model_search` | Buscar un modelo especifico en todas las fuentes (local + HuggingFace + OpenRouter) | BAJO |
| `model_info` | Detalle completo de un modelo: benchmarks, VRAM, precios, categorias, donde esta disponible | BAJO |

**Tools de instalacion/gestion**:

| Tool | Descripcion | Riesgo | Permisos |
|------|-------------|--------|----------|
| `model_install` | Descargar e instalar un modelo (via Ollama pull, HF download, etc.) | ALTO | Pedir permiso (puede ser varios GB) |
| `model_uninstall` | Desinstalar un modelo instalado por este sistema | MEDIO | Pedir permiso |
| `model_update` | Actualizar un modelo a version mas reciente | MEDIO | Pedir permiso |
| `model_disk_usage` | Espacio total en disco usado por modelos | BAJO | Auto |

**Flujo de instalacion de modelos**:

```
El auto-ranker recomienda Qwen2.5 Coder 7B Q8 pero no esta instalado
     |
     v
Proponer al usuario:
  "El modelo recomendado para tu GPU (RTX 3060, 12 GB) es:
   Qwen2.5 Coder 7B Q8 — HumanEval 75%, usa 8.5 GB VRAM

   No esta instalado. Quieres que lo descargue?
   Tamano: ~7.8 GB | Via: ollama pull qwen2.5-coder:7b-q8
   Espacio en disco actual usado por modelos: 4.2 GB

   [1] Instalar (descargar ~7.8 GB)
   [2] Ver alternativas ya instaladas
   [3] No instalar"
     |
     v
Si instala:
  - Ejecutar: ollama pull qwen2.5-coder:7b-q8
  - Mostrar progreso de descarga
  - Registrar en InstalledPackageRegistry (seccion 20.5)
  - Actualizar HybridStrategy del Butler automaticamente
```

**Gestion de espacio**:

```
El usuario pide: model_disk_usage

"Modelos de IA instalados por ai_assistant:
  ollama:qwen2.5-coder:7b-q8     7.8 GB   instalado 2026-02-15  EN USO (exploration)
  ollama:llama3.1:8b-q4           4.7 GB   instalado 2026-02-10  EN USO (general chat)
  ollama:codellama:13b-q4         8.2 GB   instalado 2026-02-08  NO EN USO
  ─────────────────────────────────────────
  Total: 20.7 GB

  Sugerencia: codellama:13b-q4 no se usa en ninguna configuracion activa.
  Desinstalar liberaria 8.2 GB. [Desinstalar?]"
```

**Deteccion de duplicados**:

Un mismo modelo puede estar instalado en varios proveedores (Ollama + LM Studio),
o el usuario puede tener varias quantizaciones del mismo modelo base,
o el mismo modelo bajo nombres distintos.

```rust
struct DuplicateDetector;

enum DuplicateKind {
    /// Mismo modelo exacto en diferentes proveedores
    /// ej: qwen2.5-coder:7b-q8 en Ollama Y en LM Studio
    ExactDuplicate {
        model_family: String,
        params: f32,
        quantization: String,
        providers: Vec<String>,
        wasted_gb: f32,
    },
    /// Mismo modelo base, diferente quantizacion
    /// ej: qwen2.5-coder:7b-q4 Y qwen2.5-coder:7b-q8
    QuantizationVariants {
        model_family: String,
        params: f32,
        variants: Vec<(String, f32)>,  // (quant, size_gb)
        suggestion: String,            // "quedarte solo con Q8 si tienes VRAM"
    },
    /// Mismo modelo base, diferente tamano
    /// ej: llama3.1:8b Y llama3.1:70b
    SizeVariants {
        model_family: String,
        variants: Vec<(f32, String, f32)>,  // (params, quant, size_gb)
        suggestion: String,
    },
    /// Modelos de la misma familia donde uno es estrictamente mejor
    /// ej: codellama:13b cuando ya tienes qwen2.5-coder:7b (que puntua mejor en benchmarks)
    Superseded {
        inferior: String,
        superior: String,
        reason: String,  // "Qwen2.5 Coder 7B puntua mejor en HumanEval (75% vs 62%)"
    },
}
```

```
El usuario pide: model_list_local --check-duplicates

"Modelos instalados (3 proveedores detectados):

  Ollama:
    qwen2.5-coder:7b-q8       7.8 GB
    qwen2.5-coder:7b-q4       4.2 GB
    llama3.1:8b-q4             4.7 GB
    codellama:13b-q4           8.2 GB

  LM Studio:
    Qwen2.5-Coder-7B-Q8       7.8 GB
    Llama-3.1-8B-Q6            5.9 GB

  Total: 38.6 GB en 6 entradas

  ⚠ Duplicados detectados:

  [EXACTO] qwen2.5-coder:7b-q8
    En Ollama (7.8 GB) Y en LM Studio (7.8 GB)
    -> Puedes eliminar uno. Ahorro: 7.8 GB
    -> Sugerencia: eliminar de LM Studio (Ollama es tu provider principal)

  [QUANTIZACION] qwen2.5-coder:7b
    Q4 (4.2 GB) Y Q8 (7.8 GB) en Ollama
    -> Si tienes VRAM suficiente (8.5 GB), Q8 es mejor. Eliminar Q4 ahorra 4.2 GB
    -> Si VRAM es justa, quedarse con Q4 y eliminar Q8 ahorra 7.8 GB

  [SUPERSEDED] codellama:13b-q4
    CodeLlama 13B Q4 (HumanEval: 62%) es inferior a Qwen2.5 Coder 7B Q8 (75%)
    y ademas usa mas VRAM (10 GB vs 8.5 GB)
    -> Sugerencia: eliminar codellama:13b-q4. Ahorro: 8.2 GB

  [TAMANO] llama3.1
    8B Q4 en Ollama (4.7 GB) Y 8B Q6 en LM Studio (5.9 GB)
    -> Mismo modelo, diferente quant. Q6 es mejor calidad.
    -> Sugerencia: quedarte con Q6 en LM Studio, eliminar Q4 de Ollama. Ahorro: 4.7 GB

  Ahorro potencial total: 24.9 GB"
```

La deteccion de duplicados funciona comparando la familia del modelo (normalizada)
y los parametros. Para identificar que "Qwen2.5-Coder-7B-Q8" de LM Studio y
"qwen2.5-coder:7b-q8" de Ollama son el mismo modelo, se usa un normalizador de nombres:

```rust
fn normalize_model_name(raw_name: &str) -> (String, Option<f32>, Option<String>) {
    // "Qwen2.5-Coder-7B-Q8" -> ("qwen2.5-coder", Some(7.0), Some("q8"))
    // "qwen2.5-coder:7b-q8"  -> ("qwen2.5-coder", Some(7.0), Some("q8"))
    // "codellama:13b-q4"     -> ("codellama", Some(13.0), Some("q4"))
    // Extraer familia, param count, quantizacion con regex
}
```

### 30.7 Integracion con Butler

El Butler (seccion 29) usa `gpu_detect` en su flujo automatico:

```
Butler detecta:
  OS: Windows 11
  GPU: NVIDIA RTX 3060 (12 GB VRAM)
  Ollama: instalado, 3 modelos descargados
  API key Anthropic: valida
       |
       v
Butler configura:
  HybridStrategy:
    exploration: Ollama/qwen2.5-coder:7b   <- gpu_recommend_models dice: mejor Q8 con 12 GB
    generation: Anthropic/claude-sonnet-4-5  <- cloud para calidad maxima
    verification: Ollama/qwen2.5-coder:7b   <- mismo modelo local
    escalation: Anthropic/claude-opus-4-6
       |
       v
Butler informa:
  "GPU detectada: RTX 3060 (12 GB VRAM)
   Modelos locales recomendados para tu hardware:
     - Qwen2.5 Coder 7B Q8 (ya descargado en Ollama) — exploración y verificación
     - Alternativa: descargar CodeLlama 13B Q4 para mejor calidad (~8 GB)
   He configurado la estrategia híbrida con Qwen local + Claude cloud."
```

Si la GPU NO es detectada o no tiene VRAM suficiente:

```
Butler informa:
  "⚠ No se detectó GPU compatible con inferencia de LLMs.
   Opciones:
     [1] Usar CPU para modelos locales (lento pero funciona, recomiendo Phi-3 Mini 3.8B)
     [2] Usar solo modelos cloud (todo de pago, sin fase local gratuita)
     [3] Instalar Ollama y descargar modelo pequeño para probar"
```

### 30.8 Registro de Modelos como Paquetes

Los modelos descargados por Ollama/llama.cpp se registran en el InstalledPackageRegistry
(seccion 20.5) como `PackageKind::AiModel`:

```
InstalledPackage {
    name: "ollama:qwen2.5-coder:7b-q8",
    version: "2026.01",
    kind: AiModel,
    size_bytes: 7_800_000_000,  // ~7.8 GB
    installed_paths: ["~/.ollama/models/manifests/.../qwen2.5-coder"],
    install_method: ViaTool { tool: "ollama", command: "ollama pull qwen2.5-coder:7b-q8" },
    installed_by: "ai_assistant",
    can_uninstall: true,
    uninstall_command: Some("ollama rm qwen2.5-coder:7b-q8"),
}
```

Asi `package_list` muestra tambien los modelos, `package_disk_usage` incluye su tamano,
y `package_uninstall` puede limpiar modelos que ya no se usan.

---

## 31. UX de Conversacion: Navegacion y Widgets

### 31.1 Problema

Las respuestas de la IA pueden ser muy largas (cientos de lineas de codigo, explicaciones
detalladas, informes). Cuando el usuario scrollea por la conversacion, necesita poder
saltar rapidamente al prompt anterior o al siguiente sin tener que scrollear manualmente
a traves de respuestas enormes.

### 31.2 Navegacion entre Prompts (REQUISITO)

> **REQUISITO DE DOCUMENTACION**: Todo widget de conversacion (egui, web, terminal)
> DEBE implementar navegacion rapida entre prompts del usuario.

**Botones flotantes**:

```
    ┌─────────────────────────────────────────────┐
    │ [Usuario] Dame un análisis de auth.rs        │
    ├─────────────────────────────────────────────┤
    │ [IA] Aquí tienes el análisis:                │
    │                                              │
    │   (200 líneas de respuesta...)               │
    │                                              │  ┌──┐
    │   ... el módulo usa JWT con HMAC-SHA256 ...  │  │ ▲ │ <- Ir al prompt anterior
    │   ... las funciones principales son ...      │  ├──┤
    │   ... recomendaciones de seguridad ...       │  │ ▼ │ <- Ir al prompt siguiente
    │                                              │  └──┘
    │   (más líneas...)                            │   botones
    │                                              │   flotantes
    ├─────────────────────────────────────────────┤
    │ [Usuario] Ahora implementa MFA               │
    ├─────────────────────────────────────────────┤
    │ [IA] ...                                     │
    └─────────────────────────────────────────────┘
```

### 31.3 Implementacion en egui (widgets.rs)

```rust
/// Estado de navegacion entre prompts
pub struct PromptNavigator {
    /// Indices de los mensajes que son del usuario
    user_prompt_indices: Vec<usize>,
    /// Indice actual (para saber entre que prompts estamos)
    current_visible_prompt: Option<usize>,
}

impl PromptNavigator {
    /// Construir desde la lista de mensajes
    pub fn from_messages(messages: &[ChatMessage]) -> Self {
        let indices: Vec<usize> = messages.iter()
            .enumerate()
            .filter(|(_, m)| m.is_user())
            .map(|(i, _)| i)
            .collect();
        Self { user_prompt_indices: indices, current_visible_prompt: None }
    }

    /// Indice del prompt anterior al scroll actual
    pub fn previous_prompt(&self, current_scroll_index: usize) -> Option<usize>;

    /// Indice del prompt siguiente al scroll actual
    pub fn next_prompt(&self, current_scroll_index: usize) -> Option<usize>;
}

/// Widget de botones flotantes de navegacion
pub fn prompt_nav_buttons(
    ui: &mut Ui,
    navigator: &PromptNavigator,
    current_scroll: usize,
) -> Option<NavigationAction> {
    // Renderizar botones ▲ / ▼ en esquina inferior derecha
    // Devolver ScrollTo(index) si se pulsa
}

enum NavigationAction {
    ScrollToMessage(usize),
}
```

### 31.4 Atajos de Teclado

| Atajo | Accion |
|-------|--------|
| `Ctrl+↑` / `Cmd+↑` | Ir al prompt del usuario anterior |
| `Ctrl+↓` / `Cmd+↓` | Ir al prompt del usuario siguiente |
| `Home` | Ir al primer mensaje |
| `End` | Ir al ultimo mensaje |
| `Ctrl+F` | Buscar en la conversacion |

### 31.5 Funcionalidades Adicionales de Navegacion

| Feature | Descripcion |
|---------|-------------|
| **Indice lateral** | Lista compacta de todos los prompts del usuario (como tabla de contenidos) |
| **Busqueda** | Buscar texto en toda la conversacion, resaltar resultados |
| **Colapsar respuestas** | Boton para colapsar/expandir respuestas largas |
| **Anclas por tarea** | En modo multi-task, separadores visuales por tarea con nombre |
| **Minimap** | Barra lateral que muestra la posicion actual en la conversacion (como en editores de codigo) |
| **Contadores** | "Prompt 3 de 12" en los botones flotantes |
| **Marcadores** | El usuario puede marcar mensajes importantes para volver a ellos |

### 31.6 Requisito para Todas las Interfaces

> **REGLA**: Cualquier interfaz de conversacion que se implemente (egui, web/HTML, terminal TUI)
> DEBE incluir como minimo:
> 1. Botones flotantes ▲/▼ para saltar entre prompts del usuario
> 2. Atajos de teclado equivalentes
> 3. Indicador de posicion ("Prompt N de M")
>
> Esto se verifica en el CompletionVerifier (seccion 21) como requisito de tipo
> `Documentation` si se esta trabajando en widgets/UI.

---

## 32. Proteccion de Contexto Contra Compresion

### 32.1 Problema

Cuando la ventana de contexto se llena, los LLMs comprimen/resumen mensajes antiguos
o directamente los eliminan. Esto puede hacer que se pierdan:
- Requisitos del usuario mencionados al inicio de la conversacion
- Decisiones de diseno tomadas en prompts anteriores
- Configuraciones o preferencias expresadas por el usuario
- Contexto critico sobre el proyecto

Esto es especialmente problematico para el RequirementRegistry (seccion 21):
los requisitos se extraen de los mensajes, pero si los mensajes se comprimen,
el contexto original se pierde.

### 32.2 Solucion: Zonas Protegidas de Contexto

```rust
struct ContextProtection {
    /// Mensajes/fragmentos que NUNCA deben comprimirse
    pinned_content: Vec<PinnedContent>,
    /// Resumen protegido de requisitos (se mantiene siempre)
    requirement_summary: String,
    /// Configuracion activa (se inyecta siempre al principio)
    active_config_summary: String,
}

struct PinnedContent {
    /// Texto protegido
    content: String,
    /// Razon por la que esta protegido
    reason: PinReason,
    /// Prioridad (si hay que elegir que mantener, mayor prioridad gana)
    priority: u32,
    /// Tamano en tokens (estimado)
    estimated_tokens: usize,
}

enum PinReason {
    /// Es un requisito del usuario
    UserRequirement,
    /// Es una decision de diseno confirmada
    DesignDecision,
    /// Es configuracion critica
    Configuration,
    /// El usuario lo marco explicitamente como importante
    UserPinned,
    /// Es el prompt original de la tarea actual
    OriginalTaskPrompt,
    /// Es un resultado de una decision sobre contradiccion
    ContradictionResolution,
}
```

### 32.3 Flujo de Compresion Segura

```
Ventana de contexto se llena (ej: 32K tokens, quedan 2K libres)
     |
     v
Paso 1: Identificar contenido protegido
  - Todos los PinnedContent (requisitos, decisiones, config)
  - El RequirementRegistry serializado como resumen compacto
  - El prompt actual del usuario
  - Los ultimos 2-3 mensajes (contexto inmediato)
     |
     v
Paso 2: Calcular espacio disponible para compresion
  tokens_protegidos = sum(pinned.estimated_tokens) + requirement_summary_tokens
  tokens_comprimibles = total_tokens - tokens_protegidos - margen_seguridad
     |
     v
Paso 3: Comprimir solo lo NO protegido
  - Respuestas largas de la IA → resumir
  - Resultados de tools → resumir o eliminar (el resultado ya se uso)
  - Conversacion casual → resumir agresivamente
  - NUNCA tocar contenido protegido
     |
     v
Paso 4: Verificar que los requisitos siguen presentes
  Para cada requisito en RequirementRegistry:
    - Buscar en el contexto comprimido
    - Si no se encuentra referencia → inyectar resumen del requisito
     |
     v
Resultado: Contexto comprimido que MANTIENE todos los requisitos y decisiones
```

### 32.4 Resumen de Requisitos Siempre Presente

Independientemente de la compresion, se mantiene un bloque compacto al inicio del contexto:

```
[REQUISITOS ACTIVOS - no comprimir]
R1 [Must] Endpoint /login con JWT (pedido por usuario, sesion 5)
R2 [Must] Refresh tokens con rotacion (pedido por usuario, sesion 5)
R3 [Must] Tests para funciones publicas (README.md)
R4 [Must] Sin warnings del compilador (CONTRIBUTING.md)
R5 [Should] Usar Result<T, AppError> (patron del codigo existente)
R6 [Resuelto] Token expira en 1h (contradiccion resuelta: usuario eligio 1h)

[DECISIONES DE DISENO - no comprimir]
D1: Usar RS256 en vez de HS256 (decidido en prompt #3)
D2: No implementar MFA en esta fase (decidido en prompt #7)

[CONFIG ACTIVA]
Modo: Programacion | Modelo: Qwen local + Claude cloud
Working dir: /home/user/project | VCS: git (main)
```

Este bloque se genera automaticamente del RequirementRegistry y ocupa ~200-500 tokens.
Es una inversion pequena que evita que el agente "olvide" lo importante.

### 32.5 El Usuario Puede Proteger Contenido

```
Usuario: "Esto es importante, no lo olvides: el API debe ser retrocompatible con v1"
     |
     v
Detectar patron "no lo olvides" / "recuerda" / "importante"
     |
     v
Crear PinnedContent {
    content: "El API debe ser retrocompatible con v1",
    reason: UserPinned,
    priority: 90,
}
     |
     v
Confirmar: "He marcado como protegido: 'API retrocompatible con v1'.
 No se perdera aunque se comprima el contexto."
```

---

## 33. Seleccion Automatica de Agente/Modo

### 33.1 Problema

Cuando el usuario/cliente envia un prompt, el sistema debe decidir:
- Que tipo de agente usar (o que modo de operacion, seccion 28)
- Cuantos agentes (uno solo o assembly-line)
- Que modelo para cada fase

Actualmente en el crate, esto NO esta implementado. El `agentic_loop.rs` es un stub
y el `ReactAgent` no esta integrado con el LLM real. La seleccion de agente no existe.

### 33.2 Clasificacion de Prompts

```rust
struct PromptClassifier;

enum PromptIntent {
    /// Pregunta simple, no requiere herramientas
    Question { topic: String },
    /// Busqueda de informacion (necesita web/browser)
    Research { query: String },
    /// Tarea de codigo (necesita filesystem, git, shell)
    Coding { task_type: CodingTaskType },
    /// Tarea compleja que requiere planificacion
    ComplexTask { subtasks: Vec<String> },
    /// Conversacion casual
    Casual,
    /// Comando del sistema ("para", "deshaz", "estado")
    SystemCommand { command: String },
}

enum CodingTaskType {
    BugFix,
    NewFeature,
    Refactoring,
    Testing,
    Documentation,
    CodeReview,
    Debugging,
}
```

### 33.3 Flujo de Seleccion

```
Prompt del usuario llega
     |
     v
Paso 1: Clasificar intent (el LLM o heuristicas)
  - Contiene "implementa", "crea", "anade"? -> Coding::NewFeature
  - Contiene "bug", "error", "arregla"? -> Coding::BugFix
  - Contiene "busca", "encuentra", "investiga"? -> Research
  - Contiene "que es", "explica", "como"? -> Question
  - Contiene "refactoriza", "limpia", "mejora"? -> Coding::Refactoring
  - Si ambiguo: preguntarle al LLM con un prompt corto de clasificacion
     |
     v
Paso 2: Mapear intent -> modo de operacion
  Question     -> Chat (sin tools, respuesta directa)
  Research     -> Asistente (web search, browser)
  Coding       -> Programacion (filesystem, git, shell + hibrido)
  ComplexTask  -> Programacion o Assembly-line (segun complejidad)
  Casual       -> Chat
  SystemCommand -> Ejecutar directamente (no necesita agente)
     |
     v
Paso 3: Evaluar complejidad (si es Coding o ComplexTask)
  - Numero de archivos que probablemente se toquen
  - Si requiere diseno previo (nueva arquitectura?)
  - Si requiere multiples fases (diseno + impl + test)
  Simple (1-2 archivos, cambio directo)    -> Programacion, agente unico
  Medio (3-5 archivos, logica nueva)       -> Programacion, agente unico + planning
  Complejo (5+ archivos, diseno necesario) -> Assembly-line (si habilitado)
     |
     v
Paso 4: Evaluar paralelismo (seccion 25.6)
  - Hay items independientes? (N modulos, N archivos, N endpoints)
    SI + no comparten archivos -> fan-out (N agentes paralelos)
  - Hay fases independientes en el DAG?
    SI -> stages paralelos en assembly-line
  - Es un problema con multiples enfoques validos?
    SI + presupuesto lo permite -> ejecucion especulativa
  - Es una busqueda multi-fuente?
    SI -> N agentes de investigacion en paralelo
  - NINGUNA de las anteriores -> agente unico (secuencial)
     |
     v
Paso 5: Seleccionar modelo segun HybridStrategy
  Si modo Chat/Asistente -> modelo unico (local si disponible)
  Si modo Programacion   -> hibrido (local explore, cloud generate, local verify)
  Si modo Assembly-line  -> cada agente puede tener modelo distinto
  Si paralelo            -> todos el mismo modelo O cada uno optimizado para su subtarea
     |
     v
Paso 6: Verificar contra ModePolicy (seccion 28.2)
  Si el modo seleccionado > max_mode -> limitar
  Si el modo seleccionado < min_mode -> elevar
  Si paralelo pero max_parallel < N  -> reducir N o serializar
  Informar al usuario si se cambia de modo
```

### 33.4 Ejemplo Completo

```
Usuario: "Hay un bug en el login, cuando meto una contraseña incorrecta no muestra error"

Paso 1: Clasificar
  Palabras clave: "bug", "login", "no muestra error"
  Intent: Coding::BugFix

Paso 2: Modo
  Coding -> Programacion

Paso 3: Complejidad
  Bug fix tipicamente toca 1-2 archivos -> Simple
  -> Agente unico, sin assembly-line

Paso 4: Modelo
  HybridStrategy configurada:
    Exploracion: Ollama/qwen2.5-coder:7b (leer codigo, entender el bug)
    Generacion: Anthropic/claude-sonnet (generar el fix)
    Verificacion: Ollama/qwen2.5-coder:7b (verificar que compila y tests pasan)

Paso 5: ModePolicy
  max_mode = Autonomous -> Programacion esta dentro del rango -> OK

Resultado:
  "Modo: Programacion (bug fix)
   Modelo: hibrido local+cloud
   Agente: unico
   Empiezo a explorar el codigo de login..."
```

### 33.5 Overrides del Usuario

El usuario siempre puede forzar un modo:

```
"Usa assembly-line para esto"            -> Forzar Assembly-line
"Hazlo todo en local"                     -> Forzar LocalOnly
"Solo respondeme, no toques codigo"       -> Forzar Chat
"Usa Claude Opus para esto"               -> Override de modelo
```

---

## 34. Import/Export Universal de Configuracion

### 34.1 Principio

**Todo lo configurable debe ser exportable e importable.** El sistema no debe tener
configuraciones que solo existan en memoria o solo sean accesibles programaticamente.
Cualquier ajuste — desde la definicion de tiers hasta los prompts que tuenean el
comportamiento de los agentes — debe poder:

1. **Exportarse** a un fichero (disco o buffer en memoria)
2. **Importarse** desde un fichero (disco o buffer en memoria)
3. **Configurarse via API** programaticamente (esto ya es el default en Rust)

### 34.2 Que Es Exportable/Importable

| Categoria | Contenido | Ejemplo |
|-----------|-----------|---------|
| **Configuracion base** | `AiConfig` completo | provider, modelo, temperatura, max_tokens, system_prompt... |
| **Tiers custom** | Definiciones de RAG tiers | nombre, umbrales de relevancia, pipelines de procesamiento |
| **Politicas de permisos** | `PermissionPolicy`, danger thresholds | que tools se auto-aprueban, nivel de peligrosidad tolerado |
| **Prompts de sistema** | System prompts por modo/agente | prompt de coding, prompt de chat, prompt de research |
| **Prompts de tuneo** | Instrucciones de comportamiento | "siempre comenta en espanol", "prefiere funciones puras" |
| **Modelos registrados** | `ModelProfile` con metadata | nombre, provider, capacidades, VRAM estimada, ranking |
| **Modelos instalados** | `InstalledPackageRegistry` | modelos locales con path, tamano, fecha |
| **Estrategia hibrida** | `HybridStrategy` | que modelo para explore/generate/verify, triggers de escalation |
| **Modo de operacion** | `ModePolicy` | modo default, min_mode, max_mode, auto_escalate |
| **Butler config** | `ButlerConfig` | modo, recursos detectados, overrides |
| **Requisitos** | `RequirementRegistry` export | requisitos activos, resueltos, contradicciones |
| **Contenido protegido** | `PinnedContent` list | decisiones de diseno, requisitos importantes |
| **Iteracion** | `IterationStrategy` + `TaskContinuationPolicy` | presupuesto, limites, politica de continuacion |
| **Privacy** | Politicas por proveedor | que anonimizar, campos protegidos |
| **Plugins** | Lista de plugins + config | VCS plugins, MCP servers, tools personalizados |
| **Tools custom** | Definiciones de tools | nombre, schema de parametros, descripcion, handler |
| **Pipelines RAG** | Config de RAG | chunks, embeddings, scoring, tiers |
| **Undo policy** | Config de rollback | max_undo_depth, auto_snapshot, file_ownership |
| **Red/Anti-bloqueo** | Estrategias de red | user agents, delays, proxy config |
| **Assembly-line** | Definicion de stages | agentes, orden, back-communication rules |
| **Clasificador** | Reglas de clasificacion de prompts | keywords, mapeo intent->modo |
| **Navegacion UX** | Preferencias de UI | atajos, botones, colores |
| **Sesiones** | Sesiones completas | historial, estado, metadata |

### 34.3 Formatos de Archivo

```rust
/// Formato de archivo de configuracion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum ConfigFormat {
    /// JSON legible para humanos (default para export)
    Json,
    /// JSON5 con comentarios (solo import)
    Json5,
    /// TOML (popular en el ecosistema Rust)
    Toml,
    /// YAML (popular en DevOps/CI)
    Yaml,
    /// Bincode comprimido (rapido, compacto, no legible)
    #[cfg(feature = "binary-storage")]
    Bincode,
}
```

**Decision de diseno**: JSON como formato principal (legible, editable, portable).
TOML y YAML como alternativas populares. Bincode para backups rapidos y transferencias
entre instancias del mismo crate.

### 34.4 API de Export

```rust
/// Trait que cualquier componente configurable debe implementar
pub trait Exportable: Serialize {
    /// Nombre del componente (para el fichero)
    fn export_name(&self) -> &str;
    /// Version del schema (para migraciones futuras)
    fn schema_version(&self) -> u32;
    /// Exportar a fichero
    fn export_to_file(&self, path: &Path, format: ConfigFormat) -> Result<()>;
    /// Exportar a bytes en memoria
    fn export_to_bytes(&self, format: ConfigFormat) -> Result<Vec<u8>>;
    /// Exportar a string (solo formatos de texto)
    fn export_to_string(&self, format: ConfigFormat) -> Result<String>;
}

/// Trait para importar configuracion
pub trait Importable: DeserializeOwned {
    /// Importar desde fichero (auto-detecta formato por extension)
    fn import_from_file(path: &Path) -> Result<Self>;
    /// Importar desde bytes (auto-detecta formato)
    fn import_from_bytes(bytes: &[u8]) -> Result<Self>;
    /// Importar desde string (JSON/TOML/YAML)
    fn import_from_string(s: &str, format: ConfigFormat) -> Result<Self>;
    /// Validar despues de importar (coherencia, valores validos)
    fn validate(&self) -> Result<Vec<ValidationWarning>>;
    /// Migrar desde version anterior del schema
    fn migrate(value: serde_json::Value, from_version: u32) -> Result<Self>;
}
```

### 34.5 Paquete de Configuracion Completo

Para exportar/importar TODO el estado configurable de una vez:

```rust
/// Paquete completo de configuracion del sistema
#[derive(Serialize, Deserialize)]
struct ConfigBundle {
    /// Metadata del paquete
    meta: BundleMeta,
    /// Secciones incluidas (cada una opcional)
    sections: BundleSections,
}

struct BundleMeta {
    /// Version del formato del bundle
    bundle_version: u32,
    /// Fecha de exportacion
    exported_at: String,
    /// Version del crate que lo exporto
    crate_version: String,
    /// Descripcion opcional (ej: "Config de produccion")
    description: Option<String>,
    /// Checksum para verificar integridad
    checksum: Option<String>,
}

/// Todas las secciones son Option<> para permitir exportacion/importacion parcial
struct BundleSections {
    ai_config: Option<AiConfig>,
    permission_policy: Option<PermissionPolicy>,
    system_prompts: Option<HashMap<String, String>>,
    behavior_prompts: Option<Vec<BehaviorPrompt>>,
    model_profiles: Option<Vec<ModelProfile>>,
    installed_models: Option<Vec<InstalledPackage>>,
    hybrid_strategy: Option<HybridStrategy>,
    mode_policy: Option<ModePolicy>,
    butler_config: Option<ButlerConfig>,
    requirements: Option<Vec<TrackedRequirement>>,
    pinned_content: Option<Vec<PinnedContent>>,
    iteration_strategy: Option<IterationStrategy>,
    privacy_policies: Option<HashMap<String, PrivacyPolicy>>,
    plugin_configs: Option<Vec<PluginConfig>>,
    custom_tools: Option<Vec<ToolDefinition>>,
    rag_config: Option<RagPipelineConfig>,
    undo_policy: Option<UndoConfig>,
    network_config: Option<NetworkConfig>,
    assembly_stages: Option<Vec<StageDefinition>>,
    classifier_rules: Option<ClassifierConfig>,
    ux_preferences: Option<UxConfig>,
    custom_tiers: Option<Vec<TierDefinition>>,
}
```

### 34.6 Flujo de Export

```
ConfigBundle::export_all(&assistant)
     |
     v
Recopilar cada seccion del sistema:
  - assistant.config -> ai_config
  - assistant.permission_system -> permission_policy
  - assistant.prompt_templates -> system_prompts
  - assistant.model_ranker.profiles() -> model_profiles
  - assistant.hybrid_strategy -> hybrid_strategy
  - ... cada componente se serializa
     |
     v
Calcular checksum (SHA-256 del contenido sin el campo checksum)
     |
     v
Serializar al formato elegido:
  .json  -> serde_json::to_string_pretty()
  .toml  -> toml::to_string_pretty()
  .yaml  -> serde_yaml::to_string()
  .bin   -> bincode::serialize() + gzip
     |
     v
Escribir a disco o devolver bytes
```

### 34.7 Flujo de Import

```
ConfigBundle::import(path_or_bytes)
     |
     v
Paso 1: Detectar formato
  Extension .json -> JSON
  Extension .toml -> TOML
  Extension .yaml/.yml -> YAML
  Extension .bin -> Bincode
  Sin extension -> auto-detect por contenido
    (empieza con '{' -> JSON, empieza con '[' -> TOML section, etc.)
     |
     v
Paso 2: Deserializar a ConfigBundle
     |
     v
Paso 3: Verificar version
  Si bundle_version > actual -> error "version de bundle mas nueva que el crate"
  Si bundle_version < actual -> intentar migrate()
     |
     v
Paso 4: Verificar checksum (si presente)
  Recalcular y comparar -> warning si no coincide
     |
     v
Paso 5: Validar cada seccion presente
  Para cada Some(section) en BundleSections:
    section.validate() -> Vec<ValidationWarning>
  Acumular warnings
     |
     v
Paso 6: Aplicar (con opciones)
  ImportMode::Replace    -> Reemplazar toda la config existente
  ImportMode::Merge      -> Mergear: lo importado tiene prioridad
  ImportMode::MergeKeep  -> Mergear: lo existente tiene prioridad
  ImportMode::Preview    -> No aplicar, solo mostrar diff
     |
     v
Paso 7: Informar
  "Importadas 12 secciones, 3 warnings, 0 errores"
  Lista de cambios aplicados
```

### 34.8 Import/Export Parcial

No siempre se quiere todo. Se puede exportar/importar secciones individuales:

```rust
/// Exportar solo los prompts
assistant.system_prompts.export_to_file("my_prompts.json", Json)?;

/// Importar solo la estrategia hibrida
let strategy = HybridStrategy::import_from_file("hybrid.toml")?;
assistant.set_hybrid_strategy(strategy);

/// Exportar solo los modelos instalados
assistant.package_registry.export_to_file("models.json", Json)?;

/// Importar tiers custom
let tiers = Vec::<TierDefinition>::import_from_file("tiers.yaml")?;
assistant.rag_config.set_custom_tiers(tiers);

/// Exportar la configuracion de permisos
assistant.permission_policy.export_to_file("permissions.json", Json)?;

/// Importar assembly-line desde otro proyecto
let stages = Vec::<StageDefinition>::import_from_file("pipeline.json")?;
assistant.set_assembly_stages(stages);
```

### 34.9 Perfiles de Configuracion

Los bundles permiten crear **perfiles** que el usuario puede intercambiar:

```
~/.ai_assistant/
  profiles/
    default.json          <- Config basica
    coding_rust.json      <- Optimizado para Rust (prompts especificos, tiers ajustados)
    coding_python.json    <- Optimizado para Python
    research.json         <- Modo investigacion (browser, web search habilitados)
    secure.json           <- Maxima seguridad (todo requiere aprobacion)
    autonomous.json       <- Modo autonomo (auto-approve casi todo)
    local_only.json       <- Solo modelos locales, sin cloud
    team_shared.toml      <- Config compartida del equipo (via git)
```

```rust
/// Cargar un perfil por nombre
assistant.load_profile("coding_rust")?;

/// Guardar la config actual como perfil
assistant.save_profile("mi_config_custom")?;

/// Listar perfiles disponibles
let profiles = ConfigBundle::list_profiles()?;

/// Comparar perfiles (diff)
let diff = ConfigBundle::diff("default", "coding_rust")?;
```

### 34.10 Migracion de Versiones

Cuando el schema cambia entre versiones del crate:

```rust
impl Importable for AiConfig {
    fn migrate(value: serde_json::Value, from_version: u32) -> Result<Self> {
        match from_version {
            1 => {
                // v1 -> v2: "model" se renombro a "default_model"
                let mut v = value;
                if let Some(model) = v.get("model").cloned() {
                    v["default_model"] = model;
                    v.as_object_mut().unwrap().remove("model");
                }
                Self::migrate(v, 2) // Recurrir para siguientes migraciones
            }
            2 => {
                // v2 -> v3: "temperature" ahora es Option<f32>
                serde_json::from_value(value).map_err(Into::into)
            }
            3 => serde_json::from_value(value).map_err(Into::into), // version actual
            _ => Err(anyhow!("Version {} no soportada", from_version)),
        }
    }
}
```

### 34.11 Import/Export como Tool del Agente

El propio agente puede usar import/export como herramientas:

```
Usuario: "Exporta mi configuracion actual"
Agente: [Tool: config_export] -> genera my_config.json
        "He exportado tu configuracion a my_config.json"

Usuario: "Carga la configuracion de coding_rust"
Agente: [Tool: config_import] -> aplica perfil coding_rust
        "He cargado el perfil coding_rust. Cambios aplicados:
         - Modelo de generacion: claude-sonnet
         - System prompt: especializado en Rust
         - Tiers RAG: ajustados para documentacion de crates"

Usuario: "Ensenale mi configuracion a otro usuario"
Agente: [Tool: config_export] -> genera bundle
        "Aqui tienes tu configuracion exportada. El otro usuario
         puede importarla con:
         assistant.import_config('tu_config.json')?"
```

### 34.12 Seguridad en Import

```rust
struct ImportSecurity {
    /// No importar API keys/tokens (se omiten automaticamente)
    strip_secrets: bool,  // default: true
    /// Validar que los paths en la config existen
    validate_paths: bool, // default: true
    /// No permitir que la config importe plugins de fuentes no confiables
    restrict_plugins: bool, // default: true
    /// Maxmio tamano de archivo a importar (evitar DoS)
    max_file_size: usize, // default: 10MB
    /// Verificar firma digital del bundle (si esta presente)
    verify_signature: bool, // default: false
}
```

Campos sensibles que se omiten en export por defecto:

| Campo | Razon | Export default |
|-------|-------|---------------|
| `api_key` | Secreto | **OMITIDO** (placeholder `"***"`) |
| `api_keys` (rotacion) | Secretos | **OMITIDO** |
| `proxy_auth` | Credenciales | **OMITIDO** |
| `encryption_key` | Secreto | **OMITIDO** |
| `session_tokens` | Efimeros | **OMITIDO** |

Para exportar incluyendo secretos (ej: backup encriptado):

```rust
assistant.export_config_with_options(ExportOptions {
    include_secrets: true,
    encrypt: true,  // AES-256-GCM, pide password
    ..Default::default()
})?;
```

### 34.13 Formato de Ejemplo (JSON)

Un archivo exportado se veria asi:

```json
{
  "meta": {
    "bundle_version": 1,
    "exported_at": "2026-02-15T10:30:00Z",
    "crate_version": "0.1.0",
    "description": "Config de desarrollo Rust",
    "checksum": "sha256:a1b2c3..."
  },
  "sections": {
    "ai_config": {
      "provider": "Ollama",
      "default_model": "qwen2.5-coder:7b",
      "temperature": 0.7,
      "max_tokens": 4096,
      "api_key": "***",
      "system_prompt": "Eres un asistente de programacion Rust..."
    },
    "hybrid_strategy": {
      "explore_provider": "Ollama",
      "explore_model": "qwen2.5-coder:7b",
      "generate_provider": "Anthropic",
      "generate_model": "claude-sonnet-4-5-20250929",
      "verify_provider": "Ollama",
      "verify_model": "qwen2.5-coder:7b",
      "escalation_triggers": ["complexity_high", "security_critical"]
    },
    "mode_policy": {
      "default_mode": "Programming",
      "max_mode": "Autonomous",
      "auto_escalate": true
    },
    "permission_policy": {
      "auto_approve_reads": true,
      "auto_approve_writes": false,
      "danger_threshold": 50,
      "require_approval_above": 70
    },
    "custom_tiers": [
      {
        "name": "rust-docs",
        "priority": 1,
        "relevance_threshold": 0.8,
        "sources": ["docs.rs", "crate-local-docs"]
      }
    ],
    "behavior_prompts": [
      {
        "name": "coding_style",
        "prompt": "Siempre usa Result<T, E> en vez de unwrap(). Prefer &str sobre String en parametros.",
        "applies_to": ["Programming", "AssemblyLine"]
      }
    ]
  }
}
```

---

## 35. Escalabilidad de Busqueda Vectorial

### 35.1 Estado Actual

| Componente | Lo que tenemos | Limitacion |
|-----------|---------------|-----------|
| **SQLite** | `rusqlite` con feature `bundled` | Solo FTS5 (texto), no vectorial |
| **Busqueda vectorial** | Cosine similarity manual en Rust (`rag.rs`) | O(n) lineal — hay que comparar contra TODOS los vectores |
| **Embeddings** | Calculados via API externa (provider) | Funcional, no es cuello de botella |

**Rendimiento actual estimado**:
- <1,000 documentos: instantaneo (~1ms)
- 1,000-10,000: aceptable (~10-50ms)
- 10,000-100,000: lento (~100-500ms)
- 100,000+: inaceptable (segundos)

Para un asistente personal o de proyecto, 1K-10K documentos es lo normal.
Pero si se quiere escalar (knowledge bases corporativas, RAG sobre repos grandes),
necesitamos algo mejor.

### 35.2 Tiers de Escalabilidad

```
Tier 0 (actual): Cosine similarity en Rust — O(n)
   |
   v  hasta ~10K docs
Tier 1: sqlite-vec — indice vectorial dentro de SQLite
   |
   v  hasta ~100K docs
Tier 2: Embedded vector DB (LanceDB, usearch) — indices ANN optimizados
   |
   v  hasta ~10M docs
Tier 3: Vector DB dedicada (Qdrant, Milvus, Weaviate) — servicio separado
   |
   v  hasta ~1B docs
Tier 4: Vector DB cloud/managed (Pinecone, Qdrant Cloud) — escala ilimitada
```

### 35.3 Tier 1: sqlite-vec

**Que es**: Extension de SQLite que anade busqueda vectorial (ANN) usando `vec0` virtual tables.

| Aspecto | Detalle |
|---------|---------|
| **Ventaja** | Se integra con nuestro rusqlite existente. Sin dependencia externa. Fichero unico |
| **Limitacion** | No tiene bindings Rust oficiales maduros. Requiere compilar la extension C. Rendimiento medio |
| **Escala** | ~100K vectores razonable |
| **Integracion** | `rusqlite::Connection::load_extension("vec0")` |

```rust
// Ejemplo de uso (conceptual)
conn.execute("CREATE VIRTUAL TABLE docs_vec USING vec0(embedding float[384])", [])?;
conn.execute("INSERT INTO docs_vec(rowid, embedding) VALUES (?, ?)", [id, &embedding_bytes])?;

// Busqueda: los K mas cercanos
let results = conn.prepare(
    "SELECT rowid, distance FROM docs_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
)?.query_map([&query_embedding, &k], |row| { ... })?;
```

### 35.4 Tier 2: Embedded Vector DB

#### Opcion A: LanceDB (Rust-native) — RECOMENDADA

| Aspecto | Detalle |
|---------|---------|
| **Lenguaje** | Rust puro (core + SDK) |
| **Licencia** | Apache 2.0 (100% open source, sin restricciones) |
| **Arquitectura** | **Embebido** — se enlaza directamente en el binario, sin servidor, sin Docker |
| **Formato** | Lance (columnar, comprimido, versionado, basado en Apache Arrow) |
| **Indices** | IVF-PQ, IVF-Flat, HNSW-PQ, HNSW-SQ, IVF-HNSW-SQ (hibrido) |
| **Escala probada** | 700M vectores en produccion (1.6 TB disco). 1B+ demostrado sobre S3 |
| **Crate** | `lancedb` en crates.io (SDK oficial, first-class) |
| **Storage** | Disco local, S3, GCS, Azure Blob — disk-first por diseno |

```rust
// Ejemplo conceptual
let db = lancedb::connect("data/vectors").await?;
let table = db.create_table("docs", data).await?;
let results = table.search(&query_vec).limit(10).execute().await?;
```

**Mejor opcion para escalar hacia arriba** desde nuestro estado actual:
- Embebido (no requiere servicio externo, como SQLite)
- Rust-native (no FFI con C, no gRPC, zero-copy via Arrow)
- Rendimiento excelente: >0.95 recall en ~5ms sobre 1M vectores
- Formato Lance es columnar, comprimido, y con versionado automatico
- Soporte de full-text search ademas de vectorial

#### Opcion B: usearch (Rust bindings)

| Aspecto | Detalle |
|---------|---------|
| **Ventaja** | Muy rapido (C++ con bindings Rust). Pequeno. Solo busqueda ANN |
| **Limitacion** | Solo indice vectorial, no es DB completa. Sin persistencia nativa (hay que gestionar el storage) |
| **Escala** | Millones de vectores |
| **Crate** | `usearch` |

Bueno para: anadirlo como acelerador al sistema actual sin cambiar la arquitectura.
El indice se mantiene en memoria con flush a disco.

### 35.5 Tier 3: Vector DB Dedicada (self-hosted, GRATUITA)

> **IMPORTANTE**: Los Tier 3 son **todos open source y gratuitos** para self-hosting.
> No requieren licencia de pago. Solo el Tier 4 (cloud managed) tiene coste.

| DB | Lenguaje | Licencia | Escala | Rust client | Self-host gratis? |
|----|----------|----------|--------|-------------|-------------------|
| **Qdrant** | Rust | Apache 2.0 | Millones-Billones | `qdrant-client` (oficial, gRPC) | **Si** (Docker/binario/compilar) |
| **Milvus** | Go/C++ | Apache 2.0 | Billones | `milvus-sdk-rust` (community) | **Si** (Docker/Kubernetes) |
| **Weaviate** | Go | BSD-3 | Millones | HTTP API | **Si** (Docker/binario) |
| **ChromaDB** | Python | Apache 2.0 | Millones | HTTP API | **Si** (pip/Docker) |

> Diferencia clave con Tier 2: requieren un **proceso/servidor separado**.
> En Tier 2 (LanceDB), la DB vive dentro de tu proceso Rust.
> En Tier 3, hay un servidor aparte al que te conectas via red (gRPC/HTTP).

### 35.6 LanceDB vs Qdrant: Comparacion Detallada

Esta es la comparacion mas relevante porque ambos estan escritos en Rust:

#### 35.6.1 Arquitectura

| Aspecto | LanceDB | Qdrant |
|---------|---------|--------|
| **Modo principal** | Embebido (in-process) | Cliente-servidor |
| **Embebido en Rust?** | **SI** — native, first-class | **NO** — el crate Rust es solo un cliente gRPC |
| **Embebido en Python?** | Si (FFI al core Rust) | Si (reimplementacion Python, solo dev/testing) |
| **Servidor propio?** | No (LanceDB Cloud es SaaS aparte) | Si — REST (6333) + gRPC (6334) |
| **Docker necesario?** | No | Si (o compilar desde source) |
| **Latencia de red** | **Zero** (in-process) | Presente (incluso en localhost hay overhead) |
| **Clustering distribuido** | No (single-node, escala con mas disco/S3) | **Si** (sharding, replicacion nativa) |
| **Analogia** | "El SQLite de vector DBs" | "El PostgreSQL de vector DBs" |

> **Hallazgo critico**: El "modo embebido" de Qdrant (`QdrantClient(":memory:")`)
> solo existe en **Python**. Es una reimplementacion Python para dev/testing.
> En Rust, **obligatoriamente** necesitas un servidor Qdrant corriendo.
> LanceDB es embebido en **todos** los lenguajes (Rust, Python, TypeScript).

#### 35.6.2 Rendimiento

| Metrica | LanceDB | Qdrant | Notas |
|---------|---------|--------|-------|
| **Latencia busqueda** | 40-60ms (sin indice) / **3-5ms** (con IVF-PQ) | **20-30ms** | Qdrant mas rapido raw; LanceDB comparable con indice optimizado |
| **Recall@1** | ~88% (default) / **>95%** (con refine_factor) | **~95%** | Ambos llegan a >95% con tuning |
| **QPS (1M vectores)** | ~500-1000 (depende de nprobes) | ~1000-2000 | Qdrant mayor throughput |
| **QPS (50M vectores)** | No benchmarked publicamente | ~40 (99% recall) / ~360 (90% recall) | Qdrant tiene datos a escala |
| **Insercion** | Rapida (append columnar) | Rapida (HNSW incremental) | Similar |

#### 35.6.3 Consumo de Recursos

| Recurso | LanceDB | Qdrant | Ganador |
|---------|---------|--------|---------|
| **RAM idle** | **~4 MB** | ~400 MB | LanceDB (100x menos) |
| **RAM busqueda activa** | **~150 MB** | ~400 MB+ | LanceDB |
| **RAM 1M vectores** | Proporcional a nprobes (disk-first) | ~1.2 GB (in-memory) | LanceDB |
| **Modo bajo RAM** | Default (siempre disk-first) | `on_disk=True` (reduce pero sigue ~100MB) | LanceDB |
| **Disco 700M vectores** | 1.6 TB (Lance columnar) | ~2-3 TB estimado (HNSW graph) | LanceDB |
| **CPU idle** | Despreciable (no hay servidor) | Bajo (servidor en espera) | LanceDB |
| **CPU busqueda** | Proporcional a nprobes | Proporcional a ef | Similar |

#### 35.6.4 Indices y Cuantizacion

| Aspecto | LanceDB | Qdrant |
|---------|---------|--------|
| **Tipos de indice vectorial** | IVF-PQ, IVF-Flat, HNSW-PQ, HNSW-SQ, **IVF-HNSW-SQ** (hibrido) | HNSW (unico tipo) |
| **Cuantizacion escalar** | Si (SQ en HNSW-SQ) | Si (float32 -> uint8, 4x compresion) |
| **Cuantizacion producto** | Si (PQ en IVF-PQ, HNSW-PQ) | Si |
| **Cuantizacion binaria** | No | **Si (32x compresion, hasta 40x mas rapido)** |
| **Indices adicionales** | BTree, Bitmap, Label List, Full-text | Payload index (filtrado) |
| **Diversidad** | **Mayor** (6 tipos de indice vectorial) | HNSW unico pero con quantization rica |

#### 35.6.5 Funcionalidades Adicionales

| Feature | LanceDB | Qdrant |
|---------|---------|--------|
| **Full-text search** | Si (integrado) | Si (sparse vectors) |
| **Filtrado durante busqueda** | Si (SQL-like) | **Si (custom HNSW con filtrado nativo, muy eficiente)** |
| **Versionado de datos** | **Si (automatico, tipo git)** | No |
| **Multi-tenancy** | Via tablas | Via collections + payload filtering |
| **Replicacion** | No | **Si (distributed mode)** |
| **Sharding** | No | **Si (horizontal scaling)** |
| **Object storage** | **Si (S3, GCS, Azure nativo)** | No (solo disco local) |
| **Snapshots/Backup** | Via versionado Lance | Si (snapshot API) |

#### 35.6.6 Integracion con ai_assistant

| Criterio | LanceDB | Qdrant | Para nuestro crate |
|----------|---------|--------|--------------------|
| **Se enlaza en el binario** | **Si** | No (servidor separado) | LanceDB: una sola dependencia. Qdrant: hay que instalar y gestionar un servidor |
| **Sin Docker** | **Si** | No (o compilar) | LanceDB: cero requisitos externos |
| **Feature flag limpio** | `lancedb = ["dep:lancedb"]` | `qdrant = ["dep:qdrant-client"]` + runtime Qdrant | LanceDB: el feature flag es todo lo que necesitas |
| **Async** | Si (tokio) | Si (tonic/gRPC) | Ambos OK con nuestro async-runtime |
| **Tamano del binario** | ~15 MB extra | ~5 MB (solo cliente gRPC) | LanceDB mas pesado pero incluye TODO |

#### 35.6.7 Resumen: Cuando Usar Cada Uno

```
Usa LanceDB (Tier 2) si:
  ✓ Quieres cero dependencias externas (como SQLite)
  ✓ RAM limitada (edge, laptop, embedded)
  ✓ Hasta ~10M vectores en un solo nodo
  ✓ Necesitas que todo sea un solo binario Rust
  ✓ Quieres versionado de datos y object storage
  ✓ Es un uso personal/equipo pequeno

Usa Qdrant (Tier 3) si:
  ✓ Necesitas clustering distribuido (multiples nodos)
  ✓ Necesitas replicacion y alta disponibilidad
  ✓ Tienes >10M vectores con requisitos de latencia estrictos
  ✓ Ya tienes Docker/Kubernetes en tu infra
  ✓ Multiples aplicaciones consultan la misma base vectorial
  ✓ Necesitas binary quantization (32x compresion)
```

**Para ai_assistant, la recomendacion es clara: LanceDB como Tier 2 default.**
Qdrant solo si el usuario tiene necesidades de escala corporativa.

### 35.7 Tier 4: Cloud/Managed (unico tier de pago)

> Este es el **unico tier que implica coste**. Los tiers 0-3 son todos gratuitos.

| Servicio | Basado en | Free tier | Escala | Coste |
|----------|-----------|-----------|--------|-------|
| **Qdrant Cloud** | Qdrant | 1GB gratis | Ilimitada | Desde ~$25/mes |
| **LanceDB Cloud** | LanceDB | En desarrollo | Ilimitada | Serverless (pago por uso) |
| **Pinecone** | Propietario | 100K vectores | Ilimitada | Desde ~$70/mes |
| **Weaviate Cloud** | Weaviate | Sandbox gratis | Ilimitada | Desde ~$25/mes |

### 35.8 Diseno de Integracion: VectorBackend Trait

Para soportar todos los tiers sin cambiar el codigo de RAG:

```rust
/// Trait que abstrae el backend de busqueda vectorial
pub trait VectorBackend: Send + Sync {
    /// Insertar un vector con su ID y metadata
    fn upsert(&mut self, id: &str, vector: &[f32], metadata: &str) -> Result<()>;
    /// Buscar los K vectores mas cercanos
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>>;
    /// Borrar un vector por ID
    fn delete(&mut self, id: &str) -> Result<()>;
    /// Numero de vectores almacenados
    fn count(&self) -> Result<usize>;
    /// Flush a disco (si aplica)
    fn flush(&mut self) -> Result<()>;
    /// Info del backend para diagnostico
    fn backend_info(&self) -> BackendInfo;
    /// Exportar todos los vectores (para migracion)
    fn export_all(&self) -> Result<Vec<VectorRecord>>;
    /// Importar vectores en bulk (para migracion)
    fn import_bulk(&mut self, records: Vec<VectorRecord>) -> Result<usize>;
}

struct VectorMatch {
    id: String,
    score: f32,      // 0.0 - 1.0 (cosine similarity)
    metadata: String,
}

struct VectorRecord {
    id: String,
    vector: Vec<f32>,
    metadata: String,
}

struct BackendInfo {
    name: &'static str,         // "InMemory", "SqliteVec", "LanceDB", "Qdrant"
    tier: u8,                    // 0, 1, 2, 3, 4
    vector_count: usize,
    estimated_size_bytes: u64,
    supports_filtering: bool,
    supports_persistence: bool,
}

/// Tier 0: lo que tenemos ahora
struct InMemoryVectorBackend {
    vectors: HashMap<String, (Vec<f32>, String)>,
}

/// Tier 1: sqlite-vec
#[cfg(feature = "sqlite-vec")]
struct SqliteVecBackend {
    conn: rusqlite::Connection,
}

/// Tier 2: LanceDB
#[cfg(feature = "lancedb")]
struct LanceBackend {
    db: lancedb::Database,
    table_name: String,
}

/// Tier 3: Qdrant
#[cfg(feature = "qdrant")]
struct QdrantBackend {
    client: qdrant_client::QdrantClient,
    collection: String,
}
```

### 35.9 Auto-Seleccion de Backend

El Butler (seccion 29) puede detectar el volumen y recomendar:

```
Documentos en el proyecto: 500
  -> Tier 0 (actual) es suficiente. No se necesita nada adicional.

Documentos en el proyecto: 15,000
  -> "Recomiendo habilitar el feature 'lancedb' para busqueda vectorial mas rapida.
      Actualmente la busqueda tarda ~150ms, con LanceDB seria ~5ms."

Documentos en el proyecto: 500,000
  -> "Recomiendo usar Qdrant como servicio externo. Puedo instalarlo via Docker
      si tienes Docker disponible (seccion 20)."
```

### 35.10 Migracion Transparente

Cambiar de backend no deberia requerir re-indexar manualmente. La migracion
se basa en `export_all()` e `import_bulk()` del trait VectorBackend.

#### 35.10.1 Que Es la Migracion Transparente

Cuando el usuario cambia de tier (ej: habilita el feature `lancedb` en Cargo.toml),
el sistema detecta que hay datos en el backend anterior y los migra al nuevo backend
**sin que el usuario tenga que re-calcular embeddings ni re-indexar nada**.

Lo que se migra:
- Los vectores (embeddings) ya calculados
- Los IDs de cada documento
- La metadata asociada a cada vector

Lo que **no** se migra (se reconstruye):
- Los indices (IVF-PQ, HNSW, etc.) — se reconstruyen automaticamente al insertar

#### 35.10.2 Flujo Detallado

```
ANTES: El usuario tenia Tier 0 (InMemoryVectorBackend)
       con 8,000 vectores almacenados en SQLite (como blobs con FTS5)

PASO 1: Cambio de feature flag
  # Cargo.toml
  [features]
  rag = ["rusqlite", "aes-gcm"]
  rag-lance = ["rag", "lancedb"]     <- nuevo feature

  El usuario recompila con: cargo build --features rag-lance

PASO 2: Deteccion automatica al iniciar
  fn initialize_vector_backend(config: &RagConfig) -> Box<dyn VectorBackend> {
      // Seleccionar backend segun features habilitados
      #[cfg(feature = "lancedb")]
      let new_backend = LanceBackend::new(&config.lance_path)?;

      #[cfg(not(feature = "lancedb"))]
      let new_backend = InMemoryVectorBackend::new();

      // Detectar si hay datos en un backend anterior
      let migration_state = MigrationState::load(&config.data_dir)?;
      if migration_state.previous_backend != new_backend.backend_info().name {
          // Backend cambio -> ofrecer migracion
          return handle_migration(migration_state, new_backend);
      }

      new_backend
  }

PASO 3: Migracion
  struct MigrationState {
      /// Backend anterior (guardado en un .json pequeno)
      previous_backend: String,
      /// Numero de vectores en el backend anterior
      previous_count: usize,
      /// Se completo la migracion?
      migrated: bool,
      /// Timestamp de migracion
      migrated_at: Option<String>,
  }

  fn handle_migration(state: MigrationState, new: Box<dyn VectorBackend>) {
      println!("Detectado cambio de backend: {} -> {}",
          state.previous_backend, new.backend_info().name);
      println!("Hay {} vectores en el backend anterior.", state.previous_count);

      // En modo interactivo: preguntar al usuario
      // En modo automatico: migrar directamente
      println!("Migrando {} vectores a {}...", state.previous_count, new.backend_info().name);

      // Leer todos los vectores del backend anterior
      let old_backend = load_previous_backend(&state)?;
      let records: Vec<VectorRecord> = old_backend.export_all()?;

      // Insertar en el nuevo backend (en bulk, eficiente)
      let imported = new.import_bulk(records)?;

      // El nuevo backend reconstruye sus indices automaticamente
      // (LanceDB construye IVF-PQ al insertar; Qdrant construye HNSW)

      // Guardar estado
      state.migrated = true;
      state.migrated_at = Some(now());
      state.save()?;

      println!("Migracion completada: {} vectores importados.", imported);
      // Opcionalmente: borrar datos del backend anterior para liberar espacio
  }

PASO 4: Operacion normal
  El codigo de RAG usa VectorBackend trait.
  No sabe ni le importa si detras hay InMemory, LanceDB, o Qdrant.
  Simplemente llama a search(), upsert(), delete().

PASO 5: Rollback (si algo sale mal)
  Si la migracion falla a mitad:
    - Los datos del backend anterior siguen intactos (no se borran hasta confirmar)
    - Se puede volver al backend anterior sin perdida
    - El MigrationState guarda el progreso para continuar desde donde se quedo
```

#### 35.10.3 Migracion en Produccion (Tier 2 -> Tier 3)

Para el caso de migrar de LanceDB a Qdrant (necesidad de clustering):

```
1. El usuario instala Qdrant (Docker o binario)
2. Configura la conexion en AiConfig:
     config.qdrant_url = "http://localhost:6334"
3. Habilita feature qdrant y recompila
4. Al iniciar:
   "Detectado Qdrant disponible en localhost:6334.
    Hay 50,000 vectores en LanceDB local.
    Migrar a Qdrant? Esto permite clustering y replicacion.
    Tiempo estimado: ~2 minutos.
    [Si / No / Mantener ambos (dual-write)]"
5. Si elige "Mantener ambos":
   - Writes van a ambos backends
   - Reads van a Qdrant (mas rapido con HNSW)
   - Se puede desactivar dual-write cuando se confirme que Qdrant funciona bien
```

#### 35.10.4 Por Que No Hay Que Re-calcular Embeddings

Los embeddings son los vectores numericos que representan el contenido de cada documento.
Calcularlos requiere llamar al modelo de embeddings (API o local), que es **la parte lenta
y costosa** del proceso de indexacion.

La migracion solo mueve vectores ya calculados entre backends. Los vectores no cambian —
son los mismos numeros flotantes independientemente de si estan en memoria, en SQLite,
en LanceDB, o en Qdrant. Lo que cambia es la estructura de indice que acelera la busqueda
(IVF-PQ, HNSW, etc.), pero esos indices se reconstruyen automaticamente al insertar.

---

## 36. Sistema Distribuido Rediseñado — Alta Disponibilidad y Tolerancia a Fallos

### 36.1 Estado Actual y Problemas

El modulo `distributed.rs` original tenia limitaciones significativas. Con la implementacion del feature
`distributed-network`, la mayoria han sido resueltas:

| Componente | Estado | Funcional? |
|-----------|--------|-----------|
| **DHT** (Kademlia 160-bit) | Estructuras de datos completas | Solo local — sin networking real |
| **CRDTs** (GCounter, PNCounter, LWWRegister, ORSet, LWWMap) | Implementacion completa | Si — merge funciona correctamente |
| **MapReduce** | Pipeline Map→Shuffle→Reduce | **Paralelo via rayon** (antes single-thread) |
| **DistributedCoordinator** | Combina DHT + MapReduce + CRDTs | Solo coordinador local |
| **P2P** (p2p.rs) | Config + mensajes + reputacion | Networking = stubs |
| **Consistent Hashing** (`consistent_hash.rs`) | **IMPLEMENTADO** | Si — vnodes, add/remove, replication factor |
| **Failure Detection** (`failure_detector.rs`) | **IMPLEMENTADO** | Si — Phi Accrual + HeartbeatManager |
| **Merkle Sync** (`merkle_sync.rs`) | **IMPLEMENTADO** | Si — SHA-256 tree, diff, proofs, anti-entropy |
| **Node Security** (`node_security.rs`) | **IMPLEMENTADO** | Si — mutual TLS, join tokens, challenge-response |
| **QUIC Networking** (`distributed_network.rs`) | **IMPLEMENTADO** | Si — quinn transport, replication, event loop, LAN discovery, peer exchange, join token validation, anti-entropy sync, reputation, probation, max_connections, enforce_min_copies |
| **Distributed VectorDb** (`vector_db.rs`) | **IMPLEMENTADO** | Si — fan-out search, replication wrapper |

**Estado actual (post-implementacion completa):**
1. ~~No hay networking real~~ → **RESUELTO**: QUIC via quinn con mutual TLS (`distributed_network.rs`)
2. ~~Sin tolerancia a fallos~~ → **RESUELTO**: Phi Accrual Failure Detector + HeartbeatManager + enforce_min_copies (`failure_detector.rs`, `distributed_network.rs`)
3. ~~Sin replicacion~~ → **RESUELTO**: Consistent hashing + configurable replication factor + enforce_min_copies automatico (`consistent_hash.rs`, `distributed_network.rs`)
4. ~~Sin seguridad inter-nodo~~ → **RESUELTO**: Mutual TLS + join tokens + HMAC challenge-response + constant-time comparison + secure RNG (`node_security.rs`)
5. ~~Sin descubrimiento automatico~~ → **RESUELTO**: LAN discovery via UDP broadcast + peer exchange protocol (`distributed_network.rs`)
6. ~~Sin reputacion de nodos~~ → **RESUELTO**: Reputation tracking (0.0-1.0) + probation period para nodos nuevos (`distributed_network.rs`)
7. ~~Sin anti-entropy sync wired~~ → **RESUELTO**: Merkle tree sync periodico integrado en event loop (`distributed_network.rs`, `merkle_sync.rs`)
8. **Sin consenso**: No hay Raft ni Paxos — usa eventual consistency con quorum reads/writes
9. **Total**: 113 tests nuevos (12 consistent_hash + 14 failure_detector + 13 merkle_sync + 27 node_security + 47 distributed_network)

### 36.2 Arquitectura Propuesta

Cada nodo es una instancia de `ai_assistant` que:
- Tiene su propia VectorDb local, RAG, agentes, y knowledge graph
- Se descubre y comunica con otros nodos via protocolo P2P
- Replica datos criticos a N-1 nodos adicionales (factor configurable)
- Puede delegar trabajo (MapReduce, busquedas, agentes) a otros nodos

```
                    +-----------+
                    |  Nodo A   |
                    | VectorDb  |
                    |   RAG     |
               +--->| Agentes   |<---+
               |    +-----------+    |
               |                     |
         heartbeat              heartbeat
               |                     |
        +------v----+         +------v----+
        |  Nodo B   |<------->|  Nodo C   |
        | VectorDb  | replicas| VectorDb  |
        |   RAG     |         |   RAG     |
        | Agentes   |         | Agentes   |
        +-----------+         +-----------+
```

### 36.3 Replicacion con Factor Configurable

```rust
struct ReplicationConfig {
    /// Numero minimo de copias de cada dato (incluido el nodo origen)
    /// Valor por defecto: 2 (el dato esta en 2 nodos)
    min_copies: usize,
    /// Numero maximo de copias (para no saturar la red)
    /// Valor por defecto: 3
    max_copies: usize,
    /// Modo de escritura: sincrono (espera confirmacion de N nodos)
    /// o asincrono (escribe local y replica en background)
    write_mode: WriteMode,
    /// Quorum de lectura: cuantos nodos deben responder para confirmar lectura
    /// Regla: R + W > N para consistencia fuerte
    read_quorum: usize,
    /// Quorum de escritura
    write_quorum: usize,
}

enum WriteMode {
    /// Esperar confirmacion de write_quorum nodos antes de responder OK
    Synchronous,
    /// Escribir local y replicar en background (mas rapido, eventual consistency)
    Asynchronous,
}
```

**Principio clave**: No todo esta en todos los nodos. Usar consistent hashing para
distribuir los datos. Cada dato se asigna a `min_copies` nodos, no a todos.

### 36.4 Particionamiento (Consistent Hashing)

Anillo virtual con vnodes, como Cassandra/DynamoDB:

```
Anillo hash (0..2^64):

    Nodo A (vnodes: v1, v5, v9)
        |
   v1---*----v2----*----v3
   |                      |
   |    Anillo Virtual     |
   |                      |
   v9---*----v8----*----v7
        |
    Nodo B (vnodes: v2, v6, v8)
    Nodo C (vnodes: v3, v7, v9)
```

- Cada nodo fisico tiene multiples vnodes (16-256) distribuidos en el anillo
- Un dato con hash H se asigna al primer vnode con hash >= H
- Las `min_copies` replicas van a los siguientes vnodes (de nodos fisicos diferentes)
- Al anadir/quitar un nodo, solo se reasigna ~1/N de los datos
- Mas vnodes = distribucion mas uniforme

```rust
struct ConsistentHashRing {
    /// (hash_position, node_id) ordenados por hash
    vnodes: Vec<(u64, NodeId)>,
    /// Numero de vnodes por nodo fisico
    vnodes_per_node: usize,
    /// Factor de replicacion
    replication_factor: usize,
}

impl ConsistentHashRing {
    /// Dado un key, devolver los N nodos responsables (primario + replicas)
    fn get_nodes(&self, key: &str) -> Vec<NodeId> {
        let hash = hash_key(key);
        let mut nodes = Vec::new();
        let mut seen = HashSet::new();

        // Recorrer el anillo desde la posicion del hash
        let start = self.vnodes.partition_point(|(h, _)| *h < hash);
        for i in 0..self.vnodes.len() {
            let idx = (start + i) % self.vnodes.len();
            let (_, node_id) = &self.vnodes[idx];
            if seen.insert(*node_id) {
                nodes.push(*node_id);
                if nodes.len() >= self.replication_factor {
                    break;
                }
            }
        }
        nodes
    }
}
```

### 36.5 Tolerancia a Fallos

#### 36.5.1 Heartbeat y Failure Detection

**Phi Accrual Failure Detector** (como Cassandra) — mas inteligente que un timeout fijo:

```rust
struct PhiAccrualDetector {
    /// Historial de intervalos entre heartbeats (ventana deslizante)
    intervals: VecDeque<Duration>,
    /// Ultimo heartbeat recibido
    last_heartbeat: Instant,
    /// Umbral phi para declarar un nodo muerto (tipicamente 8-12)
    phi_threshold: f64,
}

impl PhiAccrualDetector {
    /// Calcular el valor phi actual
    /// phi = -log10(P(heartbeat no ha llegado aun | historial))
    /// Si phi > threshold -> nodo probablemente muerto
    fn phi(&self) -> f64 {
        let elapsed = self.last_heartbeat.elapsed();
        let mean = self.mean_interval();
        let variance = self.variance();

        // Distribucion normal acumulativa
        let y = (elapsed.as_secs_f64() - mean) / variance.sqrt();
        let prob = 0.5 * (1.0 + erf(y / std::f64::consts::SQRT_2));

        -prob.log10()
    }

    /// El nodo se considera sospechoso?
    fn is_suspicious(&self) -> bool {
        self.phi() > self.phi_threshold
    }
}
```

**Ventajas sobre timeout fijo:**
- Se adapta a la latencia real de la red (WiFi lento vs LAN rapido)
- Menos falsos positivos en redes con jitter alto
- Cada nodo puede tener diferentes patrones de latencia

#### 36.5.2 Respuesta ante Fallo de Nodo (**IMPLEMENTADO**)

> **IMPLEMENTADO**: `enforce_min_copies()` en el event loop re-replica datos bajo el minimo.
> Nodos muertos se detectan via HeartbeatManager y se emite `PeerFailed` event.
> Anti-entropy sync via Merkle trees reconcilia datos al reconectarse.

```
Nodo B cae:
  1. PhiAccrualDetector(B) supera threshold en nodos A y C → PeerFailed event
  2. enforce_min_copies() detecta que datos de B tienen menos de min_copies
  3. Re-replicar datos:
     - Se eligen targets del consistent hash ring (preferencia: nodos no-probation)
     - Se envian mensajes Replicate a los nuevos targets
  4. Si B vuelve: anti-entropy sync via Merkle trees
     - Intercambio de SyncRequest/SyncResponse/SyncData
     - Solo se transfieren los datos que realmente difieren
```

#### 36.5.3 Anti-Entropy Sync (**IMPLEMENTADO**)

> **IMPLEMENTADO** en `distributed_network.rs` (`run_anti_entropy_sync()`) y `merkle_sync.rs`.
> Se ejecuta periodicamente en el event loop de cada nodo.
> Usa `AntiEntropySync::needs_sync()` para determinar que peers necesitan sync.

```rust
// Implementado en merkle_sync.rs + distributed_network.rs:
// 1. Construir MerkleTree desde storage local (storage_to_btree helper)
// 2. Enviar SyncRequest { merkle_root } al peer
// 3. Peer compara con su propio Merkle tree
// 4. Si raices difieren: responde SyncResponse { diff_keys }
// 5. Se intercambian SyncData { entries } con los datos que faltan
// 6. record_sync() actualiza el timestamp de ultimo sync

// Resultado: solo se transfieren datos que realmente difieren
```

### 36.6 Seguridad de Nodos

#### 36.6.1 Autenticacion (**IMPLEMENTADO**)

> **IMPLEMENTADO** en `node_security.rs` (`CertificateManager`, `JoinToken`, `ChallengeResponse`).
> - CA generation + node cert generation via rcgen 0.13
> - Mutual TLS via quinn ServerConfig/ClientConfig
> - Join tokens con expiracion, max_uses, secure RNG (SHA-256 mixed entropy)
> - Challenge-response con HMAC-SHA256 + constant-time comparison
> - Identity save/load a disco (cert.der, key.der, ca_cert.der)

```rust
// Implementado en node_security.rs:
pub struct NodeIdentity {
    pub node_id: NodeId,
    pub cert_der: Vec<u8>,       // DER-encoded certificate
    pub key_der: Vec<u8>,        // DER-encoded private key
    pub ca_cert_der: Vec<u8>,    // CA certificate for verification
}

pub struct CertificateManager; // Static methods for cert management
pub struct JoinToken {          // Time-limited, use-limited join tokens
    pub token: String,
    pub expires_at: u64,
    pub max_uses: Option<usize>,
    pub uses: usize,
}
pub struct ChallengeResponse;   // HMAC-SHA256 challenge-response protocol
```

#### 36.6.2 Protocolo de Union (**IMPLEMENTADO**)

> **IMPLEMENTADO** en `distributed_network.rs` (`validate_join_token()`) y `node_security.rs` (`JoinToken`).
> - Si `config.join_token` es `None`: cluster abierto, cualquier nodo puede unirse.
> - Si `config.join_token` tiene valor: tras mutual TLS, se espera `JoinRequest` con token valido.
> - Token se valida via `JoinToken::consume()` (respeta expiracion y max_uses).
> - Se responde `JoinAccepted` (con lista de peers) o `JoinRejected` (con razon).
> - Token generado con `secure_random_bytes(32)` (SHA-256 mixed entropy).
> - Comparaciones de token usan `constant_time_eq()` para prevenir timing attacks.

```
Nodo nuevo quiere unirse:
  1. Establece conexion QUIC con mutual TLS al nodo bootstrap
  2. Si el cluster requiere token:
     a. Nodo nuevo envia JoinRequest { token, cert_der }
     b. Bootstrap verifica token via JoinToken::consume()
     c. Si valido: JoinAccepted { node_id, peers }
     d. Si invalido/expirado: JoinRejected { reason } + cierre de conexion
  3. Si no requiere token: conexion aceptada directamente tras TLS
  4. Nodo entra en periodo de prueba (reputation = 0.5, probation = true)
  5. Durante probacion:
     - Se evita como target de replicacion (enforce_min_copies prefiere nodos no-probation)
     - Sus mensajes se contabilizan
  6. Tras ~100 mensajes exitosos: probation = false, reputation sube
```

#### 36.6.3 Reputacion (**IMPLEMENTADO** en `distributed_network.rs`)

> **IMPLEMENTADO**: El sistema de reputacion esta integrado directamente en `PeerState` dentro de `distributed_network.rs`.
> - Cada peer tiene `reputation: f32` (0.0 a 1.0, inicio: 0.5) y `probation: bool` (inicio: true)
> - Mensajes exitosos: +0.001 de reputacion. Errores: -0.01 de reputacion.
> - Despues de ~100 mensajes exitosos: sale de probacion.
> - Metodos publicos: `peer_reputation(&NodeId)`, `peer_in_probation(&NodeId)`.
> - `enforce_min_copies()` evita nodos en probacion como targets de replicacion cuando es posible.

```rust
// Implementado en distributed_network.rs:
pub struct PeerState {
    pub reputation: f32,     // 0.0 - 1.0
    pub probation: bool,     // New nodes start in probation
    pub messages_sent: u64,
    pub messages_received: u64,
    // ...
}

// Acciones que degradan reputacion:
// - Errores en procesamiento de mensajes (-0.01 por error)
// - Reputacion se clampea a rango [0.0, 1.0]

// Acciones con reputacion baja:
// - Nodos en probacion se evitan como targets de replicacion
// - enforce_min_copies() prefiere nodos con probation=false
```

#### 36.6.4 Rate Limiting por Nodo

El modulo `distributed_rate_limit.rs` ya existente se usa para limitar operaciones por nodo:
- Maximo de requests/segundo por nodo
- Maximo de datos/segundo transferidos
- Proteccion contra nodos que intentan saturar la red

### 36.7 Protocolo de Comunicacion

> **IMPLEMENTADO**: Se eligio QUIC via quinn (seccion 36.7.2) por simplicidad (~50 crates vs ~200+ de libp2p).
> Transporte: quinn 0.11 + rustls 0.23 + rcgen 0.13. Feature flag: `distributed-network`.
> LAN discovery implementado via UDP broadcast. Message framing: length-prefixed bincode.

#### 36.7.1 Opcion Recomendada: libp2p

```toml
# En Cargo.toml futuro (no implementado aun)
libp2p = { version = "0.54", features = [
    "tcp", "tls", "noise", "yamux",
    "mdns",        # Descubrimiento local (LAN)
    "kad",         # Kademlia DHT
    "gossipsub",   # Pub/sub para eventos
    "request-response", # RPC directo
] }
```

**Ventajas de libp2p:**
- Kademlia DHT nativo (reemplaza nuestro DHT local con uno real)
- NAT traversal (hole punching, relay, AutoNAT)
- mDNS para descubrimiento en LAN sin configuracion
- Protocolos multiplexados sobre una sola conexion
- Maduro, usado por IPFS, Substrate/Polkadot, Filecoin

#### 36.7.2 Alternativa: QUIC via quinn (**ELEGIDA**)

```toml
quinn = "0.11"
```

> **IMPLEMENTADO**: Se eligio esta opcion. LAN discovery implementado via UDP broadcast (`DiscoveryAnnounce`), peer exchange para descubrimiento mas alla de la LAN. Los nodos se auto-descubren en la misma red sin configuracion manual.

#### 36.7.3 Alternativa: gRPC via tonic

```toml
tonic = "0.12"
prost = "0.13"
```

Si se prefiere arquitectura cliente-servidor clasica en vez de P2P puro.
Mejor para despliegues controlados (cloud, kubernetes).

#### 36.7.4 Serializacion de Mensajes

```rust
enum NodeMessage {
    // Heartbeat
    Ping { sender: NodeId, timestamp: u64 },
    Pong { sender: NodeId, timestamp: u64 },

    // DHT Operations
    Get { key: String, request_id: u64 },
    GetResponse { key: String, value: Option<Vec<u8>>, request_id: u64 },
    Put { key: String, value: Vec<u8>, ttl: Option<Duration> },
    PutAck { key: String, request_id: u64 },

    // Replication
    Replicate { key: String, value: Vec<u8>, version: u64 },
    ReplicateAck { key: String, version: u64 },
    SyncRequest { merkle_root: Vec<u8> },
    SyncDelta { entries: Vec<(String, Vec<u8>, u64)> },

    // MapReduce
    MapTask { job_id: String, chunk: DataChunk },
    MapResult { job_id: String, outputs: Vec<MapOutput> },
    ReduceTask { job_id: String, key: String, values: Vec<Vec<u8>> },
    ReduceResult { job_id: String, output: ReduceOutput },

    // Cluster Management
    JoinRequest { token: String, cert: Vec<u8> },
    JoinAccepted { node_id: NodeId, peers: Vec<PeerInfo> },
    JoinRejected { reason: String },
    NodeLeft { node_id: NodeId },
}
```

Serializacion recomendada: **bincode** (ya lo tenemos como dependencia).
Para interoperabilidad con otros lenguajes: protobuf.

### 36.8 Integracion con VectorDb

| Backend | Replicacion | Estrategia |
|---------|-------------|-----------|
| InMemory (Tier 0) | Nuestro sistema | Serializar/deserializar vectores entre nodos |
| LanceDB (Tier 2) | Nuestro sistema | Replicar archivos Lance o export/import via trait |
| Qdrant (Tier 3) | **Qdrant nativo** | Usar clustering propio de Qdrant (sharding + raft) |

**Estrategia hibrida:**
- Para RAG/knowledge graph (SQLite FTS5): nuestro consistent hashing + replicacion
- Para vectores en Qdrant: delegar a su clustering nativo (mas eficiente, ya optimizado)
- Para vectores en InMemory/LanceDB: nuestro sistema los replica

```rust
trait DistributedVectorDb: VectorDb {
    /// Replicar un vector a N nodos adicionales
    fn replicate_to(&self, nodes: &[NodeId], id: &str) -> AiResult<usize>;
    /// Buscar en multiples nodos y fusionar resultados
    fn distributed_search(
        &self,
        query: &[f32],
        limit: usize,
        nodes: &[NodeId],
    ) -> AiResult<Vec<VectorSearchResult>>;
    /// Verificar consistencia entre replicas
    fn verify_replicas(&self, id: &str) -> AiResult<ReplicaStatus>;
}
```

### 36.9 Migracion desde el Sistema Actual

La migracion del sistema actual al rediseñado es **incremental y no-destructiva**:

1. **Fase 1 (HECHO)**: Paralelizar MapReduce con rayon — ya completado
2. **Fase 2 (HECHO)**: Networking QUIC via quinn — transporte con mutual TLS (`distributed_network.rs`)
3. **Fase 3 (HECHO)**: Replicacion con consistent hashing (`consistent_hash.rs`, `distributed_network.rs`)
4. **Fase 4 (HECHO)**: Heartbeat + phi accrual failure detection (`failure_detector.rs`)
5. **Fase 5 (HECHO)**: Seguridad (mutual TLS, join tokens, HMAC challenge-response) (`node_security.rs`)
6. **Fase 6 (HECHO)**: Anti-entropy sync via Merkle trees (`merkle_sync.rs`)

Los CRDTs existentes funcionan sin cambios — diseñados para eventual consistency.
El modulo `p2p.rs` sigue existiendo (stubs) y no se reemplaza — `distributed-network` es un sistema paralelo mas completo.
Se eligio quinn (QUIC) en vez de libp2p por menor footprint de dependencias (~50 vs ~200+ crates).

### 36.10 Tabla Competitiva (entrada #49)

> Ver seccion 26, entrada #49.

---

## Fuentes

- [Rig](https://github.com/0xPlaygrounds/rig) — Framework Rust para LLM apps
- [Claude Code](https://github.com/anthropics/claude-code) — Agente de codificacion de Anthropic
- [Claude Code Sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) — Arquitectura de sandboxing
- [Anthropic Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime)
- [Aider](https://github.com/Aider-AI/aider) — AI pair programming
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — Princeton software engineering agent
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — Version minima de SWE-agent
- [OpenHands](https://github.com/OpenHands/OpenHands) — Plataforma de dev agents (ex-OpenDevin)
- [Devon](https://github.com/entropy-research/Devon) — Pair programmer open-source
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) — Agente autonomo
- [CrewAI](https://github.com/crewAIInc/crewAI) — Multi-agent orchestration
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — Framework de OpenAI (ex-Swarm)
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — AI software company simulation
- [E2B](https://github.com/e2b-dev/E2B) — Sandboxed code execution
- [TaskWeaver](https://github.com/microsoft/TaskWeaver) — Microsoft code-first agent
- [browser-use](https://github.com/browser-use/browser-use) — AI browser automation
- [Stagehand](https://github.com/browserbase/stagehand) — AI browser framework
- [Composio](https://github.com/ComposioHQ/composio) — Tool integration platform
- [MCP Servers](https://github.com/modelcontextprotocol/servers) — Model Context Protocol
- [MCP Filesystem Server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
- [NVIDIA Sandboxing Guide](https://developer.nvidia.com/blog/practical-security-guidance-for-sandboxing-agentic-workflows-and-managing-execution-risk/)
- [Human-in-the-Loop Guide](https://fast.io/resources/ai-agent-human-in-the-loop/)
- [AutoAgents Rust](https://github.com/liquidos-ai/AutoAgents)
- [Claude Code vs Aider comparison](https://www.selecthub.com/vibe-coding-tools/claude-code-vs-aider-ai/)
- [SWE-bench Verified leaderboard](https://www.vals.ai/benchmarks/swebench)
- [SWE-bench Pro leaderboard](https://scale.com/leaderboard/swe_bench_pro_public)
- [OpenHands Review](https://sider.ai/blog/ai-tools/opendevin-review-can-an-open-source-ai-software-engineer-deliver-in-2025)
- [AutoGPT Review 2025](https://sider.ai/blog/ai-tools/autogpt-review-is-autonomous-ai-ready-for-real-work-in-2025)
- [CrewAI Practical Lessons](https://ondrej-popelka.medium.com/crewai-practical-lessons-learned-b696baa67242)
- [CrewAI Manager-Worker Failures](https://towardsdatascience.com/why-crewais-manager-worker-architecture-fails-and-how-to-fix-it/)
- [Goose Review 2026](https://aitoolanalysis.com/goose-ai-review/)
- [Block Red Team vs Goose](https://www.theregister.com/2026/01/12/block_ai_agent_goose/)
- [Browser-use Review](https://sider.ai/blog/ai-tools/ai-browser-use-review-are-web-browsing-ai-agents-finally-useful-in-2025)
- [E2B Blog](https://e2b.dev/blog/groqs-compound-ai-models-are-powered-by-e2b)
- [E2B vs alternatives comparison](https://www.softwareseni.com/e2b-daytona-modal-and-sprites-dev-choosing-the-right-ai-agent-sandbox-platform/)
- [LanceDB GitHub](https://github.com/lancedb/lancedb) — Embedded vector DB en Rust (Apache 2.0)
- [LanceDB Rust SDK](https://docs.rs/lancedb/latest/lancedb/) — Documentacion oficial del crate
- [Lance Format](https://github.com/lance-format/lance) — Formato columnar para ML
- [LanceDB Vector Indexes](https://docs.lancedb.com/indexing/vector-index) — Documentacion de indices
- [LanceDB 700M Vectors in Production](https://sprytnyk.dev/posts/running-lancedb-in-production/) — Case study
- [LanceDB + S3 1B+ Vectors](https://aws.amazon.com/blogs/architecture/) — AWS Architecture Blog
- [Qdrant GitHub](https://github.com/qdrant/qdrant) — Vector DB en Rust (Apache 2.0)
- [Qdrant Rust Client](https://github.com/qdrant/rust-client) — SDK oficial gRPC
- [Qdrant Official Benchmarks](https://qdrant.tech/benchmarks/) — Comparativas oficiales
- [Qdrant Memory Consumption](https://qdrant.tech/articles/memory-consumption/) — Guia de consumo de RAM
- [Qdrant Quantization Guide](https://qdrant.tech/documentation/guides/quantization/) — Binary/Scalar/Product quantization
- [LanceDB Benchmarking](https://medium.com/etoai/benchmarking-lancedb-92b01032874a) — Benchmarks independientes
- [LanceDB vs Qdrant Comparison](https://zilliz.com/comparison/qdrant-vs-lancedb) — Comparativa por Zilliz

---

## 37. Sistema Autonomo Completo — Implementacion

> Fecha: 2026-02-18

### Resumen

Se ha implementado un sistema completo de agente autonomo con 14 modulos nuevos, feature flags independientes, y total integracion con la infraestructura existente. El sistema permite ejecutar agentes autonomos con loop LLM -> parse -> validate -> execute -> feed results, sandbox configurable, scheduler cron, triggers basados en eventos, perfiles pre-hechos, modos de operacion, browser automation, y ejecucion distribuida.

### Feature Flags

| Feature | Modulos | Dependencias | En `full` |
|---------|---------|-------------|-----------|
| `autonomous` | agent_policy, agent_sandbox, os_tools, user_interaction, task_board, interactive_commands, mode_manager, agent_profiles, autonomous_loop | ninguna | No |
| `scheduler` | scheduler, trigger_system | autonomous | No |
| `butler` | butler | autonomous | No |
| `browser` | browser_tools | autonomous | No |
| `distributed-agents` | distributed_agents | autonomous + distributed-network | No |

Ningun feature flag esta en `full` — todos son opt-in.

### Arquitectura de Modulos

```
agent_policy.rs          — Modelo de permisos (AutonomyLevel, InternetMode, RiskLevel, AgentPolicy)
agent_sandbox.rs         — Validacion de acciones contra policy + audit trail
os_tools.rs              — Tools de filesystem/shell/git/network registradas en ToolRegistry
user_interaction.rs      — Comunicacion agente <-> usuario durante ejecucion
task_board.rs            — Task board reactivo basado en TaskPlan existente
interactive_commands.rs  — Parser de comandos del usuario (bilingue ES/EN)
mode_manager.rs          — Escalado/de-escalado de modos de operacion
agent_profiles.rs        — Perfiles pre-hechos para agentes, conversaciones, workflows
autonomous_loop.rs       — Core loop: LLM genera -> parsear tool calls -> sandbox validate -> ejecutar -> feed back
butler.rs                — Auto-deteccion de entorno (LLMs, proyecto, GPU, Docker, etc.)
scheduler.rs             — Cron scheduler (parseo 5-field cron expressions)
trigger_system.rs        — Triggers basados en eventos (cron, file change, feed, webhook, AI event)
browser_tools.rs         — CDP browser automation (navigate, click, type, screenshot, etc.)
distributed_agents.rs    — Agentes distribuidos con task queue, node management, MapReduce
```

### Flujo del Loop Autonomo

```
1. AutonomousAgent.run(task)
2. Construir system prompt con lista de tools disponibles
3. Generar respuesta via response_generator callback
4. Parsear tool calls del response (JSON format)
5. Si no hay tool calls -> done (respuesta final)
6. Si tool call es "ask_user" -> estado = WaitingForUser -> InteractionManager.ask() -> inyectar respuesta -> continuar
7. Para cada tool call:
   a. SandboxValidator.validate(action) — checar policy
   b. Si requiere aprobacion -> InteractionManager.ask_approval()
   c. ToolRegistry.execute(tool_call) -> resultado
   d. Push resultado como AgentMessage::Tool a la conversacion
8. Actualizar task_board con progreso (si conectado)
9. Si iteraciones >= max_iterations -> stop
10. Volver al paso 2
```

### Modelo de Permisos (AgentPolicy)

```rust
AgentPolicy {
    autonomy: AutonomyLevel,          // Paranoid | Normal | Autonomous
    internet: InternetMode,           // Disabled | SearchOnly | FullAccess | AllowList
    allowed_paths: Vec<PathBuf>,      // carpetas accesibles
    denied_paths: Vec<PathBuf>,       // carpetas denegadas
    allowed_commands: Vec<String>,    // whitelist de comandos shell
    denied_commands: Vec<String>,     // blacklist de comandos
    mcp_servers: Vec<String>,         // MCPs permitidos
    max_iterations: usize,            // limite de iteraciones del loop
    max_cost_usd: f64,               // limite de coste
    max_runtime_secs: u64,           // timeout global
    require_approval_above: RiskLevel,// pedir aprobacion si risk >= X
    tool_permissions: HashMap<String, bool>,
    working_directory: Option<PathBuf>,
}
```

Presets: `AgentPolicy::default()` (Normal), `AgentPolicy::paranoid()`, `AgentPolicy::autonomous()`.

### Modos de Operacion

```
Chat         — Conversacion simple, sin tools
Assistant    — Tools basicos (busqueda, calculo)
Programming  — Filesystem + shell + git
AssemblyLine — Multi-agente coordinado
Autonomous   — Loop autonomo completo
```

`ModeManager` gestiona escalado/de-escalado con `allowed_max` como techo configurable.

### MultiAgentSession

La clase `MultiAgentSession` integra:
- `AgentOrchestrator` para gestion de tareas/agentes
- `AutonomousAgent`s para ejecucion real
- `TaskBoard` compartido para progreso visible
- `InteractionManager` compartido para comunicacion agente-usuario
- `CommandProcessor` para interpretar comandos del usuario durante ejecucion

### Perfiles Pre-hechos

**Agentes**: coding-assistant, research-agent, devops-agent, data-analyst, content-writer, code-reviewer, sysadmin, paranoid

**Conversaciones**: casual, technical, brainstorm, interview

**Workflows**: code-review-pipeline, research-report, bug-fix

### Butler (Auto-deteccion)

9 detectores built-in:
1. OllamaDetector (localhost:11434)
2. LmStudioDetector (localhost:1234)
3. CloudApiDetector (env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
4. ProjectTypeDetector (Cargo.toml, package.json, pyproject.toml, etc.)
5. GitDetector (.git, branch, remotes)
6. DockerDetector (docker command, Dockerfile)
7. GpuDetector (nvidia-smi)
8. BrowserDetector (chrome/chromium)
9. NetworkDetector (connectivity)

### Integracion con AiAssistant

`AiAssistant` expone metodos de conveniencia:
- `operation_mode()`, `set_operation_mode()`, `escalate_mode()`, `de_escalate_mode()`
- `profiles()`, `profiles_mut()` — acceso al ProfileRegistry
- `create_agent(profile, generator)` — crea agente desde perfil con sandbox + OS tools
- `create_agent_headless(profile, generator)` — agente headless para tests
- `butler_scan()`, `auto_configure()` — deteccion y configuracion automatica
- `scheduler()`, `scheduler_mut()`, `trigger_manager()`, `trigger_manager_mut()`
- `init_browser()`, `browser_session()`, `browser_session_mut()` — acceso directo a BrowserSession (feature `browser`)
- `init_distributed_agents(node_id)`, `distributed_agents()`, `distributed_agents_mut()` — acceso directo a DistributedAgentManager (feature `distributed-agents`)

### Tests

| Modulo | Tests |
|--------|-------|
| agent_policy | 16 |
| agent_sandbox | 10 |
| os_tools | 11 |
| user_interaction | 12 |
| task_board | 15 |
| interactive_commands | 11 |
| mode_manager | 12 |
| agent_profiles | 12 |
| autonomous_loop | 15 |
| butler | 12 |
| scheduler | 12 |
| trigger_system | 14 |
| browser_tools | 10 |
| distributed_agents | 12 |
| multi_agent (session) | 13 |
| **Total** | **~187** |

Total de tests del crate con todos los features (al momento de Phase 5): **1719**

> **Nota (Phase 9)**: El total actual es **2356 tests** (2318 lib + 38 integration). Ver Seccion 38 para el desglose actualizado.

---

## 38. Infraestructura de Testing

> Seccion anadida: 2026-02-19

### Capas de Testing

El proyecto utiliza cuatro capas de testing complementarias:

| Capa | Descripcion | Ubicacion |
|------|-------------|-----------|
| **Unit tests** (`#[test]`) | Tests en cada modulo, aislados | `src/*.rs` en bloques `#[cfg(test)]` |
| **Integration tests** | Tests que prueban modulos combinados | `tests/integration_tests.rs` |
| **Test harness** (CLI) | Binary independiente con menu interactivo | `src/bin/ai_test_harness.rs` |
| **Feature-gated tests** | Tests condicionales por feature flag | `#[cfg(feature = "p2p")]` etc. |

### Estrategia de Testing P2P

El modulo P2P interactua con red (STUN, UPnP, NAT-PMP, TCP sockets). Los tests cubren este codigo sin infraestructura de red real:

1. **Paths deshabilitados**: `enable_upnp: false` → verifica error handling
2. **Configs vacias**: `stun_servers: vec![]` → manejo graceful de configuracion ausente
3. **Direcciones inalcanzables**: RFC 5737 TEST-NET (`198.51.100.1:3478`) → paths de timeout
4. **Datos sinteticos**: Construccion manual de `PeerReputation` → logica pura sin I/O
5. **Aserciones de estado**: `manager.start()` + `manager.stats()` → lifecycle sin peers reales

### Test Harness: Compilacion Condicional

El harness soporta categorias condicionales mediante `#[cfg(feature)]`:

```rust
// Categorias base siempre disponibles
let mut categories = vec![
    ("core", tests_core as fn() -> CategoryResult),
    // ... 108 categorias base ...
];

// Categorias P2P solo con feature "p2p"
#[cfg(feature = "p2p")]
{
    categories.push(("p2p_nat", tests_p2p_nat as fn() -> CategoryResult));
    categories.push(("p2p_reputation", tests_p2p_reputation as fn() -> CategoryResult));
    categories.push(("p2p_manager", tests_p2p_manager as fn() -> CategoryResult));
}
```

Esto permite compilar el harness sin features opcionales pesados, manteniendo las categorias base funcionales.

### Patrones de Test para Codigo Network-Dependent

Para tests que requieren timing (como Phi Accrual Failure Detector):
- Usar thresholds generosos (`phi_threshold: 16.0`)
- Sleep intervals largos (`20ms+`) para evitar flakiness en CI
- Verificar consistencia relativa (`is_suspicious() == (phi > threshold * 0.5)`) en vez de valores absolutos

### Resumen de Tests

| Modulo / Grupo | Tests |
|----------------|-------|
| Core (features `full`) | 2360 |
| Distributed networking | 115 |
| P2P | 32 |
| Autonomous agent system | 255 |
| Integration tests | 38 |
| Test harness (base) | ~436 |
| Test harness (P2P) | 16 |
| Benchmarks (Criterion) | 6 |
| **Total lib + integration** | **2,398** |

Nuevas areas de tests anadidas en Phase 9:
- `repl` — REPL engine, command parsing, history, session management, completions (21 tests)
- `reranker` — Cross-encoder, reciprocal rank fusion, diversity (MMR), cascade, pipeline (15 tests)
- `ab_testing` — Experiment manager, variant assignment (FNV-1a), significance (Welch's t / chi-square), early stopping (20 tests)

Nuevas areas de tests anadidas en Phases 7-8:
- `providers` — configuracion de proveedores, model fetching, generacion
- `security` — validacion de input, deteccion de inyeccion, sanitizacion
- `persistence` — persistencia de sesiones, I/O de archivos
- `metrics` — coleccion de metricas, contadores, histogramas
- `error` — tipos de error, propagacion, Display/Debug
- `analysis` — analisis de texto, extraccion de temas
- `embeddings` — LocalEmbedder, similaridad coseno, embedding batch
- `rate_limiting` — rate limit, sliding window, token bucket
- `intent` — IntentClassifier, categorias, confianza
- `templates` — PromptTemplate, renderizado, categorias
- `response_effectiveness` — metricas de calidad de respuesta
- `conversation_compaction` — estrategias de compactacion, scoring de importancia

### Suite de Benchmarks

```bash
cargo bench --bench core_benchmarks --features full
```

6 benchmarks: intent_classification, conversation_compaction, prompt_shortener, sentiment_analysis, request_signing_hmac_sha256, template_rendering.

### Determinismo y Ejemplos

- **BestFit determinism fix**: La estrategia de asignacion `BestFit` en integration tests fue corregida para producir resultados deterministas independientemente de la plataforma o el orden de iteracion de HashMap.
- **Required-features en Cargo.toml**: Todos los 17 ejemplos declaran `required-features` en sus secciones `[[example]]`, de modo que `cargo build --examples` omite correctamente los que no tienen features habilitados.

### Cobertura

Cobertura de funciones publicas:
- P2P: **100%** (32 tests, 0 funciones sin cubrir)
- Failure Detector: **100%** (15 tests, 0 funciones sin cubrir)
- Todas las demas funciones publicas cubiertas via unit tests + harness

---

## 39. Phase 9 — REPL/CLI, Neural Reranking, A/B Testing

> Fecha: 2026-02-20

### Resumen

Phase 9 anade tres modulos nuevos: un motor REPL/CLI interactivo, un pipeline de neural reranking para mejorar resultados RAG, y un framework completo de A/B testing para experimentos con prompts y configuraciones.

### Modulos Nuevos

#### REPL/CLI (`repl.rs` + `bin/ai_assistant_cli.rs`)

Motor de REPL interactivo con CLI binary para uso headless del asistente.

- `ReplEngine`: motor principal con event loop, command parsing, history
- Comandos built-in: `/help`, `/clear`, `/session`, `/model`, `/config`, `/export`, `/quit`
- Tab completion para comandos y argumentos
- Historial de comandos persistible
- Temas de color configurables
- Gestion de sesiones (crear, listar, cambiar)
- Binary `ai_assistant_cli` para ejecucion directa
- Feature: `full` (no requiere feature adicional)
- 21 tests unitarios

#### Neural Reranking (`reranker.rs`)

Pipeline de reranking para mejorar la precision de resultados RAG y busquedas vectoriales.

- `CrossEncoderReranker`: reranking basado en scores de relevancia query-document
- `ReciprocalRankFusion` (RRF): fusion de rankings multiples con factor k configurable
- `DiversityReranker`: seleccion por Maximum Marginal Relevance (MMR) con lambda de diversidad
- `CascadeReranker`: pipeline de rerankers en cascada con top-k progresivo
- `RerankerPipeline`: combinacion configurable de multiples estrategias de reranking
- Feature: `full` (no requiere feature adicional)
- 15 tests unitarios

#### A/B Testing (`ab_testing.rs`)

Framework completo de experimentacion para prompts, modelos y configuraciones.

- `ExperimentManager`: gestion del ciclo de vida de experimentos (draft, running, paused, completed)
- `VariantAssigner`: asignacion determinista de variantes por user ID usando FNV-1a hashing
- `SignificanceCalculator`: tests estadisticos — Welch's t-test para metricas continuas, chi-square para proporciones
- `EarlyStopping`: finalizacion temprana cuando se alcanza significancia estadistica
- Metricas por variante: count, mean, variance, min, max, conversion rate
- Feature: `eval`
- 20 tests unitarios

### Nuevos Ejemplos

| Ejemplo | Feature | Descripcion |
|---------|---------|-------------|
| `repl_demo` | (none) | Demostracion del motor REPL interactivo |
| `ab_testing_demo` | `eval` | Demostracion del framework de A/B testing |

### Integracion con AiAssistant

Nuevos metodos en `AiAssistant`:
- `init_experiment_manager()` — inicializa el ExperimentManager
- `experiment_manager()` — acceso inmutable al ExperimentManager
- `experiment_manager_mut()` — acceso mutable al ExperimentManager

### Tests

| Modulo | Tests |
|--------|-------|
| repl | 21 |
| reranker | 15 |
| ab_testing | 20 |
| **Total nuevos Phase 9** | **56** |

**Total acumulado**: 2,356 tests (2318 lib + 38 integration), 0 failures, 0 warnings, 6 benchmarks.

---

## 40. Phase 10 — Token Counting, Cost Tracking, Multi-Modal RAG

> Fecha: 2026-02-20

### Resumen

Phase 10 anade tres modulos nuevos: un tokenizador BPE puro Rust para conteo preciso de tokens, un sistema de tracking de costes en tiempo real con dashboard y middleware, y un pipeline de RAG multi-modal para indexacion y busqueda sobre documentos con multiples modalidades (texto, imagenes, tablas, codigo, audio).

### Modulos Nuevos

#### BPE Token Counter (`token_counter.rs`)

Tokenizador BPE (Byte Pair Encoding) implementado en Rust puro sin dependencias externas, compatible con tokenizadores GPT.

- `TokenCounter` trait: interfaz comun para diferentes estrategias de conteo
- `BpeTokenCounter`: tokenizador BPE completo con vocabulario base, merge rules y encoding/decoding
- `ApproximateCounter`: estimacion rapida basada en heuristica de palabras (~75% precision)
- `ProviderTokenCounter`: conteo ajustado por ratio especifico del proveedor (OpenAI, Anthropic, etc.)
- `TokenBudget`: gestion de presupuesto de tokens por request
- `TokenAllocation`: distribucion de tokens entre system prompt, user message y response
- Feature: `full` (no requiere feature adicional)
- 15 tests unitarios

#### Cost Integration (`cost_integration.rs`)

Dashboard de costes en tiempo real con middleware para intercepcion de requests.

- `CostDashboard`: dashboard completo con historial de requests, totales por provider/modelo, alertas
- `CostMiddleware` trait: interfaz para intercepcion de requests con decision allow/warn/block
- `DefaultCostMiddleware`: implementacion por defecto con limites diarios y mensuales
- `CostAwareConfig`: configuracion de limites de coste por provider y globales
- `RequestCostEntry`: registro individual de coste por request (provider, modelo, tokens, coste USD)
- `RequestType`: tipos de request (Chat, Completion, Embedding, ImageGeneration, etc.)
- `CostDecision`: enum Allow/Warn/Block para decisiones de middleware
- Feature: `full` (no requiere feature adicional)
- 12 tests unitarios

#### Multi-Modal RAG (`multimodal_rag.rs`)

Pipeline de RAG que maneja multiples modalidades de contenido, no solo texto plano.

- `ModalityType`: enum con variantes Text, Image, Audio, Table, Code
- `MultiModalChunk`: chunk individual con modalidad, contenido, caption opcional, embedding
- `MultiModalDocument`: documento con chunks de multiples modalidades
- `MultiModalRetriever`: indexador y buscador que opera sobre todas las modalidades
- `MultiModalPipeline`: pipeline completo retrieve + synthesize
- `MultiModalResult`: resultado de query con chunks relevantes y modalidades usadas
- `ImageCaptionExtractor`: extractor de captions para chunks de imagen
- Feature: `full` (no requiere feature adicional)
- 15 tests unitarios

### Nuevos Ejemplos

| Ejemplo | Feature | Descripcion |
|---------|---------|-------------|
| `cost_tracking` | `full` | Demostracion del dashboard de costes, middleware y alertas de presupuesto |
| `multimodal` | `full` | Demostracion de indexacion y busqueda multi-modal |

### REPL: Nuevo Comando `/cost`

El REPL ahora incluye el comando `/cost` que muestra el dashboard de costes en tiempo real, incluyendo totales por provider, ultimos requests y alertas de presupuesto.

### Integracion con AiAssistant

Nuevos metodos en `AiAssistant`:
- `init_cost_tracking()` — inicializa el CostDashboard y middleware de costes
- `cost_dashboard()` — acceso inmutable al CostDashboard
- `cost_dashboard_mut()` — acceso mutable al CostDashboard
- `cost_report()` — genera un informe de costes actual

### Tests

| Modulo | Tests |
|--------|-------|
| token_counter | 15 |
| cost_integration | 12 |
| multimodal_rag | 15 |
| **Total nuevos Phase 10** | **42** |

**Total acumulado**: 2,398 tests (2360 lib + 38 integration), 0 failures, 0 warnings, 6 benchmarks, 17 examples.

---

## 41. Phase 11 — UI Framework Hooks & Agent Graph Visualization

> Fecha: 2026-02-21

### Resumen

Phase 11 cierra los 2 ultimos items del roadmap (39/39 completados). Se anaden:
1. **UI Framework Hooks** (`ui_hooks.rs`) — infraestructura de eventos tipados para frameworks frontend
2. **Agent Graph Visualization** (`agent_graph.rs`) — exportacion programatica de grafos y trazas de ejecucion

### Modulos Nuevos

#### UI Framework Hooks (`ui_hooks.rs`)
- `ChatStreamEvent` — 8 variantes: MessageStart/Delta/End, ToolCallStart/Delta/End, Error, StatusChange
- `ChatHooks` — gestion de suscriptores con `on_event()`, `emit()`, broadcast
- `StreamAdapter` — conversion de chunks crudos a eventos tipados
- `ChatSession` — estado de sesion ligero con export JSON para frontend
- `ChatStatus` — Idle/Thinking/Streaming/ToolCalling/Error
- `UsageInfo` — tracking de tokens + costo
- Integrado en AiAssistant: `init_chat_hooks()`, `chat_hooks()`, `emit_chat_event()`

#### Agent Graph Visualization (`agent_graph.rs`)
- `AgentGraph` — grafo con nodos y aristas, CRUD completo
- `AgentNode` / `AgentEdge` — nodos con capacidades, aristas tipadas (5 tipos)
- `topological_sort()` — algoritmo de Kahn con deteccion de ciclos
- Export: `export_dot()`, `export_mermaid()`, `export_json()`
- `from_dag()` — construccion desde DagDefinition existente
- `ExecutionTrace` / `TraceStep` — registro paso a paso de ejecucion
- `GraphAnalytics` — critical path, bottlenecks, agent utilization

### Ejemplos Nuevos
| Ejemplo | Descripcion |
|---------|-------------|
| `ui_hooks_demo` | Suscripcion a eventos, StreamAdapter, ChatSession |
| `agent_graph_demo` | Grafo manual, export DOT/Mermaid/JSON, trazas, analytics |

### Tests
| Modulo | Tests |
|--------|-------|
| ui_hooks | 16 |
| agent_graph | 17 |
| **Total nuevos Phase 11** | **33** |

**Total acumulado**: 2,431 tests (2393 lib + 38 integration), 0 failures, 0 warnings, 6 benchmarks, 19 examples.

**ROADMAP COMPLETADO: 39/39 items implementados.**
