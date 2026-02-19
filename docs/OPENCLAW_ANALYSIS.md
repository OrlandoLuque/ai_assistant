# Análisis Completo de OpenClaw

> Documento generado el 2026-02-13 a partir del análisis del código fuente de OpenClaw (v2026.2.13).
> Propósito: entender su arquitectura para extraer ideas aplicables a `ai_assistant`.

---

## 1. Qué es OpenClaw

OpenClaw es un **gateway de IA multi-canal y local-first** escrito en TypeScript/Node.js.
Actúa como plano de control unificado que conecta múltiples plataformas de mensajería
(WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Google Chat, MS Teams, Matrix, WebChat...)
con modelos de lenguaje (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama local, OpenRouter,
y muchos más).

**En esencia**: un usuario envía un mensaje por cualquier canal → OpenClaw lo enruta a una sesión
→ ejecuta un agente IA con herramientas → devuelve la respuesta por el canal original.

---

## 2. Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                   Canales de Mensajería                      │
│  WhatsApp · Telegram · Slack · Discord · Signal · iMessage   │
│  Google Chat · MS Teams · Matrix · WebChat · LINE · Zalo     │
└──────────────────────────┬──────────────────────────────────┘
                           │ mensajes entrantes/salientes
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Gateway (WebSocket Control Plane)                │
│              ws://127.0.0.1:18789 (local)                    │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌───────────────┐  │
│  │ Routing  │  │ Sessions │  │ Config │  │ Broadcasting  │  │
│  │ Engine   │  │ Manager  │  │ System │  │ (WS events)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬───┘  └───────────────┘  │
│       │              │             │                          │
│       ▼              ▼             ▼                          │
│  ┌──────────────────────────────────────────┐               │
│  │        Pi Agent Runtime (RPC)             │               │
│  │  - Model selection + failover             │               │
│  │  - Context window management              │               │
│  │  - Tool orchestration + streaming         │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                        │
│       ┌─────────────┼──────────────┐                        │
│       ▼             ▼              ▼                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Browser │  │ Memory   │  │ Cron     │  + más tools       │
│  │ Tool    │  │ (Vector) │  │ (Jobs)   │                    │
│  └─────────┘  └──────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Proveedores de Modelos IA                      │
│  Anthropic · OpenAI · Google · Ollama · OpenRouter ·         │
│  Bedrock · Together · Venice · MiniMax · Moonshot · etc.     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Funcionalidades IA

### 3.1 Sistema Multi-Proveedor con Failover

**Archivos clave**: `src/agents/models-config.providers.ts`, `src/agents/pi-embedded-runner/run.ts`

OpenClaw no depende de un solo proveedor de IA. Soporta **20+ proveedores** y tiene un sistema
de failover automático:

```
Intenta perfil de auth primario
  → Si falla: marca cooldown, intenta siguiente perfil
  → Si todos en cooldown: comprueba config de fallback
  → Si hay fallbacks: lanza FailoverError (el cliente lo maneja)
  → Si no: lanza error original
```

Cada proveedor tiene su propio módulo con adaptación de formato (Anthropic Messages API,
OpenAI Chat Completions, Google Gemini, etc.). Los auth profiles permiten rotar entre
múltiples API keys del mismo proveedor.

**Idea clave**: No asumir que un proveedor siempre estará disponible. Tener fallbacks.

### 3.2 Gestión de Contexto Inteligente

**Archivos clave**: `src/agents/context-window-guard.ts`, `src/agents/pi-embedded-runner/run.ts`

- Ventana de contexto configurable por sesión y por modelo
- Umbrales de advertencia (`CONTEXT_WINDOW_WARN_BELOW_TOKENS`)
- Mínimo duro (`CONTEXT_WINDOW_HARD_MIN_TOKENS`)
- **Auto-compactación**: cuando el contexto se desborda, compacta automáticamente
- **Truncado de resultados de herramientas** como fallback si la compactación no basta
- Seguimiento de tokens por caché (prompt caching de Anthropic)

### 3.3 Sistema de Herramientas (Tools)

**Directorio**: `src/agents/tools/`

Las herramientas son el mecanismo principal por el que el agente interactúa con el mundo:

| Herramienta | Archivo | Función |
|-------------|---------|---------|
| **Browser** | `browser-tool.ts` | Navegación web, capturas, acciones DOM |
| **Canvas** | `canvas-tool.ts` | Interfaz visual en tiempo real (A2UI) |
| **Memory** | `memory-tool.ts` | Almacenamiento y búsqueda vectorial |
| **Sessions** | (varios) | Listar, enviar, crear sesiones |
| **Message** | `message-tool.ts` | Enviar mensajes a canales |
| **Web Fetch** | `web-fetch.ts` | Peticiones HTTP, scraping (Firecrawl) |
| **Web Search** | `web-search.ts` | Búsqueda web (Perplexity/OpenRouter) |
| **Cron** | `cron-tool.ts` | Programar tareas periódicas |
| **Nodes** | `nodes-tool.ts` | Integración con dispositivos (cámara, pantalla, ubicación) |
| **Image** | `image-tool.ts` | Análisis de imágenes (VLM) |

**Modelo de ejecución**:
- Las herramientas se definen como `AgentTool` (de `@mariozechner/pi-agent-core`)
- Aceptan parámetros estructurados con schema JSON
- Devuelven resultados con bloques de contenido
- Soportan streaming parcial de resultados
- Tienen **gating de permisos** configurable por canal/sesión

### 3.4 Prompt Engineering

**Archivos**: `src/agents/`, `src/auto-reply/templating.js`

- **System prompts contextuales**: varían según identidad del agente, capacidades, estado de sesión
- **Templating con variables**: sustitución dinámica en prompts
- **Niveles de pensamiento**: soporte para extended thinking de Claude (off/low/medium/high)
- **Protección contra injection**: sanitización de magic strings de Anthropic
- **Boot scripts**: archivos `BOOT.md` que se ejecutan al arrancar para inicialización

### 3.5 Streaming en Tiempo Real

**Archivos**: `src/gateway/server-chat.ts`, `src/agents/pi-embedded-runner/`

- **Block streaming**: coalescencia de múltiples tool calls antes de enviar
- **Tool streaming**: resultados parciales de herramientas en tiempo real
- **Broadcasting por WebSocket**: eventos a todos los clientes conectados
- **Tracking de secuencia**: por cliente, con detección de gaps

### 3.6 Memoria Vectorial (RAG)

**Directorio**: `src/memory/`, `extensions/memory-core`, `extensions/memory-lancedb`

- Base de datos SQLite local con extensión `sqlite-vec` para búsqueda vectorial
- Embeddings computados localmente o via API (OpenAI, Gemini, Voyage)
- Indexación por agente (aislamiento de workspaces)
- Búsqueda semántica con puntuaciones de relevancia

---

## 4. Funcionalidades NO-IA

### 4.1 CLI Completa

**Directorio**: `src/cli/`, `src/commands/`

| Comando | Función |
|---------|---------|
| `openclaw gateway` | Arranca el gateway WebSocket |
| `openclaw agent` | Ejecuta un mensaje de agente directamente |
| `openclaw message send` | Envía mensajes a canales |
| `openclaw onboard` | Wizard de configuración interactivo |
| `openclaw pairing` | Aprobar/rechazar emparejamientos |
| `openclaw config` | Ver/editar configuración |
| `openclaw channels` | Estado de canales, logout |
| `openclaw sessions` | Listar, borrar, resetear, previsualizar sesiones |
| `openclaw models` | Mostrar modelos disponibles |
| `openclaw cron` | Gestionar tareas programadas |
| `openclaw status` | Salud del sistema y canales |
| `openclaw doctor` | Diagnosticar problemas |
| `openclaw security audit` | Auditoría de seguridad |

### 4.2 Sistema de Configuración

**Directorio**: `src/config/`

- **Formato**: JSON5 con sustitución de variables de entorno
- **Validación**: Schemas Zod (`zod-schema.ts`) con errores descriptivos
- **Jerarquía de overrides**: Global → Agente → Sesión → Canal
- **Hot reload**: Detección de cambios en archivo con validación antes de aplicar
- **Backups**: Rotación automática (5 copias)
- **Includes**: Configuración modular con detección de referencias circulares
- **Tipos separados por módulo**: `types.agents.ts`, `types.channels.ts`, `types.models.ts`, etc.

### 4.3 Sistema de Canales (Adapter Pattern)

**Directorio**: `src/channels/`

Cada canal implementa múltiples **adaptadores** de capacidades:

```typescript
ChannelMessagingAdapter  // Enviar/recibir mensajes
ChannelAuthAdapter       // Login/logout
ChannelSecurityAdapter   // Comprobación de permisos
ChannelOutboundAdapter   // Routing de respuestas
ChannelStatusAdapter     // Health checks
ChannelSetupAdapter      // Configuración inicial
ChannelHeartbeatAdapter  // Keep-alive
```

**Resolución de configuración por canal** (con fallback):
```
Directo → Normalizado → Parent → Wildcard (*)
```

### 4.4 Routing de Mensajes

**Directorio**: `src/routing/`

- **Resolución de sesión**: canal + cuenta + peer → session key
- **Routing de grupos**: Direct → Account → Channel → Wildcard
- **Delivery context**: recuerda por dónde responder
- **Gating de comandos**: herramientas disponibles por canal/grupo
- **Mention gating**: solo responder a menciones/replies en grupos

### 4.5 Sistema de Permisos

**Archivos**: `src/gateway/server-methods.ts`, `src/security/`

**Scopes del gateway**:
- `operator.admin` - Acceso total
- `operator.read` - Solo lectura (estado, logs, sesiones)
- `operator.write` - Enviar mensajes, ejecutar agente
- `operator.approvals` - Aprobar/rechazar peticiones de ejecución
- `operator.pairing` - Aprobar emparejamiento de dispositivos

**Seguridad de DMs**:
- Modo `pairing` (por defecto): desconocidos reciben código de emparejamiento
- Modo `open`: permite todos los DMs (opt-in explícito)
- Allowlists por canal

### 4.6 Hooks y Eventos

**Directorio**: `src/hooks/`

- **Orientado a eventos**: triggers en session:start, session:end, command:*, etc.
- **Fuentes**: bundled, managed, workspace, plugin
- **Metadata**: bins requeridos, variables de entorno, restricciones de SO
- **Ejecución**: en paralelo o en serie según configuración

### 4.7 Cola de Comandos con Lanes

**Archivo**: `src/process/command-queue.ts`

- **Lane global**: procesamiento serial de todas las operaciones
- **Lanes por sesión**: límites de concurrencia por sesión
- **Ordenamiento**: previene condiciones de carrera en estado de sesión

### 4.8 Extensiones y Plugins

**Directorios**: `src/plugin-sdk/`, `src/plugins/`, `extensions/`

- **Tipos**: Channel plugins, Gateway HTTP handlers, Skills
- **API de plugin**: interfaces de adaptador, definición de herramientas, schemas, hooks, rutas HTTP
- **Gestión de skills**: instalación desde npm/git, bundled/managed/workspace

### 4.9 Apps Companion

- **macOS**: Menu bar, Voice Wake, Talk Mode, Canvas, WebChat
- **iOS/Android**: Talk Mode, Canvas, cámara, grabación de pantalla
- **Descubrimiento**: Bonjour/mDNS para pairing de dispositivos

---

## 5. Patrones de Diseño Clave

### 5.1 Sesión como Event Log

Las sesiones se almacenan como archivos `.jsonl` (append-only):
- **Inmutabilidad**: solo se añaden eventos, nunca se modifican
- **Compactación**: resumen cuando el tamaño excede límites
- **Historial completo**: transcripción con metadata
- **Recuperación de estado**: reconstruir desde el log al cargar

### 5.2 Adapter Pattern para Canales

Cada canal es un conjunto de adaptadores que implementan interfaces específicas.
Esto permite añadir nuevos canales sin tocar el core.

### 5.3 Configuration-Driven Behavior

Todo el comportamiento se puede modificar via configuración:
- Herramientas disponibles por canal/sesión
- Modelos y proveedores por agente
- Políticas de DM por canal
- Niveles de pensamiento y verbosidad

### 5.4 Tool Gating

Las herramientas tienen permisos configurables:
- Por canal (ej: browser solo en WebChat)
- Por usuario/grupo
- Por policy de ejecución (sandbox)
- Con aprobación humana opcional para operaciones peligrosas

### 5.5 Model Failover Chain

```
Perfil primario → cooldown → siguiente perfil → cooldown → fallback → error
```

Con tracking de cooldown por perfil de autenticación, permitiendo múltiples API keys.

### 5.6 Message Chunking Adaptativo

Las respuestas se dividen según los límites de cada plataforma:
- Discord: 2000 chars
- Telegram: 4096 chars
- WhatsApp: 4096 chars
- Etc.

Con conversión de markdown al formato nativo de cada plataforma.

### 5.7 Broadcasting con Backpressure

El sistema de WebSocket maneja clientes lentos sin bloquear a los rápidos,
con scoping de eventos según los permisos del operador.

---

## 6. Stack Tecnológico

| Componente | Tecnología |
|-----------|------------|
| **Lenguaje** | TypeScript (Node.js 22.12+) |
| **Build** | tsdown (esbuild-based) |
| **Testing** | Vitest con cobertura V8 |
| **Linting** | Oxlint + Oxfmt (Rust-based) |
| **CLI** | Commander.js |
| **WebSocket** | ws (nativo Node.js) |
| **HTTP** | Express (para hooks, Canvas, Control UI) |
| **DB Vectorial** | SQLite + sqlite-vec / LanceDB |
| **WhatsApp** | Baileys |
| **Telegram** | grammY |
| **Slack** | Bolt |
| **Discord** | discord.js |
| **Browser** | Playwright |
| **Validación** | Zod |
| **Paquetes** | pnpm (monorepo con workspaces) |

---

## 7. Flujo Completo: Mensaje de WhatsApp → Respuesta IA

```
1. Usuario envía mensaje por WhatsApp
2. Baileys (librería WhatsApp) recibe el mensaje
3. Canal extrae: remitente, texto, media
4. Routing: mapea a session key (canal + cuenta + peer)
5. Carga sesión desde ~/.openclaw/sessions/
6. Construye prompt: historial + adjuntos + system prompt
7. Llama a gateway.agent() (RPC)
8. Gateway ejecuta Pi Agent:
   a. Selecciona modelo (Anthropic/OpenAI/etc.)
   b. Resuelve auth profile (con failover)
   c. Verifica tamaño de ventana de contexto
   d. Envía al modelo con herramientas disponibles
   e. Ejecuta herramientas si el modelo las invoca
   f. Streams de resultados parciales
9. Formatea respuesta según capacidades de WhatsApp
10. Divide si >4096 chars (chunking)
11. Envía de vuelta por WhatsApp
12. Guarda estado de sesión + transcripción
```

---

## 8. Números del Proyecto

- **Archivos TypeScript**: ~800+ (src/ + extensions/)
- **Líneas de código**: estimado 100K+ (código fuente sin tests)
- **Tests**: extensivos (Vitest), incluyendo unit, e2e, y live tests
- **Canales soportados**: 15+
- **Proveedores IA**: 20+
- **Herramientas de agente**: 15+
- **Extensiones**: 10+ (memory, diagnostics, canales adicionales)
