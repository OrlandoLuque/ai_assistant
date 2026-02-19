# AI Assistant — Documentacion de modulos y funcionalidad pendiente

**Ultima actualizacion**: 2026-02-19

---

## Modulos implementados

### Recoleccion y procesamiento de datos

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `document_parsing` | Parse EPUB/DOCX/ODT/HTML a texto plano con metadatos | `documents` | 21 |
| `table_extraction` | Detectar y parsear tablas (Markdown, ASCII, HTML, CSV/TSV) | - | 10 |
| `data_source_client` | Cliente HTTP generico con auth, rate-limit, paginacion, cache | - | 21 |
| `crawl_policy` | Parse robots.txt, sitemaps, rate limiting adaptativo por dominio | - | 11 |
| `feed_monitor` | Parse RSS 2.0/Atom, monitorizar feeds, detectar entradas nuevas | - | 10 |
| `html_extraction` | Extraccion estructurada de HTML (tablas, listas, links, metadata, OG, Schema.org) | - | 25 |
| `content_versioning` | Snapshots de contenido, diffs LCS entre versiones, historial | `rag` (SQLite store) | 16 |

### Analisis y enriquecimiento

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `translation_analysis` | Analisis de calidad de traduccion, alineamiento, glosarios, prompts LLM | - | 7 |
| `entity_enrichment` | Enriquecimiento de entidades, dedup fuzzy (Jaccard n-grams), auto-tagging, merge | - | 16 |
| `auto_indexing` | Indexado RAG automatico de documentos importados, chunking adaptativo, incremental | `rag` | 13 |
| `knowledge_graph` | Knowledge Graph para Graph RAG con SQLite, extraccion de entidades (LLM/patron), relaciones, traversal multi-hop | `rag` | 27 |
| `encrypted_knowledge` | Paquetes KPKG encriptados con system_prompt, examples, rag_config, metadata | `rag` | 18 |

### Busqueda vectorial y distribucion

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `vector_db` | Trait VectorDb con InMemory y Qdrant backends, builder, hybrid search, migracion entre backends | `embeddings` | 8 |
| `vector_db_lance` | Backend LanceDB embebido (Tier 2): persistente, ANN search, export/import, Arrow RecordBatch | `vector-lancedb` | 14 |
| `distributed` | DHT Kademlia, CRDTs (GCounter, PNCounter, LWWRegister, ORSet, LWWMap), MapReduce paralelo (rayon) | `distributed` | 25 |
| `p2p` | P2P networking: NAT traversal (STUN/UPnP/NAT-PMP), ICE connectivity, bootstrap, knowledge broadcast/query, consensus, reputation | `p2p` | 19 |

### Resiliencia y red

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `http_client` | Trait `HttpClient` con `UreqClient` (produccion) y `MockHttpClient` (tests), metodos get_json/post_json/post_streaming | - | 4 |
| `async_providers` | Async HTTP client con `reqwest`+`tokio`, `AsyncHttpClient` trait, fetch/generate async, bridge blocking | `async-runtime` | 11 |
| `cloud_providers` | Soporte nativo OpenAI y Anthropic con API key resolution, endpoints /v1/chat y /v1/messages | - | 9 |
| `log_redaction` | Redaccion automatica de API keys, tokens, passwords, PEM keys en logs; macro `safe_log!` | - | 8 |
| `internal_storage` | Almacenamiento bincode+gzip con auto-deteccion de formato, herramientas de debug, macros | `binary-storage` | 8 |
| `request_queue` | Cola thread-safe con prioridad (Low/Normal/High), Condvar blocking, session removal | - | 13 |
| `server` | Servidor HTTP embebido con TcpListener, endpoints REST (/health, /models, /chat, /config), CORS | - | 9 |
| `events` | Event bus con EventHandler trait, 20+ variantes AiEvent, handlers filtered/logging/collecting | - | 12 |
| `unified_tools` | Sistema de herramientas unificado: builder, registry, validacion, multi-format parsing, built-ins | `tools` | 39 |

### Sistema de agentes autonomos

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `autonomous_loop` | Bucle de ejecucion autonomo con CostConfig, parser multi-formato (JSON/OpenAI/XML), tracking de tools | `autonomous` | 23 |
| `mode_manager` | 5 niveles de autonomia (Chat→Autonomous), escalado, historial | `autonomous` | 17 |
| `agent_sandbox` | Validacion de acciones (SandboxValidator), audit trail, politicas | `autonomous` | 8 |
| `user_interaction` | Human-in-the-loop: AutoApproveHandler, BufferedHandler con almacenamiento | `autonomous` | 21 |
| `interactive_commands` | Parser de comandos bilingue (EN/ES), intents, CommandProcessor, comando undo | `autonomous` | 16 |
| `task_board` | Tablero Kanban con prioridades, columnas, export Markdown, sistema undo | `autonomous` | 16 |
| `agent_profiles` | Perfiles pre-configurados (coding-assistant, research-agent, devops-agent, paranoid), workflows | `autonomous` | 15 |
| `agent_policy` | Politicas de seguridad: internet modes, risk levels, cost caps, command allowlists | `autonomous` | 12 |
| `scheduler` | Cron scheduler con parser de expresiones, jobs con lifecycle completo | `scheduler` | 16 |
| `trigger_system` | Triggers por evento (Manual/Cron/FileChange/FeedUpdate), cooldowns, max-fires | `scheduler` | 20 |
| `butler` | Auto-deteccion de entorno: Ollama/LM Studio (HTTP real), GPU (nvidia-smi), Docker, Browser, Network | `butler` | 18 |
| `os_tools` | Operaciones del SO con validacion sandbox: archivos, procesos, HTTP | `autonomous` | 11 |
| `browser_tools` | Chrome DevTools Protocol real via WebSocket, lanzamiento de Chrome, CDP JSON-RPC | `browser` | 19 |
| `distributed_agents` | Distribucion de tareas entre nodos, heartbeats, MapReduce distribuido | `distributed-agents` | 17 |

### Razonamiento adaptativo

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `adaptive_thinking` | Clasificacion de complejidad de queries, ajuste automatico de temperatura/tokens/RAG/CoT, parsing de `<think>` tags en streaming | - | 49 |

### Logica y planificacion

| Modulo | Funcionalidad | Feature | Tests |
|--------|---------------|---------|-------|
| `decision_tree` | Arboles de decision con condiciones, acciones, terminales, traversal con contexto | - | 17 |
| `task_planning` | Planes/listas con pasos, subpasos, estados, progreso, persistencia JSON, export Markdown | - | 23 |

---

## Detalle por modulo

### `document_parsing`

Tipos: `DocumentFormat`, `DocumentSection`, `DocumentMetadata`, `ParsedDocument`, `DocumentParserConfig`, `DocumentParser`.

- Formatos soportados: EPUB (ZIP+OPF+XHTML), DOCX (ZIP+word/document.xml), ODT (ZIP+content.xml), HTML (regex-based), PlainText
- EPUB/DOCX/ODT requieren feature `documents` (dependencia `zip` + `pdf-extract`)
- HTML siempre disponible (sin dependencias extra)
- Helpers publicos: `strip_xml_tags()`, `extract_xml_text()`, `extract_xml_metadata()`, `normalize_text()`
- Autodeteccion de formato por extension de archivo
- Limite de tamano configurable (`max_size_bytes`)

### `table_extraction`

Tipos: `TableCell`, `ExtractedTable`, `TableSourceFormat`, `TableExtractorConfig`, `TableExtractor`.

- Formatos: Markdown (`|---|---|`), ASCII art (`+---+---+`), HTML (`<table>`), CSV/TSV delimitado
- Export: `to_csv()`, `to_json()`, `to_markdown()`, `to_grid()`
- Acceso: `cell(row, col)`, `column(col)`, `column_by_name(header)`, `row_count()`
- Configuracion: flags de deteccion por formato, min_columns, min_rows
- Deteccion inteligente: ASCII tables no se confunden con Markdown

### `data_source_client`

Tipos: `AuthMethod`, `PaginationStrategy`, `RateLimitPolicy`, `DataSourceConfig`, `DataSourceResponse`, `PaginatedResponse`, `DataSourceClient`.

- Auth: None, Bearer, ApiKey (header/query), Basic
- Paginacion: Offset, PageNumber, Cursor
- Rate limiting: max requests por ventana + delay minimo entre requests
- Cache en memoria con TTL y eviccion
- Retry con exponential backoff en errores 5xx
- Duraciones como `u64` millis para serializacion

### `crawl_policy`

Tipos: `RobotsRule`, `RobotsDirectives`, `ParsedRobotsTxt`, `SitemapEntry`, `ChangeFrequency`, `ParsedSitemap`, `CrawlPolicyConfig`, `CrawlPolicy`.

- Parse completo de robots.txt (User-agent, Allow, Disallow, Crawl-delay, Sitemap)
- Matching de rutas con wildcards (`*`) y end-anchors (`$`)
- Parse de sitemaps XML (urlset y sitemapindex)
- Rate limiting adaptativo por dominio con respeto a Crawl-delay
- Cache de robots.txt con TTL configurable
- Descubrimiento automatico de sitemaps

### `feed_monitor`

Tipos: `FeedEntry`, `FeedMetadata`, `FeedFormat`, `ParsedFeed`, `FeedMonitorConfig`, `FeedMonitorState`, `FeedCheckResult`, `FeedParser`, `FeedMonitor`.

- Parse RSS 2.0: channel metadata, items (title, link, description, pubDate, guid, author, categories, content:encoded)
- Parse Atom: feed metadata, entries (title, link[href], summary, content, published, updated, id, author, categories)
- Autodeteccion de formato (RSS vs Atom)
- Monitorizacion: detecta entradas nuevas comparando IDs con estado previo
- Estado persistible (export/import JSON)
- Parse de fechas: RFC3339, RFC2822, ISO 8601 variantes

### `html_extraction`

Tipos: `HtmlSelector`, `HtmlElement`, `HtmlMetadata`, `ExtractionRule`, `HtmlExtractionConfig`, `HtmlList`, `HtmlLink`, `HtmlExtractionResult`, `HtmlExtractor`.

- Extraccion de metadata: title, meta tags, OpenGraph, Twitter Card, Schema.org (JSON-LD), feeds RSS, favicon, language
- Selectores CSS simplificados: tag, .class, #id, [attr=val]
- Extraccion de links con resolucion de URLs relativas y deteccion de externos
- Extraccion de listas (ol/ul con items)
- Extraccion de tablas via `table_extraction`
- Reglas de extraccion por dominio (content/title/author/date selectors)

### `content_versioning`

Tipos: `ContentSnapshot`, `ContentChange`, `ChangeType`, `VersionDiff`, `VersioningConfig`, `VersionHistory`, `ContentVersionStore`.

- Snapshots con hash, timestamp, metadata, labels
- Diff line-based usando algoritmo LCS (Longest Common Subsequence)
- Deduplicacion por hash (no almacena versiones identicas)
- Threshold de cambio configurable (minimo de diferencia para guardar)
- Export a unified diff format
- `SqliteVersionStore` (feature `rag`) para persistencia en SQLite
- Export/import JSON del historial completo

### `translation_analysis`

Tipos: `GlossaryEntry`, `Glossary`, `TranslationIssueType`, `TranslationIssue`, `AlignedSegment`, `TranslationStats`, `TranslationAnalysisResult`, `TranslationAnalysisConfig`, `ComparisonPrompt`, `ComparisonResponse`, `TranslationAnalyzer`.

- Alineamiento de parrafos con confianza basada en ratio de longitud
- Verificacion de consistencia de glosario (case-insensitive)
- Deteccion de numeros faltantes/extra entre origen y traduccion
- Analisis de completitud (ratio de palabras)
- Generacion de prompts para comparacion LLM
- Parsing de respuestas JSON de LLM (con fallback)
- Deteccion de idioma por bloques Unicode (CJK, Cyrillic, Arabic, Devanagari, Thai, Latin)

### `entity_enrichment`

Tipos: `EnrichableEntity`, `EnrichmentData`, `EnrichedEntity`, `MergeStrategy`, `DuplicateMatch`, `DuplicateReason`, `EnrichmentSource`, `EnrichmentConfig`, `EntityEnricher`.

- Dedup fuzzy por Jaccard index sobre character bigrams
- Matching: exacto, normalizado (lowercase + strip punctuation), fuzzy (threshold configurable)
- Merge strategies: KeepFirst, KeepLatest, KeepHighestConfidence, MergeAll, Manual
- Auto-tagging basado en tipo de entidad y atributos
- Cross-referencing via `data_source_client` (query a APIs externas)
- Registro de entidades conocidas para deteccion continua de duplicados

### `auto_indexing`

Tipos: `IndexChunkingStrategy`, `ChunkPosition`, `IndexedDocumentMeta`, `IndexableChunk`, `ChunkMetadata`, `IndexingResult`, `AutoIndexConfig`, `IndexState`, `IndexStats`, `AutoIndexer`.

- Chunking: Paragraph, Sentence, SlidingWindow, Adaptive (default)
- Adaptive chunking: merge parrafos cortos, split largos, respeta limites min/max
- Indexado incremental: solo re-indexa documentos con hash cambiado
- Almacenamiento SQLite (tablas `indexed_chunks` e `indexed_documents`)
- Integracion con `document_parsing` (feature `documents`)
- Export/import de estado
- Limite de tamano de archivo configurable

### `knowledge_graph`

Tipos: `Entity`, `EntityType`, `Relation`, `EntityMention`, `GraphChunk`, `GraphStats`, `ExtractionResult`, `ExtractedEntity`, `ExtractedRelation`, `IndexingResult`, `GraphQueryResult`, `KnowledgeGraphConfig`, `KnowledgeGraphStore`, `KnowledgeGraph`, `KnowledgeGraphBuilder`, `KnowledgeGraphCallback`, `EntityExtractor`, `LlmEntityExtractor`, `PatternEntityExtractor`.

- Almacenamiento SQLite con tablas: `entities`, `entity_aliases`, `relations`, `chunks`, `entity_mentions`
- Full-text search via FTS5 sobre chunks
- Thread-safe via `Mutex<Connection>` (Send + Sync)
- Entity types: Organization, Product, Person, Location, Concept, Event, Other
- Relation types: manufactures, located_in, part_of, variant_of, uses, related_to, custom
- Extraccion de entidades: PatternEntityExtractor (sin LLM), LlmEntityExtractor (con LLM)
- Alias resolution configurable
- Traversal con profundidad configurable y threshold de confianza
- Chunking automatico de documentos con overlap
- Deduplicacion de chunks por hash
- Builder pattern con entidades pre-configuradas de Star Citizen
- Integracion con RagPipeline via trait `GraphCallback`

### `encrypted_knowledge`

Tipos: `ExamplePair`, `RagPackageConfig`, `KpkgMetadata`, `KpkgManifest`, `ExtractedDocument`, `KpkgIndexResult`, `KpkgIndexResultExt`, `KpkgError`, `KeyProvider`, `AppKeyProvider`, `CustomKeyProvider`, `KpkgReader`, `KpkgBuilder`, `RagDbKpkgExt`.

- Paquetes de conocimiento encriptados (AES-256-GCM)
- Estructura ZIP interna con manifest.json y documentos .md/.txt
- System prompt y persona configurables por paquete
- Ejemplos few-shot (ExamplePair) con categorias opcionales
- Configuracion RAG por paquete (chunk_size, top_k, min_relevance, priority_boost)
- Metadata del paquete: autor, fecha, idioma, licencia, tags, custom fields
- Dos tipos de clave: AppKeyProvider (embebida) y CustomKeyProvider (passphrase)
- KpkgBuilder con pattern builder fluido para todos los campos
- KpkgReader con metodos read(), read_manifest_only(), read_with_manifest()
- KpkgIndexResultExt con helpers para acceder al manifest despues de indexar
- Trait RagDbKpkgExt para extension de RagDb con index_kpkg_ext()
- CLI tool (`kpkg_tool`) con comandos create, list, inspect, extract
- Compatibilidad hacia atras: manifests antiguos cargan sin error (serde(default))
- 18 tests unitarios

### `adaptive_thinking`

Tipos: `ThinkingDepth`, `RagTierPriority`, `ClassificationSignals`, `ThinkingStrategy`, `AdaptiveThinkingConfig`, `QueryClassifier`, `ThinkingTagParser`, `ThinkingParseResult`.

- 5 niveles de profundidad: Trivial, Simple, Moderate, Complex, Expert (con Ord)
- Clasificacion heuristica sin llamada LLM: reutiliza `IntentClassifier` + analisis estructural
- Señales: word_count, question_marks, has_comparison, has_analysis, has_code, is_multi_part, concept_count, expert_patterns
- Estrategia completa: temperatura, max_tokens, RAG tier hint, CoT prompt addition, profile suggestion
- Mapeo por defecto: Trivial(0.8/256), Simple(0.7/1024), Moderate(0.6/2048), Complex(0.4/4096), Expert(0.2/None)
- `ThinkingTagParser`: maquina de estados para streaming de `<think>...</think>` (maneja tags parciales entre chunks)
- Configurable: min/max depth, temperature_map, cot_instructions_override, rag_tier_priority
- `RagTierPriority`: Adaptive (default), Explicit, Highest — con warning en log cuando hay conflicto
- Deshabilitado por defecto para retrocompatibilidad
- Integracion en `AiAssistant`: `enable_adaptive_thinking()`, `classify_query()`, transparent parsing en `poll_response`
- 49 tests unitarios

### `decision_tree`

Tipos: `ConditionOperator`, `Condition`, `DecisionBranch`, `DecisionNodeType`, `DecisionNode`, `DecisionPath`, `DecisionTree`, `DecisionTreeBuilder`.

- 14 operadores de condicion: Equals, NotEquals, Contains, StartsWith, EndsWith, GreaterThan, LessThan, >=, <=, Matches (regex), IsEmpty, IsNotEmpty, InList
- Nodos: Condition, Action, Terminal, Question
- Traversal con contexto (`HashMap<String, Value>`) y deteccion de ciclos
- Evaluacion paso a paso (`evaluate_step`) o completa (`evaluate`)
- Validacion: detecta referencias colgantes y nodos inalcanzables
- Export a Mermaid flowchart
- Serializacion JSON completa (round-trip)
- Builder pattern

### `task_planning`

Tipos: `StepStatus`, `StepPriority`, `PlanStep`, `StepNote`, `TaskPlan`, `PlanSummary`, `PlanBuilder`.

- Status: Pending, InProgress, Done, Blocked, Skipped
- Prioridad: Critical, High, Medium, Low, Optional
- Pasos anidados recursivos (subpasos ilimitados)
- Progreso calculado recursivamente (% completado)
- Notas/comentarios por paso con timestamp y autor
- Etiquetas (tags) por paso
- Busqueda recursiva de pasos por ID
- `next_actionable()`: encuentra el siguiente paso ejecutable
- Export: JSON (round-trip) y Markdown
- Builder pattern para construccion fluida

### `vector_db`

Tipos: `VectorDbConfig`, `DistanceMetric`, `StoredVector`, `VectorSearchResult`, `MetadataFilter`, `FilterOperation`, `VectorDb` (trait), `InMemoryVectorDb`, `QdrantClient`, `VectorDbBuilder`, `VectorDbBackend`, `HybridVectorSearch`, `BackendInfo`, `VectorMigrationResult`.

- Trait `VectorDb` con 11 metodos: insert, insert_batch, search, get, delete, count, clear, health_check, backend_info, export_all, import_bulk
- `InMemoryVectorDb`: Tier 0, in-memory con eviccion LRU, 4 metricas de distancia (Cosine, Euclidean, DotProduct, Manhattan)
- `QdrantClient`: Tier 3, HTTP/REST via ureq, IDs numericos via hash, scroll API para export
- `VectorDbBuilder`: builder pattern con soporte para InMemory, Qdrant, y LanceDB (con feature flag)
- `BackendInfo`: tier, capacidades, max_recommended_vectors
- `migrate_vectors()`: migracion entre cualquier par de backends via export_all/import_bulk
- `string_id_to_u64()`: conversion determinista de IDs string a u64 (para Qdrant)
- `HybridVectorSearch`: combina vectorial + keyword con pesos configurables
- Feature: `embeddings`
- 8 tests unitarios

### `vector_db_lance`

Tipos: `LanceVectorDb`.

- Backend LanceDB embebido (Tier 2): almacenamiento persistente en disco en formato Lance columnar
- Implementa todo el trait `VectorDb` con 11 metodos
- Async-to-sync bridge via tokio::runtime::Runtime::block_on
- Datos en Arrow RecordBatch: id (Utf8), vector (FixedSizeList<Float32>), metadata (Utf8/JSON), timestamp (UInt64)
- Insert con semantica upsert (borra y reinserta si el ID ya existe)
- Vector search via ANN (Approximate Nearest Neighbor)
- Export/import completo para migracion entre backends
- SQL predicates para filtros de metadata
- SQL injection prevention en predicados
- Feature: `vector-lancedb` (dependencias: `lancedb`, `arrow-array`, `arrow-schema`, `tokio`, `futures`)
- 14 tests unitarios (incluyendo test de migracion InMemory -> LanceDB)

### `distributed`

Tipos: `NodeId`, `DhtNode`, `KBucket`, `RoutingTable`, `DhtValue`, `DhtConfig`, `Dht`, `DhtStats`, `GCounter`, `PNCounter`, `LWWRegister`, `ORSet`, `LWWMap`, `JobStatus`, `DataChunk`, `MapOutput`, `ReduceOutput`, `MapReduceConfig`, `MapReduceJob`, `MapReduceBuilder`, `DistributedCoordinator`.

- **DHT Kademlia**: NodeId 160-bit, XOR distance, K-buckets, routing table con 160 buckets
- **CRDTs**: GCounter (grow-only), PNCounter (increment/decrement), LWWRegister (last-writer-wins), ORSet (observed-remove), LWWMap (last-writer-wins map). Todos con `merge()` para eventual consistency
- **MapReduce paralelo**: Map y Reduce paralelizados con rayon (work-stealing thread pool), combiner opcional, split de input, progress tracking
- **DistributedCoordinator**: coordina DHT + CRDTs + MapReduce, merge_state entre coordinadores
- Feature: `distributed` (dependencia: `rayon`)
- 25 tests unitarios

### `http_client`

Tipos: `HttpClient` (trait), `UreqClient`, `MockHttpClient`.

- Trait `HttpClient`: `get_json()`, `post_json()`, `post_streaming()` con timeout configurable
- `UreqClient`: implementacion real usando `ureq`, con headers (Authorization, Content-Type)
- `MockHttpClient`: mock para tests con respuestas pre-configuradas, conteo de llamadas
- Usado internamente por `providers.rs` via funciones `_with_client()`
- 4 tests unitarios

### `log_redaction`

Tipos: `RedactionPattern`, `RedactionConfig`, `LogRedactor`.

- Patrones: API keys (sk-*, key-*), Bearer tokens, passwords en URLs, PEM keys, key=value secrets
- `RedactionConfig`: enable/disable, patrones custom, placeholder configurable
- `LogRedactor::redact()`: aplica todos los patrones configurados
- Macro `safe_log!`: reemplaza `eprintln!` con redaccion automatica
- Funcion `redact()` standalone para uso rapido
- 8 tests unitarios

### `internal_storage`

Tipos: `StorageFormat`, `InternalFileInfo`.

- `save_internal<T>()`: serializa con bincode+gzip (o JSON fallback sin feature)
- `load_internal<T>()`: auto-detecta formato (gzip magic bytes vs JSON)
- `serialize_internal<T>()`/`deserialize_internal<T>()`: variantes in-memory
- `dump_as_json<T>()`: convierte archivos binarios a JSON legible
- `convert_to_json<T>()`/`convert_json_to_binary<T>()`: conversion entre formatos
- `file_info()`: informacion de archivo (tamano, formato, timestamps)
- Macro `debug_dump!`: volcado de datos en builds de debug
- 8 tests unitarios

### `events`

Tipos: `AiEvent`, `EventHandler` (trait), `EventBus`, `FilteredHandler`, `TimestampedEvent`, `LoggingHandler`, `LogLevel`, `CollectingHandler`, `EventTimer`.

- `AiEvent`: enum con 20+ variantes (MessageSent, ResponseComplete, ProviderFailed, FallbackTriggered, SessionCreated, ContextWarning, ModelSelected, ToolExecuted, etc.)
- `EventHandler` trait: `on_event(&self, event: &AiEvent)`
- `EventBus`: registro de handlers, emision de eventos, historial opcional
- `FilteredHandler`: wrapper que filtra eventos por categoria
- `LoggingHandler`: handler built-in que logea eventos via `log` crate
- `CollectingHandler`: handler que acumula eventos en `Vec<AiEvent>` (util para tests)
- `EventTimer`: utilidad para medir duracion de ejecucion de tools
- Categorias: response, provider, session, context, model, rag, tool
- 12 tests unitarios

### `request_queue`

Tipos: `RequestPriority`, `QueuedRequest`, `QueueStats`, `RequestQueue`.

- `RequestPriority`: Low (background), Normal (user messages), High (system commands)
- `QueuedRequest`: builder pattern con id, session_id, priority, message, knowledge_context
- `RequestQueue`: cola thread-safe con `Mutex` + `Condvar`
- Operaciones: `enqueue()`, `try_dequeue()`, `dequeue_blocking()`, `dequeue_timeout()`
- `remove_session()`: elimina todas las peticiones de una sesion
- `close()`: señaliza cierre, desbloquea threads en espera
- `stats()`: estadisticas (pending, por prioridad, processed, dropped)
- 13 tests unitarios

### `unified_tools`

Tipos: `ParamType`, `ParamSchema`, `ToolDef`, `ToolBuilder`, `ToolCall`, `ToolOutput`, `ToolError`, `ToolChoice`, `ToolHandler`, `ToolRegistry`, `ProviderPlugin` (trait), `ProviderRegistry`.

- Fusion de 4 modulos existentes (tools, tool_use, tool_calling, function_calling) en una API unificada
- `ParamSchema`: tipos JSON Schema con constructors (`string()`, `number()`, `enum_type()`, `array()`), rango numerico, valores default
- `ToolDef`: definicion con export a OpenAI function format y JSON Schema
- `ToolBuilder`: patron builder fluido (`required_string()`, `optional_number()`, etc.)
- `ToolCall`: accessors tipados (`get_string()`, `get_number()`, `get_bool()`, `get_array()`)
- `ToolRegistry`: registro, validacion (enums, rangos), ejecucion, export de schemas
- Multi-format parsing: JSON array, OpenAI response, `[TOOL:name()]`, XML `<tool>`, function_call
- `ProviderPlugin` trait y `ProviderRegistry` para extensibilidad
- Built-in tools: get_current_time, calculate, string_length, validate_json
- `evaluate_math()`: parser de expresiones con +,-,*,/,%,** y parentesis
- Feature: `tools`
- 39 tests unitarios

### `async_providers`

Tipos: `AsyncHttpClient` (trait), `ReqwestClient`, `MockAsyncClient`.

- `AsyncHttpClient` trait: `get_json()`, `post_json()` async
- `ReqwestClient`: implementacion con `reqwest`, soporte para api_key (Bearer auth)
- Funciones async: `fetch_models_async()`, `fetch_ollama_models_async()`, `fetch_openai_models_async()`, `fetch_kobold_models_async()`
- Generacion async: `generate_response_async()`, `generate_response_streaming_async()`
- `block_on_async()`: bridge para llamar codigo async desde contexto sincrono
- `create_runtime()`: helper para crear runtime tokio
- `MockAsyncClient`: mock para tests con respuestas pre-configuradas
- Feature: `async-runtime` (tokio + reqwest)
- 11 tests unitarios

### `cloud_providers`

Tipos: funciones standalone (sin structs nuevos, extiende `AiProvider` con variantes `OpenAI` y `Anthropic`).

- `resolve_api_key()`: resolucion de API key desde config y env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- `generate_openai_cloud()`: /v1/chat/completions con Bearer token auth
- `generate_anthropic_cloud()`: /v1/messages con x-api-key header, anthropic-version, system como parametro top-level
- `fetch_openai_cloud_models()`: lista de modelos via API
- `fetch_anthropic_cloud_models()`: lista hardcoded de modelos Claude conocidos
- `generate_cloud_response()`: dispatcher unificado por provider
- `fetch_cloud_models()`: dispatcher unificado para fetch
- `AiProvider::is_cloud()`: metodo helper
- `AiConfig::get_api_key()`: resolucion con fallback a env var
- 9 tests unitarios

### `server`

Tipos: `ServerConfig`, `AiServer`, `ServerHandle`.

- `ServerConfig`: host, port, max_body_size, read_timeout_secs
- `AiServer`: servidor HTTP envolviendo `Arc<Mutex<AiAssistant>>`
- `ServerHandle`: handle para servidor en background con direccion real
- Endpoints REST:
  - `GET /health` — Health check (status, version, model, provider)
  - `GET /models` — Lista de modelos disponibles
  - `POST /chat` — Enviar mensaje (no-streaming, poll until complete)
  - `GET /config` — Configuracion actual (sin exponer api_key)
  - `POST /config` — Actualizar configuracion (model, temperature)
  - `OPTIONS` — CORS preflight
- Parsing HTTP manual con `std::net::TcpListener` — zero dependencias extra
- `run_blocking()`: servidor bloqueante
- `start_background()`: servidor en thread separado con auto-port
- 9 tests unitarios

### Distributed Networking (`distributed-network` feature)

Modulos de networking real para sistema distribuido:

- **`consistent_hash.rs`**: Consistent hash ring con vnodes para particionamiento de datos
  - `ConsistentHashRing`: BTreeMap-based ring, add/remove nodes, get primary + replicas
  - `KeyRange`: rango de hash afectado por cambios
  - 12 tests unitarios
- **`failure_detector.rs`**: Phi Accrual Failure Detector (como Cassandra)
  - `PhiAccrualDetector`: deteccion estadistica de fallos con CDF normal
  - `HeartbeatManager`: gestion de heartbeats por nodo
  - `NodeStatus`: Alive/Suspicious/Dead/Unknown
  - 13 tests unitarios
- **`merkle_sync.rs`**: Merkle tree SHA-256 para anti-entropy sync
  - `MerkleTree`: flat array binary tree, diff, proof generation/verification
  - `AntiEntropySync`: sync timing por peer
  - `SyncDelta`: keys to send/request
  - 14 tests unitarios
- **`node_security.rs`**: Seguridad de nodos con Ed25519
  - `CertificateManager`: generacion CA + node certs con rcgen, save/load, quinn configs
  - `JoinToken`: tokens temporales para admision al cluster
  - `ChallengeResponse`: autenticacion HMAC-SHA256 con comparacion constant-time
  - `secure_random_bytes()`: generador SHA-256 mixed entropy (timestamp+thread+counter+process)
  - `constant_time_eq()`: comparacion resistente a timing attacks
  - 27 tests unitarios
- **`distributed_network.rs`**: Core networking con QUIC
  - `NetworkNode`: API sincrona con event loop async en background (patron LanceVectorDb)
  - Transporte QUIC via quinn con mutual TLS
  - Replicacion configurable (sync/async, quorum reads/writes, min_copies enforcement)
  - LAN discovery via UDP broadcast (DiscoveryConfig: enable_broadcast, broadcast_port, broadcast_interval)
  - Peer exchange: descubrimiento de peers via PeerExchangeRequest/PeerExchangeResponse
  - Anti-entropy sync: comparacion periodica de Merkle trees con peers, intercambio de deltas
  - Join token validation: validacion de tokens en conexiones entrantes, rechazo de tokens invalidos
  - max_connections enforcement: limite de conexiones concurrentes
  - Reputation tracking: incremento gradual por mensajes exitosos, decremento por errores
  - Probation: peers nuevos en periodo de prueba (100 mensajes para salir)
  - process_message maneja TODOS los tipos de NodeMessage (sin wildcards)
  - 47 tests unitarios (incluyendo tests de conexion real entre 2 nodos)
- **`distributed.rs` (actualizado)**: `NodeMessage` enum con Serialize/Deserialize, `NodeId` con serde derives, `PeerExchangeRequest`/`PeerExchangeResponse` variants
- **`vector_db.rs` (actualizado)**: `DistributedVectorDb<V>` wrapper para busqueda distribuida

**Total tests nuevos**: 113 (1531 total con `--features full,distributed-network`)

---

## Funcionalidad aun NO implementada (para futura revision)

### 1. OCR (Reconocimiento optico de caracteres)
- **Estado**: No implementado
- **Descripcion**: Extraer texto de imagenes y PDFs escaneados
- **Dependencia potencial**: `tesseract` bindings o API externa
- **Relevancia**: Necesario para documentos antiguos escaneados

### 2. Audio/Voz
- **Estado**: No implementado
- **TTS**: Sintesis de voz para respuestas
- **STT**: Transcripcion de audio como input
- **Relevancia**: Baja para uso actual (asistente de texto)

### 3. Video understanding
- **Estado**: No implementado
- **Descripcion**: Analisis de video frame-by-frame
- **Relevancia**: Baja

### 4. Knowledge Graphs
- **Estado**: IMPLEMENTADO (ver `knowledge_graph` module)
- **Descripcion**: Representacion relacional de entidades como grafo en SQLite
- **Funcionalidad**:
  - `KnowledgeGraph`: API de alto nivel con indexado de documentos y queries
  - `KnowledgeGraphStore`: Almacenamiento SQLite con FTS5
  - `PatternEntityExtractor`: Extractor de entidades basado en patrones (sin LLM)
  - `LlmEntityExtractor`: Extractor de entidades usando LLM
  - `KnowledgeGraphCallback`: Integracion con RagPipeline via `GraphCallback` trait
  - `KnowledgeGraphBuilder`: Builder con entidades pre-configuradas de Star Citizen
- **Tests**: 8 tests unitarios + 19 tests en ai_test_harness
- **Feature**: `rag` (requiere `rusqlite`)

### 5. Aprendizaje continuo / Online Learning
- **Estado**: No implementado
- **Descripcion**: Integrar correcciones del usuario en tiempo real al modelo
- **Relevancia**: Media - requiere infraestructura significativa

### 6. Explicabilidad avanzada
- **Estado**: Solo `confidence_scoring` implementado
- **Pendiente**: Attention visualization, feature attribution, counterfactuals
- **Relevancia**: Baja para uso actual

### 7. Model distillation / Compression
- **Estado**: Solo deteccion de formato en `quantization.rs`
- **Pendiente**: Knowledge distillation, pruning real
- **Relevancia**: Fuera de alcance (se usan modelos pre-entrenados)

### 8. Reinforcement Learning
- **Estado**: No implementado
- **Pendiente**: RLHF pipeline, reward modeling, PPO/DPO
- **Relevancia**: Baja - complejo y requiere infraestructura dedicada

### 9. Orquestacion DAG de workflows
- **Estado**: Parcial (`prompt_chaining` + `agentic_loop` + `decision_tree`)
- **Pendiente**: Grafos dirigidos con paralelismo real, human-in-the-loop
- **Relevancia**: Media

### 10. Monitorizacion automatica de cambios web
- **Estado**: `content_versioning` + `feed_monitor` implementados
- **Pendiente**: Scheduler periodico automatico, alertas push, integracion activa con `crawl_policy`
- **Relevancia**: Media

### 11. Extraccion de imagenes de documentos
- **Estado**: No implementado
- **Descripcion**: Extraer imagenes embebidas de EPUB/DOCX/PDF (mapas, diagramas, tokens)
- **Dependencia**: Ya se tiene acceso al ZIP; falta guardar los binarios
- **Relevancia**: Media para documentos con contenido visual importante

### 12. APIs de bases de datos externas
- **Estado**: `data_source_client` implementado (infraestructura generica)
- **Pendiente**: Conectores especificos para APIs publicas (Open5e, Wikidata, etc.)
- **Relevancia**: Media - el cliente generico ya soporta auth/paginacion/cache

---

## Notas de arquitectura

- Todos los modulos usan nomenclatura **generica** (no especifica de dominio)
- Modulos con dependencias opcionales usan feature flags:
  - `documents`: habilita parsing de EPUB/DOCX/ODT/PDF (dependencias `zip`, `pdf-extract`)
  - `rag`: habilita `auto_indexing`, `SqliteVersionStore`, y sesiones encriptadas (dependencias `rusqlite`, `aes-gcm`)
  - `egui-widgets`: widgets de chat para egui
  - `binary-storage`: habilita serializacion bincode+gzip para datos internos (dependencia `bincode`)
  - `async-runtime`: habilita providers async con reqwest+tokio (dependencias `tokio`, `reqwest`)
  - `tools`: habilita sistema unificado de herramientas (`unified_tools`)
  - `distributed`: habilita DHT, CRDTs, MapReduce paralelo (dependencia `rayon`)
  - `distributed-network`: habilita networking real QUIC, consistent hashing, failure detection, Merkle sync, node security, replication, LAN discovery (dependencias `quinn`, `rustls`, `rcgen`, `sha2`, `tokio`, `bincode`). **No incluido en `full`** por peso de deps.
  - `vector-lancedb`: habilita backend LanceDB embebido (dependencias `lancedb`, `arrow-array`, `arrow-schema`, `tokio`, `futures`)
  - `p2p`: P2P networking con NAT traversal (STUN/UPnP/NAT-PMP), ICE, bootstrap TCP, knowledge broadcast/query, consensus voting, reputation. Requiere `distributed`.
  - `wasm`: soporte WebAssembly real con `web-sys`, `js-sys`, `wasm-bindgen`, `getrandom`. Console logging, tiempo, RNG criptografico. Solo significativo en `target_arch = "wasm32"`.
  - `autonomous`: agente autonomo core (14 modulos). No incluido en `full`.
  - `scheduler`: cron scheduler + trigger system. Requiere `autonomous`.
  - `butler`: auto-deteccion de entorno. Requiere `autonomous`.
  - `browser`: herramientas de navegador Chrome CDP. Requiere `autonomous`.
  - `distributed-agents`: ejecucion distribuida de agentes. Requiere `autonomous` + `distributed-network`.
- Patron comun: `Config` struct con `Default`, structs de resultado con `Serialize/Deserialize`
- Duraciones almacenadas como `u64` millis para compatibilidad de serializacion
- Cross-references entre modulos via `super::module::Type`
- Tests unitarios en cada modulo (`#[cfg(test)] mod tests`)
- Total de tests en los 13 nuevos modulos de contenido: **240+**
- Total de tests en los 14 modulos de agentes autonomos: **255**
- Compilacion verificada con todas las combinaciones de features
- **Zero `.unwrap()` en codigo de produccion**: 554 llamadas `.unwrap()` reemplazadas en 76 archivos con manejo apropiado de errores:
  - Lock poisoning: `.unwrap_or_else(|e| e.into_inner())` — recupera datos del Mutex/RwLock envenenado
  - NaN safety: `.partial_cmp().unwrap_or(Ordering::Equal)` — maneja NaN en comparaciones f64
  - Regex infallible: `.expect("valid regex")` — documenta la intencion en patrones conocidos
  - Duration fallback: `.unwrap_or_default()` — extraccion segura de timestamps
- **Sistema de undo en task_board**: Soporte completo para deshacer operaciones (AddTask, StartTask, PauseTask, ResumeTask, CancelTask, CompleteTask) con historial de comandos
- Renombrado `dirs_next_stub` → `platform_dirs` (eliminacion de nomenclatura de stub)

## Dependencias

```toml
[features]
documents = ["zip", "pdf-extract"]
rag = ["rusqlite", "aes-gcm"]
egui-widgets = ["egui"]
binary-storage = ["bincode"]
async-runtime = ["tokio", "reqwest"]
distributed = ["rayon"]
vector-lancedb = ["lancedb", "arrow-array", "arrow-schema", "tokio", "futures"]
full = ["core", "multi-agent", "security", "analytics", "vision", "embeddings",
        "advanced-streaming", "adapters", "tools", "documents", "eval", "rag",
        "distributed", "binary-storage", "async-runtime"]

[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde", "clock"] }
ureq = { version = "2", features = ["json"] }
uuid = { version = "1", features = ["v4"] }
flate2 = "1"
regex = "1"
urlencoding = "2"
log = "0.4"
bincode = { version = "1", optional = true }         # binary-storage
zip = { version = "2", optional = true }             # documents
pdf-extract = { version = "0.7", optional = true }   # documents
egui = { version = "0.27", optional = true }         # egui-widgets
rusqlite = { version = "0.31", optional = true }     # rag
aes-gcm = { version = "0.10", optional = true }      # rag
tokio = { version = "1", optional = true }           # async-runtime, vector-lancedb
reqwest = { version = "0.12", optional = true }      # async-runtime
rayon = { version = "1", optional = true }           # distributed (parallel MapReduce)
lancedb = { version = "0.26", optional = true }      # vector-lancedb
arrow-array = { version = "57", optional = true }    # vector-lancedb
arrow-schema = { version = "57", optional = true }   # vector-lancedb
futures = { version = "0.3", optional = true }       # vector-lancedb
```
