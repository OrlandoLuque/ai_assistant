# AI Assistant — Documentacion de modulos y funcionalidad pendiente

**Ultima actualizacion**: 2026-02-02

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
  - `rag`: habilita `auto_indexing` y `SqliteVersionStore` (dependencia `rusqlite`)
  - `egui-widgets`: widgets de chat para egui
- Patron comun: `Config` struct con `Default`, structs de resultado con `Serialize/Deserialize`
- Duraciones almacenadas como `u64` millis para compatibilidad de serializacion
- Cross-references entre modulos via `super::module::Type`
- Tests unitarios en cada modulo (`#[cfg(test)] mod tests`)
- Total de tests en los 12 nuevos modulos: **190+**
- Compilacion verificada con todas las combinaciones de features

## Dependencias

```toml
[features]
documents = ["zip", "pdf-extract"]
rag = ["rusqlite", "aes-gcm"]
egui-widgets = ["egui"]

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
zip = { version = "2", optional = true }            # documents
pdf-extract = { version = "0.7", optional = true }  # documents
egui = { version = "0.27", optional = true }         # egui-widgets
rusqlite = { version = "0.31", optional = true }     # rag
aes-gcm = { version = "0.10", optional = true }     # rag (encrypted knowledge)
```
