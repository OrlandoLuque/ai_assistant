# V39 — API Stability Hardening

**Tesis**: Endurecimiento sistemático de la API pública para minimizar breaking changes
en futuras versiones de la librería. Cualquier adición de variantes a enums, campos a
structs, o métodos a traits ya no romperá el código del cliente.

**Estado**: HECHO
**Fecha**: 2026-03-15
**LOC nuevas**: ~1,000 (anotaciones) + ~200 (constructores/Default) + ~200 (docs) ≈ 1,400 total

---

## Resumen de cambios

### Fase 1: `#[non_exhaustive]` en todos los enums públicos

Añadido `#[non_exhaustive]` a **454 enums públicos** en 213+ archivos fuente.

**Efecto**: Los crates externos que hagan `match` sobre estos enums deben incluir un
brazo wildcard `_ => {}`. Esto permite añadir variantes en futuras versiones sin romper
clientes.

**Dentro del propio crate**: Sin efecto. Los tests y código interno no necesitan cambios.

**Binarios** (`src/bin/`): Son crates separados. Se añadieron ~14 brazos wildcard en
`ai_assistant_cli.rs` y `ai_test_harness.rs` para los `match` sobre `ReplAction`,
`ReplCommand`, `SanitizationResult`, y `AccessResult`.

**Ejemplos** (`examples/`): Se añadió 1 brazo wildcard en `vision_demo.rs` para `ImageData`.

### Fase 2: `#[non_exhaustive]` en Config structs públicos

Añadido `#[non_exhaustive]` a **246 structs Config** con campos públicos.

**Efecto**: Los crates externos no pueden construir estos structs mediante struct literal
(`FooConfig { field: val }`). Deben usar `FooConfig::default()` + asignación de campos,
o constructores `new()`. Esto permite añadir campos en futuras versiones sin romper clientes.

**Migración aplicada en binarios, tests, ejemplos y benchmarks**:
- `src/bin/ai_test_harness.rs`: 37 conversiones de struct literal → `default()` + asignación
- `src/bin/ai_assistant_server.rs`: 2 conversiones (AuthConfig, TlsConfig)
- `tests/integration_tests.rs`: 16 correcciones
- `tests/rag_tier_tests.rs`: 11 conversiones
- `benches/core_benchmarks.rs`: 8 conversiones
- `examples/`: 7 conversiones en 6 archivos

**Structs Config más impactados** (por frecuencia de uso en tests/ejemplos):
- `ChunkingConfig`, `EmbeddingConfig`, `ContextWindowConfig`, `RateLimitConfig`
- `PiiConfig`, `InjectionConfig`, `SanitizationConfig`, `CoalescingConfig`
- `AuthConfig`, `TlsConfig`, `GraphRagConfig`, `ExpansionConfig`

### Fase 3: Constructores `Default` y `new()` para Config structs

**Default implementations añadidas** (8 structs):
| Struct | Archivo | Nota |
|--------|---------|------|
| `ContextComposerConfig` | context_composer.rs | total_budget=4096, response_reserve=1024 |
| `AlertConfig` | online_eval.rs | threshold=0.0, consecutive_failures=0 |
| `LayerConfig` | multi_layer_graph.rs | priority=0, SyncPolicy::Shared |
| `JudgeConfig` | prompt_signature/judge.rs | temperature=0.0, rubric vacío |
| `AutonomousAgentConfig` | autonomous_loop.rs | max_iterations=0, usa AgentConfig::default() |
| `RouterConfig` | rag_methods.rs | default_retriever vacío |

**Constructores `new()` añadidos** (5 structs con campos obligatorios):
| Struct | Archivo | Parámetros |
|--------|---------|-----------|
| `TlsConfig` | server.rs | `cert_path, key_path` |
| `PushNotificationConfig` | a2a_protocol.rs | `url` |
| `TurnConfig` | p2p.rs | `url` |
| `McpOAuthConfig` | mcp_protocol/oauth.rs | `client_id, client_secret` |
| `McpV2OAuthConfig` | mcp_protocol/v2_oauth.rs | `client_id, client_secret` |

### Fase 4: Documentación de estabilidad en traits públicos

Añadida sección `# Stability` a los 10 traits públicos más críticos:

| Trait | Archivo |
|-------|---------|
| `VectorDb` | vector_db.rs |
| `Guard` | guardrail_pipeline.rs |
| `ModelProvider` | model_integration.rs |
| `EmbeddingProvider` | embedding_providers.rs |
| `Plugin` | plugins.rs |
| `AgentCallback` | agent.rs |
| `CloudStorage` | cloud_connectors.rs |
| `SpeechProvider` | speech.rs |
| `SearchProvider` | web_search.rs |
| `TokenCounter` | token_counter.rs |

Contenido del doc comment:
> New methods may be added to this trait in minor versions with default
> implementations. Required methods will only change in major versions.

---

## Estadísticas

| Métrica | Valor |
|---------|-------|
| Enums protegidos | 454 |
| Structs protegidos | 246 |
| Total anotaciones `#[non_exhaustive]` | ~700 |
| Archivos fuente modificados | ~250 |
| Default impls añadidos | 8 |
| Constructores `new()` añadidos | 5 |
| Traits documentados | 10 |
| Archivos externos adaptados | ~12 (bins, tests, examples, benches) |
| Tests | 6,829 passing (0 failures) |

---

## Guía de migración para clientes

### Enums: añadir brazo wildcard

```rust
// ANTES (compila hoy, romperá mañana):
match action {
    ReplAction::SendMessage(msg) => { /* ... */ }
    ReplAction::ExecuteCommand(cmd) => { /* ... */ }
    ReplAction::Exit => { /* ... */ }
}

// DESPUÉS (futuro-compatible):
match action {
    ReplAction::SendMessage(msg) => { /* ... */ }
    ReplAction::ExecuteCommand(cmd) => { /* ... */ }
    ReplAction::Exit => { /* ... */ }
    _ => { /* handle future variants */ }
}
```

### Structs: usar Default + asignación o constructores

```rust
// ANTES (compila hoy, romperá mañana):
let config = EmbeddingConfig {
    dimensions: 384,
    ..Default::default()
};

// DESPUÉS (futuro-compatible):
let mut config = EmbeddingConfig::default();
config.dimensions = 384;

// O para structs con campos obligatorios:
let tls = TlsConfig::new("cert.pem", "key.pem");
```

### Traits: sin cambios requeridos

Los traits solo reciben documentación de estabilidad. No se cambian firmas.
Nuevos métodos siempre tendrán implementaciones por defecto.

---

## Próximos pasos

- [ ] Considerar añadir builder pattern a Config structs complejos (>5 campos)
- [ ] Publicar guía de migración si se publica la librería
- [ ] Revisar periódicamente nuevos `pub enum` / `pub struct` sin `#[non_exhaustive]`
