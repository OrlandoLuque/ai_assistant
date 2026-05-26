# V140 — Butler LLM-driven recommendation (0.2.87)

**Status**: DRAFT — pending V137-V139 close
**Scope**: Butler API que dado (tarea, restricciones) recomienda
modelo + cuantización + LoRA + parámetros, usando catálogo (V137) +
hardware (V139), con LLM como razonador opcional.
**No-goals**: ejecutar la recomendación (sigue siendo del caller),
auto-tuning runtime.

## Por qué

Hoy elegir modelo es manual. El catálogo extendido (V137) y la
detección de hardware (V139) son inputs ricos; el Butler debe poder
traducirlos a una decisión informada — sin que el usuario tenga que
saber qué es Q4_K_M ni cuánta VRAM gasta cada cosa.

## API propuesta

```rust
pub struct RecommendationRequest {
    pub task: TaskKind,                // Coding, Writing, Reasoning, Roleplay, …
    pub language: Option<String>,      // "es", "en", "zh", …
    pub privacy: PrivacyConstraint,    // LocalOnly, AllowCloud, …
    pub max_latency_ms: Option<u32>,
    pub min_quality_tier: QualityTier, // Best, Balanced, Cheap, Tiny
    pub allow_uncensored: bool,        // default false
    pub allow_abliterated: bool,       // default false
    pub user_hint: Option<String>,     // texto libre — interpretado por LLM
}

pub struct Recommendation {
    pub primary: ModelChoice,
    pub fallbacks: Vec<ModelChoice>,
    pub reasoning: String,             // explicación legible
    pub estimated_vram_bytes: u64,
    pub estimated_tokens_per_sec: Option<f32>,
}

pub struct ModelChoice {
    pub family_id: String,
    pub variant_id: String,
    pub lora_id: Option<String>,
    pub backend: Backend,
    pub params: SuggestedParams,
}

pub struct SuggestedParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub n_gpu_layers: i32,             // -1 = all on GPU
    pub ctx_size: u32,
    pub batch_size: u32,
}

impl Butler {
    pub async fn recommend(
        &self,
        req: RecommendationRequest,
    ) -> Result<Recommendation, ButlerError>;
}
```

## Pipeline de decisión

```
RecommendationRequest
        │
        ▼
1. Filtro duro:                       (rule-based, deterministic)
   - HardwareInfo.gpus → variantes que caben
   - privacy=LocalOnly → drop cloud entries
   - allow_uncensored=false → drop modifier=Uncensored
        │
        ▼
2. Scoring base:                      (rule-based)
   - sweet_spot tags vs task → +bonus
   - benchmarks per tarea (si los hay) → +score
   - latency budget vs estimated_tps → drop si no llega
        │
        ▼
3. Top-K candidatos
        │
        ▼
4. LLM advisor (opcional):            (feature `butler-llm-advisor`)
   - prompt estructurado con candidatos + request
   - LLM elige + razona + tunea params
   - fallback: si LLM no disponible, devuelve top-1 con razón rule-based
        │
        ▼
5. Recommendation con primary + fallbacks
```

## Cadena de fallback VRAM-aware

Si sweet spot ideal (`llama-3.1-70b-Q4_K_M`) no cabe:

1. Probar misma familia, cuantización menor (`Q3_K_M`, `Q2_K`)
   mientras siga siendo *acceptable quality* (umbral configurable).
2. Probar tamaño inmediatamente inferior misma familia (`8b` en lugar
   de `70b`).
3. Probar familia hermana de tamaño similar (`mistral-7b` si
   `llama-3.1-8b` no cabe).
4. Devolver mejor-esfuerzo + `reasoning` claro: "you don't have VRAM
   for Llama-3.1-70B (40 GB). Falling back to Llama-3.1-8B-Q4_K_M
   (6 GB), which trades reasoning depth for fit."

## LLM advisor — diseño

- Sólo se activa con feature `butler-llm-advisor`.
- Usa la trait `LlmEnhancer` ya existente (V68) → no añade
  dependencia nueva.
- Prompt template estructurado con: top-K candidatos, hardware,
  request. Pide JSON con `choice` + `reasoning` + `params`.
- Resultado validado contra schema (si LLM hallucinates `temperature:
  17`, se descarta y se cae al rule-based).
- Coste: 1 inferencia LLM por recomendación, cacheable por
  (task, constraints, hw_fingerprint).

## Wiring

- `butler.rs` (4862 líneas) gana método `recommend()`. Cambios detrás
  de feature `butler-llm-advisor` propia para no contaminar la
  superficie estable.
- MCP tool nuevo `butler_recommend_model` — invocable desde agentes.
- `ai_serve` endpoint `/butler/recommend` (POST JSON).
- `ai_setup_gui` tab "Recommend Model" que llama y muestra resultado.
- CLI `ai_setup recommend --task coding --max-vram 8G`.

## Tests

- Rule-based path con catálogo + hardware mock → recomendaciones
  determinísticas verificables.
- VRAM fallback chain: simular hardware con 4 GB / 8 GB / 16 GB /
  24 GB / 80 GB y verificar primary diferente.
- LLM advisor con `MockLlm` (V68) devolviendo JSON canned.
- Edge: catálogo vacío → ButlerError::NoCandidates con mensaje útil.
- Edge: privacidad imposible (LocalOnly + sin modelos locales) →
  ButlerError::PrivacyConstraintUnsatisfiable.

## Riesgos / vectores de ataque

- **Prompt injection en `user_hint`** → LLM advisor podría ser
  manipulado a recomendar modelos maliciosos o exfiltrar info. *Mit*:
  user_hint pasa por sanitizer, advisor sólo elige de la lista
  prefiltrada.
- **Catálogo trucado** → si fetcher (V138) sirve catálogo manipulado,
  recommender promociona modelo malicioso. *Mit*: V143 firma del
  catálogo.
- **Resource exhaustion** — LLM advisor llamado en bucle. *Mit*:
  cache + rate-limit por sesión.

## Iteraciones del plan

- **iter 1**: sólo rule-based (sin LLM). Rechazado: pierde la
  ventaja real del "asesor inteligente" que pediste.
- **iter 2**: LLM siempre. Rechazado: dependencia dura de un provider
  configurado.
- **iter 3 (actual)**: pipeline híbrido — rule-based filtra y
  ordena, LLM advisor refina opcionalmente.

## Out of scope para V140

- Ejecutar la recomendación (descargar GGUF, lanzar llama-server) —
  sigue siendo del caller.
- Auto-tuning runtime (ajustar `n_gpu_layers` en vivo si VRAM cambia).
- Benchmarking automático de candidatos.
