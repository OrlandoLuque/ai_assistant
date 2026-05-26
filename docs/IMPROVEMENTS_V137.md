# V137 — Catálogo extendido: schema + parsers + fixtures (0.2.84)

**Status**: DRAFT — plan agreed 2026-05-26, implementation not started
**Scope**: ampliar `src/models_dev.rs` para soportar el universo real de
modelos LLM (cloud + open-weights + variantes), sin tocar HTTP todavía.
**No-goals**: HTTP fetcher (eso es V138), wiring a consumers (V140).

## Por qué

Hoy `ModelRegistry` modela bien el catálogo *cloud* de models.dev
(precio, contexto, capability flags), pero no captura el ecosistema
real de modelos *open-weights*:

- Múltiples cuantizaciones por familia (Q2_K, Q3_K, Q4_K_M, Q5_K_S,
  Q6_K, Q8_0, fp16, BF16…) — cada una con tradeoffs de calidad/VRAM
  distintos.
- Variantes especiales: *abliterated* (capas de rechazo eliminadas vía
  técnicas de orthogonalización), *uncensored* fine-tunes, mezclas
  (MoE), *distilled* versions.
- Sweet spots — la cuantización/tamaño que la comunidad considera
  "punto óptimo de calidad por VRAM" para esa familia.
- LoRA adapters — parches ligeros (50-200 MB) que especializan el
  modelo base para una tarea. Un base + N adapters = N especialistas
  por el coste de VRAM de uno solo.
- Requirements per variante — VRAM mínima, RAM mínima, arquitecturas
  GPU soportadas, drivers, forks específicos de llama.cpp.

Esto es prerequisito de V140 (Butler recommender) y V138 (HTTP fetcher,
que tiene que parsear hacia este schema).

## Schema propuesto

```rust
pub struct ModelFamily {
    pub id: String,                    // "llama-3.1-8b"
    pub display_name: String,
    pub creator: String,               // "Meta", "Mistral AI", "DeepSeek"
    pub description: String,
    pub modality: Modality,            // Text, VisionText, Audio, …
    pub context_window: Option<u32>,
    pub training_cutoff: Option<NaiveDate>,
    pub family_tags: Vec<FamilyTag>,   // [Reasoning, Coding, Multilingual, …]
    pub variants: Vec<ModelVariant>,
    pub lora_adapters: Vec<LoraAdapter>,
}

pub struct ModelVariant {
    pub id: String,                    // "llama-3.1-8b-Q4_K_M"
    pub variant_kind: VariantKind,
    pub quantization: Option<Quantization>,
    pub modifier: Option<VariantModifier>, // Abliterated, Uncensored, Distilled
    pub size_bytes: u64,
    pub requirements: HardwareRequirements,
    pub source: ModelSource,           // HuggingFace { repo, file }, Ollama { tag }, …
    pub sweet_spot_for: Vec<SweetSpot>, // e.g. Quality, VramEfficiency, Speed
    pub provenance: Provenance,        // Official, CommunityFork { author }, …
    pub license: String,
}

pub struct LoraAdapter {
    pub id: String,
    pub base_family: String,
    pub purpose: AdapterPurpose,       // Coding, Writing, MedicalQA, RoleplayUncensored
    pub size_bytes: u64,
    pub source: ModelSource,
    pub license: String,
}

pub struct HardwareRequirements {
    pub min_vram_bytes: Option<u64>,   // None = CPU-only viable
    pub min_ram_bytes: u64,
    pub gpu_archs: Vec<GpuArch>,       // CudaCompute(sm_75), Rocm, Metal, Vulkan
    pub backends: Vec<Backend>,        // LlamaCppMainline, LlamaCppPrismML, Vllm, Ollama, …
}

pub enum Quantization {
    Fp32, Fp16, Bf16,
    Q8_0,
    Q6_K,
    Q5_K_S, Q5_K_M, Q5_0, Q5_1,
    Q4_K_S, Q4_K_M, Q4_0, Q4_1,
    Q3_K_S, Q3_K_M, Q3_K_L,
    Q2_K,
    Q1_0,                              // PrismML fork only
    Iq4_NL, Iq3_S, Iq2_XS,             // imatrix variants
    Other(String),                     // forward-compat for new schemes
}

pub enum SweetSpot {
    Quality,            // best output for this family
    VramEfficiency,     // best quality/VRAM tradeoff
    Speed,              // fastest tok/s on consumer GPU
    Lowest,             // smallest viable for the family
}
```

## Por qué este schema y no otro

- **`#[non_exhaustive]` en todas las enums + structs**: tu V39 establece
  esto como invariante. Permite añadir nuevas variantes sin romper
  callers en el futuro (nuevas cuantizaciones aparecen cada poco).
- **`Quantization::Other(String)` como escape hatch**: la comunidad
  GGUF inventa esquemas nuevos (IQ4_NL, IQ3_S salieron hace un año).
  Mejor un fallback abierto que un panic.
- **`HardwareRequirements` desglosado por backend**: el mismo modelo
  con Q4_K_M corre con 6 GB VRAM en llama.cpp mainline pero necesita
  el fork de PrismML para Q1_0. No es uniforme.
- **`LoraAdapter` separado de `ModelVariant`**: un adapter no es una
  variante del modelo (mismo .gguf base), es un parche que se aplica
  encima. La separación deja claro qué cuesta cargar.
- **`SweetSpot` como vector, no scalar**: una variante puede ser
  "sweet spot de VRAM" Y "sweet spot de velocidad" simultáneamente.

## Fuentes que el schema debe representar

| Fuente | Cobertura | Acceso |
|---|---|---|
| models.dev `api.json` | Cloud (OpenAI, Anthropic, Google, Mistral…) | Público, JSON estático |
| HuggingFace Hub | Open-weights, variantes, LoRAs | API REST (auth opcional, rate-limited) |
| Ollama library | Curated GGUF + tags | API REST pública |
| `curated_models.rs` (in-crate) | Fallback offline opinated | Static slice |

V137 sólo define el schema y un loader de fixtures. Las fuentes se
implementan en V138 (HTTP fetcher).

## Migración del schema actual

`ModelRegistry` ya existe (712 líneas). Estrategia:

- Mantener `ModelInfo` legacy como sub-vista — `family.into_legacy() ->
  Vec<ModelInfo>` para no romper callers actuales (cuya superficie real
  son los tests del propio módulo: nadie más consume `models_dev` hoy).
- `ModelRegistry::families: Vec<ModelFamily>` añadido junto al
  `models: Vec<ModelInfo>` actual.
- `lookup()` busca primero en `families`, fallback en `models`.

## Verificación (cuando se cierre la fase)

```bash
cargo test --lib -- models_dev::
cargo clippy --lib -- -D warnings
cargo deny check
```

Más: una fixture JSON con ≥1 familia completa (Llama-3.1-8B con 4
cuantizaciones + 1 LoRA + sweet spot tagging) parseada round-trip.

## Riesgos identificados

- **Combinatoria de variantes**: una familia popular puede tener >20
  variantes en HF. El schema permite muchas, pero hay que aplicar
  filtros en V138 para no descargar metadata de variantes que nadie
  usará. *Mitigation*: limitar por defecto a top-N populares + las que
  ya tienes locales; expansión on-demand.
- **Drift de esquema en fuentes**: HF cambia campos JSON sin avisar.
  *Mitigation*: `#[serde(default)]` + ignorar campos desconocidos
  (patrón ya usado en `models_dev.rs`).
- **License diversity**: open-weights van desde Apache-2.0 hasta
  licencias restrictivas (Llama Community, Gemma, …). El schema lleva
  el campo pero no hace enforcement; eso pertenece a V143 audit.

## Iteraciones del plan

- **iter 1**: schema básico (sin LoRAs, sin sweet spots).
- **iter 2**: añadido sweet spots tras feedback usuario.
- **iter 3 (actual)**: LoRA adapters, hardware requirements por
  backend, Quantization::Other como escape hatch, provenance.

## Out of scope para V137

- Descarga real desde HF / models.dev (V138).
- Detección de hardware del host (V139).
- Decisiones de recomendación (V140).
- Persistencia en SQLite vs JSON (decisión diferida; el módulo expone
  serde — el storage es del caller).
