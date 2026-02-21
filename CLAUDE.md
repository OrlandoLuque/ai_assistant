# Contexto del Proyecto: ai_assistant

## Sobre el Proyecto
- **Librería Rust** para integración con LLMs locales y cloud
- **203.427 líneas de código**, 214 archivos fuente (.rs), 2.010+ tests
- **Autor único**: Orlando José Luque Moraira (Lander) — orlando.luque@gmail.com
- **Estado**: NO publicado en ningún sitio (ni crates.io, ni GitHub público, ni distribuido a terceros)
- **Feature flags**: 20+ (core, multi-agent, security, analytics, rag, distributed, autonomous, etc.)

## Módulos Principales
- Multi-proveedor LLM: Ollama, LM Studio, OpenAI, Anthropic, Gemini, Bedrock, HuggingFace, y más
- RAG 5 niveles: Self-RAG, CRAG, Graph RAG, RAPTOR — 7 backends vector DB
- Multi-agente: 5 roles, orquestación, memoria compartida
- Agente autónomo: 5 niveles autonomía, scheduler cron, browser automation (CDP)
- Distribuido: CRDTs, DHT Kademlia, MapReduce, QUIC/TLS 1.3
- Seguridad: RBAC, PII detection, guardrails constitucionales, AES-256-GCM
- Streaming: SSE, WebSocket RFC 6455, compresión, resumible
- MCP protocol, WASM, egui widgets, HTTP server embebido

## Decisiones Estratégicas Tomadas (Febrero 2026)

### Licencia
- **Decisión**: Cambiar de MIT/Apache-2.0 a **PolyForm Shield 1.0.0** (recomendada) o PolyForm Noncommercial 1.0.0 (alternativa más restrictiva) antes de publicar
- **Razón**: Al no haber publicado nada, tiene libertad total para elegir licencia. No hay código "regalado" bajo MIT.
- **PolyForm Shield** (recomendada): Permite uso comercial general, pero prohíbe crear productos competidores. Más adopción, menos gestión para un dev solo.
- **PolyForm Noncommercial** (alternativa): Prohíbe TODO uso comercial sin licencia. Más control pero menos adopción y más negociaciones.
- **Sin fecha de caducidad** (descartada BSL porque se convierte en open source tras 4 años)
- **SPDX**: Ambas son identificadores SPDX reconocidos (PolyForm-Shield-1.0.0, PolyForm-Noncommercial-1.0.0)
- **Opción adicional**: Considerar publicar un módulo básico pequeño (solo providers) bajo MIT en crates.io como gancho de marketing

### Monetización
- **Modelo elegido**: PolyForm Noncommercial + negociación caso a caso con empresas
- **Motivo**: El autor es desarrollador solo, con tiempo y presupuesto limitados, con familia
- **Estrategia de precios**: Caso a caso, sin infraestructura SaaS compleja por ahora
- **Futuro**: Si crece, evolucionar a Open-Core con features premium

### Protección de Propiedad Intelectual
- **Registro PI España** (cultura.gob.es): 13,59€ — PENDIENTE
- **WIPO PROOF** (OMPI/ONU): ~20€ — PENDIENTE
- **Safe Creative**: 15-30€ — PENDIENTE
- **Depósito notarial**: 50-150€ — PENDIENTE (recomendado)
- **CLA**: Implementar antes de recibir contribuciones externas (CLA Assistant en GitHub)
- **Patentes**: DESCARTADAS (no merece la pena en Europa para software puro)

### Estrategia de Features para Monetización Futura
| Tier | Features | Precio |
|------|----------|--------|
| Gratuito | core, embeddings, streaming, tools, documents, rag | 0€ |
| Pro | multi-agent, security, analytics, eval, vision | Pago |
| Enterprise | distributed, p2p, autonomous, scheduler, browser | Personalizado |

## Reglas para Claude Code
1. **NUNCA modificar código sin permiso explícito** — el autor es muy protector con su trabajo
2. El código ha sido desarrollado iterativamente con Claude (prompts del autor + generación asistida)
3. Respetar la estructura modular existente basada en feature flags de Cargo
4. Zero `.unwrap()` en producción — usar proper error handling siempre
5. Zero warnings del compilador — compilación limpia en todas las combinaciones de features
6. Tests para todo — el proyecto tiene 2.010+ tests y debe mantenerse así

## Documentos Generados
- `Informe_Viabilidad_ai_assistant.docx` — Informe completo de viabilidad v2 (monetización, PI, licenciamiento)
  - Actualizado 21/02/2026 con estrategia PolyForm Shield (recomendada) vs Noncommercial (alternativa)
  - Incluye secciones sobre evolución del código, plan de acción y costes

## Tareas Pendientes
- [x] Actualizar el informe .docx con la estrategia PolyForm (v2 completada 21/02/2026)
- [ ] Decidir entre PolyForm Shield (recomendada) o PolyForm Noncommercial
- [ ] Sustituir LICENSE-MIT y LICENSE-APACHE por la licencia PolyForm elegida
- [ ] Actualizar campo `license` en Cargo.toml
- [ ] Registrar PI en España, WIPO PROOF y Safe Creative
- [ ] Implementar CLA en el repositorio
- [ ] Decidir si publicar módulo básico bajo MIT como gancho
- [ ] Crear página web/landing con info de licencia comercial
