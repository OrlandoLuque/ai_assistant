# El subsistema de investigación de `ai_assistant`

> **Para qué es este documento.** Es el resumen de lo que la librería tiene montado en
> materia de investigación académica, escrito para poder pasárselo a alguien que no ha
> visto el código — por ejemplo, para material de currículum o de perfil profesional.
> Todo lo que aparece aquí está implementado y con tests; lo que no está hecho tiene su
> propio apartado al final, porque un inventario que solo lista aciertos no sirve para
> nada.
>
> Fecha: 3 de septiembre de 2026 · versión 0.2.242 · feature flag `research`

---

## En una frase

Una tubería completa de investigación bibliográfica en Rust: busca en cinco APIs
académicas, normaliza los resultados a un tipo común, los deduplica por DOI, genera la
revisión de literatura en Markdown con su bibliografía en BibTeX, y — desde V289 — los
mete en un índice RAG para poder preguntarles después. Sin depender de ningún modelo de
lenguaje para la parte de buscar y estructurar.

## Tamaño

| Módulo | Qué hace | Líneas | Tests |
|---|---|---:|---:|
| `academic_search.rs` | Los cinco proveedores + motor multi-fuente | 2 409 | 45 |
| `bibtex.rs` | Parser y generador de BibTeX | 958 | 23 |
| `citations.rs` | Atribución de fuentes en respuestas generadas | 904 | 16 |
| `paper_metadata.rs` | Extracción estructurada del texto de un paper | 903 | 20 |
| `literature_review.rs` | La tubería de revisión bibliográfica | 862 | 20 |
| `mcp_research_tools.rs` | Las seis herramientas expuestas por MCP | 460 | 11 |
| `research_rag.rs` | El puente papers → índice RAG | 298 | 8 |
| **Total** | | **6 794** | **143** |

Sobre un total de ~523 000 líneas y 9 600+ tests en la librería entera.

---

## 1. Búsqueda académica — cinco proveedores tras una sola interfaz

Todos implementan el mismo trait (`search_papers`, `get_paper`, `get_citations`,
`get_references`) y devuelven el mismo `AcademicPaper`, así que el resto del sistema no
sabe de dónde vino cada resultado.

| Proveedor | API | Para qué es bueno |
|---|---|---|
| **arXiv** | Atom/XML | Preprints; lo último, antes de revisión por pares |
| **Semantic Scholar** | REST/JSON | Grafo de citas; estrangula fuerte sin clave |
| **PubMed** | NCBI E-utilities | Biomedicina |
| **OpenAlex** | REST/JSON | ~250 M trabajos de todas las disciplinas; el más general |
| **Crossref** | REST/JSON | El registro de DOIs: la ficha autoritativa de cualquier paper con DOI |

Las decisiones que tiene sentido contar, porque son las que separan «llama a una API» de
«esto aguanta en producción»:

- **Los abstracts de OpenAlex hay que reconstruirlos.** OpenAlex no puede redistribuir
  abstracts como prosa por motivos legales, así que envía un **índice invertido**
  (`{"palabra": [posiciones]}`) y deja el rearmado al cliente. Sin implementarlo, *todos*
  sus papers llegan sin abstract — que es casi todo lo que hace que un paper valga la pena
  indexar.
- **Backoff ante estrangulamiento, con `Retry-After`.** Un 429 llegaba antes al usuario
  como «error de red», que le mandaba a mirar su conexión en lugar de su ritmo de
  peticiones. Ahora hay un tipo de error propio y reintentos exponenciales que respetan la
  cabecera del servidor: ignorarla es como un cliente pasa de estrangulado a baneado.
- **El filtro por años va al servidor.** Filtrar en cliente significa pedir diez
  resultados y quedarse con tres; así es como una búsqueda acotada por años devuelve casi
  nada sin que se note.
- **Direcciones de contacto para el *polite pool*.** OpenAlex, Crossref y NCBI dan cuota
  mucho mayor a los clientes que se identifican. No es cortesía, es rendimiento.
- **Crossref devuelve error, no lista vacía, al pedirle quién cita un paper.** No aloja
  ese dato (está en Event Data / OpenCitations). Una lista vacía se leería como «a este
  paper no lo cita nadie», que es otra respuesta y es falsa.

El motor multi-fuente (`AcademicSearchEngine`) busca en todos los proveedores
configurados y **deduplica por DOI**, que es el único identificador que sobrevive a
encontrar el mismo paper por dos sitios distintos.

## 2. Revisión de literatura — de un tema a un documento

`LiteratureReviewPipeline` toma un tema y devuelve un `LiteratureReview`: secciones con
título y contenido, la lista de papers, y `to_markdown()` / `bib_entries()` para sacarlo.

- Dos presets: `quick()` y `systematic()`, que se diferencian en cuántos papers, qué
  profundidad de búsqueda y qué estilo de síntesis.
- Cinco formatos de bibliografía: BibTeX, APA 7ª, MLA 9ª, Chicago 17ª e IEEE.
- Cuatro estilos de síntesis: narrativo, sistemático, bibliografía anotada y comparativo.
- **No interviene ningún modelo**: la tubería busca, agrupa y estructura. Corre en una
  máquina sin GPU y sin clave de API de ningún LLM.

Verificado de punta a punta contra arXiv: 10 papers, 1 841 palabras, agrupados por año con
su cita y su abstract.

## 3. BibTeX — en los dos sentidos

`BibParser` lee `.bib` (con sus tipos de entrada, campos y errores de parseo tipados) y
`BibGenerator` los escribe. `BibGenerator::from_papers()` convierte directamente los
resultados de una búsqueda en una bibliografía. Es decir: se puede importar la biblioteca
que ya tiene el usuario y exportar la que se acaba de encontrar.

## 4. Metadatos de papers — del texto a la estructura

`paper_metadata.rs` extrae título, autores, abstract, DOI, referencias y **secciones**
(abstract, introducción, método, resultados, conclusión…) del texto plano de un paper,
por heurísticas sobre la estructura habitual de un artículo académico. Es lo que permite
tratar un PDF convertido a texto como un documento con partes, no como un bloque.

## 5. Puente con RAG — encontrar papers y poder preguntarles *(V289, lo más reciente)*

Búsqueda y RAG eran dos subsistemas que no se hablaban: se podían encontrar cincuenta
papers y quedarse con una lista. `research_rag.rs` los mete en el índice:

- **La clave es el DOI cuando lo hay.** Es lo que hace que repetir una búsqueda no
  duplique nada, porque es el único identificador estable entre proveedores. Normalizado
  (mayúsculas, prefijos `doi:` y `https://doi.org/`), con respaldo a la URL y, en último
  lugar, a un id cualificado por proveedor — uno desnudo dejaría que dos APIs que usan
  «12345» se pisaran el paper.
- **Los metadatos van dentro del texto indexado.** El índice guarda trozos; «quién
  escribió el paper de 2024 sobre X» no se responde desde un trozo que solo tiene el
  abstract.
- Un fallo no pierde el lote: el informe separa «ya estaba» de «falló», que son cosas
  distintas.

## 6. Superficie de uso

**CLI** (`ai_cli`, sin modelo para nada de esto):

```
ai_cli research <query> --providers openalex,crossref --max-results 10 --bibtex
ai_cli research <query> --index papers.db          # ingiere lo encontrado
ai_cli research ask "<pregunta>" --index papers.db # pregunta al índice
ai_cli research review <tema> --mode systematic --out revision.md --bibtex
```

**MCP** — seis herramientas expuestas a cualquier cliente del protocolo:
`search_papers`, `get_paper_metadata`, `import_bibtex`, `export_bibtex`,
`literature_review`, `extract_paper_metadata`.

**Multi-agente** — tres roles pensados para este dominio: `ResearchAssistant` (busca,
filtra, resume), `PeerReviewer` (critica borradores y verifica afirmaciones) y
`WritingCoach` (estilo académico).

**Rust** — todo lo anterior es API pública de la librería.

---

## Lo que NO está hecho

Esto importa tanto como lo anterior:

- **Los roles de agente de investigación son prompts de sistema, no tuberías cerradas.**
  Están cableados al bucle de agente (`agent_wiring.rs`) y se orquestan como cualquier
  otro rol, pero no hay un «revisor por pares» que coja un borrador y lo devuelva anotado
  de principio a fin.
- **No hay descarga ni parseo de PDF dentro del subsistema.** `paper_metadata` trabaja
  sobre texto; convertir el PDF es cosa del módulo de documentos y no está encadenado con
  la búsqueda.
- **La síntesis de la revisión es estructural, no interpretativa.** Agrupa, ordena y cita;
  no argumenta. Para argumentar hace falta un modelo, y esa parte no está atada a la
  tubería.
- **`get_citations` de arXiv y de Crossref no existe** porque esas APIs no lo ofrecen. Se
  informa del hecho en lugar de devolver vacío.
- **La feature `research` a secas no compila** (arrastra `mcp_protocol` sin declararlo en
  el manifiesto). Compila dentro del conjunto mínimo soportado, que incluye `rag` y
  `tools`. Está registrado como deuda conocida.

## Cómo verificar cualquier afirmación de este documento

Todo lo anterior tiene tests y casi todo se ha ejercitado contra las APIs reales, no solo
contra ejemplos guardados. Los 143 tests corren con:

```
cargo test --features "full" --lib academic_search
cargo test --features "full" --lib research_rag
cargo test --features "full" --lib literature_review
cargo test --features "full" --lib bibtex
```

Las verificaciones en vivo están anotadas, con sus cifras, en las entradas V287–V290 del
`CHANGELOG.md`.
