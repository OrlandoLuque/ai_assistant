# V106 — Recipes (Phase A.1): declarative YAML workflows

**Version:** 0.2.52 → 0.2.53
**Date:** 2026-05-03
**Scope:** Land Phase A.1 of the Tier-1 competitive-gaps plan: recipes
as portable, declarative workflows with sub-recipe composition. Inspired
by Goose's recipes but rebuilt on our existing discovery + trust model
(slash_commands pattern) so callers don't need a new mental model.

This is the first installment of a multi-phase effort to close the
recipes / IDE-protocol / in-process-inference gap that competitors
(Goose, Hermes, Autocode, OpenHands) all ship. Phase A.2 (ACP server)
and A.3 (in-process local inference) follow. See
`ai_assistant_plans/plan_tier1_competitive_gaps.md`.

---

## 1. Recipes module (`src/recipes.rs`)

A recipe is a `.yaml` file describing a multi-step LLM workflow:
variables, prompt steps, tool steps, and sub-recipe composition. The
parser is a hand-rolled YAML *subset* tailored to the schema — no
generic YAML deps, no anchors/refs/tags, narrow trust surface.

### Schema (apiVersion: recipes/v1)

```yaml
apiVersion: recipes/v1
name: code-review
description: Review a code file for bugs and clarity
version: "1.0.0"
author: orlando.luque@gmail.com
tags: [code, review]

variables:
  file_path:
    description: Path to the file
    required: true
  focus:
    description: What to focus on
    default: "general bugs"

model: claude-opus-4-7
provider: anthropic

steps:
  - id: read_file
    type: tool
    tool: file_read
    args:
      path: "{{file_path}}"

  - id: review
    type: prompt
    prompt: |
      Review {{file_path}}, focus on {{focus}}:
      {{steps.read_file.output}}

  - id: format
    type: recipe
    recipe: format-markdown
    args:
      content: "{{steps.review.output}}"

output: "{{steps.format.output}}"
```

### Step kinds

| Kind | Effect |
|------|--------|
| `prompt` | Send rendered template to LLM via callback, capture reply |
| `tool` | Invoke a tool by name with named args via callback |
| `recipe` | Call another recipe with bound args (sub-recipe) |
| `shell` | Run a shell command — *disabled by default* |

### Variable substitution

`{{var_name}}` resolves from variable bindings. `{{steps.<id>.output}}`
resolves from prior step outputs. Unbound placeholders are left
verbatim — same defensive behavior as `slash_commands`.

### Trust model (security)

| Defense | Default |
|---------|---------|
| File size cap | 256 KiB |
| Symlinks | rejected |
| Encoding | UTF-8 enforced |
| Extensions | `.yaml`/`.yml` only |
| Sub-recipe depth | 8 max |
| Steps per recipe | 64 max |
| `shell` step | disabled (set `allow_shell` to enable) |
| YAML features | scalar key:value, block scalars `|`, inline `[a,b]`, block lists `- a` only — no anchors, refs, tags, flow `{...}` |
| Variable execution | pure substitution, never `eval` |

## 2. Discovery + Registry

`discover_recipes(roots, cfg)` mirrors `slash_commands`: ordered roots,
later roots **override** earlier on duplicate names. Default roots:

1. user-global: `<config-dir>/ai_assistant/recipes/`
2. project: `<project>/.ai_assistant/recipes/`

`RecipeRegistry::load_errors` surfaces per-file failures so callers can
warn without aborting discovery.

## 3. Execution engine (`RecipeEngine`)

Builder-style:

```rust
let engine = RecipeEngine::default()
    .with_llm(|prompt| Some(call_my_llm(prompt)))
    .with_tool(|name, args| run_my_tool(name, args));
let result = engine.run(&recipe, &bindings, &registry)?;
```

Callbacks decouple the engine from `AiAssistant` directly — same
pattern we used for CoVe LLM verification in V89. Sub-recipes are
resolved from the registry; recursion limit triggers `RecursionLimit`.

`RecipeRunResult` carries every step's output plus the final rendered
output template.

## 4. CLI verbs (`ai_cli recipes`)

| Verb | Effect |
|------|--------|
| `list` | Show discovered recipes (name / version / description) |
| `show <name>` | Print metadata + variables + steps |
| `validate <name\|path>` | Validate against schema |
| `init <name> [--out PATH]` | Scaffold a recipe template |
| `run <name> [--var k=v]` | Execute end-to-end (LLM via existing `AiAssistant`) |
| `share <name> [--out PATH]` | Produce a portable bundle |

`--user-dir` / `--project-dir` override the default roots.
`--provider` / `--model` / `--url` override the recipe's hints.

## 5. Auditor binaries (per memory rule `feedback_auditable_subsystems`)

### `ai_recipes` CLI

Read-only audit. No required features — recipes module compiles always.

```text
ai_recipes list [--dir PATH]
ai_recipes inspect <FILE|NAME> [--dir PATH]
ai_recipes validate <FILE|NAME> [--dir PATH]
ai_recipes graph [--dir PATH]      # show sub-recipe call graph
ai_recipes audit [--dir PATH]      # aggregate counts + issues, exit 1 if invalid
```

### `ai_recipes_gui` (feature `gui-recipes`)

egui visual auditor: list, metadata grid, validation status, sub-recipe
call-graph view, summary panel. Read-only.

`gui-recipes = ["dep:eframe"]` — narrower than `gui-pro`, so users can
build the auditor without dragging in the rest of the desktop stack.

## 6. Tests

25 unit tests in `recipes::tests`:

- Parser: minimal recipe, block scalar prompt, variables with
  default/required, tool steps with args, sub-recipe steps, inline
  tags lists, rejection of unsupported apiVersion, rejection of
  missing apiVersion.
- Substitution: variables, step outputs, unbound left verbatim.
- Discovery: yaml/yml loading, later root overrides earlier, oversized
  files rejected.
- Validation: duplicate step IDs, shell-when-disabled, step limit.
- Engine: prompt step with mock LLM, output chaining across steps,
  tool step with callback, sub-recipe resolution, missing required
  variable, unknown sub-recipe, recursion limit.
- Scaffold: produces valid recipe parseable + validatable.

All 25/25 pass. `cargo test --lib recipes::` confirms.

## 7. Wiring (per memory rule `feedback_wiring_checklist`)

- Module declared in `src/lib.rs` (`pub mod recipes;`).
- Public re-exports for `Recipe`, `RecipeConfig`, `RecipeEngine`,
  `RecipeError`, `RecipeRegistry`, `RecipeRunResult`, `RecipeStep`,
  `RecipeVariable`, `StepKind`, `StepOutput`, `discover_recipes`,
  `parse_recipe`, `validate_recipe`, `scaffold_recipe`,
  `recipe_substitute`, `SUPPORTED_API_VERSION`.
- CLI dispatch: `recipes` subcommand in `ai_cli`.
- Two new binaries registered in `Cargo.toml`.
- One new feature flag: `gui-recipes`.
- `print_usage()` updated.

## 8. Smoke test

Created `.ai_assistant/recipes/hello.yaml`:

```text
$ ai_cli recipes list
NAME           VERSION    DESCRIPTION
hello          0.1.0      Greet the user and echo a topic

$ ai_cli recipes validate hello
OK: 'hello' (apiVersion=recipes/v1, steps=2)

$ ai_recipes audit
Recipes audit (...recipes)
  Discovered:    1
  Valid:         1
  Total steps:   2
  Shell steps:   0 (security-sensitive)
```

## 9. Pioneer-mode lessons

- Subset YAML beats full YAML when the schema is fixed: smaller
  attack surface, no anchor injection, no megabyte-of-pointers
  expansion attacks. Same defensive posture we already use in
  `slash_commands` and `config_layered`.
- Sub-recipe call graph is auditable in the standalone `ai_recipes
  graph` view — required for trust review before running an unknown
  recipe directory.
- Recipes deliberately delegate LLM and tool execution via callbacks.
  This keeps the engine usable from CLI, server, MCP, GUI, and tests
  without coupling to any specific provider.

## 10. Next (Phase A.2 / A.3)

- A.2: ACP server (handshake <200ms, first token <1s, ≥30 chunks/s)
- A.3: In-process local inference (candle + llama-cpp-2 + CUDA)

See `ai_assistant_plans/plan_tier1_competitive_gaps.md` for the full
roadmap.
