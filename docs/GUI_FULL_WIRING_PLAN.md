# Plan: Full Feature Wiring for ai_gui

## Iteration 1 — Initial Architecture

### Problem
The GUI (ai_gui.rs, 2816 lines) only exposes chat, model wizard, butler, RAG, sessions,
and basic monitoring. 20+ features compile but have no UI. Need to wire everything.

### Architecture Decision: Code Organization

**Current**: Single file `src/bin/ai_gui.rs` (2816 lines)
**Proposed**: Split into `src/bin/ai_gui/main.rs` + panel modules

```
src/bin/ai_gui/
  main.rs              — app struct, eframe::App impl, startup, routing
  panels/
    mod.rs             — panel trait + exports
    chat.rs            — chat area (existing, extracted)
    model_wizard.rs    — model library (existing, extracted)
    agents.rs          — multi-agent orchestration panel
    security.rs        — security config + audit
    memory.rs          — advanced memory viewer
    scheduler.rs       — cron job manager
    tools.rs           — MCP tools + function calling
    browser.rs         — browser automation
    sandbox.rs         — code execution sandbox
    workflows.rs       — workflow editor
    eval.rs            — evaluation + benchmarks
    analytics.rs       — full analytics dashboard
    media.rs           — vision + image/video generation
    audio.rs           — STT/TTS controls
    cloud.rs           — S3/GDrive browser
    devtools.rs        — agent debugger/profiler
    prompt_opt.rs      — prompt signature optimizer
    distributed.rs     — cluster dashboard
    settings.rs        — expanded settings (existing, extracted + expanded)
```

### Navigation: Sidebar Redesign

**Current sidebar**: Sessions | Knowledge | Butler (3 tabs)

**New sidebar with sections**:

```
CHAT
  [Chat]              — main chat (default view)
  [Sessions]          — session management

AI AGENTS
  [Agent Pool]        — create/manage agents, roles, orchestration
  [Tools / MCP]       — tool registry, function calling, MCP
  [Autonomous]        — scheduler, browser, code sandbox
  [Workflows]         — event-driven workflow editor

KNOWLEDGE
  [RAG Sources]       — .kpkg, files (existing)
  [Memory]            — episodic/procedural/entity memory
  [Cloud Storage]     — S3/GDrive browser

GENERATION
  [Vision]            — image input for vision models
  [Media]             — image/video generation (DALL-E, SD)
  [Audio]             — STT/TTS controls
  [Constrained]       — grammar editor for local models

OPTIMIZATION
  [Prompt Lab]        — prompt signatures, optimization
  [Evaluation]        — benchmarks, A/B testing, hallucination
  [Analytics]         — full metrics dashboard

SYSTEM
  [Security]          — RBAC, PII, guardrails, audit log
  [Butler]            — environment advisor (existing)
  [DevTools]          — agent debugger/profiler
  [Cluster]           — distributed nodes, DHT, CRDTs
  [Settings]          — all configuration (expanded)
```

### Panel Implementation Pattern

Each panel follows the same pattern:

```rust
// In panels/agents.rs
pub struct AgentPanel {
    // panel-specific state
}

impl AgentPanel {
    pub fn new() -> Self { ... }
    pub fn render(&mut self, ui: &mut Ui, assistant: &mut AiAssistant) { ... }
}
```

Main app holds all panels and routes based on sidebar selection.

### Settings Expansion

Current settings: URLs, temperature, history depth, enter_sends

New settings sections:
- **Provider**: URLs, API keys, timeouts, default model
- **Generation**: temperature, top_p, max_tokens, frequency_penalty, presence_penalty
- **Security**: enable/disable PII detection, guardrails mode, rate limits
- **RAG**: chunk size, overlap, similarity threshold, max results
- **Agents**: default autonomy level, max concurrent agents, approval policy
- **Memory**: episodic capacity, decay rate, consolidation frequency
- **Streaming**: SSE/WebSocket preference, buffer size, compression
- **UI**: theme (dark/light), font size, enter_sends, show monitor
- **Advanced**: binary storage, integrity check, telemetry

### State Management

Each panel has its own state struct. The main app struct holds:
- `AiAssistant` (shared, mutable reference passed to panels)
- All panel structs
- Current sidebar selection
- Global UI state (toasts, dialogs)

### HITL Integration

HITL is special — it needs to pop up approval dialogs at any time, not just when
its panel is open. Implementation:
- `HitlOverlay` renders on top of everything when there are pending approvals
- Approval queue stored in main app state
- Any agent action that requires approval pushes to the queue

### Monitor Panel Expansion

Current: Overview, Metrics, Analysis, Graph, Audit (5 tabs)
Keep as bottom panel, but enrich with data from all modules.

### Implementation Order

Phase 1 — Refactor (split into modules, new sidebar):
1. Create `src/bin/ai_gui/` directory structure
2. Extract existing code into modules
3. Implement new sidebar navigation
4. Verify everything still works

Phase 2 — Settings & Config panels:
5. Expanded settings with all config sections
6. Security config panel

Phase 3 — Agent & Automation panels:
7. Agent Pool panel (multi-agent)
8. Tools/MCP panel
9. Scheduler panel
10. Browser automation panel
11. Code sandbox panel
12. Workflow editor panel
13. HITL overlay

Phase 4 — Knowledge & Memory panels:
14. Advanced Memory viewer
15. Cloud storage browser

Phase 5 — Generation panels:
16. Vision (image input in chat)
17. Media generation panel
18. Audio (STT/TTS) panel
19. Constrained decoding panel

Phase 6 — Optimization & Monitoring panels:
20. Prompt Lab panel
21. Evaluation panel
22. Full Analytics dashboard
23. DevTools panel
24. Cluster dashboard panel

Phase 7 — Polish:
25. HITL approval overlay integration
26. Cross-panel navigation (e.g., click agent → goes to agent panel)
27. Persistent panel state (save/load UI state)
28. Monitor panel enrichment from all modules


## Iteration 2 — Review & Improvements

### Issues Found:

1. **File split is risky**: Moving from single file to module directory changes the
   binary target path in Cargo.toml and can break the build. Also increases complexity
   of the refactor significantly.

   **Decision**: Keep single file for now. The file will be large (~6000-8000 lines)
   but Rust handles this fine. Each panel is a method group on AiGuiApp.
   Use `// === PANEL: AgentPool ===` section markers for navigation.
   Rationale: Less risk, faster implementation, no Cargo.toml changes.

2. **Sidebar with 20+ items is overwhelming**: The proposed sidebar has too many items.
   Users will get lost.

   **Decision**: Use collapsible category headers. Default: only CHAT expanded.
   Other categories collapsed. Each category shows its items when expanded.
   Also add a search/filter for panels.

3. **Panel state initialization**: Many panels need async initialization (e.g.,
   fetching cluster status, loading memory). Can't block the UI thread.

   **Decision**: Panels use lazy initialization with mpsc channels (same pattern as
   model fetching). State is `Option<T>` — None until loaded, show spinner.

4. **AiAssistant doesn't expose all module APIs**: Some features are behind
   module-level functions, not methods on AiAssistant.

   **Decision**: Panels can import directly from `ai_assistant::module_name`
   when needed. Not everything needs to go through AiAssistant.

5. **Missing: System prompt editor**: Users should be able to set/edit the system
   prompt from the GUI. Currently hardcoded or absent.

   **Added**: System prompt textarea in Settings > Generation section.

6. **Missing: Model comparison**: Users might want to send the same prompt to
   multiple models and compare responses side-by-side.

   **Deferred**: Nice to have but not core. Can add later.

7. **Chat area needs to support images**: For vision models, users need to
   attach images. For media generation, results should display inline.

   **Decision**: Add image attachment button to chat input. Display generated
   images inline in the message history.

### Gains from Iteration 2:
- Avoided risky file split (saves ~2h refactor, eliminates breakage risk)
- Better UX with collapsible sidebar (prevents overwhelm)
- Identified async initialization pattern (prevents UI freezes)
- Found missing system prompt editor (core usability gap)
- Planned image support in chat (required for vision/media features)


## Iteration 3 — Feasibility & Edge Cases

### Issues Found:

1. **Panel render signature**: Panels need access to different subsets of state.
   If everything is methods on AiGuiApp, they have full `&mut self` access. Fine.
   But some panels need to trigger actions that affect other panels (e.g., agent
   panel creates a task that appears in scheduler).

   **Decision**: Use the same deferred action pattern as model wizard. Panels
   collect actions as `Option<Action>`, main render loop processes them after
   all panels render. Action enum covers cross-panel operations.

2. **Sidebar selection enum will be large**: 20+ variants.

   **Decision**: Fine. It's just an enum. Use a two-level selection:
   `SidebarCategory` (7 categories) + `SidebarPanel` (specific panel within category).
   Rendering checks both.

3. **Memory viewer needs search**: Advanced memory has 3 stores (episodic,
   procedural, entity). Each needs search + filters.

   **Decision**: Memory panel has 3 sub-tabs, each with search bar and
   filter controls. Results in scrollable list.

4. **Workflow editor needs graph rendering**: Similar to existing knowledge
   graph but for workflow DAGs.

   **Decision**: Reuse existing petgraph + force-directed layout code from
   knowledge graph panel. Extract shared graph rendering helper.

5. **Code sandbox security**: Displaying arbitrary code execution in GUI.
   Need clear visual indicators of sandbox status.

   **Decision**: Color-coded execution status (green=safe, yellow=warning,
   red=blocked). Show blocked commands prominently.

6. **Browser panel without actual browser embed**: Can't embed Chrome in egui.

   **Decision**: Panel shows: URL input, action buttons (navigate, click, type),
   screenshot preview (rendered as egui image), page content as text.
   Not a full browser — a remote control UI.

7. **Audio without real-time capture**: egui doesn't do audio recording.

   **Decision**: Audio panel is for configuration and file-based STT/TTS only.
   File picker for audio input → transcription result. Text input → TTS output
   saved to file. Real-time voice would need a separate feature.

8. **Cloud connectors need credentials**: S3/GDrive need auth.

   **Decision**: Config fields in Settings (access key, secret, bucket, region
   for S3; OAuth token for GDrive). Panel only active when configured.

9. **Panels that have no local state to show**: Some features (like
   integrity-check) are fire-and-forget. Don't need a full panel.

   **Decision**: Integrity check goes into Settings > System as a button +
   status label. Not a separate panel. Remove from sidebar.

10. **Constrained decoding is very niche**: GBNF grammar editor is only useful
    for power users with local models.

    **Decision**: Keep but make it a sub-panel of Settings > Generation >
    Advanced, not a top-level sidebar item. Reduces sidebar clutter.

### Revised Sidebar:

```
CHAT
  Chat                — main chat (default) + vision image attach
  Sessions            — session management

AGENTS
  Agent Pool          — create/manage agents, roles, status
  Tools / MCP         — tool registry, approval queue
  Automation          — scheduler + browser + code sandbox (3 sub-tabs)

KNOWLEDGE
  RAG Sources         — .kpkg, files (existing)
  Memory              — episodic/procedural/entity (3 sub-tabs)
  Cloud Storage       — S3/GDrive browser

GENERATE
  Media               — image/video generation
  Audio               — STT/TTS

OPTIMIZE
  Prompt Lab          — signatures, optimization
  Evaluation          — benchmarks, A/B, hallucination

SYSTEM
  Security            — RBAC, PII, guardrails, audit
  Analytics           — full metrics dashboard
  DevTools            — agent debugger/profiler
  Cluster             — distributed nodes
  Butler              — environment advisor (existing)
  Settings            — all config (expanded)
```

Total: 16 sidebar items in 6 categories (down from 20+).
Constrained decoding → inside Settings.
Integrity check → inside Settings.
Workflows → inside Automation sub-tab.

### Gains from Iteration 3:
- Reduced sidebar from 20+ to 16 items (less overwhelm)
- Identified deferred action pattern for cross-panel ops
- Resolved browser/audio/cloud UX constraints
- Moved niche features to sub-panels (cleaner navigation)
- Planned graph rendering reuse (less code duplication)


## Iteration 4 — Implementation Details & Final Polish

### Issues Found:

1. **Estimated total new code**: ~3500-4500 lines across all panels.
   With existing 2816 lines, total ~6300-7300 lines. Manageable for single file
   with clear section markers.

2. **Import bloat**: Many new imports needed from ai_assistant modules.

   **Decision**: Group imports by feature area with comments.

3. **Panel state structs**: Each of the 16 panels needs its own state.
   Adding 16 fields to AiGuiApp is messy.

   **Decision**: Group panel states into a `PanelStates` struct:
   ```rust
   struct PanelStates {
       agents: AgentPanelState,
       tools: ToolsPanelState,
       // ...
   }
   ```
   AiGuiApp has one field: `panels: PanelStates`.

4. **Feature gates in GUI code**: Some features are behind cfg gates in the lib.
   The GUI code should also use cfg gates for panel code that depends on specific
   features, in case someone compiles with `--features gui` but without `full`.

   **Decision**: Since `gui` now includes `full` + all relevant optionals, this
   is not needed. All features are guaranteed available when `gui` is active.
   But add a note in comments for future reference.

5. **Testing**: How to test GUI panels?

   **Decision**: Panels are pure rendering code (no side effects during render,
   actions collected as deferred). Testing is manual for now. The underlying
   library functions are already tested (2730+ tests).

6. **Performance**: Rendering 16 panels shouldn't be an issue since only one
   is visible at a time. But the sidebar itself renders all category headers.

   **Decision**: Fine. Category headers are just labels. No performance concern.

7. **Accessibility of sub-tabs**: Automation has 3 sub-tabs (scheduler, browser,
   sandbox). Memory has 3 sub-tabs. These are tabs within the panel.

   **Decision**: Use `ui.selectable_value()` horizontal tab bar at top of panel,
   same pattern as model wizard and monitor panel.

### Final Panel Specifications:

| Panel | State Fields | Key Widgets | Actions |
|-------|-------------|-------------|---------|
| Agent Pool | agents list, selected agent, new agent form | Agent cards, role selector, status badges | Create/delete agent, assign task, view messages |
| Tools/MCP | tool list, selected tool, approval queue | Tool cards, parameter viewer, approval buttons | Register tool, approve/deny, test tool |
| Automation | scheduler state, browser state, sandbox state | Sub-tabs, cron editor, URL bar, code editor | Add job, navigate, execute code |
| Memory | episodic/procedural/entity lists, search | Sub-tabs, timeline, search bar, detail view | Search, add/delete entries, consolidate |
| Cloud Storage | file list, current path, upload state | File tree, breadcrumb, upload button | Browse, upload, download, delete |
| Media | prompt, model selector, gallery | Text input, model combo, image grid | Generate image, save, delete |
| Audio | STT config, TTS config, file picker | Language/voice selectors, file drop | Transcribe file, synthesize text |
| Prompt Lab | signature fields, optimization state | Field editor, metric chart, results | Create signature, run optimization |
| Evaluation | test suite, results, experiments | Sample editor, results table, A/B chart | Run eval, create experiment |
| Security | RBAC config, PII config, audit log | Toggle switches, rule editor, log viewer | Enable/disable features, add rules |
| Analytics | metric selectors, time range, charts | Metric combo, time selector, bar/line charts | Export data, set alert thresholds |
| DevTools | recording state, breakpoints, events | Timeline, state inspector, breakpoint list | Start/stop recording, add breakpoint, replay |
| Cluster | node list, DHT status, tasks | Node cards, topology view, task queue | Add node, submit task, bootstrap |
| Settings | all config fields organized by section | Collapsible sections, sliders, inputs | Apply, reset defaults, export/import |

### Gains from Iteration 4:
- PanelStates struct prevents AiGuiApp bloat
- Confirmed no cfg gates needed (gui includes everything)
- Defined exact panel specifications (clear implementation target)
- Estimated code size: manageable
- Minor: no new issues found → plan is stable


## Iteration 5 — Diminishing Returns Check

### Reviewed:
- Architecture: single file ✓
- Navigation: collapsible sidebar with 16 items in 6 categories ✓
- Panel pattern: methods on AiGuiApp with PanelStates ✓
- Cross-panel: deferred actions ✓
- Async: mpsc channels for background ops ✓
- Edge cases: browser, audio, cloud, HITL all addressed ✓
- Settings: comprehensive expansion ✓
- Missing features: all accounted for ✓

### New findings: None significant.

**Conclusion**: Plan is stable. Iteration gains are now insignificant. Proceeding to implementation.


## Auto-Detection of ai_assistant Binaries

When Butler scans the environment, it should also detect other ai_assistant binaries
running as services/nodes on the local machine:

| Binary | Default Port | Detection |
|--------|-------------|-----------|
| `ai_assistant_server` | 3000 | HTTP GET /health or /api/v1/status |
| `ai_cluster_node` | 4000 | HTTP GET /health or QUIC handshake |

Detection logic:
1. During Butler scan, probe `localhost:3000` and `localhost:4000` for known ai_assistant endpoints
2. Check `AI_ASSISTANT_SERVER_PORT` and `AI_CLUSTER_NODE_PORT` env vars for custom ports
3. If detected, show them in the Butler panel as "Local Services" section
4. In the Cluster panel, auto-add detected nodes to the node list
5. Optional: check Windows services / systemd units for ai_assistant processes

This integrates naturally into the existing Butler scan → `apply_scan_result()` flow.


## Gain Summary

| Iteration | Key Gains |
|-----------|-----------|
| 1 (Initial) | Full architecture, navigation design, 28-step implementation order |
| 2 (+35%) | Avoided risky file split, collapsible sidebar, async init pattern, system prompt, image support |
| 3 (+25%) | Reduced sidebar 20→16, deferred actions for cross-panel, resolved browser/audio/cloud UX, moved niche features |
| 4 (+10%) | PanelStates struct, no cfg gates needed, exact panel specs, code size estimate |
| 5 (<5%) | Confirmed stability, no new issues → STOP |
