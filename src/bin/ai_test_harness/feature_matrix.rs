use super::*;

use std::collections::BTreeSet;
use std::process::Command;

// ─── Does every feature combination actually build? ───────────────────────────
//
// The crate has ninety-odd feature flags. CI has had a "feature matrix" job for a
// long time, and it was measuring the wrong thing: it ran
//
//     cargo check --features "<flag>"
//
// without `--no-default-features`. Since `default = ["full"]`, every entry compiled
// **full + flag** — thirty jobs that were near-duplicates of the full build. Nothing
// was ever compiled BELOW `full`, which is how the reduced build rotted until it did
// not compile at all (V267).
//
// This category compiles `MIN + flag` for each flag, where MIN is the smallest set
// that builds. When a combination fails it reports the first real error, because
// "some combination is broken" is not actionable and "`rag` alone is missing
// rusqlite at src/x.rs:12" is.
//
// The flag list is READ FROM Cargo.toml rather than hardcoded, so a flag added
// tomorrow is covered without anyone remembering to add it here.
//
// Slow by nature — one cargo invocation per flag. Excluded from `--all`; run it
// deliberately with `--category=feature_matrix`, optionally narrowed with
// `--filter=<substring>`.
//
// IT REPORTS WHAT *THIS* PLATFORM BUILDS. Development here is Windows and CI is
// ubuntu, so the two can legitimately disagree — a flag whose C dependency
// generates different bindings per target will differ, and a `#[cfg(unix)]` path
// is never compiled here at all (which is exactly how CI stayed red for days in
// V268). A failure seen locally is a lead, not a verdict, until CI agrees.

/// The smallest set that builds. Must stay in step with `FEATURES_MIN` in
/// `.github/workflows/ci.yml`; `min_set_matches_ci` below fails if they drift.
const MIN: &str = "tools,security,advanced-streaming,rag,adapters,analytics,embeddings,documents";

/// Flags that cannot be checked this way, with the reason. Kept explicit — a
/// silent skip is how coverage quietly shrinks.
const SKIP: &[(&str, &str)] = &[
    ("default", "alias for `full`, not a real combination"),
    (
        "full",
        "the always-tested baseline, covered by every other CI job",
    ),
];

/// Features deliberately absent from CI's matrix, each with the reason.
///
/// The list exists so that "not in CI" is a decision someone wrote down rather
/// than an accident nobody noticed. `whisper-local` is the case that prompted
/// it: a declared feature that **no CI job has ever compiled**, so neither its
/// breakage nor its repair would have shown up anywhere.
#[cfg(test)]
const NOT_IN_CI: &[(&str, &str)] = &[
    ("default", "alias for `full`"),
    ("full", "the baseline every other job builds on"),
    (
        "whisper-local",
        "needs libclang on the runner, and currently fails on Windows/MSVC: \
         whisper-rs-sys' bindgen emits glibc types (_IO_FILE, _G_fpos_t) that \
         overflow a layout assert. See N27; workaround is WHISPER_DONT_GENERATE_BINDINGS",
    ),
    (
        "local-inference-llama-cpp",
        "needs libclang on the runner, same bindgen prerequisite",
    ),
    ("gui", "brings windowing deps; not worth runner time"),
    ("gui-pro", "as `gui`"),
    ("egui-widgets", "as `gui`"),
    ("audio-io", "needs host audio devices"),
    ("video-io", "needs host video devices"),
    (
        "webrtc",
        "heavy native stack, covered by `voice-agent` in practice",
    ),
    (
        "vector-lancedb",
        "heavy native dep; the other vector backends cover the trait",
    ),
    ("aws-bedrock", "needs AWS credentials"),
    ("server-tls", "needs PEM material at build time"),
];

/// The members of `full`, which CI builds in its check / clippy / test jobs via
/// `FEATURES_STD`. A feature inside `full` is compiled by CI even when it has no
/// matrix entry of its own, and counting it as uncovered would be false.
#[cfg(test)]
fn full_members(manifest: &str) -> Vec<String> {
    let Some(start) = manifest.find(
        "
full = [",
    ) else {
        return Vec::new();
    };
    let Some(end) = manifest[start..].find(
        "
]",
    ) else {
        return Vec::new();
    };
    manifest[start..start + end]
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with('#') {
                return None;
            }
            let t = t.trim_start_matches("full = [").trim();
            let t = t.trim_end_matches(',').trim();
            let t = t.trim_matches('"');
            (!t.is_empty()
                && t.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
            .then(|| t.to_string())
        })
        .collect()
}

/// Declared features that **no CI job compiles today**, recorded so the number
/// cannot grow unnoticed.
///
/// This is a ratchet, not an excuse list: every name here is a real gap awaiting
/// a decision (put it in CI, or drop the feature). What the test enforces is
/// that a *new* feature cannot join them silently — which is exactly how
/// `whisper-local` came to be declared, shipped and never built by anything.
///
/// Shrinking this list is the goal; growing it must be deliberate.
///
/// **Measured, not assumed (2026-09-11, full battery, 84/87):** every feature
/// below *compiles* on top of the minimum set. None of them is broken — they are
/// simply absent from CI, so nothing would notice if one broke tomorrow. The only
/// three combinations that fail are the native-toolchain ones already excused in
/// [`NOT_IN_CI`] (`vector-lancedb`, `whisper-local`, `local-inference-llama-cpp`),
/// and they fail here for a *local* reason — no libclang / C++ toolchain on the
/// dev laptop — which is a lead about this machine, not a verdict about the code.
///
/// So the cost of closing this list is CI minutes, not repair work. That is a
/// budget decision for the maintainer, which is why the list records the fact
/// rather than acting on it.
#[cfg(test)]
const UNCOVERED_BACKLOG: &[&str] = &[
    // Platform-specific: the Linux runner cannot build these at all.
    "hardware-rocm",
    "hardware-metal",
    // Genuinely uncovered. Each needs triage.
    "wasm",
    "vector-pgvector",
    "prompt-fragments",
    "stall-detection",
    "stall-detection-llm",
    "sub-agents",
    "acp",
    // V330 removed `local-inference` from here by putting it in CI: it declares
    // no dependencies of its own (the umbrella is the `Backend` trait plus the
    // stub), so the CI minutes this list was weighing were almost none, and
    // leaving it out meant the V330 `LlmProvider` bridge compiled on no job.
    "local-inference-candle",
    "embeddings-local",
    "redis-backend",
    "skill-forge",
    "prompt-synthesis",
    "feedback-loop",
    "prompt-breeder",
    // GUI variants: windowing deps, same reason as `gui` itself.
    "gui-logs",
    "gui-recipes",
    "gui-acp",
    "gui-corrections",
    "gui-local-inference",
];

/// Features that **cargo does enable** in some CI job --- so a compile error in
/// them turns CI red --- but that **no `cargo test` job exercises**.
///
/// Esta lista nacio al partir en dos una pregunta que estaba junta (2026-10-26).
/// Al ensenar al ratchet a seguir el grafo de features --- necesario, porque si
/// no acusaba a `tabular-sqlite` de no compilarse mientras un trabajo lo
/// probaba --- cinco entradas del backlog pasaron a contar como cubiertas.
///
/// Tres de las cinco lo estaban de verdad: `aes-gcm`, `zip` y `pdf-extract` los
/// enciende `security` / `documents`, que SI corren en trabajos de `cargo test`,
/// asi que sus tests se ejecutan. La nota que las listaba decia «nada las
/// ejercita por si mismas», que es otra propiedad --- compilarlas en SOLITARIO
/// --- y de esa ya se encarga esta misma categoria del harness.
///
/// Las otras dos no: `code-sandbox` y `cloud-connectors` entran solo por
/// `gui-pro`, y ese trabajo es `cargo clippy --bin`. `cloud-connectors` tiene
/// **57 tests, diez de ellos de SSRF, que no ejecuta nadie** --- justo el
/// agujero que motivo todo esto.
///
/// O sea que «lo compila alguien» y «lo prueba alguien» son dos preguntas
/// distintas, y responder solo la primera deja pasar regresiones de
/// comportamiento enteras. Mezclarlas habria hecho que la puerta dejara de
/// senalar precisamente lo que habia que senalar.
#[cfg(test)]
const COMPILED_BUT_NOT_EXERCISED: &[(&str, &str)] = &[
    (
        "code-sandbox",
        "lo enciende `gui-pro`, y ese trabajo es `cargo clippy --bin`: compila la \
         libreria y no corre un solo test",
    ),
    (
        "cloud-connectors",
        "lo enciende `gui-pro`, y ese trabajo es `cargo clippy --bin`: sus 57 tests \
         --- diez de ellos de SSRF, los que comprueban que apuntar S3 a \
         169.254.169.254 se bloquea --- no se ejecutan en ningun sitio. Tarea N162",
    ),
];

/// Every feature name CI compiles: the matrix entries plus the `FEATURES_*`
/// environment sets used by the check, clippy and test jobs.
#[cfg(test)]
fn ci_matrix_features(ci_yml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_matrix = false;
    for line in ci_yml.lines() {
        let t = line.trim();
        // The env blocks are feature sets CI really builds too: the check,
        // clippy and test jobs run against FEATURES_STD and FEATURES_NETWORK,
        // so a feature named there is compiled even with no matrix entry.
        for key in ["FEATURES_STD:", "FEATURES_NETWORK:", "FEATURES_MIN:"] {
            if let Some(rest) = t.strip_prefix(key) {
                for name in rest.trim().trim_matches('"').split(',') {
                    let name = name.trim();
                    if !name.is_empty() {
                        out.push(name.to_string());
                    }
                }
            }
        }
        if t.starts_with("features:") {
            in_matrix = true;
            continue;
        }
        if !in_matrix {
            continue;
        }
        // The list is `- "a"` / `- "a,b"`; anything else ends it.
        let Some(rest) = t.strip_prefix("- ") else {
            if !t.is_empty() && !t.starts_with('#') {
                in_matrix = false;
            }
            continue;
        };
        for name in rest.trim().trim_matches('"').split(',') {
            let name = name.trim();
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Feature names written straight into a `--features` argument of a CI step.
///
/// [`ci_matrix_features`] reads the matrix entries and the `FEATURES_*` env
/// blocks, which is where most of CI lives --- but not all of it. Several jobs
/// name their features inline: `--features "tabular,tabular-polars"`,
/// `--features gui,gui-pro`, `--features "full,browser"`. Those are builds as
/// real as any other, and not reading them made the ratchet report
/// `tabular-polars` as uncovered while a job was testing it (found 2026-10-26,
/// by running the ratchet for the first time in a while).
///
/// A gate that cries wolf is worse than no gate: the cheap way to silence it is
/// to add the name to the backlog with an invented reason, and then the list
/// says "genuinely uncovered" about something CI builds. A written reason that
/// is false looks reviewed, so nobody ever re-reads it.
#[cfg(test)]
fn inline_features(ci_yml: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (_, rest) in ci_yml
        .match_indices("--features")
        .map(|(i, _)| (i, &ci_yml[i + "--features".len()..]))
    {
        // `--features "a,b"` or `--features a,b`, up to whitespace or quote.
        let rest = rest.trim_start();
        let rest = rest.strip_prefix('"').unwrap_or(rest);
        let end = rest.find(['"', ' ', '\n', '\r']).unwrap_or(rest.len());
        for name in rest[..end].split(',') {
            let name = name.trim();
            // `$FEATURES_STD` and friends are expanded by `ci_matrix_features`
            // from the env block; skip the variable spellings here.
            if name.is_empty() || name.starts_with('$') || name.contains("${{") {
                continue;
            }
            if name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Las features que alguna linea de `cargo test` ejercita de verdad.
///
/// Distinta de [`ci_matrix_features`] a proposito: esa dice que se COMPILA, y
/// esta que se PRUEBA. Los trabajos de `cargo check` y `cargo clippy` cazan un
/// fallo de compilacion y ninguna regresion de comportamiento --- que es
/// exactamente la situacion de `cloud-connectors`, cuyos 57 tests no corre
/// nadie porque su feature solo entra por el clippy de la GUI.
///
/// `$FEATURES_STD` y companía se expanden desde el bloque `env`, y
/// `${{ matrix.features }}` desde las entradas de la matriz, porque ese trabajo
/// corre `cargo test --lib` sobre cada una.
#[cfg(test)]
fn test_job_features(ci_yml: &str, env: &[(&str, String)], matrix: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for line in ci_yml.lines() {
        if !line.contains("cargo test") {
            continue;
        }
        let Some(at) = line.find("--features") else {
            continue;
        };
        let rest = line[at + "--features".len()..].trim_start();
        let rest = rest.strip_prefix('"').unwrap_or(rest);
        let end = rest.find(['"', ' ']).unwrap_or(rest.len());
        for name in rest[..end].split(',') {
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            // El orden de estas dos ramas importa, y al reves esta MAL: el token
            // de la matriz es `${{` (la expresion lleva espacios, asi que al
            // cortar por el primer espacio solo queda eso), y `${{` empieza por
            // `$`. Con la rama de variables primero, se buscaba una variable
            // llamada `{{`, no se encontraba, y la entrada se descartaba EN
            // SILENCIO: las 36 entradas de la matriz desaparecian de lo
            // «probado» y la puerta acusaba a `server-axum` de no tener tests.
            if name.contains("${{") {
                out.extend(matrix.iter().cloned());
            } else if let Some(var) = name.strip_prefix('$') {
                if let Some((_, value)) = env.iter().find(|(k, _)| *k == var) {
                    out.extend(value.split(',').map(|s| s.trim().to_string()));
                }
            } else if name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// What each feature pulls in, read from `[features]`.
///
/// Needed because "CI compiles this feature" has to include what cargo enables
/// *because* of it: the tabular job builds `tabular`, and cargo therefore builds
/// `tabular-sqlite`, which `tabular` lists. Without this the ratchet flags the
/// member and the only ways to quiet it are both wrong --- excusing a feature
/// that is built, or adding a CI job that duplicates one.
#[cfg(test)]
fn feature_graph(manifest: &str) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut in_features = false;
    let mut current: Option<(String, Vec<String>)> = None;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') && !t.starts_with("[[") {
            if let Some((name, deps)) = current.take() {
                out.insert(name, deps);
            }
            in_features = t == "[features]";
            continue;
        }
        if !in_features || t.starts_with('#') {
            continue;
        }
        // Collect the names inside the brackets, which may span lines.
        if let Some((name, rest)) = t.split_once('=') {
            let name = name.trim();
            if !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                if let Some((prev, deps)) = current.take() {
                    out.insert(prev, deps);
                }
                current = Some((name.to_string(), members_of(rest)));
                if rest.contains(']') {
                    if let Some((prev, deps)) = current.take() {
                        out.insert(prev, deps);
                    }
                }
                continue;
            }
        }
        if let Some((_, deps)) = current.as_mut() {
            deps.extend(members_of(t));
            if t.contains(']') {
                if let Some((name, deps)) = current.take() {
                    out.insert(name, deps);
                }
            }
        }
    }
    if let Some((name, deps)) = current.take() {
        out.insert(name, deps);
    }
    out
}

/// The quoted names on a line, minus `dep:` and `crate/feature` spellings:
/// those name cargo dependencies, not features of this crate.
#[cfg(test)]
fn members_of(fragment: &str) -> Vec<String> {
    fragment
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|n| !n.starts_with("dep:") && !n.contains('/'))
        .map(|n| n.to_string())
        .collect()
}

/// Everything cargo ends up enabling, starting from `seed`.
#[cfg(test)]
fn expand(
    seed: impl IntoIterator<Item = String>,
    graph: &std::collections::BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut pending: Vec<String> = seed.into_iter().collect();
    while let Some(f) = pending.pop() {
        if !out.insert(f.clone()) {
            continue;
        }
        if let Some(deps) = graph.get(&f) {
            pending.extend(deps.iter().cloned());
        }
    }
    out
}

/// Every feature name declared in `[features]`.
///
/// Parsed from the manifest so the matrix cannot silently fall behind the crate.
fn declared_features(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_features = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if !in_features || t.starts_with('#') || t.is_empty() {
            continue;
        }
        // `name = [...]` — take the left of the first `=`.
        let Some((name, _)) = t.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            out.push(name.to_string());
        }
    }
    out
}

/// First line that looks like a compiler error, plus where it happened.
///
/// The whole point of the category: report WHY, not just that it broke.
fn first_error(stderr: &str) -> String {
    let mut msg = None;
    for (i, line) in stderr.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("error") && msg.is_none() {
            msg = Some((i, t.to_string()));
        }
        if let Some((mi, ref m)) = msg {
            // The `--> file:line` note usually follows within a couple of lines.
            if i > mi && t.starts_with("-->") {
                return format!("{m}  at {}", t.trim_start_matches("-->").trim());
            }
            if i > mi + 6 {
                return m.clone();
            }
        }
    }
    msg.map(|(_, m)| m)
        .unwrap_or_else(|| "failed with no recognisable error line".to_string())
}

fn cargo_cmd() -> Option<&'static str> {
    let ok = Command::new("cargo")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some("cargo")
}

/// `cargo check --no-default-features --features "<features>"`.
fn check(cargo: &str, features: &str) -> Result<(), String> {
    let out = Command::new(cargo)
        .args([
            "check",
            "--quiet",
            "--no-default-features",
            "--features",
            features,
        ])
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    Err(first_error(&String::from_utf8_lossy(out.stderr.as_slice())))
}

pub(crate) fn tests_feature_matrix() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Feature matrix (does each flag build on top of the minimum set?)"
        ))
    );
    let mut results = Vec::new();

    let Some(cargo) = cargo_cmd() else {
        println!("  {} skipping — cargo not available", yellow("SKIP"));
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some("Skipped — cargo not on PATH".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "feature_matrix".to_string(),
            results,
        };
    };

    let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    let declared = declared_features(&manifest);
    let skip: BTreeSet<&str> = SKIP.iter().map(|(n, _)| *n).collect();
    let min_members: BTreeSet<&str> = MIN.split(',').collect();

    // Guard against the manifest parser silently returning nothing — an empty
    // matrix would pass and prove nothing.
    results.push(run_test("feature-matrix: manifest is readable", || {
        if declared.len() < 20 {
            return Err(format!(
                "only {} features parsed from Cargo.toml — the parser or the manifest \
                 layout changed, and an empty matrix would pass while testing nothing",
                declared.len()
            ));
        }
        Ok(())
    }));

    // The baseline itself.
    results.push(run_test("feature-matrix: the minimum set", || {
        check(cargo, MIN)
    }));

    println!(
        "  {} {} flags declared; checking each on top of the minimum set",
        cyan("·"),
        declared.len()
    );

    for feat in &declared {
        if skip.contains(feat.as_str()) || min_members.contains(feat.as_str()) {
            continue;
        }
        let name = format!("feature-matrix: min + {feat}");
        if !crate::should_run(&name) {
            continue;
        }
        let combo = format!("{MIN},{feat}");
        results.push(run_test(&name, || check(cargo, &combo)));
    }

    let failed = results.iter().filter(|r| !r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} feature_matrix: {}/{} combinations build",
        bold(&cyan("∑")),
        total - failed,
        total
    );
    for (name, why) in SKIP {
        println!("  {} skipped `{}` — {}", cyan("·"), name, why);
    }

    CategoryResult {
        name: "feature_matrix".to_string(),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feature_names_from_a_manifest() {
        let manifest = "\
[package]
name = \"x\"

[features]
# a comment
default = [\"full\"]
security = [\"dep:sha2\"]
rag = [\"rusqlite\", \"aes-gcm\"]
multi-agent = []

[dependencies]
serde = \"1\"
";
        let f = declared_features(manifest);
        assert_eq!(f, vec!["default", "security", "rag", "multi-agent"]);
        // Crucially, it must stop at the next section: `serde` is a dependency,
        // not a feature, and checking `--features serde` would fail confusingly.
        assert!(!f.contains(&"serde".to_string()));
    }

    #[test]
    fn min_set_matches_ci() {
        // Drift here would mean the local battery and CI test different things,
        // and the one that matters would be whichever nobody reads.
        let ci = std::fs::read_to_string(".github/workflows/ci.yml").unwrap_or_default();
        if ci.is_empty() {
            return; // not running from the repo root; nothing to compare
        }
        let line = ci
            .lines()
            .find(|l| l.trim_start().starts_with("FEATURES_MIN:"))
            .unwrap_or("");
        assert!(
            line.contains(MIN),
            "MIN here and FEATURES_MIN in ci.yml have drifted apart.\n  here: {MIN}\n  ci:   {line}"
        );
    }

    /// Every declared feature is either built by CI or excluded on purpose.
    ///
    /// The gap this closes: `whisper-local` is declared in `Cargo.toml`, is not
    /// in CI's matrix, and is excluded from release builds — so **no job has
    /// ever compiled it**, and nothing would have said so. The harness's own
    /// matrix derives from the manifest and does cover it, but that category is
    /// excluded from `--all` because it shells out to cargo per feature.
    ///
    /// This test does not force anything into CI. It forces the *decision* to be
    /// written down, so a feature added tomorrow cannot fall out of CI silently.
    #[test]
    fn every_declared_feature_is_built_by_ci_or_excluded_on_purpose() {
        let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
        let ci = std::fs::read_to_string(".github/workflows/ci.yml").unwrap_or_default();
        if manifest.is_empty() || ci.is_empty() {
            return; // not running from the repo root
        }

        // Three sources, because CI names features in three places: the matrix
        // entries, the `FEATURES_*` env blocks, and inline `--features` on the
        // run lines of the bespoke jobs (tabular, the GUI clippy, the binaries).
        // Reading only the first two is what made this gate report
        // `tabular-polars` as uncovered while a job was testing it.
        let named: Vec<String> = ci_matrix_features(&ci)
            .into_iter()
            .chain(inline_features(&ci))
            .collect();
        assert!(
            named.len() > 20,
            "only {} features parsed out of ci.yml — the workflow layout changed and this \
             test would pass while checking nothing",
            named.len()
        );

        let min_members: BTreeSet<&str> = MIN.split(',').collect();
        let excused: BTreeSet<&str> = NOT_IN_CI.iter().map(|(n, _)| *n).collect();
        // Membership of `full` counts as coverage: CI's check, clippy and test
        // jobs all build FEATURES_STD, which starts from `full`.
        let in_full: BTreeSet<String> = full_members(&manifest).into_iter().collect();
        assert!(
            in_full.len() > 10,
            "only {} members parsed out of `full` — the manifest layout changed and this              test would report covered features as orphans",
            in_full.len()
        );

        // And what cargo enables BECAUSE of those: building `tabular` builds
        // `tabular-sqlite`. Without this the gate flags members of features CI
        // already builds, and both ways of quieting it are wrong --- excusing
        // something that is built, or adding a job that duplicates one.
        let graph = feature_graph(&manifest);
        assert!(
            graph.len() > 50,
            "only {} features parsed out of [features] — the manifest layout changed",
            graph.len()
        );
        let built: BTreeSet<String> = expand(
            named
                .into_iter()
                .chain(in_full.iter().cloned())
                .chain(min_members.iter().map(|s| s.to_string())),
            &graph,
        );

        // Y lo que ademas se PRUEBA, que es un subconjunto: `cargo check` y
        // `cargo clippy` compilan sin correr un test.
        let env: Vec<(&str, String)> = ["FEATURES_STD", "FEATURES_NETWORK", "FEATURES_MIN"]
            .iter()
            .filter_map(|k| {
                ci.lines()
                    .find_map(|l| l.trim().strip_prefix(&format!("{k}:")))
                    .map(|v| (*k, v.trim().trim_matches('"').to_string()))
            })
            .collect();
        let tested: BTreeSet<String> = expand(
            test_job_features(&ci, &env, &ci_matrix_features(&ci)),
            &graph,
        );
        assert!(
            tested.len() > 20 && tested.len() < built.len(),
            "lo probado ({}) deberia ser un subconjunto propio de lo compilado ({}) — \
             si no, el analizador dejo de leer una de las dos cosas",
            tested.len(),
            built.len()
        );

        let orphans: Vec<String> = declared_features(&manifest)
            .into_iter()
            .filter(|f| {
                !built.contains(f)
                    && !in_full.contains(f)
                    && !min_members.contains(f.as_str())
                    && !excused.contains(f.as_str())
                    && !UNCOVERED_BACKLOG.contains(&f.as_str())
            })
            .collect();

        assert!(
            orphans.is_empty(),
            "a NEW feature is declared that nothing in CI compiles: {orphans:?}\n\
             Add them to the matrix in ci.yml, or to NOT_IN_CI with the reason. \
             A feature no job builds is where declared debt accumulates unseen."
        );

        // The same check in the other direction, added V330. The assertion above
        // only catches features nothing covers; it says nothing about a backlog
        // entry that quietly became covered. So the list could keep calling a
        // feature "genuinely uncovered" forever after someone fixed it, and the
        // next reader would budget CI minutes for work already done. A list that
        // is only checked in the direction that grows it is how a ratchet rusts.
        let now_covered: Vec<&str> = UNCOVERED_BACKLOG
            .iter()
            .copied()
            .filter(|f| built.contains(*f))
            .collect();

        assert!(
            now_covered.is_empty(),
            "UNCOVERED_BACKLOG still lists features CI now compiles: {now_covered:?}\n\
             Move them to COMPILED_BUT_NOT_EXERCISED if no test job runs them, or \
             delete them if one does. Leaving them makes the list overstate the gap, \
             which is the same defect as understating it."
        );

        // Y la tercera direccion, que es la que importa de verdad: una feature
        // que SI se compila pero que ningun `cargo test` ejercita. Un fallo de
        // compilacion ahi se caza; una regresion de COMPORTAMIENTO no.
        //
        // Se comprueba que cada entrada siga siendo cierta en las dos mitades:
        // que se compile (si no, su sitio es UNCOVERED_BACKLOG) y que siga sin
        // probarse (si ya se prueba, sobra). Una lista que solo se comprueba en
        // la direccion que la hace crecer es como se oxida un ratchet.
        for (name, reason) in COMPILED_BUT_NOT_EXERCISED {
            assert!(
                declared_features(&manifest).iter().any(|d| d == name),
                "COMPILED_BUT_NOT_EXERCISED nombra {name:?} ({reason}), que ya no se declara"
            );
            assert!(
                built.contains(*name),
                "COMPILED_BUT_NOT_EXERCISED dice que CI compila {name:?} ({reason}), y ya no: \
                 su sitio es UNCOVERED_BACKLOG"
            );
            assert!(
                !tested.contains(*name),
                "COMPILED_BUT_NOT_EXERCISED sigue listando {name:?} ({reason}), pero ya hay un \
                 trabajo de `cargo test` que lo ejercita. Borralo de la lista."
            );
        }

        // Y al reves: nada declarado puede estar compilado-sin-probar sin figurar
        // en la lista, que es como `cloud-connectors` llego a tener 57 tests que
        // no corre nadie sin que constara en ninguna parte.
        let excused_unexercised: BTreeSet<&str> =
            COMPILED_BUT_NOT_EXERCISED.iter().map(|(n, _)| *n).collect();
        let silently_unexercised: Vec<String> = declared_features(&manifest)
            .into_iter()
            .filter(|f| {
                built.contains(f)
                    && !tested.contains(f)
                    && !excused_unexercised.contains(f.as_str())
                    && !excused.contains(f.as_str())
            })
            .collect();
        assert!(
            silently_unexercised.is_empty(),
            "estas features se compilan pero ningun `cargo test` las ejercita, y no constan: \
             {silently_unexercised:?}\n\
             Anadelas a COMPILED_BUT_NOT_EXERCISED con el motivo, o mete la feature en un \
             trabajo de test. Un fallo de comportamiento ahi no lo ve nadie."
        );
    }

    #[test]
    fn the_ci_exclusion_list_does_not_name_features_that_stopped_existing() {
        // The other direction: an excuse for a feature that was renamed or
        // deleted is a stale note that hides the fact nothing is excluded.
        let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
        if manifest.is_empty() {
            return;
        }
        let declared: BTreeSet<String> = declared_features(&manifest).into_iter().collect();
        for (name, reason) in NOT_IN_CI {
            assert!(
                declared.contains(*name),
                "NOT_IN_CI excuses {name:?} ({reason}), but no such feature is declared"
            );
        }
    }

    /// Las dos piezas que faltaban, probadas por separado: sin esto la puerta
    /// vuelve a dar falsos positivos y nadie sabria cual de las dos fallo.
    #[test]
    fn inline_features_reads_the_bespoke_jobs() {
        let yml = "\
      - name: Tabular\n        run: cargo test --features \"tabular,tabular-polars\" --lib\n\
      - name: GUI\n        run: cargo clippy --features gui,gui-pro --bin ai_gui\n\
      - name: Env\n        run: cargo check --features \"$FEATURES_STD\"\n\
      - name: Matrix\n        run: cargo check --features \"${{ matrix.features }}\"\n";
        let got = inline_features(yml);
        assert_eq!(got, vec!["tabular", "tabular-polars", "gui", "gui-pro"]);
        // `$FEATURES_STD` lo expande `ci_matrix_features` desde el bloque env, y
        // `${{ matrix.features }}` desde la matriz. Colarlos aqui como si fueran
        // nombres de feature haria que la puerta diera por cubierto algo que no
        // existe.
    }

    #[test]
    fn the_feature_graph_expands_what_cargo_would_enable() {
        let manifest = "\
[features]\n\
tabular = [\"tabular-sqlite\"]\n\
tabular-sqlite = [\"dep:rusqlite\", \"csv\"]\n\
csv = []\n\
aparte = []\n\
\n[dependencies]\n\
csv = \"1\"\n";
        let graph = feature_graph(manifest);
        assert_eq!(graph.get("tabular").map(Vec::len), Some(1));
        // `dep:rusqlite` NO es una feature de esta crate y no debe contarse.
        assert_eq!(graph.get("tabular-sqlite"), Some(&vec!["csv".to_string()]));

        let built = expand(vec!["tabular".to_string()], &graph);
        assert!(built.contains("tabular-sqlite"), "{built:?}");
        assert!(
            built.contains("csv"),
            "y lo que ese a su vez arrastra: {built:?}"
        );
        assert!(!built.contains("aparte"), "y nada mas: {built:?}");
    }

    #[test]
    fn the_ci_matrix_parser_reads_the_real_list() {
        let yml = "\
jobs:\n  matrix:\n    strategy:\n      matrix:\n        features:\n          - \"security\"\n          - \"autonomous,scheduler\"\n          - \"rag\"\n    steps:\n      - uses: actions/checkout@v5\n";
        let got = ci_matrix_features(yml);
        assert_eq!(got, vec!["security", "autonomous", "scheduler", "rag"]);
    }

    #[test]
    fn extracts_the_error_and_its_location() {
        let stderr = "\
   Compiling ai_assistant v0.2.0
error[E0432]: unresolved import `crate::guardrail_pipeline`
  --> src/prelude.rs:33:16
   |
33 | pub use crate::guardrail_pipeline::{Guard};
";
        let e = first_error(stderr);
        assert!(e.contains("E0432"), "{e}");
        assert!(e.contains("src/prelude.rs:33"), "{e}");
    }

    #[test]
    fn reports_something_useful_when_there_is_no_error_line() {
        assert!(first_error("linker died\n").contains("no recognisable error"));
    }
}
