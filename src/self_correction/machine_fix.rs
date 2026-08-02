//! Applying a toolchain's own proposed fixes — and refusing to trust them.
//!
//! The retry loop itself lives in [`super::engine`]: a [`super::CorrectableTask`]
//! executes, is validated, and is re-executed with feedback. This module covers
//! the step *before* spending a single token on the model: modern toolchains
//! emit machine-readable replacements (rustc's `suggested_replacement`, clippy,
//! `eslint --fix`), and applying those is free.
//!
//! # Why the toolchain's confidence is not the acceptance criterion
//!
//! rustc labels suggestions `MachineApplicable`, meaning "this can be applied
//! automatically" — and **it is sometimes wrong**. Observed in this project's
//! benchmark: rustc proposed adding `+ Ord` to a generic bound, marked it
//! `MachineApplicable`, and the edit compiled while breaking the task's `f64`
//! case. `cargo fix` would have committed it.
//!
//! So the rule this module enforces is: **apply, re-verify, and keep the edit
//! only if verification passes — otherwise restore the original.** The tests
//! decide, never the compiler's self-assessment. [`Suggestion::applicability`]
//! is carried for reporting and deliberately never consulted.

/// A replacement a toolchain proposes for a byte range of a file.
///
/// Modelled on rustc's `--message-format=json` spans, but nothing here is
/// Rust-specific: any tool that can say "replace bytes X..Y of this file with
/// this text" fits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    /// Path as the toolchain reported it.
    pub file: String,
    /// Byte offset of the first replaced byte, in the *original* file.
    pub start: usize,
    /// Byte offset one past the last replaced byte.
    pub end: usize,
    /// Text to splice in.
    pub replacement: String,
    /// The toolchain's own confidence (e.g. rustc's `MachineApplicable`).
    /// Recorded for reporting only — see the module docs on why it is never
    /// used to decide whether to keep the edit.
    pub applicability: String,
}

/// Splice a set of suggestions into `source`.
///
/// Returns `None` when nothing could be applied or the result is unchanged, so
/// the caller can skip a pointless verification round.
///
/// Two details carry the correctness of this function:
///
/// * **Edits are applied back to front**, so each splice leaves the offsets of
///   the ones before it untouched. Applying front to back silently corrupts
///   every subsequent range.
/// * **Ranges that are out of bounds or land mid-character are skipped**, not
///   clamped. A toolchain reporting offsets against a file that has since
///   changed would otherwise panic or produce mojibake.
pub fn apply_suggestions(source: &str, suggestions: &[Suggestion]) -> Option<String> {
    let mut ordered: Vec<&Suggestion> = suggestions
        .iter()
        .filter(|s| s.start <= s.end && s.end <= source.len())
        .collect();
    if ordered.is_empty() {
        return None;
    }
    ordered.sort_by(|a, b| b.start.cmp(&a.start));

    let mut patched = source.to_string();
    let mut applied = 0usize;
    for s in ordered {
        if s.end > patched.len()
            || !patched.is_char_boundary(s.start)
            || !patched.is_char_boundary(s.end)
        {
            continue;
        }
        patched.replace_range(s.start..s.end, &s.replacement);
        applied += 1;
    }
    if applied == 0 || patched == source {
        return None;
    }
    Some(patched)
}

/// Apply the suggestions, then let `verify` decide whether they survive.
///
/// `verify` receives the patched source and returns whether the work still
/// holds up — in practice, whether the tests pass. Returns the accepted source
/// together with a description of what was applied, or `None` when there was
/// nothing to do or the edit failed verification. **`None` means the caller
/// should keep exactly what it had**, which is what makes a wrong
/// `MachineApplicable` suggestion harmless.
pub fn apply_if_verified<F>(
    source: &str,
    suggestions: &[Suggestion],
    mut verify: F,
) -> Option<(String, String)>
where
    F: FnMut(&str) -> bool,
{
    let patched = apply_suggestions(source, suggestions)?;
    if !verify(&patched) {
        return None;
    }
    let what = suggestions
        .iter()
        .map(|s| format!("{} ({})", s.replacement.trim(), s.applicability))
        .collect::<Vec<_>>()
        .join(", ");
    Some((patched, what))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sugg(start: usize, end: usize, replacement: &str) -> Suggestion {
        Suggestion {
            file: "src/lib.rs".to_string(),
            start,
            end,
            replacement: replacement.to_string(),
            applicability: "MachineApplicable".to_string(),
        }
    }

    #[test]
    fn splices_back_to_front() {
        let src = "let a = 1; let b = 2;";
        // Two edits whose ranges would corrupt each other if applied in order.
        let out = apply_suggestions(src, &[sugg(4, 5, "alpha"), sugg(15, 16, "beta")])
            .expect("both edits apply");
        assert_eq!(out, "let alpha = 1; let beta = 2;");
    }

    #[test]
    fn rejects_nothing_to_do() {
        let src = "fn main() {}";
        assert!(apply_suggestions(src, &[]).is_none());
        // Replacement identical to what is already there: not a change.
        assert!(apply_suggestions(src, &[sugg(0, 2, "fn")]).is_none());
        // Entirely out of range.
        assert!(apply_suggestions(src, &[sugg(100, 200, "x")]).is_none());
        // Inverted range.
        assert!(apply_suggestions(src, &[sugg(5, 2, "x")]).is_none());
    }

    #[test]
    fn skips_ranges_that_split_a_character() {
        // 'é' is two bytes; offset 2 lands inside it. Clamping would corrupt the
        // string and slicing would panic, so the edit must simply be skipped.
        let src = "aéb";
        assert!(apply_suggestions(src, &[sugg(2, 3, "X")]).is_none());
        // A boundary-respecting edit on the same string still works.
        assert_eq!(apply_suggestions(src, &[sugg(1, 3, "e")]).unwrap(), "aeb");
    }

    #[test]
    fn a_verified_fix_is_accepted() {
        let src = "let a = 1;";
        let (out, what) =
            apply_if_verified(src, &[sugg(4, 5, "alpha")], |_| true).expect("accepted");
        assert_eq!(out, "let alpha = 1;");
        assert!(what.contains("alpha"), "{what}");
    }

    #[test]
    fn a_fix_that_fails_verification_is_refused() {
        // The heart of this module: the toolchain saying MachineApplicable is
        // not evidence. rustc has proposed edits that compile and are wrong.
        let src = "let a = 1;";
        let mut seen = String::new();
        let out = apply_if_verified(src, &[sugg(4, 5, "alpha")], |patched| {
            seen = patched.to_string();
            false
        });
        assert!(out.is_none(), "a fix that does not verify must be refused");
        assert_eq!(
            seen, "let alpha = 1;",
            "verification must be shown the PATCHED source, not the original"
        );
    }

    #[test]
    fn nothing_to_apply_means_verification_is_never_run() {
        let src = "fn main() {}";
        let out = apply_if_verified(src, &[], |_| panic!("must not verify a no-op"));
        assert!(out.is_none());
    }
}
