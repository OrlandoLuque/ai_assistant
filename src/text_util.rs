//! Small UTF-8-safe string helpers (V170).
//!
//! The audit found ~90 reachable panics of the form `&s[..n]` /
//! `String::truncate(n)` where `n` is a byte offset that can fall inside a
//! multi-byte UTF-8 character (very common for accented / CJK / emoji text),
//! crashing the process. These helpers truncate on a char boundary instead.

/// Largest byte index `<= max` that is a valid char boundary of `s`.
///
/// Stable-Rust equivalent of the nightly `str::floor_char_boundary`.
pub fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Truncate `s` to at most `max` bytes **without** splitting a UTF-8 char.
/// Never panics. Returns the whole string when `max >= s.len()`.
pub fn truncate_str(s: &str, max: usize) -> &str {
    &s[..floor_char_boundary(s, max)]
}

/// Truncate a `String` in place to at most `max` bytes on a char boundary.
/// Never panics (unlike `String::truncate`, which panics off-boundary).
pub fn truncate_string(s: &mut String, max: usize) {
    let cut = floor_char_boundary(s, max);
    s.truncate(cut);
}

/// Case-insensitive substring search that returns the `[start, end)` byte
/// range **of the original `haystack`** (not of a lowercased copy).
///
/// The audit found many panics of the form
/// `let lo = s.to_lowercase(); let p = lo.find(x); &s[p..]` — `p` is a byte
/// offset into the *lowercased* copy but is used to slice the *original*,
/// and `to_lowercase()` is not length-preserving, so `p` can land inside a
/// multi-byte char and panic. This walks the original directly, so the
/// returned offsets are always valid char boundaries of `haystack`.
pub fn find_ci_range(haystack: &str, needle: &str) -> Option<(usize, usize)> {
    let needle_lower = needle.to_lowercase();
    if needle_lower.is_empty() {
        return Some((0, 0));
    }
    for (start, _) in haystack.char_indices() {
        // Accumulate the lowercased haystack char-by-char from `start` until
        // it reaches (or overshoots) the needle length, then compare.
        let mut acc = String::new();
        for (off, ch) in haystack[start..].char_indices() {
            acc.extend(ch.to_lowercase());
            if acc.len() >= needle_lower.len() {
                if acc == needle_lower {
                    return Some((start, start + off + ch.len_utf8()));
                }
                break;
            }
        }
    }
    None
}

/// Case-insensitive `find` returning the start byte offset in the original
/// `haystack`. See [`find_ci_range`].
pub fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    find_ci_range(haystack, needle).map(|(s, _)| s)
}

/// Case-insensitive `rfind`: the `[start, end)` byte range **of the original
/// `haystack`** of the *last* case-insensitive occurrence of `needle`.
/// Same char-boundary guarantee as [`find_ci_range`].
pub fn rfind_ci_range(haystack: &str, needle: &str) -> Option<(usize, usize)> {
    let mut best = None;
    let mut start = 0;
    while start <= haystack.len() {
        let Some((rel_start, rel_end)) = find_ci_range(&haystack[start..], needle) else {
            break;
        };
        let abs = (start + rel_start, start + rel_end);
        best = Some(abs);
        // Advance one char past this match's start to look for a later one.
        let step = haystack[abs.0..]
            .chars()
            .next()
            .map(|c| c.len_utf8())
            .unwrap_or(1);
        start = abs.0 + step;
    }
    best
}

/// Case-insensitive `rfind` returning the start byte offset in the original
/// `haystack` of the last match. See [`rfind_ci_range`].
pub fn rfind_ci(haystack: &str, needle: &str) -> Option<usize> {
    rfind_ci_range(haystack, needle).map(|(s, _)| s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_ci_handles_multibyte_prefix() {
        // 'ẞ' (3 bytes) lowercases to 'ß' (2 bytes): offsets diverge.
        let hay = "ẞ start INPUT rest";
        let (s, e) = find_ci_range(hay, "input").unwrap();
        assert_eq!(&hay[s..e], "INPUT"); // valid original slice, no panic
        assert_eq!(find_ci("áéí hello", "HELLO"), Some("áéí ".len()));
        assert_eq!(find_ci("abc", "xyz"), None);
        assert_eq!(find_ci("abc", ""), Some(0));
    }

    #[test]
    fn rfind_ci_returns_last_match() {
        // Two matches; rfind must return the later one, with valid offsets.
        let hay = "ẞ tag ONE tag two";
        let (s, e) = rfind_ci_range(hay, "tag").unwrap();
        assert_eq!(&hay[s..e], "tag"); // the second, lowercase "tag"
        assert_eq!(s, hay.rfind("tag").unwrap());
        // Case-insensitive last match across a multibyte prefix.
        assert_eq!(rfind_ci("áéí AB xy AB", "ab"), Some("áéí AB xy ".len()));
        assert_eq!(rfind_ci("abc", "xyz"), None);
    }

    #[test]
    fn no_panic_on_multibyte() {
        let s = "áéíóúñ界🚀"; // all multi-byte
        for n in 0..=s.len() + 3 {
            let t = truncate_str(s, n); // must not panic
            assert!(s.starts_with(t));
        }
    }

    #[test]
    fn floors_to_boundary() {
        let s = "aé"; // 'a'=1 byte, 'é'=2 bytes -> len 3
        assert_eq!(floor_char_boundary(s, 0), 0);
        assert_eq!(floor_char_boundary(s, 1), 1);
        assert_eq!(floor_char_boundary(s, 2), 1); // byte 2 is inside 'é'
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 99), 3);
    }

    #[test]
    fn truncate_string_in_place() {
        let mut s = "aébc".to_string();
        truncate_string(&mut s, 2); // byte 2 splits 'é' -> floors to 1
        assert_eq!(s, "a");
    }
}
