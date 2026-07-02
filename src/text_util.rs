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

#[cfg(test)]
mod tests {
    use super::*;

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
