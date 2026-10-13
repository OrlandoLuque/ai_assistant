// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! SHA-256, in one place, over the `sha2` crate.
//!
//! # Why this module exists at all
//!
//! The compression function was written **five times** in `src/`:
//! `compute_proof`, `binary_integrity`, `request_signing`, `cloud_connectors`
//! and `mcp_protocol::v2_oauth`. Same algorithm, three different names
//! (`sha256_bytes`, `sha256`, `sha256_hash`), and **one** of the five had a test
//! against published digests. The other four were a promise. V397 collapsed
//! them onto one hand-written implementation here; V404 replaced that
//! implementation with `sha2`.
//!
//! The module stays rather than having the callers use `sha2` directly, and
//! that is deliberate: the five call sites keep the names and signatures they
//! have always had, the published-vector test below keeps guarding them, and a
//! future change of implementation is one file instead of five.
//!
//! # Why `sha2` now, when V397 argued against it
//!
//! V397 kept a hand-written copy because `sha2` was an **optional** dependency
//! and four of the five callers live in modules with no feature gate, while
//! `cargo check --no-default-features` has to keep working. That reasoning was
//! sound about the constraint and wrong about its weight, and three
//! measurements settled it (2026-10-13, V404):
//!
//! | question | answer |
//! |---|---|
//! | dependencies added by making `sha2` non-optional | **one**: `sha2` itself. All eight of its deps were already in the lock |
//! | compile time added | **0.57 s** (from `cargo build --timings`) |
//! | `cargo check --no-default-features` | **clean** with `sha2` non-optional |
//! | speed, this machine | `sha2` is **7.4x–8.8x** faster: ~1.7 GiB/s against ~200 MiB/s |
//!
//! The 8x is hardware acceleration: `sha2` pulls `cpufeatures` to detect the
//! x86 SHA extensions at run time, which a scalar loop cannot use. And the
//! change removes roughly 150 lines of hand-rolled cryptography from a public
//! repository, which is worth something on its own.
//!
//! # What this module does NOT give you
//!
//! Hashing, not authentication. [`request_signing`](crate::request_signing)
//! builds HMAC-SHA256 on top of this; whether its signature *comparison* is
//! constant-time is a separate question and is tracked as its own task. A hash
//! has no secret-dependent branches, so this function is safe to feed secret
//! data; comparing the result with `==` is not.

use sha2::{Digest, Sha256};

/// SHA-256 of a byte slice.
///
/// ```
/// let digest = ai_assistant::sha256::sha256(b"abc");
/// assert_eq!(
///     ai_assistant::sha256::sha256_hex(b"abc"),
///     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
/// );
/// assert_eq!(digest.len(), 32);
/// ```
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// SHA-256 as lowercase hex, which is how every caller that prints it wants it.
///
/// Three of the five former copies had their own version of this two-line
/// helper; a fourth formatted the bytes inline at each call site. `sha2` does
/// not provide it, so it stays here.
pub fn sha256_hex(data: &[u8]) -> String {
    sha256(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_published_vectors() {
        // Kept across BOTH implementation changes (V397 hand-written, V404
        // `sha2`), and that is the point of it: the test is what makes either
        // implementation trustworthy, so it must not be tied to one of them.
        // It was once the only test in the crate vouching for any of the five
        // copies, and it vouched for exactly one.
        for (input, want) in [
            (
                "".as_bytes(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                "abc".as_bytes(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                // The FIPS 180-4 two-block vector.
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_bytes(),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ] {
            assert_eq!(
                sha256_hex(input),
                want,
                "SHA-256 of {} byte(s) does not match the published digest",
                input.len()
            );
        }

        // The padding boundaries. 56 is the one that matters: the length field
        // needs the last eight bytes of a block, so at 56 the message spills
        // into a second block of pure padding. These caught nothing in `sha2`
        // and were not expected to — they are here so that swapping the
        // implementation again cannot quietly reintroduce an off-by-one.
        for (len, want) in [
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
        ] {
            let input = vec![b'a'; len];
            assert_eq!(
                sha256_hex(&input),
                want,
                "SHA-256 of {len} 'a' bytes does not match the published digest"
            );
        }
    }

    #[test]
    fn hex_is_lowercase_and_sixty_four_characters() {
        // Cheap, and it pins the thing a caller actually compares against. A
        // digest formatted with `{:X}` or without zero-padding compares unequal
        // to every published value while being the same bytes.
        let hex = sha256_hex(b"anything");
        assert_eq!(hex.len(), 64, "{hex}");
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{hex}");
        assert_eq!(hex, hex.to_lowercase(), "{hex}");
        // A zero byte must print as "00" and not as "0".
        let with_zero = sha256_hex(&[0u8; 1]);
        assert_eq!(with_zero.len(), 64, "{with_zero}");
    }

    #[test]
    fn the_same_input_hashes_the_same_way_every_time() {
        // Guards against anything stateful creeping in: the function takes no
        // &mut self and must be a pure function of its input. With `sha2` the
        // risk moved rather than vanished -- a misused `Digest` can carry state
        // between calls if the hasher is reused instead of recreated.
        let a = sha256(b"determinism");
        let b = sha256(b"determinism");
        assert_eq!(a, b);
        assert_ne!(sha256(b"determinism"), sha256(b"determinisn"));
    }

    #[test]
    fn a_long_input_hashes_the_same_as_the_same_input_fed_in_one_go() {
        // New in V404. `sha2` is an incremental API and this module hides that
        // behind a one-shot function; this pins that the hiding is correct for
        // an input far larger than one block, where an implementation that
        // mishandled buffering would diverge.
        let big = vec![0x5au8; 1_000_003]; // deliberately not a multiple of 64
        let once = sha256(&big);
        let mut h = Sha256::new();
        for chunk in big.chunks(7919) {
            h.update(chunk);
        }
        let streamed: [u8; 32] = h.finalize().into();
        assert_eq!(once, streamed, "one-shot and incremental must agree");
    }
}
