// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! SHA-256, in one place.
//!
//! # Why this module exists, and why it is not the `sha2` crate
//!
//! The compression function was written **five times** in `src/`:
//! `compute_proof`, `binary_integrity`, `request_signing`, `cloud_connectors`
//! and `mcp_protocol::v2_oauth`. Same algorithm, three different names
//! (`sha256_bytes`, `sha256`, `sha256_hash`), and **one** of the five had a test
//! against published digests. The other four were a promise.
//!
//! The obvious fix is `sha2`, and it is wrong here. `sha2` is an **optional**
//! dependency (`security = ["dep:sha2"]`, among others), and four of the five
//! callers live in modules with **no feature gate at all** — this crate compiles
//! with zero features, and `cargo check --no-default-features` is part of what
//! must keep working. Reaching for `sha2` from ungated code would mean either
//! making it unconditional (every build, including the portable kit, pays for
//! it) or gating four modules that have no reason to be gated.
//!
//! So: one hand-written implementation, ungated, with the tests that make it
//! trustworthy. Callers that already have `sha2` available keep using it —
//! `config_security`, `secure_backup` and `ai_proxy` do, and they are gated
//! behind features that imply it.
//!
//! # What makes this copy the one that survived
//!
//! [`sha256`] carries the only test in the crate that can tell a correct
//! SHA-256 from a plausible one: digests from FIPS 180-4 and from `hashlib`,
//! including the **padding boundaries** at 55, 56 and 64 bytes. 56 is where the
//! length field no longer fits in the first block, so an off-by-one in the
//! padding loop produces a wrong digest for exactly those lengths and a right
//! one everywhere else — which is why a test on `"abc"` alone proves very
//! little.

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
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Padding: the 0x80 terminator, zeroes up to 56 bytes mod 64, then the
    // length in BITS as a big-endian u64. The `56` is the whole subtlety: the
    // length needs the last eight bytes of a block, so a message of 56..64
    // bytes spills into a second block that is pure padding.
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        result[4 * i..4 * i + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// SHA-256 as lowercase hex, which is how every caller that prints it wants it.
///
/// Three of the five copies had their own version of this two-line helper; a
/// fourth formatted the bytes inline at each call site.
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
        // Moved here with the implementation (N141 / V397). It was the only
        // test in the crate vouching for any of the five copies, and it
        // vouched for exactly one of them.
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
        // into a second block of pure padding.
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
        // &mut self and must be a pure function of its input.
        let a = sha256(b"determinism");
        let b = sha256(b"determinism");
        assert_eq!(a, b);
        assert_ne!(sha256(b"determinism"), sha256(b"determinisn"));
    }
}
