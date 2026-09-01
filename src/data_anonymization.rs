//! Data anonymization
//!
//! Anonymize personal and sensitive data.

use regex::Regex;
use std::collections::HashMap;

/// Anonymization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnonymizationStrategy {
    Redact,
    Hash,
    Pseudonymize,
    Generalize,
    Mask,
}

/// Data type for anonymization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DataType {
    Email,
    Phone,
    Name,
    Address,
    CreditCard,
    Ssn,
    IpAddress,
    Date,
    Custom,
}

/// Anonymization rule
#[derive(Debug, Clone)]
pub struct AnonymizationRule {
    pub data_type: DataType,
    pub strategy: AnonymizationStrategy,
    pub pattern: Option<String>,
    pub replacement: Option<String>,
}

impl AnonymizationRule {
    pub fn new(data_type: DataType, strategy: AnonymizationStrategy) -> Self {
        Self {
            data_type,
            strategy,
            pattern: None,
            replacement: None,
        }
    }

    pub fn with_pattern(mut self, pattern: &str) -> Self {
        self.pattern = Some(pattern.to_string());
        self
    }

    pub fn with_replacement(mut self, replacement: &str) -> Self {
        self.replacement = Some(replacement.to_string());
        self
    }
}

/// Anonymization result
#[derive(Debug, Clone)]
pub struct AnonymizationResult {
    pub original: String,
    pub anonymized: String,
    pub detections: Vec<Detection>,
    pub mapping: HashMap<String, String>,
}

/// Detection of sensitive data
#[derive(Debug, Clone)]
pub struct Detection {
    pub data_type: DataType,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}

/// Data anonymizer
pub struct DataAnonymizer {
    rules: Vec<AnonymizationRule>,
    patterns: HashMap<DataType, Regex>,
    pseudonym_counter: u64,
    pseudonym_mapping: HashMap<String, String>,
}

impl DataAnonymizer {
    pub fn new() -> Self {
        let mut anonymizer = Self {
            rules: Vec::new(),
            patterns: HashMap::new(),
            pseudonym_counter: 0,
            pseudonym_mapping: HashMap::new(),
        };

        // Add default patterns
        anonymizer.add_default_patterns();
        anonymizer
    }

    fn add_default_patterns(&mut self) {
        // Email pattern
        if let Ok(re) = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}") {
            self.patterns.insert(DataType::Email, re);
        }

        // Phone pattern (various formats)
        if let Ok(re) = Regex::new(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}") {
            self.patterns.insert(DataType::Phone, re);
        }

        // Credit card pattern
        if let Ok(re) = Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b") {
            self.patterns.insert(DataType::CreditCard, re);
        }

        // SSN pattern
        if let Ok(re) = Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b") {
            self.patterns.insert(DataType::Ssn, re);
        }

        // IP address pattern
        if let Ok(re) = Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b") {
            self.patterns.insert(DataType::IpAddress, re);
        }

        // Date pattern
        if let Ok(re) = Regex::new(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b") {
            self.patterns.insert(DataType::Date, re);
        }
    }

    pub fn add_rule(&mut self, rule: AnonymizationRule) {
        if let Some(ref pattern) = rule.pattern {
            if let Ok(re) = Regex::new(pattern) {
                self.patterns.insert(rule.data_type, re);
            }
        }
        self.rules.push(rule);
    }

    pub fn anonymize(&mut self, text: &str) -> AnonymizationResult {
        let mut result = text.to_string();
        let mut detections = Vec::new();
        let mut mapping = HashMap::new();

        // Collect pattern matches first
        let matches: Vec<_> = self
            .patterns
            .iter()
            .flat_map(|(data_type, pattern)| {
                let strategy = self.get_strategy(*data_type);
                pattern.find_iter(text).map(move |cap| {
                    (
                        *data_type,
                        strategy,
                        cap.as_str().to_string(),
                        cap.start(),
                        cap.end(),
                    )
                })
            })
            .collect();

        // Now process matches with mutable access
        for (data_type, strategy, value, start, end) in matches {
            let replacement = self.anonymize_value(&value, data_type, strategy);

            detections.push(Detection {
                data_type,
                value: value.clone(),
                start,
                end,
                replacement: replacement.clone(),
            });

            mapping.insert(value, replacement);
        }

        // Sort detections by position (reverse order for replacement)
        detections.sort_by_key(|e| std::cmp::Reverse(e.start));

        // Apply replacements
        for detection in &detections {
            if let Some(replacement) = mapping.get(&detection.value) {
                result = result.replacen(&detection.value, replacement, 1);
            }
        }

        // Re-sort for output
        detections.sort_by_key(|a| a.start);

        AnonymizationResult {
            original: text.to_string(),
            anonymized: result,
            detections,
            mapping,
        }
    }

    fn get_strategy(&self, data_type: DataType) -> AnonymizationStrategy {
        self.rules
            .iter()
            .find(|r| r.data_type == data_type)
            .map(|r| r.strategy)
            .unwrap_or(AnonymizationStrategy::Redact)
    }

    fn anonymize_value(
        &mut self,
        value: &str,
        data_type: DataType,
        strategy: AnonymizationStrategy,
    ) -> String {
        match strategy {
            AnonymizationStrategy::Redact => self.get_redaction(data_type),
            AnonymizationStrategy::Hash => {
                format!("HASH_{:08x}", self.simple_hash(value))
            }
            AnonymizationStrategy::Pseudonymize => self.get_pseudonym(value, data_type),
            AnonymizationStrategy::Generalize => self.generalize_value(value, data_type),
            AnonymizationStrategy::Mask => self.mask_value(value),
        }
    }

    fn get_redaction(&self, data_type: DataType) -> String {
        match data_type {
            DataType::Email => "[EMAIL REDACTED]".to_string(),
            DataType::Phone => "[PHONE REDACTED]".to_string(),
            DataType::Name => "[NAME REDACTED]".to_string(),
            DataType::Address => "[ADDRESS REDACTED]".to_string(),
            DataType::CreditCard => "[CARD REDACTED]".to_string(),
            DataType::Ssn => "[SSN REDACTED]".to_string(),
            DataType::IpAddress => "[IP REDACTED]".to_string(),
            DataType::Date => "[DATE REDACTED]".to_string(),
            DataType::Custom => "[REDACTED]".to_string(),
        }
    }

    fn simple_hash(&self, value: &str) -> u32 {
        let mut hash: u32 = 0;
        for byte in value.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn get_pseudonym(&mut self, value: &str, data_type: DataType) -> String {
        if let Some(existing) = self.pseudonym_mapping.get(value) {
            return existing.clone();
        }

        self.pseudonym_counter += 1;
        let pseudonym = match data_type {
            DataType::Email => format!("user{}@example.com", self.pseudonym_counter),
            DataType::Phone => format!("555-000-{:04}", self.pseudonym_counter % 10000),
            DataType::Name => format!("Person_{}", self.pseudonym_counter),
            DataType::IpAddress => format!("10.0.0.{}", self.pseudonym_counter % 256),
            _ => format!("PSEUDO_{}", self.pseudonym_counter),
        };

        self.pseudonym_mapping
            .insert(value.to_string(), pseudonym.clone());
        pseudonym
    }

    fn generalize_value(&self, value: &str, data_type: DataType) -> String {
        match data_type {
            DataType::Date => {
                // Generalize to year only
                if let Some(year) = value.split(&['/', '-'][..]).last() {
                    if year.len() == 4 {
                        return format!("YEAR_{}", year);
                    }
                }
                "[DATE]".to_string()
            }
            DataType::IpAddress => {
                // Generalize to subnet
                let parts: Vec<_> = value.split('.').collect();
                if parts.len() == 4 {
                    return format!("{}.{}.x.x", parts[0], parts[1]);
                }
                "[IP]".to_string()
            }
            _ => self.mask_value(value),
        }
    }

    fn mask_value(&self, value: &str) -> String {
        let len = value.len();
        if len <= 4 {
            "*".repeat(len)
        } else {
            format!(
                "{}{}{}",
                &value[..2],
                "*".repeat(len - 4),
                &value[len - 2..]
            )
        }
    }

    pub fn deanonymize(&self, text: &str, mapping: &HashMap<String, String>) -> String {
        let mut result = text.to_string();
        for (original, replacement) in mapping {
            result = result.replace(replacement, original);
        }
        result
    }

    pub fn reset_pseudonyms(&mut self) {
        self.pseudonym_mapping.clear();
        self.pseudonym_counter = 0;
    }

    /// Strip metadata segments from JPEG/PNG image bytes.
    ///
    /// JPEG: removes APP1 (EXIF/XMP), APP13 (Photoshop IPTC) and COM segments.
    /// PNG: removes `tEXt`, `zTXt`, `iTXt` and `eXIf` ancillary chunks.
    /// Other formats are passed through unchanged.
    ///
    /// Returns `(scrubbed_bytes, removed_segment_count)`. Pure-Rust, no
    /// external decoder — safe to call on untrusted input because no pixel
    /// decode is performed.
    pub fn scrub_exif(bytes: &[u8]) -> (Vec<u8>, usize) {
        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return scrub_exif_jpeg(bytes);
        }
        if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
            return scrub_exif_png(bytes);
        }
        (bytes.to_vec(), 0)
    }
}

fn scrub_exif_jpeg(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(bytes.len());
    let mut removed = 0usize;

    // Copy SOI (0xFFD8).
    if bytes.len() < 2 {
        return (bytes.to_vec(), 0);
    }
    out.extend_from_slice(&bytes[0..2]);
    let mut i = 2usize;

    while i + 1 < bytes.len() {
        // Find next marker (skip fill bytes, which are 0xFF 0xFF...).
        if bytes[i] != 0xFF {
            // Malformed — bail out, preserve remainder.
            out.extend_from_slice(&bytes[i..]);
            return (out, removed);
        }
        // Skip 0xFF padding.
        let mut j = i;
        while j < bytes.len() && bytes[j] == 0xFF {
            j += 1;
        }
        if j >= bytes.len() {
            out.extend_from_slice(&bytes[i..]);
            return (out, removed);
        }
        let marker = bytes[j];
        let marker_start = j - 1; // position of the leading 0xFF
        i = j + 1;

        // SOS (0xDA): start of scan; copy marker + remainder verbatim
        // (entropy-coded data follows; do not parse it).
        if marker == 0xDA {
            out.extend_from_slice(&bytes[marker_start..]);
            return (out, removed);
        }
        // EOI (0xD9): end of image, no payload.
        if marker == 0xD9 {
            out.extend_from_slice(&bytes[marker_start..j + 1]);
            return (out, removed);
        }
        // Standalone markers (RST0..RST7, TEM): no length, no payload.
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            out.extend_from_slice(&bytes[marker_start..j + 1]);
            continue;
        }
        // All other markers carry a 2-byte big-endian length (includes the
        // length bytes themselves).
        if i + 1 >= bytes.len() {
            out.extend_from_slice(&bytes[marker_start..]);
            return (out, removed);
        }
        let seg_len = ((bytes[i] as usize) << 8) | bytes[i + 1] as usize;
        if seg_len < 2 {
            out.extend_from_slice(&bytes[marker_start..]);
            return (out, removed);
        }
        let seg_end = i + seg_len;
        if seg_end > bytes.len() {
            out.extend_from_slice(&bytes[marker_start..]);
            return (out, removed);
        }
        // Strip APP1 (0xE1), APP13 (0xED), COM (0xFE).
        let strip = matches!(marker, 0xE1 | 0xED | 0xFE);
        if strip {
            removed += 1;
        } else {
            out.extend_from_slice(&bytes[marker_start..seg_end]);
        }
        i = seg_end;
    }
    (out, removed)
}

fn scrub_exif_png(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(bytes.len());
    let mut removed = 0usize;

    if bytes.len() < 8 {
        return (bytes.to_vec(), 0);
    }
    out.extend_from_slice(&bytes[0..8]);
    let mut i = 8usize;

    while i + 12 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let chunk_type = &bytes[i + 4..i + 8];
        let chunk_end = i + 8 + len + 4; // length + type + data + crc
        if chunk_end > bytes.len() {
            // Malformed: copy remainder verbatim.
            out.extend_from_slice(&bytes[i..]);
            return (out, removed);
        }
        let strip = matches!(chunk_type, b"tEXt" | b"zTXt" | b"iTXt" | b"eXIf" | b"tIME");
        if strip {
            removed += 1;
        } else {
            out.extend_from_slice(&bytes[i..chunk_end]);
        }
        // IEND terminates the file.
        if chunk_type == b"IEND" {
            return (out, removed);
        }
        i = chunk_end;
    }
    if i < bytes.len() {
        out.extend_from_slice(&bytes[i..]);
    }
    (out, removed)
}

impl Default for DataAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch anonymizer for multiple texts
pub struct BatchAnonymizer {
    anonymizer: DataAnonymizer,
}

impl BatchAnonymizer {
    pub fn new() -> Self {
        Self {
            anonymizer: DataAnonymizer::new(),
        }
    }

    pub fn anonymize_all(&mut self, texts: &[&str]) -> Vec<AnonymizationResult> {
        texts.iter().map(|t| self.anonymizer.anonymize(t)).collect()
    }
}

impl Default for BatchAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_anonymization() {
        let mut anonymizer = DataAnonymizer::new();
        let result = anonymizer.anonymize("Contact me at john@example.com");

        assert!(!result.anonymized.contains("john@example.com"));
        assert!(result.detections.len() == 1);
    }

    #[test]
    fn test_phone_anonymization() {
        let mut anonymizer = DataAnonymizer::new();
        let result = anonymizer.anonymize("Call 555-123-4567 for info");

        assert!(!result.anonymized.contains("555-123-4567"));
    }

    #[test]
    fn test_pseudonymization() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(
            DataType::Email,
            AnonymizationStrategy::Pseudonymize,
        ));

        let result1 = anonymizer.anonymize("Email: john@test.com");
        let result2 = anonymizer.anonymize("Contact: john@test.com");

        // Same email should get same pseudonym
        assert!(result1.mapping.get("john@test.com") == result2.mapping.get("john@test.com"));
    }

    #[test]
    fn test_hash_strategy() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(
            DataType::Email,
            AnonymizationStrategy::Hash,
        ));
        let r1 = anonymizer.anonymize("Email: a@b.com");
        let r2 = anonymizer.anonymize("Email: a@b.com");
        let v1 = r1.mapping.get("a@b.com").unwrap();
        let v2 = r2.mapping.get("a@b.com").unwrap();
        assert!(v1.starts_with("HASH_"));
        assert_eq!(v1, v2); // deterministic
    }

    #[test]
    fn test_mask_strategy() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(
            DataType::Email,
            AnonymizationStrategy::Mask,
        ));
        let result = anonymizer.anonymize("user@example.com is my email");
        let masked = result.mapping.get("user@example.com").unwrap();
        assert!(masked.starts_with("us"));
        assert!(masked.ends_with("om"));
        assert!(masked.contains('*'));
    }

    #[test]
    fn test_generalize_ip() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(
            DataType::IpAddress,
            AnonymizationStrategy::Generalize,
        ));
        let result = anonymizer.anonymize("IP is 10.20.30.40");
        let gen = result.mapping.get("10.20.30.40").unwrap();
        assert_eq!(gen, "10.20.x.x");
    }

    #[test]
    fn test_deanonymize() {
        let mut anonymizer = DataAnonymizer::new();
        let result = anonymizer.anonymize("Contact john@test.com please");
        let restored = anonymizer.deanonymize(&result.anonymized, &result.mapping);
        assert!(restored.contains("john@test.com"));
    }

    #[test]
    fn test_batch_anonymizer() {
        let mut batch = BatchAnonymizer::new();
        let results = batch.anonymize_all(&["Email me at a@b.com", "Call 555-123-4567"]);
        assert_eq!(results.len(), 2);
        assert!(!results[0].anonymized.contains("a@b.com"));
    }

    #[test]
    fn test_reset_pseudonyms() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(
            DataType::Email,
            AnonymizationStrategy::Pseudonymize,
        ));
        let r1 = anonymizer.anonymize("a@b.com");
        let p1 = r1.mapping.get("a@b.com").unwrap().clone();
        anonymizer.reset_pseudonyms();
        let r2 = anonymizer.anonymize("a@b.com");
        let p2 = r2.mapping.get("a@b.com").unwrap().clone();
        assert_eq!(p1, p2); // same counter restart → same pseudonym
    }

    #[test]
    fn test_multiple_types() {
        let mut anonymizer = DataAnonymizer::new();
        let text = "Email: user@test.com, Phone: 555-123-4567, IP: 192.168.1.1";
        let result = anonymizer.anonymize(text);

        assert!(result.detections.len() >= 3);
        assert!(!result.anonymized.contains("user@test.com"));
        assert!(!result.anonymized.contains("555-123-4567"));
        assert!(!result.anonymized.contains("192.168.1.1"));
    }

    fn jpeg_with_exif() -> Vec<u8> {
        let mut v = Vec::new();
        // SOI
        v.extend_from_slice(&[0xFF, 0xD8]);
        // APP1 EXIF segment: marker 0xFFE1, length=10, payload "EXIFDATA1"
        v.extend_from_slice(&[0xFF, 0xE1]);
        let payload = b"EXIFDATA1";
        let seg_len = (payload.len() + 2) as u16; // length includes the 2 length bytes
        v.extend_from_slice(&seg_len.to_be_bytes());
        v.extend_from_slice(payload);
        // APP0 JFIF (kept): marker 0xFFE0, length=6, payload "JFIF\0"
        v.extend_from_slice(&[0xFF, 0xE0]);
        let jfif = b"JFIF\0";
        let jfif_len = (jfif.len() + 2) as u16;
        v.extend_from_slice(&jfif_len.to_be_bytes());
        v.extend_from_slice(jfif);
        // COM segment (stripped): "COMMENT"
        v.extend_from_slice(&[0xFF, 0xFE]);
        let com = b"COMMENT";
        let com_len = (com.len() + 2) as u16;
        v.extend_from_slice(&com_len.to_be_bytes());
        v.extend_from_slice(com);
        // SOS — copy verbatim until EOI
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02]);
        // Some encoded entropy-coded data
        v.extend_from_slice(&[0x12, 0x34, 0x56]);
        // EOI
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    #[test]
    fn test_scrub_exif_jpeg_strips_app1_and_com() {
        let bytes = jpeg_with_exif();
        let (scrubbed, removed) = DataAnonymizer::scrub_exif(&bytes);
        assert_eq!(removed, 2); // APP1 + COM
        assert!(!scrubbed.windows(7).any(|w| w == b"COMMENT"));
        assert!(!scrubbed.windows(9).any(|w| w == b"EXIFDATA1"));
        // JFIF (APP0) preserved.
        assert!(scrubbed.windows(4).any(|w| w == b"JFIF"));
        // Entropy-coded image data preserved.
        assert!(scrubbed.ends_with(&[0xFF, 0xD9]));
    }

    #[test]
    fn test_scrub_exif_jpeg_no_metadata_pass_through() {
        // SOI + minimal SOS + EOI — no metadata.
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02, 0x99, 0xFF, 0xD9];
        let (scrubbed, removed) = DataAnonymizer::scrub_exif(&bytes);
        assert_eq!(removed, 0);
        assert_eq!(scrubbed, bytes);
    }

    fn png_with_text_chunks() -> Vec<u8> {
        let mut v = Vec::new();
        // PNG signature
        v.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk: length=13, type, 13 bytes data, 4 bytes CRC (zeros — content not validated by scrubber)
        v.extend_from_slice(&13u32.to_be_bytes());
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&[0; 13]);
        v.extend_from_slice(&[0; 4]);
        // tEXt chunk (stripped)
        let text_data = b"Comment\0secret note";
        v.extend_from_slice(&(text_data.len() as u32).to_be_bytes());
        v.extend_from_slice(b"tEXt");
        v.extend_from_slice(text_data);
        v.extend_from_slice(&[0; 4]);
        // eXIf chunk (stripped)
        let exif_data = b"EXIFPAYLOAD";
        v.extend_from_slice(&(exif_data.len() as u32).to_be_bytes());
        v.extend_from_slice(b"eXIf");
        v.extend_from_slice(exif_data);
        v.extend_from_slice(&[0; 4]);
        // IDAT chunk (kept)
        let idat = b"FAKEIDAT";
        v.extend_from_slice(&(idat.len() as u32).to_be_bytes());
        v.extend_from_slice(b"IDAT");
        v.extend_from_slice(idat);
        v.extend_from_slice(&[0; 4]);
        // IEND
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(b"IEND");
        v.extend_from_slice(&[0; 4]);
        v
    }

    #[test]
    fn test_scrub_exif_png_strips_text_and_exif() {
        let bytes = png_with_text_chunks();
        let (scrubbed, removed) = DataAnonymizer::scrub_exif(&bytes);
        assert_eq!(removed, 2);
        assert!(!scrubbed.windows(11).any(|w| w == b"secret note"));
        assert!(!scrubbed.windows(11).any(|w| w == b"EXIFPAYLOAD"));
        assert!(scrubbed.windows(8).any(|w| w == b"FAKEIDAT"));
        assert!(scrubbed.ends_with(b"IEND\0\0\0\0"));
    }

    #[test]
    fn test_scrub_exif_unknown_format_pass_through() {
        let bytes = vec![1, 2, 3, 4, 5];
        let (scrubbed, removed) = DataAnonymizer::scrub_exif(&bytes);
        assert_eq!(removed, 0);
        assert_eq!(scrubbed, bytes);
    }
}
