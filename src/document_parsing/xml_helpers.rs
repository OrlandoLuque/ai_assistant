//! XML/HTML helper functions for document parsing.

use regex::Regex;

use super::types::DocumentMetadata;

/// Remove all XML/HTML tags from a string, preserving the text content.
/// Also decodes common HTML entities.
pub fn strip_xml_tags(xml: &str) -> String {
    // Remove script and style blocks entirely
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("valid regex");
    let cleaned = script_re.replace_all(xml, "");
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("valid regex");
    let cleaned = style_re.replace_all(&cleaned, "");

    // Replace <br>, <br/>, <br /> with newlines
    let br_re = Regex::new(r"(?i)<br\s*/?\s*>").expect("valid regex");
    let cleaned = br_re.replace_all(&cleaned, "\n");

    // Replace block-level closing tags with newlines for paragraph separation
    let block_re = Regex::new(r"(?i)</?(p|div|li|tr|blockquote|article|section|header|footer|nav|aside|main|figure|figcaption|details|summary|dd|dt)\s*[^>]*>").expect("valid regex");
    let cleaned = block_re.replace_all(&cleaned, "\n");

    // Remove all remaining tags
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    let cleaned = tag_re.replace_all(&cleaned, "");

    // Decode common HTML entities
    let result = cleaned
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&mdash;", "\u{2014}")
        .replace("&ndash;", "\u{2013}")
        .replace("&hellip;", "\u{2026}")
        .replace("&copy;", "\u{00A9}")
        .replace("&reg;", "\u{00AE}");

    // Decode numeric character references: &#NNN; and &#xHHH;
    let num_entity_re = Regex::new(r"&#(\d+);").expect("valid regex");
    let result = num_entity_re.replace_all(&result, |caps: &regex::Captures| {
        if let Ok(code) = caps[1].parse::<u32>() {
            if let Some(ch) = char::from_u32(code) {
                return ch.to_string();
            }
        }
        caps[0].to_string()
    });

    let hex_entity_re = Regex::new(r"(?i)&#x([0-9a-f]+);").expect("valid regex");
    let result = hex_entity_re.replace_all(&result, |caps: &regex::Captures| {
        if let Ok(code) = u32::from_str_radix(&caps[1], 16) {
            if let Some(ch) = char::from_u32(code) {
                return ch.to_string();
            }
        }
        caps[0].to_string()
    });

    result.to_string()
}

/// Extract all text content found between opening and closing instances of the given tag.
/// For example, `extract_xml_text(xml, "title")` finds all `<title>...</title>` occurrences.
pub fn extract_xml_text(xml: &str, tag: &str) -> Vec<String> {
    let escaped_tag = regex::escape(tag);
    let pattern = format!(r"(?s)<{}[^>]*>(.*?)</{}>", escaped_tag, escaped_tag);
    let re = Regex::new(&pattern).expect("valid regex");

    re.captures_iter(xml)
        .map(|caps| {
            let inner = &caps[1];
            let text = strip_xml_tags(inner);
            text.trim().to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract common metadata elements from an XML document (Dublin Core, etc.).
pub fn extract_xml_metadata(xml: &str) -> DocumentMetadata {
    let mut meta = DocumentMetadata::default();

    // Title
    let titles = extract_xml_text(xml, "dc:title");
    if let Some(t) = titles.into_iter().next() {
        meta.title = Some(t);
    }
    // Fallback to <title>
    if meta.title.is_none() {
        let titles = extract_xml_text(xml, "title");
        if let Some(t) = titles.into_iter().next() {
            meta.title = Some(t);
        }
    }

    // Authors
    let creators = extract_xml_text(xml, "dc:creator");
    if !creators.is_empty() {
        meta.authors = creators;
    }

    // Language
    let langs = extract_xml_text(xml, "dc:language");
    if let Some(l) = langs.into_iter().next() {
        meta.language = Some(l);
    }

    // Date
    let dates = extract_xml_text(xml, "dc:date");
    if let Some(d) = dates.into_iter().next() {
        meta.date = Some(d);
    }
    // Fallback to dcterms:modified
    if meta.date.is_none() {
        let modified = extract_xml_text(xml, "dcterms:modified");
        if let Some(m) = modified.into_iter().next() {
            meta.date = Some(m);
        }
    }

    // Description
    let descs = extract_xml_text(xml, "dc:description");
    if let Some(d) = descs.into_iter().next() {
        meta.description = Some(d);
    }

    // Publisher
    let pubs = extract_xml_text(xml, "dc:publisher");
    if let Some(p) = pubs.into_iter().next() {
        meta.publisher = Some(p);
    }

    meta
}

/// Normalize text: collapse multiple whitespace into single spaces within lines,
/// collapse multiple blank lines into at most two newlines, and trim.
pub fn normalize_text(text: &str) -> String {
    // Replace \r\n with \n
    let text = text.replace("\r\n", "\n");

    // Collapse multiple spaces/tabs within lines (but preserve newlines)
    let space_re = Regex::new(r"[^\S\n]+").expect("valid regex");
    let text = space_re.replace_all(&text, " ");

    // Collapse 3+ newlines into 2
    let newline_re = Regex::new(r"\n{3,}").expect("valid regex");
    let text = newline_re.replace_all(&text, "\n\n");

    // Trim each line
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let text = lines.join("\n");

    // Trim the whole result
    text.trim().to_string()
}

/// Extract the first heading (h1-h6) text from an HTML/XHTML document.
#[allow(dead_code)]
pub(crate) fn extract_first_heading(html: &str) -> Option<String> {
    let heading_re = Regex::new(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>").expect("valid regex");
    if let Some(caps) = heading_re.captures(html) {
        let text = strip_xml_tags(&caps[1]);
        let text = text.trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}
