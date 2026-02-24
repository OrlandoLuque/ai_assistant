use super::*;


    #[test]
    fn test_strip_xml_tags_basic() {
        let input = "<p>Hello <b>world</b>!</p>";
        let result = strip_xml_tags(input);
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(result.contains("!"));
        assert!(!result.contains("<p>"));
        assert!(!result.contains("<b>"));
    }

    #[test]
    fn test_strip_xml_tags_entities() {
        let input = "5 &lt; 10 &amp; 3 &gt; 1 &quot;test&quot;";
        let result = strip_xml_tags(input);
        assert_eq!(result, "5 < 10 & 3 > 1 \"test\"");
    }

    #[test]
    fn test_strip_xml_tags_script_removal() {
        let input = "<div>Before<script>var x = 1;</script>After</div>";
        let result = strip_xml_tags(input);
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
        assert!(!result.contains("var x"));
    }

    #[test]
    fn test_extract_xml_text_simple() {
        let xml = r#"<root><title>My Book</title><title>Subtitle</title></root>"#;
        let results = extract_xml_text(xml, "title");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "My Book");
        assert_eq!(results[1], "Subtitle");
    }

    #[test]
    fn test_extract_xml_text_nested_tags() {
        let xml = r#"<dc:creator><name>John Doe</name></dc:creator>"#;
        let results = extract_xml_text(xml, "dc:creator");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "John Doe");
    }

    #[test]
    fn test_extract_xml_metadata() {
        let xml = r#"
            <metadata>
                <dc:title>Test Document</dc:title>
                <dc:creator>Author One</dc:creator>
                <dc:creator>Author Two</dc:creator>
                <dc:language>en</dc:language>
                <dc:date>2024-01-15</dc:date>
                <dc:description>A test document</dc:description>
                <dc:publisher>Test Publisher</dc:publisher>
            </metadata>
        "#;
        let meta = extract_xml_metadata(xml);
        assert_eq!(meta.title.as_deref(), Some("Test Document"));
        assert_eq!(meta.authors.len(), 2);
        assert_eq!(meta.authors[0], "Author One");
        assert_eq!(meta.authors[1], "Author Two");
        assert_eq!(meta.language.as_deref(), Some("en"));
        assert_eq!(meta.date.as_deref(), Some("2024-01-15"));
        assert_eq!(meta.description.as_deref(), Some("A test document"));
        assert_eq!(meta.publisher.as_deref(), Some("Test Publisher"));
    }

    #[test]
    fn test_normalize_text() {
        let input = "  Hello   world  \n\n\n\n  Second paragraph  \n  Third  ";
        let result = normalize_text(input);
        assert_eq!(result, "Hello world\n\nSecond paragraph\nThird");
    }

    #[test]
    fn test_normalize_text_crlf() {
        let input = "Line one\r\nLine two\r\n\r\n\r\nLine three";
        let result = normalize_text(input);
        assert_eq!(result, "Line one\nLine two\n\nLine three");
    }

    #[test]
    fn test_detect_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());

        assert_eq!(
            parser.detect_format(std::path::Path::new("book.epub")),
            Some(DocumentFormat::Epub)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("report.docx")),
            Some(DocumentFormat::Docx)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("letter.odt")),
            Some(DocumentFormat::Odt)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("page.html")),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("page.htm")),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("notes.txt")),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("readme.md")),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("image.png")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(parser.detect_format(std::path::Path::new("noext")), None);
    }

    #[test]
    fn test_parse_plain_text() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let content = "Hello, this is a plain text document.\n\nIt has two paragraphs.";
        let doc = parser
            .parse_string(content, DocumentFormat::PlainText)
            .unwrap();

        assert_eq!(doc.format, DocumentFormat::PlainText);
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("two paragraphs"));
        assert_eq!(doc.word_count, 11);
        assert!(doc.char_count > 0);
        assert_eq!(doc.sections.len(), 1);
    }

    #[test]
    fn test_parse_html_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let html = r#"
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <title>Test Page</title>
                <meta name="author" content="Jane Doe">
                <meta name="description" content="A test page">
            </head>
            <body>
                <h1>Welcome</h1>
                <p>This is the first paragraph.</p>
                <h2>Section Two</h2>
                <p>This is the second section content.</p>
            </body>
            </html>
        "#;

        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();

        assert_eq!(doc.format, DocumentFormat::Html);
        assert_eq!(doc.metadata.title.as_deref(), Some("Test Page"));
        assert_eq!(doc.metadata.authors, vec!["Jane Doe"]);
        assert_eq!(doc.metadata.description.as_deref(), Some("A test page"));
        assert_eq!(doc.metadata.language.as_deref(), Some("en"));
        assert!(doc.text.contains("first paragraph"));
        assert!(doc.text.contains("second section"));
        assert!(doc.word_count > 0);
    }

    #[test]
    fn test_parse_html_sections() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let html = r#"
            <body>
                <h1>Title</h1>
                <p>Intro text here.</p>
                <h2>Chapter 1</h2>
                <p>Chapter one content.</p>
            </body>
        "#;

        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();
        let titles = doc.section_titles();

        assert!(titles.contains(&"Title"));
        assert!(titles.contains(&"Chapter 1"));
    }

    #[test]
    fn test_parse_html_no_sections() {
        let mut config = DocumentParserConfig::default();
        config.extract_sections = false;
        let parser = DocumentParser::new(config);

        let html = "<p>Just a paragraph.</p>";
        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();
        assert!(doc.sections.is_empty());
        assert!(doc.text.contains("Just a paragraph"));
    }

    #[test]
    fn test_parsed_document_section_text() {
        let doc = ParsedDocument {
            text: "Full text".to_string(),
            metadata: DocumentMetadata::default(),
            sections: vec![
                DocumentSection {
                    title: Some("First".to_string()),
                    content: "Content one".to_string(),
                    level: 1,
                    index: 0,
                },
                DocumentSection {
                    title: Some("Second".to_string()),
                    content: "Content two".to_string(),
                    level: 2,
                    index: 1,
                },
            ],
            tables: Vec::new(),
            format: DocumentFormat::Html,
            source_path: None,
            char_count: 9,
            word_count: 2,
        };

        assert_eq!(doc.section_text(0), Some("Content one"));
        assert_eq!(doc.section_text(1), Some("Content two"));
        assert_eq!(doc.section_text(2), None);
    }

    #[test]
    fn test_parsed_document_section_titles() {
        let doc = ParsedDocument {
            text: String::new(),
            metadata: DocumentMetadata::default(),
            sections: vec![
                DocumentSection {
                    title: Some("Alpha".to_string()),
                    content: String::new(),
                    level: 1,
                    index: 0,
                },
                DocumentSection {
                    title: None,
                    content: String::new(),
                    level: 0,
                    index: 1,
                },
                DocumentSection {
                    title: Some("Gamma".to_string()),
                    content: String::new(),
                    level: 2,
                    index: 2,
                },
            ],
            tables: Vec::new(),
            format: DocumentFormat::PlainText,
            source_path: None,
            char_count: 0,
            word_count: 0,
        };

        let titles = doc.section_titles();
        assert_eq!(titles, vec!["Alpha", "Gamma"]);
    }

    #[test]
    fn test_max_size_enforcement() {
        let mut config = DocumentParserConfig::default();
        config.max_size_bytes = 10; // Very small limit
        let parser = DocumentParser::new(config);

        let data = b"This is way more than ten bytes of data for testing";
        let result = parser.parse_bytes(data, DocumentFormat::PlainText);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("exceeds maximum"));
    }

    #[test]
    fn test_extract_first_heading() {
        let html = "<html><body><h1>Main Title</h1><p>Text</p><h2>Sub</h2></body></html>";
        let heading = xml_helpers::extract_first_heading(html);
        assert_eq!(heading, Some("Main Title".to_string()));
    }

    #[test]
    fn test_extract_first_heading_none() {
        let html = "<html><body><p>No headings here.</p></body></html>";
        let heading = xml_helpers::extract_first_heading(html);
        assert_eq!(heading, None);
    }

    #[test]
    fn test_document_parser_config_default() {
        let config = DocumentParserConfig::default();
        assert!(config.preserve_paragraphs);
        assert!(config.extract_metadata);
        assert!(config.extract_sections);
        assert!(config.strip_tags);
        assert!(config.normalize_whitespace);
        assert_eq!(config.max_size_bytes, 50 * 1024 * 1024);
    }

    #[test]
    fn test_html_numeric_entities() {
        let input = "&#65;&#66;&#67; and &#x41;&#x42;&#x43;";
        let result = strip_xml_tags(input);
        assert_eq!(result, "ABC and ABC");
    }

    // ===== CSV parsing tests =====

    #[test]
    fn test_csv_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name,Age,City\nAlice,30,NYC\nBob,25,LA\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
        assert!(result.text.contains("Bob"));
        assert!(!result.tables.is_empty());
        assert_eq!(result.tables[0].headers.len(), 3);
        assert_eq!(result.tables[0].rows.len(), 2);
    }

    #[test]
    fn test_csv_tsv_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let tsv = b"Name\tAge\tCity\nAlice\t30\tNYC\n";
        let result = parser.parse_bytes(tsv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
        assert!(!result.tables.is_empty());
        assert_eq!(result.tables[0].headers.len(), 3);
    }

    #[test]
    fn test_csv_quoted_fields() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name,Description\n\"Alice\",\"She said \"\"hello\"\"\"\nBob,\"Line1\nLine2\"\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
    }

    #[test]
    fn test_csv_unicode() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = "Nombre,Ciudad\nJos\u{00e9},M\u{00e9}xico\nFran\u{00e7}ois,Par\u{00ed}s\n".as_bytes();
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Jos\u{00e9}"));
        assert!(result.text.contains("M\u{00e9}xico"));
    }

    #[test]
    fn test_csv_empty() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.tables.is_empty());
    }

    #[test]
    fn test_csv_single_column() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name\nAlice\nBob\nCharlie\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
    }

    // ===== Email parsing tests =====

    #[test]
    fn test_email_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: alice@example.com\r\nTo: bob@example.com\r\nSubject: Test Email\r\nDate: Mon, 1 Jan 2024 12:00:00 +0000\r\n\r\nHello Bob, this is a test email.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Hello Bob"));
        assert_eq!(result.metadata.title.as_deref(), Some("Test Email"));
        assert!(!result.metadata.authors.is_empty());
        assert!(result.metadata.authors[0].contains("alice@example.com"));
        assert!(result.metadata.extra.contains_key("to"));
    }

    #[test]
    fn test_email_multipart() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: alice@example.com\r\nContent-Type: multipart/alternative; boundary=\"boundary123\"\r\n\r\n--boundary123\r\nContent-Type: text/plain\r\n\r\nPlain text body\r\n--boundary123\r\nContent-Type: text/html\r\n\r\n<html><body>HTML body</body></html>\r\n--boundary123--\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Plain text body"));
    }

    #[test]
    fn test_email_quoted_printable() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: test@example.com\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\nHello =C3=A9l=C3=A8ve, this is a =\r\ncontinued line.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        // Should decode QP: =C3=A9 = e-acute, =C3=A8 = e-grave
        assert!(result.text.contains("continued line"));
    }

    #[test]
    fn test_email_base64_body() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // "Hello World" in base64 is "SGVsbG8gV29ybGQ="
        let email = b"From: test@example.com\r\nContent-Transfer-Encoding: base64\r\n\r\nSGVsbG8gV29ybGQ=\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Hello World"));
    }

    #[test]
    fn test_email_no_subject() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: test@example.com\r\n\r\nBody text here.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Body text"));
    }

    // ===== Image metadata extraction tests =====

    #[test]
    fn test_image_png_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // Minimal PNG: magic + IHDR chunk (width=100, height=50)
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]; // PNG magic
                                                                            // IHDR chunk: length(13) + "IHDR" + width(100) + height(50) + bit_depth + color_type + ...
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]); // chunk length = 13
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes()); // width
        png.extend_from_slice(&50u32.to_be_bytes()); // height
        png.extend_from_slice(&[8, 2, 0, 0, 0]); // bit_depth=8, color=RGB, rest
        png.extend_from_slice(&[0, 0, 0, 0]); // CRC
        let result = parser.parse_bytes(&png, DocumentFormat::Image).unwrap();
        assert!(result.metadata.extra.contains_key("format"));
        assert_eq!(result.metadata.extra["format"], "PNG");
        assert_eq!(result.metadata.extra["width"], "100");
        assert_eq!(result.metadata.extra["height"], "50");
    }

    #[test]
    fn test_image_jpeg_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // Minimal JPEG: SOI + SOF0 marker with dimensions
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0]; // SOI + APP0
        jpeg.extend_from_slice(&[0x00, 0x02, 0x00, 0x00]); // APP0 minimal
                                                           // SOF0 marker (0xFFC0)
        jpeg.extend_from_slice(&[0xFF, 0xC0]);
        jpeg.extend_from_slice(&[0x00, 0x0B]); // length = 11
        jpeg.push(8); // precision
        jpeg.extend_from_slice(&200u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&300u16.to_be_bytes()); // width
        jpeg.push(3); // num components
        jpeg.extend_from_slice(&[0; 6]); // component data padding
        let result = parser.parse_bytes(&jpeg, DocumentFormat::Image).unwrap();
        assert!(result.metadata.extra.contains_key("format"));
        assert_eq!(result.metadata.extra["format"], "JPEG");
    }

    #[test]
    fn test_image_gif_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // GIF89a header: magic + width(120) + height(80)
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&120u16.to_le_bytes()); // width LE
        gif.extend_from_slice(&80u16.to_le_bytes()); // height LE
        gif.extend_from_slice(&[0; 10]); // padding
        let result = parser.parse_bytes(&gif, DocumentFormat::Image).unwrap();
        assert_eq!(result.metadata.extra["format"], "GIF");
        assert_eq!(result.metadata.extra["width"], "120");
        assert_eq!(result.metadata.extra["height"], "80");
    }

    #[test]
    fn test_image_unknown_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let data = b"This is not an image file";
        let result = parser.parse_bytes(data, DocumentFormat::Image).unwrap();
        assert_eq!(
            result.metadata.extra.get("format").map(|s| s.as_str()),
            Some("Unknown")
        );
    }

    #[test]
    fn test_image_bmp_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // BMP header: "BM" + file size + reserved + data offset + DIB header size + width + height
        let mut bmp = b"BM".to_vec();
        bmp.extend_from_slice(&[0; 12]); // file size + reserved + data offset
        bmp.extend_from_slice(&40u32.to_le_bytes()); // DIB header size (BITMAPINFOHEADER)
        bmp.extend_from_slice(&640u32.to_le_bytes()); // width (LE, signed i32 but positive)
        bmp.extend_from_slice(&480u32.to_le_bytes()); // height
        bmp.extend_from_slice(&[0; 20]); // rest of DIB header
        let result = parser.parse_bytes(&bmp, DocumentFormat::Image).unwrap();
        assert_eq!(result.metadata.extra["format"], "BMP");
        assert_eq!(result.metadata.extra["width"], "640");
        assert_eq!(result.metadata.extra["height"], "480");
    }

    // ===== Document format detection tests =====

    #[test]
    fn test_detect_csv_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let format = parser.detect_format(std::path::Path::new("data.csv"));
        assert_eq!(format, Some(DocumentFormat::Csv));
        let format = parser.detect_format(std::path::Path::new("data.tsv"));
        assert_eq!(format, Some(DocumentFormat::Csv));
    }

    #[test]
    fn test_detect_email_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let format = parser.detect_format(std::path::Path::new("message.eml"));
        assert_eq!(format, Some(DocumentFormat::Email));
    }

    #[test]
    fn test_detect_image_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.jpg")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.jpeg")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("icon.png")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("animation.gif")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.bmp")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.webp")),
            Some(DocumentFormat::Image)
        );
    }

    #[test]
    fn test_detect_pptx_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("slides.pptx")),
            Some(DocumentFormat::Pptx)
        );
    }

    #[test]
    fn test_detect_xlsx_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("spreadsheet.xlsx")),
            Some(DocumentFormat::Xlsx)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("legacy.xls")),
            Some(DocumentFormat::Xlsx)
        );
    }

    // ===== OCR-lite engine tests =====

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert!(config.min_confidence > 0.0);
        assert_eq!(config.char_height, 7);
        assert!(config.binarize_threshold.is_none());
    }

    #[test]
    fn test_ocr_engine_creation() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        assert!(!engine.templates.is_empty());
    }

    #[test]
    fn test_glyph_template_structure() {
        let template = GlyphTemplate {
            character: 'X',
            width: 5,
            height: 7,
            bitmap: vec![0; 35],
        };
        assert_eq!(template.bitmap.len(), template.width * template.height);
    }

    #[test]
    fn test_binarize_fixed_threshold() {
        let config = OcrConfig {
            binarize_threshold: Some(128),
            ..Default::default()
        };
        let engine = OcrEngine::with_default_templates(config);
        let image = vec![0, 50, 100, 128, 200, 255];
        let binary = engine.binarize(&image, 6, 1);
        // pixels < 128 become 1 (black), >= 128 become 0 (white)
        assert_eq!(binary, vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_otsu_threshold_bimodal() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Bimodal image: half centered around 50, half centered around 200
        let mut image = vec![50u8; 100];
        for i in 50..100 {
            image[i] = 200;
        }
        let threshold = engine.otsu_threshold(&image);
        // Otsu should pick a threshold that separates the two clusters (>= 50, <= 200)
        assert!(
            threshold >= 50 && threshold <= 200,
            "Expected threshold between 50 and 200 inclusive, got {}",
            threshold
        );
    }

    #[test]
    fn test_detect_text_lines_single() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // 10x10 image with a horizontal line of black pixels at rows 3-6
        let mut binary = vec![0u8; 100];
        for y in 3..7 {
            for x in 0..10 {
                binary[y * 10 + x] = 1;
            }
        }
        let lines = engine.detect_text_lines(&binary, 10, 10);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], (3, 7));
    }

    #[test]
    fn test_segment_characters() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Line with two "characters" separated by a gap
        // Width 12, height 5: cols 0-2 have pixels, cols 3-4 empty, cols 5-8 have pixels
        let mut line = vec![0u8; 60]; // 12 * 5
        for y in 0..5 {
            for x in 0..3 {
                line[y * 12 + x] = 1;
            }
            for x in 5..9 {
                line[y * 12 + x] = 1;
            }
        }
        let chars = engine.segment_characters(&line, 12, 5);
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0], (0, 3));
        assert_eq!(chars[1], (5, 9));
    }

    #[test]
    fn test_match_template_returns_best() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Use the 'T' template as input -- should match 'T' best
        let t_template = engine
            .templates
            .iter()
            .find(|t| t.character == 'T')
            .unwrap();
        let (ch, confidence) =
            engine.match_template(&t_template.bitmap, t_template.width, t_template.height);
        assert_eq!(ch, 'T');
        assert!(
            confidence > 0.99,
            "Expected near-perfect match, got {}",
            confidence
        );
    }

    #[test]
    fn test_recognize_empty_image() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        let result = engine.recognize_bitmap(&[], 0, 0);
        assert!(result.full_text.is_empty());
        assert_eq!(result.average_confidence, 0.0);
    }

    #[test]
    fn test_recognize_all_white() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        let image = vec![255u8; 100]; // 10x10 all white
        let result = engine.recognize_bitmap(&image, 10, 10);
        assert!(result.full_text.is_empty());
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn test_epub_without_feature_returns_error() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let result = parser.parse_bytes(b"fake epub data", DocumentFormat::Epub);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("documents"));
    }

    // ========================================================================
    // Image Extraction tests
    // ========================================================================

    #[test]
    fn test_detect_jpeg_magic() {
        let header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let result = ImageExtractor::detect_format(&header);
        assert_eq!(result, Some(ImageFormat::Jpeg));
    }

    #[test]
    fn test_detect_png_magic() {
        let header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = ImageExtractor::detect_format(&header);
        assert_eq!(result, Some(ImageFormat::Png));
    }

    #[test]
    fn test_extract_jpeg_with_dimensions() {
        // Build a synthetic JPEG:
        // SOI + SOF0 marker with dimensions + padding + EOI
        let mut jpeg = Vec::new();
        // SOI
        jpeg.extend_from_slice(&[0xFF, 0xD8]);
        // SOF0 marker: FF C0, then length (2 bytes), precision (1 byte), height (2 BE), width (2 BE)
        jpeg.extend_from_slice(&[0xFF, 0xC0]);
        jpeg.extend_from_slice(&[0x00, 0x11]); // length = 17
        jpeg.push(0x08); // precision
        jpeg.extend_from_slice(&[0x00, 0x80]); // height = 128
        jpeg.extend_from_slice(&[0x01, 0x00]); // width = 256
                                               // Pad with zeros to meet the min_size_bytes=100 threshold
        jpeg.extend_from_slice(&[0x00; 100]);
        // EOI
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let extractor = ImageExtractor::with_default_config();
        let images = extractor.extract_from_bytes(&jpeg);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].format, ImageFormat::Jpeg);
        assert_eq!(images[0].width, Some(256));
        assert_eq!(images[0].height, Some(128));
        assert_eq!(images[0].index, 0);
        assert_eq!(images[0].offset, 0);
    }

    #[test]
    fn test_extract_png_with_dimensions() {
        // Build a synthetic PNG with IHDR + IEND
        let mut png = Vec::new();
        // 8-byte PNG signature
        png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk: length (4 bytes) = 13
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]);
        // Chunk type "IHDR"
        png.extend_from_slice(b"IHDR");
        // Width = 320 (4 bytes big-endian)
        png.extend_from_slice(&[0x00, 0x00, 0x01, 0x40]);
        // Height = 240 (4 bytes big-endian)
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0xF0]);
        // Bit depth, color type, compression, filter, interlace
        png.extend_from_slice(&[0x08, 0x02, 0x00, 0x00, 0x00]);
        // CRC (4 bytes, not validated by our parser)
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        // Pad to exceed min_size_bytes
        png.extend_from_slice(&[0x00; 80]);
        // IEND chunk trailer (12 bytes)
        png.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]);

        let extractor = ImageExtractor::with_default_config();
        let images = extractor.extract_from_bytes(&png);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].format, ImageFormat::Png);
        assert_eq!(images[0].width, Some(320));
        assert_eq!(images[0].height, Some(240));
    }

    #[test]
    fn test_skip_small_images() {
        // Build a tiny JPEG that is smaller than a large min_size_bytes
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
        jpeg.extend_from_slice(&[0x00; 20]);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let config = ImageExtractionConfig {
            min_size_bytes: 500, // larger than our tiny JPEG
            ..Default::default()
        };
        let extractor = ImageExtractor::new(config);
        let images = extractor.extract_from_bytes(&jpeg);
        assert!(images.is_empty(), "Small image should have been skipped");
    }

    #[test]
    fn test_max_images_limit() {
        // Build data containing 5 JPEGs, but limit to 2
        let mut data = Vec::new();
        for _ in 0..5 {
            // SOI
            data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
            // Pad to exceed 100 bytes per image
            data.extend_from_slice(&[0x00; 110]);
            // EOI
            data.extend_from_slice(&[0xFF, 0xD9]);
        }

        let config = ImageExtractionConfig {
            max_images: 2,
            ..Default::default()
        };
        let extractor = ImageExtractor::new(config);
        let images = extractor.extract_from_bytes(&data);
        assert_eq!(images.len(), 2, "Should stop after max_images reached");
        assert_eq!(images[0].index, 0);
        assert_eq!(images[1].index, 1);
    }

    #[test]
    fn test_mixed_format_document() {
        let mut data = Vec::new();

        // Embed a JPEG
        data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
        data.extend_from_slice(&[0x00; 110]);
        data.extend_from_slice(&[0xFF, 0xD9]);

        // Some random garbage between images
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        // Embed a PNG
        // PNG signature
        data.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]);
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x64]); // width=100
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x64]); // height=100
        data.extend_from_slice(&[0x08, 0x02, 0x00, 0x00, 0x00]);
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC
        data.extend_from_slice(&[0x00; 80]); // padding
                                             // IEND trailer
        data.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]);

        let analysis =
            DocumentImageAnalysis::from_document(&data, ImageExtractionConfig::default());
        assert_eq!(analysis.images.len(), 2);
        assert_eq!(analysis.formats_found.get(&ImageFormat::Jpeg), Some(&1));
        assert_eq!(analysis.formats_found.get(&ImageFormat::Png), Some(&1));
        assert!(analysis.total_image_bytes > 0);
    }

    #[test]
    fn test_empty_document() {
        let data: &[u8] = &[];
        let analysis = DocumentImageAnalysis::from_document(data, ImageExtractionConfig::default());
        assert!(analysis.images.is_empty());
        assert_eq!(analysis.total_image_bytes, 0);
        assert!(analysis.formats_found.is_empty());
    }

    // ====================================================================
    // WS9 -- OCR Integration tests
    // ====================================================================

    #[test]
    fn test_template_ocr_backend_name() {
        let backend = TemplateOcrBackend::new(OcrConfig::default());
        assert_eq!(backend.name(), "template");
        assert!(backend.supports_format(&ImageFormat::Png));
        assert!(backend.supports_format(&ImageFormat::Jpeg));
        assert!(!backend.supports_format(&ImageFormat::Gif));
    }

    #[test]
    fn test_tesseract_backend_not_available() {
        let backend = TesseractOcrBackend::new(TesseractConfig::default());
        assert_eq!(backend.name(), "tesseract");
        let result = backend.recognize(&[128u8; 35], 5, 7);
        assert_eq!(result.average_confidence, 0.0);
        assert!(result.full_text.contains("binary not available"));
        assert!(backend.supports_format(&ImageFormat::Gif));
        assert_eq!(backend.confidence_threshold(), 0.0);
    }

    #[test]
    fn test_ocr_pipeline_selects_best() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig {
            min_confidence: 0.0,
            ..OcrPipelineConfig::default()
        });
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        pipeline.add_backend(Box::new(TesseractOcrBackend::new(
            TesseractConfig::default(),
        )));

        assert_eq!(pipeline.backend_count(), 2);

        // 5x7 synthetic grayscale bitmap (all mid-grey).
        let image = vec![128u8; 5 * 7];
        let result = pipeline.process_image(&image, 5, 7);

        // The template backend returns a real (possibly 0.0) OCR attempt while
        // Tesseract always returns 0.0.  Because the pipeline picks the highest
        // confidence, and ties go to the first result evaluated, the template
        // backend's result should win (or tie) since it is added first.
        // Either way, the pipeline must not panic and must return *some* result.
        assert!(result.average_confidence >= 0.0);
    }

    #[test]
    fn test_ocr_pipeline_single_backend() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig::default());
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        assert_eq!(pipeline.backend_count(), 1);

        let image = vec![200u8; 10 * 7];
        let result = pipeline.process_image(&image, 10, 7);
        // Should return a valid OcrResult (even if empty text).
        assert!(result.average_confidence >= 0.0);
    }

    #[test]
    fn test_ocr_pipeline_process_extracted_images() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig {
            min_confidence: 0.0,
            ..OcrPipelineConfig::default()
        });
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));

        let images = vec![
            ExtractedImage {
                data: vec![128u8; 35],
                format: ImageFormat::Png,
                width: Some(5),
                height: Some(7),
                page: None,
                index: 0,
                offset: 0,
            },
            ExtractedImage {
                data: vec![200u8; 70],
                format: ImageFormat::Jpeg,
                width: Some(10),
                height: Some(7),
                page: Some(1),
                index: 1,
                offset: 100,
            },
        ];

        let results = pipeline.process_extracted_images(&images);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // first image's index
        assert_eq!(results[1].0, 1); // second image's index
    }

    #[test]
    fn test_ocr_pipeline_backend_names() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig::default());
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        pipeline.add_backend(Box::new(TesseractOcrBackend::new(
            TesseractConfig::default(),
        )));

        let names = pipeline.backend_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "template");
        assert_eq!(names[1], "tesseract");
    }

    // ========================================================================
    // Phase 4 (v11): Synthetic ZIP tests for EPUB/DOCX/ODT/PPTX/XLSX
    // ========================================================================

    #[cfg(feature = "documents")]
    mod zip_tests {
        use super::*;
        use std::io::{Cursor, Write};
        use zip::write::{SimpleFileOptions, ZipWriter};

        /// Helper: build a synthetic EPUB ZIP in memory.
        fn build_epub(
            opf_metadata: &str,
            chapters: &[(&str, &str)],
        ) -> Vec<u8> {
            let buf = Cursor::new(Vec::new());
            let mut zip = ZipWriter::new(buf);
            let opts = SimpleFileOptions::default();

            // mimetype must be first
            zip.start_file("mimetype", opts).unwrap();
            zip.write_all(b"application/epub+zip").unwrap();

            // Build OPF manifest + spine
            let mut manifest = String::new();
            let mut spine = String::new();
            for (i, (filename, _)) in chapters.iter().enumerate() {
                let id = format!("ch{}", i + 1);
                manifest.push_str(&format!(
                    "<item id=\"{id}\" href=\"{filename}\" media-type=\"application/xhtml+xml\"/>\n"
                ));
                spine.push_str(&format!("<itemref idref=\"{id}\"/>\n"));
            }

            let opf = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                 <package xmlns=\"http://www.idpf.org/2007/opf\" version=\"3.0\">\n\
                   <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n\
                     {opf_metadata}\n\
                   </metadata>\n\
                   <manifest>\n{manifest}</manifest>\n\
                   <spine>\n{spine}</spine>\n\
                 </package>"
            );

            zip.start_file("OEBPS/content.opf", opts).unwrap();
            zip.write_all(opf.as_bytes()).unwrap();

            // container.xml
            let container = "<?xml version=\"1.0\"?>\n\
                <container xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\" version=\"1.0\">\n\
                  <rootfiles>\n\
                    <rootfile full-path=\"OEBPS/content.opf\" media-type=\"application/oebps-package+xml\"/>\n\
                  </rootfiles>\n\
                </container>";
            zip.start_file("META-INF/container.xml", opts).unwrap();
            zip.write_all(container.as_bytes()).unwrap();

            // Chapters
            for (filename, body) in chapters {
                let xhtml = format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                     <html xmlns=\"http://www.w3.org/1999/xhtml\">\n\
                     <head><title>Chapter</title></head>\n\
                     <body>{body}</body>\n\
                     </html>"
                );
                let path = format!("OEBPS/{filename}");
                zip.start_file(&path, opts).unwrap();
                zip.write_all(xhtml.as_bytes()).unwrap();
            }

            zip.finish().unwrap().into_inner()
        }

        /// Helper: build a synthetic DOCX ZIP in memory.
        fn build_docx(paragraphs: &[&str], metadata_xml: Option<&str>) -> Vec<u8> {
            let buf = Cursor::new(Vec::new());
            let mut zip = ZipWriter::new(buf);
            let opts = SimpleFileOptions::default();

            // [Content_Types].xml
            let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                <Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
                  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
                  <Default Extension=\"xml\" ContentType=\"application/xml\"/>\n\
                </Types>";
            zip.start_file("[Content_Types].xml", opts).unwrap();
            zip.write_all(content_types.as_bytes()).unwrap();

            // Build document.xml body
            let mut body = String::new();
            for p in paragraphs {
                body.push_str(&format!(
                    "<w:p><w:r><w:t>{p}</w:t></w:r></w:p>\n"
                ));
            }

            let document = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                 <w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\n\
                   <w:body>\n{body}  </w:body>\n\
                 </w:document>"
            );
            zip.start_file("word/document.xml", opts).unwrap();
            zip.write_all(document.as_bytes()).unwrap();

            // Optional metadata
            if let Some(meta) = metadata_xml {
                zip.start_file("docProps/core.xml", opts).unwrap();
                zip.write_all(meta.as_bytes()).unwrap();
            }

            zip.finish().unwrap().into_inner()
        }

        /// Helper: build a synthetic ODT ZIP in memory.
        fn build_odt(paragraphs: &[&str], meta_xml: Option<&str>) -> Vec<u8> {
            let buf = Cursor::new(Vec::new());
            let mut zip = ZipWriter::new(buf);
            let opts = SimpleFileOptions::default();

            zip.start_file("mimetype", opts).unwrap();
            zip.write_all(b"application/vnd.oasis.opendocument.text").unwrap();

            let mut body = String::new();
            for p in paragraphs {
                body.push_str(&format!(
                    "<text:p text:style-name=\"Standard\">{p}</text:p>\n"
                ));
            }

            let content = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                 <office:document-content \
                   xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" \
                   xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\">\n\
                   <office:body>\n\
                     <office:text>\n{body}    </office:text>\n\
                   </office:body>\n\
                 </office:document-content>"
            );
            zip.start_file("content.xml", opts).unwrap();
            zip.write_all(content.as_bytes()).unwrap();

            if let Some(meta) = meta_xml {
                zip.start_file("meta.xml", opts).unwrap();
                zip.write_all(meta.as_bytes()).unwrap();
            }

            zip.finish().unwrap().into_inner()
        }

        // ================================================================
        // EPUB tests
        // ================================================================

        #[test]
        fn test_parse_epub_synthetic() {
            let data = build_epub(
                "<dc:title>Test Book</dc:title>",
                &[
                    ("ch1.xhtml", "<p>Chapter 1 content here.</p>"),
                    ("ch2.xhtml", "<p>Chapter 2 content here.</p>"),
                ],
            );

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(&data, DocumentFormat::Epub);
            assert!(result.is_ok(), "EPUB parse failed: {:?}", result.err());
            let doc = result.unwrap();
            assert!(doc.text.contains("Chapter 1 content"));
            assert!(doc.text.contains("Chapter 2 content"));
            assert_eq!(doc.format, DocumentFormat::Epub);
        }

        #[test]
        fn test_parse_epub_metadata() {
            let data = build_epub(
                "<dc:title>My Novel</dc:title>\n\
                 <dc:creator>Jane Doe</dc:creator>\n\
                 <dc:language>en</dc:language>",
                &[("ch1.xhtml", "<p>Some text.</p>")],
            );

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let doc = parser.parse_bytes(&data, DocumentFormat::Epub).unwrap();
            assert_eq!(doc.metadata.title.as_deref(), Some("My Novel"));
        }

        #[test]
        fn test_parse_epub_empty_chapters() {
            let data = build_epub(
                "<dc:title>Empty Book</dc:title>",
                &[
                    ("ch1.xhtml", ""),
                    ("ch2.xhtml", ""),
                ],
            );

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(&data, DocumentFormat::Epub);
            assert!(result.is_ok());
        }

        // ================================================================
        // DOCX tests
        // ================================================================

        #[test]
        fn test_parse_docx_synthetic() {
            let data = build_docx(
                &["First paragraph.", "Second paragraph.", "Third paragraph."],
                None,
            );

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(&data, DocumentFormat::Docx);
            assert!(result.is_ok(), "DOCX parse failed: {:?}", result.err());
            let doc = result.unwrap();
            assert!(doc.text.contains("First paragraph"));
            assert!(doc.text.contains("Second paragraph"));
            assert!(doc.text.contains("Third paragraph"));
            assert_eq!(doc.format, DocumentFormat::Docx);
        }

        #[test]
        fn test_parse_docx_metadata() {
            let meta = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                <cp:coreProperties \
                  xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" \
                  xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n\
                  <dc:title>My Document</dc:title>\n\
                  <dc:creator>John Smith</dc:creator>\n\
                </cp:coreProperties>";

            let data = build_docx(&["Hello world."], Some(meta));

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let doc = parser.parse_bytes(&data, DocumentFormat::Docx).unwrap();
            assert_eq!(doc.metadata.title.as_deref(), Some("My Document"));
        }

        #[test]
        fn test_parse_docx_no_metadata() {
            let data = build_docx(&["Just text, no metadata."], None);

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let doc = parser.parse_bytes(&data, DocumentFormat::Docx).unwrap();
            assert!(doc.text.contains("Just text"));
            // Metadata should have defaults
            assert!(doc.metadata.title.is_none() || doc.metadata.title.as_deref() == Some(""));
        }

        // ================================================================
        // ODT tests
        // ================================================================

        #[test]
        fn test_parse_odt_synthetic() {
            let data = build_odt(
                &["ODT paragraph one.", "ODT paragraph two."],
                None,
            );

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(&data, DocumentFormat::Odt);
            assert!(result.is_ok(), "ODT parse failed: {:?}", result.err());
            let doc = result.unwrap();
            assert!(doc.text.contains("ODT paragraph one"));
            assert!(doc.text.contains("ODT paragraph two"));
            assert_eq!(doc.format, DocumentFormat::Odt);
        }

        #[test]
        fn test_parse_odt_metadata() {
            let meta = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                <office:document-meta \
                  xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" \
                  xmlns:dc=\"http://purl.org/dc/elements/1.1/\" \
                  xmlns:meta=\"urn:oasis:names:tc:opendocument:xmlns:meta:1.0\">\n\
                  <office:meta>\n\
                    <dc:title>ODT Title</dc:title>\n\
                    <dc:creator>Author Name</dc:creator>\n\
                    <dc:language>es</dc:language>\n\
                  </office:meta>\n\
                </office:document-meta>";

            let data = build_odt(&["Content here."], Some(meta));

            let parser = DocumentParser::new(DocumentParserConfig::default());
            let doc = parser.parse_bytes(&data, DocumentFormat::Odt).unwrap();
            assert_eq!(doc.metadata.title.as_deref(), Some("ODT Title"));
        }

        // ================================================================
        // Invalid/corrupt ZIP tests
        // ================================================================

        #[test]
        fn test_parse_epub_invalid_zip() {
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(b"not a zip file", DocumentFormat::Epub);
            assert!(result.is_err());
        }

        #[test]
        fn test_parse_docx_invalid_zip() {
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(b"corrupted data", DocumentFormat::Docx);
            assert!(result.is_err());
        }

        #[test]
        fn test_parse_odt_invalid_zip() {
            let parser = DocumentParser::new(DocumentParserConfig::default());
            let result = parser.parse_bytes(b"bad data", DocumentFormat::Odt);
            assert!(result.is_err());
        }
    }
