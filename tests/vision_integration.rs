//! Cross-module vision integration tests.
//!
//! Verifies that an image attached at one surface flows correctly through
//! the rest of the stack:
//!
//! * `ChatMessage::with_image` → `messages::ChatMessage` carrier
//! * `A2AMessage::image` ↔ `extract_image_parts` round-trip
//! * `ContextBudgetAllocator` reserves image-token budget before text packing
//! * `SqliteSessionStore::attach_image` → `attachments_for_message` round-trip
//! * `AiResponse::Image` surfaces image-out through the canonical channel
//!
//! Each test exercises a real public API path (no internals reach-around)
//! so a regression in any one module breaks the corresponding test.

#![cfg(feature = "vision")]

use ai_assistant::vision::{ImageData, ImageDetail, ImageInput, ImageRef};
use ai_assistant::{AiResponse, ChatMessage};

// Build a deterministic 1x1 PNG-shaped base64 payload — content does not
// matter for these tests, only that it survives serialization and refs.
fn sample_image() -> ImageInput {
    ImageInput {
        data: ImageData::Base64("iVBORw0KGgoAAAANSUhEUg==".to_string()),
        media_type: "image/png".to_string(),
        detail: ImageDetail::Auto,
    }
}

fn sample_image_url() -> ImageInput {
    ImageInput {
        data: ImageData::Url("https://example.test/cat.jpg".to_string()),
        media_type: "image/jpeg".to_string(),
        detail: ImageDetail::Auto,
    }
}

// ---------------------------------------------------------------------------
// ChatMessage carrier
// ---------------------------------------------------------------------------

#[test]
fn chat_message_carries_images_through_builder() {
    let msg = ChatMessage::user("describe this picture")
        .with_image(sample_image())
        .with_image(sample_image_url());

    assert!(msg.has_images());
    assert_eq!(msg.images.len(), 2);
    assert_eq!(msg.images[0].media_type, "image/png");
    match &msg.images[1].data {
        ImageData::Url(u) => assert_eq!(u, "https://example.test/cat.jpg"),
        other => panic!("expected Url variant, got {:?}", other),
    }
}

#[test]
fn chat_message_default_has_no_images() {
    let msg = ChatMessage::assistant("plain text reply");
    assert!(!msg.has_images());
    assert!(msg.images.is_empty());
}

// ---------------------------------------------------------------------------
// A2A message <-> ImageInput round-trip
// ---------------------------------------------------------------------------

#[cfg(feature = "a2a")]
#[test]
fn a2a_image_message_round_trips_through_extract() {
    use ai_assistant::a2a_protocol::{A2AMessage, MessageRole};

    let original = sample_image();
    let msg = A2AMessage::image(MessageRole::User, original.clone());

    let extracted = msg.extract_image_parts();
    assert_eq!(extracted.len(), 1);
    assert_eq!(extracted[0].media_type, original.media_type);
    match (&extracted[0].data, &original.data) {
        (ImageData::Base64(a), ImageData::Base64(b)) => assert_eq!(a, b),
        _ => panic!("base64 image round-trip lost variant"),
    }
}

#[cfg(feature = "a2a")]
#[test]
fn a2a_url_image_round_trips() {
    use ai_assistant::a2a_protocol::{A2AMessage, MessageRole};

    let msg = A2AMessage::image(MessageRole::Agent, sample_image_url());
    let extracted = msg.extract_image_parts();
    assert_eq!(extracted.len(), 1);
    match &extracted[0].data {
        ImageData::Url(u) => assert_eq!(u, "https://example.test/cat.jpg"),
        other => panic!("expected Url after extract, got {:?}", other),
    }
}

#[cfg(feature = "a2a")]
#[test]
fn a2a_text_only_message_has_no_image_parts() {
    use ai_assistant::a2a_protocol::{A2AMessage, MessageRole};

    let msg = A2AMessage::text(MessageRole::User, "no images here");
    assert!(msg.extract_image_parts().is_empty());
}

// ---------------------------------------------------------------------------
// Context budget — image tokens reserved before text packing
// ---------------------------------------------------------------------------

#[test]
fn context_budget_reserves_image_tokens_before_packing() {
    use ai_assistant::context_budget::{
        ContextBudgetAllocator, ContextItem, ContextSource, ContextSourceType,
    };

    // Vision-aware source declaring a fixed image-token reservation.
    struct VisionSrc {
        image_tokens: usize,
    }
    impl ContextSource for VisionSrc {
        fn query_items(&self, _msg: &str) -> Vec<ContextItem> {
            Vec::new()
        }
        fn source_name(&self) -> &str {
            "vision"
        }
        fn source_type(&self) -> ContextSourceType {
            ContextSourceType::Custom
        }
        fn image_token_estimate(&self, _msg: &str) -> usize {
            self.image_tokens
        }
    }

    // Text source emitting one chunk that is too big to fit once images
    // have eaten the budget — proves reservation actually shrinks the
    // packing window.
    struct TextSrc;
    impl ContextSource for TextSrc {
        fn query_items(&self, _msg: &str) -> Vec<ContextItem> {
            vec![ContextItem {
                content: "x".repeat(4 * 600),
                tokens: 600,
                score: 0.9,
                source: ContextSourceType::Custom,
                label: "text-chunk".to_string(),
            }]
        }
        fn source_name(&self) -> &str {
            "text"
        }
        fn source_type(&self) -> ContextSourceType {
            ContextSourceType::Custom
        }
    }

    let mut allocator = ContextBudgetAllocator::default();
    allocator.add_source(Box::new(VisionSrc { image_tokens: 600 }));
    allocator.add_source(Box::new(TextSrc));

    // Total budget 1000; vision reserves 600 → only 400 left for text,
    // the 600-token chunk MUST be dropped.
    let result = allocator.build("user query", 1000);
    assert!(
        result.included.is_empty(),
        "text chunk should not fit after image reservation"
    );
    assert_eq!(result.dropped.len(), 1);
}

#[test]
fn context_budget_aggregates_multiple_vision_sources() {
    use ai_assistant::context_budget::{
        ContextBudgetAllocator, ContextItem, ContextSource, ContextSourceType,
    };

    struct VisionSrc {
        image_tokens: usize,
    }
    impl ContextSource for VisionSrc {
        fn query_items(&self, _msg: &str) -> Vec<ContextItem> {
            Vec::new()
        }
        fn source_name(&self) -> &str {
            "v"
        }
        fn source_type(&self) -> ContextSourceType {
            ContextSourceType::Custom
        }
        fn image_token_estimate(&self, _msg: &str) -> usize {
            self.image_tokens
        }
    }

    let mut allocator = ContextBudgetAllocator::default();
    allocator.add_source(Box::new(VisionSrc { image_tokens: 250 }));
    allocator.add_source(Box::new(VisionSrc { image_tokens: 350 }));

    assert_eq!(allocator.estimated_image_tokens("any"), 600);
}

// ---------------------------------------------------------------------------
// SQLite session store — attachment round-trip + cascade
// ---------------------------------------------------------------------------

#[cfg(feature = "rag")]
#[test]
fn sqlite_session_attachment_round_trip() {
    use ai_assistant::{ChatSession, SqliteSessionStore, UnifiedDb};
    use tempfile::NamedTempFile;

    let tmp = NamedTempFile::new().expect("temp file");
    let db = UnifiedDb::open(tmp.path()).expect("open unified db");
    let store = SqliteSessionStore::new(&db);

    let mut session = ChatSession::new("vision-flow");
    session.messages.push(ChatMessage::user("describe"));
    store.save_session(&session).expect("save session");

    let ids = store
        .message_ids_for_session(&session.id)
        .expect("message ids");
    assert_eq!(ids.len(), 1);
    let message_id = ids[0];

    let r1 = ImageRef {
        key: "k1".to_string(),
        media_type: "image/png".to_string(),
        sha256: "1".repeat(64),
    };
    let r2 = ImageRef {
        key: "k2".to_string(),
        media_type: "image/jpeg".to_string(),
        sha256: "2".repeat(64),
    };
    store.attach_image(message_id, &r1).expect("attach r1");
    store.attach_image(message_id, &r2).expect("attach r2");

    let loaded = store
        .attachments_for_message(message_id)
        .expect("load attachments");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].key, "k1");
    assert_eq!(loaded[1].sha256, "2".repeat(64));
}

// ---------------------------------------------------------------------------
// AiResponse::Image surface
// ---------------------------------------------------------------------------

#[test]
fn ai_response_image_variant_surfaces_through_accessors() {
    let img = ImageData::Base64("AAAA".to_string());
    let resp = AiResponse::Image(img);

    // Image is not text-like — text accessors stay None.
    assert!(resp.text().is_none());
    assert!(!resp.is_terminal());

    // Image accessors expose the payload.
    assert!(resp.image().is_some());
    assert_eq!(resp.images().len(), 1);
}

#[test]
fn ai_response_text_variants_have_no_image() {
    let resp = AiResponse::Complete("plain text".to_string());
    assert!(resp.image().is_none());
    assert!(resp.images().is_empty());
}
