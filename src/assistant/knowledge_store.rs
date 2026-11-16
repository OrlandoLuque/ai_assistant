//! Knowledge state for the assistant facade.
//!
//! Phase 5 of `ai_assistant_plans/HEXAGONAL_PLAN.md` ("adelgazar la fachada")
//! asks for the `#[cfg(feature = "rag")]` state of `AiAssistant` to become a
//! component the facade *composes* instead of state it *contains*. This is that
//! component: the twelve fields that lived and died behind the `rag` flag, plus
//! the one invariant that concerns nothing else -- "the database is open, or can
//! be opened from the path we were given".
//!
//! What this deliberately is NOT: a port. There is no trait here, because there
//! is no second knowledge backend to swap in -- the pluggable half of RAG
//! already lives behind `trait VectorDb`. A `trait KnowledgeStore` with exactly
//! one implementor would repeat the defect `PiiMaskingProvider` is still an
//! example of in this crate: an abstraction exported to the public with zero
//! users. If a second implementation ever shows up, the trait goes here and the
//! facade keeps composing. The same reasoning is why this type is `pub(crate)`
//! and not re-exported from `lib.rs`.

use super::IndexingProgress;
use crate::rag::{KnowledgeUsage, RagConfig, RagDb, DEFAULT_USER_ID};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

/// Everything the assistant holds about its knowledge base.
///
/// Fields are `pub(crate)` rather than private because the methods that use
/// them are `impl AiAssistant` blocks in sibling modules (`assistant::rag` and
/// friends), not methods here. That is a deliberate first step and not the end
/// state: it relocates the state without moving behaviour, so the result can be
/// checked against the old code line by line. Five of these fields were fully
/// `pub` on the facade before this; now the only way in from outside the crate
/// is through the accessors on `AiAssistant`.
pub(crate) struct KnowledgeStore {
    /// RAG database for knowledge and conversation storage.
    pub(crate) db: Option<RagDb>,
    /// RAG configuration.
    pub(crate) config: RagConfig,
    /// Current user ID for multi-user RAG operations.
    pub(crate) user_id: String,
    /// True if background indexing is in progress.
    pub(crate) is_indexing: bool,
    /// Last knowledge usage from the most recent RAG context build.
    pub(crate) last_usage: Option<KnowledgeUsage>,
    /// IDs of messages stored in the RAG DB for the current session.
    pub(crate) message_ids: Vec<i64>,
    /// Path for the RAG database (for lazy initialization).
    pub(crate) db_path: Option<PathBuf>,
    /// Registered documents pending indexing: source_name -> content.
    pub(crate) pending_documents: HashMap<String, String>,
    /// Documents that have been registered (for tracking available sources).
    pub(crate) registered_sources: Vec<String>,
    /// Channel for receiving indexing progress/results.
    pub(crate) rx_indexing: Option<Receiver<IndexingProgress>>,
    /// Cache for RAG search results.
    pub(crate) cache: Option<crate::metrics::SearchCache<Vec<crate::rag::KnowledgeChunk>>>,
    /// History of knowledge usage per message (most recent first).
    pub(crate) usage_history: Vec<KnowledgeUsage>,
}

impl Default for KnowledgeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeStore {
    /// A store with no database, no path and an empty search cache.
    ///
    /// The cache sizing (50 entries, 5 minute TTL) is the one the facade used
    /// inline before this type existed; it is kept here so that one place
    /// decides it.
    pub(crate) fn new() -> Self {
        Self {
            db: None,
            config: RagConfig::default(),
            user_id: DEFAULT_USER_ID.to_string(),
            is_indexing: false,
            last_usage: None,
            message_ids: Vec::new(),
            db_path: None,
            pending_documents: HashMap::new(),
            registered_sources: Vec::new(),
            rx_indexing: None,
            cache: Some(crate::metrics::SearchCache::new(50, 300)),
            usage_history: Vec::new(),
        }
    }

    /// Open the database if a path was set and it is not open yet.
    ///
    /// Returns true if RAG is usable afterwards. This is the invariant that
    /// belongs to this cluster alone: it reads `db_path` and writes `db`, and
    /// touches nothing else in the assistant. It used to be
    /// `AiAssistant::ensure_rag_initialized`, which still exists as a one-line
    /// delegate because a dozen call sites use it.
    pub(crate) fn ensure_initialized(&mut self) -> bool {
        if self.db.is_some() {
            return true;
        }

        if let Some(path) = self.db_path.clone() {
            match RagDb::open(&path) {
                Ok(db) => {
                    self.db = Some(db);
                    true
                }
                Err(e) => {
                    log::error!("[AI RAG] Failed to initialize database: {}", e);
                    false
                }
            }
        } else {
            false
        }
    }

    /// True if RAG is configured: a database is open, or a path was set for one.
    pub(crate) fn has_rag(&self) -> bool {
        self.db.is_some() || self.db_path.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A unique path under the system temp dir.
    ///
    /// Every path in these tests goes through here. The first version of this
    /// module used relative paths like `./no_such_directory_12345/rag.db` on the
    /// assumption that they would fail to open -- and `RagDb::open` creates the
    /// parent directory, so the test left a 100 KiB database **inside the
    /// repository working tree**. A test must not be able to dirty the tree it
    /// is testing.
    fn temp_path(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "ai_assistant_ks_{}_{}_{}",
            tag,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn a_new_store_has_no_database_and_no_path() {
        let s = KnowledgeStore::new();
        assert!(s.db.is_none());
        assert!(s.db_path.is_none());
        assert!(
            !s.has_rag(),
            "a store with neither db nor path is not usable"
        );
        assert!(s.cache.is_some(), "the search cache is created eagerly");
        assert_eq!(s.user_id, DEFAULT_USER_ID);
    }

    #[test]
    fn has_rag_is_true_with_only_a_path() {
        // The distinction matters: a path with no open database still means RAG
        // is configured, because `ensure_initialized` opens it on demand.
        let mut s = KnowledgeStore::new();
        s.db_path = Some(temp_path("haspath").join("x.db"));
        assert!(s.has_rag());
        assert!(s.db.is_none(), "has_rag does not open anything");
    }

    #[test]
    fn ensure_initialized_without_a_path_is_false_and_opens_nothing() {
        let mut s = KnowledgeStore::new();
        assert!(!s.ensure_initialized());
        assert!(s.db.is_none());
    }

    #[test]
    fn ensure_initialized_creates_the_missing_parent_directory() {
        // MEASURED, not assumed: `RagDb::open` runs `create_dir_all(parent)`
        // before opening, so "the directory does not exist" is NOT a failure
        // case -- it is a supported one. The first draft of this test asserted
        // the opposite and failed, which is how the behaviour was found.
        let dir = temp_path("mkparent");
        let path = dir.join("nested").join("rag.db");
        assert!(!dir.exists(), "precondition: nothing there yet");

        let mut s = KnowledgeStore::new();
        s.db_path = Some(path.clone());
        assert!(
            s.ensure_initialized(),
            "a missing parent is created, not fatal"
        );
        assert!(path.exists(), "and the database file is really there");

        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_initialized_reports_failure_instead_of_panicking() {
        // Something that genuinely cannot be opened on any platform: a path
        // whose parent component is an existing *file*, so `create_dir_all`
        // cannot create the directory. The contract is a `false`, not a panic
        // and not a silent `true`.
        let blocker = temp_path("blocked");
        std::fs::write(&blocker, b"not a directory").expect("write blocker file");

        let mut s = KnowledgeStore::new();
        s.db_path = Some(blocker.join("rag.db"));
        assert!(!s.ensure_initialized(), "an unopenable path is a false");
        assert!(s.db.is_none());

        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn ensure_initialized_opens_the_database_once_and_is_idempotent() {
        let dir = temp_path("idempotent");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("rag.db");
        let mut s = KnowledgeStore::new();
        s.db_path = Some(path.clone());

        assert!(s.ensure_initialized(), "first call opens the database");
        assert!(s.db.is_some());
        assert!(
            s.ensure_initialized(),
            "second call is a no-op and still true"
        );

        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_matches_new() {
        let a = KnowledgeStore::default();
        let b = KnowledgeStore::new();
        assert_eq!(a.user_id, b.user_id);
        assert_eq!(a.is_indexing, b.is_indexing);
        assert_eq!(a.message_ids.len(), b.message_ids.len());
        assert_eq!(a.registered_sources.len(), b.registered_sources.len());
    }
}
