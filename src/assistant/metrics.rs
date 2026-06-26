use super::*;

impl AiAssistant {
    // === Metrics Methods ===

    /// Get session metrics aggregated for the current session
    pub fn get_session_metrics(&self) -> crate::metrics::SessionMetrics {
        self.metrics.get_session_metrics()
    }

    /// Get RAG quality metrics
    pub fn get_rag_quality_metrics(&self) -> crate::metrics::RagQualityMetrics {
        self.metrics.get_rag_quality_metrics()
    }

    /// Get all message metrics from the current session
    pub fn get_message_metrics(&self) -> &[crate::metrics::MessageMetrics] {
        self.metrics.get_message_metrics()
    }

    /// Export all metrics as JSON
    pub fn export_metrics_json(&self) -> String {
        self.metrics.export_json()
    }

    /// Reset metrics for a new session
    pub fn reset_metrics(&mut self, session_id: &str) {
        self.metrics = crate::metrics::MetricsTracker::new(session_id);
    }

    /// Start tracking a new message (call before sending)
    pub fn start_message_tracking(&mut self) {
        self.metrics.start_message(&self.config.selected_model);
    }

    /// Mark that the first token was received
    pub fn mark_first_token_received(&mut self) {
        self.metrics.mark_first_token();
    }

    /// Finish tracking the current message (call after response complete)
    pub fn finish_message_tracking(&mut self, output_tokens: usize) {
        self.metrics.finish_message(output_tokens);
    }

    /// Clear the RAG search cache
    #[cfg(feature = "rag")]
    pub fn clear_rag_cache(&mut self) {
        if let Some(ref mut cache) = self.rag_cache {
            cache.clear();
        }
    }

    /// Get RAG cache statistics: (entries, total_hits)
    #[cfg(feature = "rag")]
    pub fn get_rag_cache_stats(&self) -> (usize, usize) {
        self.rag_cache.as_ref().map(|c| c.stats()).unwrap_or((0, 0))
    }

    #[cfg(feature = "rag")]
    /// Set priority for a knowledge source
    /// Higher priority sources appear first in search results
    pub fn set_source_priority(&mut self, source: &str, priority: i32) -> AiResult<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.set_source_priority(source, priority)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Get priority for a knowledge source
    pub fn get_source_priority(&mut self, source: &str) -> i32 {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_source_priority(source).unwrap_or(0);
            }
        }
        0
    }
}
