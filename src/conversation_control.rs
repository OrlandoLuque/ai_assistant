//! Conversation control: cancellation, regeneration, editing, branching

use crate::messages::ChatMessage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ============================================================================
// Cancellation Token
// ============================================================================

/// Token for cancelling streaming responses
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation was requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Reset the token for reuse
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

// ============================================================================
// Message Operations
// ============================================================================

/// Result of a message edit operation
#[derive(Debug, Clone)]
pub struct EditResult {
    /// Index of the edited message
    pub message_index: usize,
    /// Original content
    pub original_content: String,
    /// New content
    pub new_content: String,
    /// Messages that were removed (responses after the edit point)
    pub removed_count: usize,
}

/// Result of a regeneration operation
#[derive(Debug, Clone)]
pub struct RegenerateResult {
    /// Index of the regenerated message
    pub message_index: usize,
    /// Original response
    pub original_response: String,
    /// Prompt that triggered the original response
    pub user_prompt: String,
}

/// Manager for conversation message operations
pub struct MessageOperations;

impl MessageOperations {
    /// Edit a user message and optionally remove subsequent messages
    ///
    /// Returns the edit result if successful, or None if the index is invalid
    /// or the message is not a user message.
    pub fn edit_user_message(
        conversation: &mut Vec<ChatMessage>,
        message_index: usize,
        new_content: &str,
        remove_subsequent: bool,
    ) -> Option<EditResult> {
        if message_index >= conversation.len() {
            return None;
        }

        let msg = &conversation[message_index];
        if msg.role != "user" {
            return None;
        }

        let original_content = msg.content.clone();
        let removed_count = if remove_subsequent {
            let count = conversation.len() - message_index - 1;
            conversation.truncate(message_index + 1);
            count
        } else {
            0
        };

        // Update the message
        conversation[message_index] = ChatMessage {
            role: "user".to_string(),
            content: new_content.to_string(),
            timestamp: Utc::now(),
        };

        Some(EditResult {
            message_index,
            original_content,
            new_content: new_content.to_string(),
            removed_count,
        })
    }

    /// Prepare for regenerating the last assistant response
    ///
    /// This removes the last assistant message and returns info needed for regeneration.
    /// Returns None if there's no assistant message to regenerate.
    pub fn prepare_regeneration(conversation: &mut Vec<ChatMessage>) -> Option<RegenerateResult> {
        // Find the last assistant message
        let last_assistant_idx = conversation.iter().rposition(|m| m.role == "assistant")?;

        // Find the user message that prompted it
        let user_prompt = conversation[..last_assistant_idx]
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let original_response = conversation[last_assistant_idx].content.clone();

        // Remove the assistant message
        conversation.remove(last_assistant_idx);

        Some(RegenerateResult {
            message_index: last_assistant_idx,
            original_response,
            user_prompt,
        })
    }

    /// Get the last user message for context
    pub fn get_last_user_message(conversation: &[ChatMessage]) -> Option<&ChatMessage> {
        conversation.iter().rfind(|m| m.role == "user")
    }

    /// Count messages by role
    pub fn count_by_role(conversation: &[ChatMessage]) -> (usize, usize, usize) {
        let user = conversation.iter().filter(|m| m.role == "user").count();
        let assistant = conversation
            .iter()
            .filter(|m| m.role == "assistant")
            .count();
        let system = conversation.iter().filter(|m| m.role == "system").count();
        (user, assistant, system)
    }
}

// ============================================================================
// Conversation Branching
// ============================================================================

/// A branch point in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPoint {
    /// Unique identifier for this branch
    pub id: String,
    /// Name/label for the branch
    pub name: String,
    /// Index in the original conversation where branch starts
    pub branch_index: usize,
    /// When the branch was created
    pub created_at: DateTime<Utc>,
    /// Parent branch ID (None for main branch)
    pub parent_branch: Option<String>,
    /// Messages in this branch (from branch point onwards)
    pub messages: Vec<ChatMessage>,
}

impl BranchPoint {
    pub fn new(name: &str, branch_index: usize, parent: Option<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            branch_index,
            created_at: Utc::now(),
            parent_branch: parent,
            messages: Vec::new(),
        }
    }

    /// Get the full path of branch IDs from root to this branch
    pub fn get_lineage(&self, branches: &[BranchPoint]) -> Vec<String> {
        let mut lineage = Vec::new();
        let mut current = self.parent_branch.clone();

        while let Some(parent_id) = current {
            lineage.push(parent_id.clone());
            current = branches
                .iter()
                .find(|b| b.id == parent_id)
                .and_then(|b| b.parent_branch.clone());
        }

        lineage.reverse();
        lineage.push(self.id.clone());
        lineage
    }
}

/// Manager for conversation branches
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchManager {
    /// All branches for this session
    branches: Vec<BranchPoint>,
    /// Currently active branch ID (None = main conversation)
    active_branch: Option<String>,
}

impl BranchManager {
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            active_branch: None,
        }
    }

    /// Create a new branch from the current conversation state
    ///
    /// # Arguments
    /// * `name` - Human-readable name for the branch
    /// * `conversation` - Current conversation messages
    /// * `branch_from_index` - Index to branch from (messages after this are branch-specific)
    pub fn create_branch(
        &mut self,
        name: &str,
        conversation: &[ChatMessage],
        branch_from_index: usize,
    ) -> String {
        let branch = BranchPoint {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            branch_index: branch_from_index,
            created_at: Utc::now(),
            parent_branch: self.active_branch.clone(),
            messages: conversation[branch_from_index..].to_vec(),
        };

        let branch_id = branch.id.clone();
        self.branches.push(branch);
        branch_id
    }

    /// Switch to a different branch
    ///
    /// Returns the messages to use for the new active branch
    pub fn switch_branch(
        &mut self,
        branch_id: &str,
        base_conversation: &[ChatMessage],
    ) -> Option<Vec<ChatMessage>> {
        let branch = self.branches.iter().find(|b| b.id == branch_id)?;

        // Build conversation: base up to branch point + branch messages
        let mut conversation = base_conversation[..branch.branch_index].to_vec();
        conversation.extend(branch.messages.clone());

        self.active_branch = Some(branch_id.to_string());
        Some(conversation)
    }

    /// Return to main conversation (no active branch)
    pub fn switch_to_main(&mut self) {
        self.active_branch = None;
    }

    /// Get the currently active branch
    pub fn get_active_branch(&self) -> Option<&BranchPoint> {
        self.active_branch
            .as_ref()
            .and_then(|id| self.branches.iter().find(|b| &b.id == id))
    }

    /// Get active branch ID
    pub fn active_branch_id(&self) -> Option<&str> {
        self.active_branch.as_deref()
    }

    /// Update the active branch with new messages
    pub fn update_active_branch(&mut self, messages_from_branch_point: &[ChatMessage]) {
        if let Some(ref branch_id) = self.active_branch {
            if let Some(branch) = self.branches.iter_mut().find(|b| &b.id == branch_id) {
                branch.messages = messages_from_branch_point.to_vec();
            }
        }
    }

    /// Delete a branch
    pub fn delete_branch(&mut self, branch_id: &str) -> bool {
        if self.active_branch.as_deref() == Some(branch_id) {
            self.active_branch = None;
        }

        let initial_len = self.branches.len();
        self.branches.retain(|b| b.id != branch_id);

        // Also delete child branches
        let children: Vec<String> = self
            .branches
            .iter()
            .filter(|b| b.parent_branch.as_deref() == Some(branch_id))
            .map(|b| b.id.clone())
            .collect();

        for child_id in children {
            self.delete_branch(&child_id);
        }

        self.branches.len() != initial_len
    }

    /// Rename a branch
    pub fn rename_branch(&mut self, branch_id: &str, new_name: &str) -> bool {
        if let Some(branch) = self.branches.iter_mut().find(|b| b.id == branch_id) {
            branch.name = new_name.to_string();
            true
        } else {
            false
        }
    }

    /// Get all branches
    pub fn get_branches(&self) -> &[BranchPoint] {
        &self.branches
    }

    /// Get branch by ID
    pub fn get_branch(&self, branch_id: &str) -> Option<&BranchPoint> {
        self.branches.iter().find(|b| b.id == branch_id)
    }

    /// Get branches that stem from a specific index
    pub fn get_branches_at_index(&self, index: usize) -> Vec<&BranchPoint> {
        self.branches
            .iter()
            .filter(|b| b.branch_index == index)
            .collect()
    }

    /// Check if there are any branches
    pub fn has_branches(&self) -> bool {
        !self.branches.is_empty()
    }

    /// Get branch count
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Merge a branch into main conversation
    ///
    /// This appends the branch messages after the branch point in the main conversation.
    /// Returns the merged conversation.
    pub fn merge_branch(
        &mut self,
        branch_id: &str,
        base_conversation: &[ChatMessage],
    ) -> Option<Vec<ChatMessage>> {
        let branch = self.branches.iter().find(|b| b.id == branch_id)?.clone();

        let mut merged = base_conversation[..branch.branch_index].to_vec();
        merged.extend(branch.messages);

        // Delete the merged branch
        self.delete_branch(branch_id);

        Some(merged)
    }
}

// ============================================================================
// Response Variants Tracking
// ============================================================================

/// Tracks multiple response variants for the same prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseVariant {
    /// The response content
    pub content: String,
    /// When this variant was generated
    pub generated_at: DateTime<Utc>,
    /// Model used for this variant
    pub model: String,
    /// Temperature setting used
    pub temperature: f32,
    /// User rating (if provided)
    pub rating: Option<u8>,
    /// Is this the currently selected variant
    pub is_selected: bool,
}

/// Manages multiple response variants for a message
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantManager {
    /// Message index -> list of variants
    variants: std::collections::HashMap<usize, Vec<ResponseVariant>>,
}

impl VariantManager {
    pub fn new() -> Self {
        Self {
            variants: std::collections::HashMap::new(),
        }
    }

    /// Add a variant for a message
    pub fn add_variant(
        &mut self,
        message_index: usize,
        content: String,
        model: String,
        temperature: f32,
    ) {
        let variant = ResponseVariant {
            content,
            generated_at: Utc::now(),
            model,
            temperature,
            rating: None,
            is_selected: false,
        };

        self.variants
            .entry(message_index)
            .or_insert_with(Vec::new)
            .push(variant);
    }

    /// Select a variant for a message
    pub fn select_variant(&mut self, message_index: usize, variant_index: usize) -> bool {
        if let Some(variants) = self.variants.get_mut(&message_index) {
            for (i, v) in variants.iter_mut().enumerate() {
                v.is_selected = i == variant_index;
            }
            true
        } else {
            false
        }
    }

    /// Get all variants for a message
    pub fn get_variants(&self, message_index: usize) -> Option<&Vec<ResponseVariant>> {
        self.variants.get(&message_index)
    }

    /// Get selected variant for a message
    pub fn get_selected_variant(&self, message_index: usize) -> Option<&ResponseVariant> {
        self.variants
            .get(&message_index)
            .and_then(|vars| vars.iter().find(|v| v.is_selected))
    }

    /// Rate a variant
    pub fn rate_variant(&mut self, message_index: usize, variant_index: usize, rating: u8) -> bool {
        if let Some(variants) = self.variants.get_mut(&message_index) {
            if let Some(variant) = variants.get_mut(variant_index) {
                variant.rating = Some(rating.min(5));
                return true;
            }
        }
        false
    }

    /// Get variant count for a message
    pub fn variant_count(&self, message_index: usize) -> usize {
        self.variants
            .get(&message_index)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Check if message has multiple variants
    pub fn has_variants(&self, message_index: usize) -> bool {
        self.variant_count(message_index) > 1
    }

    /// Clear variants for a message
    pub fn clear_variants(&mut self, message_index: usize) {
        self.variants.remove(&message_index);
    }

    /// Clear all variants
    pub fn clear_all(&mut self) {
        self.variants.clear();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_edit_user_message() {
        let mut conversation = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
            ChatMessage::assistant("I'm doing well!"),
        ];

        let result =
            MessageOperations::edit_user_message(&mut conversation, 2, "What's your name?", true);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.original_content, "How are you?");
        assert_eq!(result.removed_count, 1);
        assert_eq!(conversation.len(), 3);
        assert_eq!(conversation[2].content, "What's your name?");
    }

    #[test]
    fn test_prepare_regeneration() {
        let mut conversation = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let result = MessageOperations::prepare_regeneration(&mut conversation);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.original_response, "Hi there!");
        assert_eq!(result.user_prompt, "Hello");
        assert_eq!(conversation.len(), 1);
    }

    #[test]
    fn test_branch_manager() {
        let mut manager = BranchManager::new();

        let conversation = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
            ChatMessage::user("Question 1"),
            ChatMessage::assistant("Answer 1"),
        ];

        // Create branch from index 2
        let branch_id = manager.create_branch("Alternative", &conversation, 2);

        assert_eq!(manager.branch_count(), 1);

        // Switch to branch
        let branch_conv = manager.switch_branch(&branch_id, &conversation);
        assert!(branch_conv.is_some());

        // Verify active branch
        assert!(manager.active_branch_id().is_some());

        // Switch to main
        manager.switch_to_main();
        assert!(manager.active_branch_id().is_none());
    }

    #[test]
    fn test_variant_manager() {
        let mut manager = VariantManager::new();

        manager.add_variant(1, "Response A".to_string(), "model1".to_string(), 0.7);
        manager.add_variant(1, "Response B".to_string(), "model1".to_string(), 0.8);

        assert_eq!(manager.variant_count(1), 2);
        assert!(manager.has_variants(1));

        manager.select_variant(1, 0);
        let selected = manager.get_selected_variant(1);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().content, "Response A");
    }
}
