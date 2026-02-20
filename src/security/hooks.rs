//! Message hooks for pre/post processing

/// Hook that runs before sending a message
pub type PreMessageHook = Box<dyn Fn(&str) -> HookResult + Send + Sync>;

/// Hook that runs after receiving a response
pub type PostMessageHook = Box<dyn Fn(&str, &str) -> HookResult + Send + Sync>;

/// Result from a hook
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue processing
    Continue,
    /// Modify the content
    Modify(String),
    /// Block the operation with reason
    Block(String),
}

impl HookResult {
    pub fn is_blocked(&self) -> bool {
        matches!(self, HookResult::Block(_))
    }
}

/// Manager for message hooks
pub struct HookManager {
    pre_hooks: Vec<(String, PreMessageHook)>,
    post_hooks: Vec<(String, PostMessageHook)>,
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HookManager {
    pub fn new() -> Self {
        Self {
            pre_hooks: Vec::new(),
            post_hooks: Vec::new(),
        }
    }

    /// Register a pre-message hook
    pub fn register_pre_hook<F>(&mut self, name: &str, hook: F)
    where
        F: Fn(&str) -> HookResult + Send + Sync + 'static,
    {
        self.pre_hooks.push((name.to_string(), Box::new(hook)));
    }

    /// Register a post-message hook
    pub fn register_post_hook<F>(&mut self, name: &str, hook: F)
    where
        F: Fn(&str, &str) -> HookResult + Send + Sync + 'static,
    {
        self.post_hooks.push((name.to_string(), Box::new(hook)));
    }

    /// Remove a pre-message hook by name
    pub fn remove_pre_hook(&mut self, name: &str) {
        self.pre_hooks.retain(|(n, _)| n != name);
    }

    /// Remove a post-message hook by name
    pub fn remove_post_hook(&mut self, name: &str) {
        self.post_hooks.retain(|(n, _)| n != name);
    }

    /// Run all pre-message hooks
    pub fn run_pre_hooks(&self, message: &str) -> HookChainResult {
        let mut current = message.to_string();
        let mut modifications = Vec::new();

        for (name, hook) in &self.pre_hooks {
            match hook(&current) {
                HookResult::Continue => {}
                HookResult::Modify(new_content) => {
                    modifications.push(name.clone());
                    current = new_content;
                }
                HookResult::Block(reason) => {
                    return HookChainResult::Blocked {
                        by_hook: name.clone(),
                        reason,
                    };
                }
            }
        }

        if modifications.is_empty() {
            HookChainResult::Unchanged
        } else {
            HookChainResult::Modified {
                content: current,
                by_hooks: modifications,
            }
        }
    }

    /// Run all post-message hooks
    pub fn run_post_hooks(&self, user_message: &str, response: &str) -> HookChainResult {
        let mut current = response.to_string();
        let mut modifications = Vec::new();

        for (name, hook) in &self.post_hooks {
            match hook(user_message, &current) {
                HookResult::Continue => {}
                HookResult::Modify(new_content) => {
                    modifications.push(name.clone());
                    current = new_content;
                }
                HookResult::Block(reason) => {
                    return HookChainResult::Blocked {
                        by_hook: name.clone(),
                        reason,
                    };
                }
            }
        }

        if modifications.is_empty() {
            HookChainResult::Unchanged
        } else {
            HookChainResult::Modified {
                content: current,
                by_hooks: modifications,
            }
        }
    }

    /// Get list of registered pre-hooks
    pub fn list_pre_hooks(&self) -> Vec<&str> {
        self.pre_hooks.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Get list of registered post-hooks
    pub fn list_post_hooks(&self) -> Vec<&str> {
        self.post_hooks.iter().map(|(n, _)| n.as_str()).collect()
    }
}

/// Result from running a chain of hooks
#[derive(Debug, Clone)]
pub enum HookChainResult {
    /// No modifications made
    Unchanged,
    /// Content was modified
    Modified {
        content: String,
        by_hooks: Vec<String>,
    },
    /// Processing was blocked
    Blocked { by_hook: String, reason: String },
}

impl HookChainResult {
    pub fn is_blocked(&self) -> bool {
        matches!(self, HookChainResult::Blocked { .. })
    }

    pub fn get_content(&self, original: &str) -> String {
        match self {
            HookChainResult::Unchanged => original.to_string(),
            HookChainResult::Modified { content, .. } => content.clone(),
            HookChainResult::Blocked { .. } => original.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_manager() {
        let mut manager = HookManager::new();

        manager.register_pre_hook("uppercase", |msg| HookResult::Modify(msg.to_uppercase()));

        let result = manager.run_pre_hooks("hello");
        match result {
            HookChainResult::Modified { content, .. } => {
                assert_eq!(content, "HELLO");
            }
            _ => panic!("Expected modified result"),
        }
    }

    #[test]
    fn test_hook_removal() {
        let mut manager = HookManager::new();

        manager.register_pre_hook("hook_a", |_msg| HookResult::Continue);
        manager.register_pre_hook("hook_b", |_msg| HookResult::Continue);

        assert_eq!(manager.list_pre_hooks(), vec!["hook_a", "hook_b"]);

        manager.remove_pre_hook("hook_a");

        let remaining = manager.list_pre_hooks();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], "hook_b");
    }

    #[test]
    fn test_post_hooks_execution() {
        let mut manager = HookManager::new();

        manager.register_post_hook("append_footer", |_user_msg, response| {
            HookResult::Modify(format!("{} [reviewed]", response))
        });

        let result = manager.run_post_hooks("question", "answer");
        match result {
            HookChainResult::Modified { content, by_hooks } => {
                assert_eq!(content, "answer [reviewed]");
                assert_eq!(by_hooks, vec!["append_footer"]);
            }
            _ => panic!("Expected Modified result"),
        }
    }

    #[test]
    fn test_hook_blocking() {
        let mut manager = HookManager::new();

        manager.register_pre_hook("blocker", |msg| {
            if msg.contains("forbidden") {
                HookResult::Block("Contains forbidden content".to_string())
            } else {
                HookResult::Continue
            }
        });

        // Non-forbidden message passes through
        let ok_result = manager.run_pre_hooks("hello world");
        assert!(!ok_result.is_blocked());

        // Forbidden message is blocked
        let blocked_result = manager.run_pre_hooks("this is forbidden text");
        assert!(blocked_result.is_blocked());
        match blocked_result {
            HookChainResult::Blocked { by_hook, reason } => {
                assert_eq!(by_hook, "blocker");
                assert_eq!(reason, "Contains forbidden content");
            }
            _ => panic!("Expected Blocked result"),
        }
    }
}
