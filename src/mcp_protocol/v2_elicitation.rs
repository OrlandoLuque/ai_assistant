//! MCP Elicitation Protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Action the user/host takes when responding to an elicitation request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ElicitAction {
    Accept,
    Deny,
    Dismiss,
}

/// Describes the type of an elicitation field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ElicitFieldType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "number")]
    Number,
    #[serde(rename = "boolean")]
    Boolean,
    #[serde(rename = "select")]
    Select { options: Vec<String> },
    #[serde(rename = "file_upload")]
    FileUpload { accepted_types: Vec<String> },
}

/// Schema for a single field in an elicitation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitFieldSchema {
    pub field_name: String,
    pub field_type: ElicitFieldType,
    pub description: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
}

/// An elicitation request sent from a server to the host to gather user input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitRequest {
    pub request_id: String,
    pub message: String,
    pub fields: Vec<ElicitFieldSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

/// The host's response to an elicitation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitResponse {
    pub request_id: String,
    pub action: ElicitAction,
    pub values: HashMap<String, serde_json::Value>,
}

/// Trait for components that handle elicitation requests on behalf of the host.
pub trait ElicitationHandler: Send + Sync {
    /// Handle an incoming elicitation request and return a response.
    fn handle_elicitation(&self, request: &ElicitRequest) -> ElicitResponse;
    /// A human-readable name for this handler.
    fn name(&self) -> &str;
}

/// An `ElicitationHandler` that automatically accepts every elicitation,
/// filling in configured default values.
pub struct AutoAcceptHandler {
    default_values: HashMap<String, serde_json::Value>,
}

impl AutoAcceptHandler {
    /// Create a handler with no default values (responds `Accept` with empty map).
    pub fn new() -> Self {
        Self {
            default_values: HashMap::new(),
        }
    }

    /// Create a handler that supplies the given default values on accept.
    pub fn with_defaults(defaults: HashMap<String, serde_json::Value>) -> Self {
        Self {
            default_values: defaults,
        }
    }
}

impl Default for AutoAcceptHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ElicitationHandler for AutoAcceptHandler {
    fn handle_elicitation(&self, request: &ElicitRequest) -> ElicitResponse {
        // Build the values map: for each field in the request, use the
        // handler's configured default if present, otherwise use the field's
        // own default_value if present.
        let mut values = HashMap::new();
        for field in &request.fields {
            if let Some(val) = self.default_values.get(&field.field_name) {
                values.insert(field.field_name.clone(), val.clone());
            } else if let Some(ref dv) = field.default_value {
                values.insert(field.field_name.clone(), dv.clone());
            }
        }
        ElicitResponse {
            request_id: request.request_id.clone(),
            action: ElicitAction::Accept,
            values,
        }
    }

    fn name(&self) -> &str {
        "auto_accept"
    }
}
