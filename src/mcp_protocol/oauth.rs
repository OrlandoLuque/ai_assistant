//! OAuth 2.1 types for MCP authentication (MCP spec 2025-03-26, v1 flow).

use serde::{Deserialize, Serialize};

/// OAuth 2.1 grant type for MCP authentication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum McpOAuthGrantType {
    AuthorizationCode,
    ClientCredentials,
    RefreshToken,
}

/// OAuth 2.1 scope definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthScope {
    pub name: String,
    pub description: String,
    pub resources: Vec<String>,
}

/// OAuth 2.1 configuration for MCP servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthConfig {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub scopes: Vec<McpOAuthScope>,
    pub redirect_uri: String,
    pub pkce_enabled: bool,
}

/// OAuth 2.1 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
}

/// OAuth 2.1 authorization request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAuthorizationRequest {
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub state: String,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

/// OAuth 2.1 token manager for MCP sessions
pub struct McpOAuthTokenManager {
    config: McpOAuthConfig,
    current_token: Option<McpTokenResponse>,
    token_expiry_secs: Option<u64>,
    token_obtained_at: Option<std::time::Instant>,
}

impl McpOAuthTokenManager {
    pub fn new(config: McpOAuthConfig) -> Self {
        Self {
            config,
            current_token: None,
            token_expiry_secs: None,
            token_obtained_at: None,
        }
    }

    /// Build the authorization URL for the OAuth 2.1 authorization code flow.
    /// Returns (url, state) where state is the CSRF protection value.
    pub fn build_authorization_url(&self, state: &str) -> (String, McpAuthorizationRequest) {
        let scope_str = self
            .config
            .scopes
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let request = McpAuthorizationRequest {
            response_type: "code".to_string(),
            client_id: self.config.client_id.clone(),
            redirect_uri: self.config.redirect_uri.clone(),
            scope: scope_str.clone(),
            state: state.to_string(),
            code_challenge: None,
            code_challenge_method: None,
        };

        let url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}",
            self.config.authorization_endpoint,
            urlencoding::encode(&self.config.client_id),
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(&scope_str),
            urlencoding::encode(state),
        );

        (url, request)
    }

    /// Build the token request body for authorization code exchange.
    pub fn build_token_request_authorization_code(&self, code: &str) -> Vec<(String, String)> {
        let mut params = vec![
            ("grant_type".to_string(), "authorization_code".to_string()),
            ("code".to_string(), code.to_string()),
            ("redirect_uri".to_string(), self.config.redirect_uri.clone()),
            ("client_id".to_string(), self.config.client_id.clone()),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Build the token request body for client credentials flow.
    pub fn build_token_request_client_credentials(&self) -> Vec<(String, String)> {
        let scope_str = self
            .config
            .scopes
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let mut params = vec![
            ("grant_type".to_string(), "client_credentials".to_string()),
            ("client_id".to_string(), self.config.client_id.clone()),
            ("scope".to_string(), scope_str),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Build the token request body for refreshing an access token.
    pub fn build_token_request_refresh(&self, refresh_token: &str) -> Vec<(String, String)> {
        let mut params = vec![
            ("grant_type".to_string(), "refresh_token".to_string()),
            ("refresh_token".to_string(), refresh_token.to_string()),
            ("client_id".to_string(), self.config.client_id.clone()),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Store a token response.
    pub fn set_token(&mut self, token: McpTokenResponse) {
        self.token_expiry_secs = token.expires_in;
        self.current_token = Some(token);
        self.token_obtained_at = Some(std::time::Instant::now());
    }

    /// Get the current access token, if available and not expired.
    pub fn get_access_token(&self) -> Option<&str> {
        if self.is_token_expired() {
            return None;
        }
        self.current_token.as_ref().map(|t| t.access_token.as_str())
    }

    /// Check if the current token has expired.
    pub fn is_token_expired(&self) -> bool {
        match (&self.token_obtained_at, &self.token_expiry_secs) {
            (Some(obtained), Some(expires)) => obtained.elapsed().as_secs() >= *expires,
            (None, _) => true,        // No token obtained
            (Some(_), None) => false, // No expiry = never expires
        }
    }

    /// Generate a PKCE code challenge from a code verifier (S256 method).
    /// Uses SHA-256 + base64url encoding per RFC 7636 §4.2.
    pub fn generate_pkce_challenge(verifier: &str) -> (String, String) {
        use crate::request_signing::sha256;
        let hash = sha256::sha256(verifier.as_bytes());
        // RFC 7636 requires base64url encoding (no padding) of the SHA-256 hash
        let challenge = Self::base64url_encode_no_pad(&hash);
        (challenge, "S256".to_string())
    }

    /// Base64url encode without padding (RFC 4648 §5, used by RFC 7636).
    fn base64url_encode_no_pad(data: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::with_capacity((data.len() * 4 + 2) / 3);
        let chunks = data.chunks(3);
        for chunk in chunks {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(triple & 0x3F) as usize] as char);
            }
        }
        out
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &McpOAuthConfig {
        &self.config
    }
}
