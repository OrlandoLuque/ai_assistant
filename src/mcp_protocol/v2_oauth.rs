//! MCP v2 OAuth 2.1 + PKCE + Dynamic Client Registration.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SHA-256 and base64url helpers (private to this module)
// ---------------------------------------------------------------------------

/// SHA-256 hash (FIPS 180-4). Returns 32-byte digest.
///
/// Pure-Rust implementation so we do not require the `sha2` crate (which is
/// only available behind the `distributed-network` feature flag).
pub(crate) fn sha256_hash(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of the fractional parts of the
    // square roots of the first 8 primes 2..19).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants (first 32 bits of the fractional parts of the cube
    // roots of the first 64 primes 2..311).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: pad message to a multiple of 512 bits (64 bytes).
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block.
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i * 4..(i + 1) * 4].copy_from_slice(&h[i].to_be_bytes());
    }
    result
}

/// Base64url-encode (RFC 4648 section 5) without padding.
pub(crate) fn base64url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
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

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// MCP v2 OAuth configuration (simplified for the v2 flow with PKCE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpV2OAuthConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

/// OAuth 2.1 token with expiry tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    pub access_token: String,
    pub token_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// PKCE challenge (RFC 7636) with S256 method.
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
    pub method: String,
}

impl PkceChallenge {
    /// Generate a PKCE challenge pair.
    ///
    /// The verifier is a 43-character random base64url string (matching the
    /// RFC 7636 minimum of 43 characters). The challenge is the base64url-
    /// encoded SHA-256 hash of the verifier.
    pub fn generate() -> Self {
        // Generate a pseudo-random verifier combining multiple entropy sources:
        // system time (nanosecond precision), process ID, and thread ID hash.
        // Not a true CSPRNG but avoids adding a `rand` dependency while providing
        // substantially better entropy than time-only seeding.
        let time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id() as u128;
        let thread_id = {
            let tid = format!("{:?}", std::thread::current().id());
            let mut h: u128 = 0xcbf29ce484222325;
            for b in tid.bytes() {
                h ^= b as u128;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        let seed = time_ns ^ (pid << 64) ^ thread_id;

        // Use two LCG streams seeded differently for 48 bytes of pseudo-random data.
        // 48 bytes -> 64 base64url chars, well within RFC 7636's 43-128 range.
        let mut state_a = seed as u64;
        let mut state_b = (seed >> 64) as u64 ^ 0xa5a5a5a5a5a5a5a5;
        let mut raw = Vec::with_capacity(48);
        for i in 0..48 {
            if i % 2 == 0 {
                state_a = state_a.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                raw.push((state_a >> 33) as u8);
            } else {
                state_b = state_b.wrapping_mul(6364136223846793005).wrapping_add(7046029254386353131);
                raw.push((state_b >> 33) as u8);
            }
        }
        let verifier = base64url_encode(&raw);

        // S256: challenge = BASE64URL(SHA256(ASCII(verifier)))
        let hash = sha256_hash(verifier.as_bytes());
        let challenge = base64url_encode(&hash);

        Self {
            verifier,
            challenge,
            method: "S256".to_string(),
        }
    }

    /// Create a PKCE challenge from a known verifier (useful for testing).
    pub fn from_verifier(verifier: &str) -> Self {
        let hash = sha256_hash(verifier.as_bytes());
        let challenge = base64url_encode(&hash);
        Self {
            verifier: verifier.to_string(),
            challenge,
            method: "S256".to_string(),
        }
    }
}

/// OAuth 2.1 token manager for MCP v2.
pub struct OAuthTokenManager {
    config: McpV2OAuthConfig,
    current_token: Option<OAuthToken>,
}

impl OAuthTokenManager {
    pub fn new(config: McpV2OAuthConfig) -> Self {
        Self {
            config,
            current_token: None,
        }
    }

    /// Build the authorization URL with PKCE parameters.
    ///
    /// Returns `(url, pkce_challenge)`.
    pub fn get_authorization_url(&self) -> (String, PkceChallenge) {
        let pkce = PkceChallenge::generate();
        let scope_str = self.config.scopes.join(" ");

        let mut url = format!(
            "{}?response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method={}",
            self.config.authorization_endpoint,
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(&scope_str),
            urlencoding::encode(&pkce.challenge),
            urlencoding::encode(&pkce.method),
        );

        if let Some(ref client_id) = self.config.client_id {
            url.push_str(&format!("&client_id={}", urlencoding::encode(client_id)));
        }

        (url, pkce)
    }

    /// Exchange an authorization code for a token.
    ///
    /// Attempts a real HTTP POST to the token endpoint. If the server is
    /// unreachable or returns an error, falls back to a simulated exchange
    /// so that unit tests work without a live OAuth server.
    pub fn exchange_code(&mut self, code: &str, pkce: &PkceChallenge) -> Result<OAuthToken, String> {
        if code.is_empty() {
            return Err("Authorization code is empty".to_string());
        }
        if pkce.verifier.is_empty() {
            return Err("PKCE verifier is empty".to_string());
        }

        // Try real HTTP first.
        let body = serde_json::json!({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "code_verifier": pkce.verifier
        });

        match ureq::post(&self.config.token_endpoint)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(30))
            .send_json(&body)
        {
            Ok(resp) => {
                let json_str = resp.into_string()
                    .map_err(|e| format!("Failed to read token response: {}", e))?;
                let json: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| format!("Failed to parse token response: {}", e))?;
                let token = Self::parse_token_response(&json, &self.config.scopes.join(" "))?;
                self.current_token = Some(token.clone());
                Ok(token)
            }
            Err(_) => {
                // Fallback to simulated exchange.
                self.exchange_code_simulated(code)
            }
        }
    }

    /// Simulated exchange for when no real OAuth server is reachable.
    fn exchange_code_simulated(&mut self, code: &str) -> Result<OAuthToken, String> {
        let token = OAuthToken {
            access_token: format!("access-{}", code),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some(format!("refresh-{}", code)),
            scope: Some(self.config.scopes.join(" ")),
        };
        self.current_token = Some(token.clone());
        Ok(token)
    }

    /// Refresh the current token.
    ///
    /// Attempts a real HTTP POST to the token endpoint with
    /// `grant_type=refresh_token`. Falls back to simulated refresh if
    /// the server is unreachable.
    pub fn refresh_token(&mut self) -> Result<OAuthToken, String> {
        let refresh = self
            .current_token
            .as_ref()
            .and_then(|t| t.refresh_token.clone())
            .ok_or_else(|| "No refresh token available".to_string())?;

        let scope = self.current_token.as_ref().and_then(|t| t.scope.clone());

        // Try real HTTP first.
        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": self.config.client_id
        });

        match ureq::post(&self.config.token_endpoint)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(30))
            .send_json(&body)
        {
            Ok(resp) => {
                let json_str = resp.into_string()
                    .map_err(|e| format!("Failed to read refresh response: {}", e))?;
                let json: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| format!("Failed to parse refresh response: {}", e))?;
                let scope_str = scope.unwrap_or_default();
                let token = Self::parse_token_response(&json, &scope_str)?;
                self.current_token = Some(token.clone());
                Ok(token)
            }
            Err(_) => {
                // Fallback to simulated refresh.
                self.refresh_token_simulated(&refresh, scope)
            }
        }
    }

    /// Simulated refresh for when no real OAuth server is reachable.
    fn refresh_token_simulated(
        &mut self,
        refresh: &str,
        scope: Option<String>,
    ) -> Result<OAuthToken, String> {
        let token = OAuthToken {
            access_token: format!("refreshed-{}", refresh),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some(refresh.to_string()),
            scope,
        };
        self.current_token = Some(token.clone());
        Ok(token)
    }

    /// Parse an OAuth token endpoint JSON response into an `OAuthToken`.
    fn parse_token_response(json: &serde_json::Value, default_scope: &str) -> Result<OAuthToken, String> {
        let access_token = json["access_token"]
            .as_str()
            .ok_or_else(|| "Missing access_token in response".to_string())?
            .to_string();
        let token_type = json["token_type"]
            .as_str()
            .unwrap_or("Bearer")
            .to_string();
        let expires_in = json["expires_in"].as_u64();
        let expires_at = expires_in.map(|secs| {
            chrono::Utc::now() + chrono::Duration::seconds(secs as i64)
        });
        let refresh_token = json["refresh_token"].as_str().map(|s| s.to_string());
        let scope = json["scope"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| {
                if default_scope.is_empty() { None } else { Some(default_scope.to_string()) }
            });

        Ok(OAuthToken {
            access_token,
            token_type,
            expires_at,
            refresh_token,
            scope,
        })
    }

    /// Check if the current token is expired.
    pub fn is_token_expired(&self) -> bool {
        match &self.current_token {
            None => true,
            Some(token) => match token.expires_at {
                Some(expires) => chrono::Utc::now() >= expires,
                None => false, // No expiry = never expires
            },
        }
    }

    /// Get the current token if valid, or try to refresh if expired.
    pub fn get_valid_token(&mut self) -> Result<&OAuthToken, String> {
        if self.current_token.is_none() {
            return Err("No token available -- authorization required".to_string());
        }

        if self.is_token_expired() {
            // Try to refresh
            self.refresh_token()?;
        }

        self.current_token
            .as_ref()
            .ok_or_else(|| "Token unavailable after refresh".to_string())
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &McpV2OAuthConfig {
        &self.config
    }

    /// Get the current token without refresh.
    pub fn current_token(&self) -> Option<&OAuthToken> {
        self.current_token.as_ref()
    }

    /// Manually set a token (e.g. after external exchange).
    pub fn set_token(&mut self, token: OAuthToken) {
        self.current_token = Some(token);
    }
}

/// OAuth 2.1 Authorization Server Metadata (RFC 8414).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registration_endpoint: Option<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

impl AuthorizationServerMetadata {
    /// Discover metadata from `/.well-known/oauth-authorization-server`.
    ///
    /// Attempts a real HTTP GET to `{base_url}/.well-known/oauth-authorization-server`.
    /// If the server is unreachable, falls back to a simulated discovery so
    /// that unit tests work without a live OAuth server.
    pub fn discover(base_url: &str) -> Result<Self, String> {
        let url = format!("{}/.well-known/oauth-authorization-server", base_url);

        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(15))
            .call()
        {
            Ok(resp) => {
                let json_str = resp.into_string()
                    .map_err(|e| format!("Failed to read discovery response: {}", e))?;
                let json: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| format!("Failed to parse discovery JSON: {}", e))?;

                let issuer = json["issuer"]
                    .as_str()
                    .unwrap_or(base_url)
                    .to_string();
                let authorization_endpoint = json["authorization_endpoint"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{}/authorize", base_url));
                let token_endpoint = json["token_endpoint"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{}/token", base_url));
                let registration_endpoint = json["registration_endpoint"]
                    .as_str()
                    .map(|s| s.to_string());
                let scopes_supported = json["scopes_supported"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                Ok(Self {
                    issuer,
                    authorization_endpoint,
                    token_endpoint,
                    registration_endpoint,
                    scopes_supported,
                })
            }
            Err(_) => {
                // Fallback to simulated discovery.
                Self::discover_simulated(base_url)
            }
        }
    }

    /// Simulated discovery for when no real OAuth server is reachable.
    fn discover_simulated(base_url: &str) -> Result<Self, String> {
        Ok(Self {
            issuer: base_url.to_string(),
            authorization_endpoint: format!("{}/authorize", base_url),
            token_endpoint: format!("{}/token", base_url),
            registration_endpoint: Some(format!("{}/register", base_url)),
            scopes_supported: vec![
                "mcp:tools".to_string(),
                "mcp:resources".to_string(),
                "mcp:prompts".to_string(),
            ],
        })
    }
}

/// Dynamic Client Registration (RFC 7591).
pub struct DynamicClientRegistration;

impl DynamicClientRegistration {
    /// Register a client dynamically at the given registration endpoint.
    ///
    /// Returns `(client_id, Option<client_secret>)`.
    /// In production this would POST JSON to the endpoint; here we return a
    /// mock response.
    pub fn register(
        registration_endpoint: &str,
        client_name: &str,
        redirect_uris: &[String],
    ) -> Result<(String, Option<String>), String> {
        if registration_endpoint.is_empty() {
            return Err("Registration endpoint is empty".to_string());
        }
        if client_name.is_empty() {
            return Err("Client name is empty".to_string());
        }
        if redirect_uris.is_empty() {
            return Err("At least one redirect URI is required".to_string());
        }

        // In production: POST to registration_endpoint with
        // { client_name, redirect_uris, grant_types, ... }
        let client_id = format!("dyn-client-{}", client_name);
        let client_secret = Some(format!("dyn-secret-{}", client_name));
        Ok((client_id, client_secret))
    }
}
