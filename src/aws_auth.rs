// aws_auth.rs — AWS Signature V4 authentication and Bedrock provider.
//
// Implements AWS SigV4 request signing from scratch using hmac + sha2.
// Used by the Bedrock provider for Claude-on-AWS and other foundation models.

#[cfg(feature = "aws-bedrock")]
use hmac::{Hmac, Mac};
#[cfg(feature = "aws-bedrock")]
use sha2::{Digest, Sha256};

use anyhow::{Context, Result};

// ============================================================================
// AWS Credentials
// ============================================================================

/// AWS credentials for Signature V4 signing.
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl AwsCredentials {
    pub fn new(access_key_id: String, secret_access_key: String) -> Self {
        Self {
            access_key_id,
            secret_access_key,
            session_token: None,
        }
    }

    pub fn with_session_token(mut self, token: String) -> Self {
        self.session_token = Some(token);
        self
    }

    /// Resolve credentials from environment variables.
    /// Checks AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_SESSION_TOKEN.
    pub fn from_env() -> Result<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").context("AWS_ACCESS_KEY_ID not set")?;
        let secret_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").context("AWS_SECRET_ACCESS_KEY not set")?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(Self {
            access_key_id: access_key,
            secret_access_key: secret_key,
            session_token,
        })
    }
}

// ============================================================================
// SigV4 Signing (behind feature flag)
// ============================================================================

/// A signed HTTP request ready to send.
#[derive(Debug, Clone)]
pub struct SignedRequest {
    pub url: String,
    pub method: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Parameters for SigV4 signing.
#[derive(Debug, Clone)]
pub struct SigV4Params {
    pub method: String,
    pub url: String,
    pub region: String,
    pub service: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[cfg(feature = "aws-bedrock")]
pub fn sign_request(params: &SigV4Params, credentials: &AwsCredentials) -> Result<SignedRequest> {
    // Implementation of AWS Signature Version 4
    let now = chrono_free_now();
    let date_stamp = &now[..8]; // YYYYMMDD
    let amz_date = &now; // YYYYMMDD'T'HHMMSS'Z'

    // Parse URL to get host and path
    let (host, path, query) = parse_url(&params.url)?;

    // Step 1: Create canonical request
    let payload_hash = hex_sha256(&params.body);

    let mut headers_to_sign = params.headers.clone();
    headers_to_sign.push(("host".to_string(), host.clone()));
    headers_to_sign.push(("x-amz-date".to_string(), amz_date.to_string()));
    headers_to_sign.push(("x-amz-content-sha256".to_string(), payload_hash.clone()));
    if let Some(ref token) = credentials.session_token {
        headers_to_sign.push(("x-amz-security-token".to_string(), token.clone()));
    }

    // Sort headers by lowercase name
    headers_to_sign.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));

    let canonical_headers: String = headers_to_sign
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k.to_lowercase(), v.trim()))
        .collect();

    let signed_headers: String = headers_to_sign
        .iter()
        .map(|(k, _)| k.to_lowercase())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        params.method,
        uri_encode_path(&path),
        query.as_deref().unwrap_or(""),
        canonical_headers,
        signed_headers,
        payload_hash,
    );

    // Step 2: Create string to sign
    let credential_scope = format!(
        "{}/{}/{}/aws4_request",
        date_stamp, params.region, params.service
    );
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        hex_sha256(canonical_request.as_bytes()),
    );

    // Step 3: Calculate signature
    let signing_key = derive_signing_key(
        &credentials.secret_access_key,
        date_stamp,
        &params.region,
        &params.service,
    );
    let signature = hex_hmac_sha256(&signing_key, string_to_sign.as_bytes());

    // Step 4: Build Authorization header
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key_id, credential_scope, signed_headers, signature,
    );

    let mut final_headers = headers_to_sign;
    final_headers.push(("authorization".to_string(), authorization));
    final_headers.push(("content-type".to_string(), "application/json".to_string()));

    Ok(SignedRequest {
        url: params.url.clone(),
        method: params.method.clone(),
        headers: final_headers,
        body: params.body.clone(),
    })
}

#[cfg(not(feature = "aws-bedrock"))]
pub fn sign_request(_params: &SigV4Params, _credentials: &AwsCredentials) -> Result<SignedRequest> {
    anyhow::bail!("AWS SigV4 signing requires the `aws-bedrock` feature flag")
}

// ============================================================================
// Bedrock API
// ============================================================================

/// Bedrock model request configuration.
#[derive(Debug, Clone)]
pub struct BedrockRequest {
    pub model_id: String,
    pub region: String,
    pub messages: Vec<BedrockMessage>,
    pub system_prompt: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct BedrockMessage {
    pub role: String,
    pub content: String,
}

/// Generate a response from AWS Bedrock.
///
/// Supports Claude models on Bedrock using the Messages API format.
#[cfg(feature = "aws-bedrock")]
pub fn generate_bedrock_response(
    request: &BedrockRequest,
    credentials: &AwsCredentials,
) -> Result<String> {
    let url = format!(
        "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
        request.region, request.model_id
    );

    // Build Anthropic Messages API format (used by Claude on Bedrock)
    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    });

    if let Some(ref system) = request.system_prompt {
        body["system"] = serde_json::json!(system);
    }

    let body_bytes = serde_json::to_vec(&body)?;

    let sig_params = SigV4Params {
        method: "POST".to_string(),
        url: url.clone(),
        region: request.region.clone(),
        service: "bedrock".to_string(),
        headers: vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("accept".to_string(), "application/json".to_string()),
        ],
        body: body_bytes.clone(),
    };

    let signed = sign_request(&sig_params, credentials)?;

    let mut req = ureq::post(&signed.url);
    for (key, value) in &signed.headers {
        req = req.set(key, value);
    }

    let response = req
        .timeout(std::time::Duration::from_secs(120))
        .send_bytes(&signed.body)
        .context("Bedrock API request failed")?;

    let json: serde_json::Value = response
        .into_json()
        .context("Failed to parse Bedrock response")?;

    // Parse Anthropic Messages API response format
    json.get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| {
            arr.iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .next()
        })
        .ok_or_else(|| anyhow::anyhow!("Unexpected Bedrock response format"))
}

#[cfg(not(feature = "aws-bedrock"))]
pub fn generate_bedrock_response(
    _request: &BedrockRequest,
    _credentials: &AwsCredentials,
) -> Result<String> {
    anyhow::bail!("AWS Bedrock requires the `aws-bedrock` feature flag")
}

/// Fetch available Bedrock model IDs.
pub fn fetch_bedrock_models() -> Vec<String> {
    vec![
        "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
        "anthropic.claude-3-opus-20240229-v1:0".to_string(),
        "amazon.titan-text-express-v1".to_string(),
        "amazon.titan-text-premier-v1:0".to_string(),
        "meta.llama3-1-70b-instruct-v1:0".to_string(),
        "meta.llama3-1-8b-instruct-v1:0".to_string(),
        "mistral.mistral-large-2407-v1:0".to_string(),
    ]
}

// ============================================================================
// Helper functions
// ============================================================================

#[cfg(feature = "aws-bedrock")]
fn hex_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex_encode(&hasher.finalize())
}

#[cfg(feature = "aws-bedrock")]
type HmacSha256 = Hmac<Sha256>;

#[cfg(feature = "aws-bedrock")]
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

#[cfg(feature = "aws-bedrock")]
fn hex_hmac_sha256(key: &[u8], data: &[u8]) -> String {
    hex_encode(&hmac_sha256(key, data))
}

#[cfg(feature = "aws-bedrock")]
fn derive_signing_key(secret: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{}", secret).as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

#[cfg(any(feature = "aws-bedrock", test))]
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn chrono_free_now() -> String {
        let dur = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let total_secs = dur.as_secs();
        let secs_in_day = total_secs % 86400;
        let days = total_secs / 86400;
        let hours = secs_in_day / 3600;
        let minutes = (secs_in_day % 3600) / 60;
        let seconds = secs_in_day % 60;
        let (y, m, d) = days_to_ymd(days);
        format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
            y, m, d, hours, minutes, seconds
        )
    }

    fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
        // Civil calendar algorithm from Howard Hinnant
        days += 719468;
        let era = days / 146097;
        let doe = days - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        (y, m, d)
    }

    fn parse_url(url: &str) -> std::result::Result<(String, String, Option<String>), String> {
        let stripped = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .ok_or_else(|| "URL must start with http(s)://".to_string())?;
        let (host, rest) = stripped
            .split_once('/')
            .unwrap_or((stripped, ""));
        let (path, query) = if let Some((p, q)) = rest.split_once('?') {
            (format!("/{}", p), Some(q.to_string()))
        } else if rest.is_empty() {
            ("/".to_string(), None)
        } else {
            (format!("/{}", rest), None)
        };
        Ok((host.to_string(), path, query))
    }

    fn uri_encode_path(path: &str) -> String {
        path.split('/')
            .map(|segment| {
                segment
                    .bytes()
                    .map(|b| {
                        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~'
                        {
                            (b as char).to_string()
                        } else {
                            format!("%{:02X}", b)
                        }
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn test_aws_credentials_new() {
        let creds = AwsCredentials::new("AKIAEXAMPLE".to_string(), "wJalrXUtnFEMI".to_string());
        assert_eq!(creds.access_key_id, "AKIAEXAMPLE");
        assert_eq!(creds.secret_access_key, "wJalrXUtnFEMI");
        assert!(creds.session_token.is_none());
    }

    #[test]
    fn test_aws_credentials_with_session_token() {
        let creds = AwsCredentials::new("AKIAEXAMPLE".to_string(), "secret".to_string())
            .with_session_token("token123".to_string());
        assert_eq!(creds.session_token.as_deref(), Some("token123"));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0x0a, 0xbc]), "00ff0abc");
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x48, 0x65, 0x6c, 0x6c, 0x6f]), "48656c6c6f");
    }

    #[test]
    fn test_chrono_free_now_format() {
        let ts = chrono_free_now();
        // Should be in format YYYYMMDDTHHMMSSZ
        assert_eq!(ts.len(), 16);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[8..9], "T");
        // Year should be reasonable
        let year: u64 = ts[..4].parse().unwrap();
        assert!(year >= 2024 && year <= 2100);
    }

    #[test]
    fn test_days_to_ymd_epoch() {
        // 1970-01-01
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_known_date() {
        // 2024-01-01 = 19723 days since epoch
        let (y, m, d) = days_to_ymd(19723);
        assert_eq!((y, m, d), (2024, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_leap_year() {
        // 2024-02-29 = 19723 + 31 + 28 = 19782 (2024 is leap year, so Feb has 29 days)
        let (y, m, d) = days_to_ymd(19782);
        assert_eq!(y, 2024);
        assert_eq!(m, 2);
        assert_eq!(d, 29);
    }

    #[test]
    fn test_parse_url_basic() {
        let (host, path, query) = parse_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3/invoke",
        )
        .unwrap();
        assert_eq!(host, "bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(path, "/model/anthropic.claude-3/invoke");
        assert!(query.is_none());
    }

    #[test]
    fn test_parse_url_with_query() {
        let (host, path, query) = parse_url("https://example.com/path?key=value&foo=bar").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(path, "/path");
        assert_eq!(query.as_deref(), Some("key=value&foo=bar"));
    }

    #[test]
    fn test_parse_url_no_path() {
        let (host, path, _) = parse_url("https://example.com").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(path, "/");
    }

    #[test]
    fn test_parse_url_invalid_scheme() {
        assert!(parse_url("ftp://example.com").is_err());
    }

    #[test]
    fn test_uri_encode_path() {
        assert_eq!(uri_encode_path("/model/invoke"), "/model/invoke");
        assert_eq!(
            uri_encode_path("/path with spaces"),
            "/path%20with%20spaces"
        );
        assert_eq!(uri_encode_path("/"), "/");
    }

    #[test]
    fn test_fetch_bedrock_models() {
        let models = fetch_bedrock_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("claude")));
        assert!(models.iter().any(|m| m.contains("titan")));
        assert!(models.iter().any(|m| m.contains("llama")));
    }

    #[test]
    fn test_bedrock_request_creation() {
        let req = BedrockRequest {
            model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            region: "us-east-1".to_string(),
            messages: vec![BedrockMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            system_prompt: Some("You are helpful.".to_string()),
            max_tokens: 4096,
            temperature: 0.7,
        };
        assert_eq!(req.region, "us-east-1");
        assert_eq!(req.messages.len(), 1);
        assert!(req.system_prompt.is_some());
    }

    #[cfg(feature = "aws-bedrock")]
    #[test]
    fn test_hex_sha256() {
        // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[cfg(feature = "aws-bedrock")]
    #[test]
    fn test_derive_signing_key() {
        // Known test: derive key for a specific date/region/service
        let key = derive_signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            "iam",
        );
        assert_eq!(key.len(), 32); // HMAC-SHA256 output is 32 bytes
    }

    #[cfg(feature = "aws-bedrock")]
    #[test]
    fn test_sign_request_produces_authorization() {
        let creds = AwsCredentials::new(
            "AKIDEXAMPLE".to_string(),
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
        );
        let params = SigV4Params {
            method: "POST".to_string(),
            url: "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke".to_string(),
            region: "us-east-1".to_string(),
            service: "bedrock".to_string(),
            headers: vec![],
            body: b"{}".to_vec(),
        };
        let signed = sign_request(&params, &creds).unwrap();
        // Should have an authorization header
        let auth = signed.headers.iter().find(|(k, _)| k == "authorization");
        assert!(auth.is_some());
        let auth_value = &auth.unwrap().1;
        assert!(auth_value.starts_with("AWS4-HMAC-SHA256"));
        assert!(auth_value.contains("AKIDEXAMPLE"));
        assert!(auth_value.contains("us-east-1/bedrock/aws4_request"));
    }

    #[cfg(not(feature = "aws-bedrock"))]
    #[test]
    fn test_sign_request_without_feature() {
        let creds = AwsCredentials::new("AKID".to_string(), "secret".to_string());
        let params = SigV4Params {
            method: "POST".to_string(),
            url: "https://example.com".to_string(),
            region: "us-east-1".to_string(),
            service: "bedrock".to_string(),
            headers: vec![],
            body: vec![],
        };
        let result = sign_request(&params, &creds);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("aws-bedrock"));
    }

    #[cfg(not(feature = "aws-bedrock"))]
    #[test]
    fn test_generate_bedrock_without_feature() {
        let creds = AwsCredentials::new("AKID".to_string(), "secret".to_string());
        let req = BedrockRequest {
            model_id: "anthropic.claude-3".to_string(),
            region: "us-east-1".to_string(),
            messages: vec![],
            system_prompt: None,
            max_tokens: 4096,
            temperature: 0.7,
        };
        let result = generate_bedrock_response(&req, &creds);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("aws-bedrock"));
    }
}
