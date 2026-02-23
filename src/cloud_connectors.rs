// cloud_connectors.rs — Cloud storage connectors for S3 and Google Drive.
//
// Provides a unified CloudStorage trait with implementations for:
// - Amazon S3 (using AWS SigV4 authentication from aws_auth module)
// - Google Drive (using OAuth2 Bearer token with REST API v3)
//
// Both implementations use ureq for HTTP requests.

use anyhow::{Context, Result};
use std::collections::HashMap;

// ============================================================================
// CloudStorage Trait
// ============================================================================

/// Metadata for a cloud storage object.
#[derive(Debug, Clone)]
pub struct CloudObject {
    /// Object key/path
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Last modified timestamp (Unix seconds)
    pub last_modified: u64,
    /// Content type (MIME)
    pub content_type: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Options for listing objects.
#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    /// Prefix filter (e.g., "folder/")
    pub prefix: Option<String>,
    /// Maximum number of results
    pub max_results: Option<usize>,
    /// Continuation token for pagination
    pub page_token: Option<String>,
}

/// Result of a list operation.
#[derive(Debug, Clone)]
pub struct ListResult {
    /// Objects matching the query
    pub objects: Vec<CloudObject>,
    /// Token for the next page, if any
    pub next_page_token: Option<String>,
}

/// Unified cloud storage interface.
pub trait CloudStorage {
    /// List objects in the storage.
    fn list(&self, options: &ListOptions) -> Result<ListResult>;

    /// Get an object's content.
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Upload/put an object.
    fn put(&self, key: &str, data: &[u8], content_type: Option<&str>) -> Result<()>;

    /// Delete an object.
    fn delete(&self, key: &str) -> Result<()>;

    /// Check if an object exists.
    fn exists(&self, key: &str) -> Result<bool>;

    /// Get the storage provider name.
    fn provider_name(&self) -> &str;
}

// ============================================================================
// S3 Client
// ============================================================================

/// Amazon S3 storage client configuration.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// AWS region
    pub region: String,
    /// AWS access key ID
    pub access_key_id: String,
    /// AWS secret access key
    pub secret_access_key: String,
    /// Optional session token (for temporary credentials)
    pub session_token: Option<String>,
    /// Custom endpoint URL (for S3-compatible services like MinIO)
    pub endpoint: Option<String>,
    /// Use path-style addressing (required for some S3-compatible services)
    pub path_style: bool,
}

impl S3Config {
    /// Create config for a standard AWS S3 bucket.
    pub fn new(bucket: &str, region: &str, access_key_id: &str, secret_access_key: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: region.to_string(),
            access_key_id: access_key_id.to_string(),
            secret_access_key: secret_access_key.to_string(),
            session_token: None,
            endpoint: None,
            path_style: false,
        }
    }

    /// Create config for an S3-compatible service (MinIO, DigitalOcean Spaces, etc.)
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self.path_style = true;
        self
    }

    /// Get the base URL for S3 operations.
    pub fn base_url(&self) -> String {
        if let Some(ref endpoint) = self.endpoint {
            if self.path_style {
                format!("{}/{}", endpoint.trim_end_matches('/'), self.bucket)
            } else {
                endpoint.clone()
            }
        } else if self.path_style {
            format!("https://s3.{}.amazonaws.com/{}", self.region, self.bucket)
        } else {
            format!("https://{}.s3.{}.amazonaws.com", self.bucket, self.region)
        }
    }

    /// Get the hostname for S3 operations.
    pub fn host(&self) -> String {
        if let Some(ref endpoint) = self.endpoint {
            // Extract host from endpoint URL
            endpoint
                .strip_prefix("https://")
                .or_else(|| endpoint.strip_prefix("http://"))
                .unwrap_or(endpoint)
                .split('/')
                .next()
                .unwrap_or("localhost")
                .to_string()
        } else if self.path_style {
            format!("s3.{}.amazonaws.com", self.region)
        } else {
            format!("{}.s3.{}.amazonaws.com", self.bucket, self.region)
        }
    }
}

/// Amazon S3 storage client.
pub struct S3Client {
    config: S3Config,
}

impl S3Client {
    pub fn new(config: S3Config) -> Self {
        Self { config }
    }

    /// Build the URL for an S3 object key.
    pub fn object_url(&self, key: &str) -> String {
        format!("{}/{}", self.config.base_url(), key.trim_start_matches('/'))
    }

    /// Get the bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get the region.
    pub fn region(&self) -> &str {
        &self.config.region
    }

    /// Return true if credentials are available (non-empty access key and secret key).
    fn has_credentials(&self) -> bool {
        !self.config.access_key_id.is_empty() && !self.config.secret_access_key.is_empty()
    }

    /// Create a signer from the current config.
    fn signer(&self) -> AwsSigV4Signer {
        AwsSigV4Signer::new(
            &self.config.access_key_id,
            &self.config.secret_access_key,
            &self.config.region,
        )
    }

    /// Execute a signed GET request.
    fn signed_get(&self, url: &str) -> Result<ureq::Response> {
        if self.has_credentials() {
            let signer = self.signer();
            let auth_headers = signer.sign_request("GET", url, &[], &[]);
            let mut req = ureq::get(url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(30));
            for (k, v) in &auth_headers {
                req = req.set(k, v);
            }
            req.call().context("S3 signed GET failed")
        } else {
            ureq::get(url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(30))
                .call()
                .context("S3 GET failed")
        }
    }

    /// Execute a signed PUT request.
    fn signed_put(&self, url: &str, data: &[u8], content_type: &str) -> Result<ureq::Response> {
        if self.has_credentials() {
            let signer = self.signer();
            let extra = [("Content-Type", content_type)];
            let auth_headers = signer.sign_request("PUT", url, &extra, data);
            let mut req = ureq::put(url)
                .set("Host", &self.config.host())
                .set("Content-Type", content_type)
                .timeout(std::time::Duration::from_secs(60));
            for (k, v) in &auth_headers {
                req = req.set(k, v);
            }
            req.send_bytes(data).context("S3 signed PUT failed")
        } else {
            ureq::put(url)
                .set("Host", &self.config.host())
                .set("Content-Type", content_type)
                .timeout(std::time::Duration::from_secs(60))
                .send_bytes(data)
                .context("S3 PUT failed")
        }
    }

    /// Execute a signed DELETE request.
    fn signed_delete(&self, url: &str) -> Result<ureq::Response> {
        if self.has_credentials() {
            let signer = self.signer();
            let auth_headers = signer.sign_request("DELETE", url, &[], &[]);
            let mut req = ureq::request("DELETE", url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(30));
            for (k, v) in &auth_headers {
                req = req.set(k, v);
            }
            req.call().context("S3 signed DELETE failed")
        } else {
            ureq::request("DELETE", url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(30))
                .call()
                .context("S3 DELETE failed")
        }
    }

    /// Execute a signed HEAD request.
    #[allow(clippy::result_large_err)]
    fn signed_head(&self, url: &str) -> std::result::Result<ureq::Response, ureq::Error> {
        if self.has_credentials() {
            let signer = self.signer();
            let auth_headers = signer.sign_request("HEAD", url, &[], &[]);
            let mut req = ureq::request("HEAD", url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(10));
            for (k, v) in &auth_headers {
                req = req.set(k, v);
            }
            req.call()
        } else {
            ureq::request("HEAD", url)
                .set("Host", &self.config.host())
                .timeout(std::time::Duration::from_secs(10))
                .call()
        }
    }
}

impl CloudStorage for S3Client {
    fn list(&self, options: &ListOptions) -> Result<ListResult> {
        let mut url = format!("{}?list-type=2", self.config.base_url());
        if let Some(ref prefix) = options.prefix {
            url.push_str(&format!("&prefix={}", prefix));
        }
        if let Some(max) = options.max_results {
            url.push_str(&format!("&max-keys={}", max));
        }
        if let Some(ref token) = options.page_token {
            url.push_str(&format!("&continuation-token={}", token));
        }

        let resp = self.signed_get(&url)?;

        let body = resp
            .into_string()
            .context("Failed to read S3 list response")?;

        // Parse XML response (simplified)
        let objects = parse_s3_list_xml(&body);
        let next_token = extract_xml_value(&body, "NextContinuationToken");

        Ok(ListResult {
            objects,
            next_page_token: next_token,
        })
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let url = self.object_url(key);
        let resp = self.signed_get(&url)?;

        let mut bytes = Vec::new();
        resp.into_reader()
            .read_to_end(&mut bytes)
            .context("Failed to read S3 object")?;
        Ok(bytes)
    }

    fn put(&self, key: &str, data: &[u8], content_type: Option<&str>) -> Result<()> {
        let url = self.object_url(key);
        let ct = content_type.unwrap_or("application/octet-stream");
        self.signed_put(&url, data, ct)?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let url = self.object_url(key);
        self.signed_delete(&url)?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let url = self.object_url(key);
        match self.signed_head(&url) {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(404, _)) => Ok(false),
            Err(e) => Err(anyhow::anyhow!("S3 HEAD failed: {}", e)),
        }
    }

    fn provider_name(&self) -> &str {
        "Amazon S3"
    }
}

// ============================================================================
// Google Drive Client
// ============================================================================

/// Google Drive storage client configuration.
#[derive(Debug, Clone)]
pub struct GoogleDriveConfig {
    /// OAuth2 access token
    pub access_token: String,
    /// Root folder ID (default: "root")
    pub root_folder_id: String,
}

impl GoogleDriveConfig {
    pub fn new(access_token: &str) -> Self {
        Self {
            access_token: access_token.to_string(),
            root_folder_id: "root".to_string(),
        }
    }

    pub fn with_folder(mut self, folder_id: &str) -> Self {
        self.root_folder_id = folder_id.to_string();
        self
    }
}

/// Google Drive storage client.
pub struct GoogleDriveClient {
    config: GoogleDriveConfig,
}

impl GoogleDriveClient {
    pub fn new(config: GoogleDriveConfig) -> Self {
        Self { config }
    }

    /// Build the API URL for Drive v3.
    pub fn api_url(&self, path: &str) -> String {
        format!("https://www.googleapis.com/drive/v3{}", path)
    }

    /// Build the upload URL for Drive v3.
    pub fn upload_url(&self) -> String {
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart".to_string()
    }

    /// Get the root folder ID.
    pub fn root_folder(&self) -> &str {
        &self.config.root_folder_id
    }
}

impl CloudStorage for GoogleDriveClient {
    fn list(&self, options: &ListOptions) -> Result<ListResult> {
        let mut query_parts = Vec::new();
        query_parts.push(format!("'{}' in parents", self.config.root_folder_id));

        if let Some(ref prefix) = options.prefix {
            query_parts.push(format!("name contains '{}'", prefix));
        }

        let q = query_parts.join(" and ");
        let mut url = format!(
            "{}?q={}&fields=files(id,name,size,modifiedTime,mimeType)",
            self.api_url("/files"),
            urlencoded(&q)
        );
        if let Some(max) = options.max_results {
            url.push_str(&format!("&pageSize={}", max));
        }
        if let Some(ref token) = options.page_token {
            url.push_str(&format!("&pageToken={}", token));
        }

        let resp = ureq::get(&url)
            .set(
                "Authorization",
                &format!("Bearer {}", self.config.access_token),
            )
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .context("Google Drive list failed")?;

        let json: serde_json::Value = resp.into_json().context("Failed to parse Drive response")?;

        let mut objects = Vec::new();
        if let Some(files) = json.get("files").and_then(|f| f.as_array()) {
            for file in files {
                objects.push(CloudObject {
                    key: file
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string(),
                    size: file
                        .get("size")
                        .and_then(|s| s.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0),
                    last_modified: 0, // Would need to parse ISO 8601
                    content_type: file
                        .get("mimeType")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string()),
                    metadata: {
                        let mut m = HashMap::new();
                        if let Some(id) = file.get("id").and_then(|i| i.as_str()) {
                            m.insert("drive_id".to_string(), id.to_string());
                        }
                        m
                    },
                });
            }
        }

        let next_token = json
            .get("nextPageToken")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());

        Ok(ListResult {
            objects,
            next_page_token: next_token,
        })
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        // For Google Drive, 'key' is the file ID
        let url = format!("{}?alt=media", self.api_url(&format!("/files/{}", key)));
        let resp = ureq::get(&url)
            .set(
                "Authorization",
                &format!("Bearer {}", self.config.access_token),
            )
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .context("Google Drive GET failed")?;

        let mut bytes = Vec::new();
        resp.into_reader()
            .read_to_end(&mut bytes)
            .context("Failed to read Drive file")?;
        Ok(bytes)
    }

    fn put(&self, key: &str, data: &[u8], content_type: Option<&str>) -> Result<()> {
        let ct = content_type.unwrap_or("application/octet-stream");
        let metadata = serde_json::json!({
            "name": key,
            "parents": [self.config.root_folder_id]
        });

        // Multipart upload: metadata + file content
        let boundary = "ai_assistant_boundary_12345";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
        body.extend_from_slice(serde_json::to_string(&metadata)?.as_bytes());
        body.extend_from_slice(format!("\r\n--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(format!("Content-Type: {}\r\n\r\n", ct).as_bytes());
        body.extend_from_slice(data);
        body.extend_from_slice(format!("\r\n--{}--", boundary).as_bytes());

        ureq::post(&self.upload_url())
            .set(
                "Authorization",
                &format!("Bearer {}", self.config.access_token),
            )
            .set(
                "Content-Type",
                &format!("multipart/related; boundary={}", boundary),
            )
            .timeout(std::time::Duration::from_secs(60))
            .send_bytes(&body)
            .context("Google Drive upload failed")?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let url = self.api_url(&format!("/files/{}", key));
        ureq::request("DELETE", &url)
            .set(
                "Authorization",
                &format!("Bearer {}", self.config.access_token),
            )
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .context("Google Drive delete failed")?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let url = format!("{}?fields=id", self.api_url(&format!("/files/{}", key)));
        match ureq::get(&url)
            .set(
                "Authorization",
                &format!("Bearer {}", self.config.access_token),
            )
            .timeout(std::time::Duration::from_secs(10))
            .call()
        {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(404, _)) => Ok(false),
            Err(e) => Err(anyhow::anyhow!("Google Drive check failed: {}", e)),
        }
    }

    fn provider_name(&self) -> &str {
        "Google Drive"
    }
}

// ============================================================================
// SHA-256 Implementation (FIPS 180-4)
// ============================================================================

/// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes).
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// Initial hash values for SHA-256 (first 32 bits of fractional parts of square roots of first 8 primes).
const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

/// Minimal SHA-256 implementation for AWS SigV4 signing.
/// Based on FIPS 180-4 specification.
fn sha256(data: &[u8]) -> [u8; 32] {
    // Pre-processing: pad the message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();

    // Append bit '1' (byte 0x80)
    msg.push(0x80);

    // Append zeros until message length is 56 mod 64
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }

    // Append original length in bits as 64-bit big-endian
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    let mut h = SHA256_H0;

    for chunk in msg.chunks_exact(64) {
        // Create message schedule W[0..63]
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
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

        // Initialize working variables
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        // Compression function
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
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

        // Add the compressed chunk to the current hash value
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce the final hash value (big-endian)
    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    result
}

/// SHA-256 hash as lowercase hex string.
fn sha256_hex(data: &[u8]) -> String {
    let hash = sha256(data);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// HMAC-SHA256 using standard HMAC construction: H((K XOR opad) || H((K XOR ipad) || message))
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 64;

    // Step 1: If key is longer than block size, hash it
    let key_block = if key.len() > BLOCK_SIZE {
        sha256(key).to_vec()
    } else {
        key.to_vec()
    };

    // Step 2: Pad key to block size
    let mut k_padded = [0u8; BLOCK_SIZE];
    k_padded[..key_block.len()].copy_from_slice(&key_block);

    // Step 3: Create inner and outer padded keys
    let mut i_key_pad = vec![0u8; BLOCK_SIZE];
    let mut o_key_pad = vec![0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        i_key_pad[i] = k_padded[i] ^ 0x36;
        o_key_pad[i] = k_padded[i] ^ 0x5c;
    }

    // Step 4: Inner hash = SHA-256(i_key_pad || data)
    let mut inner_msg = i_key_pad;
    inner_msg.extend_from_slice(data);
    let inner_hash = sha256(&inner_msg);

    // Step 5: Outer hash = SHA-256(o_key_pad || inner_hash)
    let mut outer_msg = o_key_pad;
    outer_msg.extend_from_slice(&inner_hash);
    sha256(&outer_msg).to_vec()
}

/// HMAC-SHA256 result as lowercase hex string.
fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    hmac_sha256(key, data)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

// ============================================================================
// AWS Signature Version 4 Signer
// ============================================================================

/// AWS Signature Version 4 signer for S3 requests.
struct AwsSigV4Signer {
    access_key: String,
    secret_key: String,
    region: String,
    service: String,
}

impl AwsSigV4Signer {
    fn new(access_key: &str, secret_key: &str, region: &str) -> Self {
        Self {
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            region: region.to_string(),
            service: "s3".to_string(),
        }
    }

    /// Get current UTC time as (date_stamp "YYYYMMDD", amz_date "YYYYMMDDTHHMMSSZ").
    fn current_time() -> (String, String) {
        // Use std::time to compute UTC date/time
        let dur = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self::format_time(dur.as_secs())
    }

    /// Format a Unix timestamp into (date_stamp, amz_date).
    fn format_time(unix_secs: u64) -> (String, String) {
        // Convert unix timestamp to calendar date/time (UTC)
        let secs = unix_secs;
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Civil date from days since epoch (1970-01-01) using a well-known algorithm
        let z = days as i64 + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u64; // day of era [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };

        let date_stamp = format!("{:04}{:02}{:02}", y, m, d);
        let amz_date = format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
            y, m, d, hours, minutes, seconds
        );
        (date_stamp, amz_date)
    }

    /// Parse URL into (path, canonical_query_string).
    fn parse_url(url: &str) -> (String, String) {
        // Strip scheme and host to get path + query
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url);

        let after_host = without_scheme
            .find('/')
            .map(|i| &without_scheme[i..])
            .unwrap_or("/");

        let (path, query) = if let Some(q_idx) = after_host.find('?') {
            (&after_host[..q_idx], &after_host[q_idx + 1..])
        } else {
            (after_host, "")
        };

        // Sort query parameters for canonical form
        let canonical_query = if query.is_empty() {
            String::new()
        } else {
            let mut pairs: Vec<&str> = query.split('&').collect();
            pairs.sort();
            pairs.join("&")
        };

        (path.to_string(), canonical_query)
    }

    /// Extract hostname from URL.
    fn extract_host(url: &str) -> String {
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url);
        without_scheme
            .split('/')
            .next()
            .unwrap_or("localhost")
            .to_string()
    }

    /// Build canonical headers string (sorted, lowercased, trimmed).
    fn canonical_headers(&self, extra_headers: &[(&str, &str)], amz_date: &str, url: &str) -> String {
        let host = Self::extract_host(url);
        let mut headers: Vec<(String, String)> = Vec::new();
        headers.push(("host".to_string(), host));
        headers.push(("x-amz-date".to_string(), amz_date.to_string()));
        for (k, v) in extra_headers {
            let lower = k.to_lowercase();
            if lower != "host" && lower != "x-amz-date" {
                headers.push((lower, v.trim().to_string()));
            }
        }
        headers.sort_by(|a, b| a.0.cmp(&b.0));
        let mut result = String::new();
        for (k, v) in &headers {
            result.push_str(k);
            result.push(':');
            result.push_str(v);
            result.push('\n');
        }
        result
    }

    /// Build signed headers string (sorted, semicolon-delimited).
    fn signed_headers(&self, extra_headers: &[(&str, &str)]) -> String {
        let mut names: Vec<String> = vec!["host".to_string(), "x-amz-date".to_string()];
        for (k, _) in extra_headers {
            let lower = k.to_lowercase();
            if lower != "host" && lower != "x-amz-date" {
                names.push(lower);
            }
        }
        names.sort();
        names.dedup();
        names.join(";")
    }

    /// Derive the signing key: HMAC chain of date/region/service/aws4_request.
    fn derive_signing_key(&self, date_stamp: &str) -> Vec<u8> {
        let k_date = hmac_sha256(
            format!("AWS4{}", self.secret_key).as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, self.region.as_bytes());
        let k_service = hmac_sha256(&k_region, self.service.as_bytes());
        hmac_sha256(&k_service, b"aws4_request")
    }

    /// Sign a request and return the headers to add.
    fn sign_request(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        payload: &[u8],
    ) -> Vec<(String, String)> {
        let (date_stamp, amz_date) = Self::current_time();
        self.sign_request_at(method, url, headers, payload, &date_stamp, &amz_date)
    }

    /// Sign a request at a specific time (for testing determinism).
    fn sign_request_at(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        payload: &[u8],
        date_stamp: &str,
        amz_date: &str,
    ) -> Vec<(String, String)> {
        // Step 1: Create canonical request
        let (path, query) = Self::parse_url(url);
        let canonical_headers = self.canonical_headers(headers, amz_date, url);
        let signed_headers = self.signed_headers(headers);
        let payload_hash = sha256_hex(payload);

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, path, query, canonical_headers, signed_headers, payload_hash
        );

        // Step 2: Create string to sign
        let scope = format!(
            "{}/{}/{}/aws4_request",
            date_stamp, self.region, self.service
        );
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            amz_date,
            scope,
            sha256_hex(canonical_request.as_bytes())
        );

        // Step 3: Calculate signing key
        let signing_key = self.derive_signing_key(date_stamp);

        // Step 4: Calculate signature
        let signature = hmac_sha256_hex(&signing_key, string_to_sign.as_bytes());

        // Step 5: Create Authorization header
        let auth = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.access_key, scope, signed_headers, signature
        );

        vec![
            ("Authorization".to_string(), auth),
            ("x-amz-date".to_string(), amz_date.to_string()),
            ("x-amz-content-sha256".to_string(), payload_hash),
        ]
    }
}

// ============================================================================
// Helpers
// ============================================================================

use std::io::Read;

fn urlencoded(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
                format!("{}", b as char)
            } else if b == b' ' {
                "+".to_string()
            } else {
                format!("%{:02X}", b)
            }
        })
        .collect()
}

/// Simplified S3 XML list response parser.
fn parse_s3_list_xml(xml: &str) -> Vec<CloudObject> {
    let mut objects = Vec::new();
    // Simple parser: find <Contents> blocks
    for contents in xml.split("<Contents>").skip(1) {
        let key = extract_xml_value(contents, "Key").unwrap_or_default();
        let size: u64 = extract_xml_value(contents, "Size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if !key.is_empty() {
            objects.push(CloudObject {
                key,
                size,
                last_modified: 0,
                content_type: None,
                metadata: HashMap::new(),
            });
        }
    }
    objects
}

/// Extract a value between XML tags: <Tag>value</Tag>
fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

// ============================================================================
// Cloud Provider
// ============================================================================

/// Supported cloud providers.
#[derive(Debug, Clone, PartialEq)]
pub enum CloudProvider {
    Aws,
    Azure,
    Gcp,
    Generic,
}

// ============================================================================
// S3 Storage Request Types
// ============================================================================

/// S3 operation types for request building.
#[derive(Debug, Clone, PartialEq)]
pub enum S3Operation {
    GetObject,
    PutObject,
    DeleteObject,
    ListObjects,
    HeadObject,
    CopyObject,
}

/// An S3-style request builder.
#[derive(Debug, Clone)]
pub struct S3Request {
    pub bucket: String,
    pub key: String,
    pub operation: S3Operation,
    pub content_type: Option<String>,
    pub body: Option<Vec<u8>>,
    pub headers: HashMap<String, String>,
}

impl S3Request {
    pub fn new(bucket: &str, key: &str, operation: S3Operation) -> Self {
        Self {
            bucket: bucket.to_string(),
            key: key.to_string(),
            operation,
            content_type: None,
            body: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_content_type(mut self, ct: &str) -> Self {
        self.content_type = Some(ct.to_string());
        self
    }

    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the S3 URL for this request.
    pub fn to_url(&self, region: &str) -> String {
        format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            self.bucket, region, self.key
        )
    }

    /// Return the HTTP method for this operation.
    pub fn to_method(&self) -> &str {
        match self.operation {
            S3Operation::GetObject => "GET",
            S3Operation::PutObject => "PUT",
            S3Operation::DeleteObject => "DELETE",
            S3Operation::ListObjects => "GET",
            S3Operation::HeadObject => "HEAD",
            S3Operation::CopyObject => "PUT",
        }
    }

    /// Build headers for this request, merging custom headers with required ones.
    pub fn to_headers(&self, region: &str) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        headers.insert(
            "host".to_string(),
            format!("{}.s3.{}.amazonaws.com", self.bucket, region),
        );
        if let Some(ref ct) = self.content_type {
            headers.insert("content-type".to_string(), ct.clone());
        }
        headers
    }
}

// ============================================================================
// Azure Blob Storage Request Types
// ============================================================================

/// Azure Blob Storage operation types.
#[derive(Debug, Clone, PartialEq)]
pub enum AzureBlobOperation {
    GetBlob,
    PutBlob,
    DeleteBlob,
    ListBlobs,
    GetProperties,
}

/// An Azure Blob Storage request builder.
#[derive(Debug, Clone)]
pub struct AzureBlobRequest {
    pub container: String,
    pub blob: String,
    pub operation: AzureBlobOperation,
    pub account: String,
    pub body: Option<Vec<u8>>,
    pub headers: HashMap<String, String>,
}

impl AzureBlobRequest {
    pub fn new(account: &str, container: &str, blob: &str, operation: AzureBlobOperation) -> Self {
        Self {
            container: container.to_string(),
            blob: blob.to_string(),
            operation,
            account: account.to_string(),
            body: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the Azure Blob Storage URL for this request.
    pub fn to_url(&self) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}/{}",
            self.account, self.container, self.blob
        )
    }

    /// Return the HTTP method for this operation.
    pub fn to_method(&self) -> &str {
        match self.operation {
            AzureBlobOperation::GetBlob => "GET",
            AzureBlobOperation::PutBlob => "PUT",
            AzureBlobOperation::DeleteBlob => "DELETE",
            AzureBlobOperation::ListBlobs => "GET",
            AzureBlobOperation::GetProperties => "HEAD",
        }
    }

    /// Build headers for this request, merging custom headers with required ones.
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        headers.insert(
            "host".to_string(),
            format!("{}.blob.core.windows.net", self.account),
        );
        headers.insert("x-ms-version".to_string(), "2021-08-06".to_string());
        headers
    }
}

// ============================================================================
// Google Cloud Storage Request Types
// ============================================================================

/// Google Cloud Storage operation types.
#[derive(Debug, Clone, PartialEq)]
pub enum GcsOperation {
    GetObject,
    InsertObject,
    DeleteObject,
    ListObjects,
    GetMetadata,
}

/// A Google Cloud Storage request builder.
#[derive(Debug, Clone)]
pub struct GcsRequest {
    pub bucket: String,
    pub object: String,
    pub operation: GcsOperation,
    pub project_id: String,
    pub body: Option<Vec<u8>>,
    pub headers: HashMap<String, String>,
}

impl GcsRequest {
    pub fn new(project_id: &str, bucket: &str, object: &str, operation: GcsOperation) -> Self {
        Self {
            bucket: bucket.to_string(),
            object: object.to_string(),
            operation,
            project_id: project_id.to_string(),
            body: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the GCS URL for this request.
    /// For ListObjects, omits the object suffix.
    pub fn to_url(&self) -> String {
        match self.operation {
            GcsOperation::ListObjects => {
                format!(
                    "https://storage.googleapis.com/storage/v1/b/{}/o",
                    self.bucket
                )
            }
            _ => {
                format!(
                    "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
                    self.bucket, self.object
                )
            }
        }
    }

    /// Return the HTTP method for this operation.
    pub fn to_method(&self) -> &str {
        match self.operation {
            GcsOperation::GetObject => "GET",
            GcsOperation::InsertObject => "POST",
            GcsOperation::DeleteObject => "DELETE",
            GcsOperation::ListObjects => "GET",
            GcsOperation::GetMetadata => "GET",
        }
    }

    /// Build headers for this request, merging custom headers with required ones.
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        headers.insert("host".to_string(), "storage.googleapis.com".to_string());
        headers
    }
}

// ============================================================================
// Unified Storage Connector
// ============================================================================

/// Generic storage operation (provider-agnostic).
#[derive(Debug, Clone, PartialEq)]
pub enum StorageOperation {
    Get,
    Put,
    Delete,
    List,
    Head,
}

/// A fully-resolved storage request ready to execute.
#[derive(Debug, Clone)]
pub struct StorageRequest {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

/// Unified storage connector that dispatches to the appropriate cloud provider.
pub struct StorageConnector {
    provider: CloudProvider,
    region: String,
}

impl StorageConnector {
    pub fn new(provider: CloudProvider, region: &str) -> Self {
        Self {
            provider,
            region: region.to_string(),
        }
    }

    /// Build a provider-specific storage request from generic parameters.
    pub fn build_request(
        &self,
        account_or_project: &str,
        bucket: &str,
        key: &str,
        operation: StorageOperation,
    ) -> StorageRequest {
        match self.provider {
            CloudProvider::Aws | CloudProvider::Generic => {
                let s3_op = match operation {
                    StorageOperation::Get => S3Operation::GetObject,
                    StorageOperation::Put => S3Operation::PutObject,
                    StorageOperation::Delete => S3Operation::DeleteObject,
                    StorageOperation::List => S3Operation::ListObjects,
                    StorageOperation::Head => S3Operation::HeadObject,
                };
                let req = S3Request::new(bucket, key, s3_op);
                StorageRequest {
                    url: req.to_url(&self.region),
                    method: req.to_method().to_string(),
                    headers: req.to_headers(&self.region),
                    body: None,
                }
            }
            CloudProvider::Azure => {
                let az_op = match operation {
                    StorageOperation::Get => AzureBlobOperation::GetBlob,
                    StorageOperation::Put => AzureBlobOperation::PutBlob,
                    StorageOperation::Delete => AzureBlobOperation::DeleteBlob,
                    StorageOperation::List => AzureBlobOperation::ListBlobs,
                    StorageOperation::Head => AzureBlobOperation::GetProperties,
                };
                let req = AzureBlobRequest::new(account_or_project, bucket, key, az_op);
                StorageRequest {
                    url: req.to_url(),
                    method: req.to_method().to_string(),
                    headers: req.to_headers(),
                    body: None,
                }
            }
            CloudProvider::Gcp => {
                let gcs_op = match operation {
                    StorageOperation::Get => GcsOperation::GetObject,
                    StorageOperation::Put => GcsOperation::InsertObject,
                    StorageOperation::Delete => GcsOperation::DeleteObject,
                    StorageOperation::List => GcsOperation::ListObjects,
                    StorageOperation::Head => GcsOperation::GetMetadata,
                };
                let req = GcsRequest::new(account_or_project, bucket, key, gcs_op);
                StorageRequest {
                    url: req.to_url(),
                    method: req.to_method().to_string(),
                    headers: req.to_headers(),
                    body: None,
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_new() {
        let config = S3Config::new("my-bucket", "us-east-1", "AKID", "secret");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.region, "us-east-1");
        assert!(!config.path_style);
        assert!(config.endpoint.is_none());
    }

    #[test]
    fn test_s3_config_base_url_virtual_host() {
        let config = S3Config::new("my-bucket", "us-east-1", "AKID", "secret");
        assert_eq!(
            config.base_url(),
            "https://my-bucket.s3.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn test_s3_config_base_url_path_style() {
        let mut config = S3Config::new("my-bucket", "us-east-1", "AKID", "secret");
        config.path_style = true;
        assert_eq!(
            config.base_url(),
            "https://s3.us-east-1.amazonaws.com/my-bucket"
        );
    }

    #[test]
    fn test_s3_config_custom_endpoint() {
        let config = S3Config::new("data", "us-east-1", "AKID", "secret")
            .with_endpoint("http://localhost:9000");
        assert_eq!(config.base_url(), "http://localhost:9000/data");
        assert!(config.path_style);
    }

    #[test]
    fn test_s3_config_host_virtual() {
        let config = S3Config::new("my-bucket", "eu-west-1", "AKID", "secret");
        assert_eq!(config.host(), "my-bucket.s3.eu-west-1.amazonaws.com");
    }

    #[test]
    fn test_s3_config_host_custom_endpoint() {
        let config = S3Config::new("data", "us-east-1", "AKID", "secret")
            .with_endpoint("https://minio.example.com:9000");
        assert_eq!(config.host(), "minio.example.com:9000");
    }

    #[test]
    fn test_s3_client_object_url() {
        let config = S3Config::new("docs", "us-west-2", "AKID", "secret");
        let client = S3Client::new(config);
        assert_eq!(
            client.object_url("path/to/file.txt"),
            "https://docs.s3.us-west-2.amazonaws.com/path/to/file.txt"
        );
    }

    #[test]
    fn test_s3_client_object_url_leading_slash() {
        let config = S3Config::new("docs", "us-west-2", "AKID", "secret");
        let client = S3Client::new(config);
        assert_eq!(
            client.object_url("/path/to/file.txt"),
            "https://docs.s3.us-west-2.amazonaws.com/path/to/file.txt"
        );
    }

    #[test]
    fn test_s3_provider_name() {
        let config = S3Config::new("b", "r", "k", "s");
        let client = S3Client::new(config);
        assert_eq!(client.provider_name(), "Amazon S3");
    }

    #[test]
    fn test_google_drive_config() {
        let config = GoogleDriveConfig::new("test-token-123");
        assert_eq!(config.access_token, "test-token-123");
        assert_eq!(config.root_folder_id, "root");
    }

    #[test]
    fn test_google_drive_config_with_folder() {
        let config = GoogleDriveConfig::new("token").with_folder("folder-abc");
        assert_eq!(config.root_folder_id, "folder-abc");
    }

    #[test]
    fn test_google_drive_api_url() {
        let client = GoogleDriveClient::new(GoogleDriveConfig::new("token"));
        assert_eq!(
            client.api_url("/files"),
            "https://www.googleapis.com/drive/v3/files"
        );
        assert_eq!(
            client.api_url("/files/abc123"),
            "https://www.googleapis.com/drive/v3/files/abc123"
        );
    }

    #[test]
    fn test_google_drive_upload_url() {
        let client = GoogleDriveClient::new(GoogleDriveConfig::new("token"));
        let url = client.upload_url();
        assert!(url.contains("upload/drive/v3/files"));
        assert!(url.contains("uploadType=multipart"));
    }

    #[test]
    fn test_google_drive_provider_name() {
        let client = GoogleDriveClient::new(GoogleDriveConfig::new("t"));
        assert_eq!(client.provider_name(), "Google Drive");
    }

    #[test]
    fn test_urlencoded() {
        assert_eq!(urlencoded("hello world"), "hello+world");
        assert_eq!(urlencoded("a/b"), "a%2Fb");
        assert_eq!(urlencoded("test"), "test");
    }

    #[test]
    fn test_parse_s3_list_xml() {
        let xml = r#"
            <ListBucketResult>
                <Contents>
                    <Key>file1.txt</Key>
                    <Size>1024</Size>
                </Contents>
                <Contents>
                    <Key>folder/file2.pdf</Key>
                    <Size>2048</Size>
                </Contents>
            </ListBucketResult>
        "#;
        let objects = parse_s3_list_xml(xml);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].key, "file1.txt");
        assert_eq!(objects[0].size, 1024);
        assert_eq!(objects[1].key, "folder/file2.pdf");
        assert_eq!(objects[1].size, 2048);
    }

    #[test]
    fn test_parse_s3_list_xml_empty() {
        let xml = "<ListBucketResult></ListBucketResult>";
        let objects = parse_s3_list_xml(xml);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_extract_xml_value() {
        assert_eq!(
            extract_xml_value("<Key>test.txt</Key>", "Key"),
            Some("test.txt".to_string())
        );
        assert_eq!(extract_xml_value("<NoMatch>value</NoMatch>", "Key"), None);
    }

    #[test]
    fn test_cloud_object_creation() {
        let obj = CloudObject {
            key: "test.txt".to_string(),
            size: 100,
            last_modified: 1234567890,
            content_type: Some("text/plain".to_string()),
            metadata: HashMap::new(),
        };
        assert_eq!(obj.key, "test.txt");
        assert_eq!(obj.size, 100);
    }

    #[test]
    fn test_list_options_default() {
        let opts = ListOptions::default();
        assert!(opts.prefix.is_none());
        assert!(opts.max_results.is_none());
        assert!(opts.page_token.is_none());
    }

    // ========================================================================
    // Storage Request Types Tests
    // ========================================================================

    #[test]
    fn test_s3_url_us_east() {
        let req = S3Request::new("my-data", "reports/q1.csv", S3Operation::GetObject);
        assert_eq!(
            req.to_url("us-east-1"),
            "https://my-data.s3.us-east-1.amazonaws.com/reports/q1.csv"
        );
    }

    #[test]
    fn test_s3_url_eu_west() {
        let req = S3Request::new("euro-bucket", "images/logo.png", S3Operation::PutObject);
        assert_eq!(
            req.to_url("eu-west-1"),
            "https://euro-bucket.s3.eu-west-1.amazonaws.com/images/logo.png"
        );
    }

    #[test]
    fn test_s3_method_mapping() {
        assert_eq!(
            S3Request::new("b", "k", S3Operation::GetObject).to_method(),
            "GET"
        );
        assert_eq!(
            S3Request::new("b", "k", S3Operation::PutObject).to_method(),
            "PUT"
        );
        assert_eq!(
            S3Request::new("b", "k", S3Operation::DeleteObject).to_method(),
            "DELETE"
        );
        assert_eq!(
            S3Request::new("b", "k", S3Operation::ListObjects).to_method(),
            "GET"
        );
        assert_eq!(
            S3Request::new("b", "k", S3Operation::HeadObject).to_method(),
            "HEAD"
        );
        assert_eq!(
            S3Request::new("b", "k", S3Operation::CopyObject).to_method(),
            "PUT"
        );
    }

    #[test]
    fn test_azure_blob_url() {
        let req = AzureBlobRequest::new(
            "myaccount",
            "mycontainer",
            "path/to/blob.bin",
            AzureBlobOperation::GetBlob,
        );
        assert_eq!(
            req.to_url(),
            "https://myaccount.blob.core.windows.net/mycontainer/path/to/blob.bin"
        );
    }

    #[test]
    fn test_azure_method_mapping() {
        assert_eq!(
            AzureBlobRequest::new("a", "c", "b", AzureBlobOperation::GetBlob).to_method(),
            "GET"
        );
        assert_eq!(
            AzureBlobRequest::new("a", "c", "b", AzureBlobOperation::PutBlob).to_method(),
            "PUT"
        );
        assert_eq!(
            AzureBlobRequest::new("a", "c", "b", AzureBlobOperation::DeleteBlob).to_method(),
            "DELETE"
        );
        assert_eq!(
            AzureBlobRequest::new("a", "c", "b", AzureBlobOperation::ListBlobs).to_method(),
            "GET"
        );
        assert_eq!(
            AzureBlobRequest::new("a", "c", "b", AzureBlobOperation::GetProperties).to_method(),
            "HEAD"
        );
    }

    #[test]
    fn test_gcs_url_format() {
        let req = GcsRequest::new(
            "my-project",
            "my-bucket",
            "data/file.json",
            GcsOperation::GetObject,
        );
        assert_eq!(
            req.to_url(),
            "https://storage.googleapis.com/storage/v1/b/my-bucket/o/data/file.json"
        );
    }

    #[test]
    fn test_gcs_list_url() {
        let req = GcsRequest::new(
            "my-project",
            "my-bucket",
            "ignored",
            GcsOperation::ListObjects,
        );
        assert_eq!(
            req.to_url(),
            "https://storage.googleapis.com/storage/v1/b/my-bucket/o"
        );
    }

    #[test]
    fn test_storage_connector_s3() {
        let connector = StorageConnector::new(CloudProvider::Aws, "us-west-2");
        let req =
            connector.build_request("ignored", "my-bucket", "file.txt", StorageOperation::Get);
        assert_eq!(
            req.url,
            "https://my-bucket.s3.us-west-2.amazonaws.com/file.txt"
        );
        assert_eq!(req.method, "GET");
        assert_eq!(
            req.headers.get("host").unwrap(),
            "my-bucket.s3.us-west-2.amazonaws.com"
        );
    }

    #[test]
    fn test_storage_connector_azure() {
        let connector = StorageConnector::new(CloudProvider::Azure, "eastus");
        let req = connector.build_request(
            "storageacct",
            "container1",
            "blob.dat",
            StorageOperation::Put,
        );
        assert_eq!(
            req.url,
            "https://storageacct.blob.core.windows.net/container1/blob.dat"
        );
        assert_eq!(req.method, "PUT");
        assert_eq!(
            req.headers.get("host").unwrap(),
            "storageacct.blob.core.windows.net"
        );
        assert_eq!(req.headers.get("x-ms-version").unwrap(), "2021-08-06");
    }

    #[test]
    fn test_storage_connector_gcs() {
        let connector = StorageConnector::new(CloudProvider::Gcp, "us-central1");
        let req = connector.build_request(
            "my-project",
            "gcs-bucket",
            "obj.bin",
            StorageOperation::Delete,
        );
        assert_eq!(
            req.url,
            "https://storage.googleapis.com/storage/v1/b/gcs-bucket/o/obj.bin"
        );
        assert_eq!(req.method, "DELETE");
        assert_eq!(req.headers.get("host").unwrap(), "storage.googleapis.com");
    }

    // ========================================================================
    // AWS SigV4 Signing Tests
    // ========================================================================

    #[test]
    fn test_sigv4_sha256_known_vector() {
        // SHA-256 of empty string = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        // SHA-256 of "abc" = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hash2 = sha256_hex(b"abc");
        assert_eq!(
            hash2,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sigv4_hmac_sha256_known_vector() {
        // HMAC-SHA256 with key "key" and message "The quick brown fox jumps over the lazy dog"
        // = f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
        let result = hmac_sha256_hex(
            b"key",
            b"The quick brown fox jumps over the lazy dog",
        );
        assert_eq!(
            result,
            "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
        );
    }

    #[test]
    fn test_sigv4_signing_key_derivation() {
        // Test the key derivation chain with known values
        let signer = AwsSigV4Signer::new("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1");
        let key = signer.derive_signing_key("20150830");
        // The signing key should be 32 bytes (SHA-256 output)
        assert_eq!(key.len(), 32);
        // Verify it's deterministic
        let key2 = signer.derive_signing_key("20150830");
        assert_eq!(key, key2);
        // Different date should produce different key
        let key3 = signer.derive_signing_key("20150831");
        assert_ne!(key, key3);
    }

    #[test]
    fn test_sigv4_canonical_request_format() {
        let signer = AwsSigV4Signer::new("AKID", "SECRET", "us-east-1");
        let url = "https://examplebucket.s3.us-east-1.amazonaws.com/test.txt";
        let amz_date = "20130524T000000Z";

        let (path, query) = AwsSigV4Signer::parse_url(url);
        let canonical_headers = signer.canonical_headers(&[], amz_date, url);
        let signed_headers = signer.signed_headers(&[]);
        let payload_hash = sha256_hex(b"");

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            "GET", path, query, canonical_headers, signed_headers, payload_hash
        );

        // Verify the canonical request has the right structure
        let lines: Vec<&str> = canonical_request.split('\n').collect();
        assert_eq!(lines[0], "GET"); // Method
        assert_eq!(lines[1], "/test.txt"); // Path
        assert_eq!(lines[2], ""); // Empty query string
        // Headers should contain host and x-amz-date
        assert!(lines[3].starts_with("host:"));
        assert!(lines[4].starts_with("x-amz-date:"));
        // Blank line after headers (the trailing \n in canonical_headers)
        assert_eq!(lines[5], ""); // blank line
        assert_eq!(lines[6], "host;x-amz-date"); // signed headers
    }

    #[test]
    fn test_sigv4_authorization_header_format() {
        let signer = AwsSigV4Signer::new("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1");
        let url = "https://examplebucket.s3.us-east-1.amazonaws.com/test.txt";
        let date_stamp = "20130524";
        let amz_date = "20130524T000000Z";

        let headers = signer.sign_request_at("GET", url, &[], b"", date_stamp, amz_date);

        // Should produce 3 headers: Authorization, x-amz-date, x-amz-content-sha256
        assert_eq!(headers.len(), 3);

        let auth = &headers[0];
        assert_eq!(auth.0, "Authorization");
        assert!(auth.1.starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"));
        assert!(auth.1.contains("20130524/us-east-1/s3/aws4_request"));
        assert!(auth.1.contains("SignedHeaders=host;x-amz-date"));
        assert!(auth.1.contains("Signature="));

        let amz_date_header = &headers[1];
        assert_eq!(amz_date_header.0, "x-amz-date");
        assert_eq!(amz_date_header.1, "20130524T000000Z");

        let content_hash = &headers[2];
        assert_eq!(content_hash.0, "x-amz-content-sha256");
        // Empty payload hash
        assert_eq!(
            content_hash.1,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_s3_unsigned_fallback() {
        // When credentials are empty, has_credentials() returns false (unsigned fallback)
        let config = S3Config::new("bucket", "us-east-1", "", "");
        let client = S3Client::new(config);
        assert!(!client.has_credentials());

        // When credentials are present, has_credentials() returns true
        let config2 = S3Config::new("bucket", "us-east-1", "AKID", "secret");
        let client2 = S3Client::new(config2);
        assert!(client2.has_credentials());
    }

    #[test]
    fn test_sigv4_parse_url_simple() {
        let (path, query) = AwsSigV4Signer::parse_url("https://bucket.s3.amazonaws.com/key");
        assert_eq!(path, "/key");
        assert_eq!(query, "");
    }

    #[test]
    fn test_sigv4_parse_url_with_query() {
        let (path, query) = AwsSigV4Signer::parse_url(
            "https://bucket.s3.amazonaws.com/?list-type=2&prefix=docs/",
        );
        assert_eq!(path, "/");
        // Query params should be sorted
        assert!(query.contains("list-type=2"));
        assert!(query.contains("prefix=docs/"));
    }

    #[test]
    fn test_sigv4_format_time() {
        // 2023-05-24T12:30:45Z = 1684931445 unix timestamp
        let (date_stamp, amz_date) = AwsSigV4Signer::format_time(1684931445);
        assert_eq!(date_stamp, "20230524");
        assert_eq!(amz_date, "20230524T123045Z");
    }

    #[test]
    fn test_sigv4_deterministic_signing() {
        // Same inputs should produce same output
        let signer = AwsSigV4Signer::new("AKID", "SECRET", "us-east-1");
        let url = "https://bucket.s3.us-east-1.amazonaws.com/key";
        let h1 = signer.sign_request_at("GET", url, &[], b"", "20230524", "20230524T120000Z");
        let h2 = signer.sign_request_at("GET", url, &[], b"", "20230524", "20230524T120000Z");
        assert_eq!(h1, h2);
    }
}
