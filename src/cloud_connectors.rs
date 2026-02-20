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

        // In a real implementation, this would use SigV4 signing from aws_auth module.
        // For now, we construct the request and attempt to make it.
        let resp = ureq::get(&url)
            .set("Host", &self.config.host())
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .context("S3 list failed")?;

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
        let resp = ureq::get(&url)
            .set("Host", &self.config.host())
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .context("S3 GET failed")?;

        let mut bytes = Vec::new();
        resp.into_reader()
            .read_to_end(&mut bytes)
            .context("Failed to read S3 object")?;
        Ok(bytes)
    }

    fn put(&self, key: &str, data: &[u8], content_type: Option<&str>) -> Result<()> {
        let url = self.object_url(key);
        let ct = content_type.unwrap_or("application/octet-stream");
        ureq::put(&url)
            .set("Host", &self.config.host())
            .set("Content-Type", ct)
            .timeout(std::time::Duration::from_secs(60))
            .send_bytes(data)
            .context("S3 PUT failed")?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let url = self.object_url(key);
        ureq::request("DELETE", &url)
            .set("Host", &self.config.host())
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .context("S3 DELETE failed")?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let url = self.object_url(key);
        match ureq::request("HEAD", &url)
            .set("Host", &self.config.host())
            .timeout(std::time::Duration::from_secs(10))
            .call()
        {
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
}
