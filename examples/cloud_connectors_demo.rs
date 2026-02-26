//! Example: cloud_connectors_demo -- Demonstrates multi-cloud storage connectors.
//!
//! Run with: cargo run --example cloud_connectors_demo --features cloud-connectors
//!
//! This example showcases S3, Google Drive, Azure Blob, and GCS request builders,
//! the unified CloudStorage trait, and the StorageConnector multi-cloud dispatcher.
//! No actual cloud connections are made — all operations build request objects.

use ai_assistant::{
    // S3
    S3Client, S3Config, S3Operation, S3Request,
    // Google Drive
    GoogleDriveClient, GoogleDriveConfig,
    // Azure Blob
    AzureBlobOperation, AzureBlobRequest,
    // Google Cloud Storage
    GcsOperation, GcsRequest,
    // Unified
    CloudStorage, StorageConnector, StorageOperation,
    // Common
    CloudObject, ListOptions, ListResult,
};
use std::collections::HashMap;

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Cloud Connectors Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. S3 Configuration & Client
    // ------------------------------------------------------------------
    println!("--- 1. Amazon S3 ---\n");

    let s3_config = S3Config::new(
        "my-ai-bucket",
        "us-east-1",
        "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    );

    println!("  Bucket:   {}", s3_config.bucket);
    println!("  Region:   {}", s3_config.region);
    println!("  Base URL: {}", s3_config.base_url());
    println!("  Host:     {}", s3_config.host());

    // With custom endpoint (MinIO)
    let minio_config = S3Config::new("local-bucket", "us-east-1", "minioadmin", "minioadmin")
        .with_endpoint("http://localhost:9000");
    println!("\n  MinIO endpoint: {:?}", minio_config.endpoint);
    println!("  MinIO base URL: {}", minio_config.base_url());

    let s3_client = S3Client::new(s3_config);
    println!("\n  Client bucket:  {}", s3_client.bucket());
    println!("  Client region:  {}", s3_client.region());
    println!("  Object URL:     {}", s3_client.object_url("models/llama3.gguf"));
    println!("  Provider:       {}", s3_client.provider_name());

    // ------------------------------------------------------------------
    // 2. S3 Request Builder
    // ------------------------------------------------------------------
    println!("\n--- 2. S3 Request Builder ---\n");

    let operations = [
        ("GetObject", S3Operation::GetObject),
        ("PutObject", S3Operation::PutObject),
        ("DeleteObject", S3Operation::DeleteObject),
        ("ListObjects", S3Operation::ListObjects),
        ("HeadObject", S3Operation::HeadObject),
    ];

    for (name, op) in &operations {
        let req = S3Request::new("my-ai-bucket", "models/llama3.gguf", op.clone());
        println!("  {:<14} method={:<6} url={}",
            name, req.to_method(), req.to_url("us-east-1"));
    }

    // Request with body
    let put_req = S3Request::new("my-ai-bucket", "data/config.json", S3Operation::PutObject)
        .with_content_type("application/json")
        .with_body(b"{\"model\": \"llama3\"}".to_vec())
        .with_header("x-amz-meta-version", "1.0");

    println!("\n  PutObject with body:");
    println!("    Content-Type: {:?}", put_req.content_type);
    println!("    Body size:    {} bytes", put_req.body.as_ref().map(|b| b.len()).unwrap_or(0));
    println!("    Headers:      {:?}", put_req.headers);

    // ------------------------------------------------------------------
    // 3. Google Drive
    // ------------------------------------------------------------------
    println!("\n--- 3. Google Drive ---\n");

    let drive_config = GoogleDriveConfig::new("ya29.example-access-token")
        .with_folder("1ABC_folder_id");

    let drive_client = GoogleDriveClient::new(drive_config);
    println!("  Root folder: {}", drive_client.root_folder());
    println!("  API URL:     {}", drive_client.api_url("files"));
    println!("  Upload URL:  {}", drive_client.upload_url());
    println!("  Provider:    {}", drive_client.provider_name());

    // ------------------------------------------------------------------
    // 4. Azure Blob Storage
    // ------------------------------------------------------------------
    println!("\n--- 4. Azure Blob Storage ---\n");

    let azure_ops = [
        ("GetBlob", AzureBlobOperation::GetBlob),
        ("PutBlob", AzureBlobOperation::PutBlob),
        ("DeleteBlob", AzureBlobOperation::DeleteBlob),
        ("ListBlobs", AzureBlobOperation::ListBlobs),
    ];

    for (name, op) in &azure_ops {
        let req = AzureBlobRequest::new("myaccount", "aicontainer", "models/llama3.bin", op.clone());
        println!("  {:<12} method={:<6} url={}",
            name, req.to_method(), req.to_url());
    }

    let azure_req = AzureBlobRequest::new(
        "myaccount", "aicontainer", "data.json", AzureBlobOperation::PutBlob,
    )
    .with_body(b"test data".to_vec())
    .with_header("x-ms-blob-type", "BlockBlob");
    println!("\n  Azure headers: {:?}", azure_req.to_headers());

    // ------------------------------------------------------------------
    // 5. Google Cloud Storage
    // ------------------------------------------------------------------
    println!("\n--- 5. Google Cloud Storage ---\n");

    let gcs_ops = [
        ("GetObject", GcsOperation::GetObject),
        ("InsertObject", GcsOperation::InsertObject),
        ("DeleteObject", GcsOperation::DeleteObject),
        ("ListObjects", GcsOperation::ListObjects),
    ];

    for (name, op) in &gcs_ops {
        let req = GcsRequest::new("my-project", "ai-bucket", "embeddings/v1.bin", op.clone());
        println!("  {:<14} method={:<6} url={}",
            name, req.to_method(), req.to_url());
    }

    // ------------------------------------------------------------------
    // 6. CloudObject & ListOptions
    // ------------------------------------------------------------------
    println!("\n--- 6. Common Types ---\n");

    let obj = CloudObject {
        key: "models/llama3-8b.gguf".to_string(),
        size: 4_500_000_000,
        last_modified: 1708900000,
        content_type: Some("application/octet-stream".to_string()),
        metadata: {
            let mut m = HashMap::new();
            m.insert("model-version".to_string(), "3.0".to_string());
            m
        },
    };
    println!("  CloudObject: key={}, size={:.1} GB",
        obj.key, obj.size as f64 / 1e9);

    let opts = ListOptions {
        prefix: Some("models/".to_string()),
        max_results: Some(100),
        page_token: None,
    };
    println!("  ListOptions: prefix={:?}, max={:?}",
        opts.prefix, opts.max_results);

    let list_result = ListResult {
        objects: vec![obj],
        next_page_token: Some("token_abc".to_string()),
    };
    println!("  ListResult:  {} objects, has_next={}",
        list_result.objects.len(), list_result.next_page_token.is_some());

    // ------------------------------------------------------------------
    // 7. StorageConnector (Unified Multi-Cloud)
    // ------------------------------------------------------------------
    println!("\n--- 7. StorageConnector (Multi-Cloud Dispatch) ---\n");

    let operations_unified = [
        StorageOperation::Get,
        StorageOperation::Put,
        StorageOperation::Delete,
        StorageOperation::List,
        StorageOperation::Head,
    ];

    // AWS
    let aws_connector = StorageConnector::new(
        ai_assistant::cloud_connectors::CloudProvider::Aws,
        "us-east-1",
    );
    println!("  AWS S3:");
    for op in &operations_unified {
        let req = aws_connector.build_request("account", "my-bucket", "data.bin", op.clone());
        println!("    {:?}: {} {}", op, req.method, req.url);
    }

    // Azure
    let azure_connector = StorageConnector::new(
        ai_assistant::cloud_connectors::CloudProvider::Azure,
        "eastus",
    );
    println!("\n  Azure Blob:");
    let req = azure_connector.build_request("myaccount", "container", "blob.bin", StorageOperation::Get);
    println!("    Get: {} {}", req.method, req.url);

    // GCP
    let gcp_connector = StorageConnector::new(
        ai_assistant::cloud_connectors::CloudProvider::Gcp,
        "us-central1",
    );
    println!("\n  GCP Storage:");
    let req = gcp_connector.build_request("my-project", "gcs-bucket", "object.bin", StorageOperation::Get);
    println!("    Get: {} {}", req.method, req.url);

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  Cloud connectors demo complete.");
    println!("  Capabilities: S3 (+ MinIO), Google Drive, Azure Blob,");
    println!("    GCS, and unified multi-cloud dispatch.");
    println!("==========================================================");
}
