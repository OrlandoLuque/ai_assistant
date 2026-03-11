//! Tests for the mcp_protocol module.

use super::*;
use std::collections::HashMap;

    #[test]
    fn test_mcp_request() {
        let request = McpRequest::new("tools/list").with_id(1u64);

        assert_eq!(request.method, "tools/list");
        assert_eq!(request.jsonrpc, "2.0");
    }

    #[test]
    fn test_mcp_tool() {
        let tool = McpTool::new("search", "Search the web")
            .with_property("query", "string", "Search query", true)
            .with_property("limit", "number", "Max results", false);

        assert_eq!(tool.name, "search");
        assert!(tool.input_schema["properties"]["query"].is_object());
    }

    #[test]
    fn test_mcp_server() {
        let mut server = McpServer::new("test-server", "1.0.0");

        server.register_tool(McpTool::new("echo", "Echo the input"), |args| {
            let text = args.get("text").and_then(|t| t.as_str()).unwrap_or("");
            Ok(serde_json::json!({ "echo": text }))
        });

        // Test initialize
        let init_request =
            McpRequest::new("initialize")
                .with_id(1u64)
                .with_params(serde_json::json!({
                    "protocolVersion": MCP_VERSION,
                    "clientInfo": { "name": "test" },
                    "capabilities": {}
                }));

        let response = server.handle_request(init_request);
        assert!(response.result.is_some());

        // Test tools list
        let list_request = McpRequest::new("tools/list").with_id(2u64);
        let response = server.handle_request(list_request);
        assert!(response.result.is_some());
    }

    #[test]
    fn test_mcp_error() {
        let error = McpError::method_not_found("unknown");
        assert_eq!(error.code, -32601);
    }

    #[test]
    fn test_mcp_resource() {
        let resource = McpResource::new("file:///test.txt", "test.txt")
            .with_description("A test file")
            .with_mime_type("text/plain");

        assert_eq!(resource.uri, "file:///test.txt");
        assert_eq!(resource.mime_type.unwrap(), "text/plain");
    }

    // ===== 2025-03-26 spec tests =====

    #[test]
    fn test_mcp_version_2025() {
        assert_eq!(MCP_VERSION, "2025-03-26");
        assert_eq!(MCP_VERSION_PREVIOUS, "2024-11-05");
    }

    #[test]
    fn test_tool_annotations_serde() {
        let ann = McpToolAnnotation {
            title: Some("My Tool".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: None,
        };
        let json = serde_json::to_value(&ann).unwrap();
        assert_eq!(json["title"], "My Tool");
        assert_eq!(json["readOnlyHint"], true);
        assert_eq!(json["destructiveHint"], false);
        assert_eq!(json["idempotentHint"], true);
        assert!(json.get("openWorldHint").is_none());
    }

    #[test]
    fn test_tool_with_annotations() {
        let tool = McpTool::new("search", "Search the web").with_annotations(McpToolAnnotation {
            title: Some("Web Search".to_string()),
            read_only_hint: Some(true),
            ..Default::default()
        });
        assert!(tool.annotations.is_some());
        let ann = tool.annotations.unwrap();
        assert_eq!(ann.title.unwrap(), "Web Search");
        assert_eq!(ann.read_only_hint.unwrap(), true);
    }

    #[test]
    fn test_pagination_serde() {
        let page = McpPagination {
            next_cursor: Some("abc123".to_string()),
        };
        let json = serde_json::to_value(&page).unwrap();
        assert_eq!(json["nextCursor"], "abc123");

        let empty_page = McpPagination { next_cursor: None };
        let json = serde_json::to_value(&empty_page).unwrap();
        assert!(json.get("nextCursor").is_none());
    }

    #[test]
    fn test_version_negotiation_current() {
        let server = McpServer::new("test", "1.0.0");
        let req = McpRequest::new("initialize")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2025-03-26",
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }));
        let resp = server.handle_request(req);
        let version = resp.result.unwrap()["protocolVersion"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(version, "2025-03-26");
    }

    #[test]
    fn test_version_negotiation_previous() {
        let server = McpServer::new("test", "1.0.0");
        let req = McpRequest::new("initialize")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": { "name": "old-client" },
                "capabilities": {}
            }));
        let resp = server.handle_request(req);
        let version = resp.result.unwrap()["protocolVersion"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(version, "2024-11-05");
    }

    #[test]
    fn test_transport_types() {
        assert_eq!(McpTransport::default(), McpTransport::StreamableHttp);

        let json = serde_json::to_value(&McpTransport::StreamableHttp).unwrap();
        assert_eq!(json, "streamable_http");

        let json = serde_json::to_value(&McpTransport::Stdio).unwrap();
        assert_eq!(json, "stdio");
    }

    #[test]
    fn test_streamable_session() {
        let server = McpServer::new("test", "1.0.0");
        let mut session = McpStreamableSession::new(server);

        assert!(!session.initialized);
        assert!(session.session_id.starts_with("mcp-"));

        // POST an initialize request
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_VERSION,
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }
        });
        let resp = session.handle_post(&body.to_string()).unwrap();
        assert!(session.initialized);
        let resp_json: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(resp_json["result"]["protocolVersion"].is_string());
    }

    #[test]
    fn test_streamable_session_events() {
        let server = McpServer::new("test", "1.0.0");
        let mut session = McpStreamableSession::new(server);

        session.push_event("message", "hello world");
        session.push_event("progress", "50%");

        let events = session.drain_events();
        assert_eq!(events.len(), 2);
        assert!(events[0].contains("event: message"));
        assert!(events[0].contains("data: hello world"));
        assert!(events[1].contains("event: progress"));

        // After drain, should be empty
        assert!(session.drain_events().is_empty());
    }

    #[test]
    fn test_streamable_session_header() {
        let server = McpServer::new("test", "1.0.0");
        let session = McpStreamableSession::new(server);
        let (key, value) = session.session_header();
        assert_eq!(key, "Mcp-Session-Id");
        assert!(value.starts_with("mcp-"));
    }

    #[test]
    fn test_tool_annotations_default() {
        let ann = McpToolAnnotation::default();
        assert!(ann.title.is_none());
        assert!(ann.read_only_hint.is_none());
        assert!(ann.destructive_hint.is_none());
        assert!(ann.idempotent_hint.is_none());
        assert!(ann.open_world_hint.is_none());
    }

    // -----------------------------------------------------------------------
    // OAuth 2.1 tests
    // -----------------------------------------------------------------------

    fn make_test_oauth_config() -> McpOAuthConfig {
        McpOAuthConfig {
            client_id: "test-client".to_string(),
            client_secret: Some("test-secret".to_string()),
            authorization_endpoint: "https://auth.example.com/authorize".to_string(),
            token_endpoint: "https://auth.example.com/token".to_string(),
            scopes: vec![
                McpOAuthScope {
                    name: "tools:read".to_string(),
                    description: "Read tools".to_string(),
                    resources: vec!["tools/*".to_string()],
                },
                McpOAuthScope {
                    name: "resources:read".to_string(),
                    description: "Read resources".to_string(),
                    resources: vec!["resources/*".to_string()],
                },
            ],
            redirect_uri: "http://localhost:8080/callback".to_string(),
            pkce_enabled: true,
        }
    }

    #[test]
    fn test_oauth_grant_type_serde() {
        let grant = McpOAuthGrantType::AuthorizationCode;
        let json = serde_json::to_string(&grant).unwrap();
        let back: McpOAuthGrantType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, McpOAuthGrantType::AuthorizationCode);

        let cc = McpOAuthGrantType::ClientCredentials;
        let json2 = serde_json::to_string(&cc).unwrap();
        let back2: McpOAuthGrantType = serde_json::from_str(&json2).unwrap();
        assert_eq!(back2, McpOAuthGrantType::ClientCredentials);

        let rt = McpOAuthGrantType::RefreshToken;
        let json3 = serde_json::to_string(&rt).unwrap();
        let back3: McpOAuthGrantType = serde_json::from_str(&json3).unwrap();
        assert_eq!(back3, McpOAuthGrantType::RefreshToken);
    }

    #[test]
    fn test_oauth_config_creation() {
        let config = make_test_oauth_config();
        assert_eq!(config.client_id, "test-client");
        assert_eq!(config.client_secret.as_deref(), Some("test-secret"));
        assert!(config.authorization_endpoint.starts_with("https://"));
        assert!(config.token_endpoint.starts_with("https://"));
        assert_eq!(config.scopes.len(), 2);
        assert!(config.pkce_enabled);
    }

    #[test]
    fn test_oauth_scope_creation() {
        let scope = McpOAuthScope {
            name: "tools:write".to_string(),
            description: "Write access to tools".to_string(),
            resources: vec!["tools/create".to_string(), "tools/update".to_string()],
        };
        assert_eq!(scope.name, "tools:write");
        assert_eq!(scope.resources.len(), 2);
    }

    #[test]
    fn test_authorization_url_building() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let (url, request) = manager.build_authorization_url("csrf-state-123");

        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=test-client"));
        assert!(url.contains("state=csrf-state-123"));
        assert!(url.contains("scope=tools%3Aread%20resources%3Aread"));

        assert_eq!(request.response_type, "code");
        assert_eq!(request.client_id, "test-client");
        assert_eq!(request.state, "csrf-state-123");
        assert_eq!(request.scope, "tools:read resources:read");
    }

    #[test]
    fn test_token_request_authorization_code() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_authorization_code("auth-code-xyz");

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "authorization_code"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "code" && v == "auth-code-xyz"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "redirect_uri" && v == "http://localhost:8080/callback"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
        assert_eq!(params.len(), 5);
    }

    #[test]
    fn test_token_request_client_credentials() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_client_credentials();

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "client_credentials"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "scope" && v == "tools:read resources:read"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
    }

    #[test]
    fn test_token_request_refresh() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_refresh("refresh-tok-abc");

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "refresh_token"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "refresh_token" && v == "refresh-tok-abc"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_pkce_challenge_generation() {
        let (challenge1, method1) = McpOAuthTokenManager::generate_pkce_challenge("my-verifier");
        let (challenge2, method2) = McpOAuthTokenManager::generate_pkce_challenge("my-verifier");
        // Deterministic
        assert_eq!(challenge1, challenge2);
        // H12: PKCE now uses SHA-256 instead of weak hash
        assert_eq!(method1, "S256");
        assert_eq!(method2, "S256");
        // SHA-256 (32 bytes) in base64url without padding = 43 chars (per RFC 7636)
        assert_eq!(challenge1.len(), 43);

        // Different verifiers produce different challenges
        let (challenge3, _) = McpOAuthTokenManager::generate_pkce_challenge("other-verifier");
        assert_ne!(challenge1, challenge3);
    }

    #[test]
    fn test_token_manager_set_and_get() {
        let config = make_test_oauth_config();
        let mut manager = McpOAuthTokenManager::new(config);

        // No token initially
        assert!(manager.get_access_token().is_none());

        let token = McpTokenResponse {
            access_token: "access-123".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(3600),
            refresh_token: Some("refresh-456".to_string()),
            scope: Some("tools:read".to_string()),
        };

        manager.set_token(token);
        assert_eq!(manager.get_access_token(), Some("access-123"));
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_token_expiry_no_token() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        // No token => considered expired
        assert!(manager.is_token_expired());
        assert!(manager.get_access_token().is_none());
    }

    #[test]
    fn test_token_no_expiry_never_expires() {
        let config = make_test_oauth_config();
        let mut manager = McpOAuthTokenManager::new(config);

        let token = McpTokenResponse {
            access_token: "permanent-token".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: None, // No expiry
            refresh_token: None,
            scope: None,
        };

        manager.set_token(token);
        // Token with no expiry should never be considered expired
        assert!(!manager.is_token_expired());
        assert_eq!(manager.get_access_token(), Some("permanent-token"));
    }

    #[test]
    fn test_token_response_serde() {
        let token = McpTokenResponse {
            access_token: "abc".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(7200),
            refresh_token: Some("def".to_string()),
            scope: Some("tools:read resources:read".to_string()),
        };

        let json = serde_json::to_string(&token).unwrap();
        let back: McpTokenResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.access_token, "abc");
        assert_eq!(back.token_type, "Bearer");
        assert_eq!(back.expires_in, Some(7200));
        assert_eq!(back.refresh_token.as_deref(), Some("def"));
        assert_eq!(back.scope.as_deref(), Some("tools:read resources:read"));
    }

    // =========================================================================
    // Session MCP resource tests (v4 - item 8.2)
    // =========================================================================

    fn make_test_session_info(id: &str) -> SessionResourceInfo {
        SessionResourceInfo {
            session_id: id.to_string(),
            name: Some(format!("Session {}", id)),
            created_at: 1000,
            updated_at: 2000,
            message_count: 5,
            closed_cleanly: true,
        }
    }

    #[test]
    fn test_session_manager_new() {
        let mgr = SessionMcpManager::new();
        assert!(mgr.list_sessions().is_empty());
    }

    #[test]
    fn test_register_session() {
        let mut mgr = SessionMcpManager::new();
        let info = make_test_session_info("s1");
        mgr.register_session(info);
        assert_eq!(mgr.list_sessions().len(), 1);
    }

    #[test]
    fn test_unregister_session() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("s1"));
        mgr.register_session(make_test_session_info("s2"));
        assert_eq!(mgr.list_sessions().len(), 2);

        let removed = mgr.unregister("s1");
        assert!(removed.is_some());
        assert_eq!(removed.as_ref().map(|r| r.session_id.as_str()), Some("s1"));
        assert_eq!(mgr.list_sessions().len(), 1);

        // Removing non-existent returns None
        assert!(mgr.unregister("s999").is_none());
    }

    #[test]
    fn test_get_session() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("s1"));

        let s = mgr.get_session("s1");
        assert!(s.is_some());
        assert_eq!(s.map(|s| s.message_count), Some(5));

        assert!(mgr.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_list_sessions() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("a"));
        mgr.register_session(make_test_session_info("b"));
        mgr.register_session(make_test_session_info("c"));

        let list = mgr.list_sessions();
        assert_eq!(list.len(), 3);

        let ids: Vec<&str> = list.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_sessions_to_json() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("j1"));

        let json = mgr.sessions_to_json();
        assert!(json.is_array());
        let arr = json.as_array().expect("should be array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["session_id"], "j1");
        assert_eq!(arr[0]["message_count"], 5);
    }

    #[test]
    fn test_generate_summary_empty() {
        let summary = SessionMcpManager::generate_summary("s-empty", &[]);
        assert_eq!(summary.session_id, "s-empty");
        assert_eq!(summary.message_count, 0);
        assert!(summary.key_topics.is_empty());
        assert!(summary.decisions.is_empty());
        assert!(summary.open_questions.is_empty());
        assert!(summary.summary.is_empty());
    }

    #[test]
    fn test_generate_summary_with_content() {
        let messages = vec![
            ("user".to_string(), "The architecture uses microservices for deployment".to_string()),
            ("assistant".to_string(), "Microservices architecture provides scalability benefits".to_string()),
            ("user".to_string(), "We need more microservices testing coverage".to_string()),
        ];
        let summary = SessionMcpManager::generate_summary("s-content", &messages);
        assert_eq!(summary.message_count, 3);
        assert!(!summary.key_topics.is_empty());
        // "microservices" should be the top topic (appears 3 times)
        assert!(summary.key_topics.contains(&"microservices".to_string()));
        assert!(summary.summary.contains("3 messages"));
    }

    #[test]
    fn test_generate_summary_decisions_and_questions() {
        let messages = vec![
            ("user".to_string(), "We decided to use Rust for the backend".to_string()),
            ("assistant".to_string(), "Agreed, Rust is a good choice".to_string()),
            ("user".to_string(), "What about the frontend framework?".to_string()),
            ("user".to_string(), "Have we chosen a database yet?".to_string()),
        ];
        let summary = SessionMcpManager::generate_summary("s-dq", &messages);
        assert_eq!(summary.decisions.len(), 3); // "decided", "agreed", "chosen"
        assert_eq!(summary.open_questions.len(), 2); // two questions with "?"
    }

    #[test]
    fn test_extract_highlights_conclusions() {
        let messages = vec![
            ("user".to_string(), "This is an important finding about performance".to_string()),
            ("assistant".to_string(), "A critical issue was identified in the pipeline".to_string()),
            ("assistant".to_string(), "In conclusion, we should refactor the module".to_string()),
            ("user".to_string(), "Therefore, the next step is clear".to_string()),
        ];
        let hl = SessionMcpManager::extract_highlights("s-hl", &messages);
        assert_eq!(hl.highlights.len(), 2); // "important" and "critical"
        assert_eq!(hl.conclusions.len(), 2); // "in conclusion" and "therefore"
    }

    #[test]
    fn test_extract_highlights_action_items() {
        let messages = vec![
            ("user".to_string(), "We need to update the dependencies".to_string()),
            ("assistant".to_string(), "TODO: add error handling for edge cases".to_string()),
            ("user".to_string(), "The weather is nice today".to_string()),
            ("assistant".to_string(), "Action item: review the PR before merging".to_string()),
        ];
        let hl = SessionMcpManager::extract_highlights("s-ai", &messages);
        assert_eq!(hl.action_items.len(), 3); // "need to", "todo", "action item"
        assert!(hl.highlights.is_empty());
    }

    #[test]
    fn test_extract_context_entities() {
        let messages = vec![
            ("user".to_string(), "The system uses Rust and PostgreSQL for storage".to_string()),
            ("assistant".to_string(), "Indeed, Rust provides great performance with PostgreSQL".to_string()),
        ];
        let ctx = SessionMcpManager::extract_context("s-ctx", &messages);
        // Entities should include capitalized words (not sentence-initial)
        assert!(ctx.entities.contains(&"Rust".to_string()));
        assert!(ctx.entities.contains(&"PostgreSQL".to_string()));
    }

    #[test]
    fn test_extract_beliefs() {
        let messages = vec![
            ("user".to_string(), "I think Rust is the best language for this project".to_string()),
            ("assistant".to_string(), "I believe the architecture is sound and well-designed".to_string()),
            ("user".to_string(), "We should add more comprehensive tests to the suite".to_string()),
            ("user".to_string(), "The sky is blue".to_string()), // no belief pattern
        ];
        let beliefs = SessionMcpManager::extract_beliefs("s-bel", &messages);
        assert_eq!(beliefs.beliefs.len(), 3);
        assert_eq!(beliefs.beliefs[0].belief_type, "opinion"); // "i think"
        assert!((beliefs.beliefs[0].confidence - 0.6).abs() < f32::EPSILON);
        assert_eq!(beliefs.beliefs[1].belief_type, "conviction"); // "i believe"
        assert_eq!(beliefs.beliefs[2].belief_type, "recommendation"); // "we should"
    }

    #[test]
    fn test_repair_session_valid_json() {
        let raw = r#"[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]"#;
        let result = SessionMcpManager::repair_session("s-ok", raw);
        assert!(result.success);
        assert_eq!(result.messages_recovered, 2);
        assert_eq!(result.messages_lost, 0);
        assert_eq!(result.repair_notes.len(), 1);
        assert!(result.repair_notes[0].contains("JSON array"));
    }

    #[test]
    fn test_repair_session_partial() {
        let raw = r#"{"role":"user","content":"hello"}
{"role":"assistant","content":"hi"}
{"role":"user","content":"broken"
this is garbage"#;
        let result = SessionMcpManager::repair_session("s-partial", raw);
        assert!(result.success);
        assert_eq!(result.messages_recovered, 3); // 2 valid + 1 repaired (missing brace)
        assert_eq!(result.messages_lost, 1); // "this is garbage"
    }

    #[test]
    fn test_repair_session_corrupted() {
        let raw = "not json at all\njust plain text\nnothing useful";
        let result = SessionMcpManager::repair_session("s-bad", raw);
        assert!(!result.success);
        assert_eq!(result.messages_recovered, 0);
        assert_eq!(result.messages_lost, 3);
        assert!(result.repair_notes.iter().any(|n| n.contains("No messages could be recovered")));
    }

    #[test]
    fn test_session_beliefs_serde() {
        let beliefs = SessionBeliefs {
            session_id: "s-serde".to_string(),
            beliefs: vec![
                SessionBelief {
                    statement: "I think this works".to_string(),
                    belief_type: "opinion".to_string(),
                    confidence: 0.6,
                },
            ],
        };
        let json = serde_json::to_string(&beliefs).unwrap();
        let back: SessionBeliefs = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-serde");
        assert_eq!(back.beliefs.len(), 1);
        assert_eq!(back.beliefs[0].statement, "I think this works");
        assert!((back.beliefs[0].confidence - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_session_context_serde() {
        let ctx = SessionContext {
            session_id: "s-ctx-ser".to_string(),
            entities: vec!["Rust".to_string(), "PostgreSQL".to_string()],
            relations: vec![
                ("Rust".to_string(), "is".to_string(), "fast".to_string()),
            ],
            key_facts: vec!["Rust is a systems programming language with good safety".to_string()],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let back: SessionContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-ctx-ser");
        assert_eq!(back.entities.len(), 2);
        assert_eq!(back.relations.len(), 1);
        assert_eq!(back.relations[0].0, "Rust");
        assert_eq!(back.relations[0].1, "is");
        assert_eq!(back.relations[0].2, "fast");
    }

    #[test]
    fn test_session_repair_result_fields() {
        let result = SessionRepairResult {
            session_id: "s-fields".to_string(),
            success: true,
            messages_recovered: 10,
            messages_lost: 2,
            repair_notes: vec!["note1".to_string(), "note2".to_string()],
        };
        assert_eq!(result.session_id, "s-fields");
        assert!(result.success);
        assert_eq!(result.messages_recovered, 10);
        assert_eq!(result.messages_lost, 2);
        assert_eq!(result.repair_notes.len(), 2);

        // Also test serde round-trip
        let json = serde_json::to_string(&result).unwrap();
        let back: SessionRepairResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages_recovered, 10);
        assert_eq!(back.messages_lost, 2);
    }

    // =========================================================================
    // MCP v2 Phase 2 tests (v5 roadmap: items 2.1, 2.2, 2.3)
    // =========================================================================

    // --- 2.1 Streamable HTTP Transport ---

    #[test]
    fn test_transport_mode_detect_json() {
        let mode = StreamableHttpTransport::detect_transport("application/json");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_json_charset() {
        let mode = StreamableHttpTransport::detect_transport("application/json; charset=utf-8");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_sse() {
        let mode = StreamableHttpTransport::detect_transport("text/event-stream");
        assert_eq!(mode, TransportMode::SSE);
    }

    #[test]
    fn test_transport_mode_detect_unknown() {
        let mode = StreamableHttpTransport::detect_transport("text/html");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_case_insensitive() {
        let mode = StreamableHttpTransport::detect_transport("TEXT/EVENT-STREAM");
        assert_eq!(mode, TransportMode::SSE);
    }

    #[test]
    fn test_transport_mode_serde() {
        let json = serde_json::to_value(TransportMode::StreamableHTTP).unwrap();
        assert_eq!(json, "streamable_http");
        let json2 = serde_json::to_value(TransportMode::SSE).unwrap();
        assert_eq!(json2, "sse");
        let json3 = serde_json::to_value(TransportMode::StdIO).unwrap();
        assert_eq!(json3, "std_io");
    }

    #[test]
    fn test_streamable_http_transport_new() {
        let transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        assert_eq!(transport.base_url(), "http://localhost:8080/mcp");
        assert_eq!(transport.mode(), TransportMode::StreamableHTTP);
        assert!(transport.get_session_id().is_none());
    }

    #[test]
    fn test_streamable_http_transport_session_id() {
        let mut transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        assert!(transport.get_session_id().is_none());

        transport.set_session_id("sess-abc-123".to_string());
        assert_eq!(transport.get_session_id(), Some("sess-abc-123"));
    }

    #[test]
    fn test_streamable_http_transport_send_request_returns_error() {
        let mut transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        let request = McpRequest::new("tools/list").with_id(1u64);
        let result = transport.send_request(&request);
        // With real HTTP transport, connecting to localhost:8080 (no server)
        // should fail with a connection error.
        assert!(result.is_err());
    }

    // --- Session Store ---

    #[test]
    fn test_in_memory_session_store_create() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        assert!(session.session_id.starts_with("mcp-session-"));
        assert!(!session.session_id.is_empty());
    }

    #[test]
    fn test_in_memory_session_store_create_multiple() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let s2 = store.create_session();
        let s3 = store.create_session();
        assert_ne!(s1.session_id, s2.session_id);
        assert_ne!(s2.session_id, s3.session_id);
        assert_eq!(store.list_sessions().len(), 3);
    }

    #[test]
    fn test_in_memory_session_store_get() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        let id = session.session_id.clone();

        let retrieved = store.get_session(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().session_id, id);

        assert!(store.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_in_memory_session_store_touch() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        let id = session.session_id.clone();
        let original_last_active = session.last_active;

        // Small sleep to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        store.touch_session(&id);
        let touched = store.get_session(&id).unwrap();
        assert!(touched.last_active >= original_last_active);
    }

    #[test]
    fn test_in_memory_session_store_delete() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let s2 = store.create_session();
        let id1 = s1.session_id.clone();

        assert_eq!(store.list_sessions().len(), 2);
        store.delete_session(&id1);
        assert_eq!(store.list_sessions().len(), 1);
        assert!(store.get_session(&id1).is_none());
        assert!(store.get_session(&s2.session_id).is_some());
    }

    #[test]
    fn test_in_memory_session_store_list() {
        let mut store = InMemorySessionStore::new();
        assert!(store.list_sessions().is_empty());

        store.create_session();
        store.create_session();
        assert_eq!(store.list_sessions().len(), 2);
    }

    #[test]
    fn test_in_memory_session_store_cleanup_expired() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let id1 = s1.session_id.clone();

        // Manually backdate the session to make it "expired"
        if let Some(session) = store.sessions.get_mut(&id1) {
            session.last_active = chrono::Utc::now() - chrono::Duration::seconds(120);
        }

        // Create a fresh session
        let _s2 = store.create_session();
        assert_eq!(store.list_sessions().len(), 2);

        // Cleanup sessions older than 60 seconds
        store.cleanup_expired(60);
        assert_eq!(store.list_sessions().len(), 1);
        assert!(store.get_session(&id1).is_none());
    }

    #[test]
    fn test_mcp_session_serde() {
        let session = McpSession {
            session_id: "s-test".to_string(),
            created_at: chrono::Utc::now(),
            last_active: chrono::Utc::now(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".to_string(), "value".to_string());
                m
            },
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: McpSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-test");
        assert_eq!(back.metadata.get("key").map(|v| v.as_str()), Some("value"));
    }

    // --- 2.2 OAuth 2.1 + PKCE ---

    #[test]
    fn test_sha256_known_vectors() {
        // SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
        let hash = v2_oauth::sha256_hash(b"abc");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = v2_oauth::sha256_hash(b"");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_longer_input() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let hash = v2_oauth::sha256_hash(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_base64url_encode_basic() {
        // Known value: base64url of [0, 1, 2, 3] should produce "AAECAT"
        // Standard base64: AAECAw== -> base64url without padding: AAECAw
        let encoded = v2_oauth::base64url_encode(&[0, 1, 2, 3]);
        assert_eq!(encoded, "AAECAw");
    }

    #[test]
    fn test_base64url_no_plus_or_slash() {
        // base64url must not contain + or / (unlike standard base64)
        let data: Vec<u8> = (0..=255).collect();
        let encoded = v2_oauth::base64url_encode(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_pkce_challenge_generate() {
        let pkce = PkceChallenge::generate();
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
        assert_eq!(pkce.method, "S256");
        // Verifier should be at least 43 chars per RFC 7636
        assert!(pkce.verifier.len() >= 43);
    }

    #[test]
    fn test_pkce_from_verifier_deterministic() {
        let pkce1 = PkceChallenge::from_verifier("test-verifier-12345678901234567890123456789");
        let pkce2 = PkceChallenge::from_verifier("test-verifier-12345678901234567890123456789");
        assert_eq!(pkce1.challenge, pkce2.challenge);
        assert_eq!(pkce1.verifier, pkce2.verifier);
        assert_eq!(pkce1.method, "S256");
    }

    #[test]
    fn test_pkce_different_verifiers_different_challenges() {
        let pkce1 = PkceChallenge::from_verifier("verifier-aaa");
        let pkce2 = PkceChallenge::from_verifier("verifier-bbb");
        assert_ne!(pkce1.challenge, pkce2.challenge);
    }

    #[test]
    fn test_pkce_challenge_is_base64url() {
        let pkce = PkceChallenge::from_verifier("my-test-verifier");
        // base64url characters only: A-Z a-z 0-9 - _
        assert!(pkce
            .challenge
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }

    fn make_test_v2_oauth_config() -> McpV2OAuthConfig {
        McpV2OAuthConfig {
            authorization_endpoint: "https://auth.example.com/authorize".to_string(),
            token_endpoint: "https://auth.example.com/token".to_string(),
            client_id: Some("test-client-v2".to_string()),
            client_secret: Some("test-secret-v2".to_string()),
            scopes: vec!["mcp:tools".to_string(), "mcp:resources".to_string()],
            redirect_uri: "http://localhost:9090/callback".to_string(),
        }
    }

    #[test]
    fn test_oauth_token_manager_new() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        assert!(manager.current_token().is_none());
        assert!(manager.is_token_expired());
    }

    #[test]
    fn test_get_authorization_url() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        let (url, pkce) = manager.get_authorization_url();

        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("client_id=test-client-v2"));
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
    }

    #[test]
    fn test_exchange_code() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");

        let token = manager.exchange_code("auth-code-xyz", &pkce).unwrap();
        assert_eq!(token.access_token, "access-auth-code-xyz");
        assert_eq!(token.token_type, "Bearer");
        assert!(token.expires_at.is_some());
        assert!(token.refresh_token.is_some());
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_exchange_code_empty_code() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");
        let result = manager.exchange_code("", &pkce);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_exchange_code_empty_verifier() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge {
            verifier: String::new(),
            challenge: "abc".to_string(),
            method: "S256".to_string(),
        };
        let result = manager.exchange_code("code", &pkce);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("verifier"));
    }

    #[test]
    fn test_refresh_token_flow() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");

        // First get a token
        manager.exchange_code("code-1", &pkce).unwrap();

        // Now refresh
        let refreshed = manager.refresh_token().unwrap();
        assert!(refreshed.access_token.starts_with("refreshed-"));
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_refresh_token_no_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let result = manager.refresh_token();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No refresh token"));
    }

    #[test]
    fn test_is_token_expired_no_token() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        assert!(manager.is_token_expired());
    }

    #[test]
    fn test_is_token_expired_valid_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "valid".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: None,
            scope: None,
        });
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_is_token_expired_no_expiry() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "permanent".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: None,
            refresh_token: None,
            scope: None,
        });
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_get_valid_token_no_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let result = manager.get_valid_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_valid_token_with_valid() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "good-token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some("refresh-1".to_string()),
            scope: None,
        });
        let token = manager.get_valid_token().unwrap();
        assert_eq!(token.access_token, "good-token");
    }

    #[test]
    fn test_oauth_token_serde() {
        let token = OAuthToken {
            access_token: "tok-abc".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now()),
            refresh_token: Some("ref-def".to_string()),
            scope: Some("mcp:tools".to_string()),
        };
        let json = serde_json::to_string(&token).unwrap();
        let back: OAuthToken = serde_json::from_str(&json).unwrap();
        assert_eq!(back.access_token, "tok-abc");
        assert_eq!(back.token_type, "Bearer");
        assert!(back.expires_at.is_some());
        assert_eq!(back.refresh_token.as_deref(), Some("ref-def"));
        assert_eq!(back.scope.as_deref(), Some("mcp:tools"));
    }

    // --- Authorization Server Metadata ---

    #[test]
    fn test_authorization_server_metadata_discover() {
        let metadata = AuthorizationServerMetadata::discover("https://auth.example.com").unwrap();
        assert_eq!(metadata.issuer, "https://auth.example.com");
        assert_eq!(
            metadata.authorization_endpoint,
            "https://auth.example.com/authorize"
        );
        assert_eq!(
            metadata.token_endpoint,
            "https://auth.example.com/token"
        );
        assert!(metadata.registration_endpoint.is_some());
        assert!(!metadata.scopes_supported.is_empty());
    }

    #[test]
    fn test_authorization_server_metadata_serde() {
        let metadata = AuthorizationServerMetadata {
            issuer: "https://example.com".to_string(),
            authorization_endpoint: "https://example.com/auth".to_string(),
            token_endpoint: "https://example.com/token".to_string(),
            registration_endpoint: None,
            scopes_supported: vec!["scope1".to_string()],
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let back: AuthorizationServerMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.issuer, "https://example.com");
        assert!(back.registration_endpoint.is_none());
        assert_eq!(back.scopes_supported.len(), 1);
    }

    // --- Dynamic Client Registration ---

    #[test]
    fn test_dynamic_client_registration() {
        let (client_id, client_secret) = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "my-app",
            &["http://localhost:8080/callback".to_string()],
        )
        .unwrap();
        assert!(client_id.contains("my-app"));
        assert!(client_secret.is_some());
    }

    #[test]
    fn test_dynamic_client_registration_empty_endpoint() {
        let result = DynamicClientRegistration::register(
            "",
            "my-app",
            &["http://localhost/cb".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_client_registration_empty_name() {
        let result = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "",
            &["http://localhost/cb".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_client_registration_no_redirects() {
        let result = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "my-app",
            &[],
        );
        assert!(result.is_err());
    }

    // --- 2.3 Tool Annotations ---

    #[test]
    fn test_tool_annotations_v2_defaults() {
        let ann = ToolAnnotations::default();
        assert!(!ann.read_only);
        assert!(ann.destructive);
        assert!(!ann.idempotent);
        assert!(ann.open_world);
    }

    #[test]
    fn test_tool_annotations_is_safe() {
        let safe = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        assert!(safe.is_safe());

        let not_safe = ToolAnnotations {
            read_only: true,
            destructive: true,
            ..Default::default()
        };
        assert!(!not_safe.is_safe());

        let not_readonly = ToolAnnotations {
            read_only: false,
            destructive: false,
            ..Default::default()
        };
        assert!(!not_readonly.is_safe());
    }

    #[test]
    fn test_tool_annotations_needs_confirmation() {
        // Default: destructive=true, open_world=true => needs confirmation
        let default_ann = ToolAnnotations::default();
        assert!(default_ann.needs_confirmation());

        // Only destructive
        let destructive_only = ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: false,
        };
        assert!(destructive_only.needs_confirmation());

        // Only open_world
        let open_only = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: true,
        };
        assert!(open_only.needs_confirmation());

        // Neither destructive nor open_world
        let no_confirmation = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        assert!(!no_confirmation.needs_confirmation());
    }

    #[test]
    fn test_tool_annotations_serde_v2() {
        let ann = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        let json = serde_json::to_string(&ann).unwrap();
        let back: ToolAnnotations = serde_json::from_str(&json).unwrap();
        assert!(back.read_only);
        assert!(!back.destructive);
        assert!(back.idempotent);
        assert!(!back.open_world);
    }

    #[test]
    fn test_tool_annotations_serde_defaults_on_missing_fields() {
        // When fields are missing, defaults should apply
        let json = r#"{"read_only": true}"#;
        let ann: ToolAnnotations = serde_json::from_str(json).unwrap();
        assert!(ann.read_only);
        assert!(ann.destructive); // default true
        assert!(!ann.idempotent); // default false
        assert!(ann.open_world); // default true
    }

    #[test]
    fn test_annotated_tool_from_tool() {
        let tool = McpTool::new("search", "Search the web");
        let annotated = AnnotatedTool::from_tool(tool);
        assert_eq!(annotated.tool.name, "search");
        // Default annotations
        assert!(!annotated.annotations.read_only);
        assert!(annotated.annotations.destructive);
    }

    #[test]
    fn test_annotated_tool_with_annotations() {
        let tool = McpTool::new("read_file", "Read a file");
        let ann = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        let annotated = AnnotatedTool::with_annotations(tool, ann);
        assert_eq!(annotated.tool.name, "read_file");
        assert!(annotated.annotations.is_safe());
        assert!(!annotated.annotations.needs_confirmation());
    }

    #[test]
    fn test_annotated_tool_serde() {
        let tool = McpTool::new("deploy", "Deploy to production");
        let ann = ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        };
        let annotated = AnnotatedTool::with_annotations(tool, ann);
        let json = serde_json::to_string(&annotated).unwrap();
        let back: AnnotatedTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool.name, "deploy");
        assert!(back.annotations.destructive);
        assert!(back.annotations.open_world);
    }

    #[test]
    fn test_tool_annotation_registry_register_and_get() {
        let mut registry = ToolAnnotationRegistry::new();
        registry.register(
            "search",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );

        let ann = registry.get("search");
        assert!(ann.is_some());
        let ann = ann.unwrap();
        assert!(ann.read_only);
        assert!(!ann.destructive);

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_tool_annotation_registry_needs_approval() {
        let mut registry = ToolAnnotationRegistry::new();

        // Safe tool -- no approval needed
        registry.register(
            "read_file",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );
        assert!(!registry.needs_approval("read_file"));

        // Destructive tool -- approval needed
        registry.register(
            "delete_file",
            ToolAnnotations {
                read_only: false,
                destructive: true,
                idempotent: false,
                open_world: false,
            },
        );
        assert!(registry.needs_approval("delete_file"));

        // Unknown tool -- approval needed (conservative default)
        assert!(registry.needs_approval("unknown_tool"));
    }

    #[test]
    fn test_tool_annotation_registry_overwrite() {
        let mut registry = ToolAnnotationRegistry::new();
        registry.register("tool1", ToolAnnotations::default());
        assert!(registry.get("tool1").unwrap().destructive);

        // Overwrite with safe annotations
        registry.register(
            "tool1",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );
        assert!(!registry.get("tool1").unwrap().destructive);
    }

    #[test]
    fn test_mcpv2_oauth_config_serde() {
        let config = make_test_v2_oauth_config();
        let json = serde_json::to_string(&config).unwrap();
        let back: McpV2OAuthConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.client_id.as_deref(), Some("test-client-v2"));
        assert_eq!(back.scopes.len(), 2);
        assert_eq!(back.redirect_uri, "http://localhost:9090/callback");
    }

    // -----------------------------------------------------------------------
    // v6 Phase 1 -- Elicitation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_elicit_request_construction() {
        let req = ElicitRequest {
            request_id: "req-1".to_string(),
            message: "Please confirm deployment".to_string(),
            fields: vec![ElicitFieldSchema {
                field_name: "confirm".to_string(),
                field_type: ElicitFieldType::Boolean,
                description: "Do you approve?".to_string(),
                required: true,
                default_value: Some(serde_json::Value::Bool(false)),
            }],
            timeout_ms: Some(30_000),
        };
        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.fields.len(), 1);
        assert_eq!(req.timeout_ms, Some(30_000));
    }

    #[test]
    fn test_elicit_field_type_all_variants() {
        let text = ElicitFieldType::Text;
        let number = ElicitFieldType::Number;
        let boolean = ElicitFieldType::Boolean;
        let select = ElicitFieldType::Select {
            options: vec!["a".into(), "b".into()],
        };
        let file = ElicitFieldType::FileUpload {
            accepted_types: vec!["image/png".into()],
        };

        // Ensure serialization round-trips
        for ft in [text, number, boolean, select, file] {
            let json = serde_json::to_string(&ft).unwrap();
            let _back: ElicitFieldType = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_elicit_action_all_variants() {
        assert_eq!(ElicitAction::Accept, ElicitAction::Accept);
        assert_eq!(ElicitAction::Deny, ElicitAction::Deny);
        assert_eq!(ElicitAction::Dismiss, ElicitAction::Dismiss);
        assert_ne!(ElicitAction::Accept, ElicitAction::Deny);
    }

    #[test]
    fn test_elicit_response_accept_action() {
        let mut vals = HashMap::new();
        vals.insert("name".to_string(), serde_json::json!("Alice"));
        let resp = ElicitResponse {
            request_id: "r-42".to_string(),
            action: ElicitAction::Accept,
            values: vals,
        };
        assert_eq!(resp.action, ElicitAction::Accept);
        assert_eq!(resp.values.get("name").unwrap(), &serde_json::json!("Alice"));
    }

    #[test]
    fn test_auto_accept_handler_returns_accept_with_defaults() {
        let handler = AutoAcceptHandler::new();
        let req = ElicitRequest {
            request_id: "r-1".to_string(),
            message: "confirm?".to_string(),
            fields: vec![ElicitFieldSchema {
                field_name: "ok".to_string(),
                field_type: ElicitFieldType::Boolean,
                description: "ok?".to_string(),
                required: true,
                default_value: Some(serde_json::json!(true)),
            }],
            timeout_ms: None,
        };
        let resp = handler.handle_elicitation(&req);
        assert_eq!(resp.action, ElicitAction::Accept);
        assert_eq!(resp.request_id, "r-1");
        // Should pick up the field's own default_value
        assert_eq!(resp.values.get("ok"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_auto_accept_handler_with_custom_defaults() {
        let mut defaults = HashMap::new();
        defaults.insert("region".to_string(), serde_json::json!("eu-west-1"));
        let handler = AutoAcceptHandler::with_defaults(defaults);

        let req = ElicitRequest {
            request_id: "r-2".to_string(),
            message: "select region".to_string(),
            fields: vec![
                ElicitFieldSchema {
                    field_name: "region".to_string(),
                    field_type: ElicitFieldType::Text,
                    description: "AWS region".to_string(),
                    required: true,
                    default_value: Some(serde_json::json!("us-east-1")),
                },
                ElicitFieldSchema {
                    field_name: "size".to_string(),
                    field_type: ElicitFieldType::Number,
                    description: "Instance size".to_string(),
                    required: false,
                    default_value: None,
                },
            ],
            timeout_ms: Some(5_000),
        };
        let resp = handler.handle_elicitation(&req);
        assert_eq!(resp.action, ElicitAction::Accept);
        // Handler default overrides field default
        assert_eq!(
            resp.values.get("region"),
            Some(&serde_json::json!("eu-west-1"))
        );
        // "size" has no handler default and no field default -> absent
        assert!(resp.values.get("size").is_none());
    }

    #[test]
    fn test_elicit_field_schema_required_optional() {
        let required_field = ElicitFieldSchema {
            field_name: "username".to_string(),
            field_type: ElicitFieldType::Text,
            description: "Your username".to_string(),
            required: true,
            default_value: None,
        };
        assert!(required_field.required);
        assert!(required_field.default_value.is_none());

        let optional_field = ElicitFieldSchema {
            field_name: "nickname".to_string(),
            field_type: ElicitFieldType::Text,
            description: "Optional nickname".to_string(),
            required: false,
            default_value: Some(serde_json::json!("anon")),
        };
        assert!(!optional_field.required);
        assert_eq!(optional_field.default_value, Some(serde_json::json!("anon")));
    }

    // -----------------------------------------------------------------------
    // v6 Phase 1 -- Audio Content tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_content_construction_and_serialization() {
        let audio = AudioContent {
            data: "dGVzdA==".to_string(),
            mime_type: "audio/wav".to_string(),
            transcript: Some("hello world".to_string()),
            duration_ms: Some(1500),
        };
        let json = serde_json::to_string(&audio).unwrap();
        assert!(json.contains("dGVzdA=="));
        assert!(json.contains("audio/wav"));
        let back: AudioContent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data, "dGVzdA==");
        assert_eq!(back.transcript.as_deref(), Some("hello world"));
    }

    #[test]
    fn test_mcp_content_audio_variant_creation() {
        let content = McpContent::Audio {
            audio: AudioContent {
                data: "AAAA".to_string(),
                mime_type: "audio/ogg".to_string(),
                transcript: None,
                duration_ms: None,
            },
        };
        if let McpContent::Audio { ref audio } = content {
            assert_eq!(audio.mime_type, "audio/ogg");
        } else {
            panic!("Expected Audio variant");
        }
    }

    #[test]
    fn test_audio_content_with_transcript() {
        let audio = AudioContent {
            data: "YXVkaW8=".to_string(),
            mime_type: "audio/mp3".to_string(),
            transcript: Some("Testing one two three".to_string()),
            duration_ms: Some(3200),
        };
        assert!(audio.transcript.is_some());
        assert_eq!(audio.transcript.unwrap(), "Testing one two three");
        assert_eq!(audio.duration_ms, Some(3200));
    }

    #[test]
    fn test_audio_content_without_transcript() {
        let audio = AudioContent {
            data: "YXVkaW8=".to_string(),
            mime_type: "audio/wav".to_string(),
            transcript: None,
            duration_ms: None,
        };
        assert!(audio.transcript.is_none());
        assert!(audio.duration_ms.is_none());
    }

    // -----------------------------------------------------------------------
    // v6 Phase 1 -- JSON-RPC Batching tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_config_default_values() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 50);
        assert!(cfg.parallel_execution);
        assert_eq!(cfg.timeout_per_request_ms, 30_000);
    }

    #[test]
    fn test_batch_config_custom_values() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            parallel_execution: false,
            timeout_per_request_ms: 5_000,
        };
        assert_eq!(cfg.max_batch_size, 10);
        assert!(!cfg.parallel_execution);
        assert_eq!(cfg.timeout_per_request_ms, 5_000);
    }

    #[test]
    fn test_batch_executor_validate_empty_batch() {
        let executor = BatchExecutor::with_defaults();
        let batch = BatchRequest {
            requests: vec![],
            config: BatchConfig::default(),
        };
        let result = executor.validate_batch(&batch);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one request"));
    }

    #[test]
    fn test_batch_executor_validate_oversized_batch() {
        let executor = BatchExecutor::new(BatchConfig {
            max_batch_size: 2,
            parallel_execution: true,
            timeout_per_request_ms: 1_000,
        });
        let requests = vec![
            JsonRpcRequest::new(serde_json::json!(1), "method_a"),
            JsonRpcRequest::new(serde_json::json!(2), "method_b"),
            JsonRpcRequest::new(serde_json::json!(3), "method_c"),
        ];
        let batch = BatchRequest {
            requests,
            config: BatchConfig::default(),
        };
        let result = executor.validate_batch(&batch);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn test_batch_executor_validate_valid_batch() {
        let executor = BatchExecutor::with_defaults();
        let requests = vec![
            JsonRpcRequest::new(serde_json::json!(1), "tools/list"),
            JsonRpcRequest::new(serde_json::json!(2), "resources/list"),
        ];
        let batch = BatchRequest {
            requests,
            config: BatchConfig::default(),
        };
        assert!(executor.validate_batch(&batch).is_ok());
    }

    #[test]
    fn test_batch_executor_create_batch() {
        let executor = BatchExecutor::with_defaults();
        let requests = vec![
            JsonRpcRequest::new(serde_json::json!(1), "ping"),
        ];
        let batch = executor.create_batch(requests);
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.config.max_batch_size, 50);
    }

    #[test]
    fn test_batch_executor_correlate_responses_matches_by_id() {
        let executor = BatchExecutor::with_defaults();
        let requests = vec![
            JsonRpcRequest::new(serde_json::json!(1), "a"),
            JsonRpcRequest::new(serde_json::json!(2), "b"),
        ];
        let batch = executor.create_batch(requests);

        let responses = vec![
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"ok": true})),
            JsonRpcResponse::success(serde_json::json!(2), serde_json::json!({"ok": true})),
            // Unrelated response should be filtered out
            JsonRpcResponse::success(serde_json::json!(99), serde_json::json!({"extra": true})),
        ];

        let result = executor.correlate_responses(&batch, responses);
        assert_eq!(result.responses.len(), 2);
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_batch_response_error_count() {
        let executor = BatchExecutor::with_defaults();
        let requests = vec![
            JsonRpcRequest::new(serde_json::json!(1), "a"),
            JsonRpcRequest::new(serde_json::json!(2), "b"),
        ];
        let batch = executor.create_batch(requests);

        let responses = vec![
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!(null)),
            JsonRpcResponse::error(
                serde_json::json!(2),
                McpError {
                    code: -32600,
                    message: "Invalid request".to_string(),
                    data: None,
                },
            ),
        ];

        let result = executor.correlate_responses(&batch, responses);
        assert_eq!(result.errors, 1);
        assert_eq!(result.responses.len(), 2);
    }

    // -----------------------------------------------------------------------
    // v6 Phase 1 -- Completions & Suggestions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_completion_request_construction() {
        let req = CompletionRequest {
            ref_type: CompletionRefType::ResourceUri("file:///tmp".to_string()),
            argument_name: "path".to_string(),
            partial_value: "/usr".to_string(),
        };
        assert_eq!(req.argument_name, "path");
        assert_eq!(req.partial_value, "/usr");
    }

    #[test]
    fn test_completion_ref_type_variants() {
        let uri = CompletionRefType::ResourceUri("file:///data".to_string());
        let prompt = CompletionRefType::PromptName("summarize".to_string());
        assert_ne!(uri, prompt);
        if let CompletionRefType::ResourceUri(ref s) = uri {
            assert_eq!(s, "file:///data");
        }
        if let CompletionRefType::PromptName(ref s) = prompt {
            assert_eq!(s, "summarize");
        }
    }

    #[test]
    fn test_completion_suggestion_construction() {
        let s = CompletionSuggestion {
            value: "hello".to_string(),
            label: Some("Hello World".to_string()),
            description: Some("A greeting".to_string()),
        };
        assert_eq!(s.value, "hello");
        assert_eq!(s.label.as_deref(), Some("Hello World"));
        assert_eq!(s.description.as_deref(), Some("A greeting"));
    }

    #[test]
    fn test_completion_result_with_suggestions() {
        let result = CompletionResult {
            suggestions: vec![
                CompletionSuggestion {
                    value: "foo".to_string(),
                    label: None,
                    description: None,
                },
                CompletionSuggestion {
                    value: "bar".to_string(),
                    label: None,
                    description: None,
                },
            ],
            has_more: false,
            total: Some(2),
        };
        assert_eq!(result.suggestions.len(), 2);
        assert!(!result.has_more);
        assert_eq!(result.total, Some(2));
    }

    #[test]
    fn test_static_completion_provider_add_values_and_complete() {
        let mut provider = StaticCompletionProvider::new();
        provider.add_values("lang".to_string(), vec!["Rust".into(), "Ruby".into(), "Python".into()]);

        let req = CompletionRequest {
            ref_type: CompletionRefType::PromptName("code".to_string()),
            argument_name: "lang".to_string(),
            partial_value: "Ru".to_string(),
        };
        let result = provider.complete(&req);
        assert_eq!(result.suggestions.len(), 2);
        let vals: Vec<&str> = result.suggestions.iter().map(|s| s.value.as_str()).collect();
        assert!(vals.contains(&"Rust"));
        assert!(vals.contains(&"Ruby"));
    }

    #[test]
    fn test_static_completion_provider_case_insensitive_prefix() {
        let mut provider = StaticCompletionProvider::new();
        provider.add_values("color".to_string(), vec!["Red".into(), "Blue".into(), "Green".into()]);

        let req = CompletionRequest {
            ref_type: CompletionRefType::ResourceUri("x".to_string()),
            argument_name: "color".to_string(),
            partial_value: "re".to_string(), // lowercase prefix matches "Red"
        };
        let result = provider.complete(&req);
        assert_eq!(result.suggestions.len(), 1);
        assert_eq!(result.suggestions[0].value, "Red");
    }

    #[test]
    fn test_static_completion_provider_empty_partial_returns_all() {
        let mut provider = StaticCompletionProvider::new();
        provider.add_values("item".to_string(), vec!["A".into(), "B".into(), "C".into()]);

        let req = CompletionRequest {
            ref_type: CompletionRefType::PromptName("p".to_string()),
            argument_name: "item".to_string(),
            partial_value: "".to_string(),
        };
        let result = provider.complete(&req);
        assert_eq!(result.suggestions.len(), 3);
    }

    #[test]
    fn test_static_completion_provider_no_match_returns_empty() {
        let mut provider = StaticCompletionProvider::new();
        provider.add_values("fruit".to_string(), vec!["Apple".into(), "Banana".into()]);

        let req = CompletionRequest {
            ref_type: CompletionRefType::PromptName("p".to_string()),
            argument_name: "fruit".to_string(),
            partial_value: "Zz".to_string(),
        };
        let result = provider.complete(&req);
        assert!(result.suggestions.is_empty());
    }

    #[test]
    fn test_completion_registry_no_providers() {
        let registry = CompletionRegistry::new();
        assert_eq!(registry.provider_count(), 0);

        let req = CompletionRequest {
            ref_type: CompletionRefType::PromptName("p".to_string()),
            argument_name: "x".to_string(),
            partial_value: "".to_string(),
        };
        let result = registry.complete(&req);
        assert!(result.suggestions.is_empty());
    }

    #[test]
    fn test_completion_registry_multiple_providers_merges_results() {
        let mut registry = CompletionRegistry::new();

        let mut p1 = StaticCompletionProvider::new();
        p1.add_values("arg".to_string(), vec!["Alpha".into(), "Apex".into()]);
        registry.register(Box::new(p1));

        let mut p2 = StaticCompletionProvider::new();
        p2.add_values("arg".to_string(), vec!["Ant".into(), "Bear".into()]);
        registry.register(Box::new(p2));

        assert_eq!(registry.provider_count(), 2);

        let req = CompletionRequest {
            ref_type: CompletionRefType::PromptName("p".to_string()),
            argument_name: "arg".to_string(),
            partial_value: "A".to_string(),
        };
        let result = registry.complete(&req);
        // p1 contributes Alpha, Apex; p2 contributes Ant (Bear doesn't match)
        assert_eq!(result.suggestions.len(), 3);
        let vals: Vec<&str> = result.suggestions.iter().map(|s| s.value.as_str()).collect();
        assert!(vals.contains(&"Alpha"));
        assert!(vals.contains(&"Apex"));
        assert!(vals.contains(&"Ant"));
    }

    #[test]
    fn test_completion_result_has_more_flag() {
        let result = CompletionResult {
            suggestions: vec![CompletionSuggestion {
                value: "partial".to_string(),
                label: None,
                description: None,
            }],
            has_more: true,
            total: Some(100),
        };
        assert!(result.has_more);
        assert_eq!(result.total, Some(100));
        assert_eq!(result.suggestions.len(), 1);
    }
