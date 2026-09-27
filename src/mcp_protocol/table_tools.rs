// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! MCP tools for asking questions of tabular data.
//!
//! Three read-only tools over an already-loaded [`TableEngine`]: list what is
//! there, describe a table's columns, run one SELECT.
//!
//! # There is deliberately no `load_table`
//!
//! The caller loads the files; the model can only query what was loaded. A
//! `load_table` tool would let any prompt name any path on the machine and read
//! the contents back through query results — the model does not need that to do
//! its job, and the blast radius of getting it wrong is every CSV on the disk.
//!
//! So the security boundary is **which files the host program chose to load**, and
//! it is enforced by the tool not existing rather than by validating a path.
//!
//! # Why a model needs `describe_table` before `query_table`
//!
//! A model that does not know the column names invents them, and invented SQL
//! that happens to parse returns a number with complete confidence. `describe`
//! also reports which columns are mixed numeric, which is what stops the model
//! trusting an `AVG` it should not — see [`ColumnInfo::is_mixed_numeric`].
//!
//! # The disclosure travels
//!
//! Every `query_table` response carries `sql_executed`, `row_count`, `truncated`,
//! `warnings` and `trustworthy`. Those are required fields on [`QueryResult`], and
//! dropping them here would undo the one design decision the tabular module is
//! built around: a wrong query that executes is worse than one that fails.

use std::sync::{Arc, Mutex};

use super::server::McpServer;
use super::types::{McpTool, McpToolAnnotation};
use crate::tabular::{Cell, QueryResult, TableEngine, TableInfo};

/// Rows a single `query_table` call may return, whatever the model asks for.
///
/// The point of this module is that the **table** never enters the prompt, only
/// the answer to a question about it. A model that asks for 100 000 rows either
/// misunderstood the tool or is being driven by something that did, and honouring
/// it would blow the context window in one call. 200 is generous for an aggregate
/// and useless for exfiltration.
pub const MAX_ROWS: usize = 200;

/// Rows returned when the caller does not say.
pub const DEFAULT_ROWS: usize = 25;

/// A [`TableEngine`] shared between tool handlers.
///
/// `Mutex` and not `RwLock`: `TableEngine::load_with` takes `&mut self`, and even
/// the read-only methods go through the same handle. Contention is not a concern
/// for a tool a model calls a few times per turn.
pub type SharedEngine = Arc<Mutex<Box<dyn TableEngine + Send>>>;

/// Wrap an engine for use by these tools.
pub fn shared<E: TableEngine + Send + 'static>(engine: E) -> SharedEngine {
    Arc::new(Mutex::new(Box::new(engine)))
}

/// Register the three tabular tools on a server.
///
/// # Tools registered
/// - `list_tables` — what is loaded, with row counts and column names
/// - `describe_table` — one table's columns, types, and which ones are unsafe to average
/// - `query_table` — one read-only SELECT, with the SQL and the warnings echoed back
///
/// # Arguments
/// * `server` — the MCP server to register on
/// * `engine` — an engine with the tables **already loaded** (see the module docs
///   for why loading is not exposed)
pub fn register_table_tools(server: &mut McpServer, engine: SharedEngine) {
    register_list_tables(server, engine.clone());
    register_describe_table(server, engine.clone());
    register_query_table(server, engine);
}

/// One cell as JSON.
///
/// `Cell::Float` is the awkward one: JSON has no NaN or infinity, and
/// `serde_json::Number::from_f64` returns `None` for both. Emitting `null` there
/// would turn "the computation produced infinity" into "there was no value",
/// which are different answers — so it becomes the string instead, which is ugly
/// and honest.
fn cell_to_json(cell: &Cell) -> serde_json::Value {
    match cell {
        Cell::Null => serde_json::Value::Null,
        Cell::Int(v) => serde_json::json!(v),
        Cell::Float(v) => match serde_json::Number::from_f64(*v) {
            Some(n) => serde_json::Value::Number(n),
            None => serde_json::Value::String(v.to_string()),
        },
        Cell::Text(v) => serde_json::Value::String(v.clone()),
        Cell::Bool(v) => serde_json::json!(v),
    }
}

fn table_to_json(info: &TableInfo) -> serde_json::Value {
    serde_json::json!({
        "name": info.name,
        "source": info.source.display().to_string(),
        "row_count": info.row_count,
        "columns": info.columns.iter().map(|c| serde_json::json!({
            "name": c.name,
            "type": c.declared_type,
            "numeric_values": c.numeric_values,
            "mixed_numeric": c.is_mixed_numeric(),
            "non_numbers_nullified": c.non_numbers_nullified,
            "values_that_are_not_numbers": c.unparseable.iter()
                .map(|(v, n)| serde_json::json!({"value": v, "count": n}))
                .collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn result_to_json(result: &QueryResult) -> serde_json::Value {
    serde_json::json!({
        // First, because it is the field a human reviewing the answer needs and
        // the one a model is least likely to volunteer.
        "sql_executed": result.sql_executed,
        "columns": result.columns,
        "rows": result.rows.iter()
            .map(|row| row.iter().map(cell_to_json).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "row_count": result.row_count,
        "truncated": result.truncated,
        "warnings": result.warnings,
        "trustworthy": result.is_trustworthy(),
    })
}

fn register_list_tables(server: &mut McpServer, engine: SharedEngine) {
    server.register_tool(
        McpTool::new(
            "list_tables",
            "List the tabular data files loaded and queryable. Returns each table's name (use it \
             in SQL), source file, row count and column names. Call this first: querying a table \
             that is not loaded fails, and the names are not guessable.",
        )
        .with_annotations(McpToolAnnotation {
            title: Some("List Loaded Tables".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |_args| {
            let guard = engine
                .lock()
                .map_err(|e| format!("Table engine lock poisoned: {e}"))?;
            let tables = guard.list_tables();
            Ok(serde_json::json!({
                "engine": guard.engine_name(),
                "tables": tables.iter().map(table_to_json).collect::<Vec<_>>(),
                "total_tables": tables.len(),
            }))
        },
    );
}

fn register_describe_table(server: &mut McpServer, engine: SharedEngine) {
    server.register_tool(
        McpTool::new(
            "describe_table",
            "Describe one loaded table: its columns, the type each was read as, and which columns \
             hold numbers MIXED with something that is not a number. Call this before query_table. \
             A column with mixed_numeric=true is the one to be careful with: check \
             non_numbers_nullified to know whether AVG and COUNT over it are right.",
        )
        .with_property(
            "table",
            "string",
            "Table name, as listed by list_tables",
            true,
        )
        .with_annotations(McpToolAnnotation {
            title: Some("Describe Table Columns".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |args| {
            let table = args
                .get("table")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: table")?;
            let guard = engine
                .lock()
                .map_err(|e| format!("Table engine lock poisoned: {e}"))?;
            // The engine's own error already names the tables that DO exist,
            // which is the difference between a model retrying usefully and
            // guessing again.
            let info = guard.describe(table).map_err(|e| e.to_string())?;
            Ok(table_to_json(&info))
        },
    );
}

fn register_query_table(server: &mut McpServer, engine: SharedEngine) {
    server.register_tool(
        McpTool::new(
            "query_table",
            "Run ONE read-only SQL query over the loaded tables and return the rows. Only \
             SELECT, WITH and VALUES are accepted; anything that writes, attaches a file or adds \
             a second statement is refused. The response always includes the SQL that ran, the \
             row count, whether the limit truncated it, and warnings — read the warnings before \
             trusting a number.",
        )
        .with_property(
            "sql",
            "string",
            "A single read-only SQL statement. Use column names from describe_table. Give every \
             aggregate an alias (SELECT SUM(x) AS total): two unaliased aggregates over one \
             column are an error on the Polars engine.",
            true,
        )
        .with_property(
            "limit",
            "integer",
            "Maximum rows to return (default 25, hard maximum 200). Prefer an aggregate over a \
             large row dump: the table must not enter the prompt, only the answer.",
            false,
        )
        .with_annotations(McpToolAnnotation {
            title: Some("Query Tables with SQL".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |args| {
            let sql = args
                .get("sql")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: sql")?;
            // Clamped and not rejected: a model asking for 10 000 rows should get
            // its answer plus `truncated: true`, which teaches it something. An
            // error teaches it to retry with 9 999.
            let requested = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_ROWS as u64) as usize;
            let limit = requested.clamp(1, MAX_ROWS);

            let guard = engine
                .lock()
                .map_err(|e| format!("Table engine lock poisoned: {e}"))?;
            let result = guard.query(sql, limit).map_err(|e| e.to_string())?;
            let mut json = result_to_json(&result);
            if requested > MAX_ROWS {
                // Said out loud, because otherwise `truncated: true` looks like the
                // data ran out rather than like a cap the tool applied.
                if let Some(obj) = json.as_object_mut() {
                    obj.insert(
                        "limit_applied".to_string(),
                        serde_json::json!({
                            "requested": requested,
                            "used": limit,
                            "reason": "query_table caps rows at MAX_ROWS so a table cannot be \
                                       pulled into the prompt",
                        }),
                    );
                }
            }
            Ok(json)
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_protocol::types::{McpRequest, MCP_VERSION};
    use std::io::Write;

    fn csv_file(name: &str, body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_mcp_table_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(body.as_bytes()).expect("write");
        path
    }

    const BODY: &str =
        "region,trimestre,importe\nnorte,Q1,100\nsur,Q1,250\nnorte,Q3,400\nsur,Q3,50\n";

    /// Everything goes through `handle_request`, not through the handler closure.
    ///
    /// Calling the closure directly would be easier and would test a path no model
    /// ever takes: `tools/call` wraps the result in `content[0].text` as a STRING,
    /// so a field that serialises badly is invisible to a test that skips the
    /// wrapping. This helper does the round trip and parses the text back.
    fn initialised(server: &McpServer) {
        let req = McpRequest::new("initialize")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "protocolVersion": MCP_VERSION,
                "clientInfo": {"name": "test"},
                "capabilities": {}
            }));
        let resp = server.handle_request(req);
        assert!(resp.error.is_none(), "initialize failed: {:?}", resp.error);
    }

    fn tool_names(server: &McpServer) -> Vec<String> {
        let resp = server.handle_request(McpRequest::new("tools/list").with_id(2u64));
        resp.result.expect("tools/list result")["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .map(|t| t["name"].as_str().expect("tool name").to_string())
            .collect()
    }

    fn try_call(
        server: &McpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let resp = server.handle_request(
            McpRequest::new("tools/call")
                .with_id(3u64)
                .with_params(serde_json::json!({"name": tool, "arguments": args})),
        );
        if let Some(err) = resp.error {
            return Err(err.message);
        }
        let text = resp.result.expect("result")["content"][0]["text"]
            .as_str()
            .expect("tools/call wraps the payload as text")
            .to_string();
        Ok(serde_json::from_str(&text).expect("the payload must be valid JSON"))
    }

    fn call(server: &McpServer, tool: &str, args: serde_json::Value) -> serde_json::Value {
        try_call(server, tool, args).unwrap_or_else(|e| panic!("{tool} failed: {e}"))
    }

    #[cfg(feature = "tabular-sqlite")]
    fn server_with_ventas() -> (McpServer, std::path::PathBuf) {
        let path = csv_file("ventas.csv", BODY);
        let mut engine = crate::tabular::sqlite_engine::SqliteTableEngine::new().expect("engine");
        engine.load("ventas", &path).expect("load");
        let mut server = McpServer::new("test", "0");
        register_table_tools(&mut server, shared(engine));
        initialised(&server);
        (server, path)
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn the_three_tools_are_registered_and_load_table_is_not() {
        let (server, _p) = server_with_ventas();
        let names = tool_names(&server);
        for expected in ["list_tables", "describe_table", "query_table"] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected}: {names:?}"
            );
        }
        // The absence IS the security boundary, so it is asserted rather than left
        // to the module docs: a `load_table` added later "for convenience" would
        // let any prompt name any path on the machine and read it back through
        // query results.
        assert!(
            !names.iter().any(|n| n.contains("load")),
            "no tool may open a file from a model-supplied path: {names:?}"
        );
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn list_tables_names_the_engine_and_the_columns() {
        let (server, _p) = server_with_ventas();
        let out = call(&server, "list_tables", serde_json::json!({}));
        assert_eq!(out["total_tables"], 1);
        assert_eq!(out["tables"][0]["name"], "ventas");
        assert_eq!(out["tables"][0]["row_count"], 4);
        let cols: Vec<&str> = out["tables"][0]["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .map(|c| c["name"].as_str().expect("name"))
            .collect();
        assert_eq!(cols, vec!["region", "trimestre", "importe"]);
        // Which engine answered is part of the answer: the two dialects differ
        // (aliases, type names), and a model that knows which one it is talking to
        // writes SQL that works the first time.
        assert_eq!(out["engine"], "sqlite");
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn query_table_echoes_the_sql_and_the_row_count() {
        let (server, _p) = server_with_ventas();
        let out = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT SUM(importe) AS total FROM ventas"}),
        );
        assert_eq!(out["rows"][0][0], 800);
        assert_eq!(out["row_count"], 1);
        assert_eq!(
            out["sql_executed"], "SELECT SUM(importe) AS total FROM ventas",
            "the SQL that ran must come back, or nobody can check the number"
        );
        assert_eq!(out["truncated"], false);
        assert_eq!(out["trustworthy"], true);
        assert!(out["warnings"].as_array().expect("warnings").is_empty());
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn a_write_is_refused_through_the_tool_too() {
        let (server, _p) = server_with_ventas();
        // The refusal lives in the engine; this asserts a new surface does not
        // bypass it, which is the only question worth asking about a new surface
        // onto an existing guard.
        for sql in [
            "DROP TABLE ventas",
            "DELETE FROM ventas",
            "SELECT 1; DROP TABLE ventas",
            "ATTACH DATABASE 'x.db' AS x",
            "PRAGMA writable_schema = ON",
        ] {
            let err = try_call(&server, "query_table", serde_json::json!({"sql": sql}))
                .expect_err(&format!("must refuse: {sql}"));
            assert!(!err.is_empty(), "a refusal must say something: {sql}");
        }
        // And the table is still there afterwards -- a refusal that arrives after
        // the damage is not a refusal.
        let after = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT COUNT(*) AS n FROM ventas"}),
        );
        assert_eq!(after["rows"][0][0], 4);
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn the_row_cap_is_applied_and_announced() {
        let (server, _p) = server_with_ventas();
        let out = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT * FROM ventas", "limit": 100_000}),
        );
        // Clamped, not refused -- and the clamp is visible, because `truncated`
        // alone reads as "the data ran out".
        assert_eq!(out["limit_applied"]["requested"], 100_000);
        assert_eq!(out["limit_applied"]["used"], MAX_ROWS);
        assert_eq!(
            out["row_count"], 4,
            "four rows, cap of 200: nothing was cut"
        );
        assert_eq!(out["truncated"], false);

        // A limit BELOW the row count truncates and says so, with no cap note.
        let cut = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT * FROM ventas", "limit": 2}),
        );
        assert_eq!(cut["row_count"], 2);
        assert_eq!(cut["truncated"], true);
        assert!(cut["limit_applied"].is_null(), "no cap was applied here");

        // And zero is not a way to ask for nothing: it clamps up to 1, because a
        // silently empty result reads as "no rows matched".
        let zero = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT * FROM ventas", "limit": 0}),
        );
        assert_eq!(zero["row_count"], 1);
        assert_eq!(zero["truncated"], true);
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn a_mixed_column_reaches_the_model_as_a_warning_not_as_a_bare_number() {
        // The end-to-end version of the AVG defect: a model must not be able to
        // read an average off this tool without also reading why it is suspect.
        let path = csv_file("na.csv", "a,importe\nx,10\ny,N/A\nz,30\n");
        let mut engine = crate::tabular::sqlite_engine::SqliteTableEngine::new().expect("engine");
        engine
            .load_with(
                "na",
                &path,
                &crate::tabular::LoadOptions::default().keeping_mixed_as_text(),
            )
            .expect("load");
        let mut server = McpServer::new("test", "0");
        register_table_tools(&mut server, shared(engine));
        initialised(&server);

        let described = call(
            &server,
            "describe_table",
            serde_json::json!({"table": "na"}),
        );
        assert_eq!(described["columns"][1]["mixed_numeric"], true);
        assert_eq!(described["columns"][1]["non_numbers_nullified"], false);
        assert_eq!(
            described["columns"][1]["values_that_are_not_numbers"][0]["value"], "N/A",
            "the marker itself is reported, so the model can decide what it means"
        );

        let out = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT AVG(importe) AS media FROM na"}),
        );
        assert_eq!(out["trustworthy"], false);
        let joined = out["warnings"]
            .as_array()
            .expect("warnings")
            .iter()
            .filter_map(|w| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!joined.is_empty(), "a suspect average must carry a warning");
        assert!(joined.contains("KEPT AS TEXT"), "{joined}");
        assert!(
            joined.contains("importe"),
            "and it names the column: {joined}"
        );
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn the_default_policy_is_trustworthy_and_says_what_it_did() {
        // The other half: nullified means the arithmetic is RIGHT, and the warning
        // must say that rather than crying wolf -- a tool that flags every mixed
        // column identically teaches the model to ignore the flag.
        let path = csv_file("na_default.csv", "a,importe\nx,10\ny,N/A\nz,30\n");
        let mut engine = crate::tabular::sqlite_engine::SqliteTableEngine::new().expect("engine");
        engine.load("na", &path).expect("load");
        let mut server = McpServer::new("test", "0");
        register_table_tools(&mut server, shared(engine));
        initialised(&server);

        let described = call(
            &server,
            "describe_table",
            serde_json::json!({"table": "na"}),
        );
        assert_eq!(described["columns"][1]["non_numbers_nullified"], true);

        let out = call(
            &server,
            "query_table",
            serde_json::json!({"sql": "SELECT AVG(importe) AS media FROM na"}),
        );
        assert_eq!(out["rows"][0][0], 20.0, "40/2, the right answer");
        let joined = out["warnings"]
            .as_array()
            .expect("warnings")
            .iter()
            .filter_map(|w| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(joined.contains("STORED AS NULL"), "{joined}");
        assert!(
            !joined.contains("KEPT AS TEXT"),
            "the wrong warning of the two would make a right number look wrong: {joined}"
        );
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn a_missing_table_tells_the_model_which_ones_exist() {
        let (server, _p) = server_with_ventas();
        let err = try_call(
            &server,
            "describe_table",
            serde_json::json!({"table": "no_existe"}),
        )
        .expect_err("unknown table");
        assert!(
            err.contains("ventas"),
            "the error must list what IS loaded, or the model just guesses again: {err}"
        );
    }

    #[cfg(feature = "tabular-sqlite")]
    #[test]
    fn a_missing_argument_is_an_error_and_not_an_empty_answer() {
        let (server, _p) = server_with_ventas();
        for (tool, args) in [
            ("query_table", serde_json::json!({})),
            ("describe_table", serde_json::json!({})),
            ("query_table", serde_json::json!({"sql": 42})),
        ] {
            let err = try_call(&server, tool, args.clone())
                .expect_err(&format!("{tool} with {args} must fail"));
            assert!(err.contains("Missing required parameter"), "{tool}: {err}");
        }
    }

    #[test]
    fn a_float_that_json_cannot_hold_becomes_a_string_not_a_null() {
        // NaN and infinity have no JSON representation, and `null` would claim
        // there was no value rather than that the computation produced one JSON
        // cannot carry. Different answers.
        assert_eq!(cell_to_json(&Cell::Float(1.5)), serde_json::json!(1.5));
        assert_eq!(cell_to_json(&Cell::Null), serde_json::Value::Null);
        assert_eq!(cell_to_json(&Cell::Int(-3)), serde_json::json!(-3));
        assert_eq!(cell_to_json(&Cell::Bool(true)), serde_json::json!(true));
        assert_eq!(
            cell_to_json(&Cell::Text("x".to_string())),
            serde_json::json!("x")
        );
        assert!(
            cell_to_json(&Cell::Float(f64::INFINITY)).is_string(),
            "infinity must not silently become null"
        );
        assert!(cell_to_json(&Cell::Float(f64::NAN)).is_string());
    }
}
