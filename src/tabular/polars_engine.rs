// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! The expensive engine: Polars, for Parquet and for files that do not fit in
//! memory.
//!
//! # What it costs, measured
//!
//! Against a hello-world with the same release profile, on 2026-09-24:
//!
//! | | hello world | + Polars | difference |
//! |---|---|---|---|
//! | crates in the graph | 1 | **397** | +396 |
//! | cold compile | 1 s | **756 s** | +755 s |
//! | release binary | 131 KB | **49.4 MiB** | **+49.3 MiB** |
//!
//! The twelve minutes are paid once; **the 49 MiB are paid forever**, on every
//! copy and every download. For scale, the whole llama.cpp payload the pendrive
//! carries is 19 MB in 8 files.
//!
//! So this is behind its own feature and is not the default. The reason to turn
//! it on is concrete and not fashion: **SQLite cannot read Parquet**, which is
//! how tabular data actually travels in business settings.
//!
//! # Where the safety comes from, and why it is not this file
//!
//! The statement checks live in the **parent module** —
//! [`ensure_single_read_only_statement`](super::ensure_single_read_only_statement)
//! — and that is the important design decision here, not an aesthetic one.
//!
//! Polars' SQL dialect accepts `INSERT`, `DELETE`, `TRUNCATE` and `DROP TABLE`,
//! and it has **no equivalent of SQLite's `Statement::readonly()`**: nothing to
//! ask whether a parsed statement writes. So this engine has *one fewer* layer
//! available than the SQLite one. If each engine carried its own copy of the
//! checks, the weaker engine would be the one that quietly lost a check, and
//! every test would still pass.
//!
//! That is the masked-defect shape this project keeps hitting: guard across every
//! configuration, not only the one that failed. An engine may add layers on top
//! of the shared ones. It may not have fewer.
//!
//! What this engine does have, and SQLite does not: the tables are **read-only
//! `LazyFrame`s built from the file**. There is no mutable store for a write to
//! land in even if one got through.

use std::collections::BTreeMap;
use std::path::Path;

use polars::prelude::*;
use polars::sql::SQLContext;

use super::{
    ensure_single_read_only_statement, is_valid_table_name, Cell, ColumnInfo, LoadOptions,
    QueryResult, TableEngine, TableError, TableInfo,
};

/// Tables held as lazy frames, queried through Polars' SQL layer.
pub struct PolarsTableEngine {
    frames: BTreeMap<String, LazyFrame>,
    tables: BTreeMap<String, TableInfo>,
}

impl std::fmt::Debug for PolarsTableEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolarsTableEngine")
            .field("tables", &self.tables.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for PolarsTableEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolarsTableEngine {
    /// An engine with no tables loaded.
    pub fn new() -> Self {
        Self {
            frames: BTreeMap::new(),
            tables: BTreeMap::new(),
        }
    }

    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    /// Reasons to distrust an answer to `sql`.
    ///
    /// Same contract as the SQLite engine's, and for the same reason: a column
    /// holding numbers and non-numbers makes aggregates lie. Polars is stricter
    /// than SQLite here — it will usually *error* rather than coerce text to zero
    /// — but "usually" is not a guarantee across versions, and a caller that
    /// switches engines must not lose a warning by doing so.
    fn warnings_for(&self, sql: &str) -> Vec<String> {
        let lowered = sql.to_lowercase();
        let mut out = Vec::new();
        for table in self.tables.values() {
            for col in table.mixed_numeric_columns() {
                if !lowered.contains(&col.name.to_lowercase()) {
                    continue;
                }
                out.push(format!(
                    "column {:?} of table {:?} holds {} numbers and {} values that are not \
                     numbers, so it was read as text. Aggregates over it are not the aggregates \
                     of the numbers. Reload with LoadOptions::treating_as_missing to make them \
                     NULL.",
                    col.name,
                    table.name,
                    col.numeric_values,
                    col.unparseable_count(),
                ));
            }
        }
        out
    }
}

/// Turn a filesystem path into the form Polars 0.55 takes.
///
/// A path that is not valid UTF-8 **errors** rather than being converted
/// lossily: a lossy conversion would silently name a *different* file, and
/// reading the wrong file is worse than refusing to read.
fn pl_path(path: &Path) -> Result<PlRefPath, TableError> {
    path.to_str().map(PlRefPath::from).ok_or_else(|| {
        TableError::Io(format!(
            "this path is not valid UTF-8, which Polars cannot represent: {}",
            path.display()
        ))
    })
}

/// Is this path a Parquet file?
///
/// By extension, which is what the caller controls. Sniffing the magic bytes
/// would be more robust and is worth doing the day somebody hits it; today the
/// wrong guess produces a clear parse error rather than a wrong answer.
fn looks_like_parquet(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("parquet") || e.eq_ignore_ascii_case("pq"))
        .unwrap_or(false)
}

/// Turn one Polars value into the engine-neutral [`Cell`].
///
/// Engine-neutral is the whole point: it is what lets one test run the same
/// queries through SQLite and Polars and demand the same answer.
fn to_cell(value: AnyValue<'_>) -> Cell {
    match value {
        AnyValue::Null => Cell::Null,
        AnyValue::Boolean(b) => Cell::Bool(b),
        AnyValue::Int8(v) => Cell::Int(v as i64),
        AnyValue::Int16(v) => Cell::Int(v as i64),
        AnyValue::Int32(v) => Cell::Int(v as i64),
        AnyValue::Int64(v) => Cell::Int(v),
        AnyValue::UInt8(v) => Cell::Int(v as i64),
        AnyValue::UInt16(v) => Cell::Int(v as i64),
        AnyValue::UInt32(v) => Cell::Int(v as i64),
        // A u64 beyond i64 cannot become an Int without lying about its value.
        // Text is lossless; a truncated integer is not.
        AnyValue::UInt64(v) => match i64::try_from(v) {
            Ok(n) => Cell::Int(n),
            Err(_) => Cell::Text(v.to_string()),
        },
        AnyValue::Float32(v) => Cell::Float(v as f64),
        AnyValue::Float64(v) => Cell::Float(v),
        AnyValue::String(s) => Cell::Text(s.to_string()),
        AnyValue::StringOwned(s) => Cell::Text(s.to_string()),
        // Dates, times, lists, structs and the rest: rendered rather than
        // dropped. Losing a column silently would be worse than showing it as
        // text.
        other => Cell::Text(other.to_string()),
    }
}

impl TableEngine for PolarsTableEngine {
    fn load_with(
        &mut self,
        name: &str,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<TableInfo, TableError> {
        if !is_valid_table_name(name) {
            return Err(TableError::BadTableName(name.to_string()));
        }
        if self.tables.contains_key(name) {
            return Err(TableError::DuplicateTable(name.to_string()));
        }

        let frame = if looks_like_parquet(path) {
            LazyFrame::scan_parquet(pl_path(path)?, ScanArgsParquet::default())
                .map_err(|e| TableError::Parse(e.to_string()))?
        } else {
            // The missing markers go to Polars, so it can type the column
            // properly instead of falling back to text — the same decision the
            // SQLite engine makes, reached through a different mechanism.
            let mut na: Vec<PlSmallStr> = vec![PlSmallStr::from_static("")];
            na.extend(options.na_values.iter().map(|s| PlSmallStr::from_str(s)));
            LazyCsvReader::new(pl_path(path)?)
                .with_has_header(true)
                .with_null_values(Some(NullValues::AllColumns(na)))
                .finish()
                .map_err(|e| TableError::Io(e.to_string()))?
        };

        // Collected once at load so `row_count` and the column report are real
        // numbers rather than "unknown". Lazy evaluation is what makes the
        // *queries* cheap; the load has to look at the data anyway to describe
        // it, and a `describe` that says "unknown rows" is not worth having.
        let df = frame
            .clone()
            .collect()
            .map_err(|e| TableError::Parse(e.to_string()))?;

        let columns: Vec<ColumnInfo> = df
            .columns()
            .iter()
            .map(|series| {
                let dtype = series.dtype();
                let numeric = dtype.is_numeric();
                // Polars types the column for us, so "unparseable" means
                // something different here than in SQLite: a text column in a
                // file whose other values are numbers. Counted the same way so
                // the two engines report the same thing.
                let (numeric_values, unparseable) = if numeric {
                    (series.len() - series.null_count(), Vec::new())
                } else {
                    count_numeric_and_rest(series)
                };
                ColumnInfo {
                    name: series.name().to_string(),
                    declared_type: format!("{dtype}"),
                    numeric_values,
                    unparseable,
                }
            })
            .collect();

        let info = TableInfo {
            name: name.to_string(),
            source: path.to_path_buf(),
            row_count: df.height(),
            columns,
        };
        self.frames.insert(name.to_string(), frame);
        self.tables.insert(name.to_string(), info.clone());
        Ok(info)
    }

    fn list_tables(&self) -> Vec<TableInfo> {
        self.tables.values().cloned().collect()
    }

    fn describe(&self, name: &str) -> Result<TableInfo, TableError> {
        self.tables
            .get(name)
            .cloned()
            .ok_or_else(|| TableError::NoSuchTable {
                asked: name.to_string(),
                available: self.table_names(),
            })
    }

    fn query(&self, sql: &str, limit: usize) -> Result<QueryResult, TableError> {
        // The shared layers. Polars has no `readonly()` to ask, so these are all
        // there is — which is exactly why they live in the parent module and not
        // in a copy per engine.
        ensure_single_read_only_statement(sql)?;

        let mut ctx = SQLContext::new();
        for (name, frame) in &self.frames {
            ctx.register(name, frame.clone());
        }

        let plan = ctx
            .execute(sql)
            .map_err(|e| TableError::BadQuery(e.to_string()))?;

        // One past the limit, so "exactly `limit` rows" can be told from "cut
        // short". Reporting 10 of 10 when there were 4000 is how a model states a
        // wrong total.
        let df = plan
            .limit((limit as u32).saturating_add(1))
            .collect()
            .map_err(|e| TableError::BadQuery(e.to_string()))?;

        let truncated = df.height() > limit;
        let take = df.height().min(limit);

        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|c| c.to_string())
            .collect();
        let mut rows = Vec::with_capacity(take);
        for r in 0..take {
            let mut cells = Vec::with_capacity(columns.len());
            for series in df.columns() {
                let v = series
                    .get(r)
                    .map_err(|e| TableError::BadQuery(e.to_string()))?;
                cells.push(to_cell(v));
            }
            rows.push(cells);
        }

        Ok(QueryResult {
            sql_executed: sql.to_string(),
            row_count: rows.len(),
            warnings: self.warnings_for(sql),
            columns,
            rows,
            truncated,
            engine: self.engine_name().to_string(),
        })
    }

    fn engine_name(&self) -> &'static str {
        "polars"
    }
}

/// For a non-numeric column, how many of its values would parse as numbers and
/// which ones would not.
fn count_numeric_and_rest(series: &Column) -> (usize, Vec<(String, usize)>) {
    let mut numeric = 0usize;
    let mut rest: Vec<(String, usize)> = Vec::new();
    for i in 0..series.len() {
        let Ok(v) = series.get(i) else { continue };
        if matches!(v, AnyValue::Null) {
            continue;
        }
        let text = match v {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            other => other.to_string(),
        };
        let t = text.trim();
        if t.parse::<f64>().is_ok() {
            numeric += 1;
        } else {
            match rest.iter_mut().find(|(s, _)| s == t) {
                Some((_, n)) => *n += 1,
                None => rest.push((t.to_string(), 1)),
            }
        }
    }
    (numeric, rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A fixture in a directory nobody else uses. `File::create` truncates, and a
    /// shared name made a test pass in one run and fail in the next.
    fn csv_file(name: &str, body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_polars_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).expect("create");
        f.write_all(body.as_bytes()).expect("write");
        p
    }

    fn loaded() -> PolarsTableEngine {
        let p = csv_file(
            "ventas.csv",
            "region,trimestre,importe\nnorte,Q1,100\nsur,Q1,250\nnorte,Q3,400\nsur,Q3,50\n",
        );
        let mut e = PolarsTableEngine::new();
        e.load("ventas", &p).expect("load");
        e
    }

    #[test]
    fn it_answers_the_aggregate_that_rag_cannot() {
        let e = loaded();
        let r = e
            .query(
                "SELECT SUM(importe) AS total FROM ventas WHERE trimestre = 'Q3'",
                10,
            )
            .expect("query");
        assert_eq!(r.row_count, 1);
        assert_eq!(r.rows[0][0], Cell::Int(450));
        assert_eq!(r.engine, "polars");
        assert!(r.sql_executed.contains("SUM"));
    }

    #[test]
    fn it_describes_what_it_loaded() {
        let e = loaded();
        let info = e.describe("ventas").expect("described");
        assert_eq!(info.row_count, 4);
        assert_eq!(
            info.columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["region", "trimestre", "importe"]
        );
        // Polars names its own types; the wording is for a human, not for
        // comparing engines.
        assert!(
            info.columns[2].declared_type.contains("64"),
            "expected an integer-ish type, got {}",
            info.columns[2].declared_type
        );
    }

    #[test]
    fn the_shared_checks_apply_here_too() {
        // The whole reason the checks live in the parent module. Polars SQL
        // accepts INSERT/DELETE/DROP and has no `readonly()` to ask, so if this
        // engine had its own copy it would be the one that quietly lost a layer.
        let e = loaded();
        for sql in [
            "DELETE FROM ventas",
            "DROP TABLE ventas",
            "INSERT INTO ventas VALUES ('x', 'Q4', 1)",
            "TRUNCATE TABLE ventas",
            "CREATE TABLE otra (a INTEGER)",
        ] {
            let err = e.query(sql, 10).expect_err("must refuse");
            assert!(
                matches!(err, TableError::NotReadOnly(_)),
                "not refused: {sql} -> {err:?}"
            );
        }
        assert_eq!(
            e.query("SELECT 1; DROP TABLE ventas", 10)
                .expect_err("must refuse"),
            TableError::MultipleStatements
        );
        // And the table is still there.
        assert!(e.describe("ventas").is_ok());
    }

    #[test]
    fn truncation_is_reported_and_not_guessed() {
        let e = loaded();
        let all = e.query("SELECT * FROM ventas", 10).expect("q");
        assert_eq!(all.row_count, 4);
        assert!(!all.truncated);

        let cut = e.query("SELECT * FROM ventas", 2).expect("q");
        assert_eq!(cut.row_count, 2);
        assert!(cut.truncated, "2 of 4 must say so");
    }

    #[test]
    fn no_rows_is_an_answer_not_an_error() {
        let e = loaded();
        let r = e
            .query("SELECT * FROM ventas WHERE region = 'este'", 10)
            .expect("a query that matches nothing still ran");
        assert!(r.is_empty());
        assert!(!r.truncated);
    }

    #[test]
    fn a_missing_table_names_the_ones_loaded() {
        let e = loaded();
        assert_eq!(
            e.describe("sales").expect_err("not loaded"),
            TableError::NoSuchTable {
                asked: "sales".to_string(),
                available: vec!["ventas".to_string()],
            }
        );
    }

    #[test]
    fn a_table_name_that_could_break_out_is_refused_at_load() {
        let p = csv_file("x.csv", "a\n1\n");
        let mut e = PolarsTableEngine::new();
        assert!(matches!(
            e.load("v\"; DROP TABLE x; --", &p),
            Err(TableError::BadTableName(_))
        ));
    }

    #[test]
    fn an_empty_field_is_null_and_a_declared_marker_too() {
        let p = csv_file("huecos.csv", "a,importe\nx,10\ny,\nz,N/A\nw,30\n");
        let mut e = PolarsTableEngine::new();
        e.load_with("h", &p, &LoadOptions::treating_as_missing(["N/A"]))
            .expect("load");
        let r = e.query("SELECT SUM(importe) FROM h", 10).expect("q");
        assert_eq!(
            r.rows[0][0],
            Cell::Int(40),
            "the two missing ones are skipped"
        );
        let n = e.query("SELECT COUNT(importe) FROM h", 10).expect("q");
        assert_eq!(n.rows[0][0], Cell::Int(2), "NULLs are not counted");
    }

    #[test]
    fn a_mixed_numeric_column_warns_here_too() {
        // Polars is stricter than SQLite and may error rather than coerce, but a
        // caller must not lose the warning by switching engines.
        let p = csv_file("na.csv", "a,importe\nx,10\ny,N/A\nz,30\n");
        let mut e = PolarsTableEngine::new();
        let info = e.load("na", &p).expect("load");
        assert!(
            info.columns[1].is_mixed_numeric(),
            "two numbers and one non-number: {:?}",
            info.columns[1]
        );
        assert_eq!(info.columns[1].numeric_values, 2);
        assert_eq!(info.columns[1].unparseable, vec![("N/A".to_string(), 1)]);
    }

    #[test]
    fn an_honest_query_carries_no_warnings() {
        let e = loaded();
        let r = e.query("SELECT SUM(importe) FROM ventas", 10).expect("q");
        assert!(r.is_trustworthy(), "{:?}", r.warnings);
    }

    #[test]
    fn a_u64_too_big_for_i64_becomes_text_rather_than_a_wrong_number() {
        // Truncating would report a different number with full confidence; text
        // is lossless.
        assert_eq!(
            to_cell(AnyValue::UInt64(u64::MAX)),
            Cell::Text(u64::MAX.to_string())
        );
        assert_eq!(to_cell(AnyValue::UInt64(7)), Cell::Int(7));
    }

    #[test]
    fn parquet_is_recognised_by_extension() {
        assert!(looks_like_parquet(Path::new("ventas.parquet")));
        assert!(looks_like_parquet(Path::new("VENTAS.PARQUET")));
        assert!(looks_like_parquet(Path::new("v.pq")));
        assert!(!looks_like_parquet(Path::new("ventas.csv")));
        assert!(!looks_like_parquet(Path::new("ventas")));
    }
}
