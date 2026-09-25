// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! The free engine: SQLite, which was already in the dependency tree.
//!
//! # Where the safety comes from — three layers, and why none alone is enough
//!
//! **1. The database lives in memory.** Nothing is opened on disk, so there is
//! no file a query could damage even if every other check failed. That is the
//! boundary; the rest is defence in depth.
//!
//! **2. One statement only**, enforced by `has_trailing_statement` — our own
//! scan, because `Connection::prepare` does **not** reject a second statement.
//! (Named in plain text: it is private, and an intra-doc link from public
//! documentation to a private item renders as dead text.)
//! Measured: `"SELECT 1; DROP TABLE ventas"` prepared fine, ran the `SELECT`,
//! returned one row and **discarded the rest without a word**. The `DROP` never
//! executed, so nothing was destroyed — but that is SQLite's design doing the
//! work, not a check of ours.
//!
//! **3. `Statement::readonly()`** asks **SQLite itself** whether the prepared
//! statement writes. Parsing the SQL to guess would be a worse answer wearing a
//! better disguise: `DELETE` appears inside string literals, `WITH` can
//! introduce an `INSERT`, and comments hide both.
//!
//! **4. And the opening keyword must be `SELECT`, `WITH` or `VALUES`** — which
//! is here because layer 3 does **not** cover `ATTACH`.
//!
//! That last point cost a corrected claim. The first version of this file said
//! "`ATTACH` is the one that reaches back out to the filesystem, and SQLite
//! reports it as not read-only, so the same check covers it." **That is false,
//! and the test caught it**: `sqlite3_stmt_readonly` classifies `ATTACH` and
//! `DETACH` as read-only, because they do not modify the *contents* of any
//! database. `ATTACH DATABASE 'C:/evil.db'` prepared fine and only failed
//! because that file did not exist. **With the file present it would have
//! worked.**
//!
//! So layers 3 and 4 cover two different classes — modifying data, and reaching
//! outside the data — and it takes both. A whitelist of three opening keywords
//! is a narrower and more honest thing than a blacklist of dangerous words: the
//! blacklist has to be complete to work, the whitelist only has to be short.
//!
//! The stronger mechanism, if this ever needs to allow more: SQLite's
//! **authorizer callback** (rusqlite's `hooks` feature), which reports each
//! action — `SQLITE_ATTACH`, `SQLITE_READ`, `SQLITE_PRAGMA` — and lets the host
//! allow or deny per action. That is the right answer for a richer surface; it
//! is more machinery than three keywords need today.

use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};

use super::{
    is_valid_table_name, Cell, ColumnInfo, LoadOptions, QueryResult, TableEngine, TableError,
    TableInfo,
};

/// The type a column was given after looking at its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inferred {
    Integer,
    Real,
    Text,
}

impl Inferred {
    fn sql(self) -> &'static str {
        match self {
            Self::Integer => "INTEGER",
            Self::Real => "REAL",
            Self::Text => "TEXT",
        }
    }
}

/// Tables loaded into an in-memory SQLite database.
pub struct SqliteTableEngine {
    conn: Connection,
    tables: BTreeMap<String, TableInfo>,
}

impl std::fmt::Debug for SqliteTableEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteTableEngine")
            .field("tables", &self.tables.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SqliteTableEngine {
    /// A fresh, empty in-memory database.
    pub fn new() -> Result<Self, TableError> {
        // No `SQLITE_OPEN_URI`: without it an "attach" through a filename is
        // not even expressible.
        let conn = Connection::open_in_memory_with_flags(
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .map_err(|e| TableError::Io(e.to_string()))?;
        Ok(Self {
            conn,
            tables: BTreeMap::new(),
        })
    }

    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    /// Reasons to distrust an answer to `sql`.
    ///
    /// Today there is one: a column holding numbers **and** something that is
    /// not a number makes `AVG` and `COUNT` silently wrong (see
    /// [`ColumnInfo::is_mixed_numeric`]).
    ///
    /// Whether the query "touches" the column is decided by looking for its name
    /// in the SQL text. That is approximate — the name could appear inside a
    /// string literal — and the inaccuracy is deliberately on the noisy side. A
    /// warning that shows up once too often is an annoyance; one that fails to
    /// show up is a confidently wrong number in front of somebody who trusted
    /// it.
    fn warnings_for(&self, sql: &str) -> Vec<String> {
        let lowered = sql.to_lowercase();
        let mut out = Vec::new();
        for table in self.tables.values() {
            for col in table.mixed_numeric_columns() {
                if !lowered.contains(&col.name.to_lowercase()) {
                    continue;
                }
                let examples: Vec<String> = col
                    .unparseable
                    .iter()
                    .take(3)
                    .map(|(v, n)| format!("{v:?} x{n}"))
                    .collect();
                let more = col.unparseable.len().saturating_sub(3);
                out.push(format!(
                    "column {:?} of table {:?} holds {} numbers and {} values that are not \
                     numbers ({}{}), so it was stored as text. SQLite coerces those to zero in \
                     aggregates: AVG divides by every row instead of skipping them, and COUNT \
                     includes them. Reload with LoadOptions::treating_as_missing to make them NULL.",
                    col.name,
                    table.name,
                    col.numeric_values,
                    col.unparseable_count(),
                    examples.join(", "),
                    if more > 0 {
                        format!(", and {more} more")
                    } else {
                        String::new()
                    },
                ));
            }
        }
        out
    }
}

/// What looking at a column's values told us.
struct Survey {
    inferred: Inferred,
    numeric_values: usize,
    /// Distinct values that are neither missing nor numbers, with their counts,
    /// in the order first seen so reports are stable.
    unparseable: Vec<(String, usize)>,
}

/// Look at a column's values and decide what it is, **counting what did not fit**.
///
/// Missing markers are skipped: they become NULL, and a column of numbers with
/// one blank is still a column of numbers. A column with no values at all is
/// TEXT, which is the choice that loses no information.
///
/// The counting is the part that matters. Before it, one stray `N/A` turned the
/// column to TEXT and nothing said so — and SQLite then answers `AVG` over that
/// column by coercing the `N/A` to a real zero, silently dividing by the wrong
/// count. Now the values that did not fit come back with the verdict.
fn survey_column(values: &[String], options: &LoadOptions) -> Survey {
    let mut numeric_values = 0usize;
    let mut all_int = true;
    let mut all_real = true;
    let mut unparseable: Vec<(String, usize)> = Vec::new();

    for v in values {
        let t = v.trim();
        if options.is_missing(t) {
            continue;
        }
        let is_int = t.parse::<i64>().is_ok();
        let is_real = t.parse::<f64>().is_ok();
        if is_real {
            numeric_values += 1;
            if !is_int {
                all_int = false;
            }
        } else {
            all_int = false;
            all_real = false;
            match unparseable.iter_mut().find(|(s, _)| s == t) {
                Some((_, n)) => *n += 1,
                None => unparseable.push((t.to_string(), 1)),
            }
        }
    }

    let inferred = if numeric_values == 0 {
        Inferred::Text
    } else if all_int {
        Inferred::Integer
    } else if all_real {
        Inferred::Real
    } else {
        // Numbers AND something else. TEXT is the only type that stores both
        // without losing anything — and `Survey::unparseable` is what stops that
        // being a silent decision.
        Inferred::Text
    };

    Survey {
        inferred,
        numeric_values,
        unparseable,
    }
}

/// The first SQL keyword, uppercased, with leading comments and whitespace
/// skipped.
///
/// Comments are skipped rather than rejected because `-- what this asks\nSELECT
/// …` is perfectly legitimate, and a model that explains itself in a comment
/// should not be refused. But they have to be skipped *correctly*, or
/// `--\nATTACH` reads as no keyword at all.
fn first_keyword(sql: &str) -> String {
    let b = sql.as_bytes();
    let mut i = 0usize;
    loop {
        // Whitespace.
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        // Line comment.
        if i + 1 < b.len() && b[i] == b'-' && b[i + 1] == b'-' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment. An unterminated one runs to the end, which leaves no
        // keyword — and no keyword is refused, which is the safe direction.
        if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            continue;
        }
        break;
    }
    let start = i;
    while i < b.len() && (b[i] as char).is_ascii_alphabetic() {
        i += 1;
    }
    sql[start..i].to_ascii_uppercase()
}

/// Is there a second statement after the first `;`?
///
/// **This exists because `Connection::prepare` does not reject one.** Measured:
/// `"SELECT 1; DROP TABLE ventas"` prepared fine, ran the `SELECT`, returned one
/// row, and **discarded the rest without a word**. `sqlite3_prepare_v2` stops at
/// the first statement and hands back the tail; rusqlite does not surface it.
///
/// The `DROP` did not execute, so nothing was destroyed — but that is SQLite's
/// design doing the work, not a check of ours, and "the second half of your query
/// was silently thrown away" is a wrong answer even when it is a safe one.
///
/// A plain search for `;` would be wrong: it appears inside string literals
/// (`SELECT ';'`), inside quoted identifiers, and inside comments. So this walks
/// the text tracking what it is inside, and only a top-level `;` counts.
fn has_trailing_statement(sql: &str) -> bool {
    let b = sql.as_bytes();
    let mut i = 0usize;
    // Position just after a top-level `;`, if we have seen one.
    let mut after_semicolon: Option<usize> = None;

    while i < b.len() {
        match b[i] {
            // Single-quoted string.
            //
            // No special case for the `''` escape, and that is deliberate: this
            // scan only needs to know whether a given character is inside a
            // literal, and `''` flips that state twice with **no character
            // between the two quotes** to misclassify. So close-then-open lands
            // in exactly the same place.
            //
            // The first version did handle it explicitly, and mutating that
            // branch away changed no answer — which is how the redundancy was
            // found. Dead code with a plausible comment is worse than no code.
            b'\'' => {
                i += 1;
                while i < b.len() && b[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            }
            // Double-quoted identifier, same reasoning.
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += 1;
                }
                i += 1;
            }
            // SQLite also accepts [bracketed] and `backticked` identifiers.
            b'[' => {
                while i < b.len() && b[i] != b']' {
                    i += 1;
                }
                i += 1;
            }
            b'`' => {
                i += 1;
                while i < b.len() && b[i] != b'`' {
                    i += 1;
                }
                i += 1;
            }
            b'-' if i + 1 < b.len() && b[i + 1] == b'-' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            }
            b';' => {
                after_semicolon = Some(i + 1);
                i += 1;
            }
            _ => i += 1,
        }
    }

    // A trailing `;` on its own is fine — plenty of tools add one. What is not
    // fine is anything else after it, comments included: `SELECT 1; -- x` is
    // harmless, but allowing it means allowing the scan to stop early, and the
    // simpler rule is easier to be sure about.
    match after_semicolon {
        None => false,
        Some(rest) => !sql[rest..].trim().is_empty(),
    }
}

/// Statement kinds this engine will run.
///
/// A whitelist, not a blacklist: a blacklist of dangerous keywords has to be
/// complete to work, and it will not be. `PRAGMA` is absent deliberately —
/// some pragmas only report, others change behaviour, and telling them apart is
/// exactly the kind of judgement that goes stale.
const ALLOWED_OPENINGS: &[&str] = &["SELECT", "WITH", "VALUES"];

/// Quote an identifier for SQLite by doubling any embedded quote.
///
/// Only ever called on names that already passed [`is_valid_table_name`] or
/// came from a file header. Belt and braces: the validator makes the dangerous
/// cases unrepresentable, this makes the merely awkward ones survivable.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

impl TableEngine for SqliteTableEngine {
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

        let mut reader = csv::ReaderBuilder::new()
            .flexible(false)
            .from_path(path)
            .map_err(|e| TableError::Io(e.to_string()))?;

        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| TableError::Parse(e.to_string()))?
            .iter()
            .map(|h| h.trim().to_string())
            .collect();
        if headers.is_empty() {
            return Err(TableError::Parse("the file has no header row".to_string()));
        }
        if let Some(blank) = headers.iter().position(|h| h.is_empty()) {
            return Err(TableError::Parse(format!(
                "column {} has no name in the header row",
                blank + 1
            )));
        }

        let mut rows: Vec<Vec<String>> = Vec::new();
        for (i, rec) in reader.records().enumerate() {
            let rec = rec.map_err(|e| {
                // The row number is the difference between a fixable file and a
                // mystery.
                TableError::Parse(format!("row {}: {}", i + 2, e))
            })?;
            rows.push(rec.iter().map(|c| c.to_string()).collect());
        }

        let surveys: Vec<Survey> = (0..headers.len())
            .map(|c| {
                let column: Vec<String> = rows.iter().filter_map(|r| r.get(c).cloned()).collect();
                survey_column(&column, options)
            })
            .collect();
        let types: Vec<Inferred> = surveys.iter().map(|s| s.inferred).collect();

        let columns: Vec<String> = headers
            .iter()
            .zip(&types)
            .map(|(h, t)| format!("{} {}", quote_ident(h), t.sql()))
            .collect();
        self.conn
            .execute_batch(&format!(
                "CREATE TABLE {} ({})",
                quote_ident(name),
                columns.join(", ")
            ))
            .map_err(|e| TableError::Parse(e.to_string()))?;

        let placeholders = vec!["?"; headers.len()].join(", ");
        let insert = format!(
            "INSERT INTO {} VALUES ({})",
            quote_ident(name),
            placeholders
        );
        {
            let tx = self
                .conn
                .unchecked_transaction()
                .map_err(|e| TableError::Io(e.to_string()))?;
            {
                let mut stmt = tx
                    .prepare(&insert)
                    .map_err(|e| TableError::Parse(e.to_string()))?;
                for row in &rows {
                    let values: Vec<rusqlite::types::Value> = row
                        .iter()
                        .zip(&types)
                        .map(|(v, t)| {
                            let s = v.trim();
                            // The caller's definition of missing, not ours.
                            if options.is_missing(s) {
                                return rusqlite::types::Value::Null;
                            }
                            match t {
                                Inferred::Integer => s
                                    .parse::<i64>()
                                    .map(rusqlite::types::Value::Integer)
                                    .unwrap_or_else(|_| {
                                        rusqlite::types::Value::Text(s.to_string())
                                    }),
                                Inferred::Real => s
                                    .parse::<f64>()
                                    .map(rusqlite::types::Value::Real)
                                    .unwrap_or_else(|_| {
                                        rusqlite::types::Value::Text(s.to_string())
                                    }),
                                Inferred::Text => rusqlite::types::Value::Text(s.to_string()),
                            }
                        })
                        .collect();
                    stmt.execute(rusqlite::params_from_iter(values))
                        .map_err(|e| TableError::Parse(e.to_string()))?;
                }
            }
            tx.commit().map_err(|e| TableError::Io(e.to_string()))?;
        }

        let info = TableInfo {
            name: name.to_string(),
            source: path.to_path_buf(),
            row_count: rows.len(),
            columns: headers
                .iter()
                .zip(&surveys)
                .map(|(h, s)| ColumnInfo {
                    name: h.clone(),
                    declared_type: s.inferred.sql().to_string(),
                    numeric_values: s.numeric_values,
                    unparseable: s.unparseable.clone(),
                })
                .collect(),
        };
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
        if has_trailing_statement(sql) {
            return Err(TableError::MultipleStatements);
        }

        // Checked BEFORE `prepare`, because preparing an `ATTACH` already opens
        // the file: by the time SQLite could tell us anything, the damage of
        // touching the filesystem is done.
        let opening = first_keyword(sql);
        if !ALLOWED_OPENINGS.contains(&opening.as_str()) {
            return Err(TableError::NotReadOnly(format!(
                "a query must start with SELECT, WITH or VALUES; this one starts with {}",
                if opening.is_empty() {
                    "no keyword at all".to_string()
                } else {
                    opening
                }
            )));
        }

        let mut stmt = self.conn.prepare(sql).map_err(|e| match e {
            rusqlite::Error::MultipleStatement => TableError::MultipleStatements,
            other => TableError::BadQuery(other.to_string()),
        })?;

        // SQLite's own answer, not our guess at one.
        if !stmt.readonly() {
            return Err(TableError::NotReadOnly(
                "the statement would modify data or reach outside the loaded tables".to_string(),
            ));
        }

        let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();

        let mut rows_out: Vec<Vec<Cell>> = Vec::new();
        let mut truncated = false;
        let mut rows = stmt.raw_query();
        // Fetch one past the limit so "exactly `limit` rows" can be told from
        // "cut short". Reporting 10 of 10 when there were 4000 is how a model
        // ends up stating a wrong total.
        while let Some(row) = rows
            .next()
            .map_err(|e| TableError::BadQuery(e.to_string()))?
        {
            if rows_out.len() == limit {
                truncated = true;
                break;
            }
            let mut cells = Vec::with_capacity(columns.len());
            for i in 0..columns.len() {
                let v = row
                    .get_ref(i)
                    .map_err(|e| TableError::BadQuery(e.to_string()))?;
                cells.push(match v {
                    ValueRef::Null => Cell::Null,
                    ValueRef::Integer(n) => Cell::Int(n),
                    ValueRef::Real(f) => Cell::Float(f),
                    ValueRef::Text(t) => Cell::Text(String::from_utf8_lossy(t).into_owned()),
                    ValueRef::Blob(b) => Cell::Text(format!("[{} bytes]", b.len())),
                });
            }
            rows_out.push(cells);
        }

        Ok(QueryResult {
            sql_executed: sql.to_string(),
            row_count: rows_out.len(),
            warnings: self.warnings_for(sql),
            columns,
            rows: rows_out,
            truncated,
            engine: self.engine_name().to_string(),
        })
    }

    fn engine_name(&self) -> &'static str {
        "sqlite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a fixture into a directory nobody else uses.
    ///
    /// The first version put every fixture in one shared directory, and
    /// `File::create` truncates: two tests running in parallel wrote the same
    /// `ventas.csv`, so one of them read a half-empty file and failed with "the
    /// file has no header row". It passed in one run and failed in the next,
    /// which is how the race announced itself.
    fn csv_file(name: &str, body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_tabular_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).expect("create");
        f.write_all(body.as_bytes()).expect("write");
        p
    }

    fn loaded() -> SqliteTableEngine {
        let p = csv_file(
            "ventas.csv",
            "region,trimestre,importe\nnorte,Q1,100\nsur,Q1,250\nnorte,Q3,400\nsur,Q3,50\n",
        );
        let mut e = SqliteTableEngine::new().expect("engine");
        e.load("ventas", &p).expect("load");
        e
    }

    #[test]
    fn it_loads_a_csv_and_infers_column_types() {
        let e = loaded();
        let info = e.describe("ventas").expect("described");
        assert_eq!(info.row_count, 4);
        assert_eq!(
            info.columns
                .iter()
                .map(|c| (c.name.as_str(), c.declared_type.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("region", "TEXT"),
                ("trimestre", "TEXT"),
                ("importe", "INTEGER")
            ]
        );
    }

    #[test]
    fn it_answers_the_aggregate_that_rag_cannot() {
        // The whole reason this module exists: "how much did Q3 add up to" is
        // not answered by retrieving anything.
        let e = loaded();
        let r = e
            .query(
                "SELECT SUM(importe) AS total FROM ventas WHERE trimestre = 'Q3'",
                10,
            )
            .expect("query");
        assert_eq!(r.row_count, 1);
        assert_eq!(r.rows[0][0], Cell::Int(450));
        // And it says what it ran.
        assert!(r.sql_executed.contains("SUM"));
        assert_eq!(r.engine, "sqlite");
    }

    #[test]
    fn a_write_is_refused() {
        let e = loaded();
        for sql in [
            "DELETE FROM ventas",
            "UPDATE ventas SET importe = 0",
            "DROP TABLE ventas",
            "INSERT INTO ventas VALUES ('x', 'Q4', 1)",
            "CREATE TABLE otra (a INTEGER)",
            "PRAGMA table_info(ventas)",
            "DETACH DATABASE main",
        ] {
            let err = e.query(sql, 10).expect_err("must refuse");
            assert!(
                matches!(err, TableError::NotReadOnly(_)),
                "not refused as read-only: {sql} -> {err:?}"
            );
        }
    }

    #[test]
    fn a_write_that_opens_with_with_is_refused_by_sqlite_itself() {
        // THE case that justifies asking `Statement::readonly()` at all. It
        // starts with WITH, so the opening-keyword whitelist lets it through;
        // only SQLite knows it writes.
        //
        // Found by mutation: removing the `readonly()` check broke no test,
        // because every other write in the suite is refused earlier by its
        // opening keyword. A layer whose removal changes nothing is either
        // redundant or untested, and this one was untested.
        let e = loaded();
        let sql = "WITH nuevo AS (SELECT 'este' AS r) \
                   INSERT INTO ventas SELECT r, 'Q4', 1 FROM nuevo";
        let err = e
            .query(sql, 10)
            .expect_err("a WITH that writes must be refused");
        assert!(
            matches!(err, TableError::NotReadOnly(_)),
            "a WITH ... INSERT reached the engine: {err:?}"
        );

        // And the read-only twin is allowed, so the refusal is about writing and
        // not about the word WITH.
        let ok = e
            .query(
                "WITH t AS (SELECT importe FROM ventas) SELECT SUM(importe) FROM t",
                10,
            )
            .expect("a WITH that only reads is fine");
        assert_eq!(ok.rows[0][0], Cell::Int(800));
    }

    #[test]
    fn attach_is_refused_even_when_the_file_exists() {
        // THE test that changed the design. `Statement::readonly()` classifies
        // ATTACH as read-only, because it does not modify the contents of any
        // database. The first version of this engine relied on that check alone,
        // and `ATTACH DATABASE 'C:/evil.db'` only failed because the file was
        // missing — with the file present it would have worked.
        //
        // So this test uses a file that DOES exist, which is the only version of
        // it that proves anything.
        let real = csv_file("existe.db", "not really a database, but present\n");
        assert!(
            real.exists(),
            "the point of this test is that the file is there"
        );

        let e = loaded();
        // MAIN_SEPARATOR rather than a literal backslash: SQLite wants forward
        // slashes, and this reads the same on both platforms.
        let path = real
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let sql = format!("ATTACH DATABASE '{path}' AS x");
        let err = e.query(&sql, 10).expect_err("must refuse");
        match &err {
            TableError::NotReadOnly(msg) => assert!(
                msg.contains("SELECT"),
                "should be refused by the opening-keyword rule: {msg}"
            ),
            other => panic!("ATTACH reached the engine: {other:?}"),
        }
    }

    #[test]
    fn a_comment_cannot_hide_the_opening_keyword() {
        let e = loaded();
        // Legitimate: a model explaining itself before the query.
        let ok = e
            .query(
                "-- suma del tercer trimestre\nSELECT SUM(importe) FROM ventas",
                10,
            )
            .expect("a leading comment is fine");
        assert_eq!(ok.row_count, 1);
        let ok2 = e
            .query("/* nota */ SELECT COUNT(*) FROM ventas", 10)
            .expect("a block comment is fine");
        assert_eq!(ok2.row_count, 1);

        // And the same trick used to hide something else.
        for sql in [
            "-- SELECT\nDROP TABLE ventas",
            "/* SELECT */ DELETE FROM ventas",
            "/* unterminated DROP TABLE ventas",
        ] {
            assert!(
                matches!(e.query(sql, 10), Err(TableError::NotReadOnly(_))),
                "hidden behind a comment: {sql}"
            );
        }
    }

    #[test]
    fn a_semicolon_inside_a_literal_is_not_a_second_statement() {
        // The reason this is a scan and not a search for ';'. If a plain search
        // were used, every one of these would be refused — and a refusal for a
        // legitimate query teaches people to work around the check.
        let e = loaded();
        for sql in [
            "SELECT ';' AS punto",
            "SELECT 'a;b' AS mezcla",
            "SELECT '' || ';' AS concat",
            "SELECT 1 -- termina aqui ;",
            "SELECT 1 /* ; */ + 1",
            // A trailing semicolon on its own: plenty of tools add one.
            "SELECT 1;",
            "SELECT 1;   ",
        ] {
            assert!(
                e.query(sql, 10).is_ok(),
                "refused a legitimate query: {sql}"
            );
        }
    }

    #[test]
    fn has_trailing_statement_knows_what_it_is_inside() {
        assert!(!has_trailing_statement("SELECT 1"));
        assert!(!has_trailing_statement("SELECT 1;"));
        assert!(!has_trailing_statement("SELECT ';'"));
        assert!(!has_trailing_statement("SELECT \"co;lumna\" FROM t"));
        assert!(!has_trailing_statement("SELECT [co;l] FROM t"));
        assert!(!has_trailing_statement("SELECT `co;l` FROM t"));
        assert!(!has_trailing_statement("SELECT 1 -- ;\n"));
        assert!(!has_trailing_statement("SELECT 1 /* ; */"));
        // These two are the ones that DISCRIMINATE. The pair above does not:
        // with comment-skipping removed the `;` looks top level, but nothing
        // follows it, so both versions answer false. A mutation survived on
        // exactly that, which is how the gap was found — the comment has to
        // contain both the `;` and something after it.
        assert!(!has_trailing_statement("SELECT 1 -- ; DROP TABLE t"));
        assert!(!has_trailing_statement("SELECT 1 /* ; DROP TABLE t */"));
        // Same shape for a literal: the `;` and what follows both inside.
        assert!(!has_trailing_statement("SELECT '; DROP TABLE t' AS s"));
        assert!(!has_trailing_statement("SELECT 'it''s ; fine' AS s"));

        assert!(has_trailing_statement("SELECT 1; DROP TABLE t"));
        assert!(has_trailing_statement("SELECT 1 ; SELECT 2"));
        assert!(has_trailing_statement("SELECT ';'; DROP TABLE t"));
        assert!(has_trailing_statement(
            "SELECT 1; -- innocuous, still a second"
        ));
    }

    #[test]
    fn first_keyword_skips_comments_and_whitespace() {
        assert_eq!(first_keyword("SELECT 1"), "SELECT");
        assert_eq!(first_keyword("  \n\tselect 1"), "SELECT");
        assert_eq!(
            first_keyword("-- hola\nWITH a AS (SELECT 1) SELECT * FROM a"),
            "WITH"
        );
        assert_eq!(first_keyword("/* x */ VALUES (1)"), "VALUES");
        assert_eq!(first_keyword("-- a\n-- b\n/* c */ SELECT 1"), "SELECT");
        // Nothing to find is refused, which is the safe direction.
        assert_eq!(first_keyword(""), "");
        assert_eq!(first_keyword("   "), "");
        assert_eq!(first_keyword("/* never closed SELECT"), "");
        assert_eq!(
            first_keyword("123 SELECT"),
            "",
            "does not start with a keyword"
        );
    }

    #[test]
    fn a_second_statement_never_reaches_execution() {
        let e = loaded();
        let err = e
            .query("SELECT 1; DROP TABLE ventas", 10)
            .expect_err("must refuse");
        assert_eq!(err, TableError::MultipleStatements);
        // And the table is still there.
        assert!(e.describe("ventas").is_ok());
    }

    #[test]
    fn a_word_that_only_looks_like_a_write_is_allowed() {
        // Guards the reason we ask SQLite instead of parsing: `DELETE` inside a
        // string literal is not a write, and a regex would refuse it.
        let e = loaded();
        let r = e
            .query("SELECT 'DELETE FROM ventas' AS texto", 10)
            .expect("a literal is not a statement");
        assert_eq!(r.rows[0][0], Cell::Text("DELETE FROM ventas".to_string()));
    }

    #[test]
    fn truncation_is_reported_and_not_guessed() {
        let e = loaded();
        let all = e.query("SELECT * FROM ventas", 10).expect("query");
        assert_eq!(all.row_count, 4);
        assert!(!all.truncated);

        let cut = e.query("SELECT * FROM ventas", 2).expect("query");
        assert_eq!(cut.row_count, 2);
        assert!(
            cut.truncated,
            "2 of 4 must say so; otherwise it reads as a table with 2 rows"
        );
    }

    #[test]
    fn no_rows_is_an_answer_not_an_error() {
        let e = loaded();
        let r = e
            .query("SELECT * FROM ventas WHERE region = 'este'", 10)
            .expect("a query that matches nothing still ran");
        assert!(r.is_empty());
        assert!(!r.truncated);
        assert_eq!(r.columns.len(), 3, "the shape is known even with no rows");
    }

    #[test]
    fn bad_sql_and_a_refused_write_are_different_errors() {
        let e = loaded();
        let bad = e
            .query("SELECT no_existe FROM ventas", 10)
            .expect_err("bad");
        assert!(matches!(bad, TableError::BadQuery(_)), "{bad:?}");
    }

    #[test]
    fn a_missing_table_names_the_ones_loaded() {
        let e = loaded();
        let err = e.describe("sales").expect_err("not loaded");
        assert_eq!(
            err,
            TableError::NoSuchTable {
                asked: "sales".to_string(),
                available: vec!["ventas".to_string()],
            }
        );
    }

    #[test]
    fn a_table_name_that_could_break_out_is_refused_at_load() {
        let p = csv_file("x.csv", "a\n1\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        let err = e
            .load("v\"; DROP TABLE x; --", &p)
            .expect_err("must refuse");
        assert!(matches!(err, TableError::BadTableName(_)), "{err:?}");
    }

    #[test]
    fn loading_the_same_name_twice_is_refused() {
        let p = csv_file("dup.csv", "a\n1\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        e.load("t", &p).expect("first");
        assert_eq!(
            e.load("t", &p).expect_err("second"),
            TableError::DuplicateTable("t".to_string())
        );
    }

    #[test]
    fn quoted_fields_with_commas_and_newlines_survive() {
        // The reason this uses the `csv` crate instead of splitting on commas.
        let p = csv_file(
            "quoted.csv",
            "nombre,nota\n\"Perez, Ana\",\"dijo:\nhola\"\n",
        );
        let mut e = SqliteTableEngine::new().expect("engine");
        let info = e.load("notas", &p).expect("load");
        assert_eq!(info.row_count, 1);
        let r = e
            .query("SELECT nombre, nota FROM notas", 10)
            .expect("query");
        assert_eq!(r.rows[0][0], Cell::Text("Perez, Ana".to_string()));
        assert_eq!(r.rows[0][1], Cell::Text("dijo:\nhola".to_string()));
    }

    #[test]
    fn an_empty_cell_becomes_null_and_not_zero() {
        // A blank in a numeric column is "unknown", not 0. Storing 0 would make
        // SUM and AVG quietly wrong, which is exactly the failure this module
        // exists to avoid.
        let p = csv_file("huecos.csv", "a,importe\nx,10\ny,\nz,30\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        e.load("huecos", &p).expect("load");

        let info = e.describe("huecos").expect("described");
        assert_eq!(
            info.columns[1].declared_type, "INTEGER",
            "one blank does not make it text"
        );

        let r = e
            .query("SELECT importe FROM huecos ORDER BY a", 10)
            .expect("q");
        assert_eq!(r.rows[1][0], Cell::Null);

        let sum = e.query("SELECT SUM(importe) FROM huecos", 10).expect("q");
        assert_eq!(sum.rows[0][0], Cell::Int(40));
        let avg = e.query("SELECT AVG(importe) FROM huecos", 10).expect("q");
        assert_eq!(
            avg.rows[0][0],
            Cell::Float(20.0),
            "AVG skips NULL, it does not average in a zero"
        );
    }

    #[test]
    fn what_one_unparseable_value_does_to_a_numeric_column() {
        // The case a real CSV will hit on day one: a column of numbers with a
        // single "N/A". Inference makes the whole column TEXT, and then SQLite
        // COERCES text to numbers in aggregates instead of refusing. This test
        // exists to record what actually happens, because the answer decides
        // whether the module is safe to expose to a model.
        let p = csv_file("na.csv", "a,importe\nx,10\ny,N/A\nz,30\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        let info = e.load("na", &p).expect("load");
        assert_eq!(
            info.columns[1].declared_type, "TEXT",
            "one unparseable value turns the column to TEXT"
        );

        // SUM's VALUE happens to be right: 'N/A' coerces to 0, which for a sum
        // is the same as skipping it. But the TYPE changed: SQLite returns
        // INTEGER when every input is an integer and REAL otherwise, so one
        // stray 'N/A' anywhere in the column turns every sum of it into a
        // float. A caller that matched on `Cell::Int` now falls through.
        let sum = e.query("SELECT SUM(importe) FROM na", 10).expect("q");
        assert_eq!(sum.rows[0][0], Cell::Float(40.0));

        // AVG is NOT. It divides by three because 'N/A' coerced to a real 0,
        // where a NULL would have been skipped. 40/3 instead of 40/2.
        let avg = e.query("SELECT AVG(importe) FROM na", 10).expect("q");
        match &avg.rows[0][0] {
            Cell::Float(v) => assert!(
                (v - 13.333_333_333_333_334).abs() < 1e-9,
                "AVG over a coerced 'N/A' gives {v}; a NULL would have given 20"
            ),
            other => panic!("expected a float, got {other:?}"),
        }

        // And COUNT of the column counts it, because TEXT 'N/A' is not NULL.
        let n = e.query("SELECT COUNT(importe) FROM na", 10).expect("q");
        assert_eq!(
            n.rows[0][0],
            Cell::Int(3),
            "COUNT sees three values where only two are numbers"
        );

        // NONE of the three errors. So the only thing standing between this and
        // a confidently wrong answer is that the result SAYS SO — which is why
        // `warnings` is a required field and not a log line.
        for r in [&sum, &avg, &n] {
            assert!(!r.is_trustworthy(), "a query over a mixed column must warn");
            let w = r.warnings.join(" ");
            assert!(
                w.contains("importe"),
                "the warning must name the column: {w}"
            );
            assert!(
                w.contains("\"N/A\" x1"),
                "and the values that did not fit: {w}"
            );
            assert!(w.contains("AVG"), "and what it does to aggregates: {w}");
        }

        // The column is reported, not guessed at.
        let col = &info.columns[1];
        assert!(col.is_mixed_numeric());
        assert_eq!(col.numeric_values, 2);
        assert_eq!(col.unparseable, vec![("N/A".to_string(), 1)]);
    }

    #[test]
    fn declaring_the_missing_marker_makes_the_column_numeric_again() {
        // The caller knows what their file uses for "no value"; we do not. Once
        // told, every number above becomes the right one.
        let p = csv_file("na2.csv", "a,importe\nx,10\ny,N/A\nz,30\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        let info = e
            .load_with("na", &p, &LoadOptions::treating_as_missing(["N/A"]))
            .expect("load");

        assert_eq!(info.columns[1].declared_type, "INTEGER");
        assert!(
            !info.columns[1].is_mixed_numeric(),
            "nothing unparseable is left"
        );

        let sum = e.query("SELECT SUM(importe) FROM na", 10).expect("q");
        assert_eq!(
            sum.rows[0][0],
            Cell::Int(40),
            "an integer again, not a float"
        );

        let avg = e.query("SELECT AVG(importe) FROM na", 10).expect("q");
        assert_eq!(avg.rows[0][0], Cell::Float(20.0), "40/2, not 40/3");

        let n = e.query("SELECT COUNT(importe) FROM na", 10).expect("q");
        assert_eq!(n.rows[0][0], Cell::Int(2), "the NULL is not counted");

        for r in [&sum, &avg, &n] {
            assert!(
                r.is_trustworthy(),
                "nothing left to warn about: {:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn the_missing_marker_is_matched_ignoring_case_and_padding() {
        // A file that writes `N/A` in one row and ` n/a ` in the next is the
        // normal case, not the exotic one.
        let p = csv_file("na3.csv", "a,importe\nx,10\ny,N/A\nz, n/a \nw,30\n");
        let mut e = SqliteTableEngine::new().expect("engine");
        let info = e
            .load_with("na", &p, &LoadOptions::treating_as_missing(["n/a"]))
            .expect("load");
        assert_eq!(info.columns[1].declared_type, "INTEGER");
        assert!(!info.columns[1].is_mixed_numeric());
    }

    #[test]
    fn an_honest_query_carries_no_warnings() {
        // So a warning means something when it appears.
        let e = loaded();
        let r = e.query("SELECT SUM(importe) FROM ventas", 10).expect("q");
        assert!(r.is_trustworthy());
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn a_column_of_honest_text_is_not_flagged() {
        // `region` holds only words. Flagging it would train people to ignore
        // warnings, which is worse than not having them.
        let e = loaded();
        let info = e.describe("ventas").expect("described");
        assert!(
            !info.columns[0].is_mixed_numeric(),
            "a text column with no numbers is not mixed"
        );
        let r = e.query("SELECT region FROM ventas", 10).expect("q");
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn inference_rules() {
        fn t(values: &[&str]) -> Survey {
            let owned: Vec<String> = values.iter().map(|s| (*s).to_string()).collect();
            survey_column(&owned, &LoadOptions::default())
        }

        assert_eq!(t(&["1", "2"]).inferred, Inferred::Integer);
        assert_eq!(t(&["1", "2.5"]).inferred, Inferred::Real);
        assert_eq!(t(&["1", "x"]).inferred, Inferred::Text);
        assert_eq!(
            t(&["1", ""]).inferred,
            Inferred::Integer,
            "one blank does not make it text"
        );
        assert_eq!(t(&[]).inferred, Inferred::Text, "nothing to go on");
        assert_eq!(t(&["", ""]).inferred, Inferred::Text, "all blank");

        // And the counting, which is the part that makes the warning possible.
        let mixed = t(&["1", "x", "2", "x", "y"]);
        assert_eq!(mixed.numeric_values, 2);
        assert_eq!(
            mixed.unparseable,
            vec![("x".to_string(), 2), ("y".to_string(), 1)],
            "distinct values with counts, in the order first seen"
        );

        // A column of honest text counts nothing as numeric, so it is not
        // "mixed" and must not be flagged.
        let words = t(&["norte", "sur"]);
        assert_eq!(words.numeric_values, 0);
        assert!(!words.unparseable.is_empty());
    }
}
