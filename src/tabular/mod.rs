// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Asking questions of tabular data, by querying it rather than retrieving it.
//!
//! # Why RAG does not apply here
//!
//! Not "works badly" — does not apply:
//!
//! - **Chunking destroys a table.** A CSV split into 500-token pieces leaves the
//!   header in chunk 1, so chunks 2..n are rows without column names.
//! - **Embedding numbers means nothing.** A row of figures has no semantic
//!   content to embed, and with TF-IDF less than nothing.
//! - **The question is usually an aggregate.** "How much did Q3 add up to?" is
//!   not answered by retrieving anything. It has to be computed.
//!
//! So the model gets a **tool**, not context. What must never happen is the
//! table entering the prompt; what enters is the result of a query.
//!
//! # The disclosure is in the type, not in the code
//!
//! [`QueryResult`] carries [`QueryResult::sql_executed`] and
//! [`QueryResult::row_count`] as required fields. That is deliberate and it is
//! the most important line in this module: **no engine can return a result
//! without saying what it ran and how much came back**, because the type will
//! not let it.
//!
//! The reason is text-to-SQL. Small models get SQL wrong often, and **a wrong
//! query that executes is the worst possible case** — it returns a number with
//! complete confidence. A rule written in a comment gets forgotten in a
//! refactor; a required field does not.
//!
//! # Two engines, chosen by cost
//!
//! | feature | engine | cost |
//! |---|---|---|
//! | `tabular-sqlite` (default) | SQLite, already in the tree for the knowledge base | ~0 |
//! | `tabular-polars` | Polars: Parquet, lazy evaluation, files that do not fit in memory | **+49.3 MiB** of binary, measured |
//!
//! SQLite cannot read Parquet. That — not novelty — is the concrete reason the
//! second engine exists.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[cfg(feature = "tabular-polars")]
pub mod polars_engine;
#[cfg(feature = "tabular-sqlite")]
pub mod sqlite_engine;

#[cfg(all(
    feature = "tabular",
    not(any(feature = "tabular-sqlite", feature = "tabular-polars"))
))]
compile_error!(
    "feature `tabular` needs an engine: enable `tabular-sqlite` (free, already in the \
     dependency tree) or `tabular-polars` (+49.3 MiB). Without one, every call would \
     fail at run time with a confusing message instead of here."
);

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// One cell, in a shape both engines can produce.
///
/// Engine-neutral on purpose: it is what lets a test run the same queries
/// through SQLite and Polars and demand the same answer. Two engines that
/// disagree silently are worse than one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cell {
    /// SQL NULL, or a missing value.
    Null,
    /// Whole number.
    Int(i64),
    /// Real number.
    Float(f64),
    /// Text.
    Text(String),
    /// Boolean.
    Bool(bool),
}

impl std::fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Printed as an empty pair of brackets rather than nothing: a blank
            // cell and a NULL read identically on screen, and they are not the
            // same answer.
            Self::Null => write!(f, "[null]"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

/// A column, as the engine understands it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnInfo {
    /// Column name, as it must be written in SQL.
    pub name: String,
    /// The type the engine inferred or was told. Engines differ in wording, so
    /// this is for a human and for the model's prompt, not for comparison.
    pub declared_type: String,
    /// How many values parsed as a number.
    pub numeric_values: usize,
    /// Values that did not parse as a number, with how often each appeared.
    ///
    /// Empty for a column that is honestly text. Non-empty **together with**
    /// `numeric_values > 0` is the dangerous case — see
    /// [`Self::is_mixed_numeric`].
    pub unparseable: Vec<(String, usize)>,
}

impl ColumnInfo {
    /// Does this column hold numbers **and** something that is not a number?
    ///
    /// # Why this is worth a method of its own
    ///
    /// Measured on SQLite, for a column `10, N/A, 30` loaded as TEXT because of
    /// the one stray value:
    ///
    /// | query | answer | what it should be |
    /// |---|---|---|
    /// | `SUM` | `40.0` — a **float**, where an all-integer column gives an integer | 40 |
    /// | `AVG` | **13.33** — divides by three, because `'N/A'` coerced to a real zero | 20 |
    /// | `COUNT` | **3** — `'N/A'` is text, and text is not NULL | 2 |
    ///
    /// So two of the three are silently wrong and the third changes type. None
    /// of them errors. A model reading `AVG` states a confidently wrong average,
    /// which is the exact failure the whole module exists to prevent — and one
    /// unparseable value in a thousand is enough to cause it.
    pub fn is_mixed_numeric(&self) -> bool {
        self.numeric_values > 0 && !self.unparseable.is_empty()
    }

    /// How many values did not parse, counting repeats.
    pub fn unparseable_count(&self) -> usize {
        self.unparseable.iter().map(|(_, n)| n).sum()
    }
}

/// A loaded table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableInfo {
    /// Name to use in SQL.
    pub name: String,
    /// Where it came from.
    pub source: PathBuf,
    /// How many rows were loaded.
    pub row_count: usize,
    /// Columns, in order.
    pub columns: Vec<ColumnInfo>,
}

impl TableInfo {
    /// Columns that hold numbers **and** something that is not a number.
    ///
    /// See [`ColumnInfo::is_mixed_numeric`] for why these are the dangerous
    /// ones.
    pub fn mixed_numeric_columns(&self) -> Vec<&ColumnInfo> {
        self.columns
            .iter()
            .filter(|c| c.is_mixed_numeric())
            .collect()
    }
}

/// What to treat as "no value" when loading.
///
/// There is no way to know what a real CSV uses. An empty field is the only
/// unambiguous case, so it is the only default; `-`, `N/A`, `NULL`, `?` and `.`
/// are all plausible and all guesses, and **guessing which one means "missing"
/// is a decision that belongs to whoever knows the data**.
///
/// What the loader does instead of guessing is *report*: a column that holds
/// numbers and something else is recorded in [`ColumnInfo::unparseable`], so the
/// caller can see the candidates and say which ones count.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadOptions {
    /// Values to store as NULL. The empty string is always included.
    pub na_values: Vec<String>,
}

impl LoadOptions {
    /// Treat these values as missing, on top of the empty string.
    pub fn treating_as_missing<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            na_values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// Is `value` a missing marker?
    ///
    /// Trimmed and case-insensitive, because a file that writes `N/A` in one row
    /// and `n/a` in the next is the normal case, not the exotic one.
    pub fn is_missing(&self, value: &str) -> bool {
        let t = value.trim();
        if t.is_empty() {
            return true;
        }
        self.na_values
            .iter()
            .any(|n| n.trim().eq_ignore_ascii_case(t))
    }
}

/// The result of one query, including what was run to get it.
///
/// Every field is required. See the module documentation for why
/// `sql_executed` and `row_count` are not optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    /// **The SQL that actually ran.** Show this to the user, not to a log.
    pub sql_executed: String,
    /// **How many rows came back.** Zero rows is an answer; it is not an error,
    /// and it must not read like one.
    pub row_count: usize,
    /// Column names, in the order the cells appear.
    pub columns: Vec<String>,
    /// The rows, capped by the caller's limit.
    pub rows: Vec<Vec<Cell>>,
    /// Whether the limit cut the result short.
    ///
    /// Without this, "10 rows" from a limit of 10 is indistinguishable from a
    /// table that happened to have exactly 10 matches — and a model reading the
    /// second as the first will state a wrong total with confidence.
    pub truncated: bool,
    /// Which engine answered.
    pub engine: String,
    /// Reasons to distrust this answer, in plain language.
    ///
    /// **Required, like `sql_executed`.** A query that touches a column holding
    /// numbers and non-numbers cannot come back without saying so, because the
    /// type will not let it — and that case makes `AVG` and `COUNT` silently
    /// wrong (see [`ColumnInfo::is_mixed_numeric`]).
    ///
    /// Erring toward warning is deliberate: a warning that appears slightly too
    /// often is an annoyance, and one that fails to appear is a confidently
    /// wrong number in front of somebody who trusted it.
    pub warnings: Vec<String>,
}

impl QueryResult {
    /// Did the query match nothing?
    ///
    /// Distinct from an error on purpose. "No rows" and "the query was wrong"
    /// are different statements and read identically if both come back empty.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Is there any reason to distrust this answer?
    pub fn is_trustworthy(&self) -> bool {
        self.warnings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a tabular operation failed.
#[derive(Debug, Clone, PartialEq)]
pub enum TableError {
    /// The file could not be read.
    Io(String),
    /// The file was read and could not be parsed as a table.
    Parse(String),
    /// No table by that name is loaded.
    NoSuchTable {
        /// What was asked for.
        asked: String,
        /// What is loaded, so the caller — or the model — can correct itself.
        available: Vec<String>,
    },
    /// The SQL did not parse, or the engine rejected it.
    ///
    /// Kept separate from [`Self::NotReadOnly`] so "you wrote bad SQL" and
    /// "you asked me to change the data" are never confused.
    BadQuery(String),
    /// The statement would modify data, or reach outside the loaded tables.
    NotReadOnly(String),
    /// More than one statement in a single query.
    MultipleStatements,
    /// A table by that name is already loaded.
    DuplicateTable(String),
    /// The name cannot be used as a SQL identifier.
    BadTableName(String),
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cannot read the file: {e}"),
            Self::Parse(e) => write!(f, "cannot parse the file as a table: {e}"),
            Self::NoSuchTable { asked, available } => {
                if available.is_empty() {
                    write!(f, "no table named {asked:?}; none are loaded")
                } else {
                    write!(
                        f,
                        "no table named {asked:?}; loaded tables are: {}",
                        available.join(", ")
                    )
                }
            }
            Self::BadQuery(e) => write!(f, "the query could not run: {e}"),
            Self::NotReadOnly(e) => write!(
                f,
                "refused: this connection answers questions and never changes \
                 anything ({e})"
            ),
            Self::MultipleStatements => write!(
                f,
                "refused: more than one statement in a single query. One question, \
                 one statement — otherwise the second could be anything"
            ),
            Self::DuplicateTable(n) => write!(f, "a table named {n:?} is already loaded"),
            Self::BadTableName(n) => write!(
                f,
                "{n:?} cannot be used as a table name: use letters, digits and \
                 underscores, starting with a letter"
            ),
        }
    }
}

impl std::error::Error for TableError {}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Somewhere a table can be loaded and then asked questions.
///
/// Implementations must answer questions and never change anything. How that is
/// enforced belongs to the engine, because only the engine can do it
/// authoritatively — SQLite can be asked whether a prepared statement is
/// read-only; parsing the SQL ourselves to guess would be a worse answer
/// wearing a better disguise.
pub trait TableEngine: Send {
    /// Load `path` as a table called `name`, saying what counts as missing.
    fn load_with(
        &mut self,
        name: &str,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<TableInfo, TableError>;

    /// Load `path` as a table called `name`, treating only empty fields as
    /// missing.
    ///
    /// Anything else that is not a number is **reported**, not guessed at — see
    /// [`LoadOptions`].
    fn load(&mut self, name: &str, path: &Path) -> Result<TableInfo, TableError> {
        self.load_with(name, path, &LoadOptions::default())
    }

    /// Every table currently loaded.
    fn list_tables(&self) -> Vec<TableInfo>;

    /// One table's columns.
    ///
    /// The tool a model needs *before* `query`: one that does not know the
    /// column names invents SQL.
    fn describe(&self, name: &str) -> Result<TableInfo, TableError>;

    /// Run a read-only query, returning at most `limit` rows.
    fn query(&self, sql: &str, limit: usize) -> Result<QueryResult, TableError>;

    /// Which engine this is, for reports and for `QueryResult::engine`.
    fn engine_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Statement checks shared by every engine
// ---------------------------------------------------------------------------

/// Statement kinds any engine will run.
///
/// A whitelist, not a blacklist: a blacklist of dangerous keywords has to be
/// complete to work, and it will not be. `PRAGMA` is absent deliberately — some
/// pragmas only report, others change behaviour (`writable_schema = ON` lets you
/// corrupt the schema), and telling them apart is exactly the kind of judgement
/// that goes stale. Nothing is lost: [`TableEngine::describe`] covers the
/// legitimate use and is better, because it answers from our own data.
pub const ALLOWED_OPENINGS: &[&str] = &["SELECT", "WITH", "VALUES"];

/// The first SQL keyword, uppercased, with leading comments and whitespace
/// skipped.
///
/// Comments are skipped rather than rejected because `-- what this asks\nSELECT
/// …` is perfectly legitimate, and a model that explains itself in a comment
/// should not be refused. But they have to be skipped *correctly*, or
/// `--\nATTACH` reads as no keyword at all.
pub fn first_keyword(sql: &str) -> String {
    let b = sql.as_bytes();
    let mut i = 0usize;
    loop {
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i + 1 < b.len() && b[i] == b'-' && b[i + 1] == b'-' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // An unterminated block comment runs to the end, which leaves no
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
/// **Measured, and the reason this is ours and not the database's:** SQLite's
/// `Connection::prepare` does *not* reject a second statement.
/// `"SELECT 1; DROP TABLE ventas"` prepared fine, ran the `SELECT`, returned one
/// row and **discarded the rest without a word**. The `DROP` never executed, so
/// nothing was destroyed — but that is SQLite's design doing the work, not a
/// check, and "the second half of your query was silently thrown away" is a
/// wrong answer even when it is a safe one.
///
/// A plain search for `;` would be wrong: it appears inside string literals
/// (`SELECT ';'`), inside quoted identifiers, and inside comments. So this walks
/// the text tracking what it is inside, and only a top-level `;` counts.
pub fn has_trailing_statement(sql: &str) -> bool {
    let b = sql.as_bytes();
    let mut i = 0usize;
    let mut after_semicolon: Option<usize> = None;

    while i < b.len() {
        match b[i] {
            // Single-quoted string.
            //
            // No special case for the `''` escape, deliberately: this scan only
            // needs to know whether a character is inside a literal, and `''`
            // flips that state twice with **no character between the two quotes**
            // to misclassify. Handling it explicitly changed no answer — mutating
            // that branch away was how the redundancy was found.
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

    // A trailing `;` on its own is fine — plenty of tools add one. Anything else
    // after it is not, comments included: the simpler rule is easier to be sure
    // about.
    match after_semicolon {
        None => false,
        Some(rest) => !sql[rest..].trim().is_empty(),
    }
}

/// The checks every engine must pass a query through before running it.
///
/// **Shared on purpose.** Polars' SQL dialect accepts `INSERT`, `DELETE` and
/// `DROP`, and has no equivalent of SQLite's `Statement::readonly()`. If each
/// engine carried its own checks, the Polars one would be born weaker than the
/// SQLite one and every test would still pass — the shape of
/// the masked-defect pattern this project keeps hitting: the one configuration
/// that fused without reranking returned nothing, while five others looked fine.
/// (Not a link — that name is a memory file, and `[…]` in rustdoc means a symbol.)
///
/// An engine may add layers **on top** of this. It may not have fewer.
pub fn ensure_single_read_only_statement(sql: &str) -> Result<(), TableError> {
    if has_trailing_statement(sql) {
        return Err(TableError::MultipleStatements);
    }
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
    Ok(())
}

/// Is `name` usable as a table identifier?
///
/// Deliberately narrow: letters, digits and underscores, starting with a
/// letter. The name reaches SQL as an identifier, and the way to keep that safe
/// is to make the unsafe cases unrepresentable rather than to escape them.
pub fn is_valid_table_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The two engines must agree, or one of them is wrong and nobody can tell which.
///
/// Compiled only when both are enabled. That is the whole point: this is the test
/// that would catch a difference, and it cannot exist in a build that has one
/// engine.
#[cfg(all(test, feature = "tabular-sqlite", feature = "tabular-polars"))]
mod both_engines_agree {
    use super::*;
    use std::io::Write;

    fn csv_file(body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_both_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join("ventas.csv");
        let mut f = std::fs::File::create(&p).expect("create");
        f.write_all(body.as_bytes()).expect("write");
        p
    }

    const BODY: &str =
        "region,trimestre,importe\nnorte,Q1,100\nsur,Q1,250\nnorte,Q3,400\nsur,Q3,50\n";

    #[test]
    fn the_same_questions_get_the_same_answers() {
        let p = csv_file(BODY);

        let mut sq = sqlite_engine::SqliteTableEngine::new().expect("sqlite");
        sq.load("ventas", &p).expect("sqlite load");
        let mut pl = polars_engine::PolarsTableEngine::new();
        pl.load("ventas", &p).expect("polars load");

        // Both must see the same shape.
        let (a, b) = (
            sq.describe("ventas").expect("a"),
            pl.describe("ventas").expect("b"),
        );
        assert_eq!(a.row_count, b.row_count, "row counts differ");
        assert_eq!(
            a.columns.iter().map(|c| &c.name).collect::<Vec<_>>(),
            b.columns.iter().map(|c| &c.name).collect::<Vec<_>>(),
            "column names differ"
        );

        // And the same answers. Column NAMES are not compared: SQLite calls an
        // aggregate `SUM(importe)` and Polars keeps `importe`, which is a
        // cosmetic difference between dialects and not a disagreement about the
        // data. Aliasing with AS is how a caller gets a stable name from either.
        for sql in [
            "SELECT SUM(importe) AS total FROM ventas",
            "SELECT COUNT(*) AS n FROM ventas",
            "SELECT AVG(importe) AS media FROM ventas",
            "SELECT MIN(importe) AS lo, MAX(importe) AS hi FROM ventas",
            "SELECT SUM(importe) AS total FROM ventas WHERE trimestre = 'Q3'",
            "SELECT region, SUM(importe) AS total FROM ventas GROUP BY region ORDER BY region",
        ] {
            let ra = sq.query(sql, 100).expect("sqlite query");
            let rb = pl.query(sql, 100).expect("polars query");
            assert_eq!(ra.row_count, rb.row_count, "row count differs for: {sql}");
            assert_eq!(
                ra.rows, rb.rows,
                "the two engines disagree about the DATA for: {sql}\n  \
                 sqlite: {:?}\n  polars: {:?}",
                ra.rows, rb.rows
            );
        }
    }

    #[test]
    fn both_engines_refuse_the_same_statements() {
        // A refusal that only one engine makes is a hole in the other, and the
        // shared checks exist to make that impossible. This is the test that
        // notices if somebody gives one engine its own copy again.
        let p = csv_file(BODY);
        let mut sq = sqlite_engine::SqliteTableEngine::new().expect("sqlite");
        sq.load("ventas", &p).expect("load");
        let mut pl = polars_engine::PolarsTableEngine::new();
        pl.load("ventas", &p).expect("load");

        for sql in [
            "DELETE FROM ventas",
            "UPDATE ventas SET importe = 0",
            "DROP TABLE ventas",
            "INSERT INTO ventas VALUES ('x', 'Q4', 1)",
            "CREATE TABLE otra (a INTEGER)",
            "PRAGMA table_info(ventas)",
            "ATTACH DATABASE 'x.db' AS x",
            "SELECT 1; DROP TABLE ventas",
            "-- SELECT\nDROP TABLE ventas",
        ] {
            let ea = sq.query(sql, 10).expect_err("sqlite must refuse");
            let eb = pl.query(sql, 10).expect_err("polars must refuse");
            // The same variant, not merely both failing: "bad SQL" and "you
            // asked me to write" are different statements to the caller.
            assert_eq!(
                std::mem::discriminant(&ea),
                std::mem::discriminant(&eb),
                "different refusals for {sql}: sqlite {ea:?} vs polars {eb:?}"
            );
        }
    }

    #[test]
    fn both_engines_allow_the_same_legitimate_queries() {
        // The other half, and the one that catches a check grown too eager: a
        // shared guard that refuses valid SQL is as bad as one that permits
        // invalid SQL, because people route around it.
        let p = csv_file(BODY);
        let mut sq = sqlite_engine::SqliteTableEngine::new().expect("sqlite");
        sq.load("ventas", &p).expect("load");
        let mut pl = polars_engine::PolarsTableEngine::new();
        pl.load("ventas", &p).expect("load");

        for sql in [
            "SELECT ';' AS punto",
            "-- una nota\nSELECT COUNT(*) AS n FROM ventas",
            "/* otra */ SELECT COUNT(*) AS n FROM ventas",
            "SELECT COUNT(*) AS n FROM ventas;",
            "WITH t AS (SELECT importe FROM ventas) SELECT SUM(importe) AS total FROM t",
        ] {
            assert!(sq.query(sql, 10).is_ok(), "sqlite refused: {sql}");
            assert!(pl.query(sql, 10).is_ok(), "polars refused: {sql}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_name_must_be_a_plain_identifier() {
        assert!(is_valid_table_name("ventas"));
        assert!(is_valid_table_name("ventas_2026"));
        assert!(is_valid_table_name("T1"));

        assert!(!is_valid_table_name(""), "empty");
        assert!(!is_valid_table_name("2026_ventas"), "leading digit");
        assert!(!is_valid_table_name("ventas-2026"), "hyphen");
        assert!(!is_valid_table_name("ventas 2026"), "space");
        // The ones that matter: anything that could end a quoted identifier and
        // start something else.
        assert!(!is_valid_table_name("ventas\"; DROP TABLE x; --"));
        assert!(!is_valid_table_name("ventas;--"));
        assert!(!is_valid_table_name("ventas'"));
    }

    #[test]
    fn a_null_cell_does_not_print_as_a_blank() {
        // A blank cell and a NULL read identically on screen, and a model that
        // sees a blank will summarise it as "empty" rather than "unknown".
        assert_eq!(Cell::Null.to_string(), "[null]");
        assert_eq!(Cell::Text(String::new()).to_string(), "");
        assert_ne!(
            Cell::Null.to_string(),
            Cell::Text(String::new()).to_string()
        );
    }

    #[test]
    fn zero_rows_is_an_answer_and_not_an_error() {
        let r = QueryResult {
            sql_executed: "SELECT * FROM t WHERE 0".to_string(),
            row_count: 0,
            columns: vec!["a".to_string()],
            rows: vec![],
            truncated: false,
            engine: "test".to_string(),
            warnings: vec![],
        };
        assert!(r.is_empty());
        // And it still says what it ran, which is the whole point of the type.
        assert!(!r.sql_executed.is_empty());
    }

    #[test]
    fn the_two_refusals_read_differently() {
        // "You wrote bad SQL" and "you asked me to change the data" must never
        // be confused: the first is the model's mistake to correct, the second
        // is a boundary it should not have crossed.
        let bad = TableError::BadQuery("no such column: x".to_string());
        let write = TableError::NotReadOnly("DELETE".to_string());
        assert_ne!(bad, write);
        assert!(bad.to_string().contains("could not run"));
        assert!(write.to_string().contains("never changes anything"));
    }

    #[test]
    fn a_missing_table_lists_the_ones_that_exist() {
        // So a model that guessed the name can correct itself instead of
        // guessing again.
        let e = TableError::NoSuchTable {
            asked: "ventas".to_string(),
            available: vec!["sales".to_string(), "clientes".to_string()],
        };
        let msg = e.to_string();
        assert!(msg.contains("sales") && msg.contains("clientes"), "{msg}");

        let none = TableError::NoSuchTable {
            asked: "ventas".to_string(),
            available: vec![],
        };
        assert!(none.to_string().contains("none are loaded"));
    }
}
