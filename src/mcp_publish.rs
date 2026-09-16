//! What the kit publishes over MCP, and what the operator lets it publish.
//!
//! # The gap this closes
//!
//! `ai_mcp_server` registered five groups of tools. The library had ready-made
//! registration functions for five more — agents, containers, events, home
//! automation, voice — and **nothing called them**. They were not experiments:
//! each is a complete `register_*` taking one shared dependency. A client
//! talking to this server simply could not reach half of what the crate can do.
//!
//! # Why a switch rather than "publish everything"
//!
//! "Search my notes" and "start a container on my machine" are not the same
//! kind of offer. MCP hands a tool list to a client that may act on it with
//! little supervision, so the question "may this be reached from outside?" has
//! to be answerable per group, and answered conservatively when nobody said.
//!
//! The default is therefore **by risk, not by convenience**: groups that only
//! read, and groups that write within the kit's own data, are published;
//! groups that **execute** — run containers, drive an agent, operate devices in
//! a house — are not, until an operator says so in writing.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// What a group of tools lets a client do, which is what decides whether it is
/// published by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Risk {
    /// Answers questions. Cannot change anything.
    ReadOnly,
    /// Changes data the kit owns — its tasks, its notes, its configuration.
    Writes,
    /// Makes something happen outside the kit: a process, a device, an agent
    /// acting on its own.
    Executes,
}

impl Risk {
    /// Published unless the operator says otherwise?
    ///
    /// `Executes` is not, and that is the whole point of this type. Everything
    /// else is, because a server that publishes nothing by default is a server
    /// nobody switches on.
    pub fn published_by_default(&self) -> bool {
        !matches!(self, Risk::Executes)
    }
}

/// One publishable group of MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolGroup {
    /// Read and change the kit's own configuration.
    Config,
    /// The task list: search, add, complete.
    Tasks,
    /// Knowledge base search and retrieval.
    Knowledge,
    /// Academic search — arXiv, Semantic Scholar, PubMed.
    Research,
    /// Run and read the benchmark suites.
    Benchmarks,
    /// Drive the multi-agent pool.
    Agents,
    /// Create, start and stop containers.
    Containers,
    /// Subscribe to and manage event sources.
    Events,
    /// Read and operate home-automation devices.
    Home,
    /// Speaker identification and voice gating.
    Voice,
}

impl ToolGroup {
    /// Every group, in the order they should be listed.
    pub fn all() -> &'static [ToolGroup] {
        &[
            ToolGroup::Config,
            ToolGroup::Tasks,
            ToolGroup::Knowledge,
            ToolGroup::Research,
            ToolGroup::Benchmarks,
            ToolGroup::Agents,
            ToolGroup::Containers,
            ToolGroup::Events,
            ToolGroup::Home,
            ToolGroup::Voice,
        ]
    }

    /// The key an operator writes in the configuration file.
    pub fn key(&self) -> &'static str {
        match self {
            ToolGroup::Config => "config",
            ToolGroup::Tasks => "tasks",
            ToolGroup::Knowledge => "knowledge",
            ToolGroup::Research => "research",
            ToolGroup::Benchmarks => "benchmarks",
            ToolGroup::Agents => "agents",
            ToolGroup::Containers => "containers",
            ToolGroup::Events => "events",
            ToolGroup::Home => "home",
            ToolGroup::Voice => "voice",
        }
    }

    /// What publishing this group lets a client do. Written for whoever has to
    /// decide, which is a person, not a program.
    pub fn describes(&self) -> &'static str {
        match self {
            ToolGroup::Config => "read and change this kit's own settings",
            ToolGroup::Tasks => "search, add and complete items on your task list",
            ToolGroup::Knowledge => "search your knowledge base and read passages from it",
            ToolGroup::Research => "search academic sources and fetch paper metadata",
            ToolGroup::Benchmarks => "run the benchmark suites and read their results",
            ToolGroup::Agents => "start and steer autonomous agents",
            ToolGroup::Containers => "create, start and stop containers on this machine",
            ToolGroup::Events => "subscribe to event sources and change their rules",
            ToolGroup::Home => "read and operate devices in your home",
            ToolGroup::Voice => "identify speakers and control who the assistant listens to",
        }
    }

    /// What this group can do, which decides the default.
    pub fn risk(&self) -> Risk {
        match self {
            ToolGroup::Knowledge | ToolGroup::Research => Risk::ReadOnly,
            ToolGroup::Config | ToolGroup::Tasks => Risk::Writes,
            // Each of these reaches outside the kit: a process on the machine,
            // a device in a house, or an agent that will keep going on its own.
            ToolGroup::Benchmarks
            | ToolGroup::Agents
            | ToolGroup::Containers
            | ToolGroup::Events
            | ToolGroup::Home
            // Deciding who the assistant will listen to is a security control,
            // not a write. It was classified as a write at first and the server
            // then announced on every start-up that it could not supply a
            // speaker verifier — a default that can never work is a bad default
            // however it is reported.
            | ToolGroup::Voice => Risk::Executes,
        }
    }

    /// Whether this is published when nobody has said anything.
    pub fn published_by_default(&self) -> bool {
        self.risk().published_by_default()
    }
}

/// Which groups this server publishes.
///
/// Absent a file, [`Self::default`] applies: everything but the groups that
/// execute. An operator turns those on explicitly, one at a time, having read
/// what each one allows.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PublishConfig {
    /// Overrides, keyed by [`ToolGroup::key`]. Anything absent falls back to
    /// the risk-based default.
    #[serde(default)]
    pub groups: BTreeMap<String, bool>,
}

impl PublishConfig {
    /// File name looked for next to the server binary.
    pub const FILE_NAME: &'static str = "mcp-publish.json";

    /// Load the operator's choices, if they wrote any.
    ///
    /// A missing file means "use the defaults" and is not an error. A malformed
    /// one **is**: quietly falling back to defaults would publish groups the
    /// operator may have been trying to switch off, which is the worst possible
    /// way to handle a typo in a security-relevant file.
    pub fn load_beside(dir: &Path) -> Result<Option<PublishConfig>, String> {
        let path = dir.join(Self::FILE_NAME);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        let parsed: PublishConfig = serde_json::from_str(&text)
            .map_err(|e| format!("{} is not valid configuration: {e}", path.display()))?;
        Ok(Some(parsed))
    }

    /// Whether a group should be published.
    pub fn publishes(&self, group: ToolGroup) -> bool {
        self.groups
            .get(group.key())
            .copied()
            .unwrap_or_else(|| group.published_by_default())
    }

    /// Every group that will not be published, with the reason — the same
    /// courtesy the server already extends for feature-gated groups. A client
    /// that cannot find a tool otherwise has no way to tell a deliberate policy
    /// from a bug.
    pub fn withheld(&self) -> Vec<(ToolGroup, String)> {
        ToolGroup::all()
            .iter()
            .filter(|g| !self.publishes(**g))
            .map(|g| {
                let reason = if self.groups.contains_key(g.key()) {
                    format!("switched off in {}", Self::FILE_NAME)
                } else {
                    format!("not published unless asked for: it can {}", g.describes())
                };
                (*g, reason)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_that_execute_are_not_published_unless_asked_for() {
        // The whole policy in one test. "Search my notes" and "start a
        // container on my machine" are not the same offer, and the second one
        // waits for an operator to say so in writing.
        let config = PublishConfig::default();

        assert!(config.publishes(ToolGroup::Knowledge));
        assert!(config.publishes(ToolGroup::Research));
        assert!(config.publishes(ToolGroup::Tasks));

        assert!(!config.publishes(ToolGroup::Containers));
        assert!(!config.publishes(ToolGroup::Agents));
        assert!(!config.publishes(ToolGroup::Home));
    }

    #[test]
    fn an_operator_can_switch_anything_on_or_off() {
        let mut config = PublishConfig::default();
        config.groups.insert("containers".into(), true);
        config.groups.insert("knowledge".into(), false);

        assert!(
            config.publishes(ToolGroup::Containers),
            "asked for explicitly"
        );
        assert!(
            !config.publishes(ToolGroup::Knowledge),
            "refused explicitly"
        );
    }

    #[test]
    fn everything_withheld_says_why() {
        // A shorter tool list with no explanation is indistinguishable from a
        // broken server.
        let config = PublishConfig::default();
        let withheld = config.withheld();

        assert!(!withheld.is_empty());
        for (group, reason) in &withheld {
            assert!(
                !reason.is_empty(),
                "{} withheld with no reason",
                group.key()
            );
            assert!(
                reason.contains("can ") || reason.contains("switched off"),
                "reason must name the policy or the capability: {reason}"
            );
        }
    }

    #[test]
    fn a_group_switched_off_says_so_rather_than_quoting_policy() {
        let mut config = PublishConfig::default();
        config.groups.insert("research".into(), false);

        let (_, reason) = config
            .withheld()
            .into_iter()
            .find(|(g, _)| *g == ToolGroup::Research)
            .expect("withheld");
        assert!(reason.contains("switched off"), "{reason}");
    }

    #[test]
    fn every_group_has_a_key_a_description_and_a_risk() {
        // Adding a variant without these is how a group ends up publishable by
        // accident, or unexplainable when it is not.
        let mut keys = std::collections::HashSet::new();
        for group in ToolGroup::all() {
            assert!(!group.key().is_empty());
            assert!(keys.insert(group.key()), "duplicate key {}", group.key());
            assert!(
                group.describes().len() > 10,
                "{} needs a description a person can decide on",
                group.key()
            );
        }
        assert_eq!(keys.len(), ToolGroup::all().len());
    }

    #[test]
    fn a_missing_file_is_the_normal_case_and_a_broken_one_is_not() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(PublishConfig::load_beside(dir.path()), Ok(None));

        // Falling back to defaults on a typo would publish groups the operator
        // may have been trying to withhold.
        std::fs::write(dir.path().join(PublishConfig::FILE_NAME), "{ nope").expect("write");
        let err = PublishConfig::load_beside(dir.path()).expect_err("must not be ignored");
        assert!(err.contains("not valid configuration"), "{err}");
    }

    #[test]
    fn an_operators_file_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(PublishConfig::FILE_NAME),
            r#"{"groups":{"containers":true,"research":false}}"#,
        )
        .expect("write");

        let config = PublishConfig::load_beside(dir.path())
            .expect("valid")
            .expect("present");
        assert!(config.publishes(ToolGroup::Containers));
        assert!(!config.publishes(ToolGroup::Research));
        // Anything unmentioned keeps the risk-based default.
        assert!(config.publishes(ToolGroup::Knowledge));
        assert!(!config.publishes(ToolGroup::Home));
    }
}
