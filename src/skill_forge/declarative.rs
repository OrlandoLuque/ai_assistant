//! Declarative skill executor — interprets a sequence of `SkillStep`s.
//!
//! Steps are structured data, authored by the LLM. The executor is
//! safe-by-construction: it does not execute arbitrary code. Side effects
//! (tool calls, LLM calls) are delegated to caller-supplied traits so the
//! caller retains control over policy and rate limits.

use super::capability::{Capability, CapabilityError, CapabilitySet};
use super::registry::{SkillError, SkillId, SkillInputs, SkillOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Step kinds
// =============================================================================

/// One step in a Declarative skill.
///
/// The `bind` field (optional) names a variable to store the step's result
/// in the executor's scope, allowing subsequent steps to reference it via
/// `{{var_name}}` in string fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SkillStep {
    pub kind: StepKind,
    /// Variable name to bind the step's result to. `None` = no binding.
    pub bind: Option<String>,
}

/// What a step does.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepKind {
    /// Call the LLM with a prompt. Result is a `String`.
    Plan { prompt: String },

    /// Invoke a named tool with JSON args. Requires
    /// `Capability::ToolCall(name)`.
    ToolCall {
        tool: String,
        args: serde_json::Value,
    },

    /// Transform the current scope with a named operation.
    Transform { op: TransformOp },

    /// Conditional branching. The `cond` expression is evaluated against
    /// current scope; true runs `then`, false runs `else_`.
    Branch {
        cond: Condition,
        then: Vec<SkillStep>,
        #[serde(rename = "else_", alias = "else")]
        else_: Vec<SkillStep>,
    },

    /// Return the value bound to a scope variable.
    Return { var: String },
}

/// Minimal transform vocabulary. Expanded over time as real skills
/// demand more operations. `op` stays `#[non_exhaustive]` to allow growth.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TransformOp {
    /// Pick a field from an object var.
    Pick { from: String, field: String },
    /// Concatenate strings from scope variables, in order.
    ConcatStrings { parts: Vec<String> },
    /// Store a literal value in scope (usually via `bind`).
    Literal { value: serde_json::Value },
}

/// Boolean condition for `StepKind::Branch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Condition {
    /// `var` equals the given value.
    Eq {
        var: String,
        value: serde_json::Value,
    },
    /// `var` is truthy (non-null, non-empty, non-zero, non-false).
    Truthy { var: String },
    /// Inverts the inner condition.
    Not { inner: Box<Condition> },
}

// =============================================================================
// Delegates (caller-supplied)
// =============================================================================

/// Caller-supplied LLM invocation for `StepKind::Plan`.
pub trait SkillPlanner: Send + Sync {
    fn plan(&self, prompt: &str) -> Result<String, String>;
}

/// Caller-supplied tool dispatcher for `StepKind::ToolCall`.
pub trait SkillToolDispatcher: Send + Sync {
    fn dispatch(&self, tool: &str, args: &serde_json::Value) -> Result<serde_json::Value, String>;
}

/// No-op planner that returns an error. Useful as a default when a skill
/// is not expected to make LLM calls.
pub struct RejectPlanner;

impl SkillPlanner for RejectPlanner {
    fn plan(&self, _prompt: &str) -> Result<String, String> {
        Err("no planner configured for this skill".into())
    }
}

/// No-op dispatcher.
pub struct RejectToolDispatcher;

impl SkillToolDispatcher for RejectToolDispatcher {
    fn dispatch(
        &self,
        _tool: &str,
        _args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        Err("no tool dispatcher configured for this skill".into())
    }
}

// =============================================================================
// Executor
// =============================================================================

/// Stateless executor of Declarative skills. Construct once with the
/// delegates, then call `execute` per invocation.
pub struct DeclarativeExecutor<P: SkillPlanner, D: SkillToolDispatcher> {
    planner: P,
    dispatcher: D,
    /// Maximum steps to execute (prevents runaway skills).
    pub max_steps: usize,
}

impl<P: SkillPlanner, D: SkillToolDispatcher> DeclarativeExecutor<P, D> {
    pub fn new(planner: P, dispatcher: D) -> Self {
        Self {
            planner,
            dispatcher,
            max_steps: 256,
        }
    }

    /// Execute the step list, returning a `SkillOutput`.
    ///
    /// `granted` is the caller's capability set; `ToolCall` steps require
    /// a matching `Capability::ToolCall` to be granted.
    pub fn execute(
        &self,
        skill_id: &SkillId,
        steps: &[SkillStep],
        inputs: &SkillInputs,
        granted: &CapabilitySet,
    ) -> Result<SkillOutput, SkillError> {
        let start = Instant::now();
        let mut scope = Scope::new(inputs);
        let mut trace = Vec::new();
        let mut budget = StepBudget::new(self.max_steps);
        self.run(
            skill_id,
            steps,
            &mut scope,
            &mut trace,
            granted,
            &mut budget,
        )?;
        Ok(SkillOutput {
            value: scope.return_value.take().unwrap_or(serde_json::Value::Null),
            trace,
            fuel_consumed: 0,
            wall_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn run(
        &self,
        skill_id: &SkillId,
        steps: &[SkillStep],
        scope: &mut Scope,
        trace: &mut Vec<String>,
        granted: &CapabilitySet,
        budget: &mut StepBudget,
    ) -> Result<(), SkillError> {
        for step in steps {
            budget.charge(skill_id)?;
            if scope.return_value.is_some() {
                return Ok(());
            }
            self.run_one(skill_id, step, scope, trace, granted, budget)?;
        }
        Ok(())
    }

    fn run_one(
        &self,
        skill_id: &SkillId,
        step: &SkillStep,
        scope: &mut Scope,
        trace: &mut Vec<String>,
        granted: &CapabilitySet,
        budget: &mut StepBudget,
    ) -> Result<(), SkillError> {
        let result: serde_json::Value =
            match &step.kind {
                StepKind::Plan { prompt } => {
                    let interpolated = scope.interpolate(prompt);
                    let text = self.planner.plan(&interpolated).map_err(|m| {
                        SkillError::ExecutionFailed {
                            skill: skill_id.clone(),
                            message: format!("plan failed: {m}"),
                        }
                    })?;
                    trace.push(format!("plan -> {} chars", text.len()));
                    serde_json::Value::String(text)
                }
                StepKind::ToolCall { tool, args } => {
                    let required = Capability::ToolCall(tool.clone());
                    if !granted.contains(&required) {
                        return Err(SkillError::CapabilityDenied {
                            skill: skill_id.clone(),
                            capability: required.to_string(),
                        });
                    }
                    let args_interp = scope.interpolate_json(args);
                    let out = self.dispatcher.dispatch(tool, &args_interp).map_err(|m| {
                        SkillError::ExecutionFailed {
                            skill: skill_id.clone(),
                            message: format!("tool '{tool}' failed: {m}"),
                        }
                    })?;
                    trace.push(format!("tool:{tool} ok"));
                    out
                }
                StepKind::Transform { op } => self.apply_transform(skill_id, op, scope)?,
                StepKind::Branch { cond, then, else_ } => {
                    let take_then = cond.evaluate(scope);
                    trace.push(format!(
                        "branch -> {}",
                        if take_then { "then" } else { "else" }
                    ));
                    let branch = if take_then {
                        then.as_slice()
                    } else {
                        else_.as_slice()
                    };
                    self.run(skill_id, branch, scope, trace, granted, budget)?;
                    return Ok(());
                }
                StepKind::Return { var } => {
                    let val = scope.vars.get(var).cloned().ok_or_else(|| {
                        SkillError::ExecutionFailed {
                            skill: skill_id.clone(),
                            message: format!("return: variable '{var}' not bound"),
                        }
                    })?;
                    scope.return_value = Some(val);
                    return Ok(());
                }
            };

        if let Some(name) = &step.bind {
            scope.vars.insert(name.clone(), result);
        }
        Ok(())
    }

    fn apply_transform(
        &self,
        skill_id: &SkillId,
        op: &TransformOp,
        scope: &mut Scope,
    ) -> Result<serde_json::Value, SkillError> {
        match op {
            TransformOp::Pick { from, field } => {
                let source = scope
                    .vars
                    .get(from)
                    .ok_or_else(|| SkillError::ExecutionFailed {
                        skill: skill_id.clone(),
                        message: format!("pick: variable '{from}' not bound"),
                    })?;
                let obj = source
                    .as_object()
                    .ok_or_else(|| SkillError::ExecutionFailed {
                        skill: skill_id.clone(),
                        message: format!("pick: '{from}' is not an object"),
                    })?;
                Ok(obj.get(field).cloned().unwrap_or(serde_json::Value::Null))
            }
            TransformOp::ConcatStrings { parts } => {
                let mut out = String::new();
                for p in parts {
                    if let Some(v) = scope.vars.get(p) {
                        if let Some(s) = v.as_str() {
                            out.push_str(s);
                        } else {
                            out.push_str(&v.to_string());
                        }
                    }
                }
                Ok(serde_json::Value::String(out))
            }
            TransformOp::Literal { value } => Ok(value.clone()),
        }
    }
}

// =============================================================================
// Scope
// =============================================================================

struct Scope {
    vars: HashMap<String, serde_json::Value>,
    return_value: Option<serde_json::Value>,
}

impl Scope {
    fn new(inputs: &SkillInputs) -> Self {
        let mut vars = HashMap::new();
        // Bind `$inputs` as the root input object.
        vars.insert("$inputs".to_string(), inputs.0.clone());
        // Also bind each top-level input field individually for convenience.
        if let Some(obj) = inputs.0.as_object() {
            for (k, v) in obj {
                vars.insert(k.clone(), v.clone());
            }
        }
        Self {
            vars,
            return_value: None,
        }
    }

    /// Replace `{{var_name}}` with the stringified value of `var_name`
    /// in `scope`. Unknown vars are replaced with empty string.
    fn interpolate(&self, template: &str) -> String {
        let mut out = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second {
                let mut name = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc == '}' {
                        break;
                    }
                    name.push(nc);
                    chars.next();
                }
                // consume }}
                if chars.next() == Some('}') && chars.next() == Some('}') {
                    // proper close
                } else {
                    // malformed — emit raw
                    out.push_str(&format!("{{{{{name}"));
                    continue;
                }
                let replacement = match self.vars.get(name.trim()) {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                out.push_str(&replacement);
            } else {
                out.push(c);
            }
        }
        out
    }

    /// Recursively interpolate string values inside a JSON value.
    fn interpolate_json(&self, v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::String(s) => serde_json::Value::String(self.interpolate(s)),
            serde_json::Value::Array(a) => {
                serde_json::Value::Array(a.iter().map(|x| self.interpolate_json(x)).collect())
            }
            serde_json::Value::Object(o) => {
                let mut m = serde_json::Map::new();
                for (k, val) in o {
                    m.insert(k.clone(), self.interpolate_json(val));
                }
                serde_json::Value::Object(m)
            }
            other => other.clone(),
        }
    }
}

impl Condition {
    fn evaluate(&self, scope: &Scope) -> bool {
        match self {
            Condition::Eq { var, value } => scope.vars.get(var).map_or(false, |v| v == value),
            Condition::Truthy { var } => scope.vars.get(var).map_or(false, is_truthy),
            Condition::Not { inner } => !inner.evaluate(scope),
        }
    }
}

fn is_truthy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_f64().map_or(false, |f| f != 0.0),
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
    }
}

struct StepBudget {
    remaining: usize,
}

impl StepBudget {
    fn new(limit: usize) -> Self {
        Self { remaining: limit }
    }
    fn charge(&mut self, skill_id: &SkillId) -> Result<(), SkillError> {
        if self.remaining == 0 {
            return Err(SkillError::ResourceExhausted {
                skill: skill_id.clone(),
                what: "step_budget",
            });
        }
        self.remaining -= 1;
        Ok(())
    }
}

// Silence dead-code warnings when the `skill-forge` feature is off.
#[allow(dead_code)]
fn _use_capability_error(e: CapabilityError) -> String {
    e.to_string()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct CannedPlanner(String);
    impl SkillPlanner for CannedPlanner {
        fn plan(&self, _p: &str) -> Result<String, String> {
            Ok(self.0.clone())
        }
    }

    struct CannedDispatcher;
    impl SkillToolDispatcher for CannedDispatcher {
        fn dispatch(
            &self,
            tool: &str,
            args: &serde_json::Value,
        ) -> Result<serde_json::Value, String> {
            match tool {
                "echo" => Ok(args.clone()),
                "fail" => Err("boom".into()),
                _ => Err(format!("unknown tool {tool}")),
            }
        }
    }

    fn exec() -> DeclarativeExecutor<CannedPlanner, CannedDispatcher> {
        DeclarativeExecutor::new(CannedPlanner("planned-output".into()), CannedDispatcher)
    }

    #[test]
    fn plan_step_runs_planner() {
        let e = exec();
        let steps = vec![
            SkillStep {
                kind: StepKind::Plan { prompt: "p".into() },
                bind: Some("out".into()),
            },
            SkillStep {
                kind: StepKind::Return { var: "out".into() },
                bind: None,
            },
        ];
        let out = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::empty(),
                &CapabilitySet::empty(),
            )
            .expect("ok");
        assert_eq!(out.value, json!("planned-output"));
    }

    #[test]
    fn tool_call_without_capability_denied() {
        let e = exec();
        let steps = vec![SkillStep {
            kind: StepKind::ToolCall {
                tool: "echo".into(),
                args: json!({"x": 1}),
            },
            bind: None,
        }];
        let err = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::empty(),
                &CapabilitySet::empty(),
            )
            .unwrap_err();
        match err {
            SkillError::CapabilityDenied { .. } => {}
            other => panic!("expected CapabilityDenied, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_with_capability_succeeds() {
        let e = exec();
        let steps = vec![
            SkillStep {
                kind: StepKind::ToolCall {
                    tool: "echo".into(),
                    args: json!({"x": 1}),
                },
                bind: Some("r".into()),
            },
            SkillStep {
                kind: StepKind::Return { var: "r".into() },
                bind: None,
            },
        ];
        let caps = CapabilitySet::empty().with(Capability::ToolCall("echo".into()));
        let out = e
            .execute(&SkillId::new("t"), &steps, &SkillInputs::empty(), &caps)
            .expect("ok");
        assert_eq!(out.value, json!({"x": 1}));
    }

    #[test]
    fn branch_takes_correct_arm() {
        let e = exec();
        let steps = vec![
            SkillStep {
                kind: StepKind::Transform {
                    op: TransformOp::Literal { value: json!(true) },
                },
                bind: Some("flag".into()),
            },
            SkillStep {
                kind: StepKind::Branch {
                    cond: Condition::Truthy { var: "flag".into() },
                    then: vec![SkillStep {
                        kind: StepKind::Transform {
                            op: TransformOp::Literal { value: json!("A") },
                        },
                        bind: Some("result".into()),
                    }],
                    else_: vec![SkillStep {
                        kind: StepKind::Transform {
                            op: TransformOp::Literal { value: json!("B") },
                        },
                        bind: Some("result".into()),
                    }],
                },
                bind: None,
            },
            SkillStep {
                kind: StepKind::Return {
                    var: "result".into(),
                },
                bind: None,
            },
        ];
        let out = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::empty(),
                &CapabilitySet::empty(),
            )
            .expect("ok");
        assert_eq!(out.value, json!("A"));
    }

    #[test]
    fn transform_pick_extracts_field() {
        let e = exec();
        let steps = vec![
            SkillStep {
                kind: StepKind::Transform {
                    op: TransformOp::Literal {
                        value: json!({"a": 1, "b": 2}),
                    },
                },
                bind: Some("obj".into()),
            },
            SkillStep {
                kind: StepKind::Transform {
                    op: TransformOp::Pick {
                        from: "obj".into(),
                        field: "b".into(),
                    },
                },
                bind: Some("picked".into()),
            },
            SkillStep {
                kind: StepKind::Return {
                    var: "picked".into(),
                },
                bind: None,
            },
        ];
        let out = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::empty(),
                &CapabilitySet::empty(),
            )
            .expect("ok");
        assert_eq!(out.value, json!(2));
    }

    #[test]
    fn step_budget_exhaustion_errors() {
        let mut e = exec();
        e.max_steps = 2;
        let mut steps = Vec::new();
        for _ in 0..5 {
            steps.push(SkillStep {
                kind: StepKind::Transform {
                    op: TransformOp::Literal { value: json!(1) },
                },
                bind: Some("x".into()),
            });
        }
        let err = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::empty(),
                &CapabilitySet::empty(),
            )
            .unwrap_err();
        match err {
            SkillError::ResourceExhausted {
                what: "step_budget",
                ..
            } => {}
            other => panic!("expected step_budget exhaustion, got {other:?}"),
        }
    }

    #[test]
    fn interpolation_substitutes_vars() {
        let mut inputs = serde_json::Map::new();
        inputs.insert("name".into(), json!("world"));
        let e = exec();
        let steps = vec![
            SkillStep {
                kind: StepKind::Transform {
                    op: TransformOp::ConcatStrings {
                        parts: vec!["name".into()],
                    },
                },
                bind: Some("greeting".into()),
            },
            SkillStep {
                kind: StepKind::Return {
                    var: "greeting".into(),
                },
                bind: None,
            },
        ];
        let out = e
            .execute(
                &SkillId::new("t"),
                &steps,
                &SkillInputs::new(serde_json::Value::Object(inputs)),
                &CapabilitySet::empty(),
            )
            .expect("ok");
        assert_eq!(out.value, json!("world"));
    }

    #[test]
    fn tool_failure_propagates_as_execution_failed() {
        let e = exec();
        let steps = vec![SkillStep {
            kind: StepKind::ToolCall {
                tool: "fail".into(),
                args: json!(null),
            },
            bind: None,
        }];
        let caps = CapabilitySet::empty().with(Capability::ToolCall("fail".into()));
        let err = e
            .execute(&SkillId::new("t"), &steps, &SkillInputs::empty(), &caps)
            .unwrap_err();
        match err {
            SkillError::ExecutionFailed { .. } => {}
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
    }
}
