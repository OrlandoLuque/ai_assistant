//! Constrained decoding demo.
//!
//! Run with: cargo run --example constrained_demo --features constrained-decoding
//!
//! Demonstrates grammar-guided generation: GBNF grammars,
//! JSON Schema constraints, and grammar builders.

use ai_assistant::{
    Grammar, GrammarRule, GrammarAlternative, GrammarElement, SchemaToGrammar,
    GrammarBuilder,
};

fn main() {
    println!("=== Constrained Decoding Demo ===\n");

    // 1. Build a grammar manually
    let mut grammar = Grammar::new("root");
    grammar.rules.push(GrammarRule {
        name: "root".to_string(),
        alternatives: vec![GrammarAlternative {
            elements: vec![
                GrammarElement::Literal("{".to_string()),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::RuleRef("pair".to_string()),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Literal("}".to_string()),
            ],
        }],
    });
    grammar.rules.push(GrammarRule {
        name: "pair".to_string(),
        alternatives: vec![GrammarAlternative {
            elements: vec![
                GrammarElement::RuleRef("string".to_string()),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Literal(":".to_string()),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::RuleRef("value".to_string()),
            ],
        }],
    });
    grammar.rules.push(GrammarRule {
        name: "string".to_string(),
        alternatives: vec![GrammarAlternative {
            elements: vec![GrammarElement::Literal("\"text\"".to_string())],
        }],
    });
    grammar.rules.push(GrammarRule {
        name: "value".to_string(),
        alternatives: vec![
            GrammarAlternative {
                elements: vec![GrammarElement::Literal("true".to_string())],
            },
            GrammarAlternative {
                elements: vec![GrammarElement::Literal("false".to_string())],
            },
        ],
    });
    grammar.rules.push(GrammarRule {
        name: "ws".to_string(),
        alternatives: vec![GrammarAlternative {
            elements: vec![GrammarElement::CharSet(vec![' ', '\t', '\n'])],
        }],
    });

    println!("Manual Grammar: root={}", grammar.root_rule);
    println!("Rules: {}", grammar.rules.len());

    // 2. Export to GBNF format
    let gbnf = grammar.to_gbnf();
    println!("\nGBNF output:");
    println!("{}", &gbnf[..gbnf.len().min(300)]);

    // 3. Compile from JSON Schema
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"}
        },
        "required": ["name", "age"]
    });

    match SchemaToGrammar::compile(&schema) {
        Ok(compiled) => {
            println!("\nJSON Schema -> Grammar:");
            println!("  Root rule: {}", compiled.root_rule);
            println!("  Rules: {}", compiled.rules.len());
            let compiled_gbnf = compiled.to_gbnf();
            println!("  GBNF (first 200 chars): {}", &compiled_gbnf[..compiled_gbnf.len().min(200)]);
        }
        Err(e) => println!("Schema compilation error: {}", e),
    }

    // 4. GrammarBuilder for fluent construction
    let mut builder = GrammarBuilder::new("root");
    builder.add_rule(GrammarRule::new("root"));
    println!("\nGrammarBuilder: root rule added");

    println!("\n=== Done ===");
}
