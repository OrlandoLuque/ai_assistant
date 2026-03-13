#!/usr/bin/env node
/**
 * Interactive chat client for ai_assistant_server — Node.js 18+
 *
 * Features demonstrated:
 *   - Basic chat (POST /chat)
 *   - SSE streaming (POST /chat/stream)
 *   - System prompts & temperature control
 *   - Model listing
 *
 * No dependencies required — uses built-in fetch and readline.
 *
 * Usage:
 *   # Start the server first:
 *   #   ai_assistant_server.exe --port 8090
 *   #
 *   node node_chat.mjs
 *   node node_chat.mjs --url http://localhost:8090 --stream
 *   node node_chat.mjs --model llama3.2:1b --system "You are a pirate"
 */

import { createInterface } from "node:readline";

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const opts = {
  url: "http://localhost:8090",
  model: null,
  system: null,
  temperature: 0.7,
  stream: false,
};

for (let i = 0; i < args.length; i++) {
  switch (args[i]) {
    case "--url":       opts.url = args[++i]; break;
    case "--model":     opts.model = args[++i]; break;
    case "--system":    opts.system = args[++i]; break;
    case "--temperature": opts.temperature = parseFloat(args[++i]); break;
    case "--stream":    opts.stream = true; break;
    case "--help":
      console.log("Usage: node node_chat.mjs [--url URL] [--model NAME] [--system PROMPT] [--temperature N] [--stream]");
      process.exit(0);
  }
}

const BASE = opts.url.replace(/\/+$/, "");

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function checkHealth() {
  try {
    const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return r.ok;
  } catch {
    return false;
  }
}

async function listModels() {
  const r = await fetch(`${BASE}/models`);
  const data = await r.json();
  const models = Array.isArray(data) ? data : data.models || [];
  console.log("\nAvailable models:");
  for (const m of models) {
    const name = typeof m === "string" ? m : m.name || m.id || JSON.stringify(m);
    console.log(`  - ${name}`);
  }
  console.log();
}

async function chatSimple(messages) {
  const body = { messages, temperature: opts.temperature };
  if (opts.model) body.model = opts.model;

  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  const data = await r.json();
  return data.response || data.content || JSON.stringify(data);
}

async function chatStream(messages) {
  const body = { messages, temperature: opts.temperature };
  if (opts.model) body.model = opts.model;

  const r = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);

  const chunks = [];
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") break;
      try {
        const chunk = JSON.parse(payload);
        const token = chunk.token || chunk.content || "";
        if (token) {
          process.stdout.write(token);
          chunks.push(token);
        }
      } catch { /* skip non-JSON lines */ }
    }
  }
  console.log();
  return chunks.join("");
}

// ---------------------------------------------------------------------------
// Main REPL
// ---------------------------------------------------------------------------

async function main() {
  if (!(await checkHealth())) {
    console.error(`Cannot connect to server at ${BASE}`);
    console.error("Start it with:  ai_assistant_server.exe --port 8090");
    process.exit(1);
  }

  console.log(`Connected to ${BASE}`);
  await listModels();

  const messages = [];
  if (opts.system) {
    messages.push({ role: "system", content: opts.system });
    console.log(`System prompt: ${opts.system}\n`);
  }

  const mode = opts.stream ? "streaming" : "sync";
  console.log(`Mode: ${mode} | Temperature: ${opts.temperature}`);
  if (opts.model) console.log(`Model: ${opts.model}`);
  console.log("Type your message (Ctrl+C to quit)\n");

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  const prompt = () => {
    rl.question("You> ", async (input) => {
      const text = input.trim();
      if (!text) return prompt();
      if (["/quit", "/exit", "exit", "quit"].includes(text.toLowerCase())) {
        rl.close();
        return;
      }
      if (text.toLowerCase() === "/models") {
        await listModels();
        return prompt();
      }

      messages.push({ role: "user", content: text });
      process.stdout.write("Assistant> ");

      try {
        const chatFn = opts.stream ? chatStream : chatSimple;
        const response = await chatFn(messages);
        if (!opts.stream) console.log(response);
        messages.push({ role: "assistant", content: response });
      } catch (err) {
        console.error(`\nError: ${err.message}`);
      }
      console.log();
      prompt();
    });
  };

  prompt();
}

main();
