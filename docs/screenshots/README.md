# Screenshots

This directory holds screenshots for documentation and the website.

## Needed

| File | What to capture | How |
|------|----------------|-----|
| `cli_startup.png` | CLI banner + provider scan output | Run `ai_assistant_cli.exe`, screenshot the startup |
| `cli_chat.png` | A short conversation in the CLI | Type a question, wait for response, screenshot |
| `gui_main.png` | GUI main window with a chat | Run `ai_gui.exe`, scan models, send a message |
| `server_curl.png` | Server startup + curl interaction | Run server, use `curl` in another terminal |

## How to capture

1. Start Ollama with a small model:
   ```
   ollama pull llama3.2:1b
   ```
2. Run the binary and take screenshots (Win+Shift+S or Snipping Tool)
3. Save as PNG in this directory
4. Use the exact filenames from the table above
