// Interactive chat client for ai_assistant_server — Go
//
// Features demonstrated:
//   - Basic chat (POST /chat)
//   - Model listing
//   - System prompts & temperature
//   - Multi-turn conversation
//
// Usage:
//   go run go_chat.go
//   go run go_chat.go -url http://localhost:8090 -model llama3.2:1b
//   go run go_chat.go -system "You are a pirate"

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
}

func checkHealth(base string) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(base + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func listModels(base string) {
	resp, err := http.Get(base + "/models")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("\nAvailable models:\n%s\n\n", string(body))
}

func chat(base string, req ChatRequest) (string, error) {
	data, _ := json.Marshal(req)
	resp, err := http.Post(base+"/chat", "application/json", bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if r, ok := result["response"].(string); ok {
		return r, nil
	}
	if c, ok := result["content"].(string); ok {
		return c, nil
	}
	out, _ := json.Marshal(result)
	return string(out), nil
}

func main() {
	base := flag.String("url", "http://localhost:8090", "Server URL")
	model := flag.String("model", "", "Model name")
	system := flag.String("system", "", "System prompt")
	temp := flag.Float64("temperature", 0.7, "Temperature (0.0-2.0)")
	flag.Parse()

	if !checkHealth(*base) {
		fmt.Fprintf(os.Stderr, "Cannot connect to %s\nStart it with: ai_assistant_server.exe --port 8090\n", *base)
		os.Exit(1)
	}

	fmt.Printf("Connected to %s\n", *base)
	listModels(*base)

	messages := []Message{}
	if *system != "" {
		messages = append(messages, Message{Role: "system", Content: *system})
		fmt.Printf("System prompt: %s\n\n", *system)
	}

	fmt.Printf("Mode: sync | Temperature: %.1f\n", *temp)
	if *model != "" {
		fmt.Printf("Model: %s\n", *model)
	}
	fmt.Println("Type your message (Ctrl+C to quit)\n")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You> ")
		if !scanner.Scan() {
			break
		}
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		if text == "/quit" || text == "/exit" {
			break
		}
		if text == "/models" {
			listModels(*base)
			continue
		}

		messages = append(messages, Message{Role: "user", Content: text})

		req := ChatRequest{
			Messages:    messages,
			Model:       *model,
			Temperature: *temp,
		}

		fmt.Print("Assistant> ")
		response, err := chat(*base, req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
			messages = messages[:len(messages)-1] // remove failed message
			continue
		}
		fmt.Println(response)
		fmt.Println()

		messages = append(messages, Message{Role: "assistant", Content: response})
	}
	fmt.Println("\nBye!")
}
