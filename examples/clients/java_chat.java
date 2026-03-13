///usr/bin/env jbang "$0" "$@" ; exit $?
// Interactive chat client for ai_assistant_server — Java 11+
//
// Features demonstrated:
//   - Basic chat (POST /chat)
//   - System prompts & temperature
//   - Multi-turn conversation
//
// Usage (with jbang):
//   jbang java_chat.java
//
// Usage (plain Java 11+):
//   java java_chat.java
//   java java_chat.java http://localhost:8090

import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

public class java_chat {

    static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    static String BASE;

    public static void main(String[] args) throws Exception {
        BASE = args.length > 0 ? args[0] : "http://localhost:8090";

        // Health check
        try {
            var health = HTTP.send(
                    HttpRequest.newBuilder(URI.create(BASE + "/health")).GET().build(),
                    HttpResponse.BodyHandlers.ofString());
            if (health.statusCode() != 200) {
                System.err.println("Server returned " + health.statusCode());
                System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Cannot connect to " + BASE);
            System.err.println("Start it with: ai_assistant_server.exe --port 8090");
            System.exit(1);
        }

        System.out.println("Connected to " + BASE);

        // List models
        var modelsResp = HTTP.send(
                HttpRequest.newBuilder(URI.create(BASE + "/models")).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        System.out.println("\nAvailable models:\n" + modelsResp.body() + "\n");

        // Conversation loop
        List<String> messageJsons = new ArrayList<>();
        var scanner = new Scanner(System.in);

        System.out.print("System prompt (Enter to skip): ");
        String systemPrompt = scanner.nextLine().trim();
        if (!systemPrompt.isEmpty()) {
            messageJsons.add(messageJson("system", systemPrompt));
        }

        System.out.println("Type your message (Ctrl+C to quit)\n");

        while (true) {
            System.out.print("You> ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine().trim();
            if (input.isEmpty()) continue;
            if (input.equals("/quit") || input.equals("/exit")) break;

            messageJsons.add(messageJson("user", input));

            String body = "{\"messages\":[" + String.join(",", messageJsons) + "],\"temperature\":0.7}";

            var response = HTTP.send(
                    HttpRequest.newBuilder(URI.create(BASE + "/chat"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .timeout(Duration.ofSeconds(120))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());

            String respBody = response.body();
            // Simple JSON parsing for "response" field
            String reply = extractField(respBody, "response");
            if (reply == null) reply = extractField(respBody, "content");
            if (reply == null) reply = respBody;

            System.out.println("Assistant> " + reply + "\n");
            messageJsons.add(messageJson("assistant", reply));
        }
        System.out.println("Bye!");
    }

    static String messageJson(String role, String content) {
        return "{\"role\":\"" + escapeJson(role) + "\",\"content\":\"" + escapeJson(content) + "\"}";
    }

    static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    static String extractField(String json, String field) {
        String key = "\"" + field + "\":\"";
        int start = json.indexOf(key);
        if (start < 0) return null;
        start += key.length();
        int end = start;
        while (end < json.length()) {
            if (json.charAt(end) == '"' && json.charAt(end - 1) != '\\') break;
            end++;
        }
        return json.substring(start, end).replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }
}
