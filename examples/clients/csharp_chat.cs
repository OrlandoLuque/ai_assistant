// Interactive chat client for ai_assistant_server — C# (.NET 6+)
//
// Features demonstrated:
//   - Basic chat (POST /chat)
//   - System prompts & temperature
//   - Multi-turn conversation
//   - Model listing
//
// Usage:
//   dotnet script csharp_chat.cs
//   -- or create a console project and paste this as Program.cs --
//   dotnet new console -n AiChat && cd AiChat
//   # replace Program.cs with this file
//   dotnet run

using System.Net.Http.Json;
using System.Text.Json;

var baseUrl = args.Length > 0 ? args[0] : "http://localhost:8090";
var client = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromSeconds(120) };

// Health check
try
{
    var health = await client.GetAsync("/health");
    if (!health.IsSuccessStatusCode)
    {
        Console.Error.WriteLine($"Server returned {health.StatusCode}");
        return 1;
    }
}
catch (HttpRequestException)
{
    Console.Error.WriteLine($"Cannot connect to {baseUrl}");
    Console.Error.WriteLine("Start it with: ai_assistant_server.exe --port 8090");
    return 1;
}

Console.WriteLine($"Connected to {baseUrl}");

// List models
var modelsResp = await client.GetStringAsync("/models");
Console.WriteLine($"\nAvailable models:\n{modelsResp}\n");

// Conversation loop
var messages = new List<object>();

// Optional system prompt
Console.Write("System prompt (Enter to skip): ");
var systemPrompt = Console.ReadLine()?.Trim();
if (!string.IsNullOrEmpty(systemPrompt))
    messages.Add(new { role = "system", content = systemPrompt });

Console.WriteLine("Type your message (Ctrl+C to quit)\n");

while (true)
{
    Console.Write("You> ");
    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(input)) continue;
    if (input is "/quit" or "/exit") break;
    if (input == "/models")
    {
        Console.WriteLine(await client.GetStringAsync("/models"));
        continue;
    }

    messages.Add(new { role = "user", content = input });

    var body = new { messages, temperature = 0.7 };
    var response = await client.PostAsJsonAsync("/chat", body);
    var json = await response.Content.ReadFromJsonAsync<JsonElement>();

    string reply;
    if (json.TryGetProperty("response", out var r))
        reply = r.GetString() ?? "";
    else if (json.TryGetProperty("content", out var c))
        reply = c.GetString() ?? "";
    else
        reply = json.GetRawText();

    Console.WriteLine($"Assistant> {reply}\n");
    messages.Add(new { role = "assistant", content = reply });
}

Console.WriteLine("Bye!");
return 0;
