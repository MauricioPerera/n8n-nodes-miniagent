# n8n-nodes-miniagent

Lightweight AI Agent node for n8n - **zero dependencies**, built-in memory, multi-LLM support.

## Features

- **Zero Dependencies**: No LangChain or external SDKs - just pure TypeScript with native `fetch`
- **Multi-LLM Support**: Works with Gemini, Claude (Anthropic), and any OpenAI-compatible API
- **Built-in Memory**: Conversation history that persists across executions
- **Tool Calls Saved**: Unlike n8n's AI Agent, this saves tool calls in memory (fixes issue #14361)
- **ReAct Pattern**: Implements Reasoning + Acting for intelligent task completion
- **Fully Serverless**: No external servers or databases required
- **n8n Cloud Ready**: Designed to pass n8n Cloud approval

## Installation

### In n8n Cloud
Search for "Mini Agent" in the community nodes section.

### Self-hosted
```bash
npm install n8n-nodes-miniagent
```

Or install via n8n Settings > Community Nodes.

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Gemini** | gemini-pro, gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash | Google AI Studio API |
| **Anthropic** | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet | Claude API |
| **OpenAI Compatible** | gpt-4, gpt-4o, gpt-3.5-turbo, llama, mistral, etc. | Works with OpenAI, OpenRouter, Groq, Ollama, LM Studio |

## Operations

### Chat
Send a message and get a response. No memory - each call is independent.

### Chat with Memory
Chat with conversation history preserved. Great for multi-turn conversations.

### Clear Memory
Clear the conversation history for a specific session.

### Get Memory
Retrieve the current conversation history for debugging.

## Tools

Tools allow the agent to perform actions. Define them as a JSON array:

### Code Tool Example
```json
[
  {
    "name": "calculate",
    "description": "Evaluate a mathematical expression",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The math expression to evaluate"
        }
      },
      "required": ["expression"]
    },
    "code": "return eval(expression)"
  }
]
```

### HTTP Tool Example
```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name"
        }
      },
      "required": ["city"]
    },
    "http": {
      "url": "https://api.weather.example/current",
      "method": "GET",
      "queryParams": {
        "q": "{{city}}"
      }
    }
  }
]
```

## Memory Types

### Buffer (Volatile)
- Stored in memory
- Fast access
- Lost when n8n restarts
- Good for: Testing, short-lived sessions

### Workflow Static Data (Persistent)
- Stored in n8n's workflow data
- Survives n8n restarts
- Good for: Production use, important conversations

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Temperature | 0.7 | Controls randomness (0-2) |
| Max Tokens | 4096 | Maximum response length |
| Max Iterations | 10 | Maximum tool-use loops |
| Max Memory Messages | 50 | Messages to keep in history |
| Include Tool Calls | true | Save tool calls in memory |

## Why Mini Agent?

### Problems with n8n's AI Agent (LangChain-based):
1. **Tool calls not saved in memory** - Agent stops using tools after a few turns
2. **Heavy dependencies** - LangChain adds complexity and version conflicts
3. **Memory requires external nodes** - No built-in persistent storage
4. **Difficult to customize** - Tied to LangChain's abstractions

### Mini Agent solves these:
1. **All messages saved** - Including tool calls and results
2. **Zero dependencies** - Just TypeScript and fetch
3. **Built-in memory** - Buffer and persistent storage included
4. **Simple architecture** - Easy to understand and extend

## Example Workflow

```
[Webhook] → [Mini Agent: Chat with Memory] → [Respond to Webhook]
```

The agent will:
1. Load conversation history for the session
2. Process the user's message
3. Use tools if needed (with proper memory of tool usage)
4. Save the updated conversation
5. Return the response

## License

MIT

## Author

Mauricio Perera (mauricio.perera@gmail.com)

## Links

- [GitHub Repository](https://github.com/MauricioPerera/n8n-nodes-miniagent)
- [Report Issues](https://github.com/MauricioPerera/n8n-nodes-miniagent/issues)
- [n8n Community Nodes](https://docs.n8n.io/integrations/community-nodes/)
