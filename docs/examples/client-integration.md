# Client Integration

Learn how to integrate SmarterRouter with various AI applications and frameworks.

## Table of Contents
- [Python OpenAI SDK](#python-openai-sdk)
- [JavaScript / TypeScript (Node.js)](#javascript--typescript-nodejs)
- [OpenCode IDE Integration](#opencode-ide-integration)
- [curl](#curl)
- [OpenWebUI](#openwebui)
- [VS Code Extensions (Continue, Cursor)](#vs-code-extensions)
- [SillyTavern & Other Chat UIs](#sillytavern--other-chat-uis)
- [Anthropic SDK](#anthropic-sdk)

SmarterRouter is compatible with any OpenAI-compatible client. See the [API Reference](../api.md) for full details.


---

## Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11436/v1",
    api_key="dummy-key"  # SmarterRouter doesn't require a key, but some clients need it
)

response = client.chat.completions.create(
    model="smarterrouter/main",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"Model used: {response.model}")  # Always 'smarterrouter/main'
# Check response signature or use /admin/explain to see actual model
```

---

## JavaScript / TypeScript (Node.js)

### Using the Official OpenAI SDK
```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:11436/v1",
  apiKey: "dummy-key", // Not required by SmarterRouter but expected by the SDK
});

const completion = await openai.chat.completions.create({
  model: "smarterrouter/main",
  messages: [{ role: "user", content: "Explain merge sort in two sentences." }],
});

console.log(completion.choices[0].message.content);
```

### Using Vercel AI SDK (`@ai-sdk/openai-compatible`)
```typescript
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText } from "ai";

const smarterrouter = createOpenAICompatible({
  name: "smarterrouter",
  baseURL: "http://localhost:11436/v1",
  apiKey: "dummy",
});

const { text } = await generateText({
  model: smarterrouter("smarterrouter/main"),
  prompt: "What is the capital of France?",
});

console.log(text);
```

---

## OpenCode IDE Integration

SmarterRouter natively supports OpenCode IDE via [opencode.json](../../opencode.json):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "smarterrouter": {
      "name": "SmarterRouter",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:11436/v1",
        "apiKey": "admin123"
      },
      "models": {
        "smarterrouter/main": {
          "name": "SmarterRouter (VRAM Adaptive & Compressed)"
        },
        "qwen2.5:3b": {
          "name": "Qwen 2.5 3B"
        }
      }
    }
  }
}
```

---


## curl

### Basic Chat Completion

```bash
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Generate Embeddings

```bash
curl -X POST http://localhost:11436/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

---

## OpenWebUI

See detailed guide: [OpenWebUI Integration](./openwebui-integration.md)

**Quick setup:**
1. OpenWebUI → Settings → Connections → Add Connection
2. Name: `SmarterRouter`
3. Base URL: `http://localhost:11436/v1`
4. API Key: (leave empty)
5. Model: `smarterrouter/main`
6. Save

---

## VS Code Extensions

### Continue

Configure `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "SmarterRouter",
      "model": "smarterrouter/main",
      "apiBase": "http://localhost:11436/v1",
      "apiKey": "dummy"
    }
  ]
}
```

### Cursor

In Cursor settings (`settings.json`):

```json
{
  "cursor.router": {
    "baseUrl": "http://localhost:11436/v1",
    "apiKey": "dummy"
  }
}
```

---

## SillyTavern & Other Chat UIs

Most chat UIs (SillyTavern, JanitorAI, etc.) support custom OpenAI endpoints.

### Configuration

1. Open API settings in your chat UI
2. Select "Custom" or "OpenAI" API mode
3. Configure:
   - **API URL:** `http://localhost:11436/v1`
   - **API Key:** (leave empty or any string)
   - **Model:** `smarterrouter/main`
4. Save and start chatting

---

## Anthropic SDK

SmarterRouter provides OpenAI-compatible endpoints. For Anthropic SDK compatibility, use the OpenAI base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11436/v1",
    api_key="dummy"
)

# Use Anthropic-compatible format if needed
# Note: SmarterRouter expects OpenAI format messages
response = client.chat.completions.create(
    model="smarterrouter/main",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

For true Anthropic API format support, you may need a compatibility layer. See [Backends Documentation](../backends.md) for details.

---

## Node.js / JavaScript

Using the official OpenAI SDK:

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:11436/v1',
  apiKey: 'dummy-key' // Not required but some clients need it
});

const completion = await openai.chat.completions.create({
  model: 'smarterrouter/main',
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(completion.choices[0].message.content);
```

---

## Go

```go
package main

import (
    "context"
    "fmt"
    "net/http"

    "github.com/gin-gonic/gin"
    openai "github.com/henomis/going-openai"
)

func main() {
    r := gin.Default()
    
    r.POST("/chat", func(c *gin.Context) {
        client := openai.NewClient(
            "http://localhost:11436/v1",
            openai.WithAPIKey("dummy"),
        )
        
        resp, err := client.ChatCompletion(
            context.Background(),
            openai.ChatCompletionRequest{
                Model: "smarterrouter/main",
                Messages: []openai.ChatCompletionMessage{
                    {Role: "user", Content: "Hello!"},
                },
            },
        )
        
        if err != nil {
            c.JSON(500, gin.H{"error": err.Error()})
            return
        }
        
        c.JSON(200, resp)
    })
    
    r.Run(":8080")
}
```

---

## Troubleshooting Integrations

### "Connection refused"

- Verify SmarterRouter is running: `curl http://localhost:11436/health`
- Check firewall rules and network connectivity
- Ensure correct port (default: 11436)

### "Model not found"

- SmarterRouter only exposes `smarterrouter/main` as the model
- Don't request other model names; the router selects internally

### "Invalid API key"

- SmarterRouter doesn't require an API key for chat endpoints
- Some clients require a non-empty key; use any string (e.g., "dummy", "sk-...")

### Slow responses

- First request to a model may be slow (cold start)
- Check [Performance Tuning](../performance.md) for optimization tips
- Consider pinning a small model: `ROUTER_PINNED_MODEL=phi3:mini`

---

## Next Steps

- [API Reference](../api.md) - Full endpoint documentation
- [Configuration](../configuration.md) - Customize SmarterRouter behavior
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
- [Performance Tuning](../performance.md) - Optimize for your workload
