# Recipe Chatbot Setup Guide

This guide explains how to configure the Recipe Chatbot to work with different LLM providers.

## Quick Start (No API Key Required)

The chatbot is configured to use **Ollama** by default, which runs models locally on your machine without requiring API keys.

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 2. Download a Model

```bash
# Small, fast model (2GB) - recommended for testing
ollama pull llama3.2:3b

# Or larger, more capable model (4.7GB)
ollama pull llama3.2

# Other good options
ollama pull qwen2.5:3b
ollama pull phi3
```

### 3. Update .env

The default `.env` is already configured for Ollama:

```bash
MODEL_NAME=ollama/llama3.2:3b
MODEL_NAME_JUDGE=ollama/llama3.2:3b
```

### 4. Run the Chatbot

```bash
uv sync
uv run uvicorn backend.main:app --reload
```

Visit http://127.0.0.1:8000

---

## Alternative Configurations

### Option 1: OpenAI (Requires API Key)

**Best for**: Production use, highest quality responses

1. Get an API key from https://platform.openai.com
2. Update `.env`:

```bash
MODEL_NAME=openai/gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

### Option 2: Anthropic Claude (Requires API Key)

**Best for**: Long conversations, complex reasoning

1. Get an API key from https://console.anthropic.com
2. Update `.env`:

```bash
MODEL_NAME=anthropic/claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your-key-here
```

### Option 3: llama.cpp (Local, No API Key)

**Best for**: Maximum control, custom models

1. Install and run llama.cpp server:

```bash
./llama-server -m path/to/model.gguf --port 8080
```

2. Update `.env`:

```bash
MODEL_NAME=openai/local-model
OPENAI_API_BASE=http://localhost:8080/v1
```

### Option 4: Other Ollama Models

```bash
# Fast and multilingual
MODEL_NAME=ollama/qwen2.5:7b

# Good for code
MODEL_NAME=ollama/codellama

# Efficient and capable
MODEL_NAME=ollama/mistral
```

---

## Recommended Models by Use Case

| Use Case | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| **Quick Testing** | `ollama/llama3.2:3b` | 2GB | ⚡⚡⚡ | ⭐⭐ |
| **Local Development** | `ollama/llama3.2` | 4.7GB | ⚡⚡ | ⭐⭐⭐ |
| **Production (Cloud)** | `openai/gpt-4o-mini` | N/A | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **Best Quality** | `anthropic/claude-3-5-sonnet` | N/A | ⚡⚡ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting

### "AuthenticationError: The api_key client option must be set"

**Solution**: You're using a cloud provider without an API key. Either:
- Switch to Ollama (no key needed)
- Or add the required API key to `.env`

### Ollama connection errors

**Check if Ollama is running**:
```bash
curl http://localhost:11434/api/tags
```

**Start Ollama if needed**:
```bash
ollama serve
```

### Model not found

**List available models**:
```bash
ollama list
```

**Pull the model**:
```bash
ollama pull llama3.2:3b
```

### Slow responses with Ollama

- Use a smaller model (`llama3.2:3b` instead of `llama3.2`)
- Ensure Ollama is using GPU acceleration
- Consider using a cloud provider for faster responses

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_NAME` | Model for chat responses | `ollama/llama3.2:3b` |
| `MODEL_NAME_JUDGE` | Model for evaluations | `ollama/llama3.2:3b` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `OPENAI_API_BASE` | Custom API endpoint | `http://localhost:8080/v1` |

---

## Next Steps

1. ✅ Configure your model provider
2. ✅ Start the server
3. ✅ Test the chat at http://127.0.0.1:8000
4. ✅ Check traces in the "Traces" tab
5. 📖 See `README.md` for homework walkthroughs
