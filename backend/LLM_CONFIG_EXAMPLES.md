# LLM Backend Configuration Examples

The recipe chatbot now uses a centralized LLM backend powered by LiteLLM, which supports multiple LLM providers through a unified interface.

## Supported Providers

LiteLLM supports 100+ LLM providers. Here are the most common ones:

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export MODEL_NAME_JUDGE="gpt-4o-mini"
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export MODEL_NAME="claude-3-5-sonnet-20241022"
export MODEL_NAME_JUDGE="claude-3-5-haiku-20241022"
```

### Azure OpenAI
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
export MODEL_NAME="azure/gpt-4o-mini"
export MODEL_NAME_JUDGE="azure/gpt-4o-mini"
```

### GitHub Copilot

**Getting Started**: GitHub Copilot uses OAuth device flow for authentication. On first use, LiteLLM will:
1. Display a device code and verification URL
2. Prompt you to visit the URL and enter the code
3. Store your credentials locally for future use

**Requirements**: Active GitHub Copilot subscription

```bash
export MODEL_NAME="github_copilot/gpt-4"
export MODEL_NAME_JUDGE="github_copilot/gpt-4"
```

### Google (Gemini)
```bash
export GEMINI_API_KEY="..."
export MODEL_NAME="gemini/gemini-1.5-pro"
export MODEL_NAME_JUDGE="gemini/gemini-1.5-flash"
```

### AWS Bedrock
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"
export MODEL_NAME="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
export MODEL_NAME_JUDGE="bedrock/anthropic.claude-3-5-haiku-20241022-v2:0"
```

### Ollama (Local)
```bash
export MODEL_NAME="ollama/llama3.1"
export MODEL_NAME_JUDGE="ollama/llama3.1"
```

### Groq
```bash
export GROQ_API_KEY="..."
export MODEL_NAME="groq/llama-3.3-70b-versatile"
export MODEL_NAME_JUDGE="groq/llama-3.1-8b-instant"
```

## Temperature Configuration

You can also configure the default temperature:

```bash
export MODEL_TEMPERATURE="0.7"  # Default: 0.7 (range: 0.0-1.0)
```

## Advanced Usage

### Using Different Models for Different Tasks

The backend supports using different models for different purposes:

```python
from backend.llm_backend import LLMBackend

# Create a backend with a powerful model for complex tasks
powerful_backend = LLMBackend(
    default_model="claude-3-5-sonnet-20241022",
    default_temperature=0.7
)

# Create a judge backend with a faster/cheaper model
judge_backend = powerful_backend.create_judge_backend(
    judge_model="claude-3-5-haiku-20241022"
)
```

### Custom Backend Configuration

```python
from backend.llm_backend import LLMBackend, set_default_backend

# Create a custom backend
custom_backend = LLMBackend(
    default_model="gpt-4o-mini",
    default_temperature=0.8,
    max_retries=5,
    retry_delay=1.0
)

# Set it as the default
set_default_backend(custom_backend)
```

### Using the Backend Directly

```python
from backend.llm_backend import get_default_backend

backend = get_default_backend()

# Single completion
response = backend.complete(
    prompt="What is the best temperature for roasting chicken?",
    temperature=0.3
)

# Chat completion
messages = [
    {"role": "user", "content": "How do I make pasta?"},
]
response = backend.chat(messages=messages)

# Override model for specific request
response = backend.complete(
    prompt="Complex reasoning task...",
    model="claude-3-opus-20240229",
    temperature=0.2
)
```

## Model Recommendations

### For Production Use

- **Main Model**: `gpt-4o-mini`, `claude-3-5-sonnet-20241022`, or `github_copilot/gpt-4`
  - Good balance of cost and quality
  - Fast response times
  - GitHub Copilot: included with GitHub Copilot subscription

- **Judge Model**: `gpt-4o-mini`, `claude-3-5-haiku-20241022`, or `github_copilot/gpt-4`
  - Fast and deterministic
  - Cost-effective for evaluations

### For Development/Testing

- **Local**: `ollama/llama3.1` or `ollama/qwen2.5`
  - Free and fast
  - Good for development

- **Fast Cloud**: `groq/llama-3.3-70b-versatile`
  - Very fast inference
  - Good quality

## Troubleshooting

### Rate Limits

The backend includes automatic retry logic with exponential backoff. You can configure:

```python
backend = LLMBackend(
    max_retries=5,      # Number of retries
    retry_delay=1.0     # Base delay in seconds
)
```

### Provider-Specific Parameters

LiteLLM automatically handles provider-specific parameters. If a provider doesn't support a parameter, it's automatically dropped.

### Logging

Enable verbose logging for debugging:

```python
import litellm
litellm.set_verbose = True
```

## Migration from Direct LiteLLM Calls

If you have existing code using `litellm.completion()` directly:

**Before:**
```python
import litellm

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
content = response.choices[0].message.content
```

**After:**
```python
from backend.llm_backend import get_default_backend

backend = get_default_backend()
content = backend.chat(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Main model for chatbot | `gpt-4o-mini` |
| `MODEL_NAME_JUDGE` | Model for evaluations | `gpt-4o-mini` |
| `MODEL_TEMPERATURE` | Default temperature | `0.7` |
| Provider API keys | See examples above | - |

For a complete list of supported providers and models, see: https://docs.litellm.ai/docs/providers
