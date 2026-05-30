# Recipe Chatbot - AI Evaluations Course

This repository contains a complete AI evaluations course built around a Recipe Chatbot. Through 5 progressive homework assignments, you'll learn practical techniques for evaluating and improving AI systems.

## Quick Start

1. **Clone & Setup**
   ```bash
   git clone https://github.com/ai-evals-course/recipe-chatbot.git
   cd recipe-chatbot
   uv sync
   ```

2. **Configure Environment**
   ```bash
   cp env.example .env
   # Edit .env to add your model and API keys
   ```
   It uses Github Copilot chat model by default. You can change the model in .env if you want Ollama / Anthropic / etc. Inference is done using `litellm`.

3. **Run the Chatbot**
   ```bash
   uv run uvicorn backend.main:app --reload --reload-include '*.md'
   # Open http://127.0.0.1:8000
   ```

