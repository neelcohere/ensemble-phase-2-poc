# Inference Module

The inference module manages LLM instantiation and provider abstraction. It provides a factory pattern for creating and configuring chat models with consistent retry/backoff logic.

## Architecture

### Files

- **`router.py`** – `ChatFactory` class that provides a unified interface for creating chat models across multiple providers
- **`cohere.py`** – `CustomChatCohere` wrapper that adds retry/backoff logic to ChatCohere

## Components

### ChatFactory

Factory class for instantiating chat models. Centralizes provider selection and configuration.

**Usage:**
```python
from ensemble_phase_2_poc.inference.router import ChatFactory

model = ChatFactory.get_model("cohere", "command-a-03-2025")
```

**Supported Providers:**
- `"cohere"` → Returns `CustomChatCohere` instance
- `"openai"` → Returns `ChatOpenAI` instance

**Parameters:**
- `provider` (str): The model provider ("cohere" or "openai")
- `model` (str): The model name/ID
- `**kwargs`: Additional provider-specific arguments

**Returns:** `BaseChatModel` subclass

### CustomChatCohere

Extends `langchain_cohere.ChatCohere` with automatic retry and exponential backoff logic.

**Features:**
- Retries failed requests up to 5 times
- Uses exponential backoff with jitter to prevent too many requests from being sent
- Wraps both `invoke()` (sync) and `ainvoke()` (async) methods

**Usage:**
```python
from ensemble_phase_2_poc.inference.cohere import CustomChatCohere

llm = CustomChatCohere(
    cohere_api_key="...",
    model="command-a-03-2025"
)
```

## Integration

Agents use `ChatFactory.get_model()` via `BaseAgent.build_agent()` to obtain a chat model:

```python
# In BaseAgent.build_agent()
model = ChatFactory.get_model("cohere", model_name)
```

This ensures all agents automatically benefit from retry/backoff logic without explicit configuration.

## Environment Variables

- `COHERE_API_KEY` – Required for Cohere provider
- `OPENAI_API_KEY` – Required for OpenAI provider

Set in `.env` file at project root.

## Adding New Providers or Models

When adding new providers or models, remember to update the pricing tables in `router.py`:

- `COHERE_MODEL_PRICING` – Input/output token costs per 1M tokens for Cohere models
- `OPENAI_MODEL_PRICING` – Input/output token costs per 1M tokens for OpenAI models

These tables are used by the `token_cost` scorer to calculate evaluation costs.
