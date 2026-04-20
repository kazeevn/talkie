# Talkie

Inference library for the Talkie 13B language model family.

Talkie models are 13-billion-parameter decoder-only transformers trained on
large-scale English text corpora. This package provides a simple Python API
and CLI to download models from HuggingFace and run inference locally.

## Models

| Name | HuggingFace | Style | Description |
|------|-------------|-------|-------------|
| `talkie-1930-13b-base` | [talkie-lm/talkie-1930-13b-base](https://huggingface.co/talkie-lm/talkie-1930-13b-base) | Base | Vintage-era base language model |
| `talkie-1930-13b-it` | [talkie-lm/talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it) | IT | Vintage instruction-tuned (chat) |
| `talkie-web-13b-base` | [talkie-lm/talkie-web-13b-base](https://huggingface.co/talkie-lm/talkie-web-13b-base) | Base | Modern web base language model |

## Installation

```bash
pip install talkie
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add talkie
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.1
- CUDA GPU with >= 28 GB VRAM (bfloat16 inference)
- ~26-50 GB disk space per model

## Quick Start

### Python API

```python
from talkie import Talkie

# Load a base model (downloads from HuggingFace on first use)
model = Talkie("talkie-1930-13b-base")

# Generate a completion
result = model.generate("The year was 1929, and", temperature=0.8, max_tokens=200)
print(result.text)

# Stream tokens
for token in model.stream("In the beginning there was"):
    print(token, end="", flush=True)
```

### Chat (instruction-tuned model)

```python
from talkie import Talkie, Message

model = Talkie("talkie-1930-13b-it")

# Single-turn
result = model.generate("Tell me about the 1920s", max_tokens=300)
print(result.text)

# Multi-turn chat
messages = [
    Message(role="user", content="Tell me about the 1920s"),
]
result = model.chat(messages, temperature=0.7)
print(result.text)

# Stream a chat reply
messages.append(Message(role="assistant", content=result.text))
messages.append(Message(role="user", content="What about the music?"))
for token in model.chat_stream(messages):
    print(token, end="", flush=True)
```

### Pre-download models

```python
from talkie import download_model

# Download before loading (useful for setup scripts)
download_model("talkie-1930-13b-base")
```

## CLI

```bash
# Generate text
talkie generate "Once upon a time" --model talkie-1930-13b-base -t 0.8

# Interactive chat
talkie chat --model talkie-1930-13b-it

# Download a model
talkie download talkie-1930-13b-base

# Download all models
talkie download all

# List available models
talkie list
```

## Architecture

All three models share the same 13B decoder-only transformer architecture:

| Parameter | Value |
|-----------|-------|
| Layers | 40 |
| Attention heads | 40 |
| Embedding dimension | 5120 |
| Head dimension | 128 |
| MLP intermediate | 13696 (SwiGLU) |
| Vocab size | 65,536 (base) / 65,540 (IT) |
| Positional encoding | RoPE (base freq 1M) |
| Normalisation | RMS Norm |
| Precision | bfloat16 |

Notable architectural features:
- **QK RMS normalisation** with learnable per-head gain
- **Embedding skip connections** in each transformer block
- **Per-layer gain scaling** initialised to $(2L)^{-0.5}$
- **Gumbel-max sampling** with temperature, top-k, and top-p (nucleus) filtering

## License

Apache 2.0
