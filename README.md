# Live LLM

Real-time signal injection for Language Models during generation.

## How It Works

Simple text append with KV cache management - **no architecture changes needed**.

```
Generation:  [tok1] → [tok2] → [tok3] → [SIGNAL] → [tok4] → [tok5]
                                          ↑
                              Just append text here,
                              model naturally reads it
```

The model already understands text. We just:
1. Generate tokens normally with KV cache
2. When signal arrives, append it as text tokens
3. Extend the KV cache (only process new signal tokens)
4. Continue generating - model sees the signal and responds

## Supported Models

Any HuggingFace causal LM works:
- **Qwen**: Qwen3-0.6B, Qwen3-4B, Qwen3-8B
- **Gemma**: gemma-3-1b-it, gemma-3-4b-it
- **Llama**, **Mistral**, and any other causal LM

## Installation

```bash
pip install torch transformers
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_stream import LiveStreamGenerator
import threading
import time

# Load any model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Create generator
generator = LiveStreamGenerator(model, tokenizer)

# Post signals from anywhere (thread-safe)
def sensor_system():
    time.sleep(2)
    generator.post_signal("ALERT: Temperature critical!")

threading.Thread(target=sensor_system, daemon=True).start()

# Generate with live signal injection
for token in generator.generate_stream("Describe the system status"):
    print(token, end="", flush=True)
```

## Example Output

```
Prompt: "Describe the weather forecast"
Signal (after 2s): "URGENT: Tornado warning!"

Output: "Today's forecast shows clear skies and mild temperatures...
        [LIVE]: URGENT: Tornado warning!
        ...however, we've just received an emergency alert. 
        A tornado warning has been issued. Seek shelter immediately."
```

## API

### LiveStreamGenerator

```python
generator = LiveStreamGenerator(model, tokenizer, device="cuda")

# Post a signal (thread-safe, can be called from anywhere)
generator.post_signal("Your signal text here", priority=1.0)

# Generate with streaming
for token in generator.generate_stream(prompt, max_tokens=200, temperature=0.7):
    print(token, end="")
```

## How KV Cache Works

Without cache (inefficient):
```
Step 1: Process [1000 tokens] → output
Step 2: Process [1001 tokens] → output  # Re-process everything!
```

With cache (efficient):
```
Step 1: Process [1000 tokens] → cache KV → output
Step 2: Process [1 token] + cached KV → extend cache → output  # Only 1 new token!

Signal arrives:
Step 3: Process [5 signal tokens] + cached KV → extend cache → output  # Only 5 tokens!
```

## License

MIT
