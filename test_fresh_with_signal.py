"""Test FRESH dual-stream model WITH signal injection (no training checkpoint)."""
import threading
import time

import torch
from dual_stream import DualStreamTransformer
from dual_stream.encoder import LiveStateEncoder, LiveBufferManager
from dual_stream.text_embedder import TrainableTextEmbedder
from dual_stream.inference import LiveInferenceEngine, GenerationConfig
from transformers import AutoTokenizer

print("Loading FRESH dual-stream model...")
model = DualStreamTransformer.from_pretrained_with_surgery(
    model_path="Qwen/Qwen3-0.6B",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# Create signal encoding pipeline
live_encoder = LiveStateEncoder(
    hidden_dim=model.config.hidden_dim,
    input_dim=768,
    num_signal_slots=8,
).to("cuda", dtype=torch.bfloat16)

text_embedder = TrainableTextEmbedder(
    output_dim=768,
    device="cuda",
)
text_embedder.projection = text_embedder.projection.to(dtype=torch.bfloat16)

live_manager = LiveBufferManager(live_encoder, text_embedder)

print("\nGate values:", model.get_layer_gate_info()[0])

# Format prompt
messages = [{"role": "user", "content": "Describe the weather conditions outside."}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False, 
    add_generation_prompt=True,
    enable_thinking=True,  # Let model think
)
input_ids = tokenizer.encode(formatted, return_tensors="pt").to("cuda")

print(f"\nPrompt: Describe the weather conditions outside.")
print("="*60)

# Manually open gates so live signals have visible influence for this demo
for idx in model.config.inject_live_at_layers:
    model.layers[idx].live_cross_attn.gate_param.data.fill_(0.6)

# Generate WITHOUT signal first (baseline)
print("\n--- Without Signal (baseline) ---")
model.eval()
generated = input_ids.clone()
live_manager.clear()

for step in range(40):
    live_state = live_manager.get_current_state()
    if live_state is not None:
        live_state = live_state.to("cuda")
    
    with torch.no_grad():
        outputs = model(generated, live_state=live_state, return_dict=True)
    
    logits = outputs["logits"][:, -1, :] / 0.7
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([generated, next_token], dim=-1)
    
    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
    print(token_text, end="", flush=True)

print("\n")

# Generate WITH signal using LiveInferenceEngine to surface interrupt text
print("\n--- With Signal (LiveInferenceEngine, high-priority) ---")
live_manager.clear()
engine = LiveInferenceEngine(model, tokenizer, live_manager, device="cuda")

def _inject_signal():
    time.sleep(1.0)
    print("\n[INJECTING: URGENT: Heavy storm warning issued!]")
    live_manager.post_signal("URGENT: Heavy storm warning issued!", priority=1.0)

inject_thread = threading.Thread(target=_inject_signal, daemon=True)
inject_thread.start()

gen_cfg = GenerationConfig(max_new_tokens=80, temperature=0.7, top_p=0.9)

try:
    for tok in engine.generate_stream("Describe the weather conditions outside.", gen_cfg):
        print(tok, end="", flush=True)
except KeyboardInterrupt:
    pass

print("\n\nDone!")
