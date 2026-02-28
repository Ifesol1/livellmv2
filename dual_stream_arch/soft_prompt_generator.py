"""
Soft Prompt Generator - Embedding Injection via Cross-Attention

This combines:
1. The model's own embeddings (no training needed)
2. Cross-attention gates (subconscious injection)

The signal's MEANING flows through the gates, not as visible text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Generator, Callable, Tuple
import threading
import time

from .native_encoder import NativeSignalEncoder, NativeSignalBuffer, LiveSignal, create_native_encoder
from .attention import GatedCrossAttention


class SoftPromptInjector(nn.Module):
    """
    Injects signal embeddings into model's hidden states.
    
    Simple approach: add scaled signal embedding to hidden states.
    This biases the model toward the signal's meaning without complex attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        gate_init: float = 0.1,  # Start small to avoid disruption
    ):
        super().__init__()
        
        # Simple scaling gate - learnable but starts small
        self.gate = nn.Parameter(torch.tensor(gate_init))
        
        # Optional: learned projection to blend signal into hidden space
        self.blend = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.blend.weight)  # Start as identity
        self.blend.weight.data *= 0.1  # Scale down initially
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        signal_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject signal meaning into hidden states.
        
        Args:
            hidden_states: [batch, seq, hidden] - model's internal state
            signal_embeddings: [batch, slots, hidden] - signal meaning vectors
            
        Returns:
            Modified hidden states with signal "felt" subconsciously
        """
        # Pool signal to single vector
        signal_vec = signal_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        # Project and scale
        signal_proj = self.blend(signal_vec)
        
        # Add to all positions with small gate
        gate_value = torch.sigmoid(self.gate)  # 0-1 range
        injection = gate_value * signal_proj
        
        return hidden_states + injection


class SoftPromptGenerator:
    """
    Generator with soft prompt injection.
    
    Signals are converted to embeddings using the model's own vocabulary,
    then injected via cross-attention gates during generation.
    
    The model "feels" the signal meaning without seeing it as text.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        injection_layer: int = -1,  # Which layer to inject at (-1 = middle)
        gate_strength: float = 0.3,
        adapter_path: Optional[str] = None,  # Path to trained adapter weights
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = next(model.parameters()).dtype
        
        # Create native encoder using model's own embeddings
        self.encoder = create_native_encoder(model, tokenizer)
        self.encoder.to(device=device, dtype=self.dtype)
        
        # Signal buffer
        self.buffer = NativeSignalBuffer(self.encoder)
        
        # Get hidden dim
        if hasattr(model.config, "hidden_size"):
            hidden_dim = model.config.hidden_size
        else:
            hidden_dim = self.encoder.hidden_dim
        
        # Create injector with matching dtype
        self.injector = SoftPromptInjector(
            hidden_dim=hidden_dim,
            gate_init=gate_strength,
        ).to(device=device, dtype=self.dtype)
        
        # Load trained adapter if provided
        self.adapter_path = adapter_path
        if adapter_path is not None:
            print(f"Loading trained adapter from {adapter_path}")
            state_dict = torch.load(adapter_path, map_location=device)
            self.injector.load_state_dict(state_dict)
            print("Adapter loaded!")
        
        # Determine injection layers (multiple for stronger effect)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            num_layers = len(model.transformer.h)
        else:
            num_layers = 24  # Default guess
        
        # Inject at multiple layers for stronger effect
        if injection_layer >= 0:
            self.injection_layers = [injection_layer]
        else:
            # Inject at 3 layers: early, middle, late
            self.injection_layers = [
                num_layers // 4,      # Early
                num_layers // 2,      # Middle
                3 * num_layers // 4,  # Late
            ]
        
        # Hook storage
        self._current_signal_state = None
        self._hook_handles = []
        
        # Callbacks
        self.on_signal_injected: Optional[Callable[[str], None]] = None
    
    def post_signal(self, content: str, priority: float = 0.5):
        """Post a signal to be injected subconsciously."""
        self.buffer.post(content, priority)
        
    def post_alert(self, message: str):
        """Post high-priority alert."""
        self.buffer.post_alert(message)
    
    def _get_layers(self):
        """Get the model's layer list."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        return None
    
    def _create_injection_hook(self):
        """Create a forward hook to inject signals at the target layer."""
        def hook(module, input, output):
            # Skip if no signal active
            if self._current_signal_state is None:
                return None  # Return None to leave output unchanged
            
            # Extract hidden states from output
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # Inject signal embedding
            hidden_states = self.injector(hidden_states, self._current_signal_state)
            
            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states
        
        return hook
    
    def _install_hook(self):
        """Install the injection hook on multiple layers."""
        layers = self._get_layers()
        if layers is None:
            return
        
        for layer_idx in self.injection_layers:
            target_layer = layers[layer_idx]
            handle = target_layer.register_forward_hook(self._create_injection_hook())
            self._hook_handles.append(handle)
    
    def _remove_hook(self):
        """Remove all injection hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text with soft prompt injection."""
        tokens = []
        for token in self.generate_stream(prompt, max_tokens, temperature, top_p):
            tokens.append(token)
        return "".join(tokens)
    
    def _process_tokens(
        self, 
        token_ids: torch.Tensor,
        past_kv: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Process tokens and return logits + updated KV cache."""
        with torch.no_grad():
            outputs = self.model(
                token_ids,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        return outputs.logits, outputs.past_key_values
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Stream tokens with soft prompt injection.
        
        Signals posted to buffer will be injected "subconsciously" -
        the model feels their meaning without seeing them as text.
        """
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except:
            text = prompt
        
        # Tokenize
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Install injection hook
        self._install_hook()
        
        self.model.eval()
        
        try:
            # Process initial prompt
            logits, past_kv = self._process_tokens(input_ids, None)
            
            for _ in range(max_tokens):
                # Check for signal
                self._current_signal_state = self.buffer.get_state()
                if self._current_signal_state is not None:
                    self._current_signal_state = self._current_signal_state.to(
                        device=self.device, dtype=self.dtype
                    )
                    if self.on_signal_injected:
                        self.on_signal_injected("signal active")
                
                # Sample next token
                next_logits = logits[:, -1, :]
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                
                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Process next token and update cache
                logits, past_kv = self._process_tokens(next_token, past_kv)
                
                # Decode and yield
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_text
                
        finally:
            self._remove_hook()
            self._current_signal_state = None


def demo():
    """Demo soft prompt injection."""
    import threading
    
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    gen = SoftPromptGenerator(model, tokenizer)
    
    # Post signal after delay
    def inject():
        time.sleep(2.0)
        print("\n>>> INJECTING: 'DANGER: Fire detected!' <<<")
        gen.post_alert("DANGER: Fire detected in building!")
    
    threading.Thread(target=inject, daemon=True).start()
    
    print("="*60)
    print("SOFT PROMPT INJECTION DEMO")
    print("Signal meaning injected subconsciously via cross-attention")
    print("="*60)
    print()
    
    prompt = "Write a calm, relaxing description of an office building."
    print(f"Prompt: {prompt}")
    print()
    print("Output: ", end="", flush=True)
    
    for token in gen.generate_stream(prompt, max_tokens=100):
        print(token, end="", flush=True)
    
    print("\n")


if __name__ == "__main__":
    demo()
