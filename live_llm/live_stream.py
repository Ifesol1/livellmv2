"""
True Live Streaming with KV Cache

This implements REAL continuous streaming where:
1. KV cache is maintained across tokens
2. When signal arrives, we EXTEND the cache (not restart)
3. Generation continues seamlessly from where it left off

This is O(n) not O(n²) - much more efficient!
"""

import torch
import threading
import queue
import time
from typing import Optional, Generator, Tuple
from dataclasses import dataclass, field


@dataclass 
class LiveSignal:
    content: str
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)


class SignalQueue:
    """Thread-safe signal queue."""
    def __init__(self):
        self._queue = queue.Queue()
    
    def post(self, content: str, priority: float = 1.0):
        self._queue.put(LiveSignal(content, priority))
    
    def get(self) -> Optional[LiveSignal]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class LiveStreamGenerator:
    """
    True streaming generator with KV cache management.
    
    When a signal arrives:
    1. We DON'T restart generation
    2. We process the signal tokens and EXTEND the KV cache
    3. Generation continues from the extended cache
    
    This is how it should work for real-time applications.
    """
    
    def __init__(self, model, tokenizer, device="cuda", signal_style: str = "user"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.signal_style = signal_style  # How signals appear: user, system, visible, subtle, raw
        self.signal_queue = SignalQueue()
        self._chat_format = self._detect_chat_format()
        print(f"[LiveStream] Detected chat format: {self._chat_format}")
    
    def _detect_chat_format(self) -> str:
        """Detect chat template format from tokenizer vocab."""
        vocab = self.tokenizer.get_vocab()
        if '<|start_header_id|>' in vocab:
            return 'llama'
        elif '<|im_start|>' in vocab:
            return 'chatml'
        return 'generic'
        
    def post_signal(self, content: str, priority: float = 1.0):
        """Post a signal to be injected."""
        print(f"[DEBUG] Signal posted to queue: {content}", flush=True)
        self.signal_queue.post(content, priority)
    
    def _format_signal(self, content: str, style: str = "user") -> str:
        """
        Format signal for injection using the correct chat template tokens.
        """
        if style == "user":
            if self._chat_format == 'llama':
                return f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif self._chat_format == 'chatml':
                return f"<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            else:
                return f"\n\nUser: {content}\n\nAssistant: "
        elif style == "system":
            return f"\n[System note: {content}]\n"
        elif style == "visible":
            return f"\n[LIVE]: {content}\n"
        elif style == "subtle":
            return f" ({content}) "
        elif style == "raw":
            return f"\n{content}\n"
        else:
            return f"\n\nUser: {content}\n\nAssistant: "
    
    def _clean_display_text(self, text: str) -> str:
        """Remove all chat template tokens for clean display."""
        import re
        # ChatML tokens
        text = re.sub(r'<\|im_start\|>\w*\n?', '', text)
        text = re.sub(r'<\|im_end\|>\n?', '', text)
        # Llama tokens
        text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>\n*', '', text)
        text = re.sub(r'<\|eot_id\|>\n?', '', text)
        text = re.sub(r'<\|begin_of_text\|>', '', text)
        text = re.sub(r'<\|end_of_text\|>', '', text)
        # Thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text
    
    def _process_tokens_into_cache(
        self, 
        token_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Process tokens and return updated KV cache.
        This extends the cache without reprocessing old tokens.
        """
        with torch.no_grad():
            outputs = self.model(
                token_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        return outputs.logits, outputs.past_key_values
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Generate with true streaming and live signal injection.
        
        The KV cache is maintained throughout. When signals arrive,
        they're processed and the cache is extended - no restart needed.
        """
        # Format prompt - DISABLE thinking for faster response
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False  # DISABLED for fast, direct responses
            )
        except:
            # Fallback for models that don't support enable_thinking
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                text = prompt
        
        # Initial tokenization
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Process prompt and initialize KV cache
        logits, past_kv = self._process_tokens_into_cache(input_ids, None)
        
        self.model.eval()
        generated_count = 0
        
        while generated_count < max_tokens:
            # Check for signal
            signal = self.signal_queue.get()
            if signal:
                print(f"[DEBUG] Processing signal from queue: {signal.content}", flush=True)
                # Format signal for model (with ChatML tokens)
                signal_text = self._format_signal(signal.content, self.signal_style)
                signal_ids = self.tokenizer.encode(
                    signal_text, 
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.device)
                
                # EXTEND the KV cache with signal tokens
                logits, past_kv = self._process_tokens_into_cache(signal_ids, past_kv)
                
                # Don't yield display text - frontend handles signal display separately
            
            # Sample next token from logits
            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Process the new token and extend cache
            logits, past_kv = self._process_tokens_into_cache(next_token, past_kv)
            
            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text
            
            generated_count += 1


def demo():
    """Demo of live streaming with signal injection."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    generator = LiveStreamGenerator(model, tokenizer)
    
    # Schedule signal injection
    def inject_later():
        time.sleep(2.0)
        print("\n>>> INJECTING SIGNAL <<<")
        generator.post_signal("URGENT: User just said they're allergic to nuts!")
    
    threading.Thread(target=inject_later, daemon=True).start()
    
    print("="*60)
    print("TRUE STREAMING WITH KV CACHE")
    print("="*60)
    print("\nPrompt: Suggest a healthy snack.")
    print("\nOutput: ", end="", flush=True)
    
    for token in generator.generate_stream(
        "Suggest some healthy snacks I could eat.",
        max_tokens=100,
    ):
        print(token, end="", flush=True)
        time.sleep(0.05)
    
    print("\n\nDone!")


if __name__ == "__main__":
    demo()
