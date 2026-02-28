"""
Native Signal Encoder - Soft Prompting / Embedding Injection

Instead of training a separate encoder, we use the model's OWN embedding layer
to convert signal text into vectors the model already understands.

Flow:
    Signal Text ("Alert: Critical") 
        → Tokenize (model's tokenizer)
        → Embed (model's embed_tokens layer)  
        → Pool (compress to fixed size)
        → Inject via cross-attention gate

Benefits:
    - No training needed - model already knows these embeddings
    - "Subconscious" effect - meaning flows through gates, not text stream
    - Works with any model immediately
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from dataclasses import dataclass
import threading
import queue
import time


@dataclass
class LiveSignal:
    """A signal to inject into the model's subconscious."""
    content: str
    priority: float = 0.5
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class NativeSignalEncoder(nn.Module):
    """
    Encodes signals using the model's own embedding layer.
    
    This is the key insight: the model already has a "dictionary" (embed_tokens)
    that maps tokens to meaning-vectors. We just use that directly.
    """
    
    def __init__(
        self,
        embed_tokens: nn.Embedding,
        tokenizer,
        hidden_dim: int,
        max_signal_tokens: int = 32,
        pooling: str = "mean",  # "mean", "max", "attention", "concat"
        num_slots: int = 4,  # Number of signal slots for cross-attention
    ):
        super().__init__()
        
        self.embed_tokens = embed_tokens  # Borrowed from model - NOT trained
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.max_signal_tokens = max_signal_tokens
        self.pooling = pooling
        self.num_slots = num_slots
        
        # Freeze the embedding layer - we don't want to modify it
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        
        # Optional: learnable signal prefix (like a "this is a signal" marker)
        # This is tiny and can be trained to help model recognize signals
        self.signal_prefix = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Optional: attention pooling
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Light projection (optional - can help shape the signal)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.proj.weight)  # Start as identity
        
    def forward(
        self,
        signal_text: Union[str, List[str]],
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Convert signal text to embeddings using model's own vocabulary.
        
        Args:
            signal_text: Signal string or list of strings
            return_all_tokens: If True, return all token embeddings (not pooled)
            
        Returns:
            Tensor of shape [batch, num_slots, hidden_dim] for cross-attention
        """
        if isinstance(signal_text, str):
            signal_text = [signal_text]
        
        batch_size = len(signal_text)
        device = self.signal_prefix.device
        
        # Tokenize signals
        encoded = self.tokenizer(
            signal_text,
            padding=True,
            truncation=True,
            max_length=self.max_signal_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get embeddings from MODEL'S vocabulary (the key insight!)
        # This gives us vectors the model already understands
        with torch.no_grad():  # Don't backprop through embeddings
            embeddings = self.embed_tokens(input_ids)  # [batch, seq, hidden]
            # Ensure dtype matches our layers
            embeddings = embeddings.to(dtype=self.proj.weight.dtype)
        
        if return_all_tokens:
            return embeddings
        
        # Pool to fixed number of slots for cross-attention
        if self.pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled.unsqueeze(1).expand(-1, self.num_slots, -1)
            
        elif self.pooling == "attention":
            # Attention pooling - learn to select important parts
            query = self.pool_query.unsqueeze(0).expand(batch_size, -1, -1)
            pooled, _ = self.pool_attn(query, embeddings, embeddings)
            
        elif self.pooling == "concat":
            # Take first N tokens as slots
            pooled = embeddings[:, :self.num_slots, :]
            if pooled.size(1) < self.num_slots:
                padding = torch.zeros(batch_size, self.num_slots - pooled.size(1), self.hidden_dim, device=device)
                pooled = torch.cat([pooled, padding], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Add signal prefix and project
        pooled = pooled + self.signal_prefix
        pooled = self.proj(pooled)
        
        return pooled  # [batch, num_slots, hidden_dim]
    
    def encode_signal(self, signal: LiveSignal) -> torch.Tensor:
        """Encode a LiveSignal object."""
        return self.forward(signal.content)


class NativeSignalBuffer:
    """
    Thread-safe buffer for managing live signals.
    
    Signals are encoded on-demand using the model's embeddings.
    """
    
    def __init__(
        self,
        encoder: NativeSignalEncoder,
        max_signals: int = 8,
        decay_seconds: float = 30.0,
    ):
        self.encoder = encoder
        self.max_signals = max_signals
        self.decay_seconds = decay_seconds
        
        self._signals: List[LiveSignal] = []
        self._lock = threading.Lock()
        self._cached_state: Optional[torch.Tensor] = None
        self._cache_valid = False
        
    def post(self, content: str, priority: float = 0.5):
        """Post a new signal to the buffer."""
        signal = LiveSignal(content=content, priority=priority)
        
        with self._lock:
            self._signals.append(signal)
            
            # Keep only most recent/important
            if len(self._signals) > self.max_signals:
                self._signals.sort(key=lambda s: (s.priority, s.timestamp))
                self._signals = self._signals[-self.max_signals:]
            
            self._cache_valid = False
    
    def post_alert(self, message: str):
        """Post high-priority alert."""
        self.post(message, priority=0.95)
    
    def get_state(self) -> Optional[torch.Tensor]:
        """
        Get current signal state as embeddings for cross-attention.
        
        Returns None if no signals in buffer.
        """
        with self._lock:
            # Remove expired signals
            now = time.time()
            self._signals = [
                s for s in self._signals 
                if now - s.timestamp < self.decay_seconds
            ]
            
            if not self._signals:
                return None
            
            # Return cached if valid
            if self._cache_valid and self._cached_state is not None:
                return self._cached_state
            
            # Encode all signals
            texts = [s.content for s in self._signals]
            priorities = torch.tensor([s.priority for s in self._signals])
            
            # Encode using model's embeddings
            embeddings = self.encoder(texts)  # [num_signals, slots, hidden]
            
            # Weight by priority and combine
            weights = priorities.softmax(dim=0).view(-1, 1, 1).to(embeddings.device)
            combined = (embeddings * weights).sum(dim=0, keepdim=True)  # [1, slots, hidden]
            
            self._cached_state = combined
            self._cache_valid = True
            
            return combined
    
    def clear(self):
        """Clear all signals."""
        with self._lock:
            self._signals.clear()
            self._cached_state = None
            self._cache_valid = False


def create_native_encoder(model, tokenizer) -> NativeSignalEncoder:
    """
    Create a NativeSignalEncoder from any HuggingFace model.
    
    This extracts the model's embedding layer and uses it to encode signals.
    """
    # Find the embedding layer
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, "embed_tokens"):
        embed_tokens = model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        embed_tokens = model.transformer.wte  # GPT-2 style
    else:
        raise ValueError("Could not find embedding layer in model")
    
    hidden_dim = embed_tokens.weight.shape[1]
    
    return NativeSignalEncoder(
        embed_tokens=embed_tokens,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
    )
