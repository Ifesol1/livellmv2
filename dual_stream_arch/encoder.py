"""
Live State Encoder - The "Senses" of the Dual-Stream Architecture

This module processes external live signals into dense vectors that can be
injected into the main transformer's cross-attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import threading
import queue
import json


@dataclass
class LiveSignal:
    """A single live signal with timestamp and content."""
    timestamp: float
    content: Dict[str, Any]
    priority: float = 1.0  # Higher priority = stronger influence
    embedding: Optional[torch.Tensor] = None


class LiveStateEncoder(nn.Module):
    """
    Encodes live state signals into dense vectors for cross-attention.
    
    This is the "sensory" pathway of the Dual-Stream architecture.
    It continuously processes incoming signals and maintains an embedding
    buffer that the main model can attend to.
    
    Architecture:
        Input (JSON/Dict) -> Text Embedding -> MLP -> Output Vector
        
    The output vector has the same dimensionality as the main model's
    hidden states, allowing seamless cross-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2560,  # Qwen3-4B hidden dim
        input_dim: int = 768,  # Size of text embeddings
        num_signal_slots: int = 8,  # How many concurrent signals to track
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_signal_slots = num_signal_slots
        
        # Text encoder for string-based signals
        # In production, this could be a small BERT or sentence transformer
        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Numeric encoder for structured data (e.g., sensor readings)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),  # Up to 64 numeric fields
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        # Priority-weighted attention for combining multiple signals
        self.signal_combiner = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # Final projection with layer norm
        self.output_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable query for signal combination
        self.combiner_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Signal type embeddings
        self.type_embeddings = nn.Embedding(16, hidden_dim)  # 16 signal types
        
        # Buffer for live signals
        self.register_buffer(
            "signal_buffer",
            torch.zeros(1, num_signal_slots, hidden_dim)
        )
        self.register_buffer(
            "signal_priorities",
            torch.zeros(1, num_signal_slots)
        )
        self.register_buffer(
            "buffer_fill_count",
            torch.tensor(0, dtype=torch.long)
        )
        
    def encode_text_signal(
        self,
        text_embedding: torch.Tensor,
        signal_type: int = 0,
    ) -> torch.Tensor:
        """
        Encode a text-based signal.
        
        Args:
            text_embedding: Pre-computed text embedding [batch, input_dim]
            signal_type: Type ID for the signal (e.g., 0=alert, 1=status, 2=sensor)
            
        Returns:
            Encoded signal vector [batch, hidden_dim]
        """
        # Project text embedding
        h = self.text_projection(text_embedding)
        
        # Add type embedding
        type_emb = self.type_embeddings(
            torch.tensor([signal_type], device=h.device, dtype=torch.long)
        )
        h = h + type_emb
        
        return self.output_norm(h)
    
    def encode_numeric_signal(
        self,
        values: torch.Tensor,
        signal_type: int = 0,
    ) -> torch.Tensor:
        """
        Encode a numeric/sensor signal.
        
        Args:
            values: Numeric values [batch, num_values] (max 64)
            signal_type: Type ID for the signal
            
        Returns:
            Encoded signal vector [batch, hidden_dim]
        """
        batch_size = values.shape[0]
        
        # Pad to expected size (preserve dtype)
        if values.shape[1] < 64:
            padding = torch.zeros(batch_size, 64 - values.shape[1], device=values.device, dtype=values.dtype)
            values = torch.cat([values, padding], dim=1)
        
        h = self.numeric_encoder(values)
        
        # Add type embedding
        type_emb = self.type_embeddings(
            torch.tensor([signal_type], device=h.device, dtype=torch.long)
        )
        h = h + type_emb
        
        return self.output_norm(h)
    
    def update_buffer(
        self,
        signal_embedding: torch.Tensor,
        priority: float = 1.0,
        slot: Optional[int] = None,
    ):
        """
        Update the live signal buffer with a new signal.
        
        Args:
            signal_embedding: Encoded signal [1, hidden_dim]
            priority: Importance weight (0-1)
            slot: Specific slot to update (auto-assigns if None)
        """
        if slot is None:
            # Use round-robin assignment
            slot = int(self.buffer_fill_count.item()) % self.num_signal_slots
            self.buffer_fill_count += 1
        
        self.signal_buffer[0, slot] = signal_embedding.squeeze(0)
        self.signal_priorities[0, slot] = priority
    
    def get_live_state(self) -> Optional[torch.Tensor]:
        """
        Get the combined live state vector for cross-attention.
        
        Returns:
            Combined state vector [1, 1, hidden_dim] or None if no signals
        """
        # Mask empty slots
        mask = self.signal_priorities > 0
        
        if not mask.any():
            # No active signals - return None (cross-attention should be skipped)
            return None
        
        # Weight signals by priority
        weighted_signals = self.signal_buffer * self.signal_priorities.unsqueeze(-1)
        
        # Use attention to combine signals
        query = self.combiner_query.expand(1, -1, -1)
        combined, _ = self.signal_combiner(
            query=query,
            key=weighted_signals,
            value=weighted_signals,
        )
        
        return self.output_projection(combined)
    
    def clear_buffer(self):
        """Clear all signals from the buffer."""
        self.signal_buffer.zero_()
        self.signal_priorities.zero_()
        self.buffer_fill_count.zero_()
    
    def forward(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        numeric_values: Optional[torch.Tensor] = None,
        signal_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass - encode signals and return combined state.
        
        Args:
            text_embeddings: Optional text embeddings [batch, seq, input_dim]
            numeric_values: Optional numeric values [batch, seq, num_values]
            signal_types: Signal type IDs [batch, seq]
            
        Returns:
            Combined live state [batch, 1, hidden_dim]
        """
        batch_size = 1
        embeddings = []
        
        if text_embeddings is not None:
            batch_size = text_embeddings.shape[0]
            for i in range(text_embeddings.shape[1]):
                sig_type = signal_types[0, i].item() if signal_types is not None else 0
                emb = self.encode_text_signal(text_embeddings[:, i], int(sig_type))
                embeddings.append(emb)
        
        if numeric_values is not None:
            batch_size = numeric_values.shape[0]
            for i in range(numeric_values.shape[1]):
                sig_type = signal_types[0, i].item() if signal_types is not None else 1
                emb = self.encode_numeric_signal(numeric_values[:, i], int(sig_type))
                embeddings.append(emb)
        
        if not embeddings:
            return torch.zeros(batch_size, 1, self.hidden_dim, device=next(self.parameters()).device)
        
        # Stack and combine
        stacked = torch.stack(embeddings, dim=1)  # [batch, num_signals, hidden_dim]
        
        # Use attention to combine
        query = self.combiner_query.expand(batch_size, -1, -1)
        combined, _ = self.signal_combiner(
            query=query,
            key=stacked,
            value=stacked,
        )
        
        return self.output_projection(combined)


class LiveBufferManager:
    """
    Thread-safe manager for the live signal buffer.
    
    This allows external systems to post signals in real-time while
    the model is generating tokens.
    """
    
    def __init__(
        self,
        encoder: LiveStateEncoder,
        text_embedder: Optional[nn.Module] = None,
    ):
        self.encoder = encoder
        self.text_embedder = text_embedder
        self.signal_queue = queue.Queue()
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None
        self._last_signal_text: Optional[str] = None
    
    @property
    def is_running(self) -> bool:
        """Return whether the background worker is active."""
        return self._running

    def ensure_running(self):
        """
        Idempotently start the background worker.
        
        The original API required callers to invoke start() manually before
        posting signals. That made mid-generation injections easy to miss.
        """
        if not self._running:
            self.start()
        
    def start(self):
        """Start the background signal processing thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_signals, daemon=True)
        self._worker_thread.start()
        
    def stop(self):
        """Stop the background processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
    
    def post_signal(
        self,
        content: Union[str, Dict[str, Any]],
        priority: float = 1.0,
        signal_type: int = 0,
    ):
        """
        Post a new signal to the buffer.
        
        Args:
            content: Signal content (string or dict)
            priority: Importance weight (0-1)
            signal_type: Type ID
        """
        payload = {
            "content": content,
            "priority": priority,
            "signal_type": signal_type,
        }

        # If the worker isn't running, process synchronously so callers
        # can inject signals without worrying about lifecycle.
        if not self._running:
            self._encode_and_buffer(payload)
            return

        self.signal_queue.put(payload)
    
    def post_alert(self, message: str, priority: float = 0.9):
        """Post a high-priority alert."""
        self.post_signal(message, priority=priority, signal_type=0)
    
    def post_status(self, status: Dict[str, Any], priority: float = 0.5):
        """Post a status update."""
        self.post_signal(status, priority=priority, signal_type=1)
    
    def post_sensor(self, values: List[float], priority: float = 0.3):
        """Post sensor readings."""
        self.post_signal({"values": values}, priority=priority, signal_type=2)
    
    def _process_signals(self):
        """Background thread that processes incoming signals."""
        while self._running:
            try:
                signal = self.signal_queue.get(timeout=0.01)
                self._encode_and_buffer(signal)
            except queue.Empty:
                continue
    
    def _encode_and_buffer(self, signal: Dict):
        """Encode a signal and add it to the buffer."""
        with self._lock:
            content = signal["content"]
            priority = signal["priority"]
            sig_type = signal["signal_type"]
            target_device = next(self.encoder.parameters()).device
            # Track last signal text for user-facing interrupts
            if isinstance(content, str):
                self._last_signal_text = content
            elif isinstance(content, dict) and "text" in content:
                self._last_signal_text = str(content.get("text"))
            else:
                self._last_signal_text = str(content)
            
            if isinstance(content, str) and self.text_embedder is not None:
                # Encode text
                with torch.no_grad():
                    text_emb = self.text_embedder(content).to(device=target_device)
                    encoded = self.encoder.encode_text_signal(text_emb, sig_type)
            elif isinstance(content, dict) and "values" in content:
                # Encode numeric
                values = torch.tensor([content["values"]], dtype=torch.float32, device=target_device)
                with torch.no_grad():
                    encoded = self.encoder.encode_numeric_signal(values, sig_type)
            else:
                # Fallback: encode as JSON string if we have text embedder
                if self.text_embedder is not None:
                    text_emb = self.text_embedder(json.dumps(content)).to(device=target_device)
                    encoded = self.encoder.encode_text_signal(text_emb, sig_type)
                else:
                    return  # Cannot encode without embedder
            
            self.encoder.update_buffer(encoded, priority)
    
    def get_current_state(self) -> Optional[torch.Tensor]:
        """Get the current live state vector (thread-safe). Returns None if no signals."""
        with self._lock:
            return self.encoder.get_live_state()
    
    def get_max_priority(self) -> float:
        """Safely read the highest priority currently buffered."""
        with self._lock:
            if self.encoder.signal_priorities.numel() == 0:
                return 0.0
            return float(self.encoder.signal_priorities.max().item())
    
    def get_last_signal_text(self) -> Optional[str]:
        """Return the most recent signal content (as text) if available."""
        with self._lock:
            return self._last_signal_text
    
    def clear(self):
        """Clear all signals."""
        with self._lock:
            self.encoder.clear_buffer()
