"""
Dual-Stream Decoder Layer - The Modified Transformer Block

This module replaces the standard decoder layer with one that includes
gated cross-attention for live state injection.

Original Architecture:
    Input -> Self-Attention -> RMSNorm -> FeedForward -> Output

New Architecture:
    Input -> Self-Attention -> [Gated Cross-Attention] -> RMSNorm -> FeedForward -> Output

The cross-attention layer "looks sideways" at the live buffer while
the self-attention "looks backward" at the token history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from .attention import GatedCrossAttention, RoPEGatedCrossAttention, AdaptiveGate


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Qwen3/Llama/Mistral)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):  # Qwen3 uses 1e-6
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DualStreamMLP(nn.Module):
    """Feed-forward network matching Qwen3/Llama/Mistral architecture (SwiGLU)."""
    
    def __init__(
        self,
        hidden_dim: int = 2560,
        intermediate_dim: int = 9728,  # Qwen3-4B default
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation (Llama/Mistral style)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DualStreamDecoderLayer(nn.Module):
    """
    Modified Transformer decoder layer with live state injection.
    
    This layer extends the standard decoder with gated cross-attention
    that allows the model to attend to external live signals during
    generation.
    
    Components:
        1. Self-Attention: Standard causal attention over token history
        2. Live Cross-Attention: Attends to external live buffer (NEW)
        3. Feed-Forward: Standard MLP transformation
        
    All components are connected via residual connections with RMSNorm.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2560,  # Qwen3-4B hidden dim
        num_heads: int = 32,
        num_kv_heads: Optional[int] = 8,  # Qwen3 uses GQA with 8 KV heads
        intermediate_dim: int = 9728,  # Qwen3-4B intermediate dim
        max_position: int = 32768,  # Qwen3 max position
        rope_theta: float = 1000000.0,  # Qwen3 uses 1M theta
        use_rope: bool = True,
        use_adaptive_gate: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_adaptive_gate = use_adaptive_gate
        
        # Pre-normalization layers
        self.input_norm = RMSNorm(hidden_dim)
        self.post_attn_norm = RMSNorm(hidden_dim)
        self.post_cross_norm = RMSNorm(hidden_dim)
        
        # Self-attention (standard causal attention)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Live cross-attention (THE KEY ADDITION)
        if use_rope:
            self.live_cross_attn = RoPEGatedCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_position=max_position,
                rope_theta=rope_theta,
                dropout=dropout,
            )
        else:
            self.live_cross_attn = GatedCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # Optional adaptive gating
        if use_adaptive_gate:
            self.adaptive_gate = AdaptiveGate(hidden_dim)
        
        # Feed-forward network
        self.mlp = DualStreamMLP(hidden_dim, intermediate_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        live_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through the dual-stream decoder layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            live_state: Live buffer states [batch, live_len, hidden_dim]
            attention_mask: Causal mask for self-attention
            position_ids: Position IDs for RoPE
            past_key_value: Cached KV for incremental decoding
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of:
                - Output hidden states
                - Updated cache (if use_cache=True)
        """
        residual = hidden_states
        
        # 1. SELF-ATTENTION (Looking backward at history)
        hidden_states = self.input_norm(hidden_states)
        
        # Create causal mask for self-attention (must match hidden_states dtype)
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        )
        
        attn_output, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            is_causal=False,  # We provide explicit mask
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # 2. LIVE CROSS-ATTENTION (Looking sideways at live buffer)
        if live_state is not None:
            residual = hidden_states
            hidden_states = self.post_attn_norm(hidden_states)
            
            cross_output, _ = self.live_cross_attn(
                hidden_states,
                live_state,
                position_ids=position_ids,
            )
            
            # Apply adaptive gating if enabled
            if self.use_adaptive_gate:
                gate = self.adaptive_gate(hidden_states, cross_output)
                cross_output = gate * cross_output
            
            hidden_states = residual + self.dropout(cross_output)
        
        # 3. FEED-FORWARD (Standard transformation)
        residual = hidden_states
        hidden_states = self.post_cross_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        
        return hidden_states, None  # Cache handling TBD
    
    def get_gate_info(self) -> dict:
        """Get information about the cross-attention gate."""
        return self.live_cross_attn.get_gate_info()


class DualStreamDecoderLayerWrapper(nn.Module):
    """
    Wrapper that adds dual-stream capability to an existing decoder layer.
    
    This approach allows you to "wrap" an existing Llama/Mistral layer
    without fully reimplementing it, preserving pretrained weights.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        inject_after_self_attn: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.inject_after_self_attn = inject_after_self_attn
        
        # Add our cross-attention layer
        self.live_cross_attn = GatedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gate_init=gate_init,
        )
        
        self.cross_norm = RMSNorm(hidden_dim)
        
        # Hook storage
        self._intermediate_output = None
    
    def _hook_after_self_attn(self, module, input, output):
        """Hook to capture output after self-attention."""
        self._intermediate_output = output
    
    def _call_original_layer(self, hidden_states: torch.Tensor, **kwargs):
        """
        Call the original layer, handling different model architectures.
        
        Gemma 3 expects position_embeddings_global and position_embeddings_local
        as positional arguments, not keyword arguments.
        """
        # Extract position embeddings for Gemma 3
        position_embeddings = kwargs.pop("position_embeddings", None)
        position_embeddings_local = kwargs.pop("position_embeddings_local", None)
        
        # Check if this is a Gemma 3 layer (has specific signature)
        layer_class_name = self.original_layer.__class__.__name__
        if "Gemma3" in layer_class_name and position_embeddings is not None:
            # Gemma 3 requires these as positional args
            return self.original_layer(
                hidden_states,
                position_embeddings,  # position_embeddings_global
                position_embeddings_local if position_embeddings_local is not None else position_embeddings,
                **kwargs
            )
        else:
            # Other models (Qwen3, Llama, etc.) use keyword args
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            return self.original_layer(hidden_states, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        live_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass that injects live state into wrapped layer.
        """
        if live_state is None:
            # No live state - use original layer directly
            return self._call_original_layer(hidden_states, **kwargs)
        
        if self.inject_after_self_attn:
            # We need to intercept after self-attention
            # This requires modifying the forward pass
            outputs = self._call_original_layer(hidden_states, **kwargs)
            
            def _extract_hidden(output):
                if isinstance(output, tuple):
                    return output[0], output[1:], "tuple"
                if isinstance(output, dict):
                    hs = output.get("hidden_states") or output.get("last_hidden_state") or output.get("out")
                    if hs is None:
                        return output, (), None
                    return hs, output, "dict"
                if hasattr(output, "last_hidden_state"):
                    return output.last_hidden_state, output, "obj"
                if hasattr(output, "hidden_states"):
                    return output.hidden_states, output, "obj"
                return output, (), None

            hidden_states, other_outputs, output_kind = _extract_hidden(outputs)
            
            # Apply cross-attention
            residual = hidden_states
            cross_output, _ = self.live_cross_attn(
                self.cross_norm(hidden_states),
                live_state,
            )
            hidden_states = residual + cross_output
            
            if output_kind == "tuple":
                return (hidden_states,) + other_outputs
            if output_kind == "dict":
                updated = dict(other_outputs)
                updated["hidden_states"] = hidden_states
                updated["last_hidden_state"] = hidden_states
                return updated
            if output_kind == "obj":
                try:
                    if hasattr(other_outputs, "hidden_states"):
                        setattr(other_outputs, "hidden_states", hidden_states)
                    if hasattr(other_outputs, "last_hidden_state"):
                        setattr(other_outputs, "last_hidden_state", hidden_states)
                    return other_outputs
                except Exception:
                    pass
            if other_outputs:
                return hidden_states
            return hidden_states
        
        return self._call_original_layer(hidden_states, **kwargs)


def convert_layer_to_dual_stream(
    layer: nn.Module,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    layer_class: str = "auto",
) -> DualStreamDecoderLayerWrapper:
    """
    Convert an existing decoder layer to dual-stream.
    
    Args:
        layer: The original decoder layer (e.g., LlamaDecoderLayer)
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        layer_class: Type of layer ("llama", "mistral", or "auto")
        
    Returns:
        Wrapped layer with dual-stream capability
    """
    return DualStreamDecoderLayerWrapper(
        original_layer=layer,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
