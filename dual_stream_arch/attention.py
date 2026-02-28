"""
Gated Cross-Attention for Live Signal Injection

This module implements the cross-attention mechanism that allows
the model to attend to external live signals during generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GatedCrossAttention(nn.Module):
    """
    Cross-attention layer with learnable gating.
    
    The gate controls how much the model attends to live signals vs.
    its own internal representations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 32,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Learnable gate parameter (scalar)
        # tanh(gate_param) gives the actual gate value in [-1, 1]
        self.gate_param = nn.Parameter(torch.tensor(gate_init))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        live_state: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for gated cross-attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            live_state: [batch, live_len, hidden_dim]
            position_ids: Optional position IDs
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q from hidden states, K/V from live state
        query = self.q_proj(hidden_states)
        key = self.k_proj(live_state)
        value = self.v_proj(live_state)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        # Apply gate
        gate = torch.tanh(self.gate_param)
        output = gate * attn_output
        
        return output, attn_weights
    
    def get_gate_info(self) -> dict:
        """Get information about the gate state."""
        gate_value = torch.tanh(self.gate_param).item()
        return {
            "gate_param": self.gate_param.item(),
            "gate_value": gate_value,
            "gate_percent": f"{abs(gate_value) * 100:.1f}%",
        }


class RoPEGatedCrossAttention(GatedCrossAttention):
    """
    Gated cross-attention with Rotary Position Embeddings.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 32,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__(hidden_dim, num_heads, dropout, gate_init)
        
        self.max_position = max_position
        self.rope_theta = rope_theta
        
        # Precompute RoPE frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        seq_len = x.shape[2]
        
        # Compute sin/cos
        freqs = torch.einsum("i,j->ij", position_ids[0].float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        rotated = torch.cat([
            x1 * cos[..., :self.head_dim // 2] - x2 * sin[..., :self.head_dim // 2],
            x1 * sin[..., self.head_dim // 2:] + x2 * cos[..., self.head_dim // 2:],
        ], dim=-1)
        
        return rotated
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        live_state: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with RoPE."""
        batch_size, seq_len, _ = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        # Q, K, V projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(live_state)
        value = self.v_proj(live_state)
        
        # Reshape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query (not to key/value from live state)
        query = self._apply_rotary_pos_emb(query, position_ids)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        # Gate
        gate = torch.tanh(self.gate_param)
        output = gate * attn_output
        
        return output, attn_weights


class AdaptiveGate(nn.Module):
    """
    Adaptive gating that learns when to attend to live signals
    based on the content of both streams.
    """
    
    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive gate value.
        
        Args:
            hidden_states: Main stream states
            cross_output: Cross-attention output
            
        Returns:
            Gate values [batch, seq_len, 1]
        """
        # Pool over sequence for gate computation
        h_pool = hidden_states.mean(dim=1, keepdim=True)
        c_pool = cross_output.mean(dim=1, keepdim=True)
        
        combined = torch.cat([h_pool, c_pool], dim=-1)
        gate = self.gate_net(combined)
        
        return gate.expand_as(cross_output[..., :1])
