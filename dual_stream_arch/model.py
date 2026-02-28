"""
DualStreamTransformer with Wrapper Strategy.

This modifies the DualStreamTransformer to wrap original HF layers
instead of reimplementing them, preserving RoPE and other specifics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union, Any
import logging
from dataclasses import dataclass

from .encoder import LiveStateEncoder, LiveBufferManager
from .decoder_layer import (
    DualStreamDecoderLayerWrapper,
    DualStreamDecoderLayer,
    RMSNorm,
)
from .attention import GatedCrossAttention
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

logger = logging.getLogger(__name__)

@dataclass
class DualStreamConfig:
    """Configuration for the Dual-Stream Transformer."""
    
    # Base model dimensions (defaults match Qwen3-4B)
    hidden_dim: int = 2560
    intermediate_dim: int = 9728
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: Optional[int] = 8
    vocab_size: int = 151936
    max_position: int = 32768
    
    # RoPE settings
    rope_theta: float = 1000000.0
    use_rope: bool = True
    
    # Sliding window attention
    sliding_window: Optional[int] = 4096
    
    # Live stream settings
    live_encoder_dim: int = 768
    num_signal_slots: int = 8
    inject_live_at_layers: List[int] = None
    
    # Gating
    use_adaptive_gate: bool = False
    gate_init: float = 0.31  # Start at ~30% (tanh(0.31)≈0.3), moderate influence
    
    # Training
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    
    # Model type for loading pretrained
    base_model: str = "qwen3"
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.inject_live_at_layers is None:
            self.inject_live_at_layers = list(range(0, self.num_layers, 4))
    
    @classmethod
    def from_qwen3_0_6b(cls) -> "DualStreamConfig":
        return cls(
            hidden_dim=1024,
            intermediate_dim=2816,
            num_layers=28,
            num_heads=16,
            num_kv_heads=8,
            vocab_size=151936,
            inject_live_at_layers=list(range(0, 28, 4)),
        )
    
    @classmethod
    def from_qwen3_4b(cls) -> "DualStreamConfig":
        return cls(
            hidden_dim=2560,
            intermediate_dim=9728,
            num_layers=36,
            num_heads=32,
            num_kv_heads=8,
            vocab_size=151936,
            inject_live_at_layers=list(range(0, 36, 4)),
        )

    @classmethod
    def from_qwen3_8b(cls) -> "DualStreamConfig":
        """
        Convenience preset for Qwen3-8B style models.
        
        Dimensions are based on published Qwen3-8B specs; adjust at load time
        if the upstream config differs.
        """
        return cls(
            hidden_dim=4096,
            intermediate_dim=11008,
            num_layers=40,
            num_heads=32,
            num_kv_heads=8,
            vocab_size=151936,
            inject_live_at_layers=list(range(0, 40, 4)),
        )

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DualStreamConfig":
        """
        Build a config directly from a HuggingFace model config.
        
        This enables plug-and-play support for other backbones like Gemma 3
        without hard-coding dimensions.
        """
        hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", 4096))
        intermediate_size = getattr(hf_config, "intermediate_size", getattr(hf_config, "ffn_dim", hidden_size * 4))
        num_layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 32))
        num_heads = getattr(hf_config, "num_attention_heads", getattr(hf_config, "n_head", 32))
        num_kv_heads = getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_kv_heads", num_heads))
        vocab_size = getattr(hf_config, "vocab_size", 32000)
        max_position = getattr(hf_config, "max_position_embeddings", getattr(hf_config, "seq_length", getattr(hf_config, "max_seq_len", 2048)))
        rope_theta = getattr(hf_config, "rope_theta", 1000000.0)
        sliding_window = getattr(hf_config, "sliding_window", getattr(hf_config, "sliding_window_size", None))
        base_model = getattr(hf_config, "model_type", "unknown")

        # Default injection cadence: every 4 layers (or at least once)
        step = max(1, num_layers // 9) if num_layers > 0 else 1
        inject_layers = list(range(0, num_layers, max(1, min(4, step))))

        return cls(
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            max_position=max_position,
            rope_theta=rope_theta,
            sliding_window=sliding_window if sliding_window is not None else 4096,
            inject_live_at_layers=inject_layers,
            base_model=base_model,
        )

class DualStreamEmbedding(nn.Module):
    def __init__(self, config: DualStreamConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

class DualStreamTransformer(nn.Module):
    """
    Dual-Stream Transformer that wraps a HuggingFace model's layers.
    
    This version preserves the exact behavior of the original model (RoPE, FlashAttn, etc.)
    by wrapping the layers instead of reimplementing them.
    """
    
    def __init__(self, config: DualStreamConfig, build_scratch: bool = True):
        super().__init__()
        self.config = config
        
        # These will be populated by from_pretrained_with_surgery
        self.embed_tokens = None
        self.layers = nn.ModuleList()
        self.norm = None
        self.lm_head = None
        self.rotary_emb = None
        self.rotary_emb_local = None  # For Gemma 3
        
        # Live state encoder (new component)
        self.live_encoder = LiveStateEncoder(
            hidden_dim=config.hidden_dim,
            input_dim=config.live_encoder_dim,
            num_signal_slots=config.num_signal_slots,
        )
        
        self.live_layers = set(config.inject_live_at_layers)
        
        # Initialize base components for scratch models when requested.
        if build_scratch:
            self._init_scratch_components()
        
        # Initialize weights of new components
        self.apply(self._init_weights)
    
    def _init_scratch_components(self):
        """Build a minimal working transformer when not loading from HF weights."""
        if self.embed_tokens is None:
            self.embed_tokens = DualStreamEmbedding(self.config)
        
        if not self.layers:
            base_layers = []
            for i in range(self.config.num_layers):
                layer = DualStreamDecoderLayer(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    intermediate_dim=self.config.intermediate_dim,
                    max_position=self.config.max_position,
                    rope_theta=self.config.rope_theta,
                    use_rope=self.config.use_rope,
                    use_adaptive_gate=self.config.use_adaptive_gate,
                    dropout=self.config.dropout,
                )
                base_layers.append(layer)
            self.layers = nn.ModuleList(base_layers)
        
        if self.norm is None:
            self.norm = RMSNorm(self.config.hidden_dim, eps=self.config.rms_norm_eps)
        
        if self.lm_head is None:
            self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)
            if self.config.tie_word_embeddings and isinstance(self.embed_tokens, nn.Embedding):
                self.lm_head.weight = self.embed_tokens.weight
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights for new components."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        live_state: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len = input_ids.shape
        past_key_values_length = 0
        if past_key_values and len(past_key_values) > 0 and past_key_values[0] is not None:
            try:
                pkv0 = past_key_values[0]
                # past_key_values structure: tuple(key, value, ...) -> use key seq length
                past_key_values_length = pkv0[0].shape[-2]
            except Exception:
                past_key_values_length = 0
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare 4D attention mask
        # Qwen3 expects [batch, 1, seq, seq] or [batch, heads, seq, seq]
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_len),
                hidden_states,
                past_key_values_length,  # account for cached tokens
                sliding_window=self.config.sliding_window
            )
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Rotary Embeddings
        position_embeddings = None
        position_embeddings_local = None
        if self.rotary_emb is not None:
            # Compute cos, sin (global)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.rotary_emb_local is not None:
            # Compute local position embeddings (Gemma 3 specific)
            position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # Layers
        new_cache = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            layer_live_state = live_state if idx in self.live_layers else None
            layer_past = None
            if past_key_values is not None and idx < len(past_key_values):
                layer_past = past_key_values[idx]
            
            # Wrapper expects: hidden_states, live_state, plus kwargs for original layer
            # Original layer expects: hidden_states, attention_mask, position_ids, ...
            
            layer_outputs = layer(
                hidden_states,
                live_state=layer_live_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                position_embeddings_local=position_embeddings_local,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if use_cache and len(layer_outputs) > 1:
                    new_cache.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        # Head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "past_key_values": new_cache,
            }
        return logits
    
    def generate_with_live_state(
        self,
        input_ids: torch.Tensor,
        live_buffer_manager: LiveBufferManager,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens while continuously attending to live state.
        """
        stop_tokens = stop_tokens or []
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            live_state = live_buffer_manager.get_current_state()
            
            outputs = self.forward(
                generated,
                live_state=live_state,
                return_dict=True,
            )
            
            next_logits = outputs["logits"][:, -1, :]
            
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() in stop_tokens:
                break
        
        return generated

    def get_layer_gate_info(self) -> Dict[int, dict]:
        info = {}
        for idx in self.live_layers:
            # layer is a DualStreamDecoderLayerWrapper
            if hasattr(self.layers[idx], 'live_cross_attn'):
                info[idx] = self.layers[idx].live_cross_attn.get_gate_info()
        return info

    def get_input_embeddings(self) -> nn.Embedding:
        # Handle wrappers vs nn.Embedding
        if isinstance(self.embed_tokens, nn.Embedding):
            return self.embed_tokens
        elif hasattr(self.embed_tokens, 'embed_tokens'):
            return self.embed_tokens.embed_tokens
        return self.embed_tokens

    @classmethod
    def from_pretrained_with_surgery(
        cls,
        model_path: str = "Qwen/Qwen3-4B",
        config: Optional[DualStreamConfig] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> "DualStreamTransformer":
        
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError("transformers required")

        logger.info(f"Loading pretrained model from {model_path}")
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load original model
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["device_map"] = "auto"
            
        original_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        # Config setup
        auto_config = DualStreamConfig.from_hf_config(hf_config)

        if config is None:
            config = auto_config
        else:
            # Keep caller-provided options (e.g., gate_init) but sync core sizes
            config.hidden_dim = auto_config.hidden_dim
            config.intermediate_dim = auto_config.intermediate_dim
            config.num_layers = auto_config.num_layers
            config.num_heads = auto_config.num_heads
            config.num_kv_heads = auto_config.num_kv_heads
            config.vocab_size = auto_config.vocab_size
            config.max_position = auto_config.max_position
            config.rope_theta = auto_config.rope_theta
            config.sliding_window = auto_config.sliding_window
            config.base_model = auto_config.base_model
            if config.inject_live_at_layers is None:
                config.inject_live_at_layers = auto_config.inject_live_at_layers

        # Create our model container
        dual_model = cls(config, build_scratch=False)
        dual_model = dual_model.to(dtype=torch_dtype)
        
        # SURGERY: Steal parts from original model
        
        # 1. Embeddings
        if hasattr(original_model, "model"):
            orig_inner = original_model.model
        else:
            orig_inner = original_model
            
        dual_model.embed_tokens = orig_inner.embed_tokens
        
        # 2. Layers (Wrap them!)
        orig_layers = orig_inner.layers
        new_layers = nn.ModuleList()
        
        logger.info(f"Wrapping {len(orig_layers)} layers...")
        
        for i, layer in enumerate(orig_layers):
            if i in config.inject_live_at_layers:
                # Wrap with dual stream capability
                wrapper = DualStreamDecoderLayerWrapper(
                    original_layer=layer,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    inject_after_self_attn=True,
                    gate_init=config.gate_init,
                )
                # Move new params to correct device/dtype
                wrapper.to(device=layer.parameters().__next__().device, dtype=torch_dtype)
                new_layers.append(wrapper)
            else:
                # Wrap simple layer to handle kwargs
                wrapper = DualStreamDecoderLayerWrapper(
                    original_layer=layer,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    gate_init=config.gate_init,
                )
                # We can disable injection by setting inject_after_self_attn=False or via logic
                wrapper.to(device=layer.parameters().__next__().device, dtype=torch_dtype)
                new_layers.append(wrapper)

        dual_model.layers = new_layers
        
        # 3. Final Norm
        dual_model.norm = orig_inner.norm
        
        # 4. Rotary Embedding
        if hasattr(orig_inner, "rotary_emb"):
            dual_model.rotary_emb = orig_inner.rotary_emb
        # Gemma 3 has both global and local rotary embeddings
        if hasattr(orig_inner, "rotary_emb_local"):
            dual_model.rotary_emb_local = orig_inner.rotary_emb_local
        
        # 5. LM Head
        dual_model.lm_head = original_model.lm_head
        
        # Move to device
        dual_model.live_encoder.to(device=device, dtype=torch_dtype)
        
        logger.info("Surgery complete: Original layers wrapped.")
        return dual_model
