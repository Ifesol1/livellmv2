"""
Dual-Stream Architecture - Cross-Attention Signal Injection

Two approaches available:

1. SOFT PROMPTING (Recommended - No Training):
   Uses model's own embeddings + cross-attention gates.
   Signal meaning injected "subconsciously".
   
   from dual_stream_arch import SoftPromptGenerator

2. FULL ARCHITECTURE (Requires Training):
   Custom encoder + modified transformer layers.
   More powerful but needs training.
   
   from dual_stream_arch import DualStreamTransformer
"""

# Soft Prompting (recommended - no training)
from .native_encoder import NativeSignalEncoder, NativeSignalBuffer, create_native_encoder
from .soft_prompt_generator import SoftPromptGenerator, SoftPromptInjector

# Full architecture (requires training)
from .encoder import LiveStateEncoder, LiveBufferManager
from .attention import GatedCrossAttention, RoPEGatedCrossAttention, AdaptiveGate
from .decoder_layer import DualStreamDecoderLayer, DualStreamDecoderLayerWrapper
from .model import DualStreamTransformer, DualStreamConfig
from .inference import LiveInferenceEngine, LiveSession

__all__ = [
    # Soft Prompting (no training)
    "SoftPromptGenerator",
    "SoftPromptInjector", 
    "NativeSignalEncoder",
    "NativeSignalBuffer",
    "create_native_encoder",
    
    # Full architecture (training required)
    "LiveStateEncoder",
    "LiveBufferManager",
    "GatedCrossAttention",
    "RoPEGatedCrossAttention",
    "AdaptiveGate",
    "DualStreamDecoderLayer",
    "DualStreamDecoderLayerWrapper",
    "DualStreamTransformer",
    "DualStreamConfig",
    "LiveInferenceEngine",
    "LiveSession",
]
