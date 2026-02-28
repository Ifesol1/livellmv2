"""
Live LLM - Simple Real-time Signal Injection

The simple, working approach: just append text to the generation stream.
No architecture changes, no training required.
Works with any HuggingFace causal LM.
"""

from .live_stream import LiveStreamGenerator, LiveSignal, SignalQueue

__all__ = ["LiveStreamGenerator", "LiveSignal", "SignalQueue"]
