"""
Live Inference Engine - Real-Time Token Generation with Live Injection

This module handles the actual inference loop where external signals
can be injected mid-generation.

Key Feature: The live buffer is checked at EVERY token step, allowing
immediate response to external events even mid-sentence.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass
import threading
import time
import queue
import logging
import inspect

from .encoder import LiveStateEncoder, LiveBufferManager
from .model import DualStreamTransformer, DualStreamConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_tokens: List[int] = None
    stream: bool = True


@dataclass
class LiveSessionConfig:
    """Configuration for a live mode session."""
    duration_seconds: float = 600.0  # Default 10 minutes
    auto_extend: bool = False  # Auto-extend on activity
    extend_on_signal: float = 60.0  # Seconds to extend when signal received
    max_duration_seconds: float = 3600.0  # Hard cap at 1 hour


class LiveSession:
    """
    A time-bounded live mode session.
    
    Users request a session for X minutes. During that window:
    - Live signals are actively processed
    - Cross-attention gates are open
    - Model reacts to real-time data
    
    After expiry:
    - Signals are ignored
    - Model reverts to standard mode
    - Session must be renewed
    
    Example:
        # Start a 10-minute live session
        session = LiveSession(engine, duration_minutes=10)
        
        with session:
            # All generation in here has live mode active
            response = engine.generate("Monitor the system...")
            
            # Post signals anytime during the session
            session.post_signal({"temp": 98.6})
        
        # Session expired or exited - back to normal mode
    """
    
    def __init__(
        self,
        engine: "LiveInferenceEngine",
        duration_minutes: float = 10.0,
        config: Optional[LiveSessionConfig] = None,
    ):
        self.engine = engine
        self.config = config or LiveSessionConfig()
        self.config.duration_seconds = duration_minutes * 60.0
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._is_active = False
        self._lock = threading.Lock()
        
        # Session stats
        self.signals_received = 0
        self.tokens_generated = 0
        self.extensions_granted = 0
    
    @property
    def is_active(self) -> bool:
        """Check if session is still valid."""
        with self._lock:
            if not self._is_active:
                return False
            if time.time() > self._end_time:
                self._is_active = False
                logger.info(f"LiveSession expired after {self.config.duration_seconds}s")
                return False
            return True
    
    @property
    def time_remaining(self) -> float:
        """Seconds remaining in session."""
        if not self._is_active:
            return 0.0
        return max(0.0, self._end_time - time.time())
    
    @property
    def session_info(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "is_active": self.is_active,
            "time_remaining_seconds": self.time_remaining,
            "signals_received": self.signals_received,
            "tokens_generated": self.tokens_generated,
            "extensions_granted": self.extensions_granted,
            "start_time": self._start_time,
            "end_time": self._end_time,
        }
    
    def start(self):
        """Start the live session."""
        with self._lock:
            self._start_time = time.time()
            self._end_time = self._start_time + self.config.duration_seconds
            self._is_active = True
            
            # Start the live buffer manager
            self.engine.live_manager.start()
            
            logger.info(
                f"LiveSession started: {self.config.duration_seconds}s "
                f"(expires at {time.strftime('%H:%M:%S', time.localtime(self._end_time))})"
            )
    
    def stop(self):
        """End the session early."""
        with self._lock:
            self._is_active = False
            self.engine.live_manager.stop()
            self.engine.live_manager.clear()
            logger.info(f"LiveSession stopped. Stats: {self.session_info}")
    
    def extend(self, seconds: float = 60.0) -> bool:
        """Extend the session duration."""
        with self._lock:
            if not self._is_active:
                return False
            
            new_end = self._end_time + seconds
            max_end = self._start_time + self.config.max_duration_seconds
            
            if new_end > max_end:
                logger.warning(f"Extension denied: would exceed max duration")
                return False
            
            self._end_time = new_end
            self.extensions_granted += 1
            logger.info(f"Session extended by {seconds}s. New end: {time.strftime('%H:%M:%S', time.localtime(self._end_time))}")
            return True
    
    def post_signal(self, content: Any, priority: float = 0.5) -> bool:
        """Post a signal if session is active."""
        if not self.is_active:
            logger.warning("Cannot post signal: session expired")
            return False
        
        self.signals_received += 1
        self.engine.live_manager.post_signal(content, priority=priority)
        
        # Auto-extend on activity if configured
        if self.config.auto_extend and self.time_remaining < 60:
            self.extend(self.config.extend_on_signal)
        
        return True
    
    def post_alert(self, message: str) -> bool:
        """Post high-priority alert if session is active."""
        return self.post_signal(message, priority=0.9)
    
    def get_live_state(self) -> Optional[torch.Tensor]:
        """Get live state only if session is active."""
        if not self.is_active:
            return None
        return self.engine.live_manager.get_current_state()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


@dataclass
class GenerationState:
    """Tracks the current state of generation."""
    tokens_generated: int = 0
    total_tokens: int = 0
    is_complete: bool = False
    stopped_by_eos: bool = False
    stopped_by_live: bool = False
    live_interrupts: int = 0


class LiveInferenceEngine:
    """
    Inference engine for dual-stream models with real-time live injection.
    
    This engine manages the generation loop and ensures that:
    1. Live signals are checked at every token step
    2. High-priority signals can interrupt generation
    3. Generation can be paused/resumed based on external events
    
    Example:
        engine = LiveInferenceEngine(model, tokenizer, live_manager)
        
        # Start generation
        for token in engine.generate_stream("Write a story"):
            print(token, end="")
            
            # External system can post signals at any time
            live_manager.post_alert("URGENT: Stop writing!")
    """
    
    def __init__(
        self,
        model: DualStreamTransformer,
        tokenizer: Any,
        live_buffer_manager: LiveBufferManager,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.live_manager = live_buffer_manager
        self.device = device
        
        # Control flags
        self._pause_generation = threading.Event()
        self._stop_generation = threading.Event()
        self._interrupt_priority = 0.0
        
        # Callbacks
        self.on_token_generated: Optional[Callable] = None
        self.on_live_signal: Optional[Callable] = None
        self.on_interrupt: Optional[Callable] = None
    
    def _ensure_live_manager(self):
        """Start live buffer processing if the caller forgot to start it."""
        try:
            if hasattr(self.live_manager, "ensure_running"):
                self.live_manager.ensure_running()
            elif hasattr(self.live_manager, "start"):
                self.live_manager.start()
        except Exception:
            # Non-fatal: generation can proceed without live signals
            logger.debug("Unable to auto-start live manager", exc_info=True)
        
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template for Qwen3 thinking models."""
        messages = [{"role": "user", "content": prompt}]
        try:
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Disable thinking mode for cleaner output
            try:
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    apply_kwargs["enable_thinking"] = False  # Disable thinking for direct responses
            except (TypeError, ValueError):
                pass
            formatted = self.tokenizer.apply_chat_template(
                messages,
                **apply_kwargs,
            )
            return formatted
        except Exception:
            # Fallback for tokenizers without chat template
            return prompt
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text with live state awareness.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            
        Returns:
            Generated text
        """
        config = config or GenerationConfig()
        self._ensure_live_manager()
        
        # Format and tokenize input
        formatted_prompt = self._format_prompt(prompt)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        
        # Generate
        output_ids = self._generate_loop(input_ids, attention_mask, config)
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0, len(input_ids[0]):],
            skip_special_tokens=True,
        )
        
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, GenerationState]:
        """
        Stream generated tokens as they are produced.
        
        This is the preferred method for live-mode generation as it
        allows immediate visibility of each token.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            
        Yields:
            Individual tokens as strings
            
        Returns:
            Final GenerationState
        """
        config = config or GenerationConfig()
        self._ensure_live_manager()
        
        # Format and tokenize
        formatted_prompt = self._format_prompt(prompt)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        
        state = GenerationState(total_tokens=len(input_ids[0]))
        generated = input_ids.clone()
        
        self._stop_generation.clear()
        self._pause_generation.clear()
        
        for step in range(config.max_new_tokens):
            # Check for stop signal
            if self._stop_generation.is_set():
                state.is_complete = True
                break
            
            # Check for pause
            while self._pause_generation.is_set():
                time.sleep(0.01)
            
            # Get live state (None if no signals in buffer)
            live_state = self.live_manager.get_current_state()
            if live_state is not None:
                live_state = live_state.to(self.device)
            
            # Check for high-priority interrupt
            if self._should_interrupt(live_state):
                state.live_interrupts += 1
                state.stopped_by_live = True
                if self.on_interrupt:
                    self.on_interrupt(live_state)
                # Emit an interrupt marker into the stream for visibility
                interrupt_text = self._emit_interrupt_text()
                yield interrupt_text
                break
            
            # Forward pass
            # NOTE: KV cache disabled due to compatibility issues with wrapped layers
            with torch.no_grad():
                outputs = self.model(
                    generated,  # Full sequence (no KV cache)
                    live_state=live_state,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            
            # Sample next token
            next_token = self._sample_token(
                outputs["logits"][:, -1, :],
                generated,
                config,
            )
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            state.tokens_generated += 1
            state.total_tokens += 1
            
            # Decode and yield
            token_text = self.tokenizer.decode(
                next_token[0],
                skip_special_tokens=True,
            )
            
            if self.on_token_generated:
                self.on_token_generated(token_text, state)
            
            yield token_text
            
            # Check for EOS
            if config.stop_tokens and next_token.item() in config.stop_tokens:
                state.stopped_by_eos = True
                state.is_complete = True
                break
        
        state.is_complete = True
        return state
    
    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Core generation loop."""
        generated = input_ids.clone()
        attn_mask = attention_mask.clone()
        past_key_values = None
        
        for step in range(config.max_new_tokens):
            # Get live state AT EVERY STEP (None if no signals)
            live_state = self.live_manager.get_current_state()
            if live_state is not None:
                live_state = live_state.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                model_input = generated if past_key_values is None else generated[:, -1:]
                outputs = self.model(
                    model_input,
                    live_state=live_state,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.get("past_key_values", None)
            
            # Sample
            next_token = self._sample_token(
                outputs["logits"][:, -1, :],
                generated,
                config,
            )
            
            generated = torch.cat([generated, next_token], dim=-1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((attn_mask.shape[0], 1), device=self.device, dtype=attn_mask.dtype)],
                dim=1,
            )
            
            # Check stop conditions
            if config.stop_tokens and next_token.item() in config.stop_tokens:
                break
        
        return generated
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Sample next token with various strategies."""
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                logits[0, token_id] /= config.repetition_penalty
        
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Apply top-k
        if config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus sampling)
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _should_interrupt(self, live_state: torch.Tensor) -> bool:
        """Check if generation should be interrupted by live signal."""
        if live_state is None:
            return False
        # Use buffer priority as a proxy for urgency
        max_priority = 0.0
        try:
            max_priority = self.live_manager.get_max_priority()
        except Exception:
            max_priority = 0.0
        
        # Allow external overrides via _interrupt_priority (defaults to 0.0)
        threshold = max(self._interrupt_priority, 0.9)
        return max_priority >= threshold
    
    def _emit_interrupt_text(self) -> str:
        """Return a human-readable interrupt message based on last signal."""
        last_text = None
        try:
            last_text = self.live_manager.get_last_signal_text()
        except Exception:
            last_text = None
        if last_text:
            return f"[LIVE-INTERRUPT] {last_text}"
        return "[LIVE-INTERRUPT]"
    
    def pause(self):
        """Pause generation (can be resumed)."""
        self._pause_generation.set()
    
    def resume(self):
        """Resume paused generation."""
        self._pause_generation.clear()
    
    def stop(self):
        """Stop generation completely."""
        self._stop_generation.set()
    
    def reset(self):
        """Reset all control flags."""
        self._pause_generation.clear()
        self._stop_generation.clear()
        self._interrupt_priority = 0.0


class InterruptibleGenerationContext:
    """
    Context manager for interruptible generation sessions.
    
    Example:
        async with InterruptibleGenerationContext(engine) as ctx:
            # Start generation in background
            ctx.start("Tell me about the weather")
            
            # At any point, inject a live signal
            ctx.inject_signal({"alert": "Temperature dropping!"})
            
            # Get generated text
            result = await ctx.get_result()
    """
    
    def __init__(self, engine: LiveInferenceEngine):
        self.engine = engine
        self._generation_thread = None
        self._result_queue = queue.Queue()
        self._error = None
        
    def __enter__(self):
        self.engine.live_manager.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.stop()
        self.engine.live_manager.stop()
        return False
    
    def start(self, prompt: str, config: Optional[GenerationConfig] = None):
        """Start generation in a background thread."""
        def _run():
            try:
                result = self.engine.generate(prompt, config)
                self._result_queue.put(("result", result))
            except Exception as e:
                self._result_queue.put(("error", e))
        
        self._generation_thread = threading.Thread(target=_run, daemon=True)
        self._generation_thread.start()
    
    def inject_signal(
        self,
        content: Dict[str, Any],
        priority: float = 0.5,
    ):
        """Inject a live signal during generation."""
        self.engine.live_manager.post_signal(content, priority=priority)
    
    def inject_alert(self, message: str):
        """Inject a high-priority alert."""
        self.engine.live_manager.post_alert(message)
    
    def get_result(self, timeout: float = None) -> str:
        """Wait for and return the generation result."""
        try:
            result_type, result = self._result_queue.get(timeout=timeout)
            if result_type == "error":
                raise result
            return result
        except queue.Empty:
            return None


class LiveStateInjector:
    """
    Utility for injecting live states from external sources.
    
    This can be used to bridge external systems (sensors, APIs, etc.)
    to the live buffer.
    """
    
    def __init__(self, live_manager: LiveBufferManager):
        self.live_manager = live_manager
        self._sources = {}
        self._running = False
        
    def register_source(
        self,
        name: str,
        poll_fn: Callable[[], Optional[Dict[str, Any]]],
        interval: float = 0.1,
        priority: float = 0.5,
    ):
        """
        Register an external source that will be polled for signals.
        
        Args:
            name: Unique name for this source
            poll_fn: Function that returns signal data or None
            interval: Polling interval in seconds
            priority: Default priority for signals from this source
        """
        self._sources[name] = {
            "poll_fn": poll_fn,
            "interval": interval,
            "priority": priority,
            "thread": None,
        }
    
    def start(self):
        """Start polling all registered sources."""
        self._running = True
        
        for name, source in self._sources.items():
            thread = threading.Thread(
                target=self._poll_source,
                args=(name, source),
                daemon=True,
            )
            source["thread"] = thread
            thread.start()
    
    def stop(self):
        """Stop all polling."""
        self._running = False
        
        for source in self._sources.values():
            if source["thread"]:
                source["thread"].join(timeout=1.0)
    
    def _poll_source(self, name: str, source: Dict):
        """Poll a single source."""
        while self._running:
            try:
                data = source["poll_fn"]()
                if data is not None:
                    self.live_manager.post_signal(
                        data,
                        priority=source["priority"],
                    )
            except Exception as e:
                logger.warning(f"Error polling source {name}: {e}")
            
            time.sleep(source["interval"])
