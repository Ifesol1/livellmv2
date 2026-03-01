"""
LLM Service - Agno Agent + Ollama backend.

Agno handles tool calling, streaming, and conversation management.
"""

import asyncio
import json
import sys
import os
from typing import AsyncGenerator, Optional, List, Dict
from agno.agent import Agent
from agno.models.ollama import Ollama

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.tools import TOOL_LIST, get_tools_schema
from server.telemetry import telemetry


class LLMService:
    """
    Async LLM service powered by Agno + Ollama.
    
    - No manual tokenizer/model loading — Ollama manages the model process.
    - No manual tool parsing — Agno handles tool call detection + execution.
    - Streaming comes from Agno's native run_response_stream.
    """
    
    DEFAULT_INSTRUCTIONS = [
        "You are a helpful AI assistant with tool calling capabilities.",
        "IMPORTANT: Only use tools when the situation CLEARLY requires it.",
        "Do NOT call tools unless explicitly needed. Most responses need NO tool calls.",
        "When calling a tool, you MUST provide the current heart rate in the 'heart_rate' parameter.",
        "Be concise in your responses.",
    ]
    
    def __init__(self):
        self.agent: Optional[Agent] = None
        self.model_name: Optional[str] = None
        self.is_loaded: bool = False
        # Pending signals queued between generations
        self._pending_signals: List[str] = []
    
    def _create_agent(self, instructions: Optional[List[str]] = None) -> Agent:
        """Create a fresh Agno Agent with the given instructions."""
        return Agent(
            model=Ollama(id=self.model_name),
            tools=TOOL_LIST,
            markdown=True,
            stream_events=True,
            instructions=instructions or self.DEFAULT_INSTRUCTIONS,
        )
    
    async def load_model(self, model_name: str = "llama3.2") -> bool:
        """
        'Load' a model by creating an Agno Agent pointed at Ollama.
        
        Ollama must be running (`ollama serve`). The model_name is the
        Ollama model tag, e.g. 'llama3.2', 'qwen3', 'mistral'.
        """
        if self.is_loaded and self.model_name == model_name:
            return True
        
        print(f"[Agno] Creating agent with Ollama model: {model_name}")
        
        self.model_name = model_name
        self.agent = self._create_agent()
        self.is_loaded = True
        print(f"[Agno] Agent ready with model: {model_name}")
        return True
    
    def create_session(self, system_prompt: Optional[str] = None):
        """
        Create a fresh Agent session, resetting conversation history.
        
        If system_prompt is provided, it becomes the Agent's instructions.
        This ensures the system prompt is a proper system message, not user content.
        """
        if not self.model_name:
            return
        
        if system_prompt:
            instructions = [system_prompt]
        else:
            instructions = self.DEFAULT_INSTRUCTIONS
        
        self.agent = self._create_agent(instructions)
        self._pending_signals.clear()
        print(f"[Agno] New session created (instructions: {len(instructions[0])} chars)", flush=True)
    
    def post_signal(self, content: str, priority: float = 1.0):
        """Queue a live signal to be injected as context in the next generation."""
        print(f"[Agno] Signal queued: {content}")
        telemetry.track_signal(source="api", priority=priority)
        self._pending_signals.append(content)
    
    def _build_prompt_with_signals(self, prompt: str) -> str:
        """Prepend any pending signals to the prompt, then clear the queue."""
        if not self._pending_signals:
            return prompt
        
        signals_text = "\n".join(
            f"[LIVE SIGNAL] {s}" for s in self._pending_signals
        )
        self._pending_signals.clear()
        return f"{signals_text}\n\n{prompt}"
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens asynchronously (plain text, no tool support).
        """
        if not self.is_loaded or not self.agent:
            yield "[Error: Model not loaded]"
            return
        
        full_prompt = self._build_prompt_with_signals(prompt)
        
        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        
        def _run():
            try:
                response_stream = self.agent.run(full_prompt, stream=True)
                for chunk in response_stream:
                    event_name = getattr(chunk, 'event', None)
                    # Only emit text content events
                    if event_name == 'RunContent' or (event_name is None and hasattr(chunk, 'content')):
                        if hasattr(chunk, 'content') and chunk.content:
                            loop.call_soon_threadsafe(token_queue.put_nowait, chunk.content)
            except Exception as e:
                print(f"[Agno] Stream error: {e}")
                loop.call_soon_threadsafe(token_queue.put_nowait, f"[Error: {e}]")
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # Sentinel
        
        import threading
        threading.Thread(target=_run, daemon=True).start()
        
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token
    
    async def generate_with_tools(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        enable_tools: bool = True,
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream tokens with automatic Agno tool calling.
        
        Yields dicts: { type: 'token' | 'tool_call' | 'tool_result' | 'done' }
        
        If system_prompt is provided, a fresh session is created with it
        as the Agent's instructions (proper system message placement).
        """
        if not self.is_loaded or not self.agent:
            yield {"type": "error", "content": "Model not loaded"}
            return
        
        # Create fresh session if system_prompt provided
        if system_prompt:
            self.create_session(system_prompt)
        
        full_prompt = self._build_prompt_with_signals(prompt)
        
        loop = asyncio.get_event_loop()
        event_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        
        def _run():
            # Strip tools when enable_tools=False so model cannot call them
            original_tools = None
            if not enable_tools and self.agent:
                original_tools = self.agent.tools
                self.agent.tools = []
                print("[Agno] Tools disabled for this generation", flush=True)
            
            try:
                response_stream = self.agent.run(full_prompt, stream=True)
                for chunk in response_stream:
                    event_name = getattr(chunk, 'event', None)
                    
                    # Tool call started
                    if event_name == 'ToolCallStarted':
                        tool = getattr(chunk, 'tool', None)
                        if tool:
                            tool_name = getattr(tool, 'tool_name', 'unknown')
                            tool_args = getattr(tool, 'tool_args', {}) or {}
                            print(f"[Agno] Tool call started: {tool_name}", flush=True)
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "tool_call",
                                "name": tool_name,
                                "arguments": tool_args if isinstance(tool_args, dict) else {}
                            })
                    
                    # Tool call completed
                    elif event_name == 'ToolCallCompleted':
                        tool = getattr(chunk, 'tool', None)
                        if tool:
                            tool_name = getattr(tool, 'tool_name', 'unknown')
                            tool_result = getattr(tool, 'result', None)
                            tool_error = getattr(tool, 'tool_call_error', None)
                            
                            # Parse result string into dict if possible
                            result_data = tool_result
                            if isinstance(result_data, str):
                                try:
                                    result_data = json.loads(result_data.replace("'", '"'))
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            
                            success = not bool(tool_error)
                            print(f"[Agno] Tool call completed: {tool_name} success={success}", flush=True)
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "tool_result",
                                "name": tool_name,
                                "result": result_data,
                                "success": success,
                                "error": str(tool_error) if tool_error else None
                            })
                    
                    # Text content token
                    elif event_name == 'RunContent':
                        content = getattr(chunk, 'content', None)
                        if content:
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "token",
                                "content": content
                            })
                    
                    # Fallback: check for content on any event type
                    elif hasattr(chunk, 'content') and chunk.content and event_name not in (
                        'RunStarted', 'RunCompleted', 'RunContentCompleted',
                        'ModelRequestStarted', 'ModelRequestCompleted',
                    ):
                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                            "type": "token",
                            "content": chunk.content
                        })
                        
            except Exception as e:
                print(f"[Agno] Tool stream error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                loop.call_soon_threadsafe(event_queue.put_nowait, {
                    "type": "error",
                    "content": str(e)
                })
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)
        
        import threading
        threading.Thread(target=_run, daemon=True).start()
        
        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event
        
        yield {"type": "done"}
    
    def get_tools(self):
        """Get available tools schema."""
        return get_tools_schema()


# Singleton instance
llm_service = LLMService()
