"""
LLM Service - Wraps the LiveStreamGenerator for web use.
Includes tool calling support.
"""

import torch
import asyncio
import re
from typing import AsyncGenerator, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_llm import LiveStreamGenerator
from server.tools import (
    ToolCallParser, ToolExecutor, 
    get_tools_prompt, get_tools_schema,
    TOOL_FUNCTIONS
)
from server.telemetry import telemetry


# Regex to strip leaked special tokens from ANY model format
_SPECIAL_TOKEN_RE = re.compile(
    r'<\|im_start\|>'
    r'|<\|im_end\|>'
    r'|<\|im_sep\|>'
    r'|<\|start_header_id\|>'
    r'|<\|end_header_id\|>'
    r'|<\|eot_id\|>'
    r'|<\|begin_of_text\|>'
    r'|<\|end_of_text\|>'
    r'|<\|pad\|>'
    r'|<\|finetune_right_pad_id\|>'
    r'|<\|reserved_special_token_\d+\|>'
)

# Bare role markers that appear as standalone tokens after special token stripping
_ROLE_LINE_RE = re.compile(r'^\s*(user|assistant|system)\s*$')


def _clean_token(token: str) -> str:
    """Strip any leaked special tokens / role markers from a single token."""
    cleaned = _SPECIAL_TOKEN_RE.sub('', token)
    if _ROLE_LINE_RE.match(cleaned):
        return ''
    return cleaned


class LLMService:
    """
    Async wrapper for LiveStreamGenerator.
    
    Handles model loading and provides async streaming interface.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_name = None
        self.is_loaded = False
        
    async def load_model(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct") -> bool:
        """Load a model asynchronously."""
        if self.is_loaded and self.model_name == model_name:
            return True
            
        print(f"Loading model: {model_name}...")
        
        # Run model loading in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, model_name)
        
        print(f"Model loaded: {model_name}")
        return True
    
    def _load_model_sync(self, model_name: str):
        """Synchronous model loading."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.generator = LiveStreamGenerator(
            self.model, 
            self.tokenizer,
            signal_style="user"  # Signals look like user follow-up messages
        )
        
        self.model_name = model_name
        self.is_loaded = True
    
    def post_signal(self, content: str, priority: float = 1.0):
        """Post a live signal."""
        print(f"[DEBUG-SERVICE] Posting signal: {content}")
        
        # Telemetry
        telemetry.track_signal(source="api", priority=priority)
        
        if self.generator:
            self.generator.post_signal(content, priority)
        else:
            print("[DEBUG-SERVICE] Generator is None, cannot post signal")
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens asynchronously.
        
        Yields tokens as they are generated.
        """
        if not self.is_loaded:
            yield "[Error: Model not loaded]"
            return
        
        # Run synchronous generator in executor
        loop = asyncio.get_event_loop()
        
        # Create a queue for cross-thread communication
        token_queue = asyncio.Queue()
        done_event = asyncio.Event()
        
        def generate_tokens():
            try:
                for token in self.generator.generate_stream(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    # Put token in queue (thread-safe with asyncio)
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(done_event.set)
        
        # Start generation in background thread
        import threading
        thread = threading.Thread(target=generate_tokens, daemon=True)
        thread.start()
        print("[DEBUG-SERVICE] Generation thread started")
        
        # Yield tokens as they arrive
        while not done_event.is_set() or not token_queue.empty():
            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                cleaned = _clean_token(token)
                if cleaned:
                    yield cleaned
            except asyncio.TimeoutError:
                continue
        
        print("[DEBUG-SERVICE] Generation finished or queue drained")
        # Drain any remaining tokens
        while not token_queue.empty():
            token = await token_queue.get()
            cleaned = _clean_token(token)
            if cleaned:
                yield cleaned
    
    async def generate_with_tools(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        enable_tools: bool = True,
        enable_thinking: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream tokens with tool calling support.
        
        Yields dicts with type: 'token', 'tool_call', 'tool_result', 'done'
        
        When a tool call is detected:
        1. Yield the tool call info
        2. Execute the tool
        3. Continue with a new generation that includes the tool result
        """
        if not self.is_loaded:
            yield {"type": "error", "content": "Model not loaded"}
            return
        
        # Build prompt with tools context
        if enable_tools:
            tools_prompt = get_tools_prompt()
            current_prompt = f"{tools_prompt}\n\nUser: {prompt}"
        else:
            current_prompt = prompt
        
        max_tool_iterations = 2  # Usually: 1st = tool call, 2nd = final answer
        total_tokens_used = 0
        tools_already_used = False  # Track if we've already used a tool
        
        for iteration in range(max_tool_iterations):
            # Buffer for detecting tool calls
            buffer = ""
            display_buffer = ""  # What we show to user (no tool tags)
            tool_call_in_progress = False
            tool_was_called = False
            tool_result_text = ""
            pending_display = ""  # Tokens waiting to be displayed
            
            loop = asyncio.get_event_loop()
            token_queue = asyncio.Queue()
            done_event = asyncio.Event()
            
            iteration_prompt = current_prompt  # Capture for thread
            
            def generate_tokens():
                try:
                    print(f"[GEN] Starting generation, prompt length: {len(iteration_prompt)}", flush=True)
                    token_count = 0
                    for token in self.generator.generate_stream(
                        iteration_prompt,
                        max_tokens=max_tokens - total_tokens_used,
                        temperature=temperature
                    ):
                        token_count += 1
                        if token_count <= 3:
                            print(f"[GEN] Token {token_count}: '{token}'", flush=True)
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
                    print(f"[GEN] Finished, generated {token_count} tokens", flush=True)
                except Exception as e:
                    print(f"[GEN] ERROR in generation: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                finally:
                    loop.call_soon_threadsafe(done_event.set)
            
            import threading
            thread = threading.Thread(target=generate_tokens, daemon=True)
            thread.start()
            print(f"[GEN] Thread started", flush=True)
            
            # Detect if buffer contains start of any tool call pattern
            def _has_open_tool_tag(buf: str) -> bool:
                """Check if buffer has an opening tool tag without a matching close."""
                if '<tool_call>' in buf and '</tool_call>' not in buf:
                    return True
                # Check for malformed tags like <call_911> without </call_911>
                import re as _re
                for m in _re.finditer(r'<(\w+)>', buf):
                    tag = m.group(1)
                    if tag in TOOL_FUNCTIONS and f'</{tag}>' not in buf:
                        return True
                return False
            
            def _has_complete_tool_call(buf: str) -> bool:
                """Check if buffer has a complete (closed) tool call."""
                if '</tool_call>' in buf:
                    return True
                import re as _re
                for m in _re.finditer(r'<(\w+)>', buf):
                    tag = m.group(1)
                    if tag in TOOL_FUNCTIONS and f'</{tag}>' in buf:
                        return True
                return False
            
            def _split_after_tool(buf: str) -> str:
                """Return text after the last closing tool tag."""
                if '</tool_call>' in buf:
                    return buf.split('</tool_call>')[-1]
                import re as _re
                # Find the last closing malformed tag
                last_end = 0
                for m in _re.finditer(r'</(\w+)>', buf):
                    if m.group(1) in TOOL_FUNCTIONS:
                        last_end = m.end()
                return buf[last_end:] if last_end else buf
            
            def _could_be_tool_tag_start(text: str) -> bool:
                """Check if text ends with a prefix that could become a tool tag."""
                candidates = ['<tool_call>']
                for tname in TOOL_FUNCTIONS:
                    candidates.append(f'<{tname}>')
                for candidate in candidates:
                    for plen in range(1, len(candidate)):
                        if text.endswith(candidate[:plen]):
                            return True
                return False
            
            pending_raw = ""
            pending_clean_tokens = []
            
            while not done_event.is_set() or not token_queue.empty():
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                    buffer += token
                    total_tokens_used += 1
                    
                    # Check for tool call start (skip if tools already used this session)
                    if _has_open_tool_tag(buffer) and not _has_complete_tool_call(buffer) and not tools_already_used:
                        if not tool_call_in_progress:
                            # Discard pending tokens — they were part of the tool tag
                            pending_raw = ""
                            pending_clean_tokens = []
                        tool_call_in_progress = True
                    
                    # Check for complete tool call
                    if tool_call_in_progress and _has_complete_tool_call(buffer) and not tools_already_used:
                        tool_call_in_progress = False
                        pending_raw = ""
                        pending_clean_tokens = []
                        
                        # Parse and execute tool call
                        tool_calls = ToolCallParser.parse(buffer)
                        
                        if tool_calls:
                            for tc in tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                }
                                
                                result = await ToolExecutor.execute(tc)
                                
                                yield {
                                    "type": "tool_result",
                                    "name": result.name,
                                    "result": result.result,
                                    "success": result.success,
                                    "error": result.error
                                }
                                
                                tool_was_called = True
                                tools_already_used = True
                                tool_result_text = ToolExecutor.format_result(result)
                            
                            buffer = _split_after_tool(buffer)
                            display_buffer = ""
                        
                    # Buffer token if not in a confirmed tool call
                    elif not tool_call_in_progress:
                        cleaned = _clean_token(token)
                        skip_token = (
                            not cleaned or
                            '<tool' in cleaned or '</tool' in cleaned or 
                            'tool_call' in cleaned.lower()
                        )
                        if not skip_token:
                            for tname in TOOL_FUNCTIONS:
                                if f'<{tname}' in cleaned or f'</{tname}' in cleaned:
                                    skip_token = True
                                    break
                        
                        pending_raw += token
                        if not skip_token:
                            pending_clean_tokens.append(cleaned)
                        
                        # If pending could still become a tool tag, hold tokens
                        if not _could_be_tool_tag_start(pending_raw):
                            # Confirmed not a tool tag — flush all pending
                            for ct in pending_clean_tokens:
                                yield {"type": "token", "content": ct}
                            pending_raw = ""
                            pending_clean_tokens = []
                        
                except asyncio.TimeoutError:
                    continue
            
            # Flush any remaining held tokens after generation ends
            for ct in pending_clean_tokens:
                yield {"type": "token", "content": ct}
            pending_raw = ""
            pending_clean_tokens = []
            
            # If a tool was called, continue with a new prompt including the result
            if tool_was_called:
                # Build clean continuation - just add tool result, model continues naturally
                # Remove any incomplete content after tool call
                clean_buffer = buffer.strip()
                if clean_buffer:
                    # Only include meaningful text before tool call
                    current_prompt = f"{current_prompt}\n\nTool result: {tool_result_text.replace('<tool_response>', '').replace('</tool_response>', '').strip()}\n\nBased on the tool result above, provide your final answer:"
                else:
                    current_prompt = f"{current_prompt}\n\nTool result: {tool_result_text.replace('<tool_response>', '').replace('</tool_response>', '').strip()}\n\nBased on the tool result above, provide your final answer:"
                print(f"[TOOLS] Continuing after tool call, iteration {iteration + 1}")
            else:
                # No tool called, we're done
                break
        
        yield {"type": "done"}
    
    def get_tools(self):
        """Get available tools schema."""
        return get_tools_schema()


# Singleton instance
llm_service = LLMService()
