"""
Tool definitions and execution for LLM tool calling.

Only essential tools — kept minimal for fast invocation.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    name: str
    arguments: Dict[str, Any]
    raw_text: str


@dataclass  
class ToolResult:
    """Result from executing a tool."""
    name: str
    result: Any
    success: bool
    error: Optional[str] = None


# Tool registry
TOOLS: Dict[str, Dict] = {}
TOOL_FUNCTIONS: Dict[str, Callable] = {}


def register_tool(name: str, description: str, parameters: Dict):
    """Decorator to register a tool function."""
    def decorator(func: Callable):
        TOOLS[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        TOOL_FUNCTIONS[name] = func
        return func
    return decorator


# ============== Tools ==============

@register_tool(
    name="call_911",
    description="Call 911 emergency services immediately. ONLY use for CRITICAL readings: heart rate below 40 BPM or above 170 BPM. No arguments needed.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)
def call_911(**kwargs) -> dict:
    """Simulate calling 911 emergency services. No args required."""
    return {
        "action": "CALLING_911",
        "status": "EMERGENCY_DISPATCHED",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": "911 called. Emergency services dispatched.",
    }


# ============== Tool Call Parser ==============

class ToolCallParser:
    """
    Parse tool calls from Qwen3 model output.
    
    Qwen3 uses this format:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "value1"}}
    </tool_call>
    """
    
    # Patterns for different tool call formats
    PATTERNS = [
        # Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL),
        # Alternative: ```json ... ```
        re.compile(r'```(?:json)?\s*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\})\s*```', re.DOTALL),
    ]
    
    # Catch malformed tool calls like <call_911>{...}</call_911>
    MALFORMED_TAG_RE = re.compile(
        r'<(\w+)>\s*(\{.*?\})\s*</\1>', re.DOTALL
    )
    # Catch bare tags with no body: <call_911></call_911> or <call_911/>
    BARE_TAG_RE = re.compile(
        r'<(\w+)>\s*</\1>|<(\w+)\s*/?>'
    )
    # Catch: call_911({"reason": "..."}) or call_911()
    FUNC_CALL_RE = re.compile(
        r'(?:^|\s)(\w+)\(\s*(\{.*?\})?\s*\)', re.DOTALL
    )
    
    @classmethod
    def parse(cls, text: str) -> List[ToolCall]:
        """Extract tool calls from model output."""
        calls = []
        
        # Try Qwen3/JSON format first
        for pattern in cls.PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    data = json.loads(match)
                    calls.append(ToolCall(
                        name=data.get("name", ""),
                        arguments=data.get("arguments", {}),
                        raw_text=match
                    ))
                except json.JSONDecodeError:
                    continue
        
        if calls:
            return calls
        
        # Try malformed tag format: <tool_name>{"key": "val"}</tool_name>
        for tag_name, json_str in cls.MALFORMED_TAG_RE.findall(text):
            if tag_name in TOOL_FUNCTIONS:
                try:
                    data = json.loads(json_str)
                    # data might be {"name":..., "arguments":...} OR direct args
                    if "name" in data and "arguments" in data:
                        calls.append(ToolCall(
                            name=data["name"],
                            arguments=data["arguments"],
                            raw_text=json_str
                        ))
                    else:
                        calls.append(ToolCall(
                            name=tag_name,
                            arguments=data,
                            raw_text=json_str
                        ))
                except json.JSONDecodeError:
                    continue
        
        if calls:
            return calls
        
        # Try bare tags with no JSON body: <call_911></call_911> or <call_911/>
        for m in cls.BARE_TAG_RE.finditer(text):
            tag_name = m.group(1) or m.group(2)
            if tag_name and tag_name in TOOL_FUNCTIONS:
                calls.append(ToolCall(
                    name=tag_name,
                    arguments={},
                    raw_text=m.group(0)
                ))
        
        if calls:
            return calls
        
        # Try function-call style: tool_name({}) or tool_name()
        for func_name, json_str in cls.FUNC_CALL_RE.findall(text):
            if func_name in TOOL_FUNCTIONS:
                args = {}
                if json_str:
                    try:
                        args = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                calls.append(ToolCall(
                    name=func_name,
                    arguments=args,
                    raw_text=json_str or ''
                ))
        
        return calls
    
    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        """Check if text contains a tool call."""
        if '<tool_call>' in text or any(p.search(text) for p in cls.PATTERNS):
            return True
        if cls.MALFORMED_TAG_RE.search(text):
            return True
        # Check bare tags
        for m in cls.BARE_TAG_RE.finditer(text):
            tag = m.group(1) or m.group(2)
            if tag and tag in TOOL_FUNCTIONS:
                return True
        if cls.FUNC_CALL_RE.search(text):
            return True
        return False
    
    @classmethod
    def extract_partial_tool_call(cls, text: str) -> Optional[str]:
        """Detect if a tool call is being formed (for streaming)."""
        if '<tool_call>' in text and '</tool_call>' not in text:
            # Tool call in progress
            start = text.rfind('<tool_call>')
            return text[start:]
        return None


# ============== Tool Executor ==============

class ToolExecutor:
    """Execute tool calls and format results."""
    
    @classmethod
    async def execute(cls, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        if tool_call.name not in TOOL_FUNCTIONS:
            return ToolResult(
                name=tool_call.name,
                result=None,
                success=False,
                error=f"Unknown tool: {tool_call.name}"
            )
        
        func = TOOL_FUNCTIONS[tool_call.name]
        try:
            # Handle both sync and async functions
            import asyncio
            if asyncio.iscoroutinefunction(func):
                result = await func(**tool_call.arguments)
            else:
                result = func(**tool_call.arguments)
            
            return ToolResult(
                name=tool_call.name,
                result=result,
                success=True
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                result=None,
                success=False,
                error=str(e)
            )
    
    @classmethod
    def format_result(cls, result: ToolResult) -> str:
        """Format tool result for injection into context."""
        if result.success:
            return f"<tool_response>\n{result.result}\n</tool_response>"
        else:
            return f"<tool_response>\nError: {result.error}\n</tool_response>"


def get_tools_prompt() -> str:
    """Generate a minimal tools description for the system prompt."""
    if not TOOLS:
        return ""
    
    return """You have one tool: call_911
To call 911, output exactly:
<tool_call>
{"name": "call_911", "arguments": {}}
</tool_call>
Only use this for CRITICAL emergencies. No arguments needed."""


def get_tools_schema() -> List[Dict]:
    """Get tools in OpenAI schema format."""
    return list(TOOLS.values())
