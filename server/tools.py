"""
Tool definitions for LLM tool calling via Agno.

Plain Python functions — Agno handles discovery, calling, and result injection.
"""

from datetime import datetime
from typing import List, Dict, Optional


# ============== Tools ==============

def call_911(heart_rate: Optional[int] = None, **kwargs) -> str:
    """
    Call 911 emergency services immediately.
    
    Args:
        heart_rate: The patient's current heart rate in BPM (optional).
    """
    print(f"\n[TOOL EXECUTING] call_911 requested. (BPM: {heart_rate})")
    
    # Tool-level guardrail: If model passes heart rate, validate it
    if heart_rate is not None:
        if 40 <= heart_rate <= 170:
            print(f"[TOOL BLOCKED] Heart rate {heart_rate} is not critical. Call rejected.")
            result = {
                "action": "CALLING_911",
                "status": "REJECTED_NON_CRITICAL",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"911 call rejected by system guardrail: heart rate {heart_rate} BPM is not critical (<40 or >170 required). Do not attempt to call 911 again for this reading.",
            }
            return str(result)
            
    print(f"[TOOL APPROVED] Dispatching emergency services...")
    result = {
        "action": "CALLING_911",
        "status": "EMERGENCY_DISPATCHED",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": "911 called. Emergency services dispatched.",
    }
    return str(result)


# ============== Tool Registry (for frontend/API) ==============

TOOL_LIST = [call_911]

def get_tools_schema() -> List[Dict]:
    """Get tools in a simple schema format for the /tools endpoint."""
    return [
        {
            "type": "function",
            "function": {
                "name": "call_911",
                "description": "Call 911 emergency services immediately. ONLY use for CRITICAL readings: heart rate below 40 BPM or above 170 BPM.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
