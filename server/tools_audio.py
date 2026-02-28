"""
Sentinel Audio Tool - ElevenLabs Integration

Allows the Live LLM to speak with dynamic emotion based on urgency.
"""

import os
import requests
import json
from typing import Optional

# Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Default Voice ID (e.g., "Rachel" or a custom "Sentinel" voice)
# You can find these IDs in your ElevenLabs dashboard
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") 

def text_to_speech(text: str, urgency: float = 0.0) -> Optional[str]:
    """
    Generate speech from text.
    
    Args:
        text: The text to speak.
        urgency: 0.0 (Calm) to 1.0 (Panic).
                 Affects 'stability' and 'similarity_boost'.
    
    Returns:
        URL or base64 path to the audio file.
    """
    if not ELEVENLABS_API_KEY:
        print("[Audio] ElevenLabs API Key missing.")
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    # Map urgency to voice settings
    # High urgency = Lower stability (more expressive/erratic)
    # Low urgency = High stability (steady, robotic)
    stability = max(0.3, 1.0 - (urgency * 0.7))
    style = urgency  # Higher urgency = more style exaggeration
    
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": 0.75,
            "style": style,
            "use_speaker_boost": True
        }
    }
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    try:
        print(f"[Audio] Generating speech (Urgency: {urgency:.2f})...")
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            # Save to static file for frontend to play
            # In a real app, you'd stream this.
            # We use a timestamp to avoid caching
            import time
            filename = f"speech_{int(time.time())}.mp3"
            
            # Ensure web/public exists or similar
            # For now, we save to a 'static' folder in server
            output_dir = os.path.join(os.path.dirname(__file__), "static")
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            print(f"[Audio] Saved to {filepath}")
            # Return relative path for frontend
            return f"/static/{filename}"
        else:
            print(f"[Audio] Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"[Audio] Exception: {e}")
        return None

# --- Tool Definition for LLM ---

def get_audio_tools_schema():
    """Return the tool schema for the LLM."""
    return [
        {
            "name": "broadcast_alert",
            "description": "Broadcasts a voice message to the facility. Use this for IMPORTANT announcements or warnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The text to speak."
                    },
                    "urgency": {
                        "type": "number",
                        "description": "Urgency level from 0.0 (Info) to 1.0 (Critical/Evacuate). Affects voice tone.",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["message", "urgency"]
            }
        }
    ]

def execute_broadcast_alert(message: str, urgency: float = 0.5):
    """Execution handler for the tool."""
    audio_path = text_to_speech(message, urgency)
    if audio_path:
        return f"Alert broadcasted: '{message}' (Urgency: {urgency})"
    return "Failed to broadcast alert."
