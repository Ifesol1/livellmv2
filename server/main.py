"""
Live LLM API Server

FastAPI server with WebSocket support for real-time streaming.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json

from llm_service import llm_service

app = FastAPI(
    title="Live LLM API",
    description="Real-time LLM with live signal injection",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class LoadModelRequest(BaseModel):
    model_name: str = "qwen2.5"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

class SignalRequest(BaseModel):
    content: str
    priority: float = 1.0

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# --- Tools Endpoint ---

@app.get("/tools")
async def get_tools():
    """Get available tools for the model."""
    return {"tools": llm_service.get_tools()}


# --- Connection Manager ---

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


# --- REST Endpoints ---

@app.get("/")
async def root():
    return {"message": "Live LLM API", "docs": "/docs"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        status="ok",
        model_loaded=llm_service.is_loaded,
        model_name=llm_service.model_name
    )

@app.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model."""
    try:
        await llm_service.load_model(request.model_name)
        return {"success": True, "model": request.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signal")
async def post_signal(request: SignalRequest):
    """Post a live signal to the current generation."""
    if not llm_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    llm_service.post_signal(request.content, request.priority)
    
    # Broadcast signal to all connected clients
    await manager.broadcast({
        "type": "signal",
        "content": request.content,
        "priority": request.priority
    })
    
    return {"success": True}

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text (non-streaming)."""
    if not llm_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    tokens = []
    async for token in llm_service.generate_stream(
        request.prompt,
        request.max_tokens,
        request.temperature
    ):
        tokens.append(token)
    
    return {"text": "".join(tokens)}


# --- WebSocket Endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time streaming with concurrent message handling.
    
    Uses two concurrent tasks:
    1. Token streaming task - sends generated tokens
    2. Message receiving task - handles signals during generation
    """
    await manager.connect(websocket)
    
    generation_task = None
    stop_generation = asyncio.Event()
    
    async def stream_tokens(prompt: str, max_tokens: int, temperature: float):
        """Stream tokens to the client."""
        try:
            async for token in llm_service.generate_stream(
                prompt, max_tokens, temperature
            ):
                if stop_generation.is_set():
                    break
                try:
                    await websocket.send_json({
                        "type": "token",
                        "content": token
                    })
                except RuntimeError:
                    # Connection closed
                    break
            try:
                await websocket.send_json({"type": "done"})
            except RuntimeError:
                pass  # Connection already closed
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except RuntimeError:
                pass  # Connection closed
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            print(f"[WS] Received: {msg_type}", flush=True)
            
            if msg_type == "generate":
                # Start generation in background task
                prompt = message.get("prompt", "")
                max_tokens = message.get("max_tokens", 200)
                temperature = message.get("temperature", 0.7)
                enable_thinking = message.get("enable_thinking", False)
                
                # Llama doesn't need special thinking suffix
                
                if not llm_service.is_loaded:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Model not loaded. Call /load first."
                    })
                    continue
                
                # Cancel any existing generation
                if generation_task and not generation_task.done():
                    stop_generation.set()
                    await generation_task
                
                stop_generation.clear()
                # Start streaming in background - allows receiving more messages
                generation_task = asyncio.create_task(
                    stream_tokens(prompt, max_tokens, temperature)
                )
            
            elif msg_type == "generate_with_tools":
                # Generation with tool calling support
                prompt = message.get("prompt", "")
                max_tokens = message.get("max_tokens", 2000)
                temperature = message.get("temperature", 0.7)
                enable_tools = message.get("enable_tools", True)
                enable_thinking = message.get("enable_thinking", False)
                system_prompt = message.get("system_prompt", None)
                
                if not llm_service.is_loaded:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Model not loaded"
                    })
                    continue
                
                # Cancel existing generation
                if generation_task and not generation_task.done():
                    stop_generation.set()
                    await generation_task
                
                stop_generation.clear()
                
                async def stream_with_tools():
                    try:
                        async for item in llm_service.generate_with_tools(
                            prompt, max_tokens, temperature, enable_tools, enable_thinking,
                            system_prompt=system_prompt
                        ):
                            if stop_generation.is_set():
                                break
                            try:
                                await websocket.send_json(item)
                            except RuntimeError:
                                break
                    except Exception as e:
                        try:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                        except RuntimeError:
                            pass
                
                generation_task = asyncio.create_task(stream_with_tools())
            
            elif msg_type == "signal":
                # Inject signal - this now works during generation!
                content = message.get("content", "")
                priority = message.get("priority", 1.0)
                print(f"[WS] Injecting signal: {content}", flush=True)
                llm_service.post_signal(content, priority)
                
                # Echo back to confirm
                await websocket.send_json({
                    "type": "signal_sent",
                    "content": content
                })
            
            elif msg_type == "stop":
                # Stop current generation
                if generation_task and not generation_task.done():
                    stop_generation.set()
            
            elif msg_type == "load":
                # Load model
                model_name = message.get("model", "qwen2.5")
                await websocket.send_json({
                    "type": "status",
                    "message": f"Loading {model_name}..."
                })
                
                try:
                    await llm_service.load_model(model_name)
                    await websocket.send_json({
                        "type": "loaded",
                        "model": model_name
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif msg_type == "status":
                await websocket.send_json({
                    "type": "status",
                    "model_loaded": llm_service.is_loaded,
                    "model_name": llm_service.model_name
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        print(f"WebSocket error: {e}")


# --- Startup ---

@app.on_event("startup")
async def startup():
    """Load default model on startup."""
    print("Starting Live LLM Server...")
    # Don't auto-load to save memory - let client request it


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
