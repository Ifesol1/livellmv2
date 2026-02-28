import asyncio
import websockets
import json
import time

async def test():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # 1. Load Model
        print("Sending load...")
        await websocket.send(json.dumps({"type": "load", "model": "Qwen/Qwen3-0.6B"}))
        
        while True:
            resp = await websocket.recv()
            data = json.loads(resp)
            print(f"Received: {data.get('type')}")
            if data.get("type") == "loaded":
                print("Model loaded!")
                break
        
        # 2. Start Generation
        print("Sending generate...")
        await websocket.send(json.dumps({
            "type": "generate", 
            "prompt": "Explain the theory of relativity in simple terms.", 
            "max_tokens": 100
        }))
        
        # 3. Wait a bit then interrupt
        tokens_received = 0
        interrupted = False
        
        while True:
            resp = await websocket.recv()
            data = json.loads(resp)
            
            if data.get("type") == "token":
                tokens_received += 1
                print(data["content"], end="", flush=True)
                
                if tokens_received > 10 and not interrupted:
                    print("\n\n>>> SENDING SIGNAL: I meant quantum mechanics <<<\n")
                    await websocket.send(json.dumps({
                        "type": "signal",
                        "content": "Actually, I meant quantum mechanics"
                    }))
                    interrupted = True
            
            elif data.get("type") == "done":
                print("\nGeneration done.")
                break
            
            elif data.get("type") == "signal_sent":
                print(f"\n[Server confirmed signal: {data['content']}]")

if __name__ == "__main__":
    asyncio.run(test())
