# Live LLM Web Application

Professional web interface for real-time LLM with live signal injection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                    (Next.js + React)                         │
│                    http://localhost:3000                     │
├─────────────────────────────────────────────────────────────┤
│                          │                                   │
│                    WebSocket                                 │
│                          │                                   │
├─────────────────────────────────────────────────────────────┤
│                        Backend                               │
│                   (FastAPI + Python)                         │
│                   http://localhost:8000                      │
├─────────────────────────────────────────────────────────────┤
│                          │                                   │
│                    live_llm module                           │
│                 (LiveStreamGenerator)                        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Backend
cd server
pip install -r requirements.txt

# Frontend
cd web
npm install
```

### 2. Start Servers

**Option A: Manual**
```bash
# Terminal 1 - Backend
cd server
python main.py

# Terminal 2 - Frontend
cd web
npm run dev
```

**Option B: PowerShell Script**
```powershell
.\start.ps1
```

### 3. Open Browser

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Features

### Chat Interface
- Real-time streaming token display
- Model selection (Qwen, Gemma)
- Connection status indicator

### Live Signal Injection
- Yellow "signal bar" to inject signals mid-generation
- Signals appear in chat with lightning bolt icon
- Model responds to signals in real-time

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Check server/model status |
| POST | `/load` | Load a model |
| POST | `/signal` | Inject a live signal |
| POST | `/generate` | Generate text (non-streaming) |

### WebSocket

Connect to `ws://localhost:8000/ws`

**Client → Server:**
```json
{"type": "load", "model": "Qwen/Qwen3-0.6B"}
{"type": "generate", "prompt": "Hello", "max_tokens": 200}
{"type": "signal", "content": "BREAKING NEWS!"}
{"type": "status"}
```

**Server → Client:**
```json
{"type": "token", "content": "Hello"}
{"type": "done"}
{"type": "signal_sent", "content": "..."}
{"type": "loaded", "model": "..."}
{"type": "error", "message": "..."}
```

## File Structure

```
livellmv2/
├── server/                 # Python Backend
│   ├── main.py            # FastAPI server
│   ├── llm_service.py     # LLM wrapper
│   └── requirements.txt
│
├── web/                    # Next.js Frontend
│   ├── app/
│   │   ├── page.tsx       # Main chat UI
│   │   ├── layout.tsx     # Root layout
│   │   └── globals.css    # Styles
│   ├── package.json
│   └── tailwind.config.js
│
├── live_llm/              # Core LLM module
│   └── live_stream.py     # LiveStreamGenerator
│
└── start.ps1              # Start script
```

## Customization

### Change Default Model
Edit `server/llm_service.py`:
```python
await llm_service.load_model("google/gemma-3-1b-it")
```

### Signal Styles
Edit `live_llm/live_stream.py`:
- `visible`: `[LIVE]: signal`
- `system`: `[System note: signal]`
- `subtle`: `(signal)`
- `interrupt`: `-- ALERT: signal --`

### Add More Models
Edit `web/app/page.tsx`:
```typescript
const MODELS = [
  { id: 'your/model', name: 'Display Name' },
  // ...
]
```
