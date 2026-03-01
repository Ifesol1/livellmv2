# Live LLM Web Application

This repository contains the Next.js frontend for the Live LLM project. The backend (Colab server and Python code) is hosted separately.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                    (Next.js + React)                         │
│                 http://localhost:3000                        │
├─────────────────────────────────────────────────────────────┤
│                          │                                   │
│                     HTTP / SSE                               │
│                          │                                   │
├─────────────────────────────────────────────────────────────┤
│                        Backend                               │
│               (Colab + ngrok/Cloudflare)                     │
│               https://your-tunnel-url.trycloudflare.com      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd web
npm install
```

### 2. Start Development Server

```bash
cd web
npm run dev
```

### 3. Open Browser

Navigate to [http://localhost:3000](http://localhost:3000)

## Features

### Chat Interface
- Real-time streaming token display using Server-Sent Events (SSE)
- Connection status indicator to your remote backend tunnel
- Markdown rendering for model responses

### Live Signal Injection
- Signal bar to inject signals mid-generation
- Signals appear in chat with lightning bolt icon
- Model responds to signals in real-time without stopping generation

### Server Connection
- Enter your Colab tunnel URL in the settings/connection banner
- Seamless reconnection and state management

## File Structure

```
livellmv2/
├── web/                    # Next.js Frontend
│   ├── app/
│   │   ├── page.tsx       # Main chat UI
│   │   ├── layout.tsx     # Root layout
│   │   └── globals.css    # Styles
│   ├── components/        # React Components
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```
