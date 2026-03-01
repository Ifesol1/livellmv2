'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Send, Loader2, Zap, Radio } from 'lucide-react'
import ChatMessage from '../components/ChatMessage'

const DEFAULT_MODEL = 'qwen2.5'

interface Message {
  role: 'user' | 'assistant' | 'signal'
  content: string
  toolCalls?: ToolCall[]
}

interface ToolCall {
  name: string
  arguments: Record<string, any>
  result?: any
  success?: boolean
  error?: string
  isExecuting?: boolean
}

export default function ChatPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  
  const [prompt, setPrompt] = useState('')
  const [liveSignal, setLiveSignal] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [currentResponse, setCurrentResponse] = useState('')
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCall[]>([])
  
  const wsRef = useRef<WebSocket | null>(null)
  const currentResponseRef = useRef('')
  const currentToolCallsRef = useRef<ToolCall[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const userScrolledUpRef = useRef(false)

  // Only auto-scroll if user is near the bottom
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, currentResponse])

  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const { scrollTop, scrollHeight, clientHeight } = container
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
    userScrolledUpRef.current = !isNearBottom
  }, [])

  const connect = useCallback(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onopen = () => {
      setIsConnected(true)
      ws.send(JSON.stringify({ type: 'status' }))
    }
    
    ws.onclose = () => {
      setIsConnected(false)
      setIsModelLoaded(false)
      setTimeout(connect, 2000)
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'token':
          currentResponseRef.current += data.content
          setCurrentResponse(currentResponseRef.current)
          break
          
        case 'done':
          setIsGenerating(false)
          const finalResponse = currentResponseRef.current
          const finalToolCalls = [...currentToolCallsRef.current]
          if (finalResponse.trim() || finalToolCalls.length > 0) {
            setMessages(prev => [...prev, { 
              role: 'assistant', 
              content: finalResponse,
              toolCalls: finalToolCalls.length > 0 ? finalToolCalls : undefined
            }])
          }
          currentResponseRef.current = ''
          currentToolCallsRef.current = []
          setCurrentResponse('')
          setCurrentToolCalls([])
          break
        
        case 'tool_call':
          const newToolCall = {
            name: data.name,
            arguments: data.arguments,
            isExecuting: true
          }
          currentToolCallsRef.current = [...currentToolCallsRef.current, newToolCall]
          setCurrentToolCalls(currentToolCallsRef.current)
          break
        
        case 'tool_result':
          currentToolCallsRef.current = currentToolCallsRef.current.map(tc => 
            tc.name === data.name && tc.isExecuting
              ? { ...tc, result: data.result, success: data.success, error: data.error, isExecuting: false }
              : tc
          )
          setCurrentToolCalls(currentToolCallsRef.current)
          break
          
        case 'signal_sent':
          setMessages(prev => [...prev, { role: 'signal', content: data.content }])
          break
          
        case 'status':
          setIsModelLoaded(data.model_loaded)
          break
          
        case 'loaded':
          setIsModelLoaded(true)
          setIsLoading(false)
          break
          
        case 'error':
          console.error(data.message || data.content)
          setIsGenerating(false)
          setIsLoading(false)
          break
      }
    }
    
    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => { wsRef.current?.close() }
  }, [connect])

  const loadModel = () => {
    if (!wsRef.current) return
    setIsLoading(true)
    wsRef.current.send(JSON.stringify({
      type: 'load',
      model: DEFAULT_MODEL
    }))
  }

  const sendPrompt = () => {
    if (!wsRef.current || !prompt.trim() || !isModelLoaded) return
    
    setMessages(prev => [...prev, { role: 'user', content: prompt }])
    setIsGenerating(true)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    setCurrentToolCalls([])
    
    wsRef.current.send(JSON.stringify({
      type: 'generate_with_tools',
      prompt: prompt,
      max_tokens: 1000,
      temperature: 0.7,
      enable_tools: true
    }))
    
    setPrompt('')
  }

  const sendSignal = () => {
    if (!wsRef.current || !liveSignal.trim()) return
    
    wsRef.current.send(JSON.stringify({
      type: 'signal',
      content: liveSignal,
      priority: 1.0
    }))
    
    setLiveSignal('')
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-800 p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Radio className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-bold">Live LLM Chat</h1>
          <span className={`text-xs px-2 py-1 rounded ${isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <a 
            href="/demo" 
            className="ml-4 px-3 py-1 text-xs bg-red-600 hover:bg-red-500 rounded-lg font-medium flex items-center gap-1"
          >
            ❤️ Heart Rate Demo
          </a>
        </div>
        
        {!isModelLoaded && isConnected && (
          <button
            onClick={loadModel}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium flex items-center gap-2 disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            {isLoading ? 'Loading...' : 'Load Model'}
          </button>
        )}
        
        {isModelLoaded && (
          <span className="text-sm text-green-400">Model: {DEFAULT_MODEL}</span>
        )}
      </div>

      {/* Messages */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-6"
      >
        <div className="max-w-4xl mx-auto">
          {messages.map((msg, i) => (
            <ChatMessage 
              key={i} 
              role={msg.role} 
              content={msg.content}
              toolCalls={msg.toolCalls}
            />
          ))}
          
          {/* Current streaming response */}
          {(currentResponse || isGenerating) && (
            <ChatMessage 
              role="assistant" 
              content={currentResponse}
              isGenerating={isGenerating}
              toolCalls={currentToolCalls.length > 0 ? currentToolCalls : undefined}
            />
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Live Signal Input */}
      <div className="border-t border-gray-800 p-4 bg-gray-900/50">
        <div className="max-w-4xl mx-auto">
          <div className="mb-3 flex gap-2">
            <div className="flex-1 relative">
              <Zap className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-yellow-500" />
              <input
                type="text"
                value={liveSignal}
                onChange={(e) => setLiveSignal(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendSignal()}
                placeholder="Type live signal to inject mid-response..."
                className="w-full pl-10 pr-4 py-2 bg-yellow-900/20 border border-yellow-600/30 rounded-lg text-yellow-100 placeholder-yellow-600/50 focus:outline-none focus:border-yellow-500"
              />
            </div>
            <button
              onClick={sendSignal}
              disabled={!liveSignal.trim()}
              className="px-4 py-2 bg-yellow-600 hover:bg-yellow-500 rounded-lg font-medium disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Zap className="w-4 h-4" />
              Inject
            </button>
          </div>
          
          {/* Prompt Input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendPrompt()}
              placeholder="Type your message..."
              disabled={!isModelLoaded}
              className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <button
              onClick={sendPrompt}
              disabled={!prompt.trim() || !isModelLoaded || isGenerating}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium disabled:opacity-50 flex items-center gap-2"
            >
              {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              Send
            </button>
          </div>
          
          <p className="text-xs text-gray-500 mt-2 text-center">
            Send a message, then inject live signals mid-response with the yellow input above
          </p>
        </div>
      </div>
    </div>
  )
}
