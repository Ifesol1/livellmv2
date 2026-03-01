'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Heart, Loader2, Activity, AlertTriangle, Phone, Send, Zap, Settings2, ChevronDown, ChevronRight, ArrowDown, Play, Square, RotateCcw } from 'lucide-react'
import ChatMessage from '../../components/ChatMessage'

const DEFAULT_MODEL = 'qwen2.5'

const DEFAULT_SYSTEM_PROMPT = `You are a medical monitoring AI assistant monitoring a patient's heart rate in real-time.

HEART RATE THRESHOLDS:
- 60-100 BPM = NORMAL resting heart rate. No action needed.
- 100-150 BPM = ELEVATED. Could be exercise or stress. Do NOT call 911.
- Below 50 BPM or above 150 BPM = ABNORMAL. Warn the user but do NOT call 911.
- Below 40 BPM or above 170 BPM = CRITICAL. Only then should you call 911.

TOOL USAGE RULES — READ CAREFULLY:
- You have access to a call_911 tool. DO NOT USE IT unless the heart rate is CRITICAL (below 40 or above 170 BPM).
- For NORMAL readings (60-100 BPM): respond with a brief status update. NO tool calls.
- For ELEVATED readings (100-150 BPM): respond with a brief note. NO tool calls.
- For ABNORMAL readings: warn the user verbally. NO tool calls.
- For CRITICAL readings (below 40 or above 170 BPM) sustained over multiple updates: use call_911.
- If you are unsure, DO NOT call 911. Only call for clear, sustained critical emergencies.
- Most of your responses should be plain text with ZERO tool calls.

Be concise. Briefly acknowledge each heart rate update in 1-2 sentences.`

const DEFAULT_SIGNAL_TEMPLATE = '[HEART RATE UPDATE] Current BPM: {bpm} - Status: {status}'

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

interface SignalLogEntry {
  bpm: number
  status: string
  timestamp: Date
}

function generateHeartRate(elapsed: number, scenario: string): number {
  switch (scenario) {
    case 'normal':
      return Math.round(75 + Math.sin(elapsed / 5) * 10 + (Math.random() - 0.5) * 5)
    case 'exercise':
      const exerciseRate = Math.min(140, 75 + elapsed * 3)
      return Math.round(exerciseRate + (Math.random() - 0.5) * 5)
    case 'critical':
      if (elapsed < 6) return Math.round(75 + (Math.random() - 0.5) * 5)
      if (elapsed < 10) return Math.round(75 + (elapsed - 6) * 20 + (Math.random() - 0.5) * 5)
      return Math.round(160 + Math.sin(elapsed) * 15 + (Math.random() - 0.5) * 5)
    case 'bradycardia':
      if (elapsed < 6) return Math.round(70 + (Math.random() - 0.5) * 5)
      if (elapsed < 12) return Math.round(70 - (elapsed - 6) * 5 + (Math.random() - 0.5) * 3)
      return Math.round(35 + (Math.random() - 0.5) * 5)
    default:
      return 75
  }
}

function getHeartRateStatus(bpm: number): { status: string; color: string; bg: string; urgent: boolean } {
  if (bpm < 40 || bpm > 150) return { status: 'CRITICAL', color: 'text-red-400', bg: 'bg-red-500/20', urgent: true }
  if (bpm < 50 || bpm > 120) return { status: 'ABNORMAL', color: 'text-amber-400', bg: 'bg-amber-500/20', urgent: false }
  return { status: 'NORMAL', color: 'text-emerald-400', bg: 'bg-emerald-500/20', urgent: false }
}

export default function HeartRateDemo() {
  const [isConnected, setIsConnected] = useState(false)
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [scenario, setScenario] = useState<string>('normal')
  
  const [messages, setMessages] = useState<Message[]>([])
  const [currentResponse, setCurrentResponse] = useState('')
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCall[]>([])
  const [currentHeartRate, setCurrentHeartRate] = useState(75)
  const [emergency911Called, setEmergency911Called] = useState(false)
  
  // New UX state
  const [userPrompt, setUserPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [signalTemplate, setSignalTemplate] = useState(DEFAULT_SIGNAL_TEMPLATE)
  const [signalLog, setSignalLog] = useState<SignalLogEntry[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [showSignalLog, setShowSignalLog] = useState(false)
  const [signalInterval, setSignalInterval] = useState(2)
  const [showScrollDown, setShowScrollDown] = useState(false)
  
  const wsRef = useRef<WebSocket | null>(null)
  const currentResponseRef = useRef('')
  const currentToolCallsRef = useRef<ToolCall[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const monitorIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const elapsedRef = useRef(0)
  const userScrolledUpRef = useRef(false)
  const signalLogEndRef = useRef<HTMLDivElement>(null)
  const systemPromptRef = useRef(DEFAULT_SYSTEM_PROMPT)
  const signalTemplateRef = useRef(DEFAULT_SIGNAL_TEMPLATE)
  const scenarioRef = useRef('normal')
  const pendingReadingsRef = useRef<{bpm: number, status: string}[]>([])
  const isMonitoringRef = useRef(false)
  const waitingForReadingsRef = useRef(false)

  // Keep refs in sync for interval closure
  useEffect(() => { systemPromptRef.current = systemPrompt }, [systemPrompt])
  useEffect(() => { signalTemplateRef.current = signalTemplate }, [signalTemplate])
  useEffect(() => { scenarioRef.current = scenario }, [scenario])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    userScrolledUpRef.current = false
    setShowScrollDown(false)
  }, [])

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, currentResponse])

  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const { scrollTop, scrollHeight, clientHeight } = container
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 120
    userScrolledUpRef.current = !isNearBottom
    setShowScrollDown(!isNearBottom)
  }, [])

  useEffect(() => {
    signalLogEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [signalLog])

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
          
          // Auto-continue monitoring: send next generation with accumulated readings
          if (isMonitoringRef.current) {
            if (pendingReadingsRef.current.length > 0) {
              waitingForReadingsRef.current = false
              const readings = [...pendingReadingsRef.current]
              pendingReadingsRef.current = []
              const latest = readings[readings.length - 1]
              const readingsText = readings.map(r => `${r.bpm} BPM (${r.status})`).join(', ')
              const nextPrompt = `[HEART RATE UPDATE] Latest readings: ${readingsText}. Current: ${latest.bpm} BPM (${latest.status}).`
              
              setTimeout(() => {
                if (!isMonitoringRef.current || !wsRef.current) return
                setIsGenerating(true)
                currentResponseRef.current = ''
                currentToolCallsRef.current = []
                setCurrentToolCalls([])
                wsRef.current.send(JSON.stringify({
                  type: 'generate_with_tools',
                  prompt: nextPrompt,
                  max_tokens: 500,
                  temperature: 0.7,
                  enable_tools: true
                }))
              }, 500)
            } else {
              waitingForReadingsRef.current = true
            }
          }
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
          
          if (data.name === 'call_911' && data.success) {
            // Check if it was actually dispatched or rejected
            const resultData = data.result
            if (resultData && typeof resultData === 'object' && resultData.status === 'EMERGENCY_DISPATCHED') {
              setEmergency911Called(true)
            }
          }
          break
          
        case 'signal_sent':
          // Don't add signal messages to chat — they go to the signal log only
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
    return () => { 
      wsRef.current?.close()
      if (monitorIntervalRef.current) {
        clearInterval(monitorIntervalRef.current)
      }
    }
  }, [connect])

  const loadModel = () => {
    if (!wsRef.current) return
    setIsLoading(true)
    wsRef.current.send(JSON.stringify({
      type: 'load',
      model: DEFAULT_MODEL
    }))
  }

  const startMonitoring = () => {
    if (!wsRef.current || !isModelLoaded) return
    
    setIsMonitoring(true)
    isMonitoringRef.current = true
    waitingForReadingsRef.current = false
    pendingReadingsRef.current = []
    setEmergency911Called(false)
    setSignalLog([])
    elapsedRef.current = 0
    
    const initialPrompt = `Patient monitoring session started. I will send you heart rate updates every ${signalInterval} seconds. Acknowledge and begin monitoring.`
    
    setMessages([{ role: 'user', content: 'Starting heart rate monitoring session...' }])
    setIsGenerating(true)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    
    wsRef.current.send(JSON.stringify({
      type: 'generate_with_tools',
      prompt: initialPrompt,
      system_prompt: systemPromptRef.current,
      max_tokens: 500,
      temperature: 0.7,
      enable_tools: true
    }))
    
    monitorIntervalRef.current = setInterval(() => {
      if (!wsRef.current) return
      
      elapsedRef.current += signalInterval
      const heartRate = generateHeartRate(elapsedRef.current, scenarioRef.current)
      setCurrentHeartRate(heartRate)
      
      const { status } = getHeartRateStatus(heartRate)
      
      setSignalLog(prev => [...prev, { bpm: heartRate, status, timestamp: new Date() }])
      
      // Accumulate readings for next generation cycle
      pendingReadingsRef.current.push({ bpm: heartRate, status })
      
      // If generation finished and was waiting for readings, trigger next cycle
      if (waitingForReadingsRef.current && wsRef.current) {
        waitingForReadingsRef.current = false
        const readings = [...pendingReadingsRef.current]
        pendingReadingsRef.current = []
        const latest = readings[readings.length - 1]
        const readingsText = readings.map(r => `${r.bpm} BPM (${r.status})`).join(', ')
        const nextPrompt = `[HEART RATE UPDATE] Latest readings: ${readingsText}. Current: ${latest.bpm} BPM (${latest.status}).`
        
        setIsGenerating(true)
        currentResponseRef.current = ''
        currentToolCallsRef.current = []
        setCurrentToolCalls([])
        wsRef.current.send(JSON.stringify({
          type: 'generate_with_tools',
          prompt: nextPrompt,
          max_tokens: 500,
          temperature: 0.7,
          enable_tools: true
        }))
      }
    }, signalInterval * 1000)
  }

  const stopMonitoring = () => {
    setIsMonitoring(false)
    isMonitoringRef.current = false
    waitingForReadingsRef.current = false
    if (monitorIntervalRef.current) {
      clearInterval(monitorIntervalRef.current)
      monitorIntervalRef.current = null
    }
  }

  const resetSession = () => {
    stopMonitoring()
    setMessages([])
    setSignalLog([])
    setCurrentResponse('')
    setCurrentToolCalls([])
    setEmergency911Called(false)
    setCurrentHeartRate(75)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    pendingReadingsRef.current = []
    isMonitoringRef.current = false
    waitingForReadingsRef.current = false
    elapsedRef.current = 0
  }

  const sendUserMessage = () => {
    if (!wsRef.current || !userPrompt.trim() || !isModelLoaded) return
    
    setMessages(prev => [...prev, { role: 'user', content: userPrompt }])
    setIsGenerating(true)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    setCurrentToolCalls([])
    
    wsRef.current.send(JSON.stringify({
      type: 'generate_with_tools',
      prompt: userPrompt,
      max_tokens: 500,
      temperature: 0.7,
      enable_tools: true
    }))
    
    setUserPrompt('')
  }

  const sendManualSignal = (content: string) => {
    if (!wsRef.current || !content.trim()) return
    wsRef.current.send(JSON.stringify({
      type: 'signal',
      content,
      priority: 1.5
    }))
    setSignalLog(prev => [...prev, { bpm: currentHeartRate, status: 'MANUAL', timestamp: new Date() }])
  }

  const { status, color, bg, urgent } = getHeartRateStatus(currentHeartRate)

  return (
    <div className="h-screen bg-gray-950 text-white flex flex-col overflow-hidden">
      {/* Emergency Banner */}
      {emergency911Called && (
        <div className="bg-red-600 text-white px-4 py-2.5 flex items-center justify-center gap-3 animate-pulse shrink-0">
          <Phone className="w-5 h-5" />
          <span className="font-bold">EMERGENCY SERVICES DISPATCHED</span>
          <Phone className="w-5 h-5" />
        </div>
      )}
      
      {/* Header */}
      <header className="border-b border-gray-800/80 px-5 py-3 flex items-center justify-between shrink-0 bg-gray-950/90 backdrop-blur-sm z-10">
        <div className="flex items-center gap-3">
          <Heart className={`w-5 h-5 ${isMonitoring ? 'text-red-500 animate-pulse' : 'text-gray-600'}`} />
          <h1 className="text-lg font-semibold tracking-tight">Heart Rate Monitor</h1>
          <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
        </div>
        
        <div className="flex items-center gap-2">
          {!isModelLoaded && isConnected && (
            <button
              onClick={loadModel}
              disabled={isLoading}
              className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-50 transition-colors"
            >
              {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              {isLoading ? 'Loading...' : 'Load Model'}
            </button>
          )}
          {isModelLoaded && (
            <span className="text-xs text-emerald-400/80 font-medium">Model Ready</span>
          )}
          <button
            onClick={() => setShowSettings(s => !s)}
            className={`p-2 rounded-lg transition-colors ${showSettings ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'}`}
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar: Vitals + Signal Log */}
        <aside className="w-72 border-r border-gray-800/80 flex flex-col shrink-0 bg-gray-900/30">
          {/* Vitals Card */}
          <div className={`p-4 border-b border-gray-800/60 transition-colors ${urgent ? 'bg-red-950/40' : ''}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-semibold uppercase tracking-widest text-gray-500">Vitals</span>
              <div className={`flex items-center gap-1.5 text-[11px] font-bold px-2 py-0.5 rounded-full ${bg} ${color}`}>
                {urgent && <AlertTriangle className="w-3 h-3" />}
                {status}
              </div>
            </div>
            <div className="flex items-end gap-2">
              <Activity className={`w-7 h-7 ${color} ${urgent ? 'animate-pulse' : ''}`} />
              <span className={`text-4xl font-bold tabular-nums leading-none ${color}`}>
                {currentHeartRate}
              </span>
              <span className={`text-sm font-medium ${color} opacity-70 mb-0.5`}>BPM</span>
            </div>
          </div>

          {/* Controls */}
          <div className="p-4 border-b border-gray-800/60 space-y-3">
            <div>
              <label className="text-[11px] font-semibold uppercase tracking-widest text-gray-500 mb-1.5 block">Scenario</label>
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700/50 rounded-lg text-sm focus:outline-none focus:border-blue-500 transition-colors cursor-pointer"
              >
                <option value="normal">Normal</option>
                <option value="exercise">Exercise</option>
                <option value="critical">Critical Emergency</option>
                <option value="bradycardia">Bradycardia (Low HR)</option>
              </select>
            </div>
            
            <div>
              <label className="text-[11px] font-semibold uppercase tracking-widest text-gray-500 mb-1.5 block">Signal Interval</label>
              <div className="flex items-center gap-2">
                <input 
                  type="range" 
                  min={1} 
                  max={10} 
                  value={signalInterval}
                  onChange={(e) => setSignalInterval(Number(e.target.value))}
                  disabled={isMonitoring}
                  className="flex-1 accent-blue-500 disabled:opacity-40"
                />
                <span className="text-xs font-mono text-gray-400 w-8 text-right">{signalInterval}s</span>
              </div>
            </div>
            
            <div className="flex gap-2">
              {!isMonitoring ? (
                <button
                  onClick={startMonitoring}
                  disabled={!isModelLoaded}
                  className="flex-1 px-3 py-2.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-medium flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Play className="w-3.5 h-3.5" />
                  Start
                </button>
              ) : (
                <button
                  onClick={stopMonitoring}
                  className="flex-1 px-3 py-2.5 bg-red-600 hover:bg-red-500 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <Square className="w-3.5 h-3.5" />
                  Stop
                </button>
              )}
              <button
                onClick={resetSession}
                className="px-3 py-2.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
                title="Reset session"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Signal Log */}
          <div className="flex-1 flex flex-col min-h-0">
            <button
              onClick={() => setShowSignalLog(s => !s)}
              className="px-4 py-2.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-widest text-gray-500 hover:text-gray-400 hover:bg-gray-800/40 transition-colors"
            >
              <span className="flex items-center gap-2">
                <Zap className="w-3 h-3" />
                Signal Log ({signalLog.length})
              </span>
              {showSignalLog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            </button>
            {showSignalLog && (
              <div className="flex-1 overflow-y-auto px-3 pb-3">
                {signalLog.length === 0 && (
                  <p className="text-xs text-gray-600 text-center py-4">No signals sent yet</p>
                )}
                {signalLog.map((entry, i) => {
                  const s = getHeartRateStatus(entry.bpm)
                  return (
                    <div key={i} className="flex items-center gap-2 py-1 text-xs">
                      <span className="text-gray-600 font-mono w-14 shrink-0">
                        {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                      </span>
                      <span className={`font-bold tabular-nums w-10 text-right ${entry.status === 'MANUAL' ? 'text-blue-400' : s.color}`}>
                        {entry.status === 'MANUAL' ? 'USR' : entry.bpm}
                      </span>
                      <span className={`text-[10px] font-medium ${entry.status === 'MANUAL' ? 'text-blue-400/70' : s.color} opacity-70`}>
                        {entry.status}
                      </span>
                    </div>
                  )
                })}
                <div ref={signalLogEndRef} />
              </div>
            )}
          </div>
        </aside>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          {/* Settings Panel (collapsible) */}
          {showSettings && (
            <div className="border-b border-gray-800/60 bg-gray-900/50 p-4 shrink-0">
              <div className="max-w-3xl mx-auto space-y-3">
                <div>
                  <label className="text-[11px] font-semibold uppercase tracking-widest text-gray-500 mb-1.5 block">System Prompt</label>
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    disabled={isMonitoring}
                    rows={4}
                    className="w-full px-3 py-2.5 bg-gray-800 border border-gray-700/50 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500 resize-y disabled:opacity-50 transition-colors font-mono leading-relaxed"
                    placeholder="System prompt for the AI..."
                  />
                </div>
                <div>
                  <label className="text-[11px] font-semibold uppercase tracking-widest text-gray-500 mb-1.5 block">
                    Signal Template
                    <span className="normal-case tracking-normal font-normal text-gray-600 ml-2">Use {'{bpm}'} and {'{status}'} as placeholders</span>
                  </label>
                  <input
                    type="text"
                    value={signalTemplate}
                    onChange={(e) => setSignalTemplate(e.target.value)}
                    className="w-full px-3 py-2.5 bg-gray-800 border border-gray-700/50 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500 transition-colors font-mono"
                    placeholder="Signal template..."
                  />
                </div>
              </div>
            </div>
          )}
          
          {/* Messages Area */}
          <div className="flex-1 min-h-0 relative">
            <div 
              ref={messagesContainerRef}
              onScroll={handleScroll}
              className="absolute inset-0 overflow-y-auto px-6 py-4"
            >
              <div className="max-w-3xl mx-auto">
                {messages.length === 0 && !isGenerating && (
                  <div className="flex flex-col items-center justify-center py-20 text-gray-600">
                    <div className="w-16 h-16 rounded-full bg-gray-800/50 flex items-center justify-center mb-4">
                      <Heart className="w-8 h-8 opacity-40" />
                    </div>
                    <p className="text-sm font-medium text-gray-500 mb-1">No session active</p>
                    <p className="text-xs text-gray-600 max-w-xs text-center">
                      Pick a scenario, customize settings if you want, then hit Start. You can also type messages to the AI during monitoring.
                    </p>
                  </div>
                )}
                
                {messages.map((msg, i) => (
                  <ChatMessage 
                    key={i} 
                    role={msg.role} 
                    content={msg.content}
                    toolCalls={msg.toolCalls}
                  />
                ))}
                
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

            {/* Scroll to bottom FAB */}
            {showScrollDown && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700/50 rounded-full text-xs text-gray-400 flex items-center gap-1.5 shadow-lg transition-all hover:text-white z-10"
              >
                <ArrowDown className="w-3 h-3" />
                New messages
              </button>
            )}
          </div>

          {/* Input Bar */}
          <div className="border-t border-gray-800/60 px-5 py-3 shrink-0 bg-gray-950/90 backdrop-blur-sm">
            <div className="max-w-3xl mx-auto flex gap-2">
              <input
                type="text"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    sendUserMessage()
                  }
                }}
                placeholder={isMonitoring ? 'Send a message to the AI during monitoring...' : 'Send a message...'}
                disabled={!isModelLoaded}
                className="flex-1 px-4 py-2.5 bg-gray-800/80 border border-gray-700/50 rounded-xl text-sm text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/60 disabled:opacity-40 transition-colors"
              />
              <button
                onClick={sendUserMessage}
                disabled={!userPrompt.trim() || !isModelLoaded}
                className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 rounded-xl text-sm font-medium disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
