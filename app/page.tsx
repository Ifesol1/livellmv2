'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Heart, Loader2, Activity, AlertTriangle, Phone, Send, Zap, Settings2,
  ChevronDown, ChevronRight, ArrowDown, Play, Square, RotateCcw, Brain,
  TrendingUp, TrendingDown, LineChart, DollarSign, BarChart3, Wifi, WifiOff, MessageSquare,
  Camera, Shield, ShieldAlert, Video, VideoOff
} from 'lucide-react'
import ChatMessage from '../components/ChatMessage'
import {
  ResponsiveContainer, AreaChart, Area, LineChart as RechartsLineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts'

// ═══════════════════════════════════════════════════════════
//  Heart Rate Helpers
// ═══════════════════════════════════════════════════════════

const HR_SYSTEM_PROMPT = `You are a medical monitoring AI assistant monitoring a patient's heart rate in real-time.

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

const HR_SIGNAL_TEMPLATE = '[HEART RATE UPDATE] Current BPM: {bpm} - Status: {status}'

function generateHeartRate(elapsed: number, scenario: string): number {
  switch (scenario) {
    case 'normal':
      return Math.round(75 + Math.sin(elapsed / 5) * 10 + (Math.random() - 0.5) * 5)
    case 'exercise':
      return Math.round(Math.min(140, 75 + elapsed * 3) + (Math.random() - 0.5) * 5)
    case 'critical':
      if (elapsed < 6) return Math.round(75 + (Math.random() - 0.5) * 5)
      if (elapsed < 10) return Math.round(75 + (elapsed - 6) * 25 + (Math.random() - 0.5) * 5)
      return Math.round(185 + Math.sin(elapsed) * 10 + (Math.random() - 0.5) * 5)
    case 'bradycardia':
      if (elapsed < 6) return Math.round(70 + (Math.random() - 0.5) * 5)
      if (elapsed < 12) return Math.round(70 - (elapsed - 6) * 5 + (Math.random() - 0.5) * 3)
      return Math.round(35 + (Math.random() - 0.5) * 5)
    default:
      return 75
  }
}

function getHeartRateStatus(bpm: number): { status: string; color: string; bg: string; urgent: boolean } {
  if (bpm < 40 || bpm > 170) return { status: 'CRITICAL', color: 'text-red-400', bg: 'bg-red-500/20', urgent: true }
  if (bpm < 50 || bpm > 150) return { status: 'ABNORMAL', color: 'text-orange-400', bg: 'bg-orange-500/20', urgent: false }
  if (bpm > 100) return { status: 'ELEVATED', color: 'text-amber-400', bg: 'bg-amber-500/20', urgent: false }
  return { status: 'NORMAL', color: 'text-emerald-400', bg: 'bg-emerald-500/20', urgent: false }
}

// ═══════════════════════════════════════════════════════════
//  Stock Trading Helpers
// ═══════════════════════════════════════════════════════════

const STOCK_SYSTEM_PROMPT = `You are an aggressive stock trading AI executing trades on AAPL in real-time.

YOUR GOAL: Make profitable trades on a single stock (AAPL). Don't just analyze — ACT.

TRADING RULES:
- You trade ONLY AAPL. No other stocks.
- When price drops more than 1.5% from recent highs, BUY using buy_stock — capture the dip.
- When price rises more than 2% from your average buy price, SELL using sell_stock — lock in profits.
- buy_stock takes only quantity (1-10 shares). sell_stock takes only quantity.
- You CANNOT sell shares you don't own. The server will reject the order.
- Track your position mentally. The server tells you shares_held and cash_remaining after each trade.

PERSONALITY: Decisive. You prefer action over inaction. You see a 2% drop as opportunity, not risk.`

const STOCK_SIGNAL_TEMPLATE = '[MARKET UPDATE] {prices}'

const SECURITY_SYSTEM_PROMPT = `You are a security monitoring AI analyzing live camera feeds in real-time.

YOUR GOAL: Monitor camera frames for threats and suspicious activity. Analyze each frame carefully.

DETECTION RULES:
- Look for: intruders, unauthorized people, fire/smoke, suspicious packages, vandalism, forced entry
- For CONFIRMED threats with high confidence, use trigger_alarm tool immediately
- For low-confidence or ambiguous observations, describe what you see and continue monitoring
- Most frames will be normal — just briefly acknowledge the scene
- Be concise: 1-2 sentences per frame unless something notable is happening

TOOL: trigger_alarm(threat_type, confidence) — Only trigger for confirmed threats above 0.7 confidence.`

interface StockState {
  price: number
  prevPrice: number
  basePrice: number
}

const INITIAL_STOCK: StockState = { price: 185.20, prevPrice: 185.20, basePrice: 185.20 }

function generateStockPrice(stock: StockState, scenario: string, elapsed: number): StockState {
  let drift = 0
  let volatility = 0

  switch (scenario) {
    case 'stable':
      drift = 0
      volatility = 0.005
      break
    case 'dip':
      drift = -0.008
      volatility = 0.006
      break
    case 'crash':
      drift = elapsed < 8 ? -0.012 : 0.008
      volatility = 0.008
      break
    case 'rally':
      drift = 0.010
      volatility = 0.005
      break
    case 'volatile':
      drift = Math.sin(elapsed / 4) * 0.015
      volatility = 0.010
      break
    default:
      volatility = 0.005
  }

  const change = drift + (Math.random() - 0.5) * 2 * volatility
  const newPrice = Math.max(stock.price * 0.5, stock.price * (1 + change))

  return {
    prevPrice: stock.price,
    price: Math.round(newPrice * 100) / 100,
    basePrice: stock.basePrice,
  }
}

function formatStockChange(price: number, prevPrice: number): { pct: string; color: string; arrow: string } {
  const pct = ((price - prevPrice) / prevPrice) * 100
  if (pct > 0) return { pct: `+${pct.toFixed(2)}%`, color: 'text-emerald-400', arrow: '▲' }
  if (pct < 0) return { pct: `${pct.toFixed(2)}%`, color: 'text-red-400', arrow: '▼' }
  return { pct: '0.00%', color: 'text-gray-400', arrow: '─' }
}

// ═══════════════════════════════════════════════════════════
//  Shared Types
// ═══════════════════════════════════════════════════════════

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
  value: string
  label: string
  timestamp: Date
}

interface TradeEntry {
  action: 'BUY' | 'SELL'
  quantity: number
  price: number
  timestamp: string
  pnl?: number
  shares_held?: number
  cash_remaining?: number
}

interface HrDataPoint {
  time: string
  bpm: number
}

interface StockDataPoint {
  time: string
  AAPL: number
}

interface SecurityAlert {
  threat_type: string
  confidence: number
  timestamp: string
}

// ═══════════════════════════════════════════════════════════
//  Main Component
// ═══════════════════════════════════════════════════════════

export default function Dashboard() {
  // ─── Core state ───
  const [activeTab, setActiveTab] = useState<'heart' | 'stock' | 'security'>('heart')
  const [isConnected, setIsConnected] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  // ─── Chat state (per-tab) ───
  const [heartMessages, setHeartMessages] = useState<Message[]>([])
  const [stockMessages, setStockMessages] = useState<Message[]>([])
  const [securityMessages, setSecurityMessages] = useState<Message[]>([])
  const [currentResponse, setCurrentResponse] = useState('')
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCall[]>([])
  const [userPrompt, setUserPrompt] = useState('')

  // ─── Heart rate state ───
  const [hrScenario, setHrScenario] = useState('normal')
  const [currentHeartRate, setCurrentHeartRate] = useState(75)
  const [emergency911Called, setEmergency911Called] = useState(false)
  const [hrSignalLog, setHrSignalLog] = useState<SignalLogEntry[]>([])

  // ─── Stock state (single: AAPL) ───
  const [stockScenario, setStockScenario] = useState('stable')
  const [stock, setStock] = useState<StockState>(INITIAL_STOCK)
  const [trades, setTrades] = useState<TradeEntry[]>([])
  const [portfolioCash, setPortfolioCash] = useState(10000)
  const [portfolioShares, setPortfolioShares] = useState(0)
  const [stockSignalLog, setStockSignalLog] = useState<SignalLogEntry[]>([])

  // ─── Security state ───
  const [securityAlerts, setSecurityAlerts] = useState<SecurityAlert[]>([])
  const [securitySignalLog, setSecuritySignalLog] = useState<SignalLogEntry[]>([])
  const [alarmTriggered, setAlarmTriggered] = useState(false)
  const [isCamActive, setIsCamActive] = useState(false)
  const [frameCount, setFrameCount] = useState(0)

  // ─── Chart history ───
  const [hrHistory, setHrHistory] = useState<HrDataPoint[]>([])
  const [stockHistory, setStockHistory] = useState<StockDataPoint[]>([])

  // ─── Shared settings ───
  const [signalInterval, setSignalInterval] = useState(3)
  const [showSignalLog, setShowSignalLog] = useState(false)
  const [showScrollDown, setShowScrollDown] = useState(false)
  const [stockStrategy, setStockStrategy] = useState('')
  const [serverUrl, setServerUrl] = useState('')

  // ─── System prompts (editable) ───
  const [hrSystemPrompt, setHrSystemPrompt] = useState(HR_SYSTEM_PROMPT)
  const [stockSystemPrompt, setStockSystemPrompt] = useState(STOCK_SYSTEM_PROMPT + (stockStrategy ? "\n\nUSER'S CUSTOM STRATEGY:\n" + stockStrategy : ''))
  const [securitySystemPrompt, setSecuritySystemPrompt] = useState(SECURITY_SYSTEM_PROMPT)

  // ─── Refs ───
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentResponseRef = useRef('')
  const currentToolCallsRef = useRef<ToolCall[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const monitorIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const elapsedRef = useRef(0)
  const userScrolledUpRef = useRef(false)
  const signalLogEndRef = useRef<HTMLDivElement>(null)
  const isMonitoringRef = useRef(false)
  const activeTabRef = useRef<'heart' | 'stock' | 'security'>('heart')
  const hrScenarioRef = useRef('normal')
  const stockScenarioRef = useRef('stable')
  const stockRef = useRef<StockState>(INITIAL_STOCK)
  const hrSystemPromptRef = useRef(HR_SYSTEM_PROMPT)
  const stockSystemPromptRef = useRef(STOCK_SYSTEM_PROMPT)
  const securitySystemPromptRef = useRef(SECURITY_SYSTEM_PROMPT)
  const signalIntervalRef = useRef(3)
  const stockStrategyRef = useRef('')
  const serverUrlRef = useRef('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const camStreamRef = useRef<MediaStream | null>(null)
  const camIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Keep refs in sync
  useEffect(() => { activeTabRef.current = activeTab }, [activeTab])
  useEffect(() => { hrScenarioRef.current = hrScenario }, [hrScenario])
  useEffect(() => { stockScenarioRef.current = stockScenario }, [stockScenario])
  useEffect(() => { stockRef.current = stock }, [stock])
  useEffect(() => { hrSystemPromptRef.current = hrSystemPrompt }, [hrSystemPrompt])
  useEffect(() => { 
    const fullPrompt = STOCK_SYSTEM_PROMPT + (stockStrategy ? "\n\nUSER'S CUSTOM STRATEGY:\n" + stockStrategy : '')
    setStockSystemPrompt(fullPrompt)
    stockSystemPromptRef.current = fullPrompt 
  }, [stockStrategy])
  useEffect(() => { signalIntervalRef.current = signalInterval }, [signalInterval])
  useEffect(() => { stockStrategyRef.current = stockStrategy }, [stockStrategy])
  useEffect(() => { serverUrlRef.current = serverUrl }, [serverUrl])
  useEffect(() => { securitySystemPromptRef.current = securitySystemPrompt }, [securitySystemPrompt])

  const messages = activeTab === 'heart' ? heartMessages : activeTab === 'stock' ? stockMessages : securityMessages
  const setMessages = activeTab === 'heart' ? setHeartMessages : activeTab === 'stock' ? setStockMessages : setSecurityMessages
  const signalLog = activeTab === 'heart' ? hrSignalLog : activeTab === 'stock' ? stockSignalLog : securitySignalLog

  // ─── Scroll ───
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    userScrolledUpRef.current = false
    setShowScrollDown(false)
  }, [])

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [heartMessages, stockMessages, securityMessages, currentResponse])

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
  }, [hrSignalLog, stockSignalLog, securitySignalLog])

  // ─── HTTP SSE Communication ───

  const getBaseUrl = useCallback(() => {
    return serverUrlRef.current.replace(/\/+$/, '')
  }, [])

  // Parse inline trigger_alarm calls from text
  const parseInlineAlarm = useCallback((text: string) => {
    // Strip live camera update markers that can interrupt the trigger_alarm call
    const cleanText = text.replace(/\[LIVE CAMERA UPDATE:[^\]]*\]/g, '')
    const match = cleanText.match(/trigger_alarm\(["']([^"']+)["'],\s*([\d.]+)\)/)
    if (match) {
      const threatType = match[1]
      const confidence = parseFloat(match[2])
      return { threat_type: threatType, confidence }
    }
    return null
  }, [])

  const executeInlineAlarm = useCallback((args: { threat_type: string; confidence: number }) => {
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    const result = {
      status: 'ALARM_ACTIVE',
      threat_type: args.threat_type,
      confidence: args.confidence,
      timestamp: ts,
      message: `ALARM: ${args.threat_type}`,
    }

    const toolCall = { name: 'trigger_alarm', arguments: args, result, success: true, isExecuting: false }
    currentToolCallsRef.current = [...currentToolCallsRef.current, toolCall]
    setCurrentToolCalls([...currentToolCallsRef.current])

    setAlarmTriggered(true)
    setSecurityAlerts(prev => [...prev, {
      threat_type: args.threat_type,
      confidence: args.confidence,
      timestamp: ts,
    }])
  }, [])

  const alarmParsedRef = useRef(false)

  const handleSSEEvent = useCallback((data: any) => {
    switch (data.type) {
      case 'token':
        currentResponseRef.current += data.content
        setCurrentResponse(currentResponseRef.current)

        // Check for inline trigger_alarm in text (only parse once per response)
        if (!alarmParsedRef.current && activeTabRef.current === 'security') {
          const alarmArgs = parseInlineAlarm(currentResponseRef.current)
          if (alarmArgs) {
            alarmParsedRef.current = true
            executeInlineAlarm(alarmArgs)
          }
        }
        break

      case 'done':
        setIsGenerating(false)
        const finalResponse = currentResponseRef.current
        const finalToolCalls = [...currentToolCallsRef.current]
        if (finalResponse.trim() || finalToolCalls.length > 0) {
          const msg: Message = {
            role: 'assistant',
            content: finalResponse,
            toolCalls: finalToolCalls.length > 0 ? finalToolCalls : undefined
          }
          if (activeTabRef.current === 'heart') setHeartMessages(prev => [...prev, msg])
          else if (activeTabRef.current === 'stock') setStockMessages(prev => [...prev, msg])
          else setSecurityMessages(prev => [...prev, msg])
        }
        currentResponseRef.current = ''
        currentToolCallsRef.current = []
        setCurrentResponse('')
        setCurrentToolCalls([])
        break

      case 'tool_call': {
        const newToolCall = { name: data.name, arguments: data.arguments, isExecuting: true }
        currentToolCallsRef.current = [...currentToolCallsRef.current, newToolCall]
        setCurrentToolCalls([...currentToolCallsRef.current])
        break
      }

      case 'tool_result':
        currentToolCallsRef.current = currentToolCallsRef.current.map(tc =>
          tc.name === data.name && tc.isExecuting
            ? { ...tc, result: data.result, success: data.success, error: data.error, isExecuting: false }
            : tc
        )
        setCurrentToolCalls([...currentToolCallsRef.current])

        if (data.name === 'call_911' && data.success) {
          if (data.result?.status === 'EMERGENCY_DISPATCHED') {
            setEmergency911Called(true)
          }
        }
        if ((data.name === 'buy_stock' || data.name === 'sell_stock') && data.success && data.result) {
          const trade: TradeEntry = {
            action: data.result.action,
            quantity: data.result.quantity,
            price: data.result.price || stockRef.current.price,
            timestamp: data.result.timestamp,
            pnl: data.result.pnl,
            shares_held: data.result.shares_held,
            cash_remaining: data.result.cash_remaining,
          }
          setTrades(prev => [trade, ...prev].slice(0, 20))
          if (trade.shares_held !== undefined) setPortfolioShares(trade.shares_held)
          if (trade.cash_remaining !== undefined) setPortfolioCash(trade.cash_remaining)
        }
        if (data.name === 'trigger_alarm' && data.success && data.result?.status === 'ALARM_ACTIVE') {
          setAlarmTriggered(true)
          setSecurityAlerts(prev => [...prev, {
            threat_type: data.result.threat_type,
            confidence: data.result.confidence,
            timestamp: data.result.timestamp,
          }])
        }
        break

      case 'signal_injected':
        {
          const prevTCs = [...currentToolCallsRef.current]
          if (prevTCs.length > 0) {
            const prevResp = currentResponseRef.current
            const msg: Message = {
              role: 'assistant',
              content: prevResp,
              toolCalls: prevTCs,
            }
            if (activeTabRef.current === 'heart') setHeartMessages(prev => [...prev, msg])
            else if (activeTabRef.current === 'stock') setStockMessages(prev => [...prev, msg])
            else setSecurityMessages(prev => [...prev, msg])
            currentResponseRef.current = ''
            currentToolCallsRef.current = []
            setCurrentResponse('')
            setCurrentToolCalls([])
          }
          else if (currentResponseRef.current.trim()) {
            currentResponseRef.current += '\n\n---\n\n'
            setCurrentResponse(currentResponseRef.current)
          }
          // Reset alarm parsing for next frame
          alarmParsedRef.current = false
        }
        break

      case 'error':
        console.error(data.message || data.content)
        setIsGenerating(false)
        break
    }
  }, [parseInlineAlarm, executeInlineAlarm])

  const readSSEStream = useCallback(async (response: Response, signal: AbortSignal) => {
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done || signal.aborted) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            handleSSEEvent(data)
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') console.error('[SSE]', e)
    }
  }, [handleSSEEvent])

  const checkServer = useCallback(async () => {
    const url = getBaseUrl()
    if (!url) { setIsConnected(false); return }
    try {
      const res = await fetch(`${url}/status`, { signal: AbortSignal.timeout(5000) })
      setIsConnected(res.ok)
    } catch {
      setIsConnected(false)
    }
  }, [getBaseUrl])

  useEffect(() => {
    if (serverUrl) checkServer()
    return () => {
      if (monitorIntervalRef.current) clearInterval(monitorIntervalRef.current)
    }
  }, [serverUrl, checkServer])

  // ─── Actions ───

  const startMonitoring = async () => {
    const url = getBaseUrl()
    if (!url || !isConnected) return

    setIsMonitoring(true)
    isMonitoringRef.current = true
    setEmergency911Called(false)
    elapsedRef.current = 0

    const tab = activeTabRef.current
    const interval = signalIntervalRef.current
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    let initialPrompt: string
    let systemPrompt: string
    let mode: string

    if (tab === 'heart') {
      setHrSignalLog([])
      initialPrompt = `Patient monitoring session started. I will send you heart rate updates every ${interval} seconds via live signals. Acknowledge each update briefly. Begin monitoring.`
      systemPrompt = hrSystemPromptRef.current
      mode = 'heartbeat'
      setHeartMessages([{ role: 'user', content: 'Starting heart rate monitoring session...' }])
    } else if (tab === 'stock') {
      setStockSignalLog([])
      setStock(INITIAL_STOCK)
      stockRef.current = INITIAL_STOCK
      setTrades([])
      setPortfolioCash(10000)
      setPortfolioShares(0)
      fetch(`${url}/reset_portfolio`, { method: 'POST' }).catch(() => {})
      initialPrompt = `Trading session started. I will send you live AAPL price updates every ${interval} seconds. You start with $10,000 cash and 0 shares. Use buy_stock(quantity) and sell_stock(quantity) to trade. Begin monitoring.`
      systemPrompt = stockSystemPromptRef.current
      mode = 'stock'
      setStockMessages([{ role: 'user', content: 'Starting AAPL trading session...' }])
    } else {
      // Security tab
      setSecuritySignalLog([])
      setAlarmTriggered(false)
      setSecurityAlerts([])
      setFrameCount(0)
      initialPrompt = `Security monitoring session started. I will send you live camera frames. Analyze each frame for threats. Use trigger_alarm if you detect a confirmed threat. Begin monitoring.`
      systemPrompt = securitySystemPromptRef.current
      mode = 'security'
      setSecurityMessages([{ role: 'user', content: 'Starting security camera monitoring...' }])
    }

    setIsGenerating(true)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    setCurrentToolCalls([])
    alarmParsedRef.current = false

    // Start signal interval — sends data to /signal via HTTP
    if (tab === 'heart') {
      monitorIntervalRef.current = setInterval(() => {
        if (!isMonitoringRef.current) return
        elapsedRef.current += signalIntervalRef.current
        const bpm = generateHeartRate(elapsedRef.current, hrScenarioRef.current)
        setCurrentHeartRate(bpm)
        const { status } = getHeartRateStatus(bpm)
        setHrHistory(prev => [...prev.slice(-59), { time: new Date().toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }), bpm }])
        setHrSignalLog(prev => [...prev, { value: `${bpm} BPM`, label: status, timestamp: new Date() }])
        const content = HR_SIGNAL_TEMPLATE.replace('{bpm}', String(bpm)).replace('{status}', status)
        fetch(`${url}/signal`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, priority: status === 'CRITICAL' ? 2.0 : 1.0 }),
        }).catch(() => {})
      }, interval * 1000)
    } else if (tab === 'stock') {
      monitorIntervalRef.current = setInterval(() => {
        if (!isMonitoringRef.current) return
        elapsedRef.current += signalIntervalRef.current
        const updated = generateStockPrice(stockRef.current, stockScenarioRef.current, elapsedRef.current)
        setStock(updated)
        stockRef.current = updated
        setStockHistory(prev => [...prev.slice(-59), {
          time: new Date().toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }),
          AAPL: +((updated.price - updated.basePrice) / updated.basePrice * 100).toFixed(2),
        }])

        // Update server-side price
        fetch(`${url}/update_price`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ price: updated.price }),
        }).catch(() => {})

        const ch = formatStockChange(updated.price, updated.prevPrice)
        const priceStr = `AAPL: $${updated.price.toFixed(2)} (${ch.pct})`

        setStockSignalLog(prev => [...prev, { value: priceStr, label: stockScenarioRef.current, timestamp: new Date() }])
        const content = STOCK_SIGNAL_TEMPLATE.replace('{prices}', priceStr)
        fetch(`${url}/signal`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, priority: 1.0 }),
        }).catch(() => {})
      }, interval * 1000)
    } else {
      // Security: start webcam and send frames
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        camStreamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setIsCamActive(true)

        // Capture frames every 4 seconds and send to vision endpoint
        const canvas = document.createElement('canvas')
        canvas.width = 384
        canvas.height = 384
        const ctx = canvas.getContext('2d')!

        camIntervalRef.current = setInterval(() => {
          if (!isMonitoringRef.current || !videoRef.current) return
          ctx.drawImage(videoRef.current, 0, 0, 384, 384)
          const dataUrl = canvas.toDataURL('image/jpeg', 0.6)
          const b64 = dataUrl.split(',')[1]

          setFrameCount(prev => prev + 1)
          const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          setSecuritySignalLog(prev => [...prev, { value: `Frame sent`, label: ts, timestamp: new Date() }])

          fetch(`${url}/signal_vision`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: b64, text: `[SECURITY CAMERA FRAME at ${ts}]` }),
          }).catch(() => {})
        }, 4000) // every 4 seconds
      } catch (e) {
        console.error('Camera access failed:', e)
      }
    }

    // Start SSE stream
    try {
      let endpoint: string
      let body: any

      if (tab === 'security') {
        endpoint = `${url}/chat_vision`
        body = { system_prompt: systemPrompt, message: initialPrompt }
      } else {
        endpoint = `${url}/chat`
        body = { message: initialPrompt, system_prompt: systemPrompt, mode, monitor: true }
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortController.signal,
      })
      await readSSEStream(res, abortController.signal)
    } catch (e: any) {
      if (e.name !== 'AbortError') console.error('[Monitor]', e)
    }
    setIsGenerating(false)
  }

  const stopMonitoring = () => {
    setIsMonitoring(false)
    isMonitoringRef.current = false
    if (monitorIntervalRef.current) {
      clearInterval(monitorIntervalRef.current)
      monitorIntervalRef.current = null
    }
    if (camIntervalRef.current) {
      clearInterval(camIntervalRef.current)
      camIntervalRef.current = null
    }
    if (camStreamRef.current) {
      camStreamRef.current.getTracks().forEach(t => t.stop())
      camStreamRef.current = null
      setIsCamActive(false)
    }
    abortControllerRef.current?.abort()
    const url = getBaseUrl()
    if (url) {
      fetch(`${url}/stop`, { method: 'POST' }).catch(() => {})
      fetch(`${url}/stop_vision`, { method: 'POST' }).catch(() => {})
    }
  }

  const resetSession = () => {
    stopMonitoring()
    if (activeTab === 'heart') {
      setHeartMessages([])
      setHrSignalLog([])
      setHrHistory([])
      setEmergency911Called(false)
      setCurrentHeartRate(75)
    } else if (activeTab === 'stock') {
      setStockMessages([])
      setStockSignalLog([])
      setStockHistory([])
      setStock(INITIAL_STOCK)
      stockRef.current = INITIAL_STOCK
      setTrades([])
      setPortfolioCash(10000)
      setPortfolioShares(0)
    } else {
      setSecurityMessages([])
      setSecuritySignalLog([])
      setSecurityAlerts([])
      setAlarmTriggered(false)
      setFrameCount(0)
    }
    setCurrentResponse('')
    setCurrentToolCalls([])
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    isMonitoringRef.current = false
    elapsedRef.current = 0
  }

  const sendUserMessage = async () => {
    const url = getBaseUrl()
    if (!url || !userPrompt.trim() || !isConnected) return
    const msg: Message = { role: 'user', content: userPrompt }
    if (activeTab === 'heart') setHeartMessages(prev => [...prev, msg])
    else if (activeTab === 'stock') setStockMessages(prev => [...prev, msg])
    else setSecurityMessages(prev => [...prev, msg])
    setIsGenerating(true)
    currentResponseRef.current = ''
    currentToolCallsRef.current = []
    setCurrentToolCalls([])

    const prompt = userPrompt
    setUserPrompt('')

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      const sysPrompt = activeTab === 'heart' ? hrSystemPromptRef.current : activeTab === 'stock' ? stockSystemPromptRef.current : securitySystemPromptRef.current
      const mode = activeTab === 'heart' ? 'heartbeat' : activeTab === 'stock' ? 'stock' : 'security'
      const res = await fetch(`${url}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          system_prompt: sysPrompt,
          mode,
          monitor: false,
        }),
        signal: abortController.signal,
      })
      await readSSEStream(res, abortController.signal)
    } catch (e: any) {
      if (e.name !== 'AbortError') console.error('[Chat]', e)
    }
    setIsGenerating(false)
  }

  const sendManualSignal = (content: string) => {
    const url = getBaseUrl()
    if (!url || !content.trim()) return
    fetch(`${url}/signal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, priority: 1.5 }),
    }).catch(() => {})
    const logSetter = activeTab === 'heart' ? setHrSignalLog : activeTab === 'stock' ? setStockSignalLog : setSecuritySignalLog
    logSetter(prev => [...prev, { value: content, label: 'MANUAL', timestamp: new Date() }])
  }

  const handleTabSwitch = (tab: 'heart' | 'stock' | 'security') => {
    if (tab === activeTab) return
    if (isMonitoring) stopMonitoring()
    setActiveTab(tab)
  }

  // ─── Derived ───
  const hrStatus = getHeartRateStatus(currentHeartRate)
  
  // Calculate total P&L from trade history
  const totalProfit = trades.reduce((sum, t) => {
    if (t.action === 'SELL') return sum + (t.quantity * t.price)
    return sum - (t.quantity * t.price)
  }, 0)
  
  // Unrealized P&L from current holdings
  const avgBuyPrice = trades.filter(t => t.action === 'BUY').length > 0
    ? trades.filter(t => t.action === 'BUY').reduce((s, t) => s + t.price * t.quantity, 0) / trades.filter(t => t.action === 'BUY').reduce((s, t) => s + t.quantity, 0)
    : 0
  const unrealizedProfit = portfolioShares > 0 ? portfolioShares * (stock.price - avgBuyPrice) : 0

  // ═══════════════════════════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════════════════════════

  return (
    <div className="h-screen bg-white text-gray-900 flex flex-col overflow-hidden font-sans selection:bg-blue-100">
      {/* Emergency Banner */}
      {emergency911Called && (
        <div className="bg-red-600 text-white px-4 py-2 flex items-center justify-center gap-2 shrink-0 text-sm font-medium">
          <Phone className="w-4 h-4 animate-pulse" />
          <span>Emergency services dispatched</span>
        </div>
      )}
      {/* Security Alarm Banner — only show on security tab */}
      {alarmTriggered && activeTab === 'security' && (
        <div className="bg-red-600 text-white px-4 py-2 flex items-center justify-center gap-2 shrink-0 text-sm font-medium animate-pulse">
          <ShieldAlert className="w-4 h-4" />
          <span>SECURITY ALARM TRIGGERED — Threat detected on camera</span>
        </div>
      )}

      {/* Header */}
      <header className="border-b border-gray-200 px-6 py-3 flex items-center justify-between shrink-0 bg-white z-20">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-900 border border-blue-800">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-gray-900">
                Nexus AI
              </h1>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[11px] font-medium text-gray-400">Workspace</span>
              </div>
            </div>
          </div>
          
          <div className="h-4 w-px bg-gray-200 hidden sm:block" />
          
          <div className={`hidden sm:flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-md font-medium border transition-colors ${
            isConnected 
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
              : 'bg-gray-50 text-gray-400 border-gray-200'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-gray-400'}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isConnected && (
            <div className="hidden md:flex items-center gap-2 px-2 py-1 bg-gray-50 border border-gray-200 rounded-md">
              <span className="text-[11px] font-mono text-gray-500">Qwen3-14B</span>
            </div>
          )}
          <button onClick={() => setShowSettings(s => !s)}
            className={`p-2 rounded-lg transition-colors ${
              showSettings 
                ? 'bg-blue-50 text-blue-900' 
                : 'text-gray-400 hover:text-gray-900 hover:bg-gray-100'
            }`}>
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Connection Banner */}
      {!isConnected && (
        <div className="border-b border-gray-200 bg-gray-50 px-6 py-3 shrink-0 z-10">
          <div className="max-w-2xl flex gap-2 items-center">
            <div className="flex-1 w-full relative">
              <input
                type="text"
                value={serverUrl}
                onChange={e => setServerUrl(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') checkServer() }}
                placeholder="Enter Colab tunnel URL (e.g., https://xxxx.trycloudflare.com)"
                className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 font-mono transition-colors"
              />
            </div>
            <button
              onClick={checkServer}
              disabled={!serverUrl.trim()}
              className="px-4 py-2 bg-blue-900 hover:bg-blue-800 text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors whitespace-nowrap"
            >
              Connect Server
            </button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white border-b border-gray-200 px-6 flex gap-4 shrink-0 z-10 pt-2">
        <button onClick={() => handleTabSwitch('heart')}
          className={`py-2 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'heart'
              ? 'border-blue-900 text-blue-900'
              : 'border-transparent text-gray-400 hover:text-gray-600'
          }`}>
          Vitals Monitor
        </button>
        <button onClick={() => handleTabSwitch('stock')}
          className={`py-2 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'stock'
              ? 'border-blue-900 text-blue-900'
              : 'border-transparent text-gray-400 hover:text-gray-600'
          }`}>
          Algorithmic Trading
        </button>
        <button onClick={() => handleTabSwitch('security')}
          className={`py-2 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'security'
              ? 'border-blue-900 text-blue-900'
              : 'border-transparent text-gray-400 hover:text-gray-600'
          }`}>
          Security Detection
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex min-h-0 bg-white">

        {/* Left Sidebar - Instrumentation */}
        <aside className="w-[320px] border-r border-gray-200 flex flex-col shrink-0 bg-gray-50/50 overflow-y-auto z-10">

          {/* ─── Heart Rate Sidebar ─── */}
          {activeTab === 'heart' && (
            <div className="flex flex-col h-full fade-in">
              {/* Vitals Card */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-medium text-gray-500">Current Heart Rate</span>
                  <div className={`flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-md border ${
                    hrStatus.urgent ? 'bg-red-50 text-red-600 border-red-200' : 
                    hrStatus.status === 'NORMAL' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                    'bg-amber-50 text-amber-700 border-amber-200'
                  }`}>
                    {hrStatus.urgent && <AlertTriangle className="w-3 h-3" />}
                    {hrStatus.status}
                  </div>
                </div>
                
                <div className="flex items-end gap-2">
                  <span className={`text-4xl font-semibold tabular-nums leading-none tracking-tight ${
                    hrStatus.urgent ? 'text-red-600' : 'text-gray-900'
                  }`}>
                    {currentHeartRate}
                  </span>
                  <span className="text-sm font-medium text-gray-400 mb-0.5">BPM</span>
                </div>
              </div>

              {/* Heart Rate Chart */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs font-medium text-gray-500">Timeline</span>
                </div>
                <div className="h-[120px] -ml-2">
                  {hrHistory.length > 1 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsLineChart data={hrHistory} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                        <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} interval="preserveStartEnd" dy={5} />
                        <YAxis domain={['dataMin - 10', 'dataMax + 10']} tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} width={30} />
                        <Tooltip
                          contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                          itemStyle={{ color: '#1f2937' }}
                          labelStyle={{ color: '#6b7280', marginBottom: '4px', fontSize: '11px' }}
                          formatter={(value: any) => [`${value} BPM`, 'Rate']}
                          cursor={{ stroke: '#d1d5db', strokeWidth: 1, strokeDasharray: '4 4' }}
                        />
                        <ReferenceLine y={170} stroke="#ef4444" strokeDasharray="2 2" strokeOpacity={0.5} />
                        <ReferenceLine y={40} stroke="#3b82f6" strokeDasharray="2 2" strokeOpacity={0.5} />
                        <Line
                          type="monotone" dataKey="bpm" stroke={hrStatus.urgent ? '#dc2626' : '#059669'}
                          strokeWidth={2} dot={false}
                          isAnimationActive={false}
                        />
                      </RechartsLineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-400">
                      Awaiting data...
                    </div>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div className="p-5 border-b border-gray-200 space-y-4">
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-gray-500 mb-1.5 block">
                      Scenario Profile
                    </label>
                    <select value={hrScenario} onChange={e => setHrScenario(e.target.value)}
                      className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-sm text-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 appearance-none cursor-pointer">
                      <option value="normal">Normal Resting (60-80 BPM)</option>
                      <option value="exercise">Exercise / Stress (120-140 BPM)</option>
                      <option value="critical">Critical Emergency (&gt;170 BPM)</option>
                      <option value="bradycardia">Bradycardia (&lt;40 BPM)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="text-xs font-medium text-gray-500 mb-1.5 flex items-center justify-between">
                      <span>Update Frequency</span>
                      <span className="text-gray-400">{signalInterval}s</span>
                    </label>
                    <input type="range" min={1} max={10} value={signalInterval}
                      onChange={e => setSignalInterval(Number(e.target.value))} disabled={isMonitoring}
                      className="w-full accent-blue-900 disabled:opacity-50" />
                  </div>
                </div>

                <div className="flex gap-2 pt-1">
                  {!isMonitoring ? (
                    <button onClick={startMonitoring} disabled={!isConnected}
                      className="flex-1 px-3 py-2 bg-blue-900 hover:bg-blue-800 text-white rounded-md text-sm font-medium disabled:opacity-50 transition-colors">
                      Start Stream
                    </button>
                  ) : (
                    <button onClick={stopMonitoring}
                      className="flex-1 px-3 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md text-sm font-medium transition-colors">
                      Stop Stream
                    </button>
                  )}
                  <button onClick={resetSession}
                    className="px-3 py-2 bg-white hover:bg-gray-100 border border-gray-300 rounded-md text-sm font-medium transition-colors text-gray-600">
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ─── Stock Trading Sidebar ─── */}
          {activeTab === 'stock' && (
            <div className="flex flex-col h-full fade-in">
              {/* P&L Card */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-medium text-gray-500">Portfolio Return</span>
                  {portfolioShares > 0 && (
                    <div className="flex items-center gap-1 text-[11px] font-medium text-gray-400">
                      AAPL:{portfolioShares}
                    </div>
                  )}
                </div>
                
                {trades.length > 0 ? (
                  <div className="space-y-4">
                    <div className="flex items-end gap-2">
                      <span className={`text-4xl font-semibold tabular-nums leading-none tracking-tight ${
                        (totalProfit + unrealizedProfit) >= 0 ? 'text-gray-900' : 'text-red-600'
                      }`}>
                        {(totalProfit + unrealizedProfit) >= 0 ? '+' : ''}{(totalProfit + unrealizedProfit).toFixed(2)}
                      </span>
                      <span className="text-sm font-medium text-gray-400 mb-0.5">USD</span>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-white rounded-md px-3 py-2 border border-gray-200">
                        <div className="text-[10px] text-gray-400 font-medium mb-1">Realized</div>
                        <div className={`text-xs font-mono ${totalProfit >= 0 ? 'text-gray-700' : 'text-red-600'}`}>
                          {totalProfit >= 0 ? '+' : ''}{totalProfit.toFixed(2)}
                        </div>
                      </div>
                      <div className="bg-white rounded-md px-3 py-2 border border-gray-200">
                        <div className="text-[10px] text-gray-400 font-medium mb-1">Unrealized</div>
                        <div className={`text-xs font-mono ${unrealizedProfit >= 0 ? 'text-gray-700' : 'text-red-600'}`}>
                          {unrealizedProfit >= 0 ? '+' : ''}{unrealizedProfit.toFixed(2)}
                        </div>
                      </div>
                      <div className="bg-white rounded-md px-3 py-2 border border-gray-200">
                        <div className="text-[10px] text-gray-400 font-medium mb-1">Trades</div>
                        <div className="text-xs font-mono text-gray-700">{trades.length}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-gray-400 py-4">
                    Awaiting execution...
                  </div>
                )}
              </div>

              {/* Live Ticker — Single Stock */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center mb-3">
                  <span className="text-xs font-medium text-gray-500">Market Data</span>
                </div>
                {(() => {
                  const ch = formatStockChange(stock.price, stock.prevPrice)
                  const fromBase = ((stock.price - stock.basePrice) / stock.basePrice * 100)
                  const positionPnL = portfolioShares > 0 ? (stock.price - avgBuyPrice) * portfolioShares : null
                  return (
                    <div className="bg-white rounded-lg border border-gray-200 p-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-semibold text-gray-800">AAPL</div>
                          <div className={`text-[11px] font-medium ${fromBase >= 0 ? 'text-gray-400' : 'text-red-500'}`}>
                            {fromBase >= 0 ? '+' : ''}{fromBase.toFixed(2)}% session
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono font-medium text-gray-900">${stock.price.toFixed(2)}</div>
                          <div className={`text-[11px] font-medium flex items-center justify-end gap-1 ${
                            ch.pct.startsWith('+') ? 'text-emerald-600' : ch.pct.startsWith('-') ? 'text-red-600' : 'text-gray-400'
                          }`}>
                            <span>{ch.pct}</span>
                          </div>
                        </div>
                      </div>
                      {positionPnL !== null && (
                        <div className="mt-2 pt-2 border-t border-gray-100 flex items-center justify-between text-[11px] font-medium">
                          <span className="text-gray-400">Position ({portfolioShares})</span>
                          <span className={`font-mono ${positionPnL >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                            {positionPnL >= 0 ? '+' : ''}${positionPnL.toFixed(2)}
                          </span>
                        </div>
                      )}
                      <div className="mt-2 pt-2 border-t border-gray-100 flex items-center justify-between text-[11px] font-medium">
                        <span className="text-gray-400">Cash</span>
                        <span className="font-mono text-gray-600">${portfolioCash.toFixed(2)}</span>
                      </div>
                    </div>
                  )
                })()}
              </div>

              {/* Stock Price Chart — Single Stock */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center mb-3">
                  <span className="text-xs font-medium text-gray-500">AAPL Trend</span>
                </div>
                <div className="h-[140px] -ml-2">
                  {stockHistory.length > 1 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsLineChart data={stockHistory} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                        <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} interval="preserveStartEnd" dy={5} />
                        <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} width={40}
                          tickFormatter={(v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`}
                          domain={['auto', 'auto']} />
                        <Tooltip
                          contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                          itemStyle={{ color: '#1f2937' }}
                          labelStyle={{ color: '#6b7280', marginBottom: '4px', fontSize: '11px' }}
                          formatter={(value: any) => [`${Number(value) > 0 ? '+' : ''}${Number(value).toFixed(2)}%`, 'AAPL']}
                          cursor={{ stroke: '#d1d5db', strokeWidth: 1, strokeDasharray: '4 4' }}
                        />
                        <ReferenceLine y={0} stroke="#d1d5db" strokeWidth={1} />
                        <Line type="monotone" dataKey="AAPL" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                      </RechartsLineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-400">
                      Awaiting data...
                    </div>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div className="p-5 border-b border-gray-200 space-y-4">
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-gray-500 mb-1.5 block">
                      Market Condition
                    </label>
                    <select value={stockScenario} onChange={e => setStockScenario(e.target.value)}
                      className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-sm text-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 appearance-none cursor-pointer">
                      <option value="stable">Stable Market</option>
                      <option value="dip">Gradual Dip</option>
                      <option value="crash">Market Crash</option>
                      <option value="rally">Bull Rally</option>
                      <option value="volatile">High Volatility</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="text-xs font-medium text-gray-500 mb-1.5 flex items-center justify-between">
                      <span>Tick Interval</span>
                      <span className="text-gray-400">{signalInterval}s</span>
                    </label>
                    <input type="range" min={1} max={10} value={signalInterval}
                      onChange={e => setSignalInterval(Number(e.target.value))} disabled={isMonitoring}
                      className="w-full accent-blue-900 disabled:opacity-50" />
                  </div>
                </div>

                <div className="flex gap-2 pt-1">
                  {!isMonitoring ? (
                    <button onClick={startMonitoring} disabled={!isConnected}
                      className="flex-1 px-3 py-2 bg-blue-900 hover:bg-blue-800 text-white rounded-md text-sm font-medium disabled:opacity-50 transition-colors">
                      Start Trading
                    </button>
                  ) : (
                    <button onClick={stopMonitoring}
                      className="flex-1 px-3 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md text-sm font-medium transition-colors">
                      Stop Trading
                    </button>
                  )}
                  <button onClick={resetSession}
                    className="px-3 py-2 bg-white hover:bg-gray-100 border border-gray-300 rounded-md text-sm font-medium transition-colors text-gray-600">
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Trade Log */}
              {trades.length > 0 && (
                <div className="p-5 border-b border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-gray-500">Trade Ledger</span>
                    <span className="text-[10px] text-gray-400">{trades.length} events</span>
                  </div>
                  <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                    {trades.map((t, i) => (
                      <div key={i} className="text-xs p-2.5 rounded-lg border border-gray-200 bg-white">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className={`font-semibold ${t.action === 'BUY' ? 'text-emerald-600' : 'text-orange-600'}`}>{t.action}</span>
                            <span className="text-gray-700">{t.quantity}× AAPL</span>
                          </div>
                          <div className="text-right">
                            <div className="font-mono text-gray-700">${t.price.toFixed(2)}</div>
                          </div>
                        </div>
                        <div className="flex items-center justify-between mt-1 text-[10px] text-gray-400">
                          <span className="font-mono">{t.timestamp}</span>
                          <span>${(t.quantity * t.price).toFixed(2)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ─── Security Sidebar ─── */}
          {activeTab === 'security' && (
            <div className="flex flex-col h-full fade-in">
              {/* Camera Preview */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-medium text-gray-500">Camera Feed</span>
                  <div className={`flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-md border ${
                    isCamActive ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-gray-50 text-gray-400 border-gray-200'
                  }`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isCamActive ? 'bg-emerald-500 animate-pulse' : 'bg-gray-400'}`} />
                    {isCamActive ? 'Live' : 'Offline'}
                  </div>
                </div>
                <div className="relative rounded-lg overflow-hidden bg-gray-900 aspect-video">
                  <video ref={videoRef} className="w-full h-full object-cover" muted playsInline autoPlay />
                  {!isCamActive && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
                      <VideoOff className="w-8 h-8 mb-2" />
                      <span className="text-xs">Camera inactive</span>
                    </div>
                  )}
                  {isCamActive && (
                    <div className="absolute top-2 right-2 bg-red-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded animate-pulse">
                      REC
                    </div>
                  )}
                </div>
                {isCamActive && (
                  <div className="mt-2 text-[11px] text-gray-400 font-mono text-center">
                    {frameCount} frames sent (every 4s)
                  </div>
                )}
              </div>

              {/* Alerts */}
              <div className="p-5 border-b border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-medium text-gray-500">Security Status</span>
                  <div className={`flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-md border ${
                    alarmTriggered ? 'bg-red-50 text-red-600 border-red-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  }`}>
                    {alarmTriggered ? <ShieldAlert className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
                    {alarmTriggered ? 'ALARM' : 'Clear'}
                  </div>
                </div>
                {securityAlerts.length > 0 ? (
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {securityAlerts.map((a, i) => (
                      <div key={i} className="text-xs p-2.5 rounded-lg border border-red-200 bg-red-50">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-red-700">{a.threat_type}</span>
                          <span className="font-mono text-red-500">{(a.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div className="text-[10px] text-red-400 mt-1 font-mono">{a.timestamp}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-400 py-4">No threats detected</div>
                )}
              </div>

              {/* Controls */}
              <div className="p-5 border-b border-gray-200 space-y-4">
                <div className="flex gap-2 pt-1">
                  {!isMonitoring ? (
                    <button onClick={startMonitoring} disabled={!isConnected}
                      className="flex-1 px-3 py-2 bg-blue-900 hover:bg-blue-800 text-white rounded-md text-sm font-medium disabled:opacity-50 transition-colors">
                      Start Monitoring
                    </button>
                  ) : (
                    <button onClick={stopMonitoring}
                      className="flex-1 px-3 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md text-sm font-medium transition-colors">
                      Stop Monitoring
                    </button>
                  )}
                  <button onClick={resetSession}
                    className="px-3 py-2 bg-white hover:bg-gray-100 border border-gray-300 rounded-md text-sm font-medium transition-colors text-gray-600">
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ─── Signal Log (shared) ─── */}
          <div className="flex-1 flex flex-col min-h-0">
            <button onClick={() => setShowSignalLog(s => !s)}
              className="px-5 py-3 flex items-center justify-between text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors sticky top-0 bg-gray-50/90 backdrop-blur-md border-b border-gray-200 z-10">
              <span className="flex items-center gap-2">
                <Zap className="w-3.5 h-3.5" />
                Signals
                <span className="bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded text-[10px]">{signalLog.length}</span>
              </span>
              <ChevronDown className={`w-4 h-4 transition-transform ${showSignalLog ? 'rotate-180' : ''}`} />
            </button>
            
            <div className={`flex-1 overflow-y-auto ${showSignalLog ? 'block' : 'hidden'}`}>
              <div className="p-3 space-y-1">
                {signalLog.length === 0 ? (
                  <div className="text-center py-6 text-xs text-gray-400">No signals recorded</div>
                ) : (
                  signalLog.map((entry, i) => (
                    <div key={i} className="text-xs p-2 rounded-md hover:bg-gray-100 transition-colors">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-semibold text-gray-400">{entry.label}</span>
                        <span className="text-[10px] font-mono text-gray-400">
                          {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                        </span>
                      </div>
                      <div className="text-gray-600 font-mono text-[11px] leading-relaxed break-words">
                        {entry.value}
                      </div>
                    </div>
                  ))
                )}
                <div ref={signalLogEndRef} className="h-1" />
              </div>
            </div>
          </div>
        </aside>

        {/* ════════════════ MAIN CHAT AREA ════════════════ */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0 bg-white">

          {/* Settings Panel Overlay */}
          {showSettings && (
            <div className="border-b border-gray-200 bg-gray-50 p-6 z-20">
              <div className="max-w-3xl mx-auto space-y-6">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-sm font-semibold text-gray-900">System Configuration</h2>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="text-xs font-medium text-gray-500 mb-1.5 block">
                        Server Endpoint
                      </label>
                      <input
                        type="text"
                        value={serverUrl}
                        onChange={e => { setServerUrl(e.target.value); setIsConnected(false) }}
                        placeholder="https://xxxx.trycloudflare.com"
                        className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm text-gray-900 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 font-mono transition-colors"
                      />
                    </div>
                    
                    {activeTab === 'stock' && (
                      <div>
                        <label className="text-xs font-medium text-gray-500 mb-1.5 block">
                          Trading Strategy
                        </label>
                        <textarea
                          value={stockStrategy}
                          onChange={e => setStockStrategy(e.target.value)}
                          disabled={isMonitoring}
                          rows={3}
                          placeholder="E.g., buy on 3% dips, focus on AAPL..."
                          className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm text-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 resize-y transition-colors"
                        />
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <label className="text-xs font-medium text-gray-500 mb-1.5 block">
                      System Prompt ({activeTab === 'heart' ? 'Medical' : activeTab === 'stock' ? 'Trading' : 'Security'})
                    </label>
                    <textarea
                      value={activeTab === 'heart' ? hrSystemPrompt : activeTab === 'stock' ? stockSystemPrompt : securitySystemPrompt}
                      onChange={e => activeTab === 'heart' ? setHrSystemPrompt(e.target.value) : activeTab === 'stock' ? setStockSystemPrompt(e.target.value) : setSecuritySystemPrompt(e.target.value)}
                      disabled={isMonitoring}
                      rows={7}
                      className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg text-xs text-gray-600 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 resize-y transition-colors font-mono"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 min-h-0 relative">
            <div ref={messagesContainerRef} onScroll={handleScroll}
              className="absolute inset-0 overflow-y-auto px-6 py-6 scroll-smooth">
              <div className="max-w-3xl mx-auto w-full pb-4">
                {messages.length === 0 && !isGenerating && (
                  <div className="flex flex-col items-center justify-center py-32 text-center fade-in">
                    <div className="w-16 h-16 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center mb-6">
                      {activeTab === 'heart'
                        ? <Heart className="w-6 h-6 text-blue-900" />
                        : activeTab === 'stock'
                        ? <BarChart3 className="w-6 h-6 text-blue-900" />
                        : <Shield className="w-6 h-6 text-blue-900" />
                      }
                    </div>
                    <h2 className="text-lg font-semibold text-gray-900 mb-2">
                      {activeTab === 'heart' ? 'Vitals Monitoring' : activeTab === 'stock' ? 'Algorithmic Trading' : 'Security Detection'}
                    </h2>
                    <p className="text-sm text-gray-500 max-w-sm mx-auto">
                      {activeTab === 'heart'
                        ? 'Select a physiological scenario and start the stream. The model will monitor vitals and call 911 if critical.'
                        : activeTab === 'stock'
                        ? 'Select market conditions and start the stream. The model will analyze AAPL prices and execute trades autonomously.'
                        : 'Start monitoring to activate your camera. The vision model will analyze live frames and trigger alarms for threats.'
                      }
                    </p>
                  </div>
                )}

                <div className="space-y-6">
                  {messages.map((msg, i) => (
                    <ChatMessage key={i} role={msg.role} content={msg.content} toolCalls={msg.toolCalls} />
                  ))}

                  {(currentResponse || isGenerating) && (
                    <ChatMessage role="assistant" content={currentResponse} isGenerating={isGenerating}
                      toolCalls={currentToolCalls.length > 0 ? currentToolCalls : undefined} />
                  )}
                </div>

                <div ref={messagesEndRef} className="h-4" />
              </div>
            </div>

            {showScrollDown && (
              <button onClick={scrollToBottom}
                className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 rounded-full text-xs font-medium text-gray-600 flex items-center gap-2 shadow-lg transition-colors z-20">
                <ArrowDown className="w-3 h-3" /> Scroll to bottom
              </button>
            )}
          </div>

          {/* Input Area */}
          <div className="p-4 sm:p-6 shrink-0 bg-white border-t border-gray-200 z-20">
            <div className="max-w-3xl mx-auto">
              <div className="relative flex items-center bg-white border border-gray-300 rounded-xl focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-400 transition-all">
                <input type="text" value={userPrompt}
                  onChange={e => setUserPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendUserMessage() } }}
                  placeholder={isMonitoring ? 'Send a message during monitoring...' : 'Send a message...'}
                  disabled={!isConnected}
                  className="flex-1 bg-transparent px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none disabled:opacity-50 w-full" />
                <div className="pr-2">
                  <button onClick={sendUserMessage}
                    disabled={!userPrompt.trim() || !isConnected}
                    className="p-2 bg-blue-900 hover:bg-blue-800 text-white rounded-lg disabled:opacity-50 disabled:bg-gray-200 disabled:text-gray-400 flex items-center justify-center transition-colors shrink-0">
                    {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
