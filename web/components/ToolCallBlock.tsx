import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Phone, Siren, ShieldAlert, CheckCircle, XCircle, Loader2, TrendingUp, TrendingDown, DollarSign, Settings2 } from 'lucide-react'

interface ToolCallBlockProps {
  name: string
  arguments: Record<string, any>
  result?: any
  success?: boolean
  error?: string
  isExecuting?: boolean
}

const ToolCallBlock: React.FC<ToolCallBlockProps> = ({
  name,
  arguments: args,
  result,
  success,
  error,
  isExecuting = false
}) => {
  const is911Call = name === 'call_911'
  const [ringCount, setRingCount] = useState(0)

  // Animate ring pulses while executing
  useEffect(() => {
    if (!isExecuting || !is911Call) return
    const interval = setInterval(() => setRingCount(c => c + 1), 600)
    return () => clearInterval(interval)
  }, [isExecuting, is911Call])

  // ─── 911 Call: Dialing State ───
  if (is911Call && isExecuting) {
    return (
      <div className="my-4 rounded-xl border border-amber-300 bg-amber-50 px-5 py-4">
        <div className="flex items-center gap-4">
          <div className="p-2.5 rounded-lg bg-amber-100 border border-amber-200 text-amber-700">
            <Phone className="w-5 h-5 animate-pulse" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-widest text-amber-700 mb-0.5">
              Emergency Override
            </div>
            <div className="text-sm font-medium text-amber-900">
              Dialing 911...
            </div>
          </div>
          <Loader2 className="w-4 h-4 text-amber-700 animate-spin" />
        </div>
      </div>
    )
  }

  // ─── 911 Call: Dispatched State ───
  if (is911Call && success) {
    return (
      <div className="my-4 rounded-xl border border-red-300 bg-red-50 px-5 py-4">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-2.5 rounded-lg bg-red-100 border border-red-200 text-red-700">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-widest text-red-700 mb-0.5">
              Emergency Protocol
            </div>
            <div className="text-sm font-medium text-red-900">
              911 Dispatched
            </div>
          </div>
          <Siren className="w-5 h-5 text-red-600" />
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 bg-white rounded-lg p-3 border border-red-200">
            <div className="text-[10px] font-medium text-red-500 mb-1">Status</div>
            <div className="flex items-center gap-2 text-sm text-gray-800">
              <CheckCircle className="w-4 h-4 text-red-500" />
              <span>{result?.message || 'Emergency services en route'}</span>
            </div>
          </div>
          {result?.timestamp && (
            <div className="bg-white rounded-lg p-3 border border-red-200 min-w-[120px]">
              <div className="text-[10px] font-medium text-red-500 mb-1">Timestamp</div>
              <div className="text-sm font-mono text-gray-800">{result.timestamp}</div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // ─── 911 Call: Error State ───
  if (is911Call && error) {
    return (
      <div className="my-4 rounded-xl border border-red-300 bg-red-50 px-5 py-4">
        <div className="flex items-center gap-3 text-red-700">
          <XCircle className="w-5 h-5 shrink-0" />
          <div className="text-sm font-medium">Failed to dispatch 911: {error}</div>
        </div>
      </div>
    )
  }

  // ─── Buy Stock ───
  const isBuy = name === 'buy_stock'
  const isSell = name === 'sell_stock'
  const isTrade = isBuy || isSell

  if (isTrade && isExecuting) {
    return (
      <div className={`my-4 rounded-xl border px-5 py-4 ${
        isBuy ? 'border-emerald-300 bg-emerald-50' : 'border-amber-300 bg-amber-50'
      }`}>
        <div className="flex items-center gap-4">
          <div className={`p-2.5 rounded-lg border ${
            isBuy ? 'bg-emerald-100 border-emerald-200 text-emerald-700' : 'bg-amber-100 border-amber-200 text-amber-700'
          }`}>
            {isBuy ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className={`text-[11px] font-semibold uppercase tracking-widest mb-0.5 ${
              isBuy ? 'text-emerald-700' : 'text-amber-700'
            }`}>
              Executing Trade
            </div>
            <div className={`text-sm font-medium flex items-center gap-2 ${
              isBuy ? 'text-emerald-900' : 'text-amber-900'
            }`}>
              <span>{isBuy ? 'BUY' : 'SELL'}</span>
              <span className="font-mono bg-white px-1.5 py-0.5 rounded border border-gray-200">{args.quantity || 1} AAPL</span>
            </div>
          </div>
          <Loader2 className={`w-4 h-4 animate-spin ${isBuy ? 'text-emerald-700' : 'text-amber-700'}`} />
        </div>
      </div>
    )
  }

  if (isTrade && (success || error)) {
    return (
      <div className={`my-4 rounded-xl border px-5 py-4 ${
        error ? 'border-red-300 bg-red-50' : 
        isBuy ? 'border-emerald-300 bg-emerald-50' : 'border-amber-300 bg-amber-50'
      }`}>
        <div className="flex items-center gap-4 mb-4">
          <div className={`p-2.5 rounded-lg border ${
            error ? 'bg-red-100 border-red-200 text-red-700' :
            isBuy ? 'bg-emerald-100 border-emerald-200 text-emerald-700' : 'bg-amber-100 border-amber-200 text-amber-700'
          }`}>
            {error ? <XCircle className="w-5 h-5" /> :
             isBuy ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className={`text-[11px] font-semibold uppercase tracking-widest mb-0.5 ${
              error ? 'text-red-700' : isBuy ? 'text-emerald-700' : 'text-amber-700'
            }`}>
              {error ? 'Execution Failed' : isBuy ? 'Buy Executed' : 'Sell Executed'}
            </div>
            <div className={`text-sm font-medium ${
              error ? 'text-red-800' : 'text-gray-800'
            }`}>
              {error ? (error) : (
                <div className="flex items-center gap-2">
                  <span className="font-mono bg-white px-1.5 py-0.5 rounded border border-gray-200">{result?.quantity} AAPL @ ${result?.price?.toFixed?.(2) || '?'}</span>
                  {result?.pnl !== undefined && (
                    <span className={`font-mono text-xs ${result.pnl >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                      P&L: {result.pnl >= 0 ? '+' : ''}${result.pnl.toFixed(2)}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
          <DollarSign className={`w-5 h-5 ${error ? 'text-red-600' : isBuy ? 'text-emerald-600' : 'text-amber-600'}`} />
        </div>
        
        {result && !error && (result.shares_held !== undefined || result.cash_remaining !== undefined) && (
          <div className={`bg-white rounded-lg p-3 border flex gap-4 ${
            isBuy ? 'border-emerald-200' : 'border-amber-200'
          }`}>
            {result.shares_held !== undefined && (
              <div>
                <div className={`text-[10px] font-medium mb-1 ${isBuy ? 'text-emerald-600' : 'text-amber-600'}`}>Shares Held</div>
                <div className="text-sm font-mono text-gray-700">{result.shares_held}</div>
              </div>
            )}
            {result.cash_remaining !== undefined && (
              <div>
                <div className={`text-[10px] font-medium mb-1 ${isBuy ? 'text-emerald-600' : 'text-amber-600'}`}>Cash</div>
                <div className="text-sm font-mono text-gray-700">${result.cash_remaining.toFixed(2)}</div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // ─── Trigger Alarm (Security) ───
  const isAlarm = name === 'trigger_alarm'

  if (isAlarm && isExecuting) {
    return (
      <div className="my-4 rounded-xl border border-red-300 bg-red-50 px-5 py-4">
        <div className="flex items-center gap-4">
          <div className="p-2.5 rounded-lg bg-red-100 border border-red-200 text-red-700">
            <ShieldAlert className="w-5 h-5 animate-pulse" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-widest text-red-700 mb-0.5">
              Security Alert
            </div>
            <div className="text-sm font-medium text-red-900">
              Triggering alarm — {args.threat_type || 'unknown threat'}...
            </div>
          </div>
          <Loader2 className="w-4 h-4 text-red-700 animate-spin" />
        </div>
      </div>
    )
  }

  if (isAlarm && (success || error)) {
    return (
      <div className={`my-4 rounded-xl border px-5 py-4 ${
        error ? 'border-red-300 bg-red-50' : 'border-red-400 bg-red-50'
      }`}>
        <div className="flex items-center gap-4 mb-4">
          <div className={`p-2.5 rounded-lg border ${
            error ? 'bg-red-100 border-red-200 text-red-700' : 'bg-red-200 border-red-300 text-red-800'
          }`}>
            {error ? <XCircle className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className={`text-[11px] font-semibold uppercase tracking-widest mb-0.5 ${
              error ? 'text-red-700' : 'text-red-800'
            }`}>
              {error ? 'Alarm Failed' : 'ALARM TRIGGERED'}
            </div>
            <div className={`text-sm font-medium ${error ? 'text-red-800' : 'text-red-900'}`}>
              {error || result?.message || 'Security alarm active'}
            </div>
          </div>
          <Siren className="w-5 h-5 text-red-600" />
        </div>
        
        {result && !error && (
          <div className="flex gap-3">
            <div className="flex-1 bg-white rounded-lg p-3 border border-red-200">
              <div className="text-[10px] font-medium text-red-500 mb-1">Threat</div>
              <div className="text-sm text-gray-800 font-medium">{result.threat_type}</div>
            </div>
            <div className="bg-white rounded-lg p-3 border border-red-200 min-w-[100px]">
              <div className="text-[10px] font-medium text-red-500 mb-1">Confidence</div>
              <div className="text-sm font-mono text-gray-800">{(result.confidence * 100).toFixed(0)}%</div>
            </div>
            {result.timestamp && (
              <div className="bg-white rounded-lg p-3 border border-red-200 min-w-[100px]">
                <div className="text-[10px] font-medium text-red-500 mb-1">Time</div>
                <div className="text-sm font-mono text-gray-800">{result.timestamp}</div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // ─── Generic tool call (fallback) ───
  return (
    <div className="my-4 rounded-xl border border-gray-200 bg-gray-50 overflow-hidden">
      <div className="px-4 py-3 flex items-center justify-between border-b border-gray-200 bg-white">
        <div className="flex items-center gap-2.5">
          <Settings2 className="w-4 h-4 text-gray-400" />
          <div className="text-sm font-medium text-gray-700 font-mono flex items-center gap-2">
            {name}
            {isExecuting && <Loader2 className="w-3 h-3 animate-spin text-gray-400" />}
          </div>
        </div>
        <div className="flex items-center">
          {success && <div className="flex items-center gap-1.5 text-emerald-600 text-xs font-medium"><CheckCircle className="w-3.5 h-3.5" /></div>}
          {error && <div className="flex items-center gap-1.5 text-red-600 text-xs font-medium"><XCircle className="w-3.5 h-3.5" /></div>}
        </div>
      </div>
      
      <div className="p-4 space-y-3">
        {Object.keys(args).length > 0 && (
          <div>
            <div className="text-[10px] font-medium text-gray-400 mb-1.5">Parameters</div>
            <pre className="p-2.5 bg-white border border-gray-200 rounded-lg text-xs text-gray-700 font-mono overflow-x-auto">
              {JSON.stringify(args, null, 2)}
            </pre>
          </div>
        )}
        
        {result && (
          <div>
            <div className="text-[10px] font-medium text-gray-400 mb-1.5">Response</div>
            <pre className="p-2.5 bg-white border border-gray-200 rounded-lg text-xs text-gray-600 font-mono overflow-x-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default ToolCallBlock
