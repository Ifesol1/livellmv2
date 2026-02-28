import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Phone, Siren, ShieldAlert, CheckCircle, XCircle, Loader2 } from 'lucide-react'

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
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="my-4 relative overflow-hidden rounded-2xl border-2 border-amber-500/60"
      >
        {/* Animated background pulse */}
        <div className="absolute inset-0 bg-gradient-to-r from-amber-950/80 via-amber-900/60 to-amber-950/80 animate-pulse" />
        
        <div className="relative px-6 py-5 flex items-center gap-5">
          {/* Ringing phone icon */}
          <div className="relative">
            <motion.div
              animate={{ rotate: [0, -15, 15, -15, 15, 0] }}
              transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 0.5 }}
              className="p-3.5 rounded-full bg-amber-500/30 border border-amber-500/40"
            >
              <Phone className="w-7 h-7 text-amber-300" />
            </motion.div>
            {/* Ring waves */}
            <motion.div
              animate={{ scale: [1, 2.5], opacity: [0.4, 0] }}
              transition={{ duration: 1.2, repeat: Infinity }}
              className="absolute inset-0 rounded-full border-2 border-amber-400/50"
            />
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="text-lg font-bold text-amber-200 tracking-wide">
              DIALING 911
            </div>
            <div className="text-sm text-amber-400/80 font-medium">
              Connecting to emergency services{'.'.repeat((ringCount % 3) + 1)}
            </div>
          </div>
          
          <Loader2 className="w-5 h-5 text-amber-400 animate-spin shrink-0" />
        </div>
      </motion.div>
    )
  }

  // ─── 911 Call: Dispatched State ───
  if (is911Call && success) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        className="my-4 relative overflow-hidden rounded-2xl border-2 border-red-500/70"
      >
        {/* Gradient BG */}
        <div className="absolute inset-0 bg-gradient-to-br from-red-950 via-red-900/80 to-red-950" />
        
        {/* Subtle scanning line */}
        <motion.div
          animate={{ top: ['0%', '100%'] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-red-400/40 to-transparent"
          style={{ position: 'absolute' }}
        />
        
        <div className="relative px-6 py-5">
          {/* Top row */}
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-red-500/25 border border-red-500/30">
              <ShieldAlert className="w-7 h-7 text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-lg font-bold text-red-200 tracking-wide">
                911 DISPATCHED
              </div>
              <div className="text-xs text-red-400/70 font-medium uppercase tracking-widest">
                Emergency services en route
              </div>
            </div>
            <motion.div
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <Siren className="w-6 h-6 text-red-400" />
            </motion.div>
          </div>
          
          {/* Info card */}
          <div className="flex items-stretch gap-3">
            <div className="flex-1 bg-red-950/60 rounded-xl p-3.5 border border-red-800/40">
              <div className="text-[10px] font-bold uppercase tracking-widest text-red-500/70 mb-1.5">Status</div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-red-400 shrink-0" />
                <span className="text-sm text-red-200 font-medium">
                  {result?.message || 'Emergency services dispatched'}
                </span>
              </div>
            </div>
            {result?.timestamp && (
              <div className="bg-red-950/60 rounded-xl p-3.5 border border-red-800/40 flex flex-col justify-center">
                <div className="text-[10px] font-bold uppercase tracking-widest text-red-500/70 mb-1">Time</div>
                <div className="text-sm font-mono text-red-300 font-bold">{result.timestamp}</div>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    )
  }

  // ─── 911 Call: Error State ───
  if (is911Call && error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="my-4 rounded-2xl border-2 border-red-700/50 bg-red-950/50 px-6 py-4"
      >
        <div className="flex items-center gap-3">
          <XCircle className="w-6 h-6 text-red-500 shrink-0" />
          <div>
            <div className="text-sm font-bold text-red-300">911 Call Failed</div>
            <div className="text-xs text-red-400/70 mt-0.5">{error}</div>
          </div>
        </div>
      </motion.div>
    )
  }

  // ─── Generic tool call (fallback) ───
  return (
    <div className="my-3 rounded-xl border border-gray-700/50 bg-gray-800/30 px-4 py-3">
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <Loader2 className={`w-4 h-4 ${isExecuting ? 'animate-spin' : ''}`} />
        <span className="font-medium">{name}</span>
        {success && <CheckCircle className="w-3.5 h-3.5 text-green-500" />}
        {error && <XCircle className="w-3.5 h-3.5 text-red-500" />}
      </div>
    </div>
  )
}

export default ToolCallBlock
