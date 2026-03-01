import React, { useState } from 'react'
import { ChevronDown, ChevronRight, BrainCircuit } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import MarkdownRenderer from './MarkdownRenderer'

interface ThinkingBlockProps {
  content: string
  isComplete: boolean
}

const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ content, isComplete }) => {
  const [isExpanded, setIsExpanded] = useState(!isComplete)

  if (!content) return null

  return (
    <div className="my-2 border border-gray-200 rounded-lg bg-gray-50 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-500 hover:bg-gray-100 transition-colors"
      >
        {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <BrainCircuit className="w-3 h-3 text-blue-800" />
        <span className="font-medium">
          {isComplete ? 'Reasoning Process' : 'Thinking...'}
        </span>
        <div className="ml-auto text-[10px] opacity-50">
          {content.length} chars
        </div>
      </button>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="px-4 py-3 border-t border-gray-200 bg-white">
              <div className="prose prose-gray prose-xs max-w-none text-gray-500 font-mono text-xs leading-relaxed">
                <MarkdownRenderer content={content} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ThinkingBlock
