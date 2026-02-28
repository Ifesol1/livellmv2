import React, { useMemo } from 'react'
import { Zap } from 'lucide-react'
import MarkdownRenderer from './MarkdownRenderer'
import ThinkingBlock from './ThinkingBlock'
import ToolCallBlock from './ToolCallBlock'

interface ToolCall {
  name: string
  arguments: Record<string, any>
  result?: any
  success?: boolean
  error?: string
  isExecuting?: boolean
}

interface ChatMessageProps {
  role: 'user' | 'assistant' | 'signal'
  content: string
  isGenerating?: boolean
  toolCalls?: ToolCall[]
}

type ContentBlock = 
  | { type: 'text'; content: string }
  | { type: 'think'; content: string; isComplete: boolean }

const ChatMessage: React.FC<ChatMessageProps> = ({ role, content, isGenerating, toolCalls }) => {
  
  // Parser for thinking blocks and text
  const blocks = useMemo(() => {
    if (role !== 'assistant') return []

    const parsedBlocks: ContentBlock[] = []
    let remaining = content
    
    while (remaining.length > 0) {
      const thinkStart = remaining.indexOf('<think>')
      
      // If no think block, rest is text
      if (thinkStart === -1) {
        if (remaining.trim()) {
          parsedBlocks.push({ type: 'text', content: remaining })
        }
        break
      }
      
      // Push text before thinking block
      if (thinkStart > 0) {
        const textBefore = remaining.substring(0, thinkStart)
        if (textBefore.trim()) {
          parsedBlocks.push({ type: 'text', content: textBefore })
        }
      }
      
      // Process thinking block
      const endTag = '</think>'
      const endTagIndex = remaining.indexOf(endTag, thinkStart)
      
      if (endTagIndex !== -1) {
        // Complete thinking block
        const thinkContent = remaining.substring(thinkStart + 7, endTagIndex)
        parsedBlocks.push({ type: 'think', content: thinkContent, isComplete: true })
        remaining = remaining.substring(endTagIndex + endTag.length)
      } else {
        // Incomplete (streaming) thinking block
        const thinkContent = remaining.substring(thinkStart + 7)
        parsedBlocks.push({ type: 'think', content: thinkContent, isComplete: false })
        remaining = ''
      }
    }
    
    return parsedBlocks
  }, [content, role])

  if (role === 'user') {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[85%] rounded-2xl px-5 py-4 bg-primary-600 text-white shadow-lg">
          <p className="whitespace-pre-wrap leading-relaxed">{content}</p>
        </div>
      </div>
    )
  }

  if (role === 'signal') {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[85%] rounded-2xl px-5 py-4 bg-yellow-600 text-white shadow-lg">
          <div className="flex items-center gap-2 text-yellow-200 text-xs mb-1">
            <Zap className="w-3 h-3" />
            <span className="font-medium uppercase tracking-wide">Follow-up</span>
          </div>
          <p className="whitespace-pre-wrap leading-relaxed">{content}</p>
        </div>
      </div>
    )
  }

  // Assistant Message
  return (
    <div className="flex justify-start mb-6 w-full group">
      <div className="w-full max-w-4xl pr-4">
        {blocks.map((block, i) => {
          if (block.type === 'think') {
            return (
              <ThinkingBlock 
                key={i} 
                content={block.content} 
                isComplete={block.isComplete} 
              />
            )
          }
          
          // Check if this text block comes after a thinking block
          const isAfterThinking = i > 0 && blocks[i - 1].type === 'think'
          
          return (
            <div key={i} className="mb-3 last:mb-0">
               {/* Label if this is the response after thinking */}
               {isAfterThinking && (
                 <div className="flex items-center gap-2 mb-2 ml-1">
                   <div className="h-px flex-1 bg-gradient-to-r from-primary-500/50 to-transparent max-w-16" />
                   <span className="text-[10px] font-semibold uppercase tracking-widest text-primary-400/70">Response</span>
                 </div>
               )}
               <div className={`rounded-2xl px-6 py-5 shadow-sm ${
                 isAfterThinking 
                   ? 'bg-gray-800/60 border-2 border-gray-700/70' 
                   : 'bg-gray-800/40 border border-gray-700/50'
               }`}>
                 <div className="text-gray-200 leading-relaxed">
                   <MarkdownRenderer content={block.content} />
                 </div>
                 {/* Cursor if this is the last block and generating */}
                 {isGenerating && i === blocks.length - 1 && (
                   <span className="inline-block w-1.5 h-4 bg-primary-400 ml-1 animate-pulse align-middle" />
                 )}
               </div>
            </div>
          )
        })}
        
        {/* Tool Calls */}
        {toolCalls && toolCalls.length > 0 && (
          <div className="mt-3">
            {toolCalls.map((tc, i) => (
              <ToolCallBlock
                key={i}
                name={tc.name}
                arguments={tc.arguments}
                result={tc.result}
                success={tc.success}
                error={tc.error}
                isExecuting={tc.isExecuting}
              />
            ))}
          </div>
        )}
        
        {/* Cursor if blocks are empty (just starting) */}
        {isGenerating && blocks.length === 0 && !toolCalls?.length && (
           <div className="bg-gray-800/40 border border-gray-700/50 rounded-2xl px-6 py-5 shadow-sm">
              <span className="inline-block w-1.5 h-4 bg-primary-400 animate-pulse align-middle" />
           </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
