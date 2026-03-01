import React, { useMemo } from 'react'
import { Zap, Sparkles } from 'lucide-react'
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
  | { type: 'inject' }

const ChatMessage: React.FC<ChatMessageProps> = ({ role, content, isGenerating, toolCalls }) => {
  
  // Parser for thinking blocks and text
  const blocks = useMemo(() => {
    if (role !== 'assistant') return []

    const parsedBlocks: ContentBlock[] = []
    // Strip tool execution results and live update markers from display
    let remaining = content
      .replace(/\[TOOL EXECUTION RESULT:[^\]]*\]/g, '')
      .replace(/\[LIVE UPDATE:[^\]]*\]/g, '')
      .replace(/\[LIVE CAMERA UPDATE:[^\]]*\]/g, '')
      .replace(/\[LIVE SIGNAL\][^\n]*/g, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
    
    // Handle signal injection markers first
    const parts = remaining.split('{{SIGNAL_INJECT}}')
    if (parts.length > 1) {
      parts.forEach((part, idx) => {
        if (part.trim()) {
          parsedBlocks.push({ type: 'text', content: part.trim() })
        }
        if (idx < parts.length - 1) {
          parsedBlocks.push({ type: 'inject' })
        }
      })
      return parsedBlocks
    }
    
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
        // Complete thinking block - skip empty ones (injected noop blocks)
        const thinkContent = remaining.substring(thinkStart + 7, endTagIndex)
        if (thinkContent.trim()) {
          parsedBlocks.push({ type: 'think', content: thinkContent, isComplete: true })
        }
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
        <div className="max-w-[80%] rounded-2xl px-5 py-3.5 bg-blue-900 text-white shadow-sm">
          <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{content}</p>
        </div>
      </div>
    )
  }

  if (role === 'signal') {
    return (
      <div className="flex justify-center mb-6">
        <div className="max-w-[85%] rounded-xl px-4 py-2 bg-gray-50 border border-gray-200 text-gray-600 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="p-1 rounded bg-gray-100 border border-gray-200">
              <Zap className="w-3.5 h-3.5 text-gray-400" />
            </div>
            <p className="whitespace-pre-wrap leading-relaxed text-[13px] font-mono text-gray-500">{content}</p>
          </div>
        </div>
      </div>
    )
  }

  // Assistant Message
  return (
    <div className="flex justify-start mb-8 w-full group">
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
          
          if (block.type === 'inject') {
            return (
              <div key={i} className="flex items-center justify-center gap-3 my-4 select-none">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-blue-300 to-transparent max-w-24" />
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-50 border border-blue-200">
                  <Sparkles className="w-3 h-3 text-blue-500" />
                  <span className="text-[10px] font-medium text-blue-600 tracking-wide">signal injected</span>
                </div>
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-blue-300 to-transparent max-w-24" />
              </div>
            )
          }
          
          // Check if this text block comes after a thinking block
          const isAfterThinking = i > 0 && blocks[i - 1].type === 'think'
          
          return (
            <div key={i} className="mb-4 last:mb-0">
               {/* Label if this is the response after thinking */}
               {isAfterThinking && (
                 <div className="flex items-center gap-2 mb-2 ml-1">
                   <div className="h-px flex-1 bg-gray-200 max-w-16" />
                   <span className="text-[10px] font-medium uppercase tracking-widest text-gray-400">Response</span>
                 </div>
               )}
               <div className={`relative rounded-2xl px-6 py-5 overflow-hidden ${
                 isAfterThinking 
                   ? 'bg-gray-50 border border-gray-200' 
                   : 'bg-transparent'
               }`}>
                 <div className="relative text-gray-800 leading-relaxed text-[15px] prose prose-gray prose-p:leading-relaxed prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:rounded-xl max-w-none">
                   <MarkdownRenderer content={block.content} />
                 </div>
                 {/* Cursor if this is the last block and generating */}
                 {isGenerating && i === blocks.length - 1 && (
                   <div className="mt-2">
                     <span className="inline-block w-2 h-4 bg-blue-900 animate-pulse align-middle rounded-sm" />
                   </div>
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
        
        {/* Show raw content if blocks are empty but content exists (fallback) */}
        {blocks.length === 0 && content.replace(/<think>[\s\S]*?<\/think>/g, '').trim() && (
          <div className="relative rounded-2xl overflow-hidden bg-transparent">
            <div className="relative text-gray-800 leading-relaxed text-[15px] prose prose-gray prose-p:leading-relaxed prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:rounded-xl max-w-none">
              <MarkdownRenderer content={content.replace(/<think>[\s\S]*?<\/think>/g, '').trim()} />
            </div>
            {isGenerating && (
              <div className="mt-2">
                <span className="inline-block w-2 h-4 bg-blue-900 animate-pulse align-middle rounded-sm" />
              </div>
            )}
          </div>
        )}
        
        {/* Cursor if blocks are empty and no content (just starting) */}
        {isGenerating && blocks.length === 0 && !content.replace(/<think>[\s\S]*?<\/think>/g, '').trim() && !toolCalls?.length && (
           <div className="relative rounded-2xl bg-transparent py-2">
              <span className="inline-block w-2 h-4 bg-blue-900 animate-pulse align-middle rounded-sm" />
           </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
