import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

interface MarkdownRendererProps {
  content: string
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <div className="prose prose-gray prose-p:text-gray-700 prose-headings:text-gray-900 prose-strong:text-gray-900 prose-li:text-gray-700 max-w-none text-sm sm:text-base break-words text-gray-700">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeKatex]}
      components={{
        // Customize components if needed
        a: ({ node, ...props }) => (
          <a {...props} className="text-blue-700 hover:text-blue-900 underline" target="_blank" rel="noopener noreferrer" />
        ),
        code: ({ node, inline, className, children, ...props }: any) => {
          const match = /language-(\w+)/.exec(className || '')
          return !inline && match ? (
            <div className="relative group">
              <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
                {/* Could add copy button here */}
              </div>
              <pre className="bg-gray-50 rounded-lg p-4 overflow-x-auto my-4 border border-gray-200">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            </div>
          ) : (
            <code className="bg-gray-100 rounded px-1.5 py-0.5 text-blue-800 font-mono text-sm" {...props}>
              {children}
            </code>
          )
        },
        // Table styling
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full divide-y divide-gray-200 border border-gray-200 rounded-lg">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-600 uppercase tracking-wider border-b border-gray-200">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700 border-b border-gray-100">
            {children}
          </td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
