import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Live LLM Chat',
  description: 'Real-time LLM with live signal injection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-b from-gray-900 to-black">
        {children}
      </body>
    </html>
  )
}
