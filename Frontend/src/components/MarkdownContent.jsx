import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/prism-light'
import oneDark from 'react-syntax-highlighter/dist/esm/styles/prism/one-dark'

import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript'
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import go from 'react-syntax-highlighter/dist/esm/languages/prism/go'
import java from 'react-syntax-highlighter/dist/esm/languages/prism/java'
import rust from 'react-syntax-highlighter/dist/esm/languages/prism/rust'
import markdown from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'

SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('javascript', javascript)
SyntaxHighlighter.registerLanguage('typescript', typescript)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('go', go)
SyntaxHighlighter.registerLanguage('java', java)
SyntaxHighlighter.registerLanguage('rust', rust)
SyntaxHighlighter.registerLanguage('markdown', markdown)

function CodeBlock({ language, children }) {
  return (
    <div className="my-4 overflow-hidden rounded-xl border border-zinc-700/50">
      {language && (
        <div className="border-b border-zinc-700/50 bg-zinc-800 px-4 py-1.5">
          <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-400">
            {language}
          </span>
        </div>
      )}
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: 0,
          fontSize: '0.8125rem',
          padding: '1rem 1.25rem',
          background: '#1e1e2e',
        }}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    </div>
  )
}

const MarkdownContent = memo(function MarkdownContent({ content }) {
  return (
    <div className="answer-content text-[15px] leading-7 text-stone-800 dark:text-zinc-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1({ children }) {
            return <h1 className="mb-4 mt-6 text-xl font-semibold text-stone-900 dark:text-zinc-50">{children}</h1>
          },
          h2({ children }) {
            return <h2 className="mb-3 mt-6 text-lg font-semibold text-stone-900 dark:text-zinc-50">{children}</h2>
          },
          h3({ children }) {
            return <h3 className="mb-2 mt-5 text-base font-semibold text-stone-900 dark:text-zinc-50">{children}</h3>
          },
          p({ children }) {
            return <p className="mb-4 last:mb-0">{children}</p>
          },
          ul({ children }) {
            return <ul className="mb-4 list-disc space-y-1.5 pl-5">{children}</ul>
          },
          ol({ children }) {
            return <ol className="mb-4 list-decimal space-y-1.5 pl-5">{children}</ol>
          },
          li({ children }) {
            return <li className="pl-0.5">{children}</li>
          },
          strong({ children }) {
            return <strong className="font-semibold text-stone-900 dark:text-zinc-50">{children}</strong>
          },
          blockquote({ children }) {
            return (
              <blockquote className="my-4 border-l-4 border-brand-400 pl-4 italic text-stone-600 dark:text-zinc-400">
                {children}
              </blockquote>
            )
          },
          hr() {
            return <hr className="my-6 border-stone-200 dark:border-zinc-700" />
          },
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '')
            if (!inline && match) {
              return <CodeBlock language={match[1]}>{children}</CodeBlock>
            }
            return (
              <code
                className="rounded-md bg-stone-200/70 px-1.5 py-0.5 font-mono text-[0.875em] text-brand-700 dark:bg-zinc-800 dark:text-brand-300"
                {...props}
              >
                {children}
              </code>
            )
          },
          pre({ children }) {
            return <>{children}</>
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="font-medium text-brand-600 underline decoration-brand-300 underline-offset-2 hover:text-brand-700 dark:text-brand-400 dark:decoration-brand-700"
              >
                {children}
              </a>
            )
          },
          table({ children }) {
            return (
              <div className="my-4 overflow-x-auto rounded-xl border border-stone-200 dark:border-zinc-700">
                <table className="min-w-full text-sm">{children}</table>
              </div>
            )
          },
          th({ children }) {
            return (
              <th className="border-b border-stone-200 bg-stone-100 px-4 py-2 text-left font-semibold dark:border-zinc-700 dark:bg-zinc-800">
                {children}
              </th>
            )
          },
          td({ children }) {
            return (
              <td className="border-b border-stone-100 px-4 py-2 dark:border-zinc-800">{children}</td>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})

export default MarkdownContent
