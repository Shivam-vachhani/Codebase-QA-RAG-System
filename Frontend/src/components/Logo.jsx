export default function Logo({ size = 32, animated = false, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={[animated ? 'logo-animated' : '', className].filter(Boolean).join(' ')}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="logoGrad" x1="8" y1="8" x2="40" y2="40" gradientUnits="userSpaceOnUse">
          <stop stopColor="#F4A261" />
          <stop offset="1" stopColor="#E76F3C" />
        </linearGradient>
        <linearGradient id="logoGradInner" x1="18" y1="18" x2="30" y2="30" gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFD6A5" />
          <stop offset="1" stopColor="#E07A4A" />
        </linearGradient>
      </defs>

      {/* Outer ring — codebase boundary */}
      <circle
        cx="24"
        cy="24"
        r="20"
        stroke="url(#logoGrad)"
        strokeWidth="2"
        fill="none"
        className={animated ? 'logo-ring' : ''}
        opacity="0.85"
      />

      {/* Left brace */}
      <path
        d="M17 14 C13 14 11 18 11 24 C11 30 13 34 17 34"
        stroke="url(#logoGrad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
        className={animated ? 'logo-brace-left' : ''}
      />

      {/* Right brace */}
      <path
        d="M31 14 C35 14 37 18 37 24 C37 30 35 34 31 34"
        stroke="url(#logoGrad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
        className={animated ? 'logo-brace-right' : ''}
      />

      {/* Center query node */}
      <circle
        cx="24"
        cy="24"
        r="4"
        fill="url(#logoGradInner)"
        className={animated ? 'logo-core' : ''}
      />

      {/* Search rays — RAG retrieval hint */}
      <path
        d="M24 8 L24 11 M24 37 L24 40 M8 24 L11 24 M37 24 L40 24"
        stroke="url(#logoGrad)"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.5"
        className={animated ? 'logo-rays' : ''}
      />
    </svg>
  )
}
