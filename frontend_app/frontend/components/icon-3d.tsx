export function Icon3D({ type, size = 24 }: { type: "dashboard" | "extract" | "reports" | "settings" | "search" | "help" | "notification" | "ai"; size?: number }) {
  const gradientId = `grad-${type}`;

  switch (type) {
    case "dashboard":
      // 3D Bar Chart (Isometric bars)
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id={`${gradientId}-1`} x1="0" y1="0" x2="0" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" />
              <stop offset="100%" stopColor="#2563eb" />
            </linearGradient>
            <linearGradient id={`${gradientId}-2`} x1="0" y1="0" x2="0" y2="100%">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
            <linearGradient id={`${gradientId}-3`} x1="0" y1="0" x2="0" y2="100%">
              <stop offset="0%" stopColor="#f472b6" />
              <stop offset="100%" stopColor="#db2777" />
            </linearGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="2" dy="5" stdDeviation="3" floodColor="#000" floodOpacity="0.3"/>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-shadow)`}>
            {/* Base grid */}
            <path d="M 10 90 L 90 90 L 70 70 L 30 70 Z" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.2)" />
            {/* Bar 1 */}
            <rect x="20" y="50" width="15" height="40" fill={`url(#${gradientId}-1)`} rx="2" />
            <path d="M 20 50 L 25 45 L 40 45 L 35 50 Z" fill="#93c5fd" />
            <path d="M 35 50 L 40 45 L 40 85 L 35 90 Z" fill="#1e40af" />
            {/* Bar 2 */}
            <rect x="45" y="30" width="15" height="60" fill={`url(#${gradientId}-2)`} rx="2" />
            <path d="M 45 30 L 50 25 L 65 25 L 60 30 Z" fill="#c4b5fd" />
            <path d="M 60 30 L 65 25 L 65 85 L 60 90 Z" fill="#5b21b6" />
            {/* Bar 3 */}
            <rect x="70" y="60" width="15" height="30" fill={`url(#${gradientId}-3)`} rx="2" />
            <path d="M 70 60 L 75 55 L 90 55 L 85 60 Z" fill="#fbcfe8" />
            <path d="M 85 60 L 90 55 L 90 85 L 85 90 Z" fill="#9d174d" />
          </g>
        </svg>
      );
    case "extract":
      // 3D Scanner / Document Box
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id={`${gradientId}-doc`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" />
              <stop offset="100%" stopColor="#cbd5e1" />
            </linearGradient>
            <linearGradient id={`${gradientId}-scan`} x1="0" y1="0" x2="0" y2="100%">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="0" dy="8" stdDeviation="4" floodColor="#000" floodOpacity="0.4"/>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-shadow)`}>
            {/* Document Base */}
            <path d="M 25 15 L 75 15 L 85 25 L 85 85 L 15 85 L 15 25 Z" fill={`url(#${gradientId}-doc)`} />
            <path d="M 75 15 L 75 25 L 85 25 Z" fill="#94a3b8" />
            {/* Lines */}
            <rect x="25" y="40" width="50" height="4" fill="#94a3b8" rx="2" />
            <rect x="25" y="55" width="40" height="4" fill="#94a3b8" rx="2" />
            <rect x="25" y="70" width="30" height="4" fill="#94a3b8" rx="2" />
            {/* Scanner beam */}
            <path d="M 10 50 L 90 50 L 80 65 L 20 65 Z" fill={`url(#${gradientId}-scan)`} opacity="0.6" />
            <rect x="10" y="48" width="80" height="4" fill="#10b981" rx="2" />
          </g>
        </svg>
      );
    case "settings":
      // 3D Gear (Isometric)
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id={`${gradientId}-gear`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#94a3b8" />
              <stop offset="100%" stopColor="#475569" />
            </linearGradient>
            <linearGradient id={`${gradientId}-gear-dark`} x1="0" y1="0" x2="0" y2="100%">
              <stop offset="0%" stopColor="#334155" />
              <stop offset="100%" stopColor="#1e293b" />
            </linearGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="3" dy="6" stdDeviation="4" floodColor="#000" floodOpacity="0.4"/>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-shadow)`}>
            {/* Outer Gear */}
            <path d="M 50 15 A 35 35 0 1 0 50 85 A 35 35 0 1 0 50 15 Z" fill={`url(#${gradientId}-gear)`} stroke="#cbd5e1" strokeWidth="4" />
            {/* Teeth - Simulated with a dashed thick stroke */}
            <circle cx="50" cy="50" r="38" fill="none" stroke={`url(#${gradientId}-gear-dark)`} strokeWidth="12" strokeDasharray="15 10" />
            {/* Inner hole */}
            <circle cx="50" cy="50" r="15" fill="#0f172a" stroke="#cbd5e1" strokeWidth="4" />
          </g>
        </svg>
      );
    case "reports":
      // 3D Pie Chart
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id={`${gradientId}-pie1`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="100%" stopColor="#dc2626" />
            </linearGradient>
            <linearGradient id={`${gradientId}-pie2`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#fbbf24" />
              <stop offset="100%" stopColor="#d97706" />
            </linearGradient>
            <linearGradient id={`${gradientId}-pie3`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" />
              <stop offset="100%" stopColor="#2563eb" />
            </linearGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="2" dy="8" stdDeviation="4" floodColor="#000" floodOpacity="0.3"/>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-shadow)`}>
            {/* Base shadow cylinder */}
            <ellipse cx="50" cy="65" rx="40" ry="20" fill="#1e3a8a" />
            {/* Pie segments extruded */}
            <path d="M 50 50 L 90 50 A 40 20 0 0 1 10 50 L 50 50 Z" fill={`url(#${gradientId}-pie3)`} />
            <path d="M 50 50 L 10 50 A 40 20 0 0 1 50 30 L 50 50 Z" fill={`url(#${gradientId}-pie2)`} />
            {/* Pulled out segment */}
            <path d="M 55 45 L 55 25 A 40 20 0 0 1 95 45 L 55 45 Z" fill={`url(#${gradientId}-pie1)`} transform="translate(5, -5)" />
            <path d="M 55 45 L 95 45 L 95 65 L 55 65 Z" fill="#991b1b" transform="translate(5, -5)" />
            <path d="M 55 45 L 55 65 L 55 25 Z" fill="#b91c1c" transform="translate(5, -5)" />
          </g>
        </svg>
      );
    case "search":
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id={`${gradientId}-glass`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="rgba(255,255,255,0.8)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0.1)" />
            </linearGradient>
            <linearGradient id={`${gradientId}-handle`} x1="0" y1="0" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#94a3b8" />
              <stop offset="100%" stopColor="#334155" />
            </linearGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="4" dy="4" stdDeviation="3" floodColor="#000" floodOpacity="0.4"/>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-shadow)`}>
            <path d="M 60 60 L 85 85" stroke={`url(#${gradientId}-handle)`} strokeWidth="14" strokeLinecap="round" />
            <circle cx="45" cy="45" r="25" fill={`url(#${gradientId}-glass)`} stroke="#cbd5e1" strokeWidth="6" />
            <path d="M 35 35 Q 45 25 55 35" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" fill="none" opacity="0.8" />
          </g>
        </svg>
      );
    case "ai":
      // 3D Brain/Core
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <radialGradient id={`${gradientId}-core`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#4c1d95" />
            </radialGradient>
            <filter id={`${gradientId}-glow`}>
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <g filter={`url(#${gradientId}-glow)`}>
            <polygon points="50,15 80,35 80,70 50,90 20,70 20,35" fill={`url(#${gradientId}-core)`} stroke="#ddd6fe" strokeWidth="2" />
            <polygon points="50,15 80,35 50,55 20,35" fill="rgba(255,255,255,0.2)" />
            <polygon points="50,55 80,35 80,70 50,90" fill="rgba(0,0,0,0.3)" />
            <circle cx="50" cy="55" r="8" fill="#fff" filter="blur(2px)" />
          </g>
        </svg>
      );
    default:
      // Generic 3D sphere
      return (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
          <defs>
            <radialGradient id={`${gradientId}-sphere`} cx="30%" cy="30%" r="70%">
              <stop offset="0%" stopColor="#ffffff" />
              <stop offset="100%" stopColor="#3b82f6" />
            </radialGradient>
            <filter id={`${gradientId}-shadow`}>
              <feDropShadow dx="0" dy="10" stdDeviation="5" floodColor="#000" floodOpacity="0.4"/>
            </filter>
          </defs>
          <circle cx="50" cy="50" r="35" fill={`url(#${gradientId}-sphere)`} filter={`url(#${gradientId}-shadow)`} />
        </svg>
      );
  }
}
