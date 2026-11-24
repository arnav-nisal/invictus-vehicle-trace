/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#F7F7F5',
        surface: '#FFFFFF',
        stone: {
          hairline: '#E6E6E2',
        },
        well: {
          DEFAULT: '#F1F1EE',
          hover: '#EAEAE6',
        },
        ink: {
          DEFAULT: '#111111',
          hover: '#262626',
          muted: '#6B6B66',
        },
        emerald: {
          deep: '#047857',
          bright: '#059669',
          tint: '#F0FDF4',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      letterSpacing: {
        tighter: '-0.02em',
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
      },
      boxShadow: {
        subtle: '0 1px 2px rgba(0, 0, 0, 0.03)',
      },
      transitionTimingFunction: {
        precision: 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [],
}
