/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      letterSpacing: {
        display: '-0.03em',
        snugger: '-0.02em',
      },
      colors: {
        alabaster: {
          50: '#FFFFFF',
          100: '#FBFBFC',
          200: '#F4F5F7',
          300: '#EAEBEF',
          400: '#DDE0E6',
        },
        obsidian: {
          DEFAULT: '#0B0F17',
          deep: '#06080D',
          surface: '#111622',
          card: '#151C2C',
          border: 'rgba(255, 255, 255, 0.08)',
          glow: 'rgba(255, 255, 255, 0.12)',
        },
        studio: {
          bg: '#F4F5F7',
          card: '#FFFFFF',
          border: 'rgba(15, 23, 42, 0.06)',
          borderSubtle: 'rgba(15, 23, 42, 0.04)',
          text: '#0F172A',
          muted: '#64748B',
        },
      },
      boxShadow: {
        'bento': '0 20px 40px -15px rgba(15, 23, 42, 0.05), 0 0 1px 1px rgba(15, 23, 42, 0.04)',
        'bento-hover': '0 25px 50px -12px rgba(15, 23, 42, 0.08), 0 0 1px 1px rgba(15, 23, 42, 0.06)',
        'obsidian': '0 25px 60px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        'hsrp': '0 20px 40px -10px rgba(0, 0, 0, 0.4), 0 1px 3px rgba(0, 0, 0, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.8), inset 0 -1px 2px rgba(0, 0, 0, 0.25)',
        'emboss': 'inset 0 1px 1px rgba(255, 255, 255, 0.9), 0 2px 4px rgba(0, 0, 0, 0.15)',
        'pill': '0 4px 12px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.06)',
      },
      borderRadius: {
        '3xl': '1.5rem', // 24px
        '4xl': '2rem',   // 32px
      },
    },
  },
  plugins: [],
}
