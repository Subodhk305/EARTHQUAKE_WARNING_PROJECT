/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        void: '#0B0F19',
        panel: '#0E1424',
        border: '#1A2540',
        neon: '#00D4FF',
        pulse: '#7B2FFF',
        danger: '#FF3B5C',
        warning: '#FFB020',
        safe: '#00E57A',
      },
      fontFamily: {
        mono: ['"Geist Mono"', '"JetBrains Mono"', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif'],
      },
      boxShadow: {
        neon: '0 0 20px rgba(0,212,255,0.3)',
        danger: '0 0 20px rgba(255,59,92,0.4)',
        pulse: '0 0 20px rgba(123,47,255,0.35)',
      },
    },
  },
  plugins: [],
}
