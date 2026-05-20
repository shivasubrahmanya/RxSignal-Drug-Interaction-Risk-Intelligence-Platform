/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'faers-bg': '#0f172a',
        'faers-card': 'rgba(30, 41, 59, 0.7)',
        'faers-accent': '#8b5cf6',
      },
    },
  },
  plugins: [],
}
