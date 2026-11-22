import type { Metadata, Viewport } from 'next'
import { Inter, JetBrains_Mono, Instrument_Serif } from 'next/font/google'
import './globals.css'

const sans = Inter({ subsets: ['latin'], variable: '--font-sans' })
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' })
const serif = Instrument_Serif({ subsets: ['latin'], weight: '400', style: ['normal', 'italic'], variable: '--font-serif' })

export const metadata: Metadata = {
  title: 'DeepS&P · LSTM forecasting on 90 years of S&P 500 data',
  description:
    'A three-layer PyTorch LSTM trained on S&P 500 closes from 1930 to 2020, with point-in-time backtests, a Monte Carlo path simulator and market analytics, all in the browser.',
}

export const viewport: Viewport = { themeColor: '#F6E4D6', width: 'device-width', initialScale: 1 }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  )
}
