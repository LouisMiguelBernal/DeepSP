'use client'

import { useEffect, useMemo, useState } from 'react'
import Chart, { Legend, type Line } from './Chart'
import {
  build, downsample, sessionOnOrBefore, mean, std, argExtreme,
  fmtUsd, fmtPct, fmtDate, type Payload, type Series,
} from '@/lib/series'
import { simulate, bands, terminal, quantile, type Sim } from '@/lib/mc'

const C = {
  accent: '#8E1B3B', // claret
  accent2: '#0E6E78', // teal
  warn: '#9A6205',
  pos: '#1F7A4D',
  neg: '#B4361F',
  faint: '#A39385',
}

const TABS = ['Prediction', 'Monte Carlo', 'Analytics', 'Model'] as const
type Tab = (typeof TABS)[number]

const usd0 = (v: number) => (v >= 1000 ? '$' + (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 'k' : '$' + v.toFixed(v < 10 ? 1 : 0))

export default function Dashboard() {
  const [s, setS] = useState<Series | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('Prediction')

  useEffect(() => {
    fetch('/data/spx.json')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<Payload> })
      .then(p => setS(build(p)))
      .catch(e => setErr(String(e)))
  }, [])

  useEffect(() => {
    const sync = () => {
      const h = window.location.hash.slice(1).replace('-', ' ')
      const t = TABS.find(x => x.toLowerCase() === h)
      if (t) setTab(t)
    }
    sync()
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  }, [])

  const select = (t: Tab) => {
    setTab(t)
    history.replaceState(null, '', '#' + t.toLowerCase().replace(' ', '-'))
  }

  return (
    <>
      <header className="hero">
        <div className="wrap">
          <div className="mast">
            <img src="/brand/deepsp.svg" alt="DeepS&amp;P" />
            <span className="mast-d">Est. 1928 · S&amp;P 500 · Neural forecasting</span>
          </div>
          <div className="eyebrow"><span className="dot" /> The forecasting desk</div>
          <h1>Deep<em>S&amp;P</em></h1>
          <p className="lede">
            A three-layer LSTM trained on ninety years of S&amp;P 500 closes, paired with a Monte Carlo engine
            for forward path simulation. Everything runs in your browser.
          </p>
          <div className="chips">
            <span className="chip accent">LSTM · 3 × 256</span>
            <span className="chip">{s ? (s.t.length - s.from).toLocaleString() : '23,000+'} sessions</span>
            <span className="chip">1930 – 2020</span>
            <span className="chip">{s ? s.meta.parameters.toLocaleString() : '1.3M'} parameters</span>
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Sections">
        <div className="wrap tabs-in" role="tablist">
          {TABS.map(t => (
            <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => select(t)}>
              {t}
            </button>
          ))}
        </div>
      </nav>

      <main className="wrap main">
        {err && <div className="empty"><b>Could not load the market data.</b><p>{err}</p></div>}
        {!s && !err && <Skeleton />}
        {s && tab === 'Prediction' && <Prediction s={s} />}
        {s && tab === 'Monte Carlo' && <MonteCarlo s={s} />}
        {s && tab === 'Analytics' && <Analytics s={s} />}
        {s && tab === 'Model' && <Model s={s} />}
      </main>

      <footer className="foot wrap">
        <span><b>DeepS&amp;P</b> · LSTM + Monte Carlo on 90 years of S&amp;P 500 data</span>
        <span>Research and education only. Nothing here is investment advice.</span>
      </footer>
    </>
  )
}

function Skeleton() {
  return (
    <div className="stack">
      <div className="stats">{Array.from({ length: 4 }, (_, i) => <div key={i} className="stat sk" />)}</div>
      <div className="card sk" style={{ height: 420 }} />
    </div>
  )
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="sec">
      <div className="sec-h">
        <h2>{title}</h2>
        {sub && <p>{sub}</p>}
      </div>
      {children}
    </section>
  )
}

type StatT = { label: string; value: string; delta?: string; tone?: 'accent' | 'pos' | 'neg' | 'warn' }
function Stats({ items }: { items: StatT[] }) {
  return (
    <div className="stats">
      {items.map(i => (
        <div key={i.label} className={'stat ' + (i.tone ?? '')}>
          <div className="stat-l">{i.label}</div>
          <div className="stat-v">{i.value}</div>
          {i.delta && <div className="stat-d">{i.delta}</div>}
        </div>
      ))}
    </div>
  )
}

/* ─────────────── Prediction ─────────────── */

function Prediction({ s }: { s: Series }) {
  const last = s.dates[s.dates.length - 1]
  const [date, setDate] = useState(last)
  const [range, setRange] = useState<'all' | '20y' | '5y' | '1y'>('all')

  const idx = sessionOnOrBefore(s.dates, date)
  const valid = idx > 0 && Number.isFinite(s.pred[idx])
  const actual = valid ? s.close[idx] : NaN
  const predicted = valid ? s.pred[idx] : NaN
  const err = predicted - actual
  const errPct = Math.abs(err) / actual

  const n = s.t.length
  const start = useMemo(() => {
    if (range === 'all') return s.from
    const yrs = { '20y': 20, '5y': 5, '1y': 1 }[range]
    const cut = s.t[n - 1] - yrs * 365.25 * 864e5
    let i = s.from
    while (i < n && s.t[i] < cut) i++
    return i
  }, [range, s, n])

  const lines = useMemo(() => {
    const B = 700
    const [cx, cy] = downsample(s.t, s.close, start, n, B)
    const [ax, ay] = downsample(s.t, s.sma50, start, n, B)
    const [bx, by] = downsample(s.t, s.sma200, start, n, B)
    const out: Line[] = [
      { name: 'Close', x: cx, y: cy, color: C.accent2, width: 1.6 },
      { name: 'SMA 50', x: ax, y: ay, color: C.warn, width: 1, dash: '2 3' },
      { name: 'SMA 200', x: bx, y: by, color: '#5E5249', width: 1, dash: '6 4' },
    ]
    if (range === '1y' || range === '5y') {
      const [px, py] = downsample(s.t, s.pred, start, n, B)
      out.push({ name: 'LSTM', x: px, y: py, color: C.accent, width: 1.4 })
    }
    return out
  }, [s, start, n, range])

  const vol = useMemo(() => {
    const B = 160
    const size = (n - start) / B
    const x: number[] = [], y: number[] = []
    for (let b = 0; b < B; b++) {
      const a = start + Math.floor(b * size), z = Math.min(n, start + Math.floor((b + 1) * size))
      if (z <= a) continue
      let sum = 0
      for (let i = a; i < z; i++) sum += s.volume[i]
      x.push(s.t[Math.floor((a + z) / 2)]); y.push(sum / (z - a))
    }
    return { x, y }
  }, [s, start, n])

  const shown = idx >= start

  return (
    <div className="stack">
      <Section
        title="Point-in-time backtest"
        sub="Pick any session. The model sees only the closes before it, then predicts that day's close, so the error is an out-of-sample read on a single bar."
      >
        <div className="controls">
          <label className="field">
            <span>Session</span>
            <input
              type="date"
              value={date}
              min={s.dates[1]}
              max={last}
              onChange={e => e.target.value && setDate(e.target.value)}
            />
          </label>
          <div className="quick">
            {[
              ['Black Monday', '1987-10-19'],
              ['Dot-com peak', '2000-03-24'],
              ['Lehman', '2008-09-15'],
              ['COVID low', '2020-03-23'],
              ['Latest', last],
            ].map(([l, d]) => (
              <button key={l} className={date === d ? 'pill on' : 'pill'} onClick={() => setDate(d)}>{l}</button>
            ))}
          </div>
        </div>
        {s.dates[idx] !== date && idx > 0 && <p className="note">Not a trading day. Snapped to {fmtDate(s.dates[idx])}.</p>}

        {valid ? (
          <Stats
            items={[
              { label: 'Actual close', value: fmtUsd(actual), delta: fmtDate(s.dates[idx]) },
              { label: 'LSTM prediction', value: fmtUsd(predicted), delta: 'Sequence-to-point forecast', tone: 'accent' },
              { label: 'Absolute error', value: fmtUsd(Math.abs(err)), delta: `${err >= 0 ? '+' : '−'}${fmtUsd(Math.abs(err))} vs actual` },
              { label: 'Error', value: fmtPct(errPct), delta: 'Relative to actual close', tone: errPct < 0.02 ? 'pos' : errPct < 0.05 ? 'warn' : 'neg' },
            ]}
          />
        ) : (
          <div className="empty"><b>No history before that date.</b><p>Pick a session after {fmtDate(s.dates[1])}.</p></div>
        )}
      </Section>

      <Section title="Price history" sub="Close with 50 and 200-session moving averages. Log scale, so ninety years stay readable.">
        <div className="card">
          <div className="card-top">
            <Legend items={lines.map(l => ({ name: l.name, color: l.color, dash: !!l.dash }))} />
            <div className="seg">
              {(['all', '20y', '5y', '1y'] as const).map(r => (
                <button key={r} className={range === r ? 'on' : ''} onClick={() => setRange(r)}>{r === 'all' ? 'All' : r.toUpperCase()}</button>
              ))}
            </div>
          </div>
          <Chart
            lines={lines}
            marker={valid && shown ? { x: s.t[idx], y: actual, label: 'Selected' } : undefined}
            height={420}
            log={range === 'all' || range === '20y'}
            xKind="time"
            yFmt={usd0}
            ariaLabel="S&P 500 close price history with moving averages"
          />
          <Chart
            bars={{ x: vol.x, y: vol.y, color: 'rgba(43,38,34,0.28)', name: 'Volume' }}
            height={110}
            xKind="time"
            yFmt={v => (v >= 1e9 ? (v / 1e9).toFixed(1) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(0) + 'M' : v.toFixed(0))}
            ariaLabel="Average daily volume"
          />
        </div>
      </Section>
    </div>
  )
}

/* ─────────────── Monte Carlo ─────────────── */

function MonteCarlo({ s }: { s: Series }) {
  const [paths, setPaths] = useState(500)
  const [days, setDays] = useState(180)
  const [seed, setSeed] = useState(42)
  const [run, setRun] = useState<{ sim: Sim; ms: number } | null>(null)

  const n = s.t.length
  const start = s.close[n - 1]
  const mu = mean(s.ret, s.from), sigma = std(s.ret, s.from)

  const go = () => {
    const t0 = performance.now()
    const sim = simulate(start, mu, sigma, days, paths, seed)
    setRun({ sim, ms: performance.now() - t0 })
  }

  const view = useMemo(() => {
    if (!run) return null
    const { sim } = run
    const x = Array.from({ length: sim.days }, (_, i) => i)
    const b = bands(sim)
    const sample = Math.min(120, sim.paths)
    const ys: Float64Array[] = []
    for (let p = 0; p < sample; p++) {
      const y = new Float64Array(sim.days)
      for (let d = 0; d < sim.days; d++) y[d] = sim.data[d * sim.paths + p]
      ys.push(y)
    }
    const f = terminal(sim)
    const p5 = quantile(f, 0.05), p50 = quantile(f, 0.5), p95 = quantile(f, 0.95)
    const sd = std(f)
    // histogram
    const bins = 50, lo = f[0], hi = f[f.length - 1], w = (hi - lo) / bins || 1
    const hx: number[] = [], hy: number[] = []
    const counts = new Array(bins).fill(0)
    for (const v of f) counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++
    for (let i = 0; i < bins; i++) { hx.push(lo + (i + 0.5) * w); hy.push(counts[i]) }
    const below = f.findIndex(v => v >= start)
    return { x, b, ys, p5, p50, p95, sd, hx, hy, pLoss: (below < 0 ? f.length : below) / f.length }
  }, [run, start])

  return (
    <div className="stack">
      <Section
        title="Forward path simulation"
        sub="Draws daily returns from the historical mean and standard deviation, then compounds them forward. It models dispersion, not direction."
      >
        <div className="card controls-card">
          <label className="field range">
            <span>Paths <b>{paths.toLocaleString()}</b></span>
            <input type="range" min={100} max={2000} step={50} value={paths} onChange={e => setPaths(+e.target.value)} />
          </label>
          <label className="field range">
            <span>Horizon <b>{days} sessions</b></span>
            <input type="range" min={30} max={365} step={15} value={days} onChange={e => setDays(+e.target.value)} />
          </label>
          <label className="field seed">
            <span>Seed</span>
            <input type="number" value={seed} onChange={e => setSeed(Math.trunc(+e.target.value) || 0)} />
          </label>
          <button className="btn" onClick={go}>Run simulation</button>
        </div>
        <p className="note">
          μ = {fmtPct(mu, 3)} and σ = {fmtPct(sigma, 2)} per session, measured on 1930–2020 daily returns.
          {run && ` Last run: ${run.sim.paths.toLocaleString()} × ${run.sim.days} steps in ${run.ms.toFixed(0)} ms.`}
        </p>
      </Section>

      {view && run ? (
        <>
          <Stats
            items={[
              { label: 'Start', value: fmtUsd(start), delta: 'Final close · ' + fmtDate(s.dates[n - 1]) },
              { label: 'Median outcome', value: fmtUsd(view.p50), delta: `${fmtPct(view.p50 / start - 1, 1, true)} over ${run.sim.days} sessions`, tone: 'accent' },
              { label: '5th percentile', value: fmtUsd(view.p5), tone: 'neg' },
              { label: '95th percentile', value: fmtUsd(view.p95), tone: 'pos' },
              { label: 'Ends below start', value: fmtPct(view.pLoss, 1), delta: 'Share of paths' },
            ]}
          />
          <Section title="Fan chart" sub="5–95% band, median path and 120 sampled paths.">
            <div className="card">
              <Legend items={[{ name: 'Median', color: C.accent }, { name: '5–95% band', color: 'rgba(142,27,59,0.3)' }, { name: 'Sampled paths', color: C.faint }]} />
              <Chart
                band={{ x: view.x, lo: view.b.p5, hi: view.b.p95, fill: 'rgba(142,27,59,0.12)', name: '95/5%' }}
                paths={{ x: view.x, ys: view.ys, color: 'rgba(43,38,34,0.12)' }}
                lines={[{ name: 'Median', x: view.x, y: view.b.p50, color: C.accent, width: 2.4 }]}
                height={420}
                xKind="number"
                xFmt={v => `+${Math.round(v)}`}
                yFmt={usd0}
                ariaLabel="Monte Carlo fan chart"
              />
            </div>
          </Section>
          <Section title="Terminal distribution" sub={`Spread of final prices. Standard deviation ${fmtUsd(view.sd)}.`}>
            <div className="card">
              <Chart
                bars={{ x: view.hx, y: view.hy, color: 'rgba(142,27,59,0.75)', name: 'Paths' }}
                vlines={[
                  { x: view.p5, color: C.neg, label: '5th' },
                  { x: view.p50, color: '#2B2622', label: 'median' },
                  { x: view.p95, color: C.pos, label: '95th' },
                ]}
                height={280}
                xKind="number"
                xFmt={usd0}
                yFmt={v => String(Math.round(v))}
                ariaLabel="Histogram of terminal prices"
              />
            </div>
          </Section>
        </>
      ) : (
        <div className="empty">
          <b>No simulation yet</b>
          <p>Set the paths and the horizon, then run it. The seed makes each run reproducible.</p>
        </div>
      )}
    </div>
  )
}

/* ─────────────── Analytics ─────────────── */

function Analytics({ s }: { s: Series }) {
  const n = s.t.length
  const firstYear = +s.dates[s.from].slice(0, 4), lastYear = +s.dates[n - 1].slice(0, 4)
  const [y0, setY0] = useState(lastYear - 20)
  const [y1, setY1] = useState(lastYear)

  const st = useMemo(() => {
    const hi = argExtreme(s.close, s.from, 1), lo = argExtreme(s.close, s.from, -1)
    const dd = argExtreme(s.drawdown, s.from, -1)
    const best = argExtreme(s.ret, s.from, 1), worst = argExtreme(s.ret, s.from, -1)
    const m = mean(s.ret, s.from)
    const B = 800
    const [ddx, ddy] = downsample(s.t, s.drawdown.map(v => v * 100), s.from, n, B)
    const [vx, vy] = downsample(s.t, s.vol30.map(v => v * 100), s.from, n, B)
    return { hi, lo, dd, best, worst, m, ddx, ddy, vx, vy }
  }, [s, n])

  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => firstYear + i)
  const a = Math.min(y0, y1), b = Math.max(y0, y1)
  const rows = useMemo(() => {
    const out: number[] = []
    for (let i = s.from; i < n; i++) {
      const y = +s.dates[i].slice(0, 4)
      if (y >= a && y <= b) out.push(i)
    }
    return out
  }, [s, n, a, b])

  const download = () => {
    const head = 'Date,Close,Volume,SMA50,SMA200,LSTM_Prediction\n'
    const f = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '')
    const body = rows.map(i => [s.dates[i], f(s.close[i]), s.volume[i], f(s.sma50[i]), f(s.sma200[i]), f(s.pred[i])].join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([head + body], { type: 'text/csv' }))
    const el = document.createElement('a')
    el.href = url; el.download = `spx_${a}_${b}.csv`; el.click()
    URL.revokeObjectURL(url)
  }

  const shown = rows.slice(-300).reverse()

  return (
    <div className="stack">
      <Section title="Historical profile" sub="Whole-series statistics, 1930 to 2020.">
        <Stats
          items={[
            { label: 'All-time high', value: fmtUsd(s.close[st.hi]), delta: fmtDate(s.dates[st.hi]) },
            { label: 'All-time low', value: fmtUsd(s.close[st.lo]), delta: fmtDate(s.dates[st.lo]) },
            { label: 'Avg daily return', value: fmtPct(st.m, 3), delta: `≈ ${fmtPct(st.m * 252, 1)} annualised`, tone: 'pos' },
            { label: '30d volatility', value: fmtPct(s.vol30[n - 1], 1), delta: 'Annualised, final bar' },
            { label: 'Max drawdown', value: fmtPct(s.drawdown[st.dd], 1), delta: 'Trough ' + fmtDate(s.dates[st.dd]), tone: 'neg' },
            { label: 'Best / worst day', value: `${fmtPct(s.ret[st.best], 1, true)} / ${fmtPct(s.ret[st.worst], 1)}`, delta: `${s.dates[st.best].slice(0, 4)} / ${s.dates[st.worst].slice(0, 4)}` },
          ]}
        />
      </Section>

      <div className="two">
        <Section title="Drawdown" sub="Distance below the running all-time high.">
          <div className="card">
            <Chart
              lines={[{ name: 'Drawdown', x: st.ddx, y: st.ddy, color: C.neg, width: 1.1, fillTo: 0, fill: 'rgba(180,54,31,0.14)' }]}
              height={240}
              xKind="time"
              yFmt={v => v.toFixed(0) + '%'}
              yMax={0}
              ariaLabel="Drawdown from all-time high"
            />
          </div>
        </Section>
        <Section title="Rolling 30-session volatility" sub="Annualised.">
          <div className="card">
            <Chart
              lines={[{ name: 'Volatility', x: st.vx, y: st.vy, color: C.accent2, width: 1.1 }]}
              height={240}
              xKind="time"
              yFmt={v => v.toFixed(0) + '%'}
              yMin={0}
              ariaLabel="Rolling 30-session volatility"
            />
          </div>
        </Section>
      </div>

      <Section title="Data explorer" sub="Filter by year and export, including the model's prediction for each session.">
        <div className="card">
          <div className="card-top">
            <div className="controls tight">
              <label className="field">
                <span>From</span>
                <select value={y0} onChange={e => setY0(+e.target.value)}>{years.map(y => <option key={y}>{y}</option>)}</select>
              </label>
              <label className="field">
                <span>To</span>
                <select value={y1} onChange={e => setY1(+e.target.value)}>{years.map(y => <option key={y}>{y}</option>)}</select>
              </label>
            </div>
            <button className="btn ghost" onClick={download}>Download {a}–{b} CSV</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Date</th><th>Close</th><th>LSTM</th><th>Error</th><th className="hide-sm">SMA 50</th><th className="hide-sm">SMA 200</th><th className="hide-sm">Volume</th></tr>
              </thead>
              <tbody>
                {shown.map(i => {
                  const e = (s.pred[i] - s.close[i]) / s.close[i]
                  return (
                    <tr key={i}>
                      <td>{s.dates[i]}</td>
                      <td>{fmtUsd(s.close[i])}</td>
                      <td>{Number.isFinite(s.pred[i]) ? fmtUsd(s.pred[i]) : '—'}</td>
                      <td className={Math.abs(e) < 0.01 ? 'pos' : Math.abs(e) < 0.03 ? 'warn' : 'neg'}>{Number.isFinite(e) ? fmtPct(e, 2, true) : '—'}</td>
                      <td className="hide-sm">{Number.isFinite(s.sma50[i]) ? fmtUsd(s.sma50[i]) : '—'}</td>
                      <td className="hide-sm">{Number.isFinite(s.sma200[i]) ? fmtUsd(s.sma200[i]) : '—'}</td>
                      <td className="hide-sm">{s.volume[i].toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="note">Showing the latest {shown.length} of {rows.length.toLocaleString()} sessions. The CSV has all of them.</p>
        </div>
      </Section>
    </div>
  )
}

/* ─────────────── Model ─────────────── */

function Model({ s }: { s: Series }) {
  const acc = useMemo(() => {
    const cut = sessionOnOrBefore(s.dates, '2015-12-31') + 1
    const n = s.t.length
    let ape = 0, k = 0, hit = 0
    for (let i = cut; i < n; i++) {
      ape += Math.abs(s.pred[i] - s.close[i]) / s.close[i]; k++
      if (Math.sign(s.pred[i] - s.close[i - 1]) === Math.sign(s.close[i] - s.close[i - 1])) hit++
    }
    return { mape: ape / k, dir: hit / k, k }
  }, [s])

  const KV = ({ title, rows }: { title: string; rows: [string, string][] }) => (
    <div className="card kv">
      <h3>{title}</h3>
      {rows.map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}
    </div>
  )

  return (
    <div className="stack">
      <Section title="Architecture and training">
        <div className="two">
          <KV title="Network" rows={[
            ['Type', 'Stacked LSTM'], ['Hidden size', '256'], ['Layers', '3'], ['Dropout', '0.20'],
            ['Input features', '1 (close)'], ['Head', 'Linear → 1'], ['Parameters', s.meta.parameters.toLocaleString()],
          ]} />
          <KV title="Training and serving" rows={[
            ['Period', '1930 – 2020'], ['Observations', s.meta.sessions.toLocaleString()], ['Scaling', 'Min-max'],
            ['Framework', `PyTorch ${s.meta.torch}`], ['Serving', 'Precomputed, static JSON'],
            ['MAPE 2016–2020', fmtPct(acc.mape)], ['Direction hit 2016–2020', fmtPct(acc.dir, 1)],
          ]} />
        </div>
      </Section>

      <Section title="Forward pass">
        <div className="card flow">
          {['Sequence', 'LSTM 256', 'LSTM 256', 'LSTM 256', 'Dense', 'Close'].map((l, i, a) => (
            <div key={i} className="flow-step">
              <span className={i === 0 ? 'node in' : i === a.length - 1 ? 'node out' : i === 4 ? 'node dense' : 'node'}>{l}</span>
              {i < a.length - 1 && <span className="arrow" aria-hidden>→</span>}
            </div>
          ))}
        </div>
      </Section>

      <div className="two">
        <div className="card prose">
          <h3>How a prediction is made</h3>
          <p>
            The closes before the selected session are min-max scaled, pushed through three LSTM layers, and the
            final hidden state is mapped to one value by the dense head, then scaled back to dollars.
          </p>
          <p>
            The network is causal, so one forward pass over the whole series gives the model&apos;s view after every
            prefix. That pass ran once in PyTorch and shipped as {(s.meta.sessions / 1000).toFixed(1)}k predictions, which is why
            this site needs no Python server and answers instantly.
          </p>
        </div>
        <div className="card prose">
          <h3>Limits worth knowing</h3>
          <ul>
            <li><b>Univariate.</b> Close price only: no volume, macro or cross-asset inputs.</li>
            <li><b>Single step.</b> It predicts one bar ahead, not a path. Its low error mostly tracks yesterday&apos;s close; direction is near a coin flip ({fmtPct(acc.dir, 1)}).</li>
            <li><b>Regime-bound.</b> Trained through 2020; the market you trade may behave differently.</li>
            <li><b>Not advice.</b> A research artefact. Do not size positions with it.</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
