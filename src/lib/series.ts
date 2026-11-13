/** Raw payload written by model/export_predictions.py. */
export type Payload = {
  meta: { index: string; first: string; last: string; sessions: number; parameters: number; torch: string }
  d: string[]
  c: number[]
  v: number[]
  p: (number | null)[]
}

export type Series = {
  meta: Payload['meta']
  /** All sessions, used for the model lookup. */
  dates: string[]
  t: Float64Array // epoch ms
  close: Float64Array
  volume: Float64Array
  pred: Float64Array // NaN where unavailable
  /** Chart window starts here: 1930 onward, as in the original app. */
  from: number
  sma50: Float64Array
  sma200: Float64Array
  ret: Float64Array
  vol30: Float64Array
  drawdown: Float64Array
}

export const CHART_START = '1930-01-01'

export function rolling(src: Float64Array, n: number, fn: 'mean' | 'std'): Float64Array {
  const out = new Float64Array(src.length).fill(NaN)
  let s = 0, s2 = 0, k = 0
  const q: number[] = []
  for (let i = 0; i < src.length; i++) {
    const x = src[i]
    if (Number.isNaN(x)) { q.length = 0; s = s2 = k = 0; continue }
    q.push(x); s += x; s2 += x * x; k++
    if (k > n) { const y = q.shift()!; s -= y; s2 -= y * y; k-- }
    if (k === n) {
      const m = s / n
      out[i] = fn === 'mean' ? m : Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1)))
    }
  }
  return out
}

export function build(p: Payload): Series {
  const n = p.d.length
  const t = new Float64Array(n), close = new Float64Array(n), volume = new Float64Array(n), pred = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    t[i] = Date.parse(p.d[i] + 'T00:00:00Z')
    close[i] = p.c[i]; volume[i] = p.v[i]; pred[i] = p.p[i] ?? NaN
  }
  const from = Math.max(0, p.d.findIndex(d => d >= CHART_START))
  const ret = new Float64Array(n).fill(NaN)
  for (let i = from + 1; i < n; i++) ret[i] = close[i] / close[i - 1] - 1
  const cl = new Float64Array(n).fill(NaN)
  cl.set(close.subarray(from), from)
  const vol30 = rolling(ret, 30, 'std').map(x => x * Math.sqrt(252))
  const drawdown = new Float64Array(n).fill(NaN)
  let peak = -Infinity
  for (let i = from; i < n; i++) { peak = Math.max(peak, close[i]); drawdown[i] = close[i] / peak - 1 }
  return {
    meta: p.meta, dates: p.d, t, close, volume, pred, from,
    sma50: rolling(cl, 50, 'mean'), sma200: rolling(cl, 200, 'mean'), ret, vol30, drawdown,
  }
}

/** Index of the session on or before an ISO date (snaps weekends and holidays back). */
export function sessionOnOrBefore(dates: string[], iso: string): number {
  let lo = 0, hi = dates.length - 1, ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (dates[mid] <= iso) { ans = mid; lo = mid + 1 } else hi = mid - 1
  }
  return ans
}

export function mean(a: ArrayLike<number>, from = 0): number {
  let s = 0, k = 0
  for (let i = from; i < a.length; i++) if (!Number.isNaN(a[i])) { s += a[i]; k++ }
  return k ? s / k : NaN
}

export function std(a: ArrayLike<number>, from = 0): number {
  const m = mean(a, from)
  let s = 0, k = 0
  for (let i = from; i < a.length; i++) if (!Number.isNaN(a[i])) { s += (a[i] - m) ** 2; k++ }
  return k > 1 ? Math.sqrt(s / (k - 1)) : NaN
}

export function argExtreme(a: ArrayLike<number>, from: number, dir: 1 | -1): number {
  let best = -1
  for (let i = from; i < a.length; i++) {
    if (Number.isNaN(a[i])) continue
    if (best < 0 || (dir > 0 ? a[i] > a[best] : a[i] < a[best])) best = i
  }
  return best
}

/**
 * Min/max bucket downsampling for line charts: keeps every spike while drawing
 * about two points per pixel instead of 23,000.
 */
export function downsample(x: ArrayLike<number>, y: ArrayLike<number>, from: number, to: number, buckets: number): [number[], number[]] {
  const xs: number[] = [], ys: number[] = []
  const n = to - from
  if (n <= buckets * 2) {
    for (let i = from; i < to; i++) if (!Number.isNaN(y[i])) { xs.push(x[i]); ys.push(y[i]) }
    return [xs, ys]
  }
  const size = n / buckets
  for (let b = 0; b < buckets; b++) {
    const a = from + Math.floor(b * size), z = Math.min(to, from + Math.floor((b + 1) * size))
    let lo = -1, hi = -1
    for (let i = a; i < z; i++) {
      if (Number.isNaN(y[i])) continue
      if (lo < 0 || y[i] < y[lo]) lo = i
      if (hi < 0 || y[i] > y[hi]) hi = i
    }
    if (lo < 0) continue
    const [p, q] = lo < hi ? [lo, hi] : [hi, lo]
    xs.push(x[p]); ys.push(y[p])
    if (q !== p) { xs.push(x[q]); ys.push(y[q]) }
  }
  return [xs, ys]
}

export const fmtUsd = (v: number, d = 2) =>
  '$' + v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
export const fmtPct = (v: number, d = 2, sign = false) => (sign && v > 0 ? '+' : '') + (v * 100).toFixed(d) + '%'
export const fmtDate = (iso: string) =>
  new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' })
