/** Seeded PRNG (mulberry32) so a run is reproducible from its seed. */
export function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Standard normal via Box–Muller, keeping the second draw for the next call. */
export function normal(u: () => number) {
  let spare: number | null = null
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s }
    let a = 0
    while (a === 0) a = u()
    const b = u()
    const r = Math.sqrt(-2 * Math.log(a))
    spare = r * Math.sin(2 * Math.PI * b)
    return r * Math.cos(2 * Math.PI * b)
  }
}

/** data[step * paths + path]; step 0 is the start price. */
export type Sim = { days: number; paths: number; data: Float64Array }

/** Geometric random walk on daily returns drawn from N(mu, sigma). */
export function simulate(start: number, mu: number, sigma: number, days: number, paths: number, seed: number): Sim {
  const z = normal(rng(seed))
  const data = new Float64Array(days * paths)
  for (let p = 0; p < paths; p++) data[p] = start
  for (let s = 1; s < days; s++) {
    const o = s * paths, prev = o - paths
    for (let p = 0; p < paths; p++) data[o + p] = data[prev + p] * (1 + mu + sigma * z())
  }
  return { days, paths, data }
}

export function quantile(sorted: ArrayLike<number>, q: number): number {
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos), hi = Math.ceil(pos)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

/** Per-step 5th, 50th and 95th percentiles. */
export function bands(sim: Sim) {
  const p5 = new Float64Array(sim.days), p50 = new Float64Array(sim.days), p95 = new Float64Array(sim.days)
  const row = new Float64Array(sim.paths)
  for (let s = 0; s < sim.days; s++) {
    row.set(sim.data.subarray(s * sim.paths, (s + 1) * sim.paths))
    row.sort()
    p5[s] = quantile(row, 0.05); p50[s] = quantile(row, 0.5); p95[s] = quantile(row, 0.95)
  }
  return { p5, p50, p95 }
}

/** Final prices, sorted ascending. */
export function terminal(sim: Sim): Float64Array {
  const f = sim.data.slice((sim.days - 1) * sim.paths)
  f.sort()
  return f
}
