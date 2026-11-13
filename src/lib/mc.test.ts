import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, bands, terminal, rng } from './mc.ts'
import { rolling, sessionOnOrBefore, downsample } from './series.ts'

test('simulation is reproducible from its seed', () => {
  const a = simulate(100, 0.0003, 0.01, 50, 200, 42)
  const b = simulate(100, 0.0003, 0.01, 50, 200, 42)
  assert.deepEqual(a.data, b.data)
  assert.notDeepEqual(a.data, simulate(100, 0.0003, 0.01, 50, 200, 7).data)
})

test('step 0 is the start price and bands are ordered', () => {
  const s = simulate(3000, 0.0003, 0.012, 120, 500, 1)
  assert.equal(s.data[0], 3000)
  const { p5, p50, p95 } = bands(s)
  for (let i = 0; i < s.days; i++) assert.ok(p5[i] <= p50[i] && p50[i] <= p95[i])
  const f = terminal(s)
  assert.equal(f.length, 500)
  for (let i = 1; i < f.length; i++) assert.ok(f[i - 1] <= f[i])
})

test('median drift matches mu over a long horizon', () => {
  const mu = 0.0004, sigma = 0.01, days = 252
  const f = terminal(simulate(1, mu, sigma, days, 4000, 9))
  const med = f[2000]
  const expected = Math.exp((days - 1) * (Math.log(1 + mu) - (sigma * sigma) / 2))
  assert.ok(Math.abs(med / expected - 1) < 0.02, `median ${med} vs ${expected}`)
})

test('rng is uniform', () => {
  const u = rng(123)
  let s = 0
  for (let i = 0; i < 1e5; i++) s += u()
  assert.ok(Math.abs(s / 1e5 - 0.5) < 0.01)
})

test('rolling mean and std', () => {
  const a = Float64Array.from([1, 2, 3, 4, 5])
  const m = rolling(a, 3, 'mean')
  assert.ok(Number.isNaN(m[1]))
  assert.equal(m[2], 2)
  assert.equal(m[4], 4)
  assert.ok(Math.abs(rolling(a, 3, 'std')[4] - 1) < 1e-12)
})

test('sessionOnOrBefore snaps back to the previous session', () => {
  const d = ['2020-01-02', '2020-01-03', '2020-01-06']
  assert.equal(sessionOnOrBefore(d, '2020-01-05'), 1)
  assert.equal(sessionOnOrBefore(d, '2020-01-06'), 2)
  assert.equal(sessionOnOrBefore(d, '2019-12-31'), -1)
})

test('downsample keeps extremes', () => {
  const n = 10000
  const x = Float64Array.from({ length: n }, (_, i) => i)
  const y = Float64Array.from({ length: n }, (_, i) => (i === 5000 ? 99 : Math.sin(i / 100)))
  const [, ys] = downsample(x, y, 0, n, 200)
  assert.ok(ys.length <= 400)
  assert.ok(ys.includes(99))
})
