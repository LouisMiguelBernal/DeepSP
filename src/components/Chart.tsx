'use client'

import { useMemo, useRef, useState, useEffect, type ReactNode } from 'react'

export type Line = {
  name: string
  x: ArrayLike<number>
  y: ArrayLike<number>
  color: string
  width?: number
  dash?: string
  /** Fill between the line and this y value (e.g. 0 for drawdown). */
  fillTo?: number
  fill?: string
  hover?: boolean
}

export type Band = { x: ArrayLike<number>; lo: ArrayLike<number>; hi: ArrayLike<number>; fill: string; name: string }

export type Bars = { x: ArrayLike<number>; y: ArrayLike<number>; color: string; name: string }

type Props = {
  lines?: Line[]
  band?: Band
  /** Many faint paths drawn as one SVG path. */
  paths?: { x: ArrayLike<number>; ys: ArrayLike<number>[]; color: string }
  bars?: Bars
  marker?: { x: number; y: number; label: string }
  vlines?: { x: number; color: string; label: string }[]
  height: number
  log?: boolean
  xKind: 'time' | 'number'
  yFmt: (v: number) => string
  xFmt?: (v: number) => string
  yMin?: number
  yMax?: number
  ariaLabel: string
  footer?: ReactNode
}

const PAD = { l: 58, r: 14, t: 12, b: 26 }

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) ?? 10 * mag
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = []
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 3]) {
      const v = m * 10 ** e
      if (v >= lo && v <= hi) out.push(v)
    }
  return out
}

function yearTicks(lo: number, hi: number, maxCount: number): number[] {
  const y0 = new Date(lo).getUTCFullYear(), y1 = new Date(hi).getUTCFullYear()
  const span = Math.max(1, y1 - y0)
  const step = [1, 2, 5, 10, 20, 25, 50].find(s => span / s <= maxCount) ?? 50
  const out: number[] = []
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) out.push(Date.UTC(y, 0, 1))
  return out
}

function nearest(x: ArrayLike<number>, v: number): number {
  let lo = 0, hi = x.length - 1
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1
    if (x[m] < v) lo = m; else hi = m
  }
  return Math.abs(x[lo] - v) <= Math.abs(x[hi] - v) ? lo : hi
}

export default function Chart(props: Props) {
  const { lines = [], band, paths, bars, marker, vlines, height, log, xKind, yFmt, ariaLabel, footer } = props
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hx, setHx] = useState<number | null>(null)
  const narrow = width < 520
  const pad = { ...PAD, l: narrow ? 46 : PAD.l }

  const geom = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    const scanX = (x: ArrayLike<number>) => { if (x.length) { x0 = Math.min(x0, x[0]); x1 = Math.max(x1, x[x.length - 1]) } }
    const scanY = (y: ArrayLike<number>) => {
      for (let i = 0; i < y.length; i++) {
        const v = y[i]
        if (Number.isFinite(v) && (!log || v > 0)) { if (v < y0) y0 = v; if (v > y1) y1 = v }
      }
    }
    lines.forEach(l => { scanX(l.x); scanY(l.y); if (l.fillTo !== undefined) scanY([l.fillTo]) })
    if (band) { scanX(band.x); scanY(band.lo); scanY(band.hi) }
    if (paths) { scanX(paths.x); paths.ys.forEach(scanY) }
    if (bars) {
      for (let i = 0; i < bars.x.length; i++) { x0 = Math.min(x0, bars.x[i]); x1 = Math.max(x1, bars.x[i]) }
      scanY(bars.y); y0 = Math.min(y0, 0)
    }
    if (props.yMin !== undefined) y0 = props.yMin
    if (props.yMax !== undefined) y1 = props.yMax
    if (!log && props.yMin === undefined && !bars) { const r = (y1 - y0) * 0.06; y0 -= r; y1 += r }
    return { x0, x1, y0, y1 }
  }, [lines, band, paths, bars, log, props.yMin, props.yMax])

  const iw = Math.max(10, width - pad.l - pad.r), ih = height - pad.t - pad.b
  const { x0, x1, y0, y1 } = geom
  const sx = (v: number) => pad.l + ((v - x0) / (x1 - x0 || 1)) * iw
  const ly0 = Math.log10(y0), ly1 = Math.log10(y1)
  const sy = (v: number) =>
    log ? pad.t + ih - ((Math.log10(v) - ly0) / (ly1 - ly0 || 1)) * ih : pad.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih

  const pathOf = (x: ArrayLike<number>, y: ArrayLike<number>) => {
    let d = '', pen = false
    for (let i = 0; i < x.length; i++) {
      const v = y[i]
      if (!Number.isFinite(v) || (log && v <= 0)) { pen = false; continue }
      d += (pen ? 'L' : 'M') + sx(x[i]).toFixed(1) + ' ' + sy(v).toFixed(1)
      pen = true
    }
    return d
  }

  const yt = log ? logTicks(y0, y1) : niceTicks(y0, y1, narrow ? 4 : 5)
  const xt = xKind === 'time' ? yearTicks(x0, x1, Math.max(3, Math.floor(iw / 80))) : niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90)))
  const xFmt = props.xFmt ?? (xKind === 'time' ? (v: number) => String(new Date(v).getUTCFullYear()) : (v: number) => String(Math.round(v)))

  const hoverLines = lines.filter(l => l.hover !== false)
  const tip = (() => {
    if (hx === null || !width) return null
    const xv = x0 + ((hx - pad.l) / iw) * (x1 - x0)
    const rows: { name: string; color: string; v: number; x: number }[] = []
    for (const l of hoverLines) {
      if (!l.x.length) continue
      const i = nearest(l.x, xv)
      if (Number.isFinite(l.y[i])) rows.push({ name: l.name, color: l.color, v: l.y[i], x: l.x[i] })
    }
    if (band && band.x.length) {
      const i = nearest(band.x, xv)
      rows.push({ name: band.name + ' high', color: 'var(--faint)', v: band.hi[i], x: band.x[i] })
      rows.push({ name: band.name + ' low', color: 'var(--faint)', v: band.lo[i], x: band.x[i] })
    }
    if (!rows.length) return null
    const at = rows[0].x
    const label = xKind === 'time'
      ? new Date(at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' })
      : xFmt(at)
    return { px: sx(at), label, rows }
  })()

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          onPointerMove={e => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
            const x = e.clientX - r.left
            setHx(x >= pad.l && x <= pad.l + iw ? x : null)
          }}
          onPointerLeave={() => setHx(null)}
        >
          {yt.map(v => (
            <g key={'y' + v}>
              <line x1={pad.l} x2={pad.l + iw} y1={sy(v)} y2={sy(v)} className="grid" />
              <text x={pad.l - 8} y={sy(v)} className="tick" textAnchor="end" dominantBaseline="middle">{yFmt(v)}</text>
            </g>
          ))}
          {xt.map(v => (
            <text key={'x' + v} x={sx(v)} y={height - 6} className="tick" textAnchor="middle">{xFmt(v)}</text>
          ))}

          {bars && (() => {
            const bw = Math.max(1, iw / bars.x.length - 1)
            return (
              <g fill={bars.color}>
                {Array.from({ length: bars.x.length }, (_, i) => (
                  <rect key={i} x={sx(bars.x[i]) - bw / 2} y={sy(bars.y[i])} width={bw} height={Math.max(0, sy(0) - sy(bars.y[i]))} rx={1} />
                ))}
              </g>
            )
          })()}

          {band && (() => {
            let d = ''
            for (let i = 0; i < band.x.length; i++) d += (i ? 'L' : 'M') + sx(band.x[i]).toFixed(1) + ' ' + sy(band.hi[i]).toFixed(1)
            for (let i = band.x.length - 1; i >= 0; i--) d += 'L' + sx(band.x[i]).toFixed(1) + ' ' + sy(band.lo[i]).toFixed(1)
            return <path d={d + 'Z'} fill={band.fill} />
          })()}

          {paths && <path d={paths.ys.map(y => pathOf(paths.x, y)).join('')} fill="none" stroke={paths.color} strokeWidth={0.8} />}

          {lines.map(l => {
            const d = pathOf(l.x, l.y)
            return (
              <g key={l.name}>
                {l.fillTo !== undefined && l.x.length > 0 && (
                  <path
                    d={`${d}L${sx(l.x[l.x.length - 1]).toFixed(1)} ${sy(l.fillTo).toFixed(1)}L${sx(l.x[0]).toFixed(1)} ${sy(l.fillTo).toFixed(1)}Z`}
                    fill={l.fill ?? 'none'}
                  />
                )}
                <path d={d} fill="none" stroke={l.color} strokeWidth={l.width ?? 1.4} strokeDasharray={l.dash} strokeLinejoin="round" />
              </g>
            )
          })}

          {vlines?.map(v => (
            <g key={v.label}>
              <line x1={sx(v.x)} x2={sx(v.x)} y1={pad.t} y2={pad.t + ih} stroke={v.color} strokeDasharray="3 4" strokeWidth={1.4} />
              <text x={sx(v.x) + 5} y={pad.t + 12} className="tick" fill={v.color} style={{ fill: v.color }}>{v.label}</text>
            </g>
          ))}

          {marker && (
            <g>
              <circle cx={sx(marker.x)} cy={sy(marker.y)} r={10} className="marker-halo" />
              <circle cx={sx(marker.x)} cy={sy(marker.y)} r={5} className="marker" />
            </g>
          )}

          {tip && (
            <g pointerEvents="none">
              <line x1={tip.px} x2={tip.px} y1={pad.t} y2={pad.t + ih} className="cross" />
              {tip.rows.slice(0, hoverLines.length).map(r => (
                <circle key={r.name} cx={tip.px} cy={sy(r.v)} r={3.5} fill={r.color} stroke="var(--bg)" strokeWidth={1.5} />
              ))}
            </g>
          )}
        </svg>
      )}
      {tip && (
        <div className="tip" style={{ left: Math.min(Math.max(tip.px, 90), width - 90) }}>
          <div className="tip-h">{tip.label}</div>
          {tip.rows.map(r => (
            <div key={r.name} className="tip-r">
              <span className="sw" style={{ background: r.color }} />
              <span>{r.name}</span>
              <b>{yFmt(r.v)}</b>
            </div>
          ))}
        </div>
      )}
      {footer}
    </div>
  )
}

export function Legend({ items }: { items: { name: string; color: string; dash?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map(i => (
        <span key={i.name}>
          <i style={{ background: i.dash ? 'transparent' : i.color, borderTop: i.dash ? `2px dashed ${i.color}` : undefined }} />
          {i.name}
        </span>
      ))}
    </div>
  )
}
