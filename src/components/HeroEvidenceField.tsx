import type { CSSProperties } from 'react'
import type { Candidate, RunArtifact } from '../types'

type HeroEvidenceFieldProps = {
  run: RunArtifact
}

type HeroPoint = Candidate & {
  x: number
  y: number
}

type HeroVisualStyle = CSSProperties & {
  '--hero-delay'?: string
  '--aperture-fill'?: number
}

const FIELD_WIDTH = 1000
const FIELD_HEIGHT = 360
const FIELD_LEFT = 390
const FIELD_RIGHT = 960
const FIELD_TOP = 44
const FIELD_BOTTOM = 316
const APERTURE_RADIUS = 74
const APERTURE_CIRCUMFERENCE = 2 * Math.PI * APERTURE_RADIUS

function projectCandidates(candidates: Candidate[]): HeroPoint[] {
  if (candidates.length === 0) return []
  const xValues = candidates.map((candidate) => candidate.embedding.x)
  const yValues = candidates.map((candidate) => candidate.embedding.y)
  const minimumX = Math.min(...xValues)
  const maximumX = Math.max(...xValues)
  const minimumY = Math.min(...yValues)
  const maximumY = Math.max(...yValues)
  const xRange = maximumX - minimumX || 1
  const yRange = maximumY - minimumY || 1

  return candidates.map((candidate, index) => ({
    ...candidate,
    x: FIELD_LEFT + ((candidate.embedding.x - minimumX) / xRange) * (FIELD_RIGHT - FIELD_LEFT),
    y: FIELD_TOP + ((candidate.embedding.y - minimumY) / yRange) * (FIELD_BOTTOM - FIELD_TOP) + ((index % 3) - 1) * 8,
  }))
}

function edgeCurve(source: HeroPoint, target: HeroPoint, index: number) {
  const midpointX = (source.x + target.x) / 2
  const midpointY = (source.y + target.y) / 2
  const normalX = target.y - source.y
  const normalY = source.x - target.x
  const length = Math.max(Math.hypot(normalX, normalY), 1)
  const bend = (index % 2 === 0 ? 1 : -1) * Math.min(34, length * 0.12)
  const controlX = midpointX + (normalX / length) * bend
  const controlY = midpointY + (normalY / length) * bend
  return `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} Q ${controlX.toFixed(1)} ${controlY.toFixed(1)} ${target.x.toFixed(1)} ${target.y.toFixed(1)}`
}


export function HeroEvidenceField({ run }: HeroEvidenceFieldProps) {
  const points = projectCandidates(run.candidates)
  const pointById = new Map(points.map((point) => [point.id, point]))
  const totalPairs = run.summary.defendedPairCount + run.summary.ambiguousPairCount
  const defendedRate = totalPairs === 0 ? 0 : run.summary.defendedPairCount / totalPairs
  const apertureStyle = {
    '--aperture-fill': defendedRate,
    strokeDasharray: `${(APERTURE_CIRCUMFERENCE * defendedRate).toFixed(1)} ${APERTURE_CIRCUMFERENCE.toFixed(1)}`,
  } as HeroVisualStyle

  return (
    <div className="hero-evidence-field" aria-hidden="true">
      <svg viewBox={`0 0 ${FIELD_WIDTH} ${FIELD_HEIGHT}`} preserveAspectRatio="xMidYMid slice">
        <defs>
          <filter id="hero-node-glow" x="-200%" y="-200%" width="400%" height="400%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <radialGradient id="hero-aperture-well" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--lime)" stopOpacity=".1" />
            <stop offset="72%" stopColor="var(--assay-0)" stopOpacity=".1" />
            <stop offset="100%" stopColor="var(--assay-0)" stopOpacity="0" />
          </radialGradient>
        </defs>
        <g className="hero-evidence-field__grid">
          {Array.from({ length: 11 }, (_, index) => <line key={`vertical-${index}`} x1={index * 100} y1="0" x2={index * 100} y2={FIELD_HEIGHT} />)}
          {Array.from({ length: 7 }, (_, index) => <line key={`horizontal-${index}`} x1="0" y1={index * 60} x2={FIELD_WIDTH} y2={index * 60} />)}
        </g>
        <g className="hero-evidence-field__edges">
          {run.pairs.map((pair, index) => {
            const source = pointById.get(pair.rejectedId)
            const target = pointById.get(pair.chosenId)
            if (!source || !target) return null
            const style = { '--hero-delay': `${120 + index * 32}ms` } as HeroVisualStyle
            return <path key={pair.id} d={edgeCurve(source, target, index)} className={`hero-edge hero-edge--${pair.verdict}`} style={style} pathLength="1" />
          })}
        </g>
        <g className="hero-aperture" transform="translate(760 176)">
          <circle className="hero-aperture__well" r="112" fill="url(#hero-aperture-well)" />
          <circle className="hero-aperture__ticks" r="96" />
          <circle className="hero-aperture__track" r={APERTURE_RADIUS} />
          <circle className="hero-aperture__value" r={APERTURE_RADIUS} style={apertureStyle} transform="rotate(-90)" />
          <text className="hero-aperture__number" textAnchor="middle" y="3">{Math.round(defendedRate * 100)}%</text>
          <text className="hero-aperture__label" textAnchor="middle" y="25">defended yield</text>
        </g>
        <g className="hero-evidence-field__nodes">
          {points.map((point, index) => {
            const confidenceValues = Object.values(point.scores).map((score) => score.confidence)
            const meanConfidence = confidenceValues.length === 0 ? 0 : confidenceValues.reduce((sum, confidence) => sum + confidence, 0) / confidenceValues.length
            const radius = 4.5 + meanConfidence * 3.5
            let tone = 'ambiguous'
            if (run.pairs.some((pair) => pair.verdict === 'defended' && pair.chosenId === point.id)) {
              tone = 'defended'
            } else if (run.pairs.some((pair) => pair.verdict === 'defended' && pair.rejectedId === point.id)) {
              tone = 'reference'
            }
            const style = { '--hero-delay': `${280 + index * 44}ms` } as HeroVisualStyle
            return (
              <g key={point.id} className={`hero-node hero-node--${tone}`} transform={`translate(${point.x.toFixed(1)} ${point.y.toFixed(1)})`} style={style}>
                <title>{`${point.id}: ${Math.round(meanConfidence * 100)}% mean verifier confidence`}</title>
                <circle className="hero-node__halo" r={radius + 7} />
                <circle className="hero-node__core" r={radius} filter="url(#hero-node-glow)" />
                <text x={radius + 7} y="3">{String(index + 1).padStart(2, '0')}</text>
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}
