import type { CSSProperties } from 'react'
import { clamp, formatPercent } from '../lib'

type ConfidenceDialProps = {
  value: number
  tone: 'defended' | 'ambiguous' | 'neutral'
  label: string
  size?: 'inline' | 'panel'
}

type ConfidenceStyle = CSSProperties & {
  '--confidence-value': number
}

export function ConfidenceDial({
  value,
  tone,
  label,
  size = 'inline',
}: ConfidenceDialProps) {
  const normalizedValue = clamp(Number.isFinite(value) ? value : 0, 0, 1)
  const formattedValue = formatPercent(normalizedValue)
  const style = { '--confidence-value': normalizedValue * 100 } as ConfidenceStyle

  return (
    <span
      className={`confidence-dial confidence-dial--${tone} confidence-dial--${size}`}
      style={style}
    >
      <span className="confidence-dial__meter" aria-hidden="true">
        <span className="confidence-dial__needle" />
      </span>
      <span className="confidence-dial__copy">
        <span className="confidence-dial__label">{label}</span>
        <strong className="confidence-dial__value">{formattedValue}</strong>
      </span>
    </span>
  )
}
