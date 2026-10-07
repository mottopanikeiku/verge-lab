import { clamp, formatPercent } from '../lib'

type ConfidenceDialProps = {
  value: number
  tone: 'defended' | 'ambiguous' | 'neutral'
  label: string
  size?: 'inline' | 'panel'
}

export function ConfidenceDial({
  value,
  tone,
  label,
  size = 'inline',
}: ConfidenceDialProps) {
  const normalizedValue = clamp(Number.isFinite(value) ? value : 0, 0, 1)
  const formattedValue = formatPercent(normalizedValue)

  return (
    <span
      role="img"
      className={`confidence-dial confidence-dial--${tone} confidence-dial--${size}`}
      aria-label={`Illustrative ${label}: ${formattedValue}`}
    >
      <span className="confidence-dial__copy">
        <span className="confidence-dial__label">Illustrative {label}</span>
        <strong className="confidence-dial__value">{formattedValue}</strong>
      </span>
    </span>
  )
}
