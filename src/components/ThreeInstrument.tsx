import { Component, Suspense, lazy, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { Activity } from 'lucide-react'
import { clamp } from '../lib'

const LazyConstellationField = lazy(() =>
  import('@designcodeio/threeui/components/ConstellationField').then((module) => ({
    default: module.ConstellationField,
  })),
)
const LazyTopologyField = lazy(() =>
  import('@designcodeio/threeui/components/TopologyField').then((module) => ({
    default: module.TopologyField,
  })),
)
const LazyDiagnosticsPanel = lazy(() =>
  import('@designcodeio/threeui/components/DiagnosticsPanel').then((module) => ({
    default: module.DiagnosticsPanel,
  })),
)

type ThreeInstrumentProps = {
  kind: 'constellation' | 'topology' | 'diagnostics'
  title: string
  description: string
  metrics: {
    speed: number
    density: number
    opacity: number
    hue: number
    brightness: number
  }
  compact?: boolean
}

type InstrumentBoundaryProps = {
  children: ReactNode
}

type InstrumentBoundaryState = {
  failed: boolean
}

class InstrumentBoundary extends Component<InstrumentBoundaryProps, InstrumentBoundaryState> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="instrument-fallback" role="status">
          Visual instrument unavailable. The run evidence remains accessible in the tables below.
        </div>
      )
    }

    return this.props.children
  }
}

function useReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const updatePreference = () => setReducedMotion(media.matches)
    media.addEventListener('change', updatePreference)
    return () => media.removeEventListener('change', updatePreference)
  }, [])

  return reducedMotion
}

function InstrumentVisual({ kind, metrics }: Pick<ThreeInstrumentProps, 'kind' | 'metrics'>) {
  const reducedMotion = useReducedMotion()
  const mapped = {
    speed: reducedMotion ? 0 : clamp(metrics.speed, 0, 1.6),
    density: clamp(metrics.density, 0.35, 1.7),
    opacity: clamp(metrics.opacity, 0.35, 1),
    hue: clamp(metrics.hue, -1, 1),
    brightness: clamp(metrics.brightness, 0.55, 1.35),
  }

  if (kind === 'constellation') {
    return (
      <LazyConstellationField
        variant="particle-network"
        mode="dark"
        speed={mapped.speed}
        density={mapped.density}
        opacity={mapped.opacity}
        hue={mapped.hue}
        brightness={mapped.brightness}
        style={{ width: '100%', height: '100%' }}
      />
    )
  }

  if (kind === 'diagnostics') {
    return (
      <LazyDiagnosticsPanel
        variant="flow"
        mode="dark"
        speed={mapped.speed}
        density={mapped.density}
        opacity={mapped.opacity}
        hue={mapped.hue}
        brightness={mapped.brightness}
        style={{ width: '100%', height: '100%' }}
      />
    )
  }

  return (
    <LazyTopologyField
      mode="dark"
      hue={mapped.hue}
      brightness={mapped.brightness}
      style={{ width: '100%', height: '100%' }}
    />
  )
}

export function ThreeInstrument({
  kind,
  title,
  description,
  metrics,
  compact = false,
}: ThreeInstrumentProps) {
  return (
    <figure className={`three-instrument${compact ? ' three-instrument--compact' : ''}`}>
      <div className="instrument-heading">
        <figcaption>
          <span className="instrument-kicker">
            <Activity size={13} aria-hidden="true" />
            Assistive field
          </span>
          <strong>{title}</strong>
        </figcaption>
        <span className="instrument-live">live mapping</span>
      </div>
      <div className="instrument-viewport" aria-hidden="true">
        <InstrumentBoundary>
          <Suspense fallback={<div className="instrument-loading">Calibrating field…</div>}>
            <InstrumentVisual kind={kind} metrics={metrics} />
          </Suspense>
        </InstrumentBoundary>
      </div>
      <p>{description} Decorative signal only; it does not expose model internals.</p>
    </figure>
  )
}
