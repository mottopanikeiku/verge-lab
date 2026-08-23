import { useId } from 'react'
import type { Checkpoint } from '../types'

type CheckpointTraceProps = {
  checkpoints: Checkpoint[]
}

type PlotPoint = {
  x: number
  y: number
}

const CHART_WIDTH = 760
const CHART_HEIGHT = 270
const PLOT_LEFT = 58
const PLOT_RIGHT = 704
const PLOT_TOP = 28
const PLOT_BOTTOM = 220
const stepFormatter = new Intl.NumberFormat('en-US')

const linePath = (points: PlotPoint[]) =>
  points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')

const formatDecimal = (value: number) => value.toFixed(3)

export function CheckpointTrace({ checkpoints }: CheckpointTraceProps) {
  const chartId = useId().replace(/:/g, '')
  const data = checkpoints
    .filter((checkpoint) =>
      [checkpoint.step, checkpoint.trainLoss, checkpoint.evalReward, checkpoint.defendedWinRate].every(Number.isFinite),
    )
    .slice()
    .sort((left, right) => left.step - right.step)

  if (data.length === 0) {
    return (
      <figure className="checkpoint-trace checkpoint-trace--empty">
        <figcaption>
          <span className="eyebrow">Evaluation telemetry</span>
          <strong>No calibrated states to plot</strong>
        </figcaption>
        <p>The checkpoint register has no finite reward, win-rate, or loss observations.</p>
      </figure>
    )
  }

  const first = data[0]
  const latest = data[data.length - 1]
  const minStep = first.step
  const maxStep = latest.step
  const stepSpan = Math.max(maxStep - minStep, 1)
  const primaryValues = data.flatMap((checkpoint) => [checkpoint.evalReward, checkpoint.defendedWinRate])
  const primaryMinimum = Math.min(0, ...primaryValues)
  const primaryMaximum = Math.max(1, ...primaryValues)
  const primarySpan = Math.max(primaryMaximum - primaryMinimum, 0.001)
  const lossMaximum = Math.max(1, ...data.map((checkpoint) => checkpoint.trainLoss))
  const plotWidth = PLOT_RIGHT - PLOT_LEFT
  const plotHeight = PLOT_BOTTOM - PLOT_TOP
  const xFor = (step: number) => PLOT_LEFT + ((step - minStep) / stepSpan) * plotWidth
  const primaryYFor = (value: number) => PLOT_BOTTOM - ((value - primaryMinimum) / primarySpan) * plotHeight
  const lossYFor = (value: number) => PLOT_BOTTOM - (value / lossMaximum) * plotHeight
  const evalPoints = data.map((checkpoint) => ({ x: xFor(checkpoint.step), y: primaryYFor(checkpoint.evalReward) }))
  const winRatePoints = data.map((checkpoint) => ({ x: xFor(checkpoint.step), y: primaryYFor(checkpoint.defendedWinRate) }))
  const lossPoints = data.map((checkpoint) => ({ x: xFor(checkpoint.step), y: lossYFor(checkpoint.trainLoss) }))
  const lastEvalPoint = evalPoints[evalPoints.length - 1]
  const evalArea = `M ${evalPoints[0].x.toFixed(2)} ${PLOT_BOTTOM} ${linePath(evalPoints).replace(/^M /, 'L ')} L ${lastEvalPoint.x.toFixed(2)} ${PLOT_BOTTOM} Z`
  const tickStride = Math.max(1, Math.ceil(data.length / 6))
  const xTicks = data.filter((_, index) => index % tickStride === 0 || index === data.length - 1)
  const titleId = `${chartId}-title`
  const descriptionId = `${chartId}-description`
  const gradientId = `${chartId}-reward-area`

  return (
    <figure className="checkpoint-trace calibration-chrome" aria-labelledby={titleId}>
      <figcaption className="checkpoint-trace__heading">
        <div>
          <span className="eyebrow">Evaluation telemetry</span>
          <strong>{data.length} calibrated {data.length === 1 ? 'state' : 'states'}</strong>
        </div>
        <ul className="checkpoint-trace__legend" aria-label="Trace legend">
          <li><i className="trace-key trace-key--reward" />Eval reward</li>
          <li><i className="trace-key trace-key--win-rate" />Defended win rate</li>
          <li><i className="trace-key trace-key--loss" />Train loss</li>
        </ul>
      </figcaption>

      <div className="checkpoint-trace__plot">
        <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
          <title id={titleId}>Checkpoint evaluation, defended win-rate, and training-loss trace</title>
          <desc id={descriptionId}>From step {stepFormatter.format(first.step)} to {stepFormatter.format(latest.step)}. Evaluation reward moves from {formatDecimal(first.evalReward)} to {formatDecimal(latest.evalReward)}, defended win rate from {Math.round(first.defendedWinRate * 100)} percent to {Math.round(latest.defendedWinRate * 100)} percent, and training loss from {formatDecimal(first.trainLoss)} to {formatDecimal(latest.trainLoss)}.</desc>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="var(--violet)" stopOpacity="0.26" />
              <stop offset="1" stopColor="var(--violet)" stopOpacity="0" />
            </linearGradient>
          </defs>

          <g className="checkpoint-trace__grid" aria-hidden="true">
            {[PLOT_TOP, PLOT_TOP + plotHeight / 2, PLOT_BOTTOM].map((y) => <line key={y} x1={PLOT_LEFT} y1={y} x2={PLOT_RIGHT} y2={y} />)}
            {xTicks.map((checkpoint) => <line key={checkpoint.step} x1={xFor(checkpoint.step)} y1={PLOT_TOP} x2={xFor(checkpoint.step)} y2={PLOT_BOTTOM} />)}
          </g>
          <g className="checkpoint-trace__axes" aria-hidden="true">
            <text x={PLOT_LEFT - 10} y={PLOT_TOP + 4} textAnchor="end">{primaryMaximum.toFixed(2)}</text>
            <text x={PLOT_LEFT - 10} y={PLOT_BOTTOM + 4} textAnchor="end">{primaryMinimum.toFixed(2)}</text>
            <text x={PLOT_RIGHT + 10} y={PLOT_TOP + 4}>loss {lossMaximum.toFixed(2)}</text>
            <text x={PLOT_RIGHT + 10} y={PLOT_BOTTOM + 4}>0.00</text>
            {xTicks.map((checkpoint) => <text key={checkpoint.step} x={xFor(checkpoint.step)} y={PLOT_BOTTOM + 28} textAnchor="middle">{stepFormatter.format(checkpoint.step)}</text>)}
            <text x={(PLOT_LEFT + PLOT_RIGHT) / 2} y={CHART_HEIGHT - 8} textAnchor="middle">training step</text>
          </g>

          <path className="checkpoint-trace__area" d={evalArea} fill={`url(#${gradientId})`} aria-hidden="true" />
          <path className="checkpoint-trace__line checkpoint-trace__line--reward" d={linePath(evalPoints)} pathLength="1" aria-hidden="true" />
          <path className="checkpoint-trace__line checkpoint-trace__line--win-rate" d={linePath(winRatePoints)} pathLength="1" aria-hidden="true" />
          <path className="checkpoint-trace__line checkpoint-trace__line--loss" d={linePath(lossPoints)} pathLength="1" aria-hidden="true" />
          <g className="checkpoint-trace__points" aria-hidden="true">
            {data.map((checkpoint, index) => (
              <g key={`${checkpoint.step}-${checkpoint.label}`}>
                <circle className="checkpoint-trace__point checkpoint-trace__point--reward" cx={evalPoints[index].x} cy={evalPoints[index].y} r="4" />
                <circle className="checkpoint-trace__point checkpoint-trace__point--win-rate" cx={winRatePoints[index].x} cy={winRatePoints[index].y} r="3.5" />
                <circle className="checkpoint-trace__point checkpoint-trace__point--loss" cx={lossPoints[index].x} cy={lossPoints[index].y} r="3" />
              </g>
            ))}
          </g>
        </svg>
      </div>

      <dl className="checkpoint-trace__readout" aria-label={`Latest checkpoint at step ${stepFormatter.format(latest.step)}`}>
        <div><dt>Eval reward</dt><dd><strong>{formatDecimal(latest.evalReward)}</strong><small>from {formatDecimal(first.evalReward)}</small></dd></div>
        <div><dt>Defended win rate</dt><dd><strong>{Math.round(latest.defendedWinRate * 100)}%</strong><small>from {Math.round(first.defendedWinRate * 100)}%</small></dd></div>
        <div><dt>Train loss</dt><dd><strong>{formatDecimal(latest.trainLoss)}</strong><small>from {formatDecimal(first.trainLoss)}</small></dd></div>
      </dl>
    </figure>
  )
}
