import { useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent } from 'react'
import { CircleAlert, Focus, GitBranch, Minus, Plus, RotateCcw } from 'lucide-react'
import { ConfidenceDial } from './ConfidenceDial'
import { candidateLabel, formatNumber, formatPercent, promptFor, shortId } from '../lib'
import type { Candidate, Pair, RunArtifact } from '../types'

type PartialOrderGraphProps = {
  run: RunArtifact
  candidates: Candidate[]
  pairs: Pair[]
  selectedCandidateId: string | null
  onSelectCandidate: (candidateId: string) => void
}

type PositionedCandidate = Candidate & { graphX: number; graphY: number }
type GraphViewport = { x: number; y: number; scale: number }
type PanOrigin = { pointerId: number; clientX: number; clientY: number; x: number; y: number; width: number; height: number }

const GRAPH_WIDTH = 880
const GRAPH_HEIGHT = 360
const MIN_ZOOM = 1
const MAX_ZOOM = 3
const ZOOM_STEP = 1.25
const INITIAL_VIEWPORT: GraphViewport = { x: 0, y: 0, scale: MIN_ZOOM }

function clampNumber(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum)
}

// Short node caption, cut at a word boundary when the first word fits.
function nodeDomain(domain: string, maximum = 12) {
  if (domain.length <= maximum) return domain
  const cut = domain.slice(0, maximum + 1).replace(/\s+\S*$/, '')
  return cut.length > maximum ? cut.slice(0, maximum) : cut
}

function positionCandidates(candidates: Candidate[]): PositionedCandidate[] {
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
    graphX: candidates.length === 1 ? GRAPH_WIDTH / 2 : 72 + ((candidate.embedding.x - minimumX) / xRange) * 736,
    graphY: candidates.length === 1 ? GRAPH_HEIGHT / 2 : 56 + ((candidate.embedding.y - minimumY) / yRange) * 244 + ((index % 3) - 1) * 7,
  }))
}

function relaxCollisions(positioned: PositionedCandidate[], iterations = 24): PositionedCandidate[] {
  const relaxed = positioned.map((candidate) => ({ ...candidate }))
  const targets = positioned.map((candidate) => ({ x: candidate.graphX, y: candidate.graphY }))
  const minimumDistance = 62
  const maximumDisplacement = 36
  const maximumStep = 4

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let moved = false

    for (let firstIndex = 0; firstIndex < relaxed.length; firstIndex += 1) {
      for (let secondIndex = firstIndex + 1; secondIndex < relaxed.length; secondIndex += 1) {
        const first = relaxed[firstIndex]
        const second = relaxed[secondIndex]
        let deltaX = second.graphX - first.graphX
        let deltaY = second.graphY - first.graphY
        let distance = Math.hypot(deltaX, deltaY)

        if (distance >= minimumDistance) continue
        if (distance < 0.001) {
          const angle = ((firstIndex + 1) * (secondIndex + 1) * 2.399963) % (Math.PI * 2)
          deltaX = Math.cos(angle)
          deltaY = Math.sin(angle)
          distance = 1
        }

        const shift = Math.min((minimumDistance - distance) / 2, maximumStep)
        const unitX = deltaX / distance
        const unitY = deltaY / distance
        first.graphX -= unitX * shift
        first.graphY -= unitY * shift
        second.graphX += unitX * shift
        second.graphY += unitY * shift
        moved = true
      }
    }

    relaxed.forEach((candidate, index) => {
      const offsetX = candidate.graphX - targets[index].x
      const offsetY = candidate.graphY - targets[index].y
      const displacement = Math.hypot(offsetX, offsetY)
      if (displacement > maximumDisplacement) {
        const correction = maximumDisplacement / displacement
        candidate.graphX = targets[index].x + offsetX * correction
        candidate.graphY = targets[index].y + offsetY * correction
      }
      candidate.graphX = clampNumber(candidate.graphX, 42, GRAPH_WIDTH - 42)
      candidate.graphY = clampNumber(candidate.graphY, 46, GRAPH_HEIGHT - 46)
    })

    if (!moved) break
  }

  return relaxed
}

function constrainViewport(viewport: GraphViewport): GraphViewport {
  const scale = clampNumber(viewport.scale, MIN_ZOOM, MAX_ZOOM)
  const width = GRAPH_WIDTH / scale
  const height = GRAPH_HEIGHT / scale
  return {
    x: clampNumber(viewport.x, 0, GRAPH_WIDTH - width),
    y: clampNumber(viewport.y, 0, GRAPH_HEIGHT - height),
    scale,
  }
}

function moveGraphFocus(event: KeyboardEvent<SVGGElement>, candidateId: string, candidates: Candidate[], onSelectCandidate: (candidateId: string) => void) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    onSelectCandidate(candidateId)
    return
  }
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return
  event.preventDefault()
  const currentIndex = candidates.findIndex((candidate) => candidate.id === candidateId)
  const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1
  const nextCandidate = candidates[(currentIndex + direction + candidates.length) % candidates.length]
  onSelectCandidate(nextCandidate.id)
  document.getElementById(`graph-node-${nextCandidate.id}`)?.focus()
}

export function PartialOrderGraph({ run, candidates, pairs, selectedCandidateId, onSelectCandidate }: PartialOrderGraphProps) {
  const positionedCandidates = useMemo(
    () => relaxCollisions(positionCandidates(candidates)),
    [candidates],
  )
  const nodeById = useMemo(
    () => Object.fromEntries(positionedCandidates.map((candidate) => [candidate.id, candidate])) as Record<string, PositionedCandidate>,
    [positionedCandidates],
  )
  const visiblePairs = useMemo(
    () => pairs.filter((pair) => nodeById[pair.chosenId] && nodeById[pair.rejectedId]),
    [nodeById, pairs],
  )
  const [viewport, setViewport] = useState<GraphViewport>(INITIAL_VIEWPORT)
  const panOrigin = useRef<PanOrigin | null>(null)
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedCandidateId)
  const viewWidth = GRAPH_WIDTH / viewport.scale
  const viewHeight = GRAPH_HEIGHT / viewport.scale

  const zoomBy = (factor: number, anchorX = 0.5, anchorY = 0.5) => {
    setViewport((current) => {
      const nextScale = clampNumber(current.scale * factor, MIN_ZOOM, MAX_ZOOM)
      const currentWidth = GRAPH_WIDTH / current.scale
      const currentHeight = GRAPH_HEIGHT / current.scale
      const nextWidth = GRAPH_WIDTH / nextScale
      const nextHeight = GRAPH_HEIGHT / nextScale
      return constrainViewport({
        x: current.x + (currentWidth - nextWidth) * anchorX,
        y: current.y + (currentHeight - nextHeight) * anchorY,
        scale: nextScale,
      })
    })
  }

  const beginPan = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'touch' || event.button !== 0 || viewport.scale <= MIN_ZOOM || (event.target as Element).closest('.graph-node')) return
    event.currentTarget.setPointerCapture(event.pointerId)
    panOrigin.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      x: viewport.x,
      y: viewport.y,
      width: viewWidth,
      height: viewHeight,
    }
  }

  const continuePan = (event: PointerEvent<HTMLDivElement>) => {
    const origin = panOrigin.current
    if (!origin || origin.pointerId !== event.pointerId) return
    event.preventDefault()
    const bounds = event.currentTarget.getBoundingClientRect()
    setViewport((current) => constrainViewport({
      x: origin.x - ((event.clientX - origin.clientX) / Math.max(bounds.width, 1)) * origin.width,
      y: origin.y - ((event.clientY - origin.clientY) / Math.max(bounds.height, 1)) * origin.height,
      scale: current.scale,
    }))
  }

  const endPan = (event: PointerEvent<HTMLDivElement>) => {
    if (panOrigin.current?.pointerId !== event.pointerId) return
    panOrigin.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }


  if (candidates.length === 0) {
    return <div className="empty-state"><CircleAlert aria-hidden="true" /><h3>No example answers match</h3><p>Clear the search or widen the domain and verdict filters.</p></div>
  }

  return (
    <div className="topology-stack">
      <figure className="order-graph calibration-chrome" data-armed={selectedCandidate ? 'true' : 'false'} aria-labelledby="order-graph-title">
        <figcaption><span><GitBranch size={15} aria-hidden="true" /><strong id="order-graph-title">Illustrative partial order</strong></span><span>{visiblePairs.length} computed example edges</span></figcaption>
        <div className="graph-key" aria-label="Example graph legend"><span><i className="key-line key-line--defended" />Arrow: selected answer</span><span><i className="key-line key-line--ambiguous" />Dashed: abstain</span><span><i className="key-node" />Example answer</span></div>
        <div className="graph-controls" role="group" aria-label="Graph viewport controls">
          <button type="button" onClick={() => zoomBy(1 / ZOOM_STEP)} disabled={viewport.scale <= MIN_ZOOM}><Minus size={14} aria-hidden="true" /><span>Zoom out</span></button>
          <button type="button" onClick={() => setViewport(INITIAL_VIEWPORT)} disabled={viewport.scale === MIN_ZOOM && viewport.x === 0 && viewport.y === 0}><RotateCcw size={14} aria-hidden="true" /><span>Reset view</span></button>
          <button type="button" onClick={() => zoomBy(ZOOM_STEP)} disabled={viewport.scale >= MAX_ZOOM}><Plus size={14} aria-hidden="true" /><span>Zoom in</span></button>
          <output aria-label="Current graph zoom">{Math.round(viewport.scale * 100)}%</output>
        </div>
        <div className="graph-viewport" onPointerDown={beginPan} onPointerMove={continuePan} onPointerUp={endPan} onPointerCancel={endPan}>
          <svg className="order-graph-svg" viewBox={`${viewport.x} ${viewport.y} ${viewWidth} ${viewHeight}`} role="group" aria-label="Illustrative candidate partial-order graph computed from authored example scores. Coordinates are authored, not model embeddings. Tab to nodes; arrow keys move between candidates." preserveAspectRatio="xMidYMid meet">
            <defs>
              <marker id="arrow-defended" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" className="marker-defended" /></marker>
            </defs>
            <g aria-label="Computed preference edges on illustrative inputs">
              {visiblePairs.map((pair) => {
                const chosen = nodeById[pair.chosenId]
                const rejected = nodeById[pair.rejectedId]
                const markerEnd = pair.verdict === 'defended' ? 'url(#arrow-defended)' : undefined
                const edgeLabel = pair.verdict === 'defended' ? 'preference edge' : 'undirected comparison'
                return <line key={pair.id} x1={rejected.graphX} y1={rejected.graphY} x2={chosen.graphX} y2={chosen.graphY} className={`graph-edge graph-edge--${pair.verdict}`} markerEnd={markerEnd}><title>{`Computed example ${pair.verdict} ${edgeLabel}, ${formatPercent(pair.confidence)} illustrative confidence, not a measurement. ${pair.reason}`}</title></line>
              })}
            </g>
            <g aria-label="Illustrative candidates">
              {positionedCandidates.map((candidate) => {
                const label = candidateLabel(run, candidate)
                const prompt = promptFor(run, candidate.promptId)
                const selected = candidate.id === selectedCandidateId
                return (
                  <g key={candidate.id} id={`graph-node-${candidate.id}`} className={`graph-node${selected ? ' graph-node--selected' : ''}`} transform={`translate(${candidate.graphX} ${candidate.graphY})`} role="button" tabIndex={0} aria-pressed={selected} aria-label={`${label} ${prompt?.domain ?? 'unknown domain'}, example answer ${shortId(candidate.id)}. Select answer.`} onClick={() => onSelectCandidate(candidate.id)} onKeyDown={(event) => moveGraphFocus(event, candidate.id, candidates, onSelectCandidate)}>
                    {/* The trailing space keeps the label and domain separate words for speech-input name matching; SVG strips it visually. */}
                    <title>{`Illustrative candidate ${candidate.id}: ${candidate.output.slice(0, 120)}`}</title><circle r={selected ? 24 : 19} /><text textAnchor="middle" dominantBaseline="central">{`${label} `}</text><text className="graph-node-domain" textAnchor="middle" y="37">{prompt ? nodeDomain(prompt.domain) : null}</text>
                  </g>
                )
              })}
            </g>
          </svg>
        </div>
        <p className="graph-note">Arrows point to the selected answer, not to a global rank. Dashed lines mean no winner was selected. Positions are authored coordinates, not measured model embeddings. Use zoom buttons, then drag with a mouse to pan. Tab to an answer; arrow keys move between answers; Enter selects it. The table below offers the same selection controls.</p>
      </figure>

      {selectedCandidate ? (
        <section className="candidate-inspector" aria-labelledby="candidate-inspector-title">
          <header><span className="eyebrow"><Focus size={13} aria-hidden="true" />Selected illustrative candidate</span><h3 id="candidate-inspector-title">{candidateLabel(run, selectedCandidate)} · {shortId(selectedCandidate.id)}</h3><p>{promptFor(run, selectedCandidate.promptId)?.domain}</p></header>
          <blockquote>{selectedCandidate.output}</blockquote>
          <dl className="specimen-facts"><div><dt>Illustrative tokens</dt><dd>{selectedCandidate.tokens}</dd></div><div><dt>Illustrative latency</dt><dd>{selectedCandidate.latencyMs} ms</dd></div><div><dt>Illustrative coordinate z</dt><dd>{formatNumber(selectedCandidate.embedding.z)}</dd></div></dl>
          <div className="score-evidence-list">
            {run.objectives.map((objective) => {
              const score = selectedCandidate.scores[objective.id]
              if (!score) return null
              return <details key={objective.id}><summary><span>Illustrative {objective.label} score</span><span className="score-evidence-value"><strong>{formatNumber(score.value)}</strong><ConfidenceDial value={score.confidence} tone="neutral" label={`${objective.label} confidence`} size="inline" /></span></summary><p>Authored example evidence: {score.evidence}</p></details>
            })}
          </div>
        </section>
      ) : null}

      <details className="candidate-register-disclosure" open>
        <summary><span>Example answer table</span><small>{candidates.length} rows · score / confidence</small></summary>
        <section className="candidate-table-section" aria-labelledby="candidate-table-title">
          <header className="section-heading section-heading--compact"><div><span className="eyebrow">Illustrative accessible register</span><h3 id="candidate-table-title">Example candidate scores and confidences</h3></div><span>{candidates.length} example rows</span></header>
          <div className="table-wrap" tabIndex={0} role="region" aria-label="Example scores table; scroll horizontally for all objectives"><table className="data-table candidate-table"><caption>All scores, confidences, token counts and latencies are illustrative authored inputs, not model measurements. Objective cells show score / confidence.</caption><thead><tr><th scope="col">Example candidate</th><th scope="col">Example domain</th>{run.objectives.map((objective) => <th scope="col" key={objective.id}>Illustrative {objective.label} score / confidence</th>)}<th scope="col">Illustrative tokens</th><th scope="col">Illustrative latency</th></tr></thead><tbody>
            {candidates.map((candidate) => {
              const selected = candidate.id === selectedCandidateId
              return <tr key={candidate.id} className={selected ? 'is-selected' : undefined}><th scope="row" data-label="Example candidate"><button type="button" onClick={() => onSelectCandidate(candidate.id)} aria-pressed={selected}><span>{candidateLabel(run, candidate)}</span>{shortId(candidate.id)}</button></th><td data-label="Example domain">{promptFor(run, candidate.promptId)?.domain}</td>{run.objectives.map((objective) => { const score = candidate.scores[objective.id]; return <td data-label={`Illustrative ${objective.label} score / confidence`} key={objective.id}>{score ? `${formatNumber(score.value)} / ${formatPercent(score.confidence)}` : '—'}</td> })}<td data-label="Illustrative tokens">{candidate.tokens}</td><td data-label="Illustrative latency">{candidate.latencyMs} ms</td></tr>
            })}
          </tbody></table></div>
        </section>
      </details>
    </div>
  )
}
