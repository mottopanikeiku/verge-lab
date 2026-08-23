import type { KeyboardEvent } from 'react'
import { CircleAlert, Focus, GitBranch } from 'lucide-react'
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
    graphX: candidates.length === 1 ? 440 : 72 + ((candidate.embedding.x - minimumX) / xRange) * 736,
    graphY: candidates.length === 1 ? 180 : 56 + ((candidate.embedding.y - minimumY) / yRange) * 244 + ((index % 3) - 1) * 7,
  }))
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
  const positionedCandidates = positionCandidates(candidates)
  const nodeById = Object.fromEntries(positionedCandidates.map((candidate) => [candidate.id, candidate])) as Record<string, PositionedCandidate>
  const visiblePairs = pairs.filter((pair) => nodeById[pair.chosenId] && nodeById[pair.rejectedId])
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedCandidateId)

  if (candidates.length === 0) {
    return <div className="empty-state"><CircleAlert aria-hidden="true" /><h3>No candidates cross this aperture</h3><p>Clear the search or widen the domain and verdict filters.</p></div>
  }

  return (
    <div className="topology-stack">
      <figure className="order-graph" aria-labelledby="order-graph-title">
        <figcaption><span><GitBranch size={15} aria-hidden="true" /><strong id="order-graph-title">Defensible partial order</strong></span><span>{visiblePairs.length} visible edges</span></figcaption>
        <div className="graph-key" aria-label="Graph legend"><span><i className="key-line key-line--defended" />Defended preference</span><span><i className="key-line key-line--ambiguous" />Ambiguous comparison</span><span><i className="key-node" />Candidate</span></div>
        <svg className="order-graph-svg" viewBox="0 0 880 360" role="group" aria-label="Candidate partial-order graph. Tab to nodes; arrow keys move between candidates." preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arrow-defended" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" className="marker-defended" /></marker>
            <marker id="arrow-ambiguous" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" className="marker-ambiguous" /></marker>
          </defs>
          <g aria-label="Preference edges">
            {visiblePairs.map((pair) => {
              const chosen = nodeById[pair.chosenId]
              const rejected = nodeById[pair.rejectedId]
              return <line key={pair.id} x1={rejected.graphX} y1={rejected.graphY} x2={chosen.graphX} y2={chosen.graphY} className={`graph-edge graph-edge--${pair.verdict}`} markerEnd={`url(#arrow-${pair.verdict})`}><title>{`${pair.verdict} edge, ${formatPercent(pair.confidence)} confidence. ${pair.reason}`}</title></line>
            })}
          </g>
          <g aria-label="Candidates">
            {positionedCandidates.map((candidate) => {
              const label = candidateLabel(run, candidate)
              const prompt = promptFor(run, candidate.promptId)
              const selected = candidate.id === selectedCandidateId
              return (
                <g key={candidate.id} id={`graph-node-${candidate.id}`} className={`graph-node${selected ? ' graph-node--selected' : ''}`} transform={`translate(${candidate.graphX} ${candidate.graphY})`} role="button" tabIndex={0} aria-pressed={selected} aria-label={`${label}, ${prompt?.domain ?? 'unknown domain'}, ${candidate.tokens} tokens. Select candidate.`} onClick={() => onSelectCandidate(candidate.id)} onKeyDown={(event) => moveGraphFocus(event, candidate.id, candidates, onSelectCandidate)}>
                  <title>{`${candidate.id}: ${candidate.output.slice(0, 120)}`}</title><circle r={selected ? 24 : 19} /><text textAnchor="middle" dominantBaseline="central">{label}</text><text className="graph-node-domain" textAnchor="middle" y="37">{prompt?.domain.slice(0, 12)}</text>
                </g>
              )
            })}
          </g>
        </svg>
        <p className="graph-note">Position uses the artifact embedding x/y coordinates. Arrows point from rejected to chosen; no scalar rank is inferred.</p>
      </figure>

      {selectedCandidate ? (
        <section className="candidate-inspector" aria-labelledby="candidate-inspector-title">
          <header><span className="eyebrow"><Focus size={13} aria-hidden="true" />Selected specimen</span><h3 id="candidate-inspector-title">{candidateLabel(run, selectedCandidate)} · {shortId(selectedCandidate.id)}</h3><p>{promptFor(run, selectedCandidate.promptId)?.domain}</p></header>
          <blockquote>{selectedCandidate.output}</blockquote>
          <dl className="specimen-facts"><div><dt>Tokens</dt><dd>{selectedCandidate.tokens}</dd></div><div><dt>Latency</dt><dd>{selectedCandidate.latencyMs} ms</dd></div><div><dt>Embed z</dt><dd>{formatNumber(selectedCandidate.embedding.z)}</dd></div></dl>
          <div className="score-evidence-list">
            {run.objectives.map((objective) => {
              const score = selectedCandidate.scores[objective.id]
              if (!score) return null
              return <details key={objective.id}><summary><span>{objective.label}</span><strong>{formatNumber(score.value)} <small>± {formatPercent(1 - score.confidence)}</small></strong></summary><p>{score.evidence}</p></details>
            })}
          </div>
        </section>
      ) : null}

      <section className="candidate-table-section" aria-labelledby="candidate-table-title">
        <header className="section-heading section-heading--compact"><div><span className="eyebrow">Accessible register</span><h3 id="candidate-table-title">Candidate evidence table</h3></div><span>{candidates.length} rows</span></header>
        <div className="table-wrap"><table className="data-table candidate-table"><thead><tr><th scope="col">Candidate</th><th scope="col">Domain</th>{run.objectives.map((objective) => <th scope="col" key={objective.id}>{objective.label}</th>)}<th scope="col">Tokens</th><th scope="col">Latency</th></tr></thead><tbody>
          {candidates.map((candidate) => {
            const selected = candidate.id === selectedCandidateId
            return <tr key={candidate.id} className={selected ? 'is-selected' : undefined}><th scope="row" data-label="Candidate"><button type="button" onClick={() => onSelectCandidate(candidate.id)} aria-pressed={selected}><span>{candidateLabel(run, candidate)}</span>{shortId(candidate.id)}</button></th><td data-label="Domain">{promptFor(run, candidate.promptId)?.domain}</td>{run.objectives.map((objective) => { const score = candidate.scores[objective.id]; return <td data-label={objective.label} key={objective.id}>{score ? `${formatNumber(score.value)} / ${formatPercent(score.confidence)}` : '—'}</td> })}<td data-label="Tokens">{candidate.tokens}</td><td data-label="Latency">{candidate.latencyMs} ms</td></tr>
          })}
        </tbody></table></div>
      </section>
    </div>
  )
}
