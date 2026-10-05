import { useMemo, useState } from 'react'
import type React from 'react'
import { ArrowLeft, ArrowRight, Beaker, CheckCircle2, CircleAlert, Clock3, Microscope, ShieldCheck } from 'lucide-react'
import { CheckpointTrace } from '../components/CheckpointTrace'
import { ConfidenceDial } from '../components/ConfidenceDial'
import { HeroEvidenceField } from '../components/HeroEvidenceField'
import { EvidenceFilters } from '../components/EvidenceFilters'
import { PartialOrderGraph } from '../components/PartialOrderGraph'
import { ThreeInstrument } from '../components/ThreeInstrument'
import { candidateFor, candidateLabel, clamp, filteredCandidates, filteredPairs, formatCompactNumber, formatDate, formatNumber, formatPercent, promptFor, shortId } from '../lib'
import type { LabFilters, Pair, RunArtifact, ViewId } from '../types'

type SharedViewProps = { run: RunArtifact }

type OverviewProps = SharedViewProps & { onNavigate: (view: ViewId) => void }

export function Overview({ run, onNavigate }: OverviewProps) {
  const totalPairs = run.summary.defendedPairCount + run.summary.ambiguousPairCount
  const defendedRate = totalPairs === 0 ? 0 : run.summary.defendedPairCount / totalPairs
  const ambiguityRate = 1 - defendedRate
  const averageConfidence = run.pairs.length === 0 ? 0 : run.pairs.reduce((sum, pair) => sum + pair.confidence, 0) / run.pairs.length
  const objectiveShiftRatios = run.objectives.map((objective) => (
    Math.abs(objective.delta) / (Math.abs(objective.mean) || 1)
  ))
  const largestObjectiveShift = Math.max(...objectiveShiftRatios, Number.EPSILON)

  return (
    <div className="view-stack">
      <section className="bench-hero">
        <header className="view-hero overview-hero">
          <div>
            <span className="eyebrow">Illustrative demo overview / {shortId(run.id)}</span>
            <h1>Example scores, explicit tradeoffs.</h1>
            <p>Explore a partial order computed from authored example scores. Defended edges satisfy the shared objectives; these examples are not model measurements.</p>
            <div className="hero-readout" role="list" aria-label="Illustrative example summary, computed from authored inputs">
              <span role="listitem"><strong>{run.summary.candidateCount}</strong> example candidates</span>
              <span role="listitem"><strong>{run.summary.defendedPairCount}</strong> example defended edges</span>
              <span role="listitem"><strong>{run.summary.ambiguousPairCount}</strong> example ambiguous pairs</span>
              <span role="listitem"><strong>{formatPercent(averageConfidence)}</strong> mean illustrative confidence</span>
            </div>
          </div>
          <dl className="run-stamp"><div><dt>Demo status</dt><dd><span className="status-dot" />{run.status}</dd></div><div><dt>Example model</dt><dd>{run.model.base}</dd></div><div><dt>Demo date</dt><dd>{formatDate(run.createdAt)}</dd></div></dl>
        </header>
        <ThreeInstrument kind="constellation" title="Illustrative candidate field" description={`${run.summary.candidateCount} example candidates; motion maps the computed ${formatPercent(ambiguityRate)} example ambiguity share and brightness maps mean illustrative confidence.`} metrics={{ speed: 0.25 + ambiguityRate, density: 0.55 + run.summary.candidateCount / 20, opacity: 0.68 + defendedRate * 0.2, hue: 0.18, brightness: 0.65 + averageConfidence * 0.5 }} />
        <HeroEvidenceField run={run} />
      </section>

      <section className="ledger" aria-labelledby="ledger-title">
        <header className="section-heading"><div><span className="eyebrow">Illustrative demo ledger</span><h2 id="ledger-title">{run.name}</h2></div><span>schema v{run.schemaVersion}</span></header>
        <div className="ledger-grid">
          <article className="ledger-lead"><span>Computed example defended yield</span><strong>{formatPercent(defendedRate)}</strong><p>{run.summary.defendedPairCount} of {totalPairs} example comparisons satisfy the confidence-bound Pareto rule on authored inputs, not measured outcomes.</p><button type="button" className="text-action" onClick={() => onNavigate('topology')}>Trace the example order <ArrowRight size={15} aria-hidden="true" /></button></article>
          <dl className="ledger-metrics"><div><dt>Example prompts</dt><dd>{run.summary.promptCount}</dd></div><div><dt>Example candidates</dt><dd>{run.summary.candidateCount}</dd></div><div><dt>Example ambiguous pairs</dt><dd className="coral-text">{run.summary.ambiguousPairCount}</dd></div><div><dt>Computed example flips</dt><dd>{run.summary.mutationFlipCount}</dd></div><div><dt>Illustrative GPU estimate</dt><dd>{formatNumber(run.summary.estimatedGpuMinutes, 1)} min</dd></div><div><dt>Illustrative cost estimate</dt><dd>${formatNumber(run.summary.estimatedCostUsd)}</dd></div></dl>
        </div>
      </section>

      <section className="objective-ledger" aria-labelledby="objective-title">
        <header className="section-heading"><div><span className="eyebrow">Illustrative reward specification</span><h2 id="objective-title">Example objective scores</h2></div><span>{run.objectives.length} example dimensions</span></header>
        <div className="objective-rows">
          {run.objectives.map((objective, index) => {
            const shiftRatio = objectiveShiftRatios[index]
            const magnitude = `${(shiftRatio / largestObjectiveShift) * 100}%`
            const direction = objective.delta > 0 ? 'positive' : objective.delta < 0 ? 'negative' : 'flat'
            return (
              <article key={objective.id} className={`objective-row objective-row--${index % 4}`} data-magnitude={shiftRatio.toFixed(4)} data-delta-direction={direction} style={{ '--objective-magnitude': magnitude } as React.CSSProperties}>
                <span className="objective-index">0{index + 1}</span>
                <div><h3>{objective.label}</h3><p>{objective.description}</p><span className="objective-magnitude" aria-label={`Illustrative authored delta magnitude is ${formatPercent(shiftRatio)} of the computed example score mean, not an observed change`}><i aria-hidden="true" /></span></div>
                <dl><div><dt>Direction</dt><dd>{objective.direction}</dd></div><div><dt>Example mean</dt><dd>{formatNumber(objective.mean)}</dd></div><div><dt>Illustrative Δ</dt><dd>{objective.delta > 0 ? '+' : ''}{formatNumber(objective.delta)}</dd></div></dl>
              </article>
            )
          })}
        </div>
      </section>

      <section className="triage-strip" aria-labelledby="triage-title">
        <header><span className="eyebrow">Next examination</span><h2 id="triage-title">Follow the uncertain edge.</h2></header>
        <div><button type="button" onClick={() => onNavigate('pairs')}><span><CircleAlert size={18} aria-hidden="true" />Pair lab</span><strong>{run.summary.ambiguousPairCount} example comparisons to inspect</strong><ArrowRight aria-hidden="true" /></button><button type="button" onClick={() => onNavigate('runbook')}><span><Clock3 size={18} aria-hidden="true" />Runbook</span><strong>{run.checkpoints.length} illustrative checkpoints</strong><ArrowRight aria-hidden="true" /></button></div>
      </section>
    </div>
  )
}

type FilteredViewProps = SharedViewProps & { filters: LabFilters; onFiltersChange: (filters: LabFilters) => void }

export function RewardTopology({ run, filters, onFiltersChange }: FilteredViewProps) {
  const visiblePairs = useMemo(() => filteredPairs(run, filters), [run, filters])
  const visibleCandidates = useMemo(() => filteredCandidates(run, filters), [run, filters])
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(visibleCandidates[0]?.id ?? null)
  const effectiveSelectedCandidateId = visibleCandidates.some((candidate) => candidate.id === selectedCandidateId)
    ? selectedCandidateId
    : visibleCandidates[0]?.id ?? null

  const defended = visiblePairs.filter((pair) => pair.verdict === 'defended').length
  const defendedRate = visiblePairs.length === 0 ? 0 : defended / visiblePairs.length
  const meanConfidence = visiblePairs.length === 0 ? 0 : visiblePairs.reduce((sum, pair) => sum + pair.confidence, 0) / visiblePairs.length

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Illustrative reward topology</span><h1>A ranking is too certain.</h1><p>This partial order is computed from authored example scores, not model evaluations. Defended arrows coexist with unresolved tradeoffs; filtering changes which examples are visible.</p></div><div className="calibration-chrome"><ThreeInstrument compact kind="topology" title="Illustrative confidence texture" description={`Hue maps the computed ${formatPercent(defendedRate)} example defended edge share; brightness maps ${formatPercent(meanConfidence)} mean illustrative confidence.`} metrics={{ speed: 0, density: 0, opacity: 1, hue: -0.32 + defendedRate * 0.68, brightness: 0.62 + meanConfidence * 0.58 }} /></div></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visibleCandidates.length} resultLabel={`candidates · ${visiblePairs.length} edges`} />
      <PartialOrderGraph run={run} candidates={visibleCandidates} pairs={visiblePairs} selectedCandidateId={effectiveSelectedCandidateId} onSelectCandidate={setSelectedCandidateId} />
    </div>
  )
}

function PairChoice({ run, pair, role }: { run: RunArtifact; pair: Pair; role: 'chosen' | 'rejected' }) {
  const candidate = candidateFor(run, role === 'chosen' ? pair.chosenId : pair.rejectedId)
  if (!candidate) return <div className="empty-inline">Candidate absent from artifact.</div>
  const neutral = pair.verdict === 'ambiguous'
  const sideLabel = neutral
    ? role === 'chosen' ? 'Candidate A' : 'Candidate B'
    : role === 'chosen' ? 'Chosen' : 'Rejected'
  const tone = neutral ? 'neutral' : role
  return (
    <article className={`pair-choice pair-choice--${tone}`}>
      <header><span>Example {sideLabel}</span><strong>{candidateLabel(run, candidate)} · {shortId(candidate.id)}</strong></header>
      <blockquote>{candidate.output}</blockquote>
      <div className="pair-score-stack">{run.objectives.map((objective) => {
        const score = candidate.scores[objective.id]
        return <div key={objective.id}><span>Illustrative {objective.label} score</span><strong>{score ? formatNumber(score.value) : '—'}</strong>{score ? <ConfidenceDial value={score.confidence} tone="neutral" label={`${sideLabel} ${objective.label} confidence`} size="inline" /> : <small>missing example score</small>}{score ? <p>Authored example evidence: {score.evidence}</p> : null}</div>
      })}</div>
    </article>
  )
}

export function PairLab({ run, filters, onFiltersChange }: FilteredViewProps) {
  const visiblePairs = useMemo(() => filteredPairs(run, filters), [run, filters])
  const [selectedPairId, setSelectedPairId] = useState<string | null>(null)
  const selectedPairIsVisible = visiblePairs.some((pair) => pair.id === selectedPairId)
  const selectedPair = visiblePairs.find((pair) => pair.id === selectedPairId) ?? visiblePairs[0]
  const relevantMutations = selectedPair ? run.mutations.filter((mutation) => mutation.sourceCandidateId === selectedPair.chosenId || mutation.sourceCandidateId === selectedPair.rejectedId) : []
  const drillDownOpen = Boolean(selectedPairId && selectedPairIsVisible)

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Illustrative pair lab</span><h1>Inspect the edge, not the score.</h1><p>Compare authored example evidence objective by objective. Verdicts and margins are deterministic calculations on these example inputs; ambiguity is a valid result, not a missing label.</p></div><dl className="verdict-counts"><div><dt>Example defended</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Example ambiguous</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visiblePairs.length} resultLabel="comparisons" />
      {selectedPair ? (
        <div className={`pair-workbench ${drillDownOpen ? 'pair-workbench--detail' : 'pair-workbench--register'}`}>
          <aside className="pair-index pair-index-panel" aria-label="Filtered illustrative comparison pairs">
            <span className="eyebrow">Illustrative pair register</span>
            {visiblePairs.map((pair) => {
              const prompt = promptFor(run, pair.promptId)
              const selected = pair.id === selectedPair.id
              return (
                <button type="button" key={pair.id} className={selected ? 'is-selected' : undefined} aria-pressed={selected} onClick={() => setSelectedPairId(pair.id)}>
                  <span><i className={`verdict-mark verdict-mark--${pair.verdict}`} />{shortId(pair.id)}</span>
                  <small>{prompt?.domain}</small>
                  <ConfidenceDial value={pair.confidence} tone={pair.verdict} label={`${shortId(pair.id)} confidence`} size="inline" />
                </button>
              )
            })}
          </aside>
          <section className="pair-evidence pair-evidence-panel pair-evidence--active calibration-chrome" data-armed="true" aria-labelledby="pair-evidence-title">
            <button type="button" className="pair-back" onClick={() => setSelectedPairId(null)}><ArrowLeft size={15} aria-hidden="true" />Back to pair register</button>
            <header className="pair-evidence-header">
              <div><span className={`verdict-label verdict-label--${selectedPair.verdict}`}>{selectedPair.verdict === 'defended' ? <CheckCircle2 size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{selectedPair.verdict}</span><h2 id="pair-evidence-title">{shortId(selectedPair.id)}</h2><p>{promptFor(run, selectedPair.promptId)?.text}</p></div>
              <dl><div><dt>Illustrative confidence</dt><dd><ConfidenceDial value={selectedPair.confidence} tone={selectedPair.verdict} label={`${shortId(selectedPair.id)} verdict confidence`} size="panel" /></dd></div><div><dt>Example domain</dt><dd>{promptFor(run, selectedPair.promptId)?.domain}</dd></div></dl>
            </header>
            <section className="specification-band" aria-labelledby="spec-title">
              <h3 id="spec-title">Example prompt-level reward specification</h3>
              <div>
                <section aria-labelledby="rubrics-title"><h4 id="rubrics-title">Rubrics</h4><ul>{promptFor(run, selectedPair.promptId)?.specification.rubrics.map((rubric) => <li key={rubric}>{rubric}</li>)}</ul></section>
                <section aria-labelledby="constraints-title"><h4 id="constraints-title">Constraints</h4><ul>{promptFor(run, selectedPair.promptId)?.specification.constraints.map((constraint) => <li key={constraint}>{constraint}</li>)}</ul></section>
              </div>
            </section>
            <div className="pair-comparison"><PairChoice run={run} pair={selectedPair} role="chosen" /><PairChoice run={run} pair={selectedPair} role="rejected" /></div>
            <section className="verdict-reason"><span className="eyebrow">Computed example verdict and margins</span><h3>{selectedPair.reason}</h3><div className="margin-strip">{run.objectives.map((objective) => <div key={objective.id}><span>{objective.label}</span><strong>{selectedPair.margins[objective.id] === undefined ? 'not shared' : `${selectedPair.margins[objective.id] > 0 ? '+' : ''}${formatNumber(selectedPair.margins[objective.id])}`}</strong></div>)}</div></section>
            <section className="mutation-evidence"><header><span className="eyebrow">Illustrative mutation comparisons</span><strong>{relevantMutations.length} linked examples</strong></header><p>Verdict flips and deltas are computed from authored before/after scores, not measured rescoring or model bias tests.</p>{relevantMutations.length === 0 ? <p className="empty-inline">No example mutation targets either candidate in this pair.</p> : relevantMutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} · computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</section>
          </section>
        </div>
      ) : <div className="empty-state"><CircleAlert aria-hidden="true" /><h3>No comparisons match</h3><p>Clear the candidate search or widen the domain and verdict filters.</p></div>}
    </div>
  )
}

export function Runbook({ run }: SharedViewProps) {
  const lastCheckpoint = run.checkpoints.at(-1)
  const firstCheckpoint = run.checkpoints[0]
  const rewardGain = lastCheckpoint && firstCheckpoint ? lastCheckpoint.evalReward - firstCheckpoint.evalReward : 0
  const maxStep = Math.max(...run.checkpoints.map((checkpoint) => checkpoint.step), 1)

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Runbook / illustrative demo</span><h1>An authored example, not a run.</h1><p>Illustrative checkpoints, mutation comparisons and model metadata show the artifact format. They are not a training record, a trained adapter or GPU diagnostics.</p></div><div className="runbook-seal"><ShieldCheck aria-hidden="true" /><span>Example artifact status</span><strong>{run.status}</strong><small>{run.id}</small></div></header>
      <section className="runbook-grid">
        <div className="calibration-chrome"><ThreeInstrument kind="diagnostics" title="Illustrative checkpoint animation" description={`${run.checkpoints.length} authored example checkpoints; speed maps illustrative GPU-minute values, density maps illustrative win-rate values and brightness maps illustrative reward values. None is observed activity.`} metrics={{ speed: clamp((lastCheckpoint?.gpuMinutes ?? 0) / Math.max(run.summary.estimatedGpuMinutes, 1), 0.15, 1.3), density: 0.45 + (lastCheckpoint?.defendedWinRate ?? 0), opacity: 0.78, hue: 0.36 + clamp(rewardGain, -0.3, 0.3), brightness: 0.72 + clamp(lastCheckpoint?.evalReward ?? 0, 0, 1) * 0.48 }} /></div>
        <section className="manifest" aria-labelledby="manifest-title"><header><span className="eyebrow">Illustrative model metadata</span><h2 id="manifest-title">Example adapter description</h2></header><dl><div><dt>Example base</dt><dd>{run.model.base}</dd></div><div><dt>Example method</dt><dd>{run.model.method}</dd></div><div><dt>Example adapter name</dt><dd>{run.model.adapter}</dd></div><div><dt>Illustrative parameters</dt><dd>{formatCompactNumber(run.model.parameterCount)}</dd></div><div><dt>Example quantization</dt><dd>{run.model.quantization}</dd></div><div><dt>Illustrative GPU estimate</dt><dd>{formatNumber(run.summary.estimatedGpuMinutes, 1)} GPU min</dd></div></dl></section>
      </section>
      <section className="timeline-section" aria-labelledby="timeline-title">
        <header className="section-heading"><div><span className="eyebrow">Illustrative checkpoint trace</span><h2 id="timeline-title">Authored demo timeline</h2></div><span>{run.checkpoints.length} illustrative states</span></header>
        {run.checkpoints.length === 0 ? (
          <div className="empty-state"><Clock3 aria-hidden="true" /><h3>No illustrative checkpoints</h3><p>The example artifact contains no authored checkpoint values.</p></div>
        ) : (
          <div className="checkpoint-trace-stack">
            <CheckpointTrace checkpoints={run.checkpoints} />
            <ol className="timeline">{run.checkpoints.map((checkpoint, index) => <li key={`${checkpoint.step}-${checkpoint.label}`} style={{ '--checkpoint-progress': `${(checkpoint.step / maxStep) * 100}%` } as React.CSSProperties}><div className="timeline-axis"><span>{String(index + 1).padStart(2, '0')}</span></div><article><header><div><span>Illustrative step {checkpoint.step}</span><h3>{checkpoint.label}</h3></div><strong>{formatPercent(checkpoint.defendedWinRate)} illustrative defended win rate</strong></header><dl><div><dt>Illustrative train loss</dt><dd>{formatNumber(checkpoint.trainLoss, 3)}</dd></div><div><dt>Illustrative eval reward</dt><dd>{formatNumber(checkpoint.evalReward, 3)}</dd></div><div><dt>Illustrative GPU minutes</dt><dd>{formatNumber(checkpoint.gpuMinutes, 1)} min</dd></div></dl><div className="checkpoint-rule" aria-hidden="true"><i /></div></article></li>)}</ol>
          </div>
        )}
      </section>
      <section className="method-note"><Beaker aria-hidden="true" /><div><span className="eyebrow">Example export rule</span><h2>Only example defended edges are eligible.</h2><p>The Python exporter retains margins and authored score evidence as metadata. Missing scores, low confidence and cross-objective tradeoffs remain ambiguous. These eligibility counts are computed on illustrative inputs, not evidence of training quality.</p></div><dl><div><dt>Example eligible edges</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Example ambiguous edges</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></section>
      <section className="mutation-register" aria-labelledby="mutation-title"><header className="section-heading"><div><span className="eyebrow"><Microscope size={14} aria-hidden="true" />Illustrative mutations</span><h2 id="mutation-title">Example mutation register</h2></div><span>{run.summary.mutationFlipCount} computed example flips</span></header><p>Mutated text and authored before/after scores illustrate the audit rule. Deltas and flips are deterministic example calculations, not measured model responses.</p>{run.mutations.length === 0 ? <p className="empty-inline">No illustrative mutations are included in this artifact.</p> : <div>{run.mutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind} · {shortId(mutation.sourceCandidateId)}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} / computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</div>}</section>
    </div>
  )
}
