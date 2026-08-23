import { useMemo, useState } from 'react'
import type React from 'react'
import { ArrowRight, Beaker, CheckCircle2, CircleAlert, Clock3, Microscope, ShieldCheck } from 'lucide-react'
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

  return (
    <div className="view-stack">
      <header className="view-hero overview-hero">
        <div><span className="eyebrow">Run overview / {shortId(run.id)}</span><h1>Evidence before preference.</h1><p>Verge keeps reward disagreements visible, then exports only the edges every shared objective can defend.</p></div>
        <dl className="run-stamp"><div><dt>Status</dt><dd><span className="status-dot" />{run.status}</dd></div><div><dt>Model</dt><dd>{run.model.base}</dd></div><div><dt>Created</dt><dd>{formatDate(run.createdAt)}</dd></div></dl>
      </header>

      <section className="ledger" aria-labelledby="ledger-title">
        <header className="section-heading"><div><span className="eyebrow">Run ledger</span><h2 id="ledger-title">{run.name}</h2></div><span>schema v{run.schemaVersion}</span></header>
        <div className="ledger-grid">
          <article className="ledger-lead"><span>Defended edge yield</span><strong>{formatPercent(defendedRate)}</strong><p>{run.summary.defendedPairCount} of {totalPairs} comparisons survive robust Pareto checks.</p><button type="button" className="text-action" onClick={() => onNavigate('topology')}>Trace the order <ArrowRight size={15} aria-hidden="true" /></button></article>
          <dl className="ledger-metrics"><div><dt>Prompts</dt><dd>{run.summary.promptCount}</dd></div><div><dt>Candidates</dt><dd>{run.summary.candidateCount}</dd></div><div><dt>Ambiguous</dt><dd className="coral-text">{run.summary.ambiguousPairCount}</dd></div><div><dt>Mutation flips</dt><dd>{run.summary.mutationFlipCount}</dd></div><div><dt>GPU estimate</dt><dd>{formatNumber(run.summary.estimatedGpuMinutes, 1)} min</dd></div><div><dt>Cost estimate</dt><dd>${formatNumber(run.summary.estimatedCostUsd)}</dd></div></dl>
          <ThreeInstrument kind="constellation" title="Candidate density field" description={`${run.summary.candidateCount} candidates; motion reflects ${formatPercent(ambiguityRate)} ambiguity and brightness reflects mean pair confidence.`} metrics={{ speed: 0.25 + ambiguityRate, density: 0.55 + run.summary.candidateCount / 20, opacity: 0.68 + defendedRate * 0.2, hue: 0.18, brightness: 0.65 + averageConfidence * 0.5 }} />
        </div>
      </section>

      <section className="objective-ledger" aria-labelledby="objective-title">
        <header className="section-heading"><div><span className="eyebrow">Reward specification</span><h2 id="objective-title">Objectives remain plural</h2></div><span>{run.objectives.length} dimensions</span></header>
        <div className="objective-rows">
          {run.objectives.map((objective, index) => <article key={objective.id} className={`objective-row objective-row--${index % 4}`}><span className="objective-index">0{index + 1}</span><div><h3>{objective.label}</h3><p>{objective.description}</p></div><dl><div><dt>Direction</dt><dd>{objective.direction}</dd></div><div><dt>Mean</dt><dd>{formatNumber(objective.mean)}</dd></div><div><dt>Δ audit</dt><dd>{objective.delta > 0 ? '+' : ''}{formatNumber(objective.delta)}</dd></div></dl></article>)}
        </div>
      </section>

      <section className="triage-strip" aria-labelledby="triage-title">
        <header><span className="eyebrow">Next examination</span><h2 id="triage-title">Follow the uncertain edge.</h2></header>
        <div><button type="button" onClick={() => onNavigate('pairs')}><span><CircleAlert size={18} aria-hidden="true" />Pair lab</span><strong>{run.summary.ambiguousPairCount} comparisons need inspection</strong><ArrowRight aria-hidden="true" /></button><button type="button" onClick={() => onNavigate('runbook')}><span><Clock3 size={18} aria-hidden="true" />Runbook</span><strong>{run.checkpoints.length} checkpoints record the path</strong><ArrowRight aria-hidden="true" /></button></div>
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
      <header className="view-hero"><div><span className="eyebrow">Reward topology</span><h1>A ranking is too certain.</h1><p>This is a partial order: defended arrows coexist with unresolved tradeoffs. Filter the aperture without collapsing evidence.</p></div><ThreeInstrument compact kind="topology" title="Edge confidence texture" description={`Hue follows ${formatPercent(defendedRate)} defended edge share; brightness follows ${formatPercent(meanConfidence)} mean confidence.`} metrics={{ speed: 0, density: 0, opacity: 1, hue: -0.32 + defendedRate * 0.68, brightness: 0.62 + meanConfidence * 0.58 }} /></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visibleCandidates.length} resultLabel={`candidates · ${visiblePairs.length} edges`} />
      <PartialOrderGraph run={run} candidates={visibleCandidates} pairs={visiblePairs} selectedCandidateId={effectiveSelectedCandidateId} onSelectCandidate={setSelectedCandidateId} />
    </div>
  )
}

function PairChoice({ run, pair, role }: { run: RunArtifact; pair: Pair; role: 'chosen' | 'rejected' }) {
  const candidate = candidateFor(run, role === 'chosen' ? pair.chosenId : pair.rejectedId)
  if (!candidate) return <div className="empty-inline">Candidate absent from artifact.</div>
  return (
    <article className={`pair-choice pair-choice--${role}`}>
      <header><span>{role === 'chosen' ? 'Chosen' : 'Rejected'}</span><strong>{candidateLabel(run, candidate)} · {shortId(candidate.id)}</strong></header>
      <blockquote>{candidate.output}</blockquote>
      <div className="pair-score-stack">{run.objectives.map((objective) => { const score = candidate.scores[objective.id]; return <div key={objective.id}><span>{objective.label}</span><strong>{score ? formatNumber(score.value) : '—'}</strong><small>{score ? `${formatPercent(score.confidence)} confidence` : 'missing score'}</small>{score ? <p>{score.evidence}</p> : null}</div> })}</div>
    </article>
  )
}

export function PairLab({ run, filters, onFiltersChange }: FilteredViewProps) {
  const visiblePairs = useMemo(() => filteredPairs(run, filters), [run, filters])
  const [selectedPairId, setSelectedPairId] = useState<string | null>(visiblePairs[0]?.id ?? null)
  const selectedPair = visiblePairs.find((pair) => pair.id === selectedPairId) ?? visiblePairs[0]
  const relevantMutations = selectedPair ? run.mutations.filter((mutation) => mutation.sourceCandidateId === selectedPair.chosenId || mutation.sourceCandidateId === selectedPair.rejectedId) : []

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Pair lab</span><h1>Inspect the edge, not the score.</h1><p>Compare verifier evidence objective by objective. Ambiguity is a valid result, never a missing label.</p></div><dl className="verdict-counts"><div><dt>Defended</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Ambiguous</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visiblePairs.length} resultLabel="comparisons" />
      {selectedPair ? (
        <div className="pair-workbench">
          <aside className="pair-index" aria-label="Filtered comparison pairs"><span className="eyebrow">Pair register</span>{visiblePairs.map((pair) => { const prompt = promptFor(run, pair.promptId); return <button type="button" key={pair.id} className={pair.id === selectedPair.id ? 'is-selected' : undefined} aria-pressed={pair.id === selectedPair.id} onClick={() => setSelectedPairId(pair.id)}><span><i className={`verdict-mark verdict-mark--${pair.verdict}`} />{shortId(pair.id)}</span><small>{prompt?.domain}</small><strong>{formatPercent(pair.confidence)}</strong></button> })}</aside>
          <main className="pair-evidence">
            <header className="pair-evidence-header"><div><span className={`verdict-label verdict-label--${selectedPair.verdict}`}>{selectedPair.verdict === 'defended' ? <CheckCircle2 size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{selectedPair.verdict}</span><h2>{shortId(selectedPair.id)}</h2><p>{promptFor(run, selectedPair.promptId)?.text}</p></div><dl><div><dt>Confidence</dt><dd>{formatPercent(selectedPair.confidence)}</dd></div><div><dt>Domain</dt><dd>{promptFor(run, selectedPair.promptId)?.domain}</dd></div></dl></header>
            <section className="specification-band" aria-labelledby="spec-title"><h3 id="spec-title">Prompt-level reward specification</h3><div><ul>{promptFor(run, selectedPair.promptId)?.specification.rubrics.map((rubric) => <li key={rubric}>{rubric}</li>)}</ul><ul>{promptFor(run, selectedPair.promptId)?.specification.constraints.map((constraint) => <li key={constraint}>{constraint}</li>)}</ul></div></section>
            <div className="pair-comparison"><PairChoice run={run} pair={selectedPair} role="chosen" /><PairChoice run={run} pair={selectedPair} role="rejected" /></div>
            <section className="verdict-reason"><span className="eyebrow">Why this verdict</span><h3>{selectedPair.reason}</h3><div className="margin-strip">{run.objectives.map((objective) => <div key={objective.id}><span>{objective.label}</span><strong>{selectedPair.margins[objective.id] === undefined ? 'not shared' : `${selectedPair.margins[objective.id] > 0 ? '+' : ''}${formatNumber(selectedPair.margins[objective.id])}`}</strong></div>)}</div></section>
            <section className="mutation-evidence"><header><span className="eyebrow">Mutation audit</span><strong>{relevantMutations.length} linked probes</strong></header>{relevantMutations.length === 0 ? <p className="empty-inline">No meaning-preserving mutation targets either candidate in this pair.</p> : relevantMutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'verdict flipped' : 'stable'} · objective deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</section>
          </main>
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
      <header className="view-hero"><div><span className="eyebrow">Runbook / deterministic trace</span><h1>The path is part of the result.</h1><p>Checkpoint evidence, mutation probes, and the adapter manifest form one inspectable training record.</p></div><div className="runbook-seal"><ShieldCheck aria-hidden="true" /><span>Artifact status</span><strong>{run.status}</strong><small>{run.id}</small></div></header>
      <section className="runbook-grid">
        <ThreeInstrument kind="diagnostics" title="Run activity diagnostic" description={`${run.checkpoints.length} checkpoints; speed follows GPU activity, density follows defended win rate, brightness follows evaluation reward.`} metrics={{ speed: clamp((lastCheckpoint?.gpuMinutes ?? 0) / Math.max(run.summary.estimatedGpuMinutes, 1), 0.15, 1.3), density: 0.45 + (lastCheckpoint?.defendedWinRate ?? 0), opacity: 0.78, hue: 0.36 + clamp(rewardGain, -0.3, 0.3), brightness: 0.72 + clamp(lastCheckpoint?.evalReward ?? 0, 0, 1) * 0.48 }} />
        <section className="manifest" aria-labelledby="manifest-title"><header><span className="eyebrow">Model manifest</span><h2 id="manifest-title">Adapter specimen</h2></header><dl><div><dt>Base</dt><dd>{run.model.base}</dd></div><div><dt>Method</dt><dd>{run.model.method}</dd></div><div><dt>Adapter</dt><dd>{run.model.adapter}</dd></div><div><dt>Parameters</dt><dd>{formatCompactNumber(run.model.parameterCount)}</dd></div><div><dt>Quantization</dt><dd>{run.model.quantization}</dd></div><div><dt>Budget</dt><dd>{formatNumber(run.summary.estimatedGpuMinutes, 1)} GPU min</dd></div></dl></section>
      </section>
      <section className="timeline-section" aria-labelledby="timeline-title"><header className="section-heading"><div><span className="eyebrow">Checkpoint trace</span><h2 id="timeline-title">Training timeline</h2></div><span>{run.checkpoints.length} recorded states</span></header>{run.checkpoints.length === 0 ? <div className="empty-state"><Clock3 aria-hidden="true" /><h3>No checkpoints recorded</h3><p>The run artifact contains no training trace.</p></div> : <ol className="timeline">{run.checkpoints.map((checkpoint, index) => <li key={`${checkpoint.step}-${checkpoint.label}`} style={{ '--checkpoint-progress': `${(checkpoint.step / maxStep) * 100}%` } as React.CSSProperties}><div className="timeline-axis"><span>{String(index + 1).padStart(2, '0')}</span></div><article><header><div><span>Step {checkpoint.step}</span><h3>{checkpoint.label}</h3></div><strong>{formatPercent(checkpoint.defendedWinRate)} defended win rate</strong></header><dl><div><dt>Train loss</dt><dd>{formatNumber(checkpoint.trainLoss, 3)}</dd></div><div><dt>Eval reward</dt><dd>{formatNumber(checkpoint.evalReward, 3)}</dd></div><div><dt>GPU elapsed</dt><dd>{formatNumber(checkpoint.gpuMinutes, 1)} min</dd></div></dl><div className="checkpoint-rule" aria-hidden="true"><i /></div></article></li>)}</ol>}</section>
      <section className="method-note"><Beaker aria-hidden="true" /><div><span className="eyebrow">Export doctrine</span><h2>Only defended edges leave the bench.</h2><p>DPO output retains margins and verifier evidence as metadata. Missing scores, low confidence, and cross-objective tradeoffs remain ambiguous rather than becoming synthetic winners.</p></div><dl><div><dt>Eligible edges</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Held for review</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></section>
      <section className="mutation-register" aria-labelledby="mutation-title"><header className="section-heading"><div><span className="eyebrow"><Microscope size={14} aria-hidden="true" />Bias probes</span><h2 id="mutation-title">Mutation register</h2></div><span>{run.summary.mutationFlipCount} flips</span></header>{run.mutations.length === 0 ? <p className="empty-inline">No mutation probes were recorded for this run.</p> : <div>{run.mutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind} · {shortId(mutation.sourceCandidateId)}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'flipped' : 'stable'} / objective deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</div>}</section>
    </div>
  )
}
