import { useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { ArrowLeft, ArrowRight, Beaker, CheckCircle2, CircleAlert, Clock3, Microscope, ShieldCheck } from 'lucide-react'
import { CheckpointTrace } from '../components/CheckpointTrace'
import { ConfidenceDial } from '../components/ConfidenceDial'
import { EvidenceFilters } from '../components/EvidenceFilters'
import { PartialOrderGraph } from '../components/PartialOrderGraph'
import { candidateFor, candidateLabel, filteredCandidates, filteredPairs, formatCompactNumber, formatDate, formatNumber, formatPercent, promptFor, shortId } from '../lib'
import type { LabFilters, Pair, RunArtifact, ViewId } from '../types'

type SharedViewProps = { run: RunArtifact }

type OverviewProps = SharedViewProps & { onNavigate: (view: ViewId) => void }

export function Overview({ run, onNavigate }: OverviewProps) {
  const totalPairs = run.summary.defendedPairCount + run.summary.ambiguousPairCount
  const defendedRate = totalPairs === 0 ? 0 : run.summary.defendedPairCount / totalPairs

  return (
    <div className="view-stack">
      <section className="bench-hero">
        <header className="view-hero overview-hero">
          <div>
            <span className="eyebrow">Pair selection / authored examples</span>
            <h1>Better on one score. Worse on none.</h1>
            <p>I compare two answers to the same prompt without averaging away their tradeoffs. A Pareto preference means no score worsens and at least one improves. If one answer gains on one score but loses on another, I abstain: neither is a winner under this rule.</p>
            <div className="hero-readout" role="list" aria-label="Counts computed from authored example inputs, not evaluation results">
              <span role="listitem"><strong>{run.summary.candidateCount}</strong> example answers</span>
              <span role="listitem"><strong>{run.summary.defendedPairCount}</strong> selected pairs</span>
              <span role="listitem"><strong>{run.summary.ambiguousPairCount}</strong> abstentions</span>
            </div>
          </div>
          <dl className="run-stamp"><div><dt>Demo status</dt><dd><span className="status-dot" />{run.status}</dd></div><div><dt>Example model</dt><dd>{run.model.base}</dd></div><div><dt>Demo date</dt><dd>{formatDate(run.createdAt)}</dd></div></dl>
        </header>
      </section>

      <section className="ledger" aria-labelledby="ledger-title">
        <header className="section-heading"><div><span className="eyebrow">The selection rule</span><h2 id="ledger-title">Select a pair, or leave it unresolved.</h2></div></header>
        <div className="ledger-grid">
          <article className="ledger-lead"><span>Selected from this example</span><strong>{formatPercent(defendedRate)}</strong><p>{run.summary.defendedPairCount} of {totalPairs} authored comparisons pass the rule. This is an example count, not a measured accuracy or win rate.</p><button type="button" className="text-action" onClick={() => onNavigate('pairs')}>Inspect a pair <ArrowRight size={15} aria-hidden="true" /></button></article>
          <div className="selection-rule"><h3>What counts as better?</h3><p>Higher is better for a score marked “maximize”; lower is better for “minimize”. The Python comparison also subtracts an uncertainty allowance from each advantage. Every adjusted margin must be nonnegative, and at least one must exceed the improvement threshold.</p><h3>When do I abstain?</h3><p>A tradeoff, near tie, missing score, low confidence or overlapping confidence bounds leaves the pair unresolved. The artifact calls a selected pair <code>defended</code> and an abstention <code>ambiguous</code>. Neither label proves that a model is better.</p><button type="button" className="text-action" onClick={() => onNavigate('topology')}>Compare example scores <ArrowRight size={15} aria-hidden="true" /></button></div>
        </div>
      </section>

      <section className="objective-ledger" aria-labelledby="objective-title">
        <header className="section-heading"><div><span className="eyebrow">Scores used in this example</span><h2 id="objective-title">Keep each objective separate.</h2></div></header>
        <div className="objective-rows">
          {run.objectives.map((objective, index) => (
            <article key={objective.id} className="objective-row">
              <span className="objective-index">0{index + 1}</span>
              <div><h3>{objective.label}</h3><p>{objective.description}</p></div>
              <dl><div><dt>Better direction</dt><dd>{objective.direction === 'maximize' ? 'Higher score' : 'Lower score'}</dd></div><div><dt>Example mean</dt><dd>{formatNumber(objective.mean)}</dd></div></dl>
            </article>
          ))}
        </div>
      </section>

      <section className="triage-strip" aria-labelledby="triage-title">
        <header><span className="eyebrow">Explore the examples</span><h2 id="triage-title">Why was no winner selected?</h2></header>
        <div><button type="button" onClick={() => onNavigate('pairs')}><span><CircleAlert size={18} aria-hidden="true" />Inspect pairs</span><strong>{run.summary.ambiguousPairCount} example abstentions to inspect</strong><ArrowRight aria-hidden="true" /></button><button type="button" onClick={() => onNavigate('runbook')}><span><Clock3 size={18} aria-hidden="true" />Example artifact</span><strong>Authored metadata, not a training record</strong><ArrowRight aria-hidden="true" /></button></div>
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


  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Compare example scores</span><h1>Not every pair has a winner.</h1><p>An arrow points to the answer selected by the Pareto rule: no adjusted score worsens, and at least one improves beyond the threshold. A dashed line means I abstain. Scores and coordinates are authored examples, not model measurements.</p></div></header>
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
  const evidenceHeading = useRef<HTMLHeadingElement>(null)

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Inspect example pairs</span><h1>See why a pair was selected.</h1><p>No score may worsen; at least one must improve. Tradeoffs abstain rather than forcing a winner. These example verdicts also account for confidence, missing scores and near ties.</p></div><dl className="verdict-counts"><div><dt>Selected pairs</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Abstentions</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visiblePairs.length} resultLabel="comparisons" />
      {selectedPair ? (
        <div className={`pair-workbench ${drillDownOpen ? 'pair-workbench--detail' : 'pair-workbench--register'}`}>
          <aside className="pair-index pair-index-panel" aria-label="Filtered illustrative comparison pairs">
            <span className="eyebrow">Illustrative pair register</span>
            {visiblePairs.map((pair) => {
              const prompt = promptFor(run, pair.promptId)
              const selected = pair.id === selectedPair.id
              return (
                <button type="button" key={pair.id} data-pair-id={pair.id} className={selected ? 'is-selected' : undefined} aria-pressed={selected} onClick={() => { flushSync(() => setSelectedPairId(pair.id)); evidenceHeading.current?.focus() }}>
                  <span><i className={`verdict-mark verdict-mark--${pair.verdict}`} />{shortId(pair.id)}</span>
                  <small>{prompt?.domain} · {pair.verdict === 'defended' ? 'Selected' : 'Abstain'}</small>
                  <ConfidenceDial value={pair.confidence} tone={pair.verdict} label={`${shortId(pair.id)} confidence`} size="inline" />
                </button>
              )
            })}
          </aside>
          <section className="pair-evidence pair-evidence-panel pair-evidence--active calibration-chrome" data-armed="true" aria-labelledby="pair-evidence-title">
            <button type="button" className="pair-back" onClick={() => { const id = selectedPair.id; flushSync(() => setSelectedPairId(null)); document.querySelector<HTMLButtonElement>(`[data-pair-id="${id}"]`)?.focus() }}><ArrowLeft size={15} aria-hidden="true" />Back to pairs</button>
            <header className="pair-evidence-header">
              <div><span className={`verdict-label verdict-label--${selectedPair.verdict}`}>{selectedPair.verdict === 'defended' ? <CheckCircle2 size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{selectedPair.verdict === 'defended' ? 'Selected (defended)' : 'Abstain (ambiguous)'}</span><h2 id="pair-evidence-title" ref={evidenceHeading} tabIndex={-1}>{shortId(selectedPair.id)}</h2><p>{promptFor(run, selectedPair.promptId)?.text}</p></div>
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
            <section className="verdict-reason"><span className="eyebrow">Why this example gets this verdict</span><h3>{selectedPair.reason}</h3><p>Margins compare {selectedPair.verdict === 'defended' ? 'the chosen answer with the rejected answer' : 'Candidate A with Candidate B; this order does not imply a preference'}. They account for score direction and subtract an uncertainty allowance. Selection requires every margin ≥ 0 and at least one above the improvement threshold. Negative margins or a near tie cannot select a winner; missing or low-confidence scores also abstain.</p><div className="margin-strip">{run.objectives.map((objective) => <div key={objective.id}><span>{objective.label}</span><strong>{selectedPair.margins[objective.id] === undefined ? 'not shared' : `${selectedPair.margins[objective.id] > 0 ? '+' : ''}${formatNumber(selectedPair.margins[objective.id], 4)}`}</strong></div>)}</div></section>
            <section className="mutation-evidence"><header><span className="eyebrow">Illustrative mutation comparisons</span><strong>{relevantMutations.length} linked examples</strong></header><p>Verdict flips and deltas are computed from authored before/after scores, not measured rescoring or model bias tests.</p>{relevantMutations.length === 0 ? <p className="empty-inline">No example mutation targets either candidate in this pair.</p> : relevantMutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} · computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</section>
          </section>
        </div>
      ) : <div className="empty-state"><CircleAlert aria-hidden="true" /><h3>No comparisons match</h3><p>Clear the candidate search or widen the domain and verdict filters.</p></div>}
    </div>
  )
}

export function Runbook({ run }: SharedViewProps) {

  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Runbook / illustrative demo</span><h1>An authored example, not a run.</h1><p>Illustrative checkpoints, mutation comparisons and model metadata show the artifact format. They are not a training record, a trained adapter or GPU diagnostics.</p></div><div className="runbook-seal"><ShieldCheck aria-hidden="true" /><span>Example artifact status</span><strong>{run.status}</strong><small>{run.id}</small></div></header>
      <section className="runbook-grid">
        <section className="manifest" aria-labelledby="manifest-title"><header><span className="eyebrow">Illustrative model metadata</span><h2 id="manifest-title">Example adapter description</h2></header><dl><div><dt>Example base</dt><dd>{run.model.base}</dd></div><div><dt>Example method</dt><dd>{run.model.method}</dd></div><div><dt>Example adapter name</dt><dd>{run.model.adapter}</dd></div><div><dt>Illustrative parameters</dt><dd>{formatCompactNumber(run.model.parameterCount)}</dd></div><div><dt>Example quantization</dt><dd>{run.model.quantization}</dd></div><div><dt>Illustrative GPU estimate</dt><dd>{formatNumber(run.summary.estimatedGpuMinutes, 1)} GPU min</dd></div></dl></section>
      </section>
      <section className="timeline-section" aria-labelledby="timeline-title">
        <header className="section-heading"><div><span className="eyebrow">Illustrative checkpoint trace</span><h2 id="timeline-title">Authored demo timeline</h2></div><span>{run.checkpoints.length} illustrative states</span></header>
        {run.checkpoints.length === 0 ? (
          <div className="empty-state"><Clock3 aria-hidden="true" /><h3>No illustrative checkpoints</h3><p>The example artifact contains no authored checkpoint values.</p></div>
        ) : (
          <div className="checkpoint-trace-stack">
            <CheckpointTrace checkpoints={run.checkpoints} />
            <ol className="timeline">{run.checkpoints.map((checkpoint, index) => <li key={`${checkpoint.step}-${checkpoint.label}`}><div className="timeline-axis"><span>{String(index + 1).padStart(2, '0')}</span></div><article><header><div><span>Illustrative step {checkpoint.step}</span><h3>{checkpoint.label}</h3></div><strong>{formatPercent(checkpoint.defendedWinRate)} illustrative defended win rate</strong></header><dl><div><dt>Illustrative train loss</dt><dd>{formatNumber(checkpoint.trainLoss, 3)}</dd></div><div><dt>Illustrative eval reward</dt><dd>{formatNumber(checkpoint.evalReward, 3)}</dd></div><div><dt>Illustrative GPU minutes</dt><dd>{formatNumber(checkpoint.gpuMinutes, 1)} min</dd></div></dl></article></li>)}</ol>
          </div>
        )}
      </section>
      <section className="method-note"><Beaker aria-hidden="true" /><div><span className="eyebrow">Example export rule</span><h2>Only example defended edges are eligible.</h2><p>The Python exporter retains margins and authored score evidence as metadata. Missing scores, low confidence and cross-objective tradeoffs remain ambiguous. These eligibility counts are computed on illustrative inputs, not evidence of training quality.</p></div><dl><div><dt>Example eligible edges</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Example ambiguous edges</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></section>
      <section className="mutation-register" aria-labelledby="mutation-title"><header className="section-heading"><div><span className="eyebrow"><Microscope size={14} aria-hidden="true" />Illustrative mutations</span><h2 id="mutation-title">Example mutation register</h2></div><span>{run.summary.mutationFlipCount} computed example flips</span></header><p>Mutated text and authored before/after scores illustrate the audit rule. Deltas and flips are deterministic example calculations, not measured model responses.</p>{run.mutations.length === 0 ? <p className="empty-inline">No illustrative mutations are included in this artifact.</p> : <div>{run.mutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind} · {shortId(mutation.sourceCandidateId)}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} / computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</div>}</section>
    </div>
  )
}
