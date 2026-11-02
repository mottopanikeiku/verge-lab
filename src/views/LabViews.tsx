import { useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { ArrowLeft, ArrowRight, Beaker, CheckCircle2, CircleAlert, FileOutput, Microscope } from 'lucide-react'
import { ConfidenceDial } from '../components/ConfidenceDial'
import { EvidenceFilters } from '../components/EvidenceFilters'
import { PartialOrderGraph } from '../components/PartialOrderGraph'
import { candidateFor, candidateLabel, filteredCandidates, filteredPairs, formatNumber, formatPercent, promptFor, shortId } from '../lib'
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
            <span className="eyebrow">Pair selection / examples</span>
            <h1>Pareto preference-pair selection</h1>
            <p>I compare answers to the same prompt across several scores. A pair is selected when no adjusted score worsens and at least one improves beyond the threshold. When scores trade off, the selector abstains.</p>
            <div className="hero-readout" role="list" aria-label="Counts computed from example inputs">
              <span role="listitem"><strong>{run.summary.candidateCount}</strong> example answers</span>
              <span role="listitem"><strong>{run.summary.defendedPairCount}</strong> selected pairs</span>
              <span role="listitem"><strong>{run.summary.ambiguousPairCount}</strong> abstentions</span>
            </div>
          </div>
        </header>
      </section>

      <section className="ledger" aria-labelledby="ledger-title">
        <header className="section-heading"><div><span className="eyebrow">The selection rule</span><h2 id="ledger-title">Selected pairs and abstentions</h2></div></header>
        <div className="ledger-grid">
          <article className="ledger-lead"><span>Selected from this example</span><strong>{formatPercent(defendedRate)}</strong><p>{run.summary.defendedPairCount} of {totalPairs} example comparisons pass the rule.</p><button type="button" className="text-action" onClick={() => onNavigate('pairs')}>Inspect a pair <ArrowRight size={15} aria-hidden="true" /></button></article>
          <div className="selection-rule"><h3>Score margins</h3><p>Higher is better for “maximize”; lower is better for “minimize”. A heuristic penalty based on the supplied confidence reduces each advantage. Every adjusted margin must be nonnegative, and at least one must exceed the improvement threshold.</p><h3>Abstentions</h3><p>Tradeoffs, near ties, missing scores, low confidence and penalties that remove dominance leave a pair unresolved. The artifact labels selected pairs <code>defended</code> and abstentions <code>ambiguous</code>.</p><button type="button" className="text-action" onClick={() => onNavigate('topology')}>Compare example scores <ArrowRight size={15} aria-hidden="true" /></button></div>
        </div>
      </section>

      <section className="objective-ledger" aria-labelledby="objective-title">
        <header className="section-heading"><div><span className="eyebrow">Scores used in this example</span><h2 id="objective-title">Separate objectives</h2></div></header>
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
        <header><span className="eyebrow">Explore the examples</span><h2 id="triage-title">Inspect the decisions</h2></header>
        <div><button type="button" onClick={() => onNavigate('pairs')}><span><CircleAlert size={18} aria-hidden="true" />Inspect pairs</span><strong>{run.summary.ambiguousPairCount} example abstentions</strong><ArrowRight aria-hidden="true" /></button><button type="button" onClick={() => onNavigate('export')}><span><FileOutput size={18} aria-hidden="true" />Export</span><strong>DPO export and text edits</strong><ArrowRight aria-hidden="true" /></button></div>
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
      <header className="view-hero"><div><span className="eyebrow">Example scores</span><h1>Compare answer scores</h1><p>An arrow points to the answer selected by the Pareto rule; a dashed line marks an abstention. Select an answer to read its scores. Graph positions are illustrative coordinates.</p></div></header>
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
        return <div key={objective.id}><span>Example {objective.label} score</span><strong>{score ? formatNumber(score.value) : '—'}</strong>{score ? <ConfidenceDial value={score.confidence} tone="neutral" label={`${sideLabel} ${objective.label} confidence`} size="inline" /> : <small>missing example score</small>}{score ? <p>Example rationale: {score.evidence}</p> : null}</div>
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
      <header className="view-hero"><div><span className="eyebrow">Example pairs</span><h1>Inspect pair decisions</h1><p>Compare the answers, their scores and adjusted margins. The selector also checks confidence, missing scores and near ties before choosing a winner.</p></div><dl className="verdict-counts"><div><dt>Selected pairs</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Abstentions</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></header>
      <EvidenceFilters run={run} filters={filters} onChange={onFiltersChange} resultCount={visiblePairs.length} resultLabel="comparisons" />
      {selectedPair ? (
        <div className={`pair-workbench ${drillDownOpen ? 'pair-workbench--detail' : 'pair-workbench--register'}`}>
          <aside className="pair-index pair-index-panel" aria-label="Filtered illustrative comparison pairs">
            <span className="eyebrow">Example comparisons</span>
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
              <h3 id="spec-title">Prompt criteria</h3>
              <div>
                <section aria-labelledby="rubrics-title"><h4 id="rubrics-title">Rubrics</h4><ul>{promptFor(run, selectedPair.promptId)?.specification.rubrics.map((rubric) => <li key={rubric}>{rubric}</li>)}</ul></section>
                <section aria-labelledby="constraints-title"><h4 id="constraints-title">Constraints</h4><ul>{promptFor(run, selectedPair.promptId)?.specification.constraints.map((constraint) => <li key={constraint}>{constraint}</li>)}</ul></section>
              </div>
            </section>
            <div className="pair-comparison"><PairChoice run={run} pair={selectedPair} role="chosen" /><PairChoice run={run} pair={selectedPair} role="rejected" /></div>
            <section className="verdict-reason"><span className="eyebrow">Selection reason</span><h3>{selectedPair.reason}</h3><p>Margins compare {selectedPair.verdict === 'defended' ? 'the chosen answer with the rejected answer' : 'Candidate A with Candidate B; this order does not imply a preference'}. They account for score direction and a heuristic confidence penalty. Selection requires every margin ≥ 0 and at least one above the improvement threshold.</p><div className="margin-strip">{run.objectives.map((objective) => <div key={objective.id}><span>{objective.label}</span><strong>{selectedPair.margins[objective.id] === undefined ? 'not shared' : `${selectedPair.margins[objective.id] > 0 ? '+' : ''}${formatNumber(selectedPair.margins[objective.id], 4)}`}</strong></div>)}</div></section>
            <section className="mutation-evidence"><header><span className="eyebrow">Example mutations</span><strong>{relevantMutations.length} linked examples</strong></header><p>These text edits come with example before/after scores. The selector computes score changes and verdict flips from them.</p>{relevantMutations.length === 0 ? <p className="empty-inline">No example mutation targets either candidate in this pair.</p> : relevantMutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} · computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</section>
          </section>
        </div>
      ) : <div className="empty-state"><CircleAlert aria-hidden="true" /><h2>No comparisons match</h2><p>Clear the candidate search or widen the domain and verdict filters.</p></div>}
    </div>
  )
}

export function ExportView({ run }: SharedViewProps) {
  return (
    <div className="view-stack">
      <header className="view-hero"><div><span className="eyebrow">Export</span><h1>Export and text edits</h1><p>The exporter writes selected pairs as DPO JSONL. Text edits show which pair decisions flip when an answer changes.</p></div></header>
      <section className="method-note"><Beaker aria-hidden="true" /><div><span className="eyebrow">DPO export</span><h2>Export selected pairs</h2><p>The Python exporter recomputes each decision and writes selected pairs with their margins and score rationales. Abstentions are excluded.</p></div><dl><div><dt>Example eligible edges</dt><dd>{run.summary.defendedPairCount}</dd></div><div><dt>Example ambiguous edges</dt><dd>{run.summary.ambiguousPairCount}</dd></div></dl></section>
      <section className="mutation-register" aria-labelledby="mutation-title"><header className="section-heading"><div><span className="eyebrow"><Microscope size={14} aria-hidden="true" />Text changes</span><h2 id="mutation-title">Example mutations</h2></div><span>{run.summary.mutationFlipCount} computed example flips</span></header><p>Each text edit has example before/after scores. The selector computes deltas and checks whether the pair decision changes.</p>{run.mutations.length === 0 ? <p className="empty-inline">No example mutations are included in this artifact.</p> : <div>{run.mutations.map((mutation) => <details key={mutation.id}><summary><span>{mutation.kind} · {shortId(mutation.sourceCandidateId)}</span><strong className={mutation.flipped ? 'coral-text' : undefined}>{mutation.flipped ? 'example verdict flipped' : 'example verdict stable'} / computed deltas</strong></summary><blockquote>{mutation.output}</blockquote><div className="mutation-deltas">{Object.entries(mutation.scoreDelta).map(([objectiveId, delta]) => <span key={objectiveId}>{objectiveId}: {delta > 0 ? '+' : ''}{formatNumber(delta)}</span>)}</div><p>{mutation.reason}</p></details>)}</div>}</section>
    </div>
  )
}
