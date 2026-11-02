import { useEffect, useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { ArrowLeft, CheckCircle2, CircleAlert } from 'lucide-react'
import type { RealPair, RealPairSample, RealPairsArtifact, ResponseNumber } from '../types'

const SAMPLE_LABELS: Record<RealPairSample, string> = {
  'hash-order': 'Start of the pool in hash order',
  'pareto-abstains': 'Added: a Pareto abstention',
  'pareto-against-human': 'Added: Pareto disagrees with the annotators',
}

const FILTERS: Record<string, { label: string; keep: (pair: RealPair) => boolean }> = {
  all: { label: 'All pairs', keep: () => true },
  keeps: { label: 'Pareto keeps', keep: (pair) => pair.selectors.pareto.pick !== null },
  abstains: { label: 'Pareto abstains', keep: (pair) => pair.selectors.pareto.pick === null },
  against: { label: 'Pareto disagrees with the annotators', keep: (pair) => ![null, pair.selectors.human.pick].includes(pair.selectors.pareto.pick) },
  gap: { label: 'In the helpfulness-gap set', keep: (pair) => pair.selectors.gap.inTrainingSet },
}

const integer = new Intl.NumberFormat('en-US')

type LoadState = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: RealPairsArtifact }

function useRealPairs() {
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  useEffect(() => {
    let active = true
    fetch(`${import.meta.env.BASE_URL}data/helpsteer2-pairs.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<RealPairsArtifact>
      })
      .then((data) => { if (active) setState({ status: 'ready', data }) })
      .catch((error: Error) => { if (active) setState({ status: 'error', message: error.message }) })
    return () => { active = false }
  }, [])
  return state
}

function ResponseCard({ data, pair, response }: { data: RealPairsArtifact; pair: RealPair; response: ResponseNumber }) {
  const pick = pair.selectors.pareto.pick
  const tone = pick === null ? 'neutral' : pick === response ? 'chosen' : 'rejected'
  const ratings = pair.ratings[response - 1]
  return (
    <article className={`pair-choice pair-choice--${tone} real-response`}>
      <header><span>Response {response}</span><strong>{pick === null ? 'Pareto abstains' : pick === response ? 'Pareto keeps this one' : 'Pareto rejects this one'}</strong></header>
      <dl className="rating-grid" aria-label={`HelpSteer2 ratings for response ${response}, 0 to 4`}>
        {data.attributes.map((name) => (
          <div key={name} className={data.paretoAttributes.includes(name) ? 'is-pareto' : undefined}><dt>{name}</dt><dd>{ratings[name]}</dd></div>
        ))}
      </dl>
      <blockquote tabIndex={0}>{pair.responses[response - 1]}</blockquote>
    </article>
  )
}

function Decisions({ data, pair }: { data: RealPairsArtifact; pair: RealPair }) {
  const { pareto, gap, human } = pair.selectors
  const helpfulness = pair.ratings.map((ratings) => ratings.helpfulness)
  return (
    <div className="table-wrap">
      <table className="data-table decision-table">
        <caption>What each selector does with this pair. Each trained on {integer.format(data.pool.trainingPairsPerSelector)} pairs from the same pool of {integer.format(data.pool.eligiblePairs)}.</caption>
        <thead><tr><th scope="col">Selector</th><th scope="col">Picks</th><th scope="col">Why</th><th scope="col">In its training set</th></tr></thead>
        <tbody>
          <tr><th scope="row">Pareto, {data.paretoAttributes.join(' + ')}</th><td>{pareto.pick === null ? 'Abstains' : `Response ${pareto.pick}`}</td><td>{pareto.reason}</td><td>{pareto.inTrainingSet ? 'Yes' : 'No'}</td></tr>
          <tr><th scope="row">Largest helpfulness gap</th><td>Response {gap.pick}</td><td>Helpfulness {helpfulness[0]} vs {helpfulness[1]}, gap {Math.abs(gap.helpfulnessGap)}</td><td>{gap.inTrainingSet ? 'Yes' : 'No, outside the largest gaps'}</td></tr>
          <tr><th scope="row">Human label</th><td>Response {human.pick}</td><td>Preference strength {pair.preferenceStrength > 0 ? '+' : '−'}{Math.abs(pair.preferenceStrength)} on a −3 to +3 scale</td><td>{human.inTrainingSet ? 'Yes' : 'No'}</td></tr>
        </tbody>
      </table>
    </div>
  )
}

export function RealPairs() {
  const state = useRealPairs()
  const [filter, setFilter] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const pairs = useMemo(() => (state.status === 'ready' ? state.data.pairs : []), [state])
  const visible = useMemo(() => pairs.filter(FILTERS[filter].keep), [pairs, filter])

  if (state.status === 'loading') return <div className="view-stack"><p className="empty-inline">Loading HelpSteer2 pairs…</p></div>
  if (state.status === 'error') return <div className="view-stack"><div className="empty-state"><CircleAlert aria-hidden="true" /><h2>Could not load the pairs</h2><p>{state.message}</p></div></div>

  const { data } = state
  const selected = visible.find((pair) => pair.id === selectedId) ?? visible[0]
  const detailOpen = Boolean(selectedId && visible.some((pair) => pair.id === selectedId))
  const abstains = pairs.filter((pair) => pair.selectors.pareto.pick === null).length
  const counts = Object.fromEntries(data.sample.map((entry) => [entry.name, entry.count])) as Record<RealPairSample, number>

  return (
    <div className="view-stack">
      <header className="view-hero">
        <div>
          <span className="eyebrow">HelpSteer2 / real data</span>
          <h1>Training pairs from the experiment</h1>
          <p>{pairs.length} HelpSteer2 prompts with two responses each, the annotators' 0–4 ratings and preference, and what each of the experiment's three selectors did with the pair. The Pareto rule keeps a pair when one response is at least as good on correctness and coherence and better on one of them.</p>
          <p className="real-sample-note">The first {counts['hash-order']} open the experiment's {integer.format(data.pool.eligiblePairs)}-pair pool in hash order. Pareto passes {integer.format(data.pool.paretoPasses)} of the {integer.format(data.pool.eligiblePairs)}, so its rare cases are added from further down the same order: {counts['pareto-abstains']} abstentions and {counts['pareto-against-human']} pairs where it disagrees with the annotators.</p>
        </div>
        <dl className="verdict-counts"><div><dt>Pareto keeps</dt><dd>{pairs.length - abstains}</dd></div><div><dt>Pareto abstains</dt><dd>{abstains}</dd></div></dl>
      </header>
      <section className="filter-bar real-filter-bar" aria-label="HelpSteer2 pair filters">
        <label><span>Show</span><select value={filter} onChange={(event) => { setFilter(event.target.value); setSelectedId(null) }}>{Object.entries(FILTERS).map(([value, { label, keep }]) => <option key={value} value={value}>{label} ({pairs.filter(keep).length})</option>)}</select></label>
        <p className="filter-result" aria-live="polite"><strong>{visible.length}</strong> pairs</p>
      </section>
      {selected ? (
        <div className={`pair-workbench ${detailOpen ? 'pair-workbench--detail' : 'pair-workbench--register'}`}>
          <aside className="pair-index pair-index-panel" aria-label="HelpSteer2 pairs">
            <span className="eyebrow">Pairs</span>
            {visible.map((pair) => {
              const isSelected = pair.id === selected.id
              const pick = pair.selectors.pareto.pick
              return (
                <button type="button" key={pair.id} data-pair-id={pair.id} className={isSelected ? 'is-selected' : undefined} aria-pressed={isSelected} onClick={() => { flushSync(() => setSelectedId(pair.id)); heading.current?.focus() }}>
                  <span><i className={`verdict-mark verdict-mark--${pick === null ? 'ambiguous' : 'defended'}`} />{pair.id.replace('helpsteer2:', '#')}</span>
                  <small>{pick === null ? 'Pareto abstains' : `Pareto: response ${pick}`}{pair.selectors.gap.inTrainingSet ? ' · gap set' : ''}</small>
                </button>
              )
            })}
          </aside>
          <section className="pair-evidence pair-evidence-panel" aria-labelledby="real-pair-title">
            <button type="button" className="pair-back" onClick={() => { const id = selected.id; flushSync(() => setSelectedId(null)); document.querySelector<HTMLButtonElement>(`[data-pair-id="${id}"]`)?.focus() }}><ArrowLeft size={15} aria-hidden="true" />Back to pairs</button>
            <header className="pair-evidence-header">
              <div>
                <span className={`verdict-label verdict-label--${selected.selectors.pareto.pick === null ? 'ambiguous' : 'defended'}`}>{selected.selectors.pareto.pick === null ? <CircleAlert size={14} aria-hidden="true" /> : <CheckCircle2 size={14} aria-hidden="true" />}{selected.selectors.pareto.pick === null ? 'Pareto abstains' : `Pareto keeps response ${selected.selectors.pareto.pick}`}</span>
                <h2 id="real-pair-title" ref={heading} tabIndex={-1}>{selected.id}</h2>
                <blockquote className="real-prompt" tabIndex={0}>{selected.prompt}</blockquote>
              </div>
              <dl><div><dt>Position in hash order</dt><dd>{selected.hashRank + 1} of {integer.format(data.pool.eligiblePairs)}</dd></div><div><dt>Why it is shown</dt><dd>{SAMPLE_LABELS[selected.sample]}</dd></div></dl>
            </header>
            <Decisions data={data} pair={selected} />
            <div className="pair-comparison"><ResponseCard data={data} pair={selected} response={1} /><ResponseCard data={data} pair={selected} response={2} /></div>
          </section>
        </div>
      ) : null}
      <p className="real-credit">Data: <a href={`https://huggingface.co/datasets/${data.dataset.huggingFace}`}>{data.dataset.name}</a>, {data.dataset.attribution}, <a href={data.dataset.licenseUrl}>{data.dataset.license}</a>. Texts and ratings are unchanged; the file is <code>public/data/helpsteer2-pairs.json</code>, written by <code>tools/export_demo_pairs.py</code>.</p>
    </div>
  )
}
