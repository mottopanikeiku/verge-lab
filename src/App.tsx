import { useState } from 'react'
import { flushSync } from 'react-dom'
import type { ComponentType } from 'react'
import { Activity, BookOpen, ChevronRight, Database, Network, TestTubes } from 'lucide-react'
import runArtifact from '../artifacts/demo-run.json'
import './App.css'
import { ExportView, Overview, PairLab, RewardTopology } from './views/LabViews'
import { RealPairs } from './views/RealPairs'
import type { LabFilters, RunArtifact, ViewId } from './types'

const run = runArtifact as RunArtifact

const NAV_ITEMS: Array<{ id: ViewId; label: string; index: string; icon: ComponentType<{ size?: number; 'aria-hidden'?: boolean }> }> = [
  { id: 'real', label: 'HelpSteer2 pairs', index: '01', icon: Database },
  { id: 'overview', label: 'How the rule works', index: '02', icon: Activity },
  { id: 'topology', label: 'Compare scores', index: '03', icon: Network },
  { id: 'pairs', label: 'Inspect pairs', index: '04', icon: TestTubes },
  { id: 'export', label: 'Export', index: '05', icon: BookOpen },
]

function App() {
  const [activeView, setActiveView] = useState<ViewId>('real')
  const [filters, setFilters] = useState<LabFilters>({ candidate: '', domain: 'all', verdict: 'all' })
  const activeLabel = NAV_ITEMS.find((item) => item.id === activeView)?.label

  const navigate = (view: ViewId) => {
    flushSync(() => setActiveView(view))
    document.getElementById('lab-main')?.focus({ preventScroll: true })
    window.scrollTo({ top: 0, behavior: 'instant' })
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#lab-main">Skip to content</a>
      <aside className="side-rail" aria-label="Verge Lab navigation">
        <header className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div><strong>VERGE / LAB</strong><small>Pair-selection demo</small></div>
        </header>
        <nav className="primary-nav" aria-label="Demo views">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            return <button type="button" key={item.id} className={activeView === item.id ? 'is-active' : undefined} aria-current={activeView === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><span className="nav-index">{item.index}</span><Icon size={17} aria-hidden={true} /><span>{item.label}</span><ChevronRight className="nav-chevron" size={15} aria-hidden="true" /></button>
          })}
        </nav>
      </aside>

      <header className="mobile-header">
        <div className="brand-lockup"><div className="brand-mark" aria-hidden="true"><span /><span /><span /></div><div><strong>VERGE / LAB</strong><small>{activeLabel}</small></div></div>
        <span className="mobile-demo-label">{activeView === 'real' ? 'HelpSteer2 data' : 'Example scores'}</span>
      </header>

      <main id="lab-main" tabIndex={-1}>
        {activeView === 'real' ? (
          <aside className="example-notice" aria-labelledby="example-notice-title">
            <strong id="example-notice-title">Real pairs from the DPO experiment</strong>
            <p>These pairs, ratings and selector decisions come from the experiment's saved data. The other views use made-up scores to show the rule's confidence penalty, near-tie threshold and export. <a href="https://github.com/mottopanikeiku/verge-lab#readme">The results</a> are in the README.</p>
            <details><summary>Where the pairs come from</summary><p>This view reads <code>data/helpsteer2-pairs.json</code>, exported by <code>tools/export_demo_pairs.py</code> from <code>results/day-dpo/data.json.gz</code> and <code>results/human-preferences/source.jsonl</code>. A test checks the file against those results and the saved pair table.</p></details>
          </aside>
        ) : (
          <aside className="example-notice" aria-labelledby="example-notice-title">
            <strong id="example-notice-title">Made-up example scores</strong>
            <p>The scores in this view are made up; the selection decisions are computed from them. The first view has real HelpSteer2 pairs.</p>
            <details><summary>Where the examples come from</summary><p>Every example view reads <code>artifacts/demo-run.json</code>, built from <code>examples/candidates.json</code>. Scores, confidences, token counts, latency and coordinates are authored inputs. The Python package computes pair decisions, margins, counts and mutation flips from them.</p></details>
          </aside>
        )}
        <div className="bench-content">
          {activeView === 'real' ? <RealPairs /> : null}
          {activeView === 'overview' ? <Overview run={run} onNavigate={navigate} /> : null}
          {activeView === 'topology' ? <RewardTopology run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'pairs' ? <PairLab run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'export' ? <ExportView run={run} /> : null}
        </div>
      </main>

      <nav className="bottom-nav" aria-label="Mobile demo views">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return <button type="button" key={item.id} className={activeView === item.id ? 'is-active' : undefined} aria-label={item.label} aria-current={activeView === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><Icon size={18} aria-hidden={true} /><span>{item.id === 'real' ? 'Real' : item.id === 'overview' ? 'Rule' : item.id === 'topology' ? 'Scores' : item.id === 'pairs' ? 'Pairs' : item.label}</span></button>
        })}
      </nav>
    </div>
  )
}

export default App
