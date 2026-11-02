import { useState } from 'react'
import { flushSync } from 'react-dom'
import type { ComponentType } from 'react'
import { Activity, BookOpen, ChevronRight, Network, TestTubes } from 'lucide-react'
import runArtifact from '../artifacts/demo-run.json'
import './App.css'
import { ExportView, Overview, PairLab, RewardTopology } from './views/LabViews'
import type { LabFilters, RunArtifact, ViewId } from './types'

const run = runArtifact as RunArtifact

const NAV_ITEMS: Array<{ id: ViewId; label: string; index: string; icon: ComponentType<{ size?: number; 'aria-hidden'?: boolean }> }> = [
  { id: 'overview', label: 'Overview', index: '01', icon: Activity },
  { id: 'topology', label: 'Compare scores', index: '02', icon: Network },
  { id: 'pairs', label: 'Inspect pairs', index: '03', icon: TestTubes },
  { id: 'export', label: 'Export', index: '04', icon: BookOpen },
]

function App() {
  const [activeView, setActiveView] = useState<ViewId>('overview')
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
        <span className="mobile-demo-label">Example only</span>
      </header>

      <main id="lab-main" tabIndex={-1}>
        <aside className="example-notice" aria-labelledby="example-notice-title">
          <strong id="example-notice-title">Pair-selection demo</strong>
          <p>The scores here are made up; the selection decisions are computed from them. <a href="https://github.com/mottopanikeiku/verge-lab#readme">The DPO experiment</a> is in the README.</p>
          <details><summary>Where the examples come from</summary><p>Every view reads <code>artifacts/demo-run.json</code>, built from <code>examples/candidates.json</code>. Scores, confidences, token counts, latency and coordinates are authored inputs. The Python package computes pair decisions, margins, counts and mutation flips from them.</p></details>
        </aside>
        <div className="bench-content">
          {activeView === 'overview' ? <Overview run={run} onNavigate={navigate} /> : null}
          {activeView === 'topology' ? <RewardTopology run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'pairs' ? <PairLab run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'export' ? <ExportView run={run} /> : null}
        </div>
      </main>

      <nav className="bottom-nav" aria-label="Mobile demo views">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return <button type="button" key={item.id} className={activeView === item.id ? 'is-active' : undefined} aria-label={item.label} aria-current={activeView === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><Icon size={18} aria-hidden={true} /><span>{item.id === 'topology' ? 'Scores' : item.id === 'pairs' ? 'Pairs' : item.label}</span></button>
        })}
      </nav>
    </div>
  )
}

export default App
