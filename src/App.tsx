import { useState } from 'react'
import { flushSync } from 'react-dom'
import type { ComponentType } from 'react'
import { Activity, BookOpen, ChevronRight, Menu, Network, TestTubes, X } from 'lucide-react'
import runArtifact from '../artifacts/demo-run.json'
import './App.css'
import { Overview, PairLab, RewardTopology, Runbook } from './views/LabViews'
import type { LabFilters, RunArtifact, ViewId } from './types'

const run = runArtifact as RunArtifact

const NAV_ITEMS: Array<{ id: ViewId; label: string; index: string; icon: ComponentType<{ size?: number; 'aria-hidden'?: boolean }> }> = [
  { id: 'overview', label: 'Overview', index: '01', icon: Activity },
  { id: 'topology', label: 'Reward topology', index: '02', icon: Network },
  { id: 'pairs', label: 'Pair lab', index: '03', icon: TestTubes },
  { id: 'runbook', label: 'Runbook', index: '04', icon: BookOpen },
]

function App() {
  const [activeView, setActiveView] = useState<ViewId>('overview')
  const [railOpen, setRailOpen] = useState(true)
  const [filters, setFilters] = useState<LabFilters>({ candidate: '', domain: 'all', verdict: 'all' })
  const activeLabel = NAV_ITEMS.find((item) => item.id === activeView)?.label

  const navigate = (view: ViewId) => {
    const commitNavigation = () => {
      setActiveView(view)
      document.getElementById('lab-main')?.focus()
    }
    const transitionDocument = document as Document & {
      startViewTransition?: (updateCallback: () => void) => unknown
    }

    if (
      typeof transitionDocument.startViewTransition === 'function'
      && !window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      transitionDocument.startViewTransition(() => {
        flushSync(() => setActiveView(view))
        document.getElementById('lab-main')?.focus()
      })
      return
    }

    commitNavigation()
  }

  return (
    <div className={`app-shell${railOpen ? '' : ' app-shell--rail-closed'}`}>
      <a className="skip-link" href="#lab-main">Skip to workbench</a>
      <aside className="side-rail" aria-label="Verge Lab navigation">
        <header className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div><strong>VERGE / LAB</strong><small>Preference calibration</small></div>
        </header>
        <nav className="primary-nav" aria-label="Workbench views">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            return <button type="button" key={item.id} className={activeView === item.id ? 'is-active' : undefined} aria-current={activeView === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><span className="nav-index">{item.index}</span><Icon size={17} aria-hidden={true} /><span>{item.label}</span><ChevronRight className="nav-chevron" size={15} aria-hidden="true" /></button>
          })}
        </nav>
        <footer className="rail-footer"><div><span className="status-dot" />{run.status}</div><dl><div><dt>Run</dt><dd>{run.id}</dd></div><div><dt>Schema</dt><dd>v{run.schemaVersion}</dd></div></dl></footer>
      </aside>

      <header className="mobile-header">
        <div className="brand-lockup"><div className="brand-mark" aria-hidden="true"><span /><span /><span /></div><div><strong>VERGE / LAB</strong><small>{activeLabel}</small></div></div>
        <button type="button" aria-label={railOpen ? 'Collapse navigation rail' : 'Expand navigation rail'} aria-expanded={railOpen} onClick={() => setRailOpen((open) => !open)}>{railOpen ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}</button>
      </header>

      <main id="lab-main" tabIndex={-1}>
        <div className="bench-content">
          {activeView === 'overview' ? <Overview run={run} onNavigate={navigate} /> : null}
          {activeView === 'topology' ? <RewardTopology run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'pairs' ? <PairLab run={run} filters={filters} onFiltersChange={setFilters} /> : null}
          {activeView === 'runbook' ? <Runbook run={run} /> : null}
        </div>
      </main>

      <nav className="bottom-nav" aria-label="Mobile workbench views">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return <button type="button" key={item.id} className={activeView === item.id ? 'is-active' : undefined} aria-current={activeView === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><Icon size={18} aria-hidden={true} /><span>{item.label === 'Reward topology' ? 'Topology' : item.label}</span></button>
        })}
      </nav>
    </div>
  )
}

export default App
