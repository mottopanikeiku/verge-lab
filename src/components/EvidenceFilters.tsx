import { Search } from 'lucide-react'
import type { LabFilters, RunArtifact } from '../types'

type EvidenceFiltersProps = {
  run: RunArtifact
  filters: LabFilters
  onChange: (filters: LabFilters) => void
  resultCount: number
  resultLabel: string
}

export function EvidenceFilters({ run, filters, onChange, resultCount, resultLabel }: EvidenceFiltersProps) {
  const domains = [...new Set(run.prompts.map((prompt) => prompt.domain))]

  return (
    <section className="filter-bar" aria-label="Illustrative example filters">
      <label className="search-field">
        <span>Example candidate or evidence</span>
        <span className="input-shell"><Search size={15} aria-hidden="true" /><input type="search" value={filters.candidate} placeholder="Search ID or output" onChange={(event) => onChange({ ...filters, candidate: event.target.value })} /></span>
      </label>
      <label><span>Domain</span><select value={filters.domain} onChange={(event) => onChange({ ...filters, domain: event.target.value })}><option value="all">All domains</option>{domains.map((domain) => <option value={domain} key={domain}>{domain}</option>)}</select></label>
      <label><span>Example verdict</span><select value={filters.verdict} onChange={(event) => onChange({ ...filters, verdict: event.target.value as LabFilters['verdict'] })}><option value="all">All verdicts</option><option value="defended">Defended</option><option value="ambiguous">Ambiguous</option></select></label>
      <p className="filter-result" aria-live="polite"><strong>{resultCount}</strong> illustrative {resultLabel}</p>
    </section>
  )
}
