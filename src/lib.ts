import type { Candidate, LabFilters, Pair, RunArtifact } from './types'

export const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value))

export const formatPercent = (value: number) => `${Math.round(value * 100)}%`

export const formatNumber = (value: number, digits = 2) =>
  new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value)

export const formatCompactNumber = (value: number) =>
  new Intl.NumberFormat('en-US', { notation: 'compact' }).format(value)

export const formatDate = (value: string) =>
  new Intl.DateTimeFormat('en-US', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'UTC',
    timeZoneName: 'short',
  }).format(new Date(value))

export const shortId = (value: string) => {
  const segments = value.split('-')
  return segments.length > 1 ? segments.slice(-2).join('-') : value.slice(0, 10)
}

export const candidateFor = (run: RunArtifact, id: string) =>
  run.candidates.find((candidate) => candidate.id === id)

export const promptFor = (run: RunArtifact, id: string) =>
  run.prompts.find((prompt) => prompt.id === id)

export const candidateLabel = (run: RunArtifact, candidate: Candidate) => {
  const promptCandidates = run.candidates.filter((item) => item.promptId === candidate.promptId)
  const position = promptCandidates.findIndex((item) => item.id === candidate.id)
  return `C${position + 1}`
}

export const pairMatchesFilters = (run: RunArtifact, pair: Pair, filters: LabFilters) => {
  const prompt = promptFor(run, pair.promptId)
  const chosen = candidateFor(run, pair.chosenId)
  const rejected = candidateFor(run, pair.rejectedId)
  const query = filters.candidate.trim().toLocaleLowerCase()
  const matchesCandidate =
    query.length === 0 ||
    [pair.id, chosen?.id, chosen?.output, rejected?.id, rejected?.output].some((value) =>
      value?.toLocaleLowerCase().includes(query),
    )

  return (
    matchesCandidate &&
    (filters.domain === 'all' || prompt?.domain === filters.domain) &&
    (filters.verdict === 'all' || pair.verdict === filters.verdict)
  )
}

export const filteredPairs = (run: RunArtifact, filters: LabFilters) =>
  run.pairs.filter((pair) => pairMatchesFilters(run, pair, filters))

export const filteredCandidates = (run: RunArtifact, filters: LabFilters) => {
  const promptIds = new Set(
    run.prompts
      .filter((prompt) => filters.domain === 'all' || prompt.domain === filters.domain)
      .map((prompt) => prompt.id),
  )
  const pairCandidateIds = new Set(
    filteredPairs(run, filters).flatMap((pair) => [pair.chosenId, pair.rejectedId]),
  )
  const query = filters.candidate.trim().toLocaleLowerCase()

  return run.candidates.filter((candidate) => {
    const matchesQuery =
      query.length === 0 ||
      candidate.id.toLocaleLowerCase().includes(query) ||
      candidate.output.toLocaleLowerCase().includes(query)
    const matchesVerdict = filters.verdict === 'all' || pairCandidateIds.has(candidate.id)
    return promptIds.has(candidate.promptId) && matchesQuery && matchesVerdict
  })
}
