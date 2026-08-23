export type ViewId = 'overview' | 'topology' | 'pairs' | 'runbook'
export type PairVerdict = 'defended' | 'ambiguous'

export type Score = {
  value: number
  confidence: number
  evidence: string
}

export type Objective = {
  id: string
  label: string
  description: string
  direction: 'maximize' | 'minimize'
  color: string
  mean: number
  delta: number
}

export type Prompt = {
  id: string
  text: string
  domain: string
  specification: {
    rubrics: string[]
    constraints: string[]
  }
}

export type Candidate = {
  id: string
  promptId: string
  output: string
  tokens: number
  latencyMs: number
  scores: Record<string, Score>
  embedding: {
    x: number
    y: number
    z: number
  }
}

export type Pair = {
  id: string
  promptId: string
  chosenId: string
  rejectedId: string
  verdict: PairVerdict
  confidence: number
  margins: Record<string, number>
  reason: string
}

export type Mutation = {
  id: string
  sourceCandidateId: string
  kind: string
  output: string
  scoreDelta: Record<string, number>
  flipped: boolean
  reason: string
}

export type Checkpoint = {
  step: number
  label: string
  trainLoss: number
  evalReward: number
  defendedWinRate: number
  gpuMinutes: number
}

export type RunArtifact = {
  schemaVersion: 1
  id: string
  name: string
  createdAt: string
  status: string
  model: {
    base: string
    method: string
    adapter: string
    parameterCount: number
    quantization: string
  }
  summary: {
    promptCount: number
    candidateCount: number
    defendedPairCount: number
    ambiguousPairCount: number
    mutationFlipCount: number
    estimatedGpuMinutes: number
    estimatedCostUsd: number
  }
  objectives: Objective[]
  prompts: Prompt[]
  candidates: Candidate[]
  pairs: Pair[]
  mutations: Mutation[]
  checkpoints: Checkpoint[]
}

export type LabFilters = {
  candidate: string
  domain: string
  verdict: 'all' | PairVerdict
}
