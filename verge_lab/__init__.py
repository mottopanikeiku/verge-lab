"""Verifier-grounded preference mining for defensible post-training data."""

from .ids import stable_id
from .models import (
    Candidate,
    Checkpoint,
    Embedding,
    ModelInfo,
    Mutation,
    Objective,
    Pair,
    Prompt,
    PromptSpecification,
    RunArtifact,
    Score,
    Summary,
)
from .pareto import compare_candidates, mine_pareto_edges

__all__ = [
    "Candidate",
    "Checkpoint",
    "Embedding",
    "ModelInfo",
    "Mutation",
    "Objective",
    "Pair",
    "Prompt",
    "PromptSpecification",
    "RunArtifact",
    "Score",
    "Summary",
    "compare_candidates",
    "mine_pareto_edges",
    "stable_id",
    "__version__",
]
__version__ = "0.1.0"
