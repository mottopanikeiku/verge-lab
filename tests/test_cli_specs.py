import json
from pathlib import Path

import pytest

from verge_lab.cli import main
from verge_lab.demo import build_artifact
from verge_lab.models import RunArtifact
from verge_lab.scorers import build_scorer
from verge_lab.specs import ScorerSpecification


def test_demo_cli_creates_canonical_artifact(tmp_path) -> None:
    destination = tmp_path / "demo.json"

    assert main(["demo", "--output", str(destination)]) == 0

    artifact = RunArtifact.read_json(destination)
    assert artifact.schema_version == 1
    assert len(artifact.prompts) >= 4
    assert len(artifact.candidates) >= 10
    assert {pair.verdict for pair in artifact.pairs} == {"defended", "ambiguous"}
    assert len(artifact.objectives) >= 3
    assert len(artifact.mutations) >= 3
    assert len(artifact.checkpoints) >= 4


def test_specifications_cannot_select_arbitrary_code() -> None:
    specification = ScorerSpecification(
        "correctness", "__import__('os').system", {"command": "false"}
    )

    with pytest.raises(ValueError, match="unknown scorer kind"):
        build_scorer(specification)


def test_analysis_rejects_objective_without_observations() -> None:
    source = Path(__file__).resolve().parent.parent / "examples" / "candidates.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    for candidate in payload["candidates"]:
        candidate["scores"].pop("clarity", None)

    with pytest.raises(ValueError, match="clarity has no observed scores"):
        build_artifact(payload)
