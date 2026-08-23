import pytest

from verge_lab.cli import main
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
