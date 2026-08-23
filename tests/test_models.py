import json

import pytest

from verge_lab.demo import build_demo_artifact
from verge_lab.models import RunArtifact, Score


def test_strict_score_rejects_unknown_fields_and_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="unknown extra"):
        Score.from_dict({"value": 1, "confidence": 1, "evidence": "ok", "extra": True})
    with pytest.raises(ValueError, match="finite"):
        Score(float("nan"), 1, "not JSON")


def test_artifact_round_trip_is_strict_and_summary_is_derived() -> None:
    artifact = build_demo_artifact()
    restored = RunArtifact.from_json(artifact.to_json())

    assert restored == artifact
    assert restored.summary.prompt_count == len(restored.prompts)
    assert restored.summary.candidate_count == len(restored.candidates)
    assert restored.summary.defended_pair_count == sum(
        pair.verdict == "defended" for pair in restored.pairs
    )
    assert restored.summary.ambiguous_pair_count == sum(
        pair.verdict == "ambiguous" for pair in restored.pairs
    )
    assert restored.summary.mutation_flip_count == sum(
        mutation.flipped for mutation in restored.mutations
    )

    raw = artifact.to_dict()
    raw["unexpected"] = "not canonical"
    with pytest.raises(ValueError, match="unknown unexpected"):
        RunArtifact.from_dict(raw)


def test_json_parser_rejects_non_standard_nan() -> None:
    with pytest.raises(ValueError, match="non-finite JSON"):
        RunArtifact.from_json(
            json.dumps(build_demo_artifact().to_dict()).replace(
                '"schemaVersion": 1', '"schemaVersion": NaN'
            )
        )


def test_artifact_rejects_invalid_created_at() -> None:
    raw = build_demo_artifact().to_dict()
    raw["createdAt"] = "not-a-date"

    with pytest.raises(ValueError, match="RFC 3339"):
        RunArtifact.from_dict(raw)
