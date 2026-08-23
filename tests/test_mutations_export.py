import json
from dataclasses import replace

import pytest

from verge_lab.demo import build_demo_artifact
from verge_lab.export import dpo_records, export_dpo_jsonl
from verge_lab.mutations import mutate_output


def test_demo_mutation_audit_detects_real_preference_reversals() -> None:
    artifact = build_demo_artifact()
    flipped = [mutation for mutation in artifact.mutations if mutation.flipped]

    assert flipped
    assert all("reversed" in mutation.reason for mutation in flipped)
    assert all(mutation.score_delta for mutation in artifact.mutations)
    candidates = {candidate.id: candidate for candidate in artifact.candidates}
    assert all(
        mutation.output
        == mutate_output(candidates[mutation.source_candidate_id].output, mutation.kind)
        for mutation in artifact.mutations
    )


def test_dpo_export_contains_only_defended_edges_and_preserves_evidence(tmp_path) -> None:
    artifact = build_demo_artifact()
    records = dpo_records(artifact)

    assert len(records) == artifact.summary.defended_pair_count
    assert len(records) < len(artifact.pairs)
    assert all(
        record["metadata"]["pairId"]
        in {pair.id for pair in artifact.pairs if pair.verdict == "defended"}
        for record in records
    )
    assert all(record["metadata"]["evidence"] for record in records)
    assert all(record["metadata"]["margins"] for record in records)

    destination = export_dpo_jsonl(artifact, tmp_path / "train.jsonl")
    exported = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert exported == list(records)


def test_dpo_export_rejects_tampered_pair_evidence() -> None:
    artifact = build_demo_artifact()
    pair = next(pair for pair in artifact.pairs if pair.verdict == "defended")
    tampered = replace(pair, margins={**pair.margins, next(iter(pair.margins)): -999.0})
    pairs = tuple(tampered if item.id == pair.id else item for item in artifact.pairs)
    forged_artifact = replace(artifact, pairs=pairs)

    with pytest.raises(ValueError, match="recomputed preference evidence"):
        dpo_records(forged_artifact)
