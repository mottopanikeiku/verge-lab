import json

from verge_lab.demo import build_demo_artifact
from verge_lab.export import dpo_records, export_dpo_jsonl


def test_demo_mutation_audit_detects_real_preference_reversals() -> None:
    artifact = build_demo_artifact()
    flipped = [mutation for mutation in artifact.mutations if mutation.flipped]

    assert flipped
    assert all("reversed" in mutation.reason for mutation in flipped)
    assert all(mutation.score_delta for mutation in artifact.mutations)


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
