"""Defended-only DPO JSONL export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import RunArtifact


def dpo_records(artifact: RunArtifact) -> tuple[dict[str, Any], ...]:
    prompts = {item.id: item for item in artifact.prompts}
    candidates = {item.id: item for item in artifact.candidates}
    records = []
    for pair in sorted(artifact.pairs, key=lambda item: (item.prompt_id, item.id)):
        if pair.verdict != "defended":
            continue
        prompt = prompts[pair.prompt_id]
        chosen = candidates[pair.chosen_id]
        rejected = candidates[pair.rejected_id]
        records.append(
            {
                "prompt": prompt.text,
                "chosen": chosen.output,
                "rejected": rejected.output,
                "metadata": {
                    "artifactId": artifact.id,
                    "pairId": pair.id,
                    "promptId": pair.prompt_id,
                    "chosenId": pair.chosen_id,
                    "rejectedId": pair.rejected_id,
                    "confidence": pair.confidence,
                    "margins": {key: pair.margins[key] for key in sorted(pair.margins)},
                    "evidence": {
                        objective.id: {
                            "chosen": chosen.scores[objective.id].evidence,
                            "rejected": rejected.scores[objective.id].evidence,
                        }
                        for objective in sorted(artifact.objectives, key=lambda item: item.id)
                        if objective.id in chosen.scores and objective.id in rejected.scores
                    },
                },
            }
        )
    return tuple(records)


def export_dpo_jsonl(artifact: RunArtifact, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
        for record in dpo_records(artifact)
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination
