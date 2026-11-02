#!/usr/bin/env python3
"""Export real HelpSteer2 training pairs and the experiment's selector decisions for the demo.

Reads only saved results (stdlib only):
- results/day-dpo/data.json.gz: texts of the 1,024-pair training sets and the selected IDs
- results/human-preferences/source.jsonl: HelpSteer2 ratings and preference strengths

The human-reference set is the first 1,024 eligible pairs in the experiment's hash order, so
its records are the eligible pool in that order. The export takes the first 30 of them, then
the next six where the Pareto rule abstains and the next four where it picks against the
annotators, so the rare cases are visible.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/day-dpo/data.json.gz"
SOURCE = ROOT / "results/human-preferences/source.jsonl"
OUTPUT = ROOT / "public/data/helpsteer2-pairs.json"
ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
PARETO_ATTRIBUTES = ("correctness", "coherence")
SAMPLE = (("hash-order", 30), ("pareto-abstains", 6), ("pareto-against-human", 4))


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pareto_decision(ratings: list[dict]) -> tuple[int | None, str]:
    """Same point-dominance rule as tools/prepare_day_dpo.py, with a readable reason."""
    differences = {name: ratings[1][name] - ratings[0][name] for name in PARETO_ATTRIBUTES}
    higher = {response: [name for name, value in differences.items() if sign(value) == direction]
              for response, direction in ((1, -1), (2, 1))}
    tied = [name for name, value in differences.items() if value == 0]
    if higher[1] and higher[2]:
        return None, (f"Trade-off: response 1 is higher on {' and '.join(higher[1])}, "
                      f"response 2 on {' and '.join(higher[2])}")
    if not higher[1] and not higher[2]:
        return None, f"Tied on {' and '.join(tied)}"
    pick = 1 if higher[1] else 2
    reason = f"Response {pick} is higher on {' and '.join(higher[pick])}"
    if tied:
        reason += f" and tied on {' and '.join(tied)}"
    return pick, reason


def build() -> dict:
    with gzip.open(DATA, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    with SOURCE.open(encoding="utf-8") as handle:
        sources = {row["id"]: row for row in map(json.loads, handle)}
    metadata = data["metadata"]
    selected = {name: set(ids) for name, ids in metadata["selected_pair_ids"].items()}
    pool = data["train"]["human"]
    if [record["pair_id"] for record in pool] != metadata["selected_pair_ids"]["human"]:
        raise ValueError("Human-reference records are not in selection order")

    pairs = []
    wanted = dict(SAMPLE)
    taken = {name: 0 for name in wanted}
    for rank, record in enumerate(pool):
        source = sources[record["pair_id"]]
        ratings = [{name: source["scores"][index][name] for name in ATTRIBUTES} for index in (0, 1)]
        human = 2 if source["preference_strength"] > 0 else 1
        pareto, reason = pareto_decision(ratings)
        if rank < wanted["hash-order"]:
            sample = "hash-order"
        elif pareto is None:
            sample = "pareto-abstains"
        elif pareto != human:
            sample = "pareto-against-human"
        else:
            continue
        if taken[sample] == wanted[sample]:
            continue
        taken[sample] += 1
        # Records store the human-preferred answer as chosen; restore HelpSteer2's order.
        responses = ([record["chosen"], record["rejected"]] if record["chosen_index"] == 1
                     else [record["rejected"], record["chosen"]])
        if [text_hash(text) for text in responses] != source["response_sha256"]:
            raise ValueError(f"Response text does not match its rating row: {record['pair_id']}")
        if text_hash(record["prompt"]) != source["prompt_sha256"]:
            raise ValueError(f"Prompt text does not match its rating row: {record['pair_id']}")
        gap = ratings[1]["helpfulness"] - ratings[0]["helpfulness"]
        pairs.append({
            "id": record["pair_id"],
            "sample": sample,
            "hashRank": rank,
            "prompt": record["prompt"],
            "responses": responses,
            "ratings": ratings,
            "preferenceStrength": source["preference_strength"],
            "selectors": {
                "pareto": {"pick": pareto, "reason": reason,
                           "inTrainingSet": record["pair_id"] in selected["pareto"]},
                "gap": {"pick": 2 if gap > 0 else 1, "helpfulnessGap": gap,
                        "inTrainingSet": record["pair_id"] in selected["gap"]},
                "human": {"pick": human, "inTrainingSet": record["pair_id"] in selected["human"]},
            },
        })
        if taken == wanted:
            break
    if taken != wanted:
        raise ValueError(f"Not enough pairs for the sample: {taken}")

    counts = metadata["counts"]
    return {
        "dataset": {
            "name": "HelpSteer2 and HelpSteer2-Preference",
            "huggingFace": metadata["dataset"],
            "revision": metadata["revision"],
            "license": metadata["license"],
            "licenseUrl": metadata["license_url"],
            "attribution": metadata["attribution"],
            "papers": metadata["papers"],
        },
        "pool": {
            "eligiblePairs": counts["common_eligible_pairs"],
            "paretoPasses": counts["defended_eligible_pairs"],
            "trainingPairsPerSelector": counts["train_pairs_each"],
        },
        "sample": [{"name": name, "count": count} for name, count in SAMPLE],
        "attributes": list(ATTRIBUTES),
        "paretoAttributes": list(PARETO_ATTRIBUTES),
        "pairs": pairs,
    }


def render(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="compare with the committed file")
    arguments = parser.parse_args()
    content = render(build())
    if arguments.check:
        if arguments.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{arguments.output} is stale; rerun tools/export_demo_pairs.py")
        print(f"{arguments.output} matches the saved results")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(content, encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
