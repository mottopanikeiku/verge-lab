#!/usr/bin/env python3
"""Join pinned HelpSteer2 texts and ratings, then select fixed DPO data (stdlib only).

The output contains modified CC-BY-4.0 data from NVIDIA and Scale AI, credited
in its metadata. No text is excerpted, normalized, or generated. Supply a local
cache containing the pinned gzip files; this script never downloads anything.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
SELECTION_SEED = 20261007
TRAIN_COUNT = 1024
EVALUATION_COUNT = 192
ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
CONDITIONS = ("pareto", "gap", "human")
PINNED_HASHES = {
    "preference/preference.jsonl.gz": (
        "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b"
    ),
    "train.jsonl.gz": "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
    "validation.jsonl.gz": "610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a",
}
COMPACT_SHA256 = "0ff82d149ac25b057d711a806c5278710c835a98f6dec1436d9677f0ea4b381a"
CARD_SHA256 = "835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            yield index, json.loads(line)


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def selection_hash(identifier: str) -> str:
    return text_hash(f"{SELECTION_SEED}:{identifier}")


def pareto_direction(scores: list[dict]) -> int:
    differences = [scores[1][aspect] - scores[0][aspect]
                   for aspect in ("correctness", "coherence")]
    if all(value >= 0 for value in differences) and any(value > 0 for value in differences):
        return 1
    if all(value <= 0 for value in differences) and any(value < 0 for value in differences):
        return -1
    return 0


def single_turn(*texts: str) -> bool:
    return all("<extra_id_1>" not in text for text in texts)


def load_verified_sources(
    cache: Path, compact_directory: Path,
) -> tuple[list[dict], list[dict], dict]:
    provenance = json.loads((compact_directory / "source-metadata.json").read_text())
    require(provenance["revision"] == REVISION, "Dataset revision mismatch")
    require(provenance["license"] == "CC-BY-4.0", "Dataset license mismatch")
    for name, expected in PINNED_HASHES.items():
        entry = provenance["files"][name]
        require(entry["compressed_sha256"] == expected, f"Provenance hash mismatch: {name}")
        require(file_hash(cache / name) == expected, f"Pinned source hash mismatch: {name}")
        require((cache / name).stat().st_size == entry["bytes"], f"Source size mismatch: {name}")
    compact_path = compact_directory / "source.jsonl"
    require(provenance["stored_source_sha256"] == COMPACT_SHA256,
            "Provenance compact hash mismatch")
    require(file_hash(compact_path) == COMPACT_SHA256, "Compact source hash mismatch")
    require(provenance["card_sha256"] == CARD_SHA256, "Provenance card hash mismatch")
    for card in (cache / "README.md", compact_directory / "source-card.txt"):
        require(file_hash(card) == CARD_SHA256, "Pinned dataset card hash mismatch")
        require("license: cc-by-4.0" in card.read_text(), "Missing dataset license declaration")
    with compact_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    with (compact_directory / "pairs.csv").open(newline="", encoding="utf-8") as source:
        pairs = list(csv.DictReader(source))
    return rows, pairs, provenance


def joined_pool(
    cache: Path, compact: list[dict], pairs: list[dict],
) -> tuple[list[dict], dict[str, str], set[str], dict]:
    """Check every join before filtering; keep full texts only for the common pool."""
    by_id = {row["id"]: row for row in compact}
    csv_by_id = {row["id"]: row for row in pairs}
    require(len(by_id) == len(compact), "Duplicate compact pair ID")
    require(len(csv_by_id) == len(pairs), "Duplicate CSV pair ID")
    require(by_id.keys() == csv_by_id.keys(), "Compact/CSV pair IDs differ")
    references = defaultdict(list)
    for row in compact:
        require(row["split"] in ("train", "validation"), "Invalid compact split")
        require(len(row["rating_rows"]) == len(row["scores"]) == len(row["response_sha256"]) == 2,
                "Compact pair must have two responses")
        for response_index, rating_index in enumerate(row["rating_rows"]):
            references[(row["split"], rating_index)].append((row, response_index))
    counts = Counter()
    train_prompt_hashes = set()
    validation_prompts = {}
    checked_references = set()
    for split in ("train", "validation"):
        for index, raw in read_gzip(cache / f"{split}.jsonl.gz"):
            counts[f"{split}_rating_responses"] += 1
            prompt_hash = text_hash(raw["prompt"])
            if split == "train":
                train_prompt_hashes.add(prompt_hash)
            else:
                validation_prompts[prompt_hash] = raw["prompt"]
            scores = {aspect: raw[aspect] for aspect in ATTRIBUTES}
            require(all(not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and 0 <= value <= 4
                        for value in scores.values()), f"Invalid rating: {split}:{index}")
            for row, response_index in references.get((split, index), ()):
                require(prompt_hash == row["prompt_sha256"], "Rating join prompt hash mismatch")
                require(text_hash(raw["response"]) == row["response_sha256"][response_index],
                        "Rating join response hash mismatch")
                require(scores == row["scores"][response_index], "Rating join scores mismatch")
                checked_references.add((split, index))
    require(checked_references == references.keys(), "Missing rating join rows")
    pool = []
    seen = set()
    for index, raw in read_gzip(cache / "preference/preference.jsonl.gz"):
        identifier = f"helpsteer2:{index:05d}"
        require(identifier in by_id, "Raw preference missing from compact source")
        row = by_id[identifier]
        split = {"train": "train", "val": "validation"}.get(raw["split"])
        require(split == row["split"] and raw["split"] == row["upstream_preference_split"],
                "Preference join split mismatch")
        require(index == row["preference_row"], "Preference row index mismatch")
        require(text_hash(raw["prompt"]) == row["prompt_sha256"],
                "Preference join prompt hash mismatch")
        responses = [raw["response_1"], raw["response_2"]]
        require([text_hash(response) for response in responses] == row["response_sha256"],
                "Preference join response hash mismatch")
        strength = raw["preference_strength"]
        require(not isinstance(strength, bool) and isinstance(strength, (int, float))
                and math.isfinite(strength) and -3 <= strength <= 3,
                "Invalid source preference strength")
        require(strength == row["preference_strength"], "Source preference strength mismatch")
        gap = row["scores"][1]["helpfulness"] - row["scores"][0]["helpfulness"]
        human = sign(strength)
        pareto = pareto_direction(row["scores"])
        csv_row = csv_by_id[identifier]
        require(csv_row["split"] == split, "CSV split mismatch")
        require(
            int(csv_row["human"]) == human and float(csv_row["preference_strength"]) == strength,
            "CSV source preference sign/strength mismatch",
        )
        require(float(csv_row["helpfulness_gap"]) == gap, "CSV helpfulness gap mismatch")
        require(int(csv_row["pareto"]) == pareto, "CSV point-dominance orientation mismatch")
        seen.add(identifier)
        counts[f"{split}_preference_pairs"] += 1
        if split != "train":
            continue
        if gap == 0:
            counts["excluded_helpfulness_tie"] += 1
        elif human == 0:
            counts["excluded_human_tie"] += 1
        elif not single_turn(raw["prompt"], *responses):
            counts["excluded_multiturn"] += 1
        elif len(raw["prompt"]) > 4000 or any(len(response) > 8000 for response in responses):
            counts["excluded_text_length"] += 1
        else:
            pool.append({"id": identifier, "prompt": raw["prompt"], "responses": responses,
                         "prompt_sha256": row["prompt_sha256"], "pareto": pareto,
                         "gap": gap, "human": human})
    require(seen == by_id.keys(), "Compact preference missing from raw source")
    counts["common_eligible_pairs"] = len(pool)
    counts["defended_eligible_pairs"] = sum(row["pareto"] != 0 for row in pool)
    counts["all_train_rating_prompt_hashes"] = len(train_prompt_hashes)
    counts["distinct_validation_rating_prompts"] = len(validation_prompts)
    return pool, validation_prompts, train_prompt_hashes, dict(counts)


def select_data(
    pool: list[dict], validation: dict[str, str], train_hashes: set[str],
    train_count: int = TRAIN_COUNT, evaluation_count: int = EVALUATION_COUNT,
) -> tuple[dict, list[dict], dict]:
    require(train_count > 0 and evaluation_count > 0, "Selection counts must be positive")
    require(len({row["id"] for row in pool}) == len(pool), "Duplicate eligible pair ID")
    require(all(row["gap"] != 0 and row["human"] in (-1, 1)
                and single_turn(row["prompt"], *row["responses"])
                and len(row["prompt"]) <= 4000
                and all(len(response) <= 8000 for response in row["responses"])
                and text_hash(row["prompt"]) == row["prompt_sha256"]
                and row["prompt_sha256"] in train_hashes for row in pool),
            "Common eligibility violation")
    hashed = sorted(pool, key=lambda row: selection_hash(row["id"]))
    defended = [row for row in hashed if row["pareto"] != 0]
    require(
        len(defended) >= train_count,
        f"Insufficient defended pool: {len(defended)} < {train_count}",
    )
    selections = {
        "pareto": defended[:train_count],
        "gap": sorted(
            pool, key=lambda row: (-abs(row["gap"]), selection_hash(row["id"]))
        )[:train_count],
        "human": hashed[:train_count],
    }
    train = {}
    for condition, rows in selections.items():
        require(len(rows) == train_count, f"Selection count mismatch: {condition}")
        records = []
        for row in rows:
            direction = sign(row[condition])
            require(direction in (-1, 1), f"Zero selector direction: {condition}")
            chosen_index = 2 if direction > 0 else 1
            records.append({"pair_id": row["id"], "prompt": row["prompt"],
                            "chosen": row["responses"][chosen_index - 1],
                            "rejected": row["responses"][2 - chosen_index],
                            "chosen_index": chosen_index, "prompt_sha256": row["prompt_sha256"]})
        train[condition] = records
    eligible_eval = {
        identifier: prompt for identifier, prompt in validation.items()
        if identifier not in train_hashes and len(prompt) <= 4000 and single_turn(prompt)
    }
    require(all(text_hash(prompt) == identifier for identifier, prompt in validation.items()),
            "Validation prompt hash mismatch")
    require(len(eligible_eval) >= evaluation_count,
            f"Insufficient held-out single-turn prompts: {len(eligible_eval)} < {evaluation_count}")
    evaluation = [{"prompt_id": identifier, "prompt": eligible_eval[identifier]}
                  for identifier in sorted(eligible_eval, key=selection_hash)[:evaluation_count]]
    require(len({row["prompt_id"] for row in evaluation}) == evaluation_count,
            "Evaluation prompt count mismatch")
    require(not {row["prompt_id"] for row in evaluation} & train_hashes,
            "Evaluation overlaps a train rating prompt")
    selected_ids = {
        condition: [row["id"] for row in rows] for condition, rows in selections.items()
    }
    overlap = {f"{left}_{right}": len(set(selected_ids[left]) & set(selected_ids[right]))
               for index, left in enumerate(CONDITIONS) for right in CONDITIONS[index + 1:]}
    details = {"eligible_evaluation_prompts": len(eligible_eval),
               "validation_prompts_overlapping_any_train_rating": (
                   len(set(validation) & train_hashes)
               ),
               "selected_pair_ids": selected_ids,
               "evaluation_prompt_ids": [row["prompt_id"] for row in evaluation],
               "selection_overlap_pairs": overlap,
               "selection_overlap_prompts": {
                   f"{left}_{right}": len({row["prompt_sha256"] for row in train[left]} &
                                          {row["prompt_sha256"] for row in train[right]})
                   for index, left in enumerate(CONDITIONS) for right in CONDITIONS[index + 1:]},
               "content_sha256": {condition: sha256(json_bytes(records))
                                  for condition, records in train.items()},
               "evaluation_content_sha256": sha256(json_bytes(evaluation))}
    return train, evaluation, details


def prepare(cache: Path, compact_directory: Path, output: Path, metadata_output: Path) -> dict:
    compact, pairs, provenance = load_verified_sources(cache, compact_directory)
    pool, validation, train_hashes, counts = joined_pool(cache, compact, pairs)
    for key in ("train_rating_responses", "validation_rating_responses",
                "train_preference_pairs", "validation_preference_pairs"):
        require(counts[key] == provenance["counts"][key], f"Source count mismatch: {key}")
    train, evaluation, selection = select_data(pool, validation, train_hashes)
    metadata = {
        "dataset": provenance["dataset"], "revision": REVISION,
        "attribution": provenance["attribution"], "license": provenance["license"],
        "license_url": provenance["license_url"], "papers": provenance["papers"],
        "changes": ("Full original single-turn texts joined to ratings; "
                    "deterministic subset and orientation; no excerpts."),
        "source_files": provenance["files"], "card_sha256": CARD_SHA256,
        "compact_source_sha256": COMPACT_SHA256,
        "pairs_csv_sha256": file_hash(compact_directory / "pairs.csv"),
        "counts": {**counts, "train_pairs_each": TRAIN_COUNT,
                   "evaluation_prompts": EVALUATION_COUNT},
        "selection_seed": SELECTION_SEED,
        "eligibility": ("train; nonzero helpfulness gap and human preference; no "
                        "<extra_id_1>; prompt <=4000 and each response <=8000 characters"),
        "selectors": {
            "pareto": ("correctness+coherence point dominance; ascending selection hash; "
                       "retain dominance direction"),
            "gap": ("descending absolute helpfulness gap, ascending selection hash; "
                    "helpfulness direction"),
            "human": "ascending selection hash from common pool; human label direction",
        },
        "selection_hash": ("sha256(UTF-8('20261007:' + pair_id)); "
                           "evaluation uses prompt SHA-256 as ID"),
        "evaluation": ("Distinct validation rating prompts; exact hash absent from ALL "
                       "train rating prompts; same character and single-turn prompt filter; "
                       "no reference answers"),
        "fixed_across_training_seeds": [1701, 1702, 1703],
        **selection,
    }
    payload = json_bytes({"train": train, "evaluation": evaluation, "metadata": metadata}) + b"\n"
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    require(len(compressed) <= 4 * 1024 * 1024, "Compressed training-text artifact exceeds 4 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(compressed)
    report = {**metadata, "data_content_sha256": sha256(payload),
              "data_gzip_sha256": sha256(compressed), "data_gzip_bytes": len(compressed),
              "gzip_mtime": 0}
    metadata_output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--compact-directory", type=Path,
                        default=ROOT / "results" / "human-preferences")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "day-dpo" / "data.json.gz")
    parser.add_argument("--metadata-output", type=Path,
                        default=ROOT / "results" / "day-dpo" / "data-metadata.json")
    args = parser.parse_args()
    report = prepare(args.cache, args.compact_directory, args.output, args.metadata_output)
    print(json.dumps({"counts": report["counts"], "data_gzip_sha256": report["data_gzip_sha256"],
                      "data_content_sha256": report["data_content_sha256"],
                      "data_gzip_bytes": report["data_gzip_bytes"],
                      "selection_overlap_pairs": report["selection_overlap_pairs"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
