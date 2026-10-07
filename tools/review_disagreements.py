#!/usr/bin/env python3
"""Fetch full texts for fixed UltraFeedback and HelpSteer2 disagreement samples.

UltraFeedback: nice -n 19 python tools/review_disagreements.py
HelpSteer2: add --human-cache PATH_TO_HELPSTEER2_CACHE (uses the existing gzip).
Selection is saved before compact excerpts or upstream text are read. The
UltraFeedback fetch consumes only its first 200 JSONL lines. Both modes store
only sampled pairs. AI-assisted judgments are recorded separately under
results/disagreement-review/; this helper does not regenerate the judgments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "public-preferences"
OUTPUT = ROOT / "results" / "disagreement-review"
REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
BASE = "https://huggingface.co/datasets/openbmb/UltraFeedback"
URL = f"{BASE}/resolve/{REVISION}/truthful_qa.jsonl"
PREFIX_SHA256 = "301b543f3587d2666ee1ded62304df635c79867c816f3ccc565cc9a74b17a125"
COMPACT_SHA256 = "2f97554fd89d1dd0e8546f2d5a546d79a800f53db786cc93d759ba6c58f772b8"
CARD_SHA256 = "70a22b5659215ab738b9dfd0dc6bd40d3bb040bf846b4e34955358f30625a4f7"
HUMAN_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
HUMAN_GZIP_SHA256 = "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b"
HUMAN_CARD_SHA256 = "835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b"
HUMAN_COMPACT_SHA256 = "0ff82d149ac25b057d711a806c5278710c835a98f6dec1436d9677f0ea4b381a"
SEED = 20261007
SAMPLE_SIZE = 12
PREFIX_LINES = 200
MAX_LINE_BYTES = 2 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def select_pairs(source: Path, output: Path) -> list[dict]:
    pair_bytes = (source / "pairs.csv").read_bytes()
    pairs = list(csv.DictReader(pair_bytes.decode("utf-8").splitlines()))
    eligible = sorted(
        (pair for pair in pairs if pair["overall_relation"] == "disagree"),
        key=lambda pair: pair["pair_id"],
    )
    ids = [pair["pair_id"] for pair in eligible]
    if len(ids) != len(set(ids)) or len(ids) < SAMPLE_SIZE:
        raise ValueError("Disagreement IDs must be unique and sufficient for the sample")
    selected = random.Random(SEED).sample(eligible, SAMPLE_SIZE)
    if any(pair["verdict"] != "defended" for pair in selected):
        raise ValueError("Selected disagreement is not a defended pair")
    manifest = {
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "eligible_count": len(eligible),
        "population": "Existing pairs.csv rows with overall_relation == disagree",
        "algorithm": ("random.Random(20261007).sample(sorted eligible pairs by pair_id, 12); "
                      "draw order retained"),
        "pairs_csv_sha256": sha256(pair_bytes),
        "eligible_pair_ids_sorted": ids,
        "selected_pair_ids_draw_order": [pair["pair_id"] for pair in selected],
        "selected_pairs": selected,
        "selection_timing": (
            "Written before reading compact excerpts or fetching upstream full texts"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection.json"
    encoded = json_bytes(manifest)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Existing selection differs; refusing to replace the fixed sample")
    else:
        path.write_bytes(encoded)
    return selected


def fetch_samples(source: Path, output: Path, pairs: list[dict]) -> None:
    compact_bytes = (source / "source.jsonl").read_bytes()
    if sha256(compact_bytes) != COMPACT_SHA256:
        raise ValueError("Compact source does not match the pinned existing hash")
    compact = {row["id"]: row for row in map(json.loads, compact_bytes.splitlines())}
    card = (source / "source-card.txt").read_bytes()
    if sha256(card) != CARD_SHA256 or b"license: mit" not in card:
        raise ValueError("Stored upstream card does not match the pinned MIT card")
    needed = {compact[pair["prompt_id"]]["source_row_index"] for pair in pairs}
    rows = {}
    prefix_digest = hashlib.sha256()
    consumed_bytes = 0
    with urllib.request.urlopen(URL, timeout=90) as response:
        for index in range(PREFIX_LINES):
            raw = response.readline(MAX_LINE_BYTES + 1)
            if not raw or len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError(f"Missing or oversized upstream line {index}")
            prefix_digest.update(raw)
            consumed_bytes += len(raw)
            if index in needed:
                row = json.loads(raw)
                saved = compact[f"truthful_qa:{index:04d}"]
                if sha256(raw) != saved["source_line_sha256"]:
                    raise ValueError(f"Upstream line {index} differs from compact source")
                if sha256(row["instruction"].encode("utf-8")) != saved["instruction_sha256"]:
                    raise ValueError(f"Instruction hash mismatch at line {index}")
                rows[index] = row
    if prefix_digest.hexdigest() != PREFIX_SHA256:
        raise ValueError("First 200 upstream lines do not match the pinned prefix hash")
    samples = []
    checked = []
    for pair in pairs:
        saved = compact[pair["prompt_id"]]
        row = rows[saved["source_row_index"]]
        completions = []
        for role, candidate_key in (("a", "candidate_a"), ("b", "candidate_b")):
            candidate_id = pair[candidate_key]
            original = next(item for item in saved["completions"] if item["id"] == candidate_id)
            upstream = row["completions"][original["completion_index"]]
            digest = sha256(upstream["response"].encode("utf-8"))
            if digest != original["response_sha256"]:
                raise ValueError(f"Full response hash mismatch: {candidate_id}")
            if upstream["overall_score"] != original["overall_score"]:
                raise ValueError(f"Overall score mismatch: {candidate_id}")
            for aspect, entries in original["annotations"].items():
                annotations = upstream.get("annotations", {}).get(aspect) or []
                if isinstance(annotations, dict):
                    annotations = [annotations]
                if [entry.get("Rating") for entry in annotations] != [
                    entry["rating"] for entry in entries
                ]:
                    raise ValueError(f"Aspect rating mismatch: {candidate_id}/{aspect}")
            completions.append({
                "role": role, "candidate_id": candidate_id,
                "response_sha256": digest, "upstream": upstream,
            })
            checked.append({
                "pair_id": pair["pair_id"], "candidate_id": candidate_id,
                "response_sha256": digest, "matches_compact_source": True,
            })
        samples.append({
            "pair": pair,
            "source_row_index": saved["source_row_index"],
            "source_line_sha256": saved["source_line_sha256"],
            "instruction_sha256": saved["instruction_sha256"],
            "upstream_prompt": {key: value for key, value in row.items() if key != "completions"},
            "completions": completions,
        })
    encoded = b"".join(
        (json.dumps(sample, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for sample in samples
    )
    (output / "samples.jsonl").write_bytes(encoded)
    provenance = {
        "dataset": "openbmb/UltraFeedback",
        "revision": REVISION,
        "source_url": URL,
        "source_prefix_lines_consumed": PREFIX_LINES,
        "source_prefix_bytes_consumed": consumed_bytes,
        "source_prefix_sha256": prefix_digest.hexdigest(),
        "compact_source_sha256": COMPACT_SHA256,
        "samples_sha256": sha256(encoded),
        "card_url": f"{BASE}/raw/{REVISION}/README.md",
        "card_sha256": CARD_SHA256,
        "card_local_path": "../public-preferences/source-card.txt",
        "license": (
            "UltraFeedback card declares MIT; this does not erase upstream source dataset terms"
        ),
        "attribution": ("UltraFeedback: Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, "
                        "Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu and Maosong Sun (2023). "
                        "TruthfulQA prompts: Stephanie Lin, Jacob Hilton and Owain Evans (2022)."),
        "truthfulqa_url": "https://github.com/sylinrl/TruthfulQA",
        "truthfulqa_caveat": ("These prompts originate in TruthfulQA. UltraFeedback's MIT card "
                             "is not a claim that every upstream asset has identical terms; "
                             "consult TruthfulQA's license and source notices before reuse."),
        "score_caveat": ("Original aspect ratings, full rationales and critiques are GPT-4 "
                        "feedback, not verified facts or independent human truth. Overall "
                        "scores come from a separate feedback field, not an independent judge."),
        "text_scope": ("All original prompt fields and both full completion objects per "
                       "sampled pair; no excerpt truncation"),
        "hash_definition": ("Responses and instructions: UTF-8 original full text; source "
                            "lines and prefix: exact downloaded bytes including newlines"),
        "response_hash_checks": checked,
        "matched_response_count": len(checked),
        "matched_unique_response_count": len({item["candidate_id"] for item in checked}),
        "matched_prompt_count": len(rows),
        "paid_compute_usd": 0,
    }
    (output / "provenance.json").write_bytes(json_bytes(provenance))
    print(json.dumps({
        "sampled_pairs": len(samples), "matched_responses": len(checked),
        "consumed_bytes": consumed_bytes, "samples_bytes": len(encoded),
    }, sort_keys=True))


def fetch_human_samples(cache: Path, output: Path) -> None:
    """Select before reading text; reuse the pinned local HelpSteer2 download."""
    source = ROOT / "results" / "human-preferences"
    pair_bytes = (source / "pairs.csv").read_bytes()
    columns = ("id", "split", "pareto", "reason", "helpfulness_gap", "human",
               "preference_strength", "matched_pareto", "matched_gap")
    pairs = [
        {column: pair[column] for column in columns}
        for pair in csv.DictReader(pair_bytes.decode("utf-8").splitlines())
    ]
    contradictions = sorted(
        (pair for pair in pairs if int(pair["pareto"]) != 0
         and int(pair["human"]) != 0
         and int(pair["pareto"]) == -int(pair["human"])),
        key=lambda pair: pair["id"],
    )
    validation = [pair for pair in contradictions if pair["split"] == "validation"]
    population = validation if len(validation) >= SAMPLE_SIZE else contradictions
    ids = [pair["id"] for pair in population]
    if len(ids) != len(set(ids)) or len(ids) < SAMPLE_SIZE:
        raise ValueError("Human contradiction IDs must be unique and sufficient")
    selected = random.Random(SEED).sample(population, SAMPLE_SIZE)
    manifest = {
        "seed": SEED, "sample_size": SAMPLE_SIZE, "eligible_count": len(population),
        "all_split_contradiction_count": len(contradictions),
        "validation_contradiction_count": len(validation),
        "population": "Validation-only" if population is validation else "All splits",
        "primary_rule": "Correctness and coherence only; overall helpfulness held out",
        "eligibility": "Nonzero defended Pareto sign opposite nonzero human preference sign",
        "algorithm": ("random.Random(20261007).sample(sorted eligible pairs by id, 12); "
                      "draw order retained"),
        "pairs_csv_sha256": sha256(pair_bytes),
        "eligible_pair_ids_sorted": ids,
        "selected_pair_ids_draw_order": [pair["id"] for pair in selected],
        "selected_pairs": selected,
        "selection_timing": "Written before reading compact source or full upstream texts",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "human-selection.json"
    encoded = json_bytes(manifest)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Existing human selection differs; refusing replacement")
    else:
        path.write_bytes(encoded)
    metadata = json.loads((source / "source-metadata.json").read_text(encoding="utf-8"))
    if metadata["revision"] != HUMAN_REVISION:
        raise ValueError("HelpSteer2 revision changed")
    card = (source / "source-card.txt").read_bytes()
    if sha256(card) != HUMAN_CARD_SHA256 or b"license: cc-by-4.0" not in card:
        raise ValueError("Pinned human source card changed")
    compact_bytes = (source / "source.jsonl").read_bytes()
    if (sha256(compact_bytes) != HUMAN_COMPACT_SHA256
            or sha256(compact_bytes) != metadata["stored_source_sha256"]):
        raise ValueError("Human compact source hash mismatch")
    compact = {row["id"]: row for row in map(json.loads, compact_bytes.splitlines())}
    filename = "preference/preference.jsonl.gz"
    archive = cache / filename
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    if (digest.hexdigest() != HUMAN_GZIP_SHA256
            or digest.hexdigest() != metadata["files"][filename]["compressed_sha256"]):
        raise ValueError("Cached full preference source hash mismatch")
    needed = {compact[pair["id"]]["preference_row"] for pair in selected}
    full = {}
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in needed:
                full[index] = json.loads(line)
                if len(full) == len(needed):
                    break
    samples = []
    checks = []
    for pair in selected:
        saved = compact[pair["id"]]
        upstream = full[saved["preference_row"]]
        prompt_hash = sha256(upstream["prompt"].encode("utf-8"))
        response_hashes = [sha256(upstream[f"response_{i}"].encode("utf-8")) for i in (1, 2)]
        if prompt_hash != saved["prompt_sha256"] or response_hashes != saved["response_sha256"]:
            raise ValueError(f"Full human prompt/response hash mismatch: {pair['id']}")
        split = {"train": "train", "val": "validation"}[upstream["split"]]
        if (
            split != saved["split"]
            or upstream["preference_strength"] != saved["preference_strength"]
        ):
            raise ValueError(f"Original human preference mismatch: {pair['id']}")
        checks.append({"id": pair["id"], "prompt_sha256": prompt_hash,
                       "response_sha256": response_hashes, "matches_compact_source": True})
        samples.append({"pair": pair, "compact_source": saved, "upstream": upstream})
    stored = b"".join(
        (json.dumps(sample, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for sample in samples
    )
    (output / "human-samples.jsonl").write_bytes(stored)
    provenance = {
        "dataset": metadata["dataset"], "revision": metadata["revision"],
        "source_url": metadata["files"][filename]["url"],
        "compressed_source_sha256": digest.hexdigest(),
        "compact_source_sha256": metadata["stored_source_sha256"],
        "samples_sha256": sha256(stored),
        "pairs_csv_sha256": sha256(pair_bytes),
        "license": metadata["license"], "license_url": metadata["license_url"],
        "attribution": metadata["attribution"], "papers": metadata["papers"],
        "card_url": metadata["card_url"], "card_sha256": metadata["card_sha256"],
        "source_card_local_path": "../human-preferences/source-card.txt",
        "changes": ("Selected pairs only; full prompt, both responses and all upstream "
                    "preference fields retained. Original numeric aspect ratings joined "
                    "from compact source by checked full-text hashes."),
        "preference_sign": "Positive favors response 2; negative favors response 1",
        "justification_provenance": ("The source card says preference statements and "
                                    "elaborations were post-processed from human-written "
                                    "justifications; all upstream raw preference fields "
                                    "are also retained in each sampled row."),
        "aggregation_caveat": ("The source card considers the three most similar preferences "
                              "for the overall label; original aspect scores are rounded "
                              "aggregates. Disagreement is not proof that a human annotator "
                              "was wrong."),
        "human_label_caveat": ("Dataset human judgments, not independently verified truth. "
                              "My qualitative review is AI-assisted and not an additional "
                              "human annotation."),
        "hash_checks": checks, "matched_prompt_count": len(checks),
        "matched_response_count": 2 * len(checks), "paid_compute_usd": 0,
    }
    (output / "human-provenance.json").write_bytes(json_bytes(provenance))
    print(json.dumps({"human_sampled_pairs": len(samples), "matched_responses": 2 * len(checks),
                      "samples_bytes": len(stored)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--human-cache", type=Path,
                        help="Review HelpSteer2 instead, using a pinned local gzip cache")
    args = parser.parse_args()
    if args.human_cache is not None:
        fetch_human_samples(args.human_cache, args.output_dir)
    else:
        pairs = select_pairs(args.source_dir, args.output_dir)
        fetch_samples(args.source_dir, args.output_dir, pairs)


if __name__ == "__main__":
    main()
