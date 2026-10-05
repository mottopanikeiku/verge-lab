#!/usr/bin/env python3
"""Mine stored UltraFeedback scores with the existing Pareto miner (no models).

Offline: nice -n 19 uv run --frozen --no-sync python tools/mine_public_preferences.py
Refresh the pinned first 200 rows: add --fetch. No third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import urllib.request
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verge_lab.models import Candidate, Embedding, Objective, Score  # noqa: E402
from verge_lab.pareto import compare_candidates, mine_pareto_edges  # noqa: E402

REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
DATASET = "openbmb/UltraFeedback"
FILE = "truthful_qa.jsonl"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}"
SOURCE_URL = f"{BASE_URL}/resolve/{REVISION}/{FILE}"
CARD_URL = f"{BASE_URL}/raw/{REVISION}/README.md"
ASPECTS = ("helpfulness", "honesty", "instruction_following", "truthfulness")
PROMPTS = 200
EXCERPT_CHARS = 240
MAX_LINE_BYTES = 2 * 1024 * 1024
RESULTS = ROOT / "results" / "public-preferences"
OBJECTIVES = tuple(
    Objective(
        aspect, aspect.replace("_", " "), "Unverified GPT-4 dataset rating", "maximize", "#666666"
    )
    for aspect in ASPECTS
)
CONFIGS = (
    ("nominal_point_scores", 1.0, 0.0),
    ("confidence_1_penalty_scale_0.05", 1.0, 0.05),
    ("confidence_0.9_no_penalty", 0.9, 0.0),
    ("confidence_0.9_penalty_scale_0.05", 0.9, 0.05),
    ("confidence_0.8_penalty_scale_0.05", 0.8, 0.05),
    ("confidence_0.8_penalty_scale_1", 0.8, 1.0),
    ("confidence_0.5_below_threshold", 0.5, 0.05),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compact_row(row: dict, row_index: int, raw: bytes) -> dict:
    instruction = row["instruction"]
    prompt_id = f"truthful_qa:{row_index:04d}"
    completions = []
    for index, completion in enumerate(row["completions"]):
        annotations = {}
        for aspect in ASPECTS:
            entries = completion.get("annotations", {}).get(aspect)
            if isinstance(entries, dict):
                entries = [entries]
            annotations[aspect] = [
                {
                    "rating": entry.get("Rating"),
                    "rationale_excerpt": str(
                        entry.get("Rationale For Rating", entry.get("Rationale", ""))
                    )[:EXCERPT_CHARS],
                }
                for entry in (entries or [])
            ]
        response = completion["response"]
        completions.append(
            {
                "id": f"{prompt_id}:completion:{index}",
                "completion_index": index,
                "model": completion["model"],
                "principle": completion.get("principle"),
                "response_excerpt": response[:EXCERPT_CHARS],
                "response_chars": len(response),
                "response_sha256": sha256(response.encode("utf-8")),
                "annotations": annotations,
                "overall_score": completion.get("overall_score"),
                "fine_grained_score": completion.get("fine-grained_score"),
                "critique_excerpt": str(completion.get("critique", ""))[:EXCERPT_CHARS],
            }
        )
    return {
        "id": prompt_id,
        "source_row_index": row_index,
        "source_id": row.get("id"),
        "source": row["source"],
        "source_line_sha256": sha256(raw),
        "instruction_excerpt": instruction[:EXCERPT_CHARS],
        "instruction_chars": len(instruction),
        "instruction_sha256": sha256(instruction.encode("utf-8")),
        "completions": completions,
    }


def fetch_source(directory: Path) -> None:
    """Consume only a bounded prefix; do not download the full dataset file."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    source_bytes = 0
    rows = []
    with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
        for index in range(PROMPTS):
            raw = response.readline(MAX_LINE_BYTES + 1)
            if not raw or len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError(f"Missing or oversized JSONL row {index}; no source written")
            digest.update(raw)
            source_bytes += len(raw)
            rows.append(compact_row(json.loads(raw), index, raw))
    with urllib.request.urlopen(CARD_URL, timeout=90) as response:
        card_bytes = response.read(64 * 1024 + 1)
    if len(card_bytes) > 64 * 1024:
        raise ValueError("Dataset card exceeds 64 KiB")
    card = card_bytes.decode("utf-8")
    if "license: mit" not in card:
        raise ValueError("Pinned dataset card does not declare expected MIT license")
    source = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    (directory / "source.jsonl").write_bytes(source)
    (directory / "source-card.txt").write_bytes(card_bytes)
    save_json(
        directory / "source-metadata.json",
        {
            "dataset": DATASET,
            "dataset_url": BASE_URL,
            "revision": REVISION,
            "source_url": SOURCE_URL,
            "source_file": FILE,
            "selection": (
                "First 200 physical JSONL rows, zero-based indices 0..199; "
                "no filtering or shuffle"
            ),
            "prompt_count": PROMPTS,
            "source_prefix_bytes_consumed": source_bytes,
            "source_prefix_sha256": digest.hexdigest(),
            "stored_source_sha256": sha256(source),
            "hash_definitions": {
                "prefix": "Concatenated exact downloaded first 200 JSONL lines, including newlines",
                "line": "Exact downloaded JSONL line, including newline",
                "instruction_and_response": "UTF-8 full original text (not excerpt)",
            },
            "card_url": CARD_URL,
            "card_sha256": sha256(card_bytes),
            "license": (
                "MIT (dataset card declaration; upstream source datasets "
                "may have additional terms)"
            ),
            "license_card_excerpt": card.split("---", 2)[1].strip(),
            "attribution": (
                "UltraFeedback, Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, "
                "Yuan Ni, Guotong Xie, Zhiyuan Liu and Maosong Sun (2023 dataset card); "
                "prompts from TruthfulQA, Stephanie Lin, Jacob Hilton and Owain Evans (2022)."
            ),
            "paper_url": "https://arxiv.org/html/2310.01377v2",
            "truthfulqa_url": "https://github.com/sylinrl/TruthfulQA",
            "judge_card_excerpt": (
                "We then ask GPT-4 to annotate the collected samples based on the instructions."
            ),
            "limitation_card_excerpt": (
                "GPT-4 also makes mistakes and provides inaccurate feedbacks."
            ),
            "overall_score_provenance": (
                "Separate overall critique-generation rating, not the mean fine-grained rating "
                "(paper sections 2.4, 2.6 and 3.1). Independent score field, "
                "NOT an independent judge or verified ground truth."
            ),
            "excerpt_chars": EXCERPT_CHARS,
            "omitted": [
                "Full responses",
                "Full critiques and rationales",
                "System prompts",
                "Reference answers",
            ],
        },
    )


def aspect_rating(completion: dict, aspect: str) -> float | None:
    entries = completion["annotations"].get(aspect, [])
    ratings = [numeric(entry["rating"]) for entry in entries]
    # Do not silently average the available subset of incomplete annotations.
    if not ratings or any(value is None or not 1 <= value <= 5 for value in ratings):
        return None
    return sum(ratings) / len(ratings)


def candidates_for(rows: list[dict], confidence: float) -> list[Candidate]:
    candidates = []
    for row in rows:
        for completion in row["completions"]:
            scores = {}
            for aspect in ASPECTS:
                rating = aspect_rating(completion, aspect)
                if rating is not None:
                    scores[aspect] = Score(
                        (rating - 1) / 4,
                        confidence,
                        f"Unverified dataset Rating; row {row['source_row_index']}, "
                        f"completion {completion['completion_index']}, aspect {aspect}; "
                        "confidence is an assumption",
                    )
            # Candidate requires a nonempty map. This metadata-only score is never
            # included in OBJECTIVES; the core miner still sees all aspects missing.
            if not scores:
                scores["unavailable_metadata"] = Score(
                    0.0, 0.0, "No usable aspect ratings; not an objective or a measurement"
                )
            candidates.append(
                Candidate(
                    completion["id"],
                    row["id"],
                    completion["response_excerpt"],
                    0,
                    0.0,
                    scores,
                    Embedding(0.0, 0.0, 0.0),
                )
            )
    return candidates


def overall_relation(pair, completion_by_id: dict) -> str:
    if pair.verdict != "defended":
        return "abstained"
    chosen = numeric(completion_by_id[pair.chosen_id]["overall_score"])
    rejected = numeric(completion_by_id[pair.rejected_id]["overall_score"])
    if chosen is None or rejected is None:
        return "overall_missing"
    if chosen == rejected:
        return "overall_tie"
    return "agree" if chosen > rejected else "disagree"


def top_comparisons(
    rows: list[dict], candidates: list[Candidate], kwargs: dict
) -> tuple[dict, dict]:
    by_id = {candidate.id: candidate for candidate in candidates}
    outcomes = Counter()
    reasons = Counter()
    prompt_counts = Counter()
    membership = {}
    for row in rows:
        scored = [
            (completion, numeric(completion["overall_score"])) for completion in row["completions"]
        ]
        if any(value is None for _, value in scored):
            prompt_counts["excluded_incomplete_overall_scores"] += 1
            continue
        maximum = max(value for _, value in scored)
        top = {completion["id"] for completion, value in scored if value == maximum}
        prompt_counts["unique_top" if len(top) == 1 else "tied_top"] += 1
        if len(top) == len(scored):
            prompt_counts["all_overall_scores_tied"] += 1
        # Each tied top is compared against every strictly lower candidate, once.
        # Tied-top pairs are not given an arbitrary dataset winner.
        for left, right in combinations(row["completions"], 2):
            left_top, right_top = left["id"] in top, right["id"] in top
            if left_top == right_top:
                continue
            top_id = left["id"] if left_top else right["id"]
            pair = compare_candidates(by_id[left["id"]], by_id[right["id"]], OBJECTIVES, **kwargs)
            membership[pair.id] = top_id
            if pair.verdict == "ambiguous":
                outcomes["abstained"] += 1
                reasons[pair.reason] += 1
            else:
                outcomes["top_defended" if pair.chosen_id == top_id else "other_defended"] += 1
    return {
        "definition": (
            "All maximum-overall candidates versus every strictly lower-overall candidate; "
            "unordered pairs counted once; top-top ties excluded"
        ),
        "comparisons": sum(outcomes.values()),
        "outcomes": {
            name: outcomes[name] for name in ("top_defended", "other_defended", "abstained")
        },
        "abstention_reasons": dict(sorted(reasons.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
    }, membership


def analyze(directory: Path) -> dict:
    stored = (directory / "source.jsonl").read_bytes()
    metadata = json.loads((directory / "source-metadata.json").read_text(encoding="utf-8"))
    if sha256(stored) != metadata["stored_source_sha256"]:
        raise ValueError("Stored source hash does not match metadata")
    rows = [json.loads(line) for line in stored.splitlines()]
    completion_by_id = {
        completion["id"]: completion for row in rows for completion in row["completions"]
    }
    rows_by_id = {row["id"]: row for row in rows}
    sensitivity = []
    baseline_pairs = None
    baseline_top = None
    for name, confidence, scale in CONFIGS:
        candidates = candidates_for(rows, confidence)
        kwargs = {"epsilon": 0.01, "min_confidence": 0.8, "uncertainty_scale": scale}
        pairs = mine_pareto_edges(candidates, OBJECTIVES, **kwargs)
        verdicts = Counter(pair.verdict for pair in pairs)
        reasons = Counter(pair.reason for pair in pairs if pair.verdict == "ambiguous")
        agreement = Counter(
            overall_relation(pair, completion_by_id) for pair in pairs if pair.verdict == "defended"
        )
        top, membership = top_comparisons(rows, candidates, kwargs)
        comparable = agreement["agree"] + agreement["disagree"]
        sensitivity.append(
            {
                "name": name,
                "assumed_confidence": confidence,
                **kwargs,
                "per_pair_uncertainty_penalty": round(2 * scale * (1 - confidence), 6),
                "all_pairs": len(pairs),
                "defended": verdicts["defended"],
                "abstained": verdicts["ambiguous"],
                "abstention_reasons": dict(sorted(reasons.items())),
                "defended_vs_overall": {
                    key: agreement[key]
                    for key in ("agree", "disagree", "overall_tie", "overall_missing")
                },
                "comparable_defended": comparable,
                "disagreement_fraction_of_comparable_defended": agreement["disagree"] / comparable
                if comparable
                else None,
                "dataset_top_vs_other": top,
            }
        )
        if baseline_pairs is None:
            baseline_pairs, baseline_top = pairs, membership
    pair_rows = []
    examples = {}
    for pair in baseline_pairs:
        first = completion_by_id[pair.chosen_id]
        second = completion_by_id[pair.rejected_id]
        relation = overall_relation(pair, completion_by_id)
        record = {
            "pair_id": pair.id,
            "prompt_id": pair.prompt_id,
            "candidate_a": pair.chosen_id,
            "candidate_b": pair.rejected_id,
            "defended_winner": pair.chosen_id if pair.verdict == "defended" else "",
            "verdict": pair.verdict,
            "reason": pair.reason,
            "overall_a": first["overall_score"],
            "overall_b": second["overall_score"],
            "overall_relation": relation,
            "dataset_top_candidate": baseline_top.get(pair.id, ""),
            **{f"margin_{aspect}": pair.margins.get(aspect, "") for aspect in ASPECTS},
        }
        pair_rows.append(record)
        categories = []
        if pair.verdict == "defended":
            categories.append("defended")
        if pair.reason == "cross-objective tradeoff":
            categories.append("tradeoff_abstention")
        if pair.reason == "equivalent within epsilon":
            categories.append("tie_abstention")
        if relation == "disagree":
            categories.append("defended_disagreement")
        if relation == "overall_tie":
            categories.append("defended_overall_tie")
        if pair.reason.startswith("missing scores"):
            categories.append("missing_aspect_abstention")
        for category in categories:
            if category not in examples:
                examples[category] = {
                    "pair": record,
                    "prompt": {
                        key: value
                        for key, value in rows_by_id[pair.prompt_id].items()
                        if key != "completions"
                    },
                    "candidates": [first, second],
                    "raw_aspect_ratings": {
                        completion["id"]: {
                            aspect: aspect_rating(completion, aspect) for aspect in ASPECTS
                        }
                        for completion in (first, second)
                    },
                }
    with (directory / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(pair_rows)
    for category in (
        "defended",
        "tradeoff_abstention",
        "tie_abstention",
        "defended_disagreement",
        "defended_overall_tie",
        "missing_aspect_abstention",
    ):
        examples.setdefault(
            category,
            {
                "observed": False,
                "note": "No instance in the nominal sample; no synthetic example substituted",
            },
        )
    save_json(directory / "examples.json", examples)
    missing = Counter()
    for completion in completion_by_id.values():
        for aspect in ASPECTS:
            if aspect_rating(completion, aspect) is None:
                missing[aspect] += 1
    overall_pairs = Counter()
    for row in rows:
        for left, right in combinations(row["completions"], 2):
            a, b = numeric(left["overall_score"]), numeric(right["overall_score"])
            overall_pairs[
                "missing" if a is None or b is None else "tied" if a == b else "strictly_ranked"
            ] += 1
    summary = {
        "source": metadata,
        "prompt_count": len(rows),
        "candidate_count": len(completion_by_id),
        "missing_aspect_ratings": {aspect: missing[aspect] for aspect in ASPECTS},
        "overall_pair_rankings": dict(sorted(overall_pairs.items())),
        "baseline": sensitivity[0],
        "sensitivity": sensitivity,
        "method": {
            "core": "verge_lab.pareto.mine_pareto_edges and compare_candidates, unchanged",
            "aspect_values": (
                "Mean of source Rating entries per aspect; "
                "(rating - 1)/4 maps 1..5 to 0..1; maximize all four aspects"
            ),
            "missing_aspects": (
                "Absent, nonnumeric, nonfinite or out-of-range ratings omit the whole aspect; "
                "core abstains on any missing objective; no imputation"
            ),
            "missing_all_aspects": (
                "Non-objective metadata-only schema entry allows Candidate construction; "
                "all actual objectives remain absent and comparison abstains"
            ),
            "overall": (
                "Source overall_score, NOT fine-grained_score or an aspect aggregate; "
                "exact numeric ties excluded from agreement/disagreement denominator "
                "and reported separately"
            ),
            "abstentions": (
                "Core verdict ambiguous is reported as abstained; "
                "CSV candidate_a/b for abstentions are identifiers, not a preference"
            ),
            "confidence": (
                "Unavailable in source. All confidence values in sensitivity are "
                "MODELING ASSUMPTIONS, not estimates or verified reliability. "
                "Nominal point-score run assumes 1 with zero uncertainty penalty; "
                "no statistical coverage claim."
            ),
            "unused_schema_fields": (
                "tokens=0, latency_ms=0 and embedding=(0,0,0) are required-schema placeholders, "
                "not measurements; not used by the miner"
            ),
            "examples": "First matching pair in stable core output order, not hand-picked",
        },
        "limitations": [
            (
                "200-row deterministic prefix of TruthfulQA subset, not a random "
                "or representative sample of UltraFeedback or human preferences."
            ),
            (
                "GPT-4 annotations and overall ratings are unverified. Separate overall field "
                "is not an independent annotator, human preference or ground truth."
            ),
            (
                "Equal-interval treatment of ordinal 1..5 ratings and confidence penalties "
                "are modeling choices. Any positive penalty can remove dominance "
                "when one aspect ties."
            ),
            (
                "No model training, inference or downstream preference-quality evaluation; "
                "disagreement measures inconsistency between recorded fields only."
            ),
            (
                "Stored excerpts and hashes permit score reproduction and linkage, not full "
                "response review offline. Full text requires pinned upstream data; MIT is the "
                "dataset-card license, not a legal assessment of upstream content."
            ),
        ],
        "outputs": [
            "source.jsonl",
            "source-metadata.json",
            "source-card.txt",
            "summary.json",
            "pairs.csv",
            "examples.json",
            "run.json",
        ],
    }
    save_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Replace compact source from pinned first 200 public rows before analysis",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing stored source and receiving outputs",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    if args.fetch:
        fetch_source(args.results_dir)
    fetched = time.perf_counter()
    summary = analyze(args.results_dir)
    finished = time.perf_counter()
    run = {
        "command": " ".join(sys.orig_argv),
        "external_invocation_note": (
            "Run under nice -n 19; process niceness recorded below. "
            "No GPU, model or paid provider calls."
        ),
        "fetch_requested": args.fetch,
        "fetch_seconds": fetched - started,
        "analysis_seconds": finished - fetched,
        "total_seconds": finished - started,
        "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "niceness": os.getpriority(os.PRIO_PROCESS, 0),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "Unknown",
        "paid_compute_usd": 0,
        "resources_label": "MEAS",
        "confidence_label": "INFERENCE (assumed, not calibrated)",
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "determinism": (
            "summary.json, pairs.csv and examples.json are deterministic for stored source "
            "and unchanged core; run.json contains variable timings and environment"
        ),
    }
    save_json(args.results_dir / "run.json", run)
    print(json.dumps({"counts": summary["baseline"], "resources": run}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
