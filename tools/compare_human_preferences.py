#!/usr/bin/env python3
"""Compare existing Pareto mining with human HelpSteer2 preferences; no inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verge_lab.models import Candidate, Embedding, Objective, Score  # noqa: E402
from verge_lab.pareto import compare_candidates  # noqa: E402

REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
BASE = f"https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/{REVISION}"
ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
CONFIGS = {
    "correctness_coherence": ("correctness", "coherence"),
    "three_quality_aspects": ("helpfulness", "correctness", "coherence"),
    "five_aspects_naive_maximize": ATTRIBUTES,
}
SEED = 20261007
RESULTS = ROOT / "results" / "human-preferences"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def key(prompt: str, response: str) -> tuple[str, str]:
    return sha256(prompt.encode()), sha256(response.encode())


def download(cache: Path, filename: str) -> Path:
    target = cache / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with (
                urllib.request.urlopen(f"{BASE}/{filename}", timeout=120) as remote,
                partial.open("wb") as output,
            ):
                while block := remote.read(1024 * 1024):
                    output.write(block)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return target


def read_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            yield index, json.loads(line)


def fetch(directory: Path, cache: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    card = download(cache, "README.md").read_bytes()
    if b"license: cc-by-4.0" not in card:
        raise ValueError("Pinned card does not declare CC-BY-4.0")
    paths = {name: download(cache, name) for name in (
        "train.jsonl.gz", "validation.jsonl.gz", "preference/preference.jsonl.gz"
    )}
    ratings = {}
    counts = Counter()
    duplicates = []
    for split in ("train", "validation"):
        for index, row in read_gzip(paths[f"{split}.jsonl.gz"]):
            scores = {aspect: row[aspect] for aspect in ATTRIBUTES}
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 4 for v in scores.values()):
                raise ValueError(f"Invalid rating at {split}:{index}")
            identifier = key(row["prompt"], row["response"])
            counts[f"{split}_rating_responses"] += 1
            if identifier in ratings:
                prior = ratings[identifier]
                if prior[0] != split or prior[2] != scores:
                    raise ValueError("Conflicting duplicate exact prompt-response ratings")
                duplicates.append({"split": split, "first_row": prior[1], "duplicate_row": index})
            else:
                ratings[identifier] = (split, index, scores)
    compact = []
    unmatched = []
    for index, row in read_gzip(paths["preference/preference.jsonl.gz"]):
        split = {"train": "train", "val": "validation"}[row["split"]]
        counts[f"{split}_preference_pairs"] += 1
        identifiers = [key(row["prompt"], row[f"response_{number}"]) for number in (1, 2)]
        if any(identifier not in ratings for identifier in identifiers):
            unmatched.append({"preference_row": index, "split": split})
            continue
        values = [ratings[identifier] for identifier in identifiers]
        if any(value[0] != split for value in values):
            raise ValueError("Preference/rating split mismatch")
        strength = row["preference_strength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) \
                or not math.isfinite(strength) or not -3 <= strength <= 3:
            raise ValueError("Invalid preference strength")
        compact.append({
            "id": f"helpsteer2:{index:05d}", "preference_row": index, "split": split,
            "upstream_preference_split": row["split"],
            "prompt_sha256": identifiers[0][0],
            "response_sha256": [identifier[1] for identifier in identifiers],
            "rating_rows": [value[1] for value in values],
            "scores": [value[2] for value in values], "preference_strength": strength,
        })
        counts[f"{split}_matched_pairs"] += 1
    stored = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                      for row in compact)
    (directory / "source.jsonl").write_bytes(stored)
    (directory / "source-card.txt").write_bytes(card)
    save_json(directory / "source-metadata.json", {
        "dataset": "nvidia/HelpSteer2", "revision": REVISION,
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ("HelpSteer2 and HelpSteer2-Preference, NVIDIA and Scale AI; "
                        "Zhilin Wang et al."),
        "papers": ["https://arxiv.org/abs/2406.08673", "https://arxiv.org/abs/2410.01257"],
        "files": {name: {"url": f"{BASE}/{name}", "compressed_sha256": sha256(path.read_bytes()),
                         "bytes": path.stat().st_size} for name, path in paths.items()},
        "card_url": f"{BASE}/README.md", "card_sha256": sha256(card),
        "stored_source_sha256": sha256(stored), "counts": dict(counts), "unmatched": unmatched,
        "identical_duplicate_rating_keys": duplicates,
        "duplicate_handling": "Use first row only when exact text, split and all scores agree",
        "join": "Exact UTF-8 prompt and response SHA-256, both responses, with split equality",
        "split_names": {"preference/train": "train", "preference/val": "validation"},
        "changes": "Text omitted; retain hashes, original numeric ratings and preference strengths",
        "reference": ("Separately collected human pairwise preference_strength; "
                      "positive prefers response 2"),
        "helpsteer3": {
            "revision": "f6d145777bcbde96137596340fab89793acd1031",
            "card": ("https://huggingface.co/datasets/nvidia/HelpSteer3/blob/"
                     "f6d145777bcbde96137596340fab89793acd1031/README.md"),
            "license": "CC-BY-4.0",
            "decision": ("Not suitable for this score rule: preferences and free-text "
                         "feedback, not matched numeric multi-aspect ratings. "
                         "No scores inferred from text."),
        },
    })


def evaluate_pair(row: dict, aspects: tuple[str, ...]) -> dict:
    objectives = tuple(Objective(a, a, "Human ordinal rating", "maximize", "#666666")
                       for a in aspects)
    candidates = [Candidate(
        f"{row['id']}:{index + 1}", row["id"], "Text identified by upstream response hash",
        0, 0.0, {a: Score(value[a] / 4, 1.0, "Nominal human point score") for a in aspects},
        Embedding(0, 0, 0),
    ) for index, value in enumerate(row["scores"])]
    pair = compare_candidates(*candidates, objectives, uncertainty_scale=0.0)
    prediction = (1 if pair.chosen_id == candidates[1].id else -1) \
        if pair.verdict == "defended" else 0
    gap = row["scores"][1]["helpfulness"] - row["scores"][0]["helpfulness"]
    return {"id": row["id"], "split": row["split"], "pareto": prediction,
            "reason": pair.reason, "helpfulness_gap": gap,
            "human": sign(row["preference_strength"]),
            "preference_strength": row["preference_strength"]}


def agreement(rows: list[dict], prediction: str) -> dict:
    counts = Counter()
    for row in rows:
        guess = row[prediction] if prediction == "pareto" else sign(row["helpfulness_gap"])
        if not guess:
            counts["no_prediction"] += 1
        elif not row["human"]:
            counts["human_tie"] += 1
        else:
            counts["agree" if guess == row["human"] else "contradict"] += 1
    n = counts["agree"] + counts["contradict"]
    p = counts["agree"] / n if n else None
    interval = None
    if n:
        z = 1.959963984540054
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [center - radius, center + radius]
    return {"selected": len(rows), **{name: counts[name] for name in (
        "agree", "contradict", "human_tie", "no_prediction")},
        "strict_human_pairs": n, "strict_agreement": p, "wilson_95": interval,
        "contradiction_among_all_selected": counts["contradict"] / len(rows) if rows else None}


def gap_selection(rows: list[dict], count: int, seed: int = SEED) -> list[dict]:
    eligible = [row for row in rows if row["helpfulness_gap"] != 0]
    if count > len(eligible):
        raise ValueError("Requested baseline yield exceeds nonzero-gap eligible pool")
    return sorted(eligible, key=lambda row: (
        -abs(row["helpfulness_gap"]), sha256(f"{seed}:{row['id']}".encode())
    ))[:count]


def compare(rows: list[dict]) -> dict:
    defended = [row for row in rows if row["pareto"]]
    eligible = [row for row in rows if row["helpfulness_gap"] != 0]
    matched_rule = [row for row in eligible if row["pareto"]]
    baseline = gap_selection(eligible, len(matched_rule))
    full_baseline = (
        gap_selection(eligible, len(defended)) if len(defended) <= len(eligible) else None
    )
    full_baseline_ids = {row["id"] for row in full_baseline} if full_baseline is not None else set()
    matched_ids = {row["id"] for row in matched_rule}
    baseline_ids = {row["id"] for row in baseline}
    for row in rows:
        row["matched_pareto"] = row["id"] in matched_ids
        row["matched_gap"] = row["id"] in baseline_ids
        row["full_yield_gap"] = row["id"] in full_baseline_ids
    cutoff = min((abs(row["helpfulness_gap"]) for row in baseline), default=None)
    strict_rates = [agreement(gap_selection(eligible, len(matched_rule), SEED + offset), "gap")
                    ["strict_agreement"] for offset in range(20)]
    strict_rates = [value for value in strict_rates if value is not None]
    return {
        "all_pairs": len(rows), "defended": len(defended), "abstained": len(rows) - len(defended),
        "yield": len(defended) / len(rows) if rows else None,
        "unrestricted_pareto_vs_human": agreement(defended, "pareto"),
        "unrestricted_pareto_vs_helpfulness": dict(Counter(
            "tie" if not row["helpfulness_gap"] else
            "agree" if sign(row["helpfulness_gap"]) == row["pareto"] else "contradict"
            for row in defended)),
        "abstention_reasons": dict(Counter(row["reason"] for row in rows if not row["pareto"])),
        "full_yield_matched": {
            "feasible": full_baseline is not None,
            "yield_pairs_each": len(defended),
            "pareto": agreement(defended, "pareto"),
            "helpfulness_gap": (
                agreement(full_baseline, "gap") if full_baseline is not None else None
            ),
            "reason_if_infeasible": (
                "Pareto yield exceeds all nonzero-helpfulness pairs; no arbitrary tie direction"
                if full_baseline is None else None
            ),
        },
        "matched": {
            "eligible_nonzero_helpfulness_pairs": len(eligible),
            "excluded_helpfulness_ties": len(rows) - len(eligible),
            "yield_pairs_each": len(matched_rule),
            "yield_of_all_pairs": len(matched_rule) / len(rows) if rows else None,
            "pareto": agreement(matched_rule, "pareto"),
            "helpfulness_gap": agreement(baseline, "gap"),
            "overlap": len({r["id"] for r in matched_rule} & {r["id"] for r in baseline}),
            "gap_cutoff": cutoff,
            "eligible_at_cutoff": sum(abs(r["helpfulness_gap"]) == cutoff for r in eligible),
            "selected_at_cutoff": sum(abs(r["helpfulness_gap"]) == cutoff for r in baseline),
            "baseline_agreement_range_20_label_blind_tie_seeds":
                [min(strict_rates), max(strict_rates)] if strict_rates else None,
        },
    }


def figure(summary: dict, target: Path) -> None:
    panels = []
    for index, split in enumerate(("train", "validation")):
        result = summary["configurations"]["correctness_coherence"][split]["matched"]
        y = 110 + index * 180
        panels.append(f'<text x="24" y="{y - 22}">{split.title()}: '
                      f'{result["yield_pairs_each"]} selected by each method</text>')
        for j, (name, label, color) in enumerate((
            ("pareto", "Pareto", "#2563eb"), ("helpfulness_gap", "Helpfulness gap", "#a34b12")
        )):
            score = result[name]
            rate = score["strict_agreement"] or 0
            by = y + j * 55
            panels.append(f'<text x="24" y="{by + 20}">{label}</text>'
                          f'<rect x="190" y="{by}" width="{rate * 390:.2f}" height="28" '
                          f'fill="{color}"/>'
                          f'<text x="600" y="{by + 20}">{rate:.2%}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 470" role="img" '
           'aria-labelledby="title desc"><title id="title">'
           'Human preference agreement at equal yield</title>'
           '<desc id="desc">Correctness and coherence Pareto selection compared with '
           'helpfulness-gap selection. Human preference ties excluded from agreement '
           'percentages; selected pair counts '
           'are identical within each split.</desc><rect width="760" height="470" fill="#fff"/>'
           '<g font-family="system-ui,sans-serif" font-size="18" fill="#172033">'
           '<text x="24" y="35" font-size="23">'
           'Human preference agreement at equal pair yield</text>'
           + ''.join(panels)
           + '<text x="24" y="425" font-size="15">Same nonzero-helpfulness pool; '
           'ties broken without preference labels.</text>'
           '<text x="24" y="449" font-size="15">Agreement excludes human ties. '
           'See summary.json for all denominators.</text></g></svg>\n')
    target.write_text(svg)


def analyze(directory: Path) -> dict:
    source = (directory / "source.jsonl").read_bytes()
    metadata = json.loads((directory / "source-metadata.json").read_text())
    if sha256(source) != metadata["stored_source_sha256"]:
        raise ValueError("Compact source hash mismatch")
    rows = [json.loads(line) for line in source.splitlines()]
    configs = {}
    primary = []
    for name, aspects in CONFIGS.items():
        pairs = [evaluate_pair(row, aspects) for row in rows]
        configs[name] = {split: compare([r for r in pairs if r["split"] == split])
                         for split in ("train", "validation")}
        if name == "correctness_coherence":
            primary = pairs
    summary = {
        "dataset_revision": REVISION, "cost_usd": 0, "model_calls": 0,
        "primary_aspects": list(CONFIGS["correctness_coherence"]),
        "method": ("Existing compare_candidates; ratings / 4, confidence 1, "
                   "epsilon 0.01, no penalty"),
        "aspect_choice": ("Hold overall helpfulness out of primary rule; complexity "
                          "and verbosity describe style, not monotone quality. "
                          "Three-quality and naive-five maximization are sensitivities, "
                          "not tuned variants."),
        "matched_comparison": ("Within each split, exclude helpfulness ties from BOTH "
                               "methods. Select all defended pairs in that pool; select "
                               "the same number of largest absolute helpfulness gaps, "
                               "predicting gap sign. Human labels never select pairs. "
                               "Human ties remain selected but not in strict "
                               "agreement denominator."),
        "tie_break": f"Ascending SHA-256 of seed:pair_id; seed {SEED}; 20-seed sensitivity",
        "uncertainty": ("Wilson intervals describe binomial variation, not annotation "
                        "uncertainty; shared dataset population and correlated methods "
                        "prevent treating them as a superiority test."),
        "configurations": configs,
    }
    save_json(directory / "summary.json", summary)
    with (directory / "pairs.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(primary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(primary)
    figure(summary, directory / "human-agreement.svg")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path(".cache/helpsteer2"))
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.output, args.cache)
    summary = analyze(args.output)
    print(json.dumps(summary["configurations"]["correctness_coherence"], indent=2))


if __name__ == "__main__":
    main()
