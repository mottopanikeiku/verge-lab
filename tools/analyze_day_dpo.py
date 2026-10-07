#!/usr/bin/env python3
"""Analyze the fixed day DPO experiment using paired prompts and training seeds."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from html import escape
from pathlib import Path
from statistics import mean, median, pstdev

CONDITIONS = ("pareto", "gap", "human")
SEEDS = (1701, 1702, 1703)
PROMPTS = 192
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261007
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"condition", "seed", "prompt_id", "reward", "response_tokens", "response",
            "generation_sha256"}


def validate_records(rows, expected_prompt_ids=None):
    """Reject anything other than one start and nine trained responses per prompt."""
    records = {}
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError(f"Record {number}: missing required fields or not an object")
        condition, seed = row["condition"], row["seed"]
        if condition not in ("start", *CONDITIONS):
            raise ValueError(f"Record {number}: invalid condition")
        if condition == "start":
            if seed is not None:
                raise ValueError(f"Record {number}: start seed must be null")
        elif type(seed) is not int or seed not in SEEDS:
            raise ValueError(f"Record {number}: invalid training seed")
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Record {number}: invalid prompt_id")
        reward = row["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(f"Record {number}: reward must be a finite number")
        try:
            finite = math.isfinite(reward)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"Record {number}: reward must be a finite number")
        if type(row["response_tokens"]) is not int or row["response_tokens"] < 0:
            raise ValueError(f"Record {number}: response_tokens must be a nonnegative integer")
        if not isinstance(row["response"], str):
            raise ValueError(f"Record {number}: response must be a string")
        digest = row["generation_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Record {number}: invalid generation_sha256")
        if digest != hashlib.sha256(row["response"].encode("utf-8")).hexdigest():
            raise ValueError(f"Record {number}: generation_sha256 does not match response")
        key = (condition, seed, prompt_id)
        if key in records:
            raise ValueError(f"Duplicate record for {key}")
        records[key] = row
    prompt_ids = {key[2] for key in records if key[0] == "start"}
    if len(prompt_ids) != PROMPTS:
        raise ValueError(f"Expected {PROMPTS} distinct start prompts, got {len(prompt_ids)}")
    if expected_prompt_ids is not None and prompt_ids != set(expected_prompt_ids):
        raise ValueError("Generation prompt IDs differ from the evaluation dataset")
    expected = {("start", None, prompt_id) for prompt_id in prompt_ids}
    expected.update((condition, seed, prompt_id) for condition in CONDITIONS
                    for seed in SEEDS for prompt_id in prompt_ids)
    missing, extra = expected - records.keys(), records.keys() - expected
    if missing or extra:
        raise ValueError(f"Unbalanced records: {len(missing)} missing, {len(extra)} unexpected")
    return records


def load_records(path: Path, expected_prompt_ids=None):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return validate_records((json.loads(line) for line in source), expected_prompt_ids)


def evaluation_prompt_ids(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        evaluation = json.load(source)["evaluation"]
    ids = [row["prompt_id"] for row in evaluation]
    if len(ids) != PROMPTS or any(not isinstance(value, str) or not value for value in ids) \
            or len(set(ids)) != PROMPTS:
        raise ValueError(f"Evaluation dataset must contain {PROMPTS} distinct prompt IDs")
    return ids


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def crossed_bootstrap(matrix, companion=None, *, draws=BOOTSTRAP_DRAWS,
                      random_seed=BOOTSTRAP_SEED):
    """Resample prompt rows and the three matched seed columns independently.

    Both matrices use the same draws. Conditions are already subtracted within
    each cell, so pairing is never broken. No resampled matrix is allocated.
    """
    if not matrix or any(len(row) != len(SEEDS) for row in matrix):
        raise ValueError("Bootstrap requires nonempty prompt rows with three matched seeds")
    if companion is not None and (len(companion) != len(matrix)
                                  or any(len(row) != len(SEEDS) for row in companion)):
        raise ValueError("Companion matrix must have the same shape")
    if type(draws) is not int or draws <= 0:
        raise ValueError("Bootstrap draw count must be positive")
    rng = random.Random(random_seed)
    randrange = rng.randrange
    count = len(matrix)
    denominator = count * len(SEEDS)
    samples, paired_samples = [], []
    for _ in range(draws):
        first, second, third = randrange(3), randrange(3), randrange(3)
        total = paired_total = 0.0
        for _ in range(count):
            prompt = randrange(count)
            row = matrix[prompt]
            total += row[first] + row[second] + row[third]
            if companion is not None:
                paired = companion[prompt]
                paired_total += paired[first] + paired[second] + paired[third]
        samples.append(total / denominator)
        if companion is not None:
            paired_samples.append(paired_total / denominator)
    return samples, paired_samples


def distribution(values):
    ordered = sorted(values)
    return {"mean": mean(values), "median": median(values), "stddev": pstdev(values),
            "min": ordered[0], "max": ordered[-1],
            "p25": percentile(ordered, 0.25), "p75": percentile(ordered, 0.75)}


def correlation(left, right):
    left_mean, right_mean = mean(left), mean(right)
    numerator = math.fsum((a - left_mean) * (b - right_mean)
                          for a, b in zip(left, right, strict=True))
    left_ss = math.fsum((a - left_mean) ** 2 for a in left)
    right_ss = math.fsum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss) * math.sqrt(right_ss)
    return numerator / denominator if denominator else None


def versus_start(rows, records):
    differences = [row["reward"] - records[("start", None, row["prompt_id"])]["reward"]
                   for row in rows]
    counts = {"win": sum(value > 0 for value in differences),
              "tie": sum(value == 0 for value in differences),
              "loss": sum(value < 0 for value in differences)}
    return {"comparisons": len(rows), "counts": counts,
            "rates": {name: count / len(rows) for name, count in counts.items()},
            "mean_reward_delta": mean(differences)}


def bootstrap_summary(matrix, samples):
    ordered = sorted(samples)
    return {"mean": mean(value for row in matrix for value in row),
            "ci95": [percentile(ordered, 0.025), percentile(ordered, 0.975)],
            "per_seed": [{"seed": seed, "mean": mean(row[index] for row in matrix)}
                         for index, seed in enumerate(SEEDS)]}


def summarize(records, *, draws=BOOTSTRAP_DRAWS, random_seed=BOOTSTRAP_SEED):
    prompt_ids = sorted(key[2] for key in records if key[0] == "start")
    conditions = {}
    for condition in ("start", *CONDITIONS):
        seeds = (None,) if condition == "start" else SEEDS
        all_rows, per_seed = [], []
        for seed in seeds:
            rows = [records[(condition, seed, prompt_id)] for prompt_id in prompt_ids]
            all_rows.extend(rows)
            per_seed.append({"seed": seed, "mean_reward": mean(row["reward"] for row in rows),
                             "response_tokens": distribution([row["response_tokens"]
                                                               for row in rows]),
                             "vs_start": None if condition == "start"
                             else versus_start(rows, records)})
        rewards = [row["reward"] for row in all_rows]
        lengths = [row["response_tokens"] for row in all_rows]
        conditions[condition] = {
            "records": len(all_rows), "mean_reward": mean(rewards),
            "reward_stddev": pstdev(rewards), "response_tokens": distribution(lengths),
            "per_seed": per_seed,
            "vs_start": None if condition == "start" else versus_start(all_rows, records),
            "reward_length_pearson": correlation(rewards, lengths),
        }
    reward_matrix, length_matrix = [], []
    for prompt_id in prompt_ids:
        reward_row, length_row = [], []
        for seed in SEEDS:
            pareto = records[("pareto", seed, prompt_id)]
            gap = records[("gap", seed, prompt_id)]
            reward_row.append(pareto["reward"] - gap["reward"])
            length_row.append(pareto["response_tokens"] - gap["response_tokens"])
        reward_matrix.append(tuple(reward_row))
        length_matrix.append(tuple(length_row))
    reward_samples, length_samples = crossed_bootstrap(
        reward_matrix, length_matrix, draws=draws, random_seed=random_seed)
    primary = bootstrap_summary(reward_matrix, reward_samples)
    low, high = primary["ci95"]
    primary["conclusion"] = ("pareto_superior" if low > 0 else
                             "gap_superior" if high < 0 else "inconclusive")
    length_difference = bootstrap_summary(length_matrix, length_samples)
    return {
        "schema_version": 1,
        "design": {"prompts": len(prompt_ids), "training_seeds": list(SEEDS),
                   "records": len(records), "prompt_ids": prompt_ids,
                   "start_generations_per_prompt": 1},
        "bootstrap": {"draws": draws, "random_seed": random_seed,
                      "method": "Crossed paired bootstrap of prompts and matched seed IDs",
                      "interval": "95% percentile; linear interpolation of sorted draws"},
        "primary_pareto_minus_gap_reward": primary,
        "pareto_minus_gap_response_tokens": length_difference,
        "conditions": conditions,
        "length_confounding": {
            "paired_delta_pearson": correlation(
                [value for row in reward_matrix for value in row],
                [value for row in length_matrix for value in row]),
            "interpretation": (
                "I report response lengths and paired length differences because the reward "
                "model may favor longer answers. These outputs are not length-matched. "
                "Correlations are descriptive, not a causal adjustment. I do not regress "
                "out length or treat the reward model as human preference truth."),
        },
        "reward_measure": {
            "model": "OpenAssistant/reward-model-deberta-v3-large-v2",
            "revision": "c355404efa9ad2ad069f3a197cae0523c14244fc",
            "units": "Raw single logit, not a probability or human preference label",
            "tokenization": "Question + answer pair, max_length=512, longest_first",
            "model_card": "https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2",
        },
    }


def render_svg(summary):
    """Draw observed seed effects, the crossed interval, and response lengths."""
    primary = summary["primary_pareto_minus_gap_reward"]
    low, high = primary["ci95"]
    effects = [row["mean"] for row in primary["per_seed"]]
    lower, upper = min(0, low, *effects), max(0, high, *effects)
    padding = max((upper - lower) * 0.12, 0.05)
    lower, upper = lower - padding, upper + padding
    left, right = 210, 810

    def x(value):
        return left + (value - lower) * (right - left) / (upper - lower)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="630" '
        'viewBox="0 0 920 630" role="img" aria-labelledby="title desc">',
        '<title id="title">Day DPO: paired reward differences and response lengths</title>',
        '<desc id="desc">Three matched training seed effects and the crossed bootstrap '
        '95 percent interval for Pareto minus gap reward, with mean response token '
        'lengths for each condition.</desc>',
        '<rect width="920" height="630" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#17212b">',
        '<text x="28" y="34" font-size="20">Pareto minus gap: raw reward difference</text>',
        f'<text x="28" y="58">{summary["design"]["prompts"]} paired prompts; '
        f'3 matched seeds; {summary["bootstrap"]["draws"]:,} crossed bootstrap draws</text>',
        f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="82" y2="255" '
        'stroke="#9aa4ae" stroke-dasharray="4 4"/>',
    ]
    for index in range(5):
        value = lower + (upper - lower) * index / 4
        position = x(value)
        parts.append(f'<text x="{position:.2f}" y="280" text-anchor="middle">'
                     f'{value:.3f}</text>')
    parts.append(f'<line x1="{left}" x2="{right}" y1="260" y2="260" stroke="#9aa4ae"/>')
    for index, row in enumerate(primary["per_seed"]):
        y = 105 + index * 40
        parts.append(f'<text x="28" y="{y + 5}">Seed {row["seed"]}</text>')
        parts.append(f'<circle cx="{x(row["mean"]):.2f}" cy="{y}" r="5" fill="#245a85"/>')
        parts.append(f'<text x="840" y="{y + 5}">{row["mean"]:+.3f}</text>')
    parts.extend([
        '<text x="28" y="235">Aggregate + 95% CI</text>',
        f'<line x1="{x(low):.2f}" x2="{x(high):.2f}" y1="230" y2="230" '
        'stroke="#245a85" stroke-width="3"/>',
        f'<circle cx="{x(primary["mean"]):.2f}" cy="230" r="6" fill="#245a85"/>',
        f'<text x="840" y="235">{primary["mean"]:+.3f}</text>',
        f'<text x="28" y="315">95% CI [{low:+.3f}, {high:+.3f}]; '
        f'{escape(primary["conclusion"].replace("_", " "))}</text>',
        '<text x="28" y="358" font-size="20">Generated response tokens</text>',
        '<text x="28" y="382">Bars: means; dots: seed means; '
        'text: median and observed range</text>',
    ])
    maximum = max(row["response_tokens"]["mean"]
                  for row in summary["conditions"].values())
    scale = 380 / max(maximum, 1)
    for index, condition in enumerate(("start", *CONDITIONS)):
        row = summary["conditions"][condition]
        lengths = row["response_tokens"]
        y = 402 + index * 36
        width = lengths["mean"] * scale
        parts.extend([
            f'<text x="28" y="{y + 17}">{condition}</text>',
            f'<rect x="{left}" y="{y}" width="{width:.2f}" height="22" fill="#d8e7f0"/>',
            f'<text x="620" y="{y + 17}">mean {lengths["mean"]:.1f}; '
            f'median {lengths["median"]:.1f}; '
            f'range {lengths["min"]}–{lengths["max"]}</text>',
        ])
        for seed_row in row["per_seed"]:
            position = left + seed_row["response_tokens"]["mean"] * scale
            parts.append(f'<circle cx="{position:.2f}" cy="{y + 11}" r="4" fill="#245a85"/>')
    length_delta = summary["pareto_minus_gap_response_tokens"]
    length_low, length_high = length_delta["ci95"]
    parts.extend([
        f'<text x="28" y="566">Paired length difference: {length_delta["mean"]:+.2f} tokens; '
        f'95% CI [{length_low:+.2f}, {length_high:+.2f}]</text>',
        '<text x="28" y="593">Reward is a model logit, not human preference truth. '
        'Length may confound the comparison.</text>',
        '</g></svg>\n',
    ])
    return "\n".join(parts)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(records_path: Path, output_dir: Path, data_path: Path):
    ids = evaluation_prompt_ids(data_path)
    records = load_records(records_path, ids)
    summary = summarize(records)
    summary["source"] = {"records_file": records_path.name,
                         "records_sha256": file_sha256(records_path),
                         "data_file": data_path.name, "data_sha256": file_sha256(data_path)}
    figure = render_svg(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    (output_dir / "reward-difference.svg").write_text(figure, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "results/day-dpo/records.jsonl.gz")
    parser.add_argument("--data", type=Path, default=ROOT / "results/day-dpo/data.json.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/day-dpo")
    args = parser.parse_args()
    summary = analyze(args.records, args.output_dir, args.data)
    print(json.dumps(summary["primary_pareto_minus_gap_reward"], sort_keys=True))


if __name__ == "__main__":
    main()
