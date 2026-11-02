#!/usr/bin/env python3
"""Analyze the fixed day DPO experiment using paired prompts and training seeds."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
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


def nice_ticks(low, high, target=5):
    span = high - low
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(size * magnitude for size in (1, 2, 5, 10) if size * magnitude >= raw)
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    return [round(index * step, 10) for index in range(first, last + 1)], step


def render_svg(summary):
    """Draw mean reward per training set, then the paired Pareto-minus-gap effect."""
    css = (
        "text{font-family:system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;"
        "font-size:14px;fill:#1f2933}"
        ".title{font-size:20px;font-weight:600}.sub{fill:#52606d}.small{font-size:12px}"
        ".axis{stroke:#9aa5b1;stroke-width:1}.grid{stroke:#e4e7eb;stroke-width:1}"
        ".bar{fill:#bcccdc}.hot{fill:#2f6f9f}.dot{fill:#102a43;stroke:#fff;stroke-width:1.5}"
        ".ci{stroke:#2f6f9f;stroke-width:4;stroke-linecap:round}"
        ".zero{stroke:#c0392b;stroke-width:1.5;stroke-dasharray:5 4}.red{fill:#c0392b}"
        "@media (prefers-color-scheme:dark){"
        "text{fill:#e4e7eb}.sub{fill:#9aa5b1}.axis{stroke:#616e7c}.grid{stroke:#323f4b}"
        ".bar{fill:#486581}.hot{fill:#6cb2eb}.dot{fill:#f0f4f8;stroke:#102a43}"
        ".ci{stroke:#6cb2eb}.zero{stroke:#ff8a80}.red{fill:#ff8a80}.bg{fill:#12181f}}"
        ".bg{fill:#fff}"
    )
    conditions = summary["conditions"]
    order = sorted(conditions, key=lambda name: conditions[name]["mean_reward"])
    labels = {"start": "Starting model", "pareto": "Pareto pairs",
              "human": "Human-label pairs", "gap": "Helpfulness-gap pairs"}
    left, right = 190, 800
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="500" '
        'viewBox="0 0 880 500" role="img" aria-labelledby="title desc">',
        '<title id="title">DPO pair selectors: mean reward and the Pareto-minus-gap effect</title>',
        '<desc id="desc">Mean reward for the starting model and for three DPO training sets, '
        'with per-seed means, and the Pareto minus gap reward difference per seed with its '
        'bootstrap 95 percent interval.</desc>',
        f'<style>{css}</style>',
        '<rect class="bg" width="880" height="500"/>',
        '<text class="title" x="24" y="34">'
        'DPO on 1,024 selected pairs: mean reward after training</text>',
        f'<text class="sub" x="24" y="58">Qwen2.5-1.5B-Instruct, {summary["design"]["prompts"]} '
        f'held-out prompts, 3 seeds per training set. Dots are seed means.</text>',
    ]
    rewards = [row["mean_reward"] for row in conditions.values()] + [
        seed["mean_reward"] for row in conditions.values() for seed in row["per_seed"]]
    lower, upper = min(0, *rewards), max(rewards)
    ticks, step = nice_ticks(lower, upper + (upper - lower) * 0.08)
    upper = max(ticks[-1], upper)

    def reward_x(value):
        return left + (value - lower) * (right - left) / (upper - lower)

    for tick in ticks:
        parts.append(f'<line class="grid" x1="{reward_x(tick):.1f}" x2="{reward_x(tick):.1f}" '
                     'y1="82" y2="238"/>')
        parts.append(f'<text class="small sub" x="{reward_x(tick):.1f}" y="256" '
                     f'text-anchor="middle">{tick:g}</text>')
    parts.append(f'<text class="small sub" x="{left}" y="274">mean reward (raw logit)</text>')
    for index, name in enumerate(order):
        y = 92 + index * 38
        row = conditions[name]
        cls = "hot" if name == "pareto" else "bar"
        parts.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end">'
                     f'{labels.get(name, name)}</text>')
        parts.append(f'<rect class="{cls}" x="{reward_x(0):.1f}" y="{y}" '
                     f'width="{reward_x(row["mean_reward"]) - reward_x(0):.1f}" height="26" '
                     'rx="3"/>')
        if name != "start":
            for seed_row in row["per_seed"]:
                parts.append(f'<circle class="dot" cx="{reward_x(seed_row["mean_reward"]):.1f}" '
                             f'cy="{y + 13}" r="4.5"/>')
        reach = max(row["mean_reward"], *(seed["mean_reward"] for seed in row["per_seed"]))
        parts.append(f'<text x="{reward_x(reach) + 14:.1f}" y="{y + 18}">'
                     f'{row["mean_reward"]:.2f}</text>')
    parts.append(f'<line class="axis" x1="{reward_x(0):.1f}" x2="{reward_x(0):.1f}" '
                 'y1="82" y2="238"/>')

    primary = summary["primary_pareto_minus_gap_reward"]
    low, high = primary["ci95"]
    effects = [row["mean"] for row in primary["per_seed"]]
    delta_low, delta_high = min(0, low, *effects), max(0, high, *effects)
    pad = max((delta_high - delta_low) * 0.1, 0.02)
    delta_low, delta_high = delta_low - pad, delta_high + pad
    delta_ticks, _ = nice_ticks(delta_low, delta_high)

    def delta_x(value):
        return left + (value - delta_low) * (right - left) / (delta_high - delta_low)

    parts.append('<text class="title" x="24" y="296" style="font-size:16px">'
                 'Pareto minus gap, same prompts and seeds</text>')
    for tick in delta_ticks:
        parts.append(f'<line class="grid" x1="{delta_x(tick):.1f}" x2="{delta_x(tick):.1f}" '
                     'y1="312" y2="416"/>')
        parts.append(f'<text class="small sub" x="{delta_x(tick):.1f}" y="434" '
                     f'text-anchor="middle">{tick:+g}</text>' if tick else
                     f'<text class="small sub" x="{delta_x(tick):.1f}" y="434" '
                     'text-anchor="middle">0</text>')
    parts.append(f'<line class="zero" x1="{delta_x(0):.1f}" x2="{delta_x(0):.1f}" '
                 'y1="308" y2="418"/>')
    parts.append(f'<text class="small red" x="{delta_x(0) + 6:.1f}" y="322">'
                 'no difference</text>')
    for index, seed_row in enumerate(primary["per_seed"]):
        y = 340 + index * 22
        parts.append(f'<text class="small" x="{left - 12}" y="{y + 4}" text-anchor="end">'
                     f'seed {seed_row["seed"]}</text>')
        parts.append(f'<circle class="dot" cx="{delta_x(seed_row["mean"]):.1f}" cy="{y}" r="4.5"/>')
    y = 404
    parts.append(f'<text x="{left - 12}" y="{y + 5}" text-anchor="end">all seeds</text>')
    parts.append(f'<line class="ci" x1="{delta_x(low):.1f}" x2="{delta_x(high):.1f}" '
                 f'y1="{y}" y2="{y}"/>')
    parts.append(f'<circle class="dot" cx="{delta_x(primary["mean"]):.1f}" cy="{y}" r="6"/>')
    verdict = {"pareto_superior": "Pareto ahead", "gap_superior": "gap ahead"}.get(
        primary["conclusion"], "interval crosses zero: no clear winner")
    parts.extend([
        f'<text x="24" y="470">{primary["mean"]:+.2f} logits, 95% CI '
        f'[{low:+.2f}, {high:+.2f}]: {verdict}.</text>',
        f'<text class="small sub" x="24" y="490">Reward: OpenAssistant DeBERTa-v3-large-v2. '
        f'{summary["bootstrap"]["draws"]:,} bootstrap draws over prompts and seeds.</text>',
        '</svg>\n',
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
