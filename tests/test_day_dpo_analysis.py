"""Synthetic checks for balanced records and crossed, paired DPO analysis."""
import gzip
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_day_dpo.py"
SPEC = importlib.util.spec_from_file_location("analyze_day_dpo", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def raw_row(condition, seed, prompt_id, reward, tokens):
    response = f"Synthetic answer: {condition}, {seed}, {prompt_id}. π"
    return {"condition": condition, "seed": seed, "prompt_id": prompt_id,
            "reward": reward, "response_tokens": tokens, "response": response,
            "generation_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest()}


def balanced_rows():
    rows = []
    for index in range(analysis.PROMPTS):
        prompt_id = f"synthetic-{index:03d}"
        rows.append(raw_row("start", None, prompt_id, 0.0, 6))
        for seed in analysis.SEEDS:
            for condition, reward, tokens in (("pareto", 2.0, 12), ("gap", 1.0, 10),
                                              ("human", -1.0, 8)):
                rows.append(raw_row(condition, seed, prompt_id, reward, tokens))
    return rows


def compressed_json(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(value, target)


def compressed_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def reference_crossed_draws(matrix, companion, draws, seed):
    """Intentionally allocate each tiny resample to check the optimized loop."""
    rng = random.Random(seed)
    reward_samples, length_samples = [], []
    for _ in range(draws):
        seeds = [rng.randrange(3) for _ in range(3)]
        prompts = [rng.randrange(len(matrix)) for _ in range(len(matrix))]
        reward_samples.append(mean([matrix[prompt][column]
                                    for prompt in prompts for column in seeds]))
        length_samples.append(mean([companion[prompt][column]
                                    for prompt in prompts for column in seeds]))
    return reward_samples, length_samples


def test_crossed_bootstrap_matches_reference_and_shares_both_axes():
    matrix = [(1, 10, 100), (3, 30, 300), (-7, -70, -700), (2, 20, 200)]
    companion = [tuple(2 * value + 5 for value in row) for row in matrix]
    actual, paired = analysis.crossed_bootstrap(matrix, companion, draws=150, random_seed=41)
    expected, expected_paired = reference_crossed_draws(matrix, companion, 150, 41)
    assert actual == pytest.approx(expected)
    assert paired == pytest.approx(expected_paired)
    assert paired == pytest.approx([2 * value + 5 for value in actual])
    assert analysis.crossed_bootstrap(matrix, companion, draws=150, random_seed=41) == \
        (actual, paired)


def test_seed_only_uncertainty_does_not_disappear_with_more_prompts():
    matrix = [(-3, 0, 3)] * analysis.PROMPTS
    actual, paired = analysis.crossed_bootstrap(matrix, draws=80, random_seed=19)
    rng = random.Random(19)
    expected = []
    for _ in range(80):
        seeds = [rng.randrange(3) for _ in range(3)]
        expected.append(mean(matrix[0][seed] for seed in seeds))
        for _ in range(analysis.PROMPTS):
            rng.randrange(analysis.PROMPTS)
    assert actual == expected
    assert paired == []
    assert min(actual) < 0 < max(actual)


def test_prompt_only_uncertainty_resamples_whole_matched_rows():
    matrix = [(-2, -2, -2), (4, 4, 4)]
    actual, _ = analysis.crossed_bootstrap(matrix, draws=100, random_seed=17)
    expected, _ = reference_crossed_draws(matrix, matrix, 100, 17)
    assert actual == expected
    assert set(actual) == {-2, 1, 4}


@pytest.mark.parametrize("difference,conclusion", [(1, "pareto_superior"),
                                                    (-1, "gap_superior"),
                                                    (0, "inconclusive")])
def test_constant_paired_effect_has_exact_interval_and_strict_conclusion(difference, conclusion):
    rows = balanced_rows()
    for row in rows:
        if row["condition"] == "pareto":
            row["reward"] = 1.0 + difference
    summary = analysis.summarize(analysis.validate_records(rows), draws=20)
    primary = summary["primary_pareto_minus_gap_reward"]
    assert primary["mean"] == difference
    assert primary["ci95"] == [difference, difference]
    assert primary["conclusion"] == conclusion
    assert primary["per_seed"] == [{"seed": seed, "mean": difference}
                                   for seed in analysis.SEEDS]


def test_known_rewards_lengths_and_start_pairing():
    summary = analysis.summarize(analysis.validate_records(balanced_rows()), draws=30)
    assert summary["design"]["records"] == 1920
    assert summary["design"]["prompts"] == 192
    assert summary["pareto_minus_gap_response_tokens"]["mean"] == 2
    assert summary["pareto_minus_gap_response_tokens"]["ci95"] == [2, 2]
    for condition, reward, tokens in (("start", 0, 6), ("pareto", 2, 12),
                                      ("gap", 1, 10), ("human", -1, 8)):
        row = summary["conditions"][condition]
        assert row["mean_reward"] == reward
        assert row["response_tokens"] == {"mean": tokens, "median": tokens,
                                           "stddev": 0, "min": tokens, "max": tokens,
                                           "p25": tokens, "p75": tokens}
        assert row["reward_length_pearson"] is None
        for seed in row["per_seed"]:
            assert seed["mean_reward"] == reward
            assert seed["response_tokens"]["mean"] == tokens
        if condition == "start":
            assert row["records"] == 192
            assert row["vs_start"] is None
        else:
            expected = {"win": 576 if reward > 0 else 0, "tie": 0,
                        "loss": 576 if reward < 0 else 0}
            assert row["vs_start"]["comparisons"] == 576
            assert row["vs_start"]["counts"] == expected
            assert row["vs_start"]["mean_reward_delta"] == reward
            assert row["vs_start"]["rates"]["win"] == (1 if reward > 0 else 0)
            for seed in row["per_seed"]:
                assert seed["vs_start"]["comparisons"] == 192
    assert summary["length_confounding"]["paired_delta_pearson"] is None


def test_reward_ties_use_exact_per_prompt_start_comparisons():
    rows = balanced_rows()
    for row in rows:
        if row["condition"] == "start":
            row["reward"] = 1.0
    summary = analysis.summarize(analysis.validate_records(rows), draws=10)
    assert summary["conditions"]["gap"]["vs_start"]["counts"] == \
        {"win": 0, "tie": 576, "loss": 0}


def test_prompt_seed_effects_are_subtracted_before_bootstrap():
    rows = balanced_rows()
    for row in rows:
        baseline = int(row["prompt_id"].rsplit("-", 1)[1]) * 50
        if row["condition"] == "start":
            row["reward"] = baseline
        else:
            seed_effect = (row["seed"] - 1701) * 200
            row["reward"] += baseline + seed_effect
    summary = analysis.summarize(analysis.validate_records(rows), draws=50)
    assert summary["primary_pareto_minus_gap_reward"]["ci95"] == [1, 1]
    assert summary["primary_pareto_minus_gap_reward"]["mean"] == 1


@pytest.mark.parametrize("kind,message", [("duplicate", "Duplicate"),
                                          ("missing", "missing"),
                                          ("missing_start", "distinct start"),
                                          ("wrong_prompt", "unexpected")])
def test_unbalanced_records_are_rejected(kind, message):
    rows = balanced_rows()
    if kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "missing":
        rows.pop()
    elif kind == "missing_start":
        rows.pop(0)
    else:
        rows[-1]["prompt_id"] = "another-prompt"
    with pytest.raises(ValueError, match=message):
        analysis.validate_records(rows)


@pytest.mark.parametrize("field,value,message", [
    ("condition", "unknown", "condition"),
    ("seed", 1704, "seed"),
    ("seed", 1701.0, "seed"),
    ("seed", True, "seed"),
    ("prompt_id", "", "prompt_id"),
    ("reward", float("nan"), "finite"),
    ("reward", float("inf"), "finite"),
    ("reward", -float("inf"), "finite"),
    ("reward", True, "finite"),
    ("reward", "1", "finite"),
    ("response_tokens", -1, "nonnegative integer"),
    ("response_tokens", 1.5, "nonnegative integer"),
    ("response_tokens", True, "nonnegative integer"),
    ("response", None, "string"),
    ("generation_sha256", "not-a-digest", "generation_sha256"),
    ("generation_sha256", "0" * 64, "does not match"),
])
def test_invalid_raw_fields_are_rejected(field, value, message):
    rows = balanced_rows()
    rows[1][field] = value
    with pytest.raises(ValueError, match=message):
        analysis.validate_records(rows)


def test_missing_fields_and_non_null_start_seed_are_rejected():
    rows = balanced_rows()
    del rows[0]["reward"]
    with pytest.raises(ValueError, match="required fields"):
        analysis.validate_records(rows)
    rows = balanced_rows()
    rows[0]["seed"] = 1701
    with pytest.raises(ValueError, match="start seed must be null"):
        analysis.validate_records(rows)


def test_raw_records_bind_to_expected_evaluation_prompt_ids():
    rows = balanced_rows()
    expected = {row["prompt_id"] for row in rows}
    analysis.validate_records(rows, expected)
    expected.remove("synthetic-000")
    expected.add("a-different-evaluation-prompt")
    with pytest.raises(ValueError, match="evaluation dataset"):
        analysis.validate_records(rows, expected)


def test_duplicate_evaluation_ids_are_rejected(tmp_path):
    data = tmp_path / "data.json.gz"
    compressed_json(data, {"evaluation": [{"prompt_id": "same"}] * analysis.PROMPTS})
    with pytest.raises(ValueError, match="distinct prompt IDs"):
        analysis.evaluation_prompt_ids(data)


def test_gzip_input_outputs_and_svg_are_deterministic(tmp_path):
    rows = balanced_rows()
    records = tmp_path / "records.jsonl.gz"
    data = tmp_path / "data.json.gz"
    compressed_rows(records, reversed(rows))
    compressed_json(data, {"evaluation": [{"prompt_id": f"synthetic-{index:03d}",
                                           "prompt": f"Synthetic prompt {index}"}
                                          for index in range(analysis.PROMPTS)]})
    assert analysis.load_records(records) == analysis.validate_records(rows)
    first = analysis.analyze(records, tmp_path, data)
    outputs = {name: (tmp_path / name).read_bytes()
               for name in ("summary.json", "reward-difference.svg")}
    assert first == analysis.analyze(records, tmp_path, data)
    assert outputs == {name: (tmp_path / name).read_bytes() for name in outputs}
    assert json.loads(outputs["summary.json"]) == first
    assert first["bootstrap"]["draws"] == 10_000
    assert first["bootstrap"]["random_seed"] == 20261007
    assert first["source"]["records_sha256"] == hashlib.sha256(records.read_bytes()).hexdigest()
    svg = ElementTree.fromstring(outputs["reward-difference.svg"])
    text = " ".join(svg.itertext())
    assert "192 paired prompts" in text
    assert "[+1.000, +1.000]" in text
    assert "mean 12.0; median 12.0" in text


def test_invalid_records_do_not_write_outputs(tmp_path):
    records = tmp_path / "records.jsonl.gz"
    data = tmp_path / "data.json.gz"
    rows = balanced_rows()
    compressed_rows(records, rows[:-1])
    compressed_json(data, {"evaluation": [{"prompt_id": f"synthetic-{index:03d}"}
                                          for index in range(analysis.PROMPTS)]})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="missing"):
        analysis.analyze(records, output, data)
    assert not output.exists()


def test_length_spread_and_seed_means_keep_seed_variation():
    rows = balanced_rows()
    for row in rows:
        if row["condition"] == "pareto":
            row["response_tokens"] = (row["seed"] - 1700) * 10
    summary = analysis.summarize(analysis.validate_records(rows), draws=20)
    condition = summary["conditions"]["pareto"]
    assert condition["response_tokens"]["mean"] == 20
    assert condition["response_tokens"]["median"] == 20
    assert condition["response_tokens"]["min"] == 10
    assert condition["response_tokens"]["max"] == 30
    assert condition["response_tokens"]["stddev"] == pytest.approx((200 / 3) ** 0.5)
    assert [row["response_tokens"]["mean"] for row in condition["per_seed"]] == [10, 20, 30]
    assert summary["pareto_minus_gap_response_tokens"]["per_seed"] == [
        {"seed": 1701, "mean": 0}, {"seed": 1702, "mean": 10}, {"seed": 1703, "mean": 20}]


def test_svg_length_domain_contains_seed_means_above_condition_mean():
    rows = balanced_rows()
    for row in rows:
        if row["condition"] == "pareto":
            row["response_tokens"] = 192 if row["seed"] == 1701 else 0
    summary = analysis.summarize(analysis.validate_records(rows), draws=20)
    svg = ElementTree.fromstring(analysis.render_svg(summary))
    dots = [element for element in svg.iter()
            if element.tag.endswith("circle") and float(element.attrib["cy"]) >= 400]
    assert len(dots) == 9
    assert all(210 <= float(dot.attrib["cx"]) <= 590 for dot in dots)
    assert max(float(dot.attrib["cx"]) for dot in dots) == 590


def test_percentile_interval_uses_linear_interpolation():
    assert analysis.percentile([0, 10, 20], 0.025) == 0.5
    assert analysis.percentile([0, 10, 20], 0.975) == 19.5
