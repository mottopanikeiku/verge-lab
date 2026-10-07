"""Human comparison adapters use synthetic scores and no network."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "compare_human_preferences.py"
SPEC = importlib.util.spec_from_file_location("compare_human_preferences", MODULE_PATH)
mining = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mining)


def source_row(identifier="p", left=(1, 1, 1), right=(2, 2, 2), strength=-1):
    def scores(values):
        return dict(zip(mining.ATTRIBUTES, (*values, 2, 2), strict=True))
    return {"id": identifier, "split": "validation", "scores": [scores(left), scores(right)],
            "preference_strength": strength}


def test_human_sign_is_positive_for_response_two_and_not_an_aspect_average():
    row = mining.evaluate_pair(source_row(strength=-0.5), ("correctness", "coherence"))
    assert row["pareto"] == 1
    assert row["human"] == -1
    assert mining.agreement([row], "pareto")["contradict"] == 1
    row["human"] = 0
    result = mining.agreement([row], "pareto")
    assert result["human_tie"] == 1
    assert result["strict_agreement"] is None


def test_primary_holds_out_helpfulness_and_abstains_on_tradeoffs():
    row = source_row(left=(4, 1, 1), right=(0, 2, 2))
    assert mining.evaluate_pair(row, ("correctness", "coherence"))["pareto"] == 1
    assert mining.evaluate_pair(row, mining.CONFIGS["three_quality_aspects"])["pareto"] == 0
    row = source_row(left=(1, 1, 3), right=(2, 2, 2))
    assert mining.evaluate_pair(row, ("correctness", "coherence"))["reason"] == \
        "cross-objective tradeoff"
    row = source_row(left=(1, 2, 2), right=(4, 2, 2))
    assert mining.evaluate_pair(row, ("correctness", "coherence"))["pareto"] == 0


def test_matching_excludes_overall_ties_from_both_methods_and_never_uses_labels():
    rows = [
        mining.evaluate_pair(source_row("tie", left=(2, 1, 1), right=(2, 2, 2)),
                             ("correctness", "coherence")),
        mining.evaluate_pair(source_row("low"), ("correctness", "coherence")),
        mining.evaluate_pair(source_row("high", left=(0, 3, 1), right=(4, 1, 3)),
                             ("correctness", "coherence")),
    ]
    result = mining.compare(rows)
    assert result["defended"] == 2
    assert result["matched"]["yield_pairs_each"] == 1
    assert result["matched"]["excluded_helpfulness_ties"] == 1
    assert result["full_yield_matched"]["feasible"] is True
    assert result["full_yield_matched"]["helpfulness_gap"]["selected"] == 2
    assert rows[0]["matched_pareto"] is False
    assert rows[1]["matched_pareto"] is True
    assert rows[2]["matched_gap"] is True
    selected = [row["id"] for row in mining.gap_selection(rows, 2)]
    for row in rows:
        row["human"] *= -1
    assert [row["id"] for row in mining.gap_selection(rows, 2)] == selected
    assert mining.gap_selection(list(reversed(rows)), 2) == mining.gap_selection(rows, 2)
    with pytest.raises(ValueError, match="exceeds"):
        mining.gap_selection(rows, 3)
    only_ties = mining.compare([rows[0]])
    assert only_ties["full_yield_matched"]["feasible"] is False
    assert only_ties["full_yield_matched"]["helpfulness_gap"] is None
    assert only_ties["matched"]["yield_pairs_each"] == 0


def test_exact_join_preserves_fractional_preferences_and_reports_unmatched(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "README.md").write_text("---\nlicense: cc-by-4.0\n---\n")
    scores = dict.fromkeys(mining.ATTRIBUTES, 2)
    def compressed(name, values):
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt") as output:
            for value in values:
                output.write(json.dumps(value) + "\n")
    compressed("train.jsonl.gz", [{"prompt": "p", "response": text, **scores}
                                   for text in ("a", "b", "a")])
    compressed("validation.jsonl.gz", [{"prompt": "q", "response": text, **scores}
                                     for text in ("c", "d")])
    preferences = [{"split": "train", "prompt": "p", "response_1": "a", "response_2": "b",
                    "preference_strength": 1 / 3},
                   {"split": "train", "prompt": "p", "response_1": "a ", "response_2": "b",
                    "preference_strength": 1},
                   {"split": "val", "prompt": "q", "response_1": "c", "response_2": "d",
                    "preference_strength": 1}]
    compressed("preference/preference.jsonl.gz", preferences)
    result = tmp_path / "result"
    mining.fetch(result, cache)
    source = [json.loads(line) for line in (result / "source.jsonl").read_text().splitlines()]
    assert len(source) == 2
    assert source[0]["preference_strength"] == 1 / 3
    assert source[0]["response_sha256"][0] == mining.sha256(b"a")
    assert source[1]["split"] == "validation"
    assert source[1]["upstream_preference_split"] == "val"
    meta = json.loads((result / "source-metadata.json").read_text())
    assert meta["unmatched"] == [{"preference_row": 1, "split": "train"}]
    assert meta["counts"]["train_rating_responses"] == 3
    assert meta["identical_duplicate_rating_keys"] == [
        {"split": "train", "first_row": 0, "duplicate_row": 2}
    ]
    compressed("train.jsonl.gz", [{"prompt": "p", "response": "a", **scores},
                                {"prompt": "p", "response": "a", **scores, "correctness": 0}])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        mining.fetch(result, cache)


def test_offline_analysis_repeats_exactly_and_checks_source_hash(tmp_path, monkeypatch):
    rows = [source_row("a"), source_row("b", left=(2, 2, 2), right=(2, 3, 3), strength=0)]
    stored = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (tmp_path / "source.jsonl").write_bytes(stored)
    mining.save_json(tmp_path / "source-metadata.json",
                     {"stored_source_sha256": mining.sha256(stored)})
    def no_network(*args, **kwargs):
        raise AssertionError("Offline analysis requested network")
    monkeypatch.setattr(mining.urllib.request, "urlopen", no_network)
    first = mining.analyze(tmp_path)
    outputs = {name: (tmp_path / name).read_bytes()
               for name in ("summary.json", "pairs.csv", "human-agreement.svg")}
    assert mining.analyze(tmp_path) == first
    assert outputs == {name: (tmp_path / name).read_bytes() for name in outputs}
    (tmp_path / "source.jsonl").write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        mining.analyze(tmp_path)
