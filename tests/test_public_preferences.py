"""Adapter tests use synthetic fixtures, never a network request or model call."""

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "mine_public_preferences.py"
SPEC = importlib.util.spec_from_file_location("mine_public_preferences", MODULE_PATH)
mining = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mining)


def completion(identifier, ratings, overall):
    return {
        "id": identifier,
        "completion_index": 0,
        "model": "synthetic-test-fixture",
        "response_excerpt": identifier,
        "annotations": {
            aspect: [{"rating": rating}]
            for aspect, rating in zip(mining.ASPECTS, ratings, strict=False)
        },
        "overall_score": overall,
    }


def prompt(*completions):
    return {"id": "p", "source_row_index": 0, "completions": list(completions)}


def pairs_for(row, confidence=1.0, scale=0.0):
    return mining.mine_pareto_edges(
        mining.candidates_for([row], confidence),
        mining.OBJECTIVES,
        uncertainty_scale=scale,
    )


def test_rating_rescaling_and_annotation_missingness():
    value = completion("a", [1, 3, 5, "N/A"], 7)
    candidate = mining.candidates_for([prompt(value)], 0.9)[0]
    assert candidate.scores["helpfulness"].value == 0.0
    assert candidate.scores["honesty"].value == 0.5
    assert candidate.scores["instruction_following"].value == 1.0
    assert "truthfulness" not in candidate.scores
    assert candidate.scores["honesty"].confidence == 0.9
    value["annotations"]["honesty"] = [{"rating": 5}, {"rating": 3}]
    assert mining.aspect_rating(value, "honesty") == 4.0
    value["annotations"]["honesty"].append({"rating": None})
    assert mining.aspect_rating(value, "honesty") is None


def test_overall_tie_and_opposite_ordering_are_not_agreement():
    strong = completion("a", [5, 5, 5, 5], 8)
    weak = completion("b", [1, 1, 1, 1], 8)
    pair = pairs_for(prompt(strong, weak))[0]
    by_id = {value["id"]: value for value in (strong, weak)}
    assert pair.verdict == "defended"
    assert mining.overall_relation(pair, by_id) == "overall_tie"
    weak["overall_score"] = 9
    assert mining.overall_relation(pair, by_id) == "disagree"
    weak["overall_score"] = None
    assert mining.overall_relation(pair, by_id) == "overall_missing"


def test_partial_and_all_missing_aspects_abstain():
    strong = completion("a", [5, 5, 5, 5], 9)
    weak = completion("b", [1, 1, 1], 1)
    pair = pairs_for(prompt(strong, weak))[0]
    assert pair.reason == "missing scores: truthfulness"
    weak["annotations"] = {}
    pair = pairs_for(prompt(strong, weak))[0]
    assert pair.verdict == "ambiguous"
    assert (
        pair.reason == "missing scores: helpfulness, honesty, instruction_following, truthfulness"
    )


def test_top_ties_compare_each_top_to_strictly_lower_only():
    row = prompt(
        completion("a", [5, 5, 5, 5], 9),
        completion("b", [4, 4, 4, 4], 9),
        completion("c", [1, 1, 1, 1], 2),
    )
    result, membership = mining.top_comparisons(
        [row],
        mining.candidates_for([row], 1.0),
        {"uncertainty_scale": 0.0},
    )
    assert result["comparisons"] == 2
    assert result["outcomes"] == {"top_defended": 2, "other_defended": 0, "abstained": 0}
    assert result["prompt_counts"] == {"tied_top": 1}
    assert set(membership.values()) == {"a", "b"}


def test_positive_penalty_removes_equal_aspect_dominance():
    row = prompt(completion("a", [5, 5, 5, 5], 9), completion("b", [5, 4, 4, 4], 7))
    assert pairs_for(row)[0].verdict == "defended"
    assert pairs_for(row, confidence=0.9, scale=0.05)[0].reason == "confidence bounds overlap"
    assert pairs_for(row, confidence=0.5)[0].reason.startswith("confidence below threshold")


def test_offline_analysis_is_deterministic_and_counts_ties(tmp_path, monkeypatch):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("Offline analysis attempted a download")

    monkeypatch.setattr(mining.urllib.request, "urlopen", network_forbidden)
    row = prompt(
        completion("a", [5, 5, 5, 5], 8),
        completion("b", [1, 1, 1, 1], 8),
        completion("c", [1, 1, 1, 1], 9),
    )
    source = (json.dumps(row) + "\n").encode()
    (tmp_path / "source.jsonl").write_bytes(source)
    mining.save_json(
        tmp_path / "source-metadata.json", {"stored_source_sha256": mining.sha256(source)}
    )
    result = mining.analyze(tmp_path)
    original = {
        name: (tmp_path / name).read_bytes()
        for name in ("summary.json", "pairs.csv", "examples.json")
    }
    assert result["baseline"]["all_pairs"] == 3
    assert result["baseline"]["defended"] == 2
    assert result["baseline"]["abstained"] == 1
    assert result["baseline"]["defended_vs_overall"] == {
        "agree": 0,
        "disagree": 1,
        "overall_tie": 1,
        "overall_missing": 0,
    }
    assert result["overall_pair_rankings"] == {"strictly_ranked": 2, "tied": 1}
    assert mining.analyze(tmp_path) == result
    for name, data in original.items():
        assert (tmp_path / name).read_bytes() == data
