"""The demo's HelpSteer2 pairs come from the saved experiment data, decisions included."""
import csv
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools/export_demo_pairs.py"
SPEC = importlib.util.spec_from_file_location("export_demo_pairs", MODULE)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)
PAYLOAD = export.build()


def test_committed_demo_pairs_match_the_export() -> None:
    assert export.OUTPUT.read_text(encoding="utf-8") == export.render(PAYLOAD)


def test_sample_is_the_pool_in_hash_order_plus_the_rare_cases() -> None:
    pairs = PAYLOAD["pairs"]
    assert Counter(pair["sample"] for pair in pairs) == dict(export.SAMPLE)
    assert [pair["hashRank"] for pair in pairs if pair["sample"] == "hash-order"] == list(range(30))
    assert [pair["hashRank"] for pair in pairs] == sorted(pair["hashRank"] for pair in pairs)
    for pair in pairs:
        pareto = pair["selectors"]["pareto"]["pick"]
        human = pair["selectors"]["human"]["pick"]
        if pair["sample"] == "pareto-abstains":
            assert pareto is None
        if pair["sample"] == "pareto-against-human":
            assert pareto not in (None, human)


def test_decisions_match_the_saved_pair_table() -> None:
    path = ROOT / "results/human-preferences/pairs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        table = {row["id"]: row for row in csv.DictReader(handle)}
    for pair in PAYLOAD["pairs"]:
        row = table[pair["id"]]
        selectors = pair["selectors"]
        assert {1: 2, -1: 1, 0: None}[int(row["pareto"])] == selectors["pareto"]["pick"]
        assert float(row["helpfulness_gap"]) == selectors["gap"]["helpfulnessGap"]
        assert {1: 2, -1: 1}[int(row["human"])] == selectors["human"]["pick"]
        assert float(row["preference_strength"]) == pair["preferenceStrength"]
        # Every sampled pair lies within the first 1,024 in hash order, so the Pareto set
        # holds exactly the ones the rule keeps.
        assert selectors["pareto"]["inTrainingSet"] == (selectors["pareto"]["pick"] is not None)
        assert selectors["human"]["inTrainingSet"]
        if selectors["gap"]["inTrainingSet"]:
            assert abs(selectors["gap"]["helpfulnessGap"]) >= 3


def test_pareto_reason_names_the_deciding_ratings() -> None:
    assert export.pareto_decision([
        {"correctness": 3, "coherence": 4}, {"correctness": 4, "coherence": 4},
    ]) == (2, "Response 2 is higher on correctness and tied on coherence")
    assert export.pareto_decision([
        {"correctness": 3, "coherence": 4}, {"correctness": 4, "coherence": 3},
    ]) == (None, "Trade-off: response 1 is higher on coherence, response 2 on correctness")
    assert export.pareto_decision([
        {"correctness": 3, "coherence": 3}, {"correctness": 3, "coherence": 3},
    ]) == (None, "Tied on correctness and coherence")
