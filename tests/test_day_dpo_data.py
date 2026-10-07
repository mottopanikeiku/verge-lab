"""Fixed DPO selection and exact text joins; no models or network."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_day_dpo.py"
SPEC = importlib.util.spec_from_file_location("prepare_day_dpo", MODULE_PATH)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def scores(values):
    return dict(zip(data.ATTRIBUTES, (*values, 2, 2), strict=True))


def write_gzip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(data.json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(gzip.compress(payload, mtime=0))


@pytest.fixture
def sources(tmp_path):
    cache = tmp_path / "cache"
    ratings = {"train": [], "validation": []}
    preferences = []
    compact = []
    pairs = []

    def add(prompt, left=(0, 1, 1), right=(1, 2, 2), strength=-1, split="train", responses=None):
        responses = responses or [f"  α {prompt} response one\n" * 20,
                                  f"β {prompt} response two  \n" * 20]
        raw = {"prompt": prompt, "response_1": responses[0], "response_2": responses[1],
               "preference_strength": strength, "split": "val" if split == "validation" else split}
        index = len(preferences)
        identifier = f"helpsteer2:{index:05d}"
        rating_rows = []
        values = [scores(left), scores(right)]
        for response, value in zip(responses, values, strict=True):
            rating_rows.append(len(ratings[split]))
            ratings[split].append({"prompt": prompt, "response": response, **value})
        row = {"id": identifier, "preference_row": index, "split": split,
               "upstream_preference_split": raw["split"], "preference_strength": strength,
               "prompt_sha256": data.text_hash(prompt), "rating_rows": rating_rows,
               "response_sha256": [data.text_hash(response) for response in responses],
               "scores": values}
        compact.append(row)
        pairs.append({"id": identifier, "split": split,
                      "pareto": str(data.pareto_direction(values)),
                      "human": str(data.sign(strength)), "preference_strength": str(strength),
                      "helpfulness_gap": str(right[0] - left[0])})
        preferences.append(raw)

    add("a")
    add("b", left=(4, 3, 3), right=(0, 1, 1), strength=1 / 3)
    add("c", left=(0, 3, 1), right=(3, 1, 3))
    add("d", left=(3, 4, 4), right=(1, 2, 2))
    add("held-out", split="validation")
    add("leak-not-in-any-preference-train-row", split="validation")
    # This prompt is deliberately absent from all compact train preferences.
    ratings["train"].append({"prompt": "leak-not-in-any-preference-train-row",
                             "response": "unused rating", **scores((2, 2, 2))})

    def persist():
        for split, rows in ratings.items():
            write_gzip(cache / f"{split}.jsonl.gz", rows)
        write_gzip(cache / "preference" / "preference.jsonl.gz", preferences)

    persist()
    return cache, compact, pairs, ratings, preferences, persist, add


def test_default_counts_match_expanded_experiment():
    assert data.TRAIN_COUNT == 1024
    assert data.EVALUATION_COUNT == 192
    assert data.SELECTION_SEED == 20261007


@pytest.mark.parametrize("left,right,expected", [
    ((0, 1, 1), (4, 2, 2), 1),
    ((4, 2, 2), (0, 1, 1), -1),
    ((0, 2, 1), (4, 1, 2), 0),
    ((0, 2, 2), (4, 2, 2), 0),
    ((4, 1, 2), (0, 2, 2), 1),
])
def test_correctness_coherence_point_dominance_ignores_helpfulness(left, right, expected):
    assert data.pareto_direction([scores(left), scores(right)]) == expected


def test_exact_full_text_join_and_all_rating_prompt_holdout(sources):
    cache, compact, pairs, _, preferences, _, _ = sources
    pool, validation, train_hashes, counts = data.joined_pool(cache, compact, pairs)
    assert len(pool) == 4
    assert counts["defended_eligible_pairs"] == 3
    assert pool[0]["prompt"] == preferences[0]["prompt"]
    assert pool[0]["responses"] == [preferences[0]["response_1"], preferences[0]["response_2"]]
    train, evaluation, details = data.select_data(pool, validation, train_hashes, 2, 1)
    assert evaluation == [{"prompt_id": data.text_hash("held-out"), "prompt": "held-out"}]
    assert details["validation_prompts_overlapping_any_train_rating"] == 1
    assert details["eligible_evaluation_prompts"] == 1
    assert all(len(rows) == 2 for rows in train.values())
    assert len({row["prompt_id"] for row in evaluation}) == 1


def test_selectors_match_independent_order_and_preserve_each_direction(sources):
    cache, compact, pairs, *_ = sources
    pool, validation, train_hashes, _ = data.joined_pool(cache, compact, pairs)
    train, evaluation, details = data.select_data(pool, validation, train_hashes, 2, 1)
    hashed = sorted(pool, key=lambda row: data.text_hash("20261007:" + row["id"]))
    expected = {"pareto": [row for row in hashed if row["pareto"]][:2],
                "gap": sorted(pool, key=lambda row: (-abs(row["gap"]),
                                                     data.text_hash("20261007:" + row["id"])))[:2],
                "human": hashed[:2]}
    for condition, rows in expected.items():
        assert [record["pair_id"] for record in train[condition]] == [row["id"] for row in rows]
        assert details["selected_pair_ids"][condition] == [row["id"] for row in rows]
        for row, record in zip(rows, train[condition], strict=True):
            chosen_index = 2 if row[condition] > 0 else 1
            assert record["chosen_index"] == chosen_index
            assert record["chosen"] == row["responses"][chosen_index - 1]
            assert record["rejected"] == row["responses"][2 - chosen_index]
        assert details["content_sha256"][condition] == data.sha256(
            data.json_bytes(train[condition])
        )
    assert data.select_data(list(reversed(pool)), dict(reversed(list(validation.items()))),
                            train_hashes, 2, 1) == (train, evaluation, details)
    flipped = copy.deepcopy(pool)
    for row in flipped:
        row["human"] *= -1
    changed, _, _ = data.select_data(flipped, validation, train_hashes, 2, 1)
    assert changed["pareto"] == train["pareto"]
    assert changed["gap"] == train["gap"]
    assert [row["pair_id"] for row in changed["human"]] == [
        row["pair_id"] for row in train["human"]
    ]
    for old, new in zip(train["human"], changed["human"], strict=True):
        assert old["chosen"] == new["rejected"]


def test_selection_overlap_is_set_intersection(sources):
    cache, compact, pairs, *_ = sources
    pool, validation, train_hashes, _ = data.joined_pool(cache, compact, pairs)
    train, _, details = data.select_data(pool, validation, train_hashes, 2, 1)
    for index, left in enumerate(data.CONDITIONS):
        for right in data.CONDITIONS[index + 1:]:
            expected = len({row["pair_id"] for row in train[left]} &
                           {row["pair_id"] for row in train[right]})
            assert details["selection_overlap_pairs"][f"{left}_{right}"] == expected


@pytest.mark.parametrize("field,value,error", [
    ("prompt", "a ", "Preference join prompt hash"),
    ("response_1", "excerpt", "Preference join response hash"),
    ("response_2", "excerpt", "Preference join response hash"),
    ("preference_strength", 1, "Source preference strength"),
    ("split", "val", "Preference join split"),
])
def test_preference_source_mismatches_are_rejected(sources, field, value, error):
    cache, compact, pairs, _, preferences, persist, _ = sources
    preferences[0][field] = value
    persist()
    with pytest.raises(ValueError, match=error):
        data.joined_pool(cache, compact, pairs)


@pytest.mark.parametrize("field,value,error", [
    ("prompt", "changed", "Rating join prompt hash"),
    ("response", "changed", "Rating join response hash"),
    ("correctness", 4, "Rating join scores"),
])
def test_rating_source_mismatches_are_rejected(sources, field, value, error):
    cache, compact, pairs, ratings, _, persist, _ = sources
    ratings["train"][0][field] = value
    persist()
    with pytest.raises(ValueError, match=error):
        data.joined_pool(cache, compact, pairs)


@pytest.mark.parametrize("field,value,error", [
    ("split", "validation", "CSV split"),
    ("human", "1", "CSV source preference sign"),
    ("pareto", "-1", "CSV point-dominance orientation"),
    ("helpfulness_gap", "4", "CSV helpfulness gap"),
])
def test_csv_selector_mismatches_are_rejected(sources, field, value, error):
    cache, compact, pairs, *_ = sources
    pairs[0][field] = value
    with pytest.raises(ValueError, match=error):
        data.joined_pool(cache, compact, pairs)


def test_common_eligibility_filters_before_all_selectors(sources):
    cache, compact, pairs, _, _, persist, add = sources
    add("gap-tie", left=(2, 1, 1), right=(2, 2, 2))
    add("human-tie", strength=0)
    add("multi <extra_id_1> turn")
    add("response-multi", responses=["a", "b <extra_id_1>"])
    add("first-response-multi", responses=["a <extra_id_1>", "b"])
    add("x" * 4001, responses=["a", "b"])
    add("long-response", responses=["a" * 8001, "b"])
    add("long-second-response", responses=["a", "b" * 8001])
    add("x" * 4000, responses=["a" * 8000, "b" * 8000])
    persist()
    pool, _, _, counts = data.joined_pool(cache, compact, pairs)
    assert len(pool) == 5
    assert pool[-1]["prompt"] == "x" * 4000
    assert counts["excluded_helpfulness_tie"] == 1
    assert counts["excluded_human_tie"] == 1
    assert counts["excluded_multiturn"] == 3
    assert counts["excluded_text_length"] == 3


def test_evaluation_filters_single_turn_length_and_any_train_overlap(sources):
    cache, compact, pairs, *_ = sources
    pool, _, train_hashes, _ = data.joined_pool(cache, compact, pairs)
    prompts = ["held-out", "x" * 4000, "y" * 4001, "<extra_id_1> dialog", "a"]
    validation = {data.text_hash(prompt): prompt for prompt in prompts}
    _, evaluation, details = data.select_data(pool, validation, train_hashes, 2, 2)
    assert {row["prompt"] for row in evaluation} == {"held-out", "x" * 4000}
    assert details["eligible_evaluation_prompts"] == 2
    with pytest.raises(ValueError, match="Insufficient held-out single-turn prompts: 2 < 3"):
        data.select_data(pool, validation, train_hashes, 2, 3)


def test_infeasible_counts_and_noncommon_pool_are_rejected(sources):
    cache, compact, pairs, *_ = sources
    pool, validation, train_hashes, _ = data.joined_pool(cache, compact, pairs)
    with pytest.raises(ValueError, match="Insufficient defended pool: 3 < 4"):
        data.select_data(pool, validation, train_hashes, 4, 1)
    with pytest.raises(ValueError, match="Selection counts must be positive"):
        data.select_data(pool, validation, train_hashes, 0, 1)
    pool[0]["human"] = 0
    with pytest.raises(ValueError, match="Common eligibility violation"):
        data.select_data(pool, validation, train_hashes, 2, 1)


def test_pinned_source_hash_corruption_is_rejected(sources):
    cache, _, _, *_ = sources
    compact_directory = MODULE_PATH.parents[1] / "results" / "human-preferences"
    with pytest.raises(ValueError, match="Pinned source hash mismatch"):
        data.load_verified_sources(cache, compact_directory)


def test_tracked_artifact_has_full_text_hashes_counts_and_zero_gzip_timestamp():
    directory = MODULE_PATH.parents[1] / "results" / "day-dpo"
    compressed = (directory / "data.json.gz").read_bytes()
    report = json.loads((directory / "data-metadata.json").read_text())
    payload = gzip.decompress(compressed)
    artifact = json.loads(payload)
    assert compressed[4:8] == bytes(4)
    assert len(compressed) <= 4 * 1024 * 1024
    assert len(compressed) == report["data_gzip_bytes"]
    assert data.sha256(compressed) == report["data_gzip_sha256"]
    assert data.sha256(payload) == report["data_content_sha256"]
    compact_path = MODULE_PATH.parents[1] / "results" / "human-preferences" / "source.jsonl"
    compact = {row["id"]: row for row in map(json.loads, compact_path.read_text().splitlines())}
    for condition, records in artifact["train"].items():
        assert len(records) == 1024
        assert len({row["pair_id"] for row in records}) == 1024
        assert [row["pair_id"] for row in records] == report["selected_pair_ids"][condition]
        for record in records:
            source = compact[record["pair_id"]]
            assert source["split"] == "train"
            assert (
                data.text_hash(record["prompt"]) == source["prompt_sha256"]
                == record["prompt_sha256"]
            )
            assert data.text_hash(record["chosen"]) == source["response_sha256"][
                record["chosen_index"] - 1
            ]
            assert data.text_hash(record["rejected"]) == source["response_sha256"][
                2 - record["chosen_index"]
            ]
            direction = {
                "pareto": data.pareto_direction(source["scores"]),
                "gap": data.sign(
                    source["scores"][1]["helpfulness"] - source["scores"][0]["helpfulness"]
                ),
                "human": data.sign(source["preference_strength"]),
            }[condition]
            assert record["chosen_index"] == (2 if direction > 0 else 1)
            assert data.single_turn(record["prompt"], record["chosen"], record["rejected"])
        assert data.sha256(data.json_bytes(records)) == report["content_sha256"][condition]
    assert len(artifact["evaluation"]) == 192
    assert len({row["prompt_id"] for row in artifact["evaluation"]}) == 192
    for row in artifact["evaluation"]:
        assert set(row) == {"prompt_id", "prompt"}
        assert data.text_hash(row["prompt"]) == row["prompt_id"]
        assert data.single_turn(row["prompt"])
    assert [row["prompt_id"] for row in artifact["evaluation"]] == report["evaluation_prompt_ids"]
    assert data.sha256(data.json_bytes(artifact["evaluation"])) == report[
        "evaluation_content_sha256"
    ]
    assert artifact["metadata"]["license"] == "CC-BY-4.0"
    assert artifact["metadata"]["revision"] == data.REVISION
