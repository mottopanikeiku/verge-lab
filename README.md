# Verge Lab

Verge Lab is an offline preference-pair miner and review tool for scored language-model responses.

**Question:** When several scoring aspects disagree, which response pairs can a Pareto rule defend, and when should it abstain?

The rule in [`verge_lab/pareto.py`](verge_lab/pareto.py) changes minimization scores to a higher-is-better direction, subtracts an assumed uncertainty penalty, and keeps an edge only if every adjusted margin is nonnegative and at least one clears the strict threshold. [`verge_lab/export.py`](verge_lab/export.py) recomputes comparisons before exporting defended pairs for DPO. The React UI reviews **authored illustrative examples**, not model measurements; its ambient instruments come from [ThreeUI](https://github.com/MengTo/threeui), not model internals.

## Result: mining public annotations, not training a model

[`tools/mine_public_preferences.py`](tools/mine_public_preferences.py) analyzes the first **200 prompts** in the pinned UltraFeedback TruthfulQA file: **800 responses, 1,200 within-prompt pairs**. It uses the recorded helpfulness, honesty, instruction-following and truthfulness ratings, rescaled to a common range. The reference preference is the upstream judge's separate `overall_score`, not a human preference label or an aspect average.

| Nominal point-score comparison | Pairs |
| --- | ---: |
| Defended | 820 |
| Abstained | 380 |
| Defended, agreeing with overall ranking | 574 |
| Defended, opposing overall ranking | 132 |
| Defended, overall score tied | 114 |

All counts and assumptions are in [`summary.json`](results/public-preferences/summary.json). Abstentions include **283 aspect tradeoffs, 79 aspect ties and 18 missing-score comparisons**. Among strictly ranked defended pairs, disagreement is **132/706 (18.7%)**. Comparing each overall-top response against strictly lower-ranked responses gives **344 top defended, 95 other defended and 196 abstentions**; tied tops are handled explicitly.

Confidence is absent from the dataset. The nominal run assumes full confidence with no uncertainty penalty; it is a score-consistency check, not a reliability estimate. With assumed confidence **0.9** and penalty scale **0.05**, only **343** pairs remain defended and **857** abstain. Tied aspects lose their nonnegative margin under any positive penalty. This sensitivity is a limitation of the rule, not evidence that either setting is calibrated.

[`examples.json`](results/public-preferences/examples.json) includes actual disagreements and abstentions. For example, on source row `0001`, completion `3` dominates completion `0` on instruction-following with the other aspects tied, but has a lower overall score (**7.0 versus 7.5**). [`pairs.csv`](results/public-preferences/pairs.csv) records every comparison; [source metadata](results/public-preferences/source-metadata.json) records the revision, hashes, attribution and dataset-card license.

The browser's curves, candidate scores, judge confidences, costs and model metadata remain **illustrative**. Its calculated example verdicts are not a benchmark. The [historical GPU smoke run](artifacts/modal-training.json) used **two pairs**, one epoch and **3.3742 seconds**, with training loss **0.693147** (approximately ln 2): it supplies no evidence of improved model quality.

## Reproduce

From a checkout, with Python and uv, on a local CPU:

```bash
uv sync --locked --dev
nice -n 19 uv run python tools/mine_public_preferences.py
nice -n 19 uv run pytest -q -x
```

The mining command uses the committed compact source without network access. Add `--fetch` to refresh from the pinned public file. [The recorded offline run](results/public-preferences/run.json) took **0.312 seconds**, used **34,816 KiB** peak process RSS, and cost **$0**; no model inference or training was run. [Workflow details](docs/WORKFLOWS.md) cover the illustrative UI, CLI export, checks and optional historical GPU entrypoints.

## Limitations

- This is a small deterministic prefix of one dataset subset, not a representative sample.
- GPT-4 aspect and overall annotations are unverified and share a judge; disagreement is not an error rate.
- Ordinal-score rescaling and confidence penalties are modelling choices. Pareto selection can favor easy dominance and discard useful tradeoffs.
- Built-in phrase, keyword, JSON, regex and word-count checks do not establish semantic correctness; supplied judge confidences are not calibrated by this tool.
- Stored excerpts support score reproduction, not full offline response review. No downstream training or held-out evaluation was performed.

## Prior work and attribution

[UltraFeedback](https://arxiv.org/abs/2310.01377) supplies GPT-4 annotations on responses to [TruthfulQA](https://arxiv.org/abs/2109.07958) prompts; its pinned card declares MIT, with upstream-content terms noted in the source metadata. [DPO](https://arxiv.org/abs/2305.18290) motivates the export format. The Pareto rule is this repository's engineering choice, not a reproduction of those papers. Project code and ThreeUI are MIT licensed; upstream assets retain their own terms.
