# The nine-adapter experiment

## Data

HelpSteer2 ratings and HelpSteer2-Preference labels are joined by exact prompt and response hashes ([`tools/prepare_day_dpo.py`](../tools/prepare_day_dpo.py)). A pair is eligible if it is in the train split, single-turn, has a nonzero helpfulness gap and a nonzero human preference, and fits the length limits (prompt ≤ 4,000 characters, responses ≤ 8,000). That leaves 3,956 pairs, 3,719 of which pass the Pareto rule.

Each training set takes 1,024 of them:

| Selector | How the 1,024 are chosen | Chosen answer |
| --- | --- | --- |
| Pareto | Passing pairs, first by SHA-256 order of `20261007:pair_id` | The dominant answer |
| Helpfulness gap | Largest absolute helpfulness gaps, same hash order for ties | Higher helpfulness |
| Human reference | Eligible pairs, same hash order | Human-preferred answer |

Pareto maximizes correctness and coherence (rescaled 0–1, confidence 1). Helpfulness is the baseline's signal. Complexity and verbosity are left out, since maximizing them rewards length whatever the prompt asks.

The 192 evaluation prompts are single-turn validation prompts picked by the same hash rule. None appears in any HelpSteer2 training rating row.

The compact data (full texts for the 1,024-pair sets and the evaluation prompts) is in `results/day-dpo/data.json.gz`. To rebuild it from upstream:

```bash
uv run python tools/compare_human_preferences.py --fetch --cache .cache/helpsteer2
uv run python tools/prepare_day_dpo.py --cache .cache/helpsteer2
```

## Training

[`tools/day_dpo_modal.py`](../tools/day_dpo_modal.py) pins the model revision and the dependency stack. Each of the three selectors trains from the same Qwen2.5-1.5B-Instruct checkpoint with a fresh LoRA adapter, for seeds 1701, 1702 and 1703.

| Setting | Value |
| --- | --- |
| Epochs / optimizer steps | 1 / 64 |
| Learning rate, schedule | 5e-5, linear, no warmup |
| DPO loss, beta | sigmoid, 0.1 |
| Batch | 2 per device × 8 accumulation = 16 |
| Max prompt / sequence tokens | 384 / 768 |
| LoRA | rank 16, alpha 32, dropout 0, all linear layers |
| Precision | bfloat16, gradient checkpointing |
| Reference model | starting checkpoint, adapter disabled |

Per-run settings, losses and dependency versions are in `results/day-dpo/training-*.json`.

## Evaluation

Every adapter and the starting model answer the same 192 prompts, greedy decoding, at most 192 new tokens. The starting model answers once per prompt and is shared across seeds.

The scorer is `OpenAssistant/reward-model-deberta-v3-large-v2`: one raw logit per prompt and answer, truncated to 512 tokens (117 of 1,920 inputs are cut). Its training sources (WebGPT comparisons, summarization feedback, synthetic instruction pairs, Anthropic HH) predate HelpSteer2.

The headline number is the mean of `reward(Pareto) − reward(gap)` over prompts and matched seeds. The interval comes from 10,000 bootstrap draws that resample prompts and seeds independently (random seed 20261007), with percentile endpoints. An interval entirely above zero favors Pareto, entirely below favors gap, otherwise no verdict. No length adjustment is applied.

`records.jsonl.gz` holds every generation with its reward. `summary.json` holds the aggregates. [`tools/analyze_day_dpo.py`](../tools/analyze_day_dpo.py) refuses missing, duplicate, nonfinite or unbalanced rows.

## Retraining on Modal

Needs a Modal account and paid L4 GPUs; full training uses up to three containers in parallel.

```bash
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode cache
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode full --size 1.5B
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode score --size 1.5B
uv run python tools/analyze_day_dpo.py --output-dir .cache/dpo
```

Finished adapters, generations and scores are saved to a Modal volume before returning, so a rerun skips completed work. `--mode cleanup` deletes the downloaded model weights and keeps the adapters. `results/day-dpo/compute.json` lists every attempt and its cost bound.

## Selection rule

For each objective, [`pareto.py`](../verge_lab/pareto.py) flips minimize scores so higher is better and leaves scales alone. The adjusted advantage of the left answer is:

```text
left_score - right_score
  - uncertainty_scale * ((1 - left_confidence) + (1 - right_confidence))
```

Defaults: `uncertainty_scale = 0.05`, minimum confidence 0.8, `epsilon = 0.01` in each objective's own units. A pair is selected when every adjusted margin is nonnegative and at least one exceeds epsilon. The same test runs in the other direction. Anything else abstains: missing scores, low confidence, near ties, tradeoffs. With a nonzero penalty, an exact tie on any objective also blocks dominance.

[`export.py`](../verge_lab/export.py) recomputes every comparison before writing DPO JSONL and rejects artifacts whose stored decisions or margins disagree. The package runs no model; scores come from whatever you supply (the built-in checkers and scorers are phrase, JSON-field, regex and word-count checks).

## Agreement with human preference labels

`results/human-preferences/` compares both selectors with HelpSteer2-Preference labels (`uv run python tools/compare_human_preferences.py`). At matched yield on validation data each selects 310 pairs; the human labels agree on 281 of 286 Pareto picks and 277 of 282 gap picks (human ties are excluded from the denominators, 24 and 28 of them).

## UltraFeedback ratings on TruthfulQA

`results/public-preferences/` runs the selector on [UltraFeedback](https://arxiv.org/abs/2310.01377) ratings for TruthfulQA (`tools/mine_public_preferences.py`). UltraFeedback's MIT license does not replace the terms of the underlying content.

## Browser demo and checks

`artifacts/demo-run.json` is generated from `examples/candidates.json` with `uv run verge demo`. Its scores are made up; the pair decisions are computed. The Pages workflow builds with Vite under `/verge-lab/`. To preview a production build:

```bash
npm ci && npm run build && npm run preview -- --host 127.0.0.1
```

Checks:

```bash
uv run pytest
uv run ruff check .
npm run check
```

`tools/check_demo.mjs` is a Puppeteer pass over the running demo (keyboard navigation, filters, overflow on mobile and desktop).
