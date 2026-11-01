# Verge Lab

Verge Lab selects DPO training pairs from multi-aspect scores. It chooses an answer only when at least one score improves and none gets worse, leaving tradeoffs unresolved.

I compared this Pareto rule with selecting the largest helpfulness gaps. Across nine Qwen2.5-1.5B-Instruct adapters, Pareto showed no clear advantage: its mean reward was 0.11 logits lower, with a 95% bootstrap interval that crossed zero. Each selector used 1,024 training pairs and three matched seeds, evaluated on 192 held-out HelpSteer2 prompts.

![Paired reward differences and response lengths](results/day-dpo/reward-difference.svg)

[Scores and generations](results/day-dpo/) · [Method and training settings](docs/WORKFLOWS.md#equal-size-dpo-training-experiment) · [Browser demo](https://mottopanikeiku.github.io/verge-lab/)

## The comparison

I use correctness and coherence for Pareto selection. Helpfulness supplies the baseline; complexity and verbosity describe style and are left out. The human-reference condition uses separately collected pairwise preferences.

All three conditions draw from the same eligible training pool and use the same DPO settings. Pareto and human-reference pairs are sampled by a fixed hash order; the gap selector takes the largest absolute helpfulness gaps. The sets differ: Pareto shares 330 pairs with gap and 958 with the human reference. This compares the full selection methods, including which pairs they choose.

| Training pairs | Mean reward | Wins vs starting model | Mean response tokens |
| --- | ---: | ---: | ---: |
| Starting model | 0.550 | — | 159.9 |
| Pareto | 0.620 | 49.83% | 161.8 |
| Helpfulness gap | 0.731 | 54.86% | 157.7 |
| Human reference | 0.651 | 50.00% | 159.7 |

Reward is the raw logit from [OpenAssistant's DeBERTa reward model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2). Each trained row covers 576 comparisons against one common starting-model answer per prompt; ties remain in the win-rate denominator.

The Pareto-minus-gap difference is −0.112 logits, with a crossed prompt/seed bootstrap 95% interval of [−0.257, +0.032]. The three seed means are −0.027, −0.118 and −0.190. The [summary](results/day-dpo/summary.json) includes full-precision values, counts and response lengths.

All nine adapters completed one epoch and 64 optimizer steps. I used LoRA rank 16, learning rate 0.00005, DPO beta 0.1 and effective batch size 16. The reference model is the starting checkpoint with its adapter disabled.

## Run the analysis locally

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are enough to recompute the result from the saved scores.

```bash
uv sync --locked --dev
uv run python tools/analyze_day_dpo.py --output-dir .cache/day-dpo-analysis
```

The script resamples matched seeds and prompts in 10,000 bootstrap draws and writes a summary and SVG. [Cloud training commands and pinned versions](docs/WORKFLOWS.md#equal-size-dpo-training-experiment) are separate.

## Select and inspect pairs

The [Python selector](verge_lab/pareto.py) supports maximize/minimize objectives and a confidence-based margin penalty. It abstains on missing scores, low confidence, ties and tradeoffs. The penalty is a heuristic; the experiment uses the human ratings directly with confidence set to one. The [exporter](verge_lab/export.py) recomputes pair decisions before writing TRL-compatible DPO JSONL.

```bash
uv run verge analyze examples/candidates.json --output artifacts/my-run.json
uv run verge export-dpo artifacts/my-run.json --output artifacts/my-run.dpo.jsonl
```

The [browser demo](https://mottopanikeiku.github.io/verge-lab/) lets you inspect score comparisons, pair decisions and the artifact format. Its scores, model metadata and checkpoint curves are illustrative. Pair decisions and margins are computed from those example inputs.

To run the UI locally with Node.js 22+:

```bash
npm ci
npm run dev
```

## Earlier label comparison

At matched yield on HelpSteer2 validation data, both selectors chose 310 pairs. Excluding human ties, Pareto agreed with 281/286 preferences (98.25%) and helpfulness gap with 277/282 (98.23%). The [full comparison](results/human-preferences/summary.json) includes tie counts and alternative objective sets.

## Scope

- This is one 1.5B model, 1,024 pairs per selector and three training seeds.
- The reward model supplies a proxy score. Its declared training sources predate HelpSteer2, but older-source overlap and Qwen's instruction-training data remain unknown.
- Training truncates long inputs. Generation stops at 192 new tokens, and every condition's median response reaches that cap. The reward scorer also truncates 117 of 1,920 inputs to 512 tokens.
- Pareto answers average 4.14 tokens longer than gap answers. The comparison is not length-matched.

## Data and licenses

[HelpSteer2](https://arxiv.org/abs/2406.08673) and [HelpSteer2-Preference](https://arxiv.org/abs/2410.01257), by NVIDIA, Scale AI and Zhilin Wang et al., are CC-BY-4.0. Qwen2.5 is Apache-2.0; the scorer and project code are MIT. Training uses the [DPO objective](https://arxiv.org/abs/2305.18290).

The earlier [public-rating experiment](results/public-preferences/) uses GPT-4 annotations from [UltraFeedback](https://arxiv.org/abs/2310.01377) on [TruthfulQA](https://arxiv.org/abs/2109.07958). UltraFeedback's MIT license does not replace the upstream content terms.
