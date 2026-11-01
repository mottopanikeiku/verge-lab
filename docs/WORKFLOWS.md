# Running Verge Lab

## Local example and review UI

The browser reads `artifacts/demo-run.json`, generated from `examples/candidates.json`. Scores, confidences, model metadata, checkpoint curves and resource estimates are illustrative. Pair decisions, margins and mutation flips are computed from those inputs.

Python 3.11+ and uv are needed for the selector; Node.js 22+ and npm are needed for the UI.

```bash
uv sync --locked --dev
uv run verge demo --output artifacts/demo-run.json
npm ci
npm run dev
```

The four views show the selection rule, a candidate graph and table, individual pair decisions, and the example artifact's metadata and checkpoints.

### GitHub Pages

The live demo is at <https://mottopanikeiku.github.io/verge-lab/>. Vite uses the `/verge-lab/` base path for scripts, styles and the favicon. To serve a production build locally:

```bash
npm ci
npm run build
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/verge-lab/`. The [Pages workflow](../.github/workflows/pages.yml) builds with the npm lockfile on pushes to `main` or manual dispatch, uploads `dist`, and deploys it. Set the repository's Pages source to GitHub Actions before deploying. The app loads no remote executable scripts.

The [Chromium checks](../tools/check_demo.mjs) cover keyboard navigation, filters, pair details, browser errors and desktop/mobile overflow. With the preview server running:

```bash
npm install --no-save --package-lock=false puppeteer
node tools/check_demo.mjs http://127.0.0.1:4173/verge-lab/
```

Puppeteer downloads Chromium if needed. Stop the preview server afterward; `npm ci` restores the locked dependencies. The older [smoke record](assets/pages-smoke.json) and [screenshot](assets/pages-pair-lab.png) show a previous version of the UI.

## Analyze and export

```bash
uv run verge analyze examples/candidates.json --output artifacts/my-run.json
uv run verge export-dpo artifacts/my-run.json --output artifacts/my-run.dpo.jsonl
```

`analyze` accepts supplied scores and confidences. The local package runs no model inference or generated code. Built-in checkers use phrase, JSON-field and regular-expression checks; scorers use keywords, phrases, JSON fields and word counts. Semantic correctness and confidence calibration depend on the score source.

### Selection rule

For each objective, [pareto.py](../verge_lab/pareto.py) flips the sign of minimization scores so higher is better. It leaves score scales unchanged. The adjusted advantage of the left answer is:

```text
left_score - right_score
  - uncertainty_scale * ((1 - left_confidence) + (1 - right_confidence))
```

The defaults are `uncertainty_scale = 0.05`, minimum confidence `0.8`, and strict improvement threshold `epsilon = 0.01`, in each objective's native units. A pair is selected when every adjusted margin is nonnegative and at least one exceeds epsilon. The same comparison is tried in the other direction.

Missing scores, low confidence, near ties and cross-objective tradeoffs cause abstention. A nonzero penalty also blocks dominance when any objective is exactly tied. This penalty is heuristic; it is separate from the statistical bootstrap used for the DPO result.

[export.py](../verge_lab/export.py) recomputes every comparison with the default thresholds and rejects stored decisions or margins that differ. Only selected pairs are exported. Artifacts do not store threshold settings, so nondefault settings are incompatible whenever they change a stored comparison.

The artifact includes run/model metadata, counts, prompt rubrics and constraints, candidate scores and coordinates, pair decisions, mutation results and optional checkpoints. Content hashes and stable ordering make repeated outputs comparable.

## Human ratings and preferences

```bash
uv run python tools/compare_human_preferences.py
uv run python tools/compare_human_preferences.py --fetch --cache .cache/helpsteer2
```

The first command analyzes the saved compact source offline. The second downloads the pinned original rating and preference files. [Source metadata](../results/human-preferences/source-metadata.json) lists revisions, hashes, join counts and exclusions.

Each preference pair joins to two rating responses by exact prompt and response text hashes. The preference split `val` maps to the rating split `validation`. Duplicate rating keys are accepted only when their split and scores agree; conflicting duplicates stop the analysis.

The primary Pareto rule maximizes correctness and coherence, rescaled from 0–4 to 0–1 with confidence set to one. Helpfulness supplies the baseline. Complexity and verbosity are omitted because maximizing style scores would favor sophistication and length regardless of the prompt. The summary also includes all three quality aspects and naive five-aspect maximization.

The reference label is the separately collected `preference_strength`: positive values prefer response 2. Human ties count toward selection yield but are excluded from strict-agreement denominators. The annotations come from the same dataset population.

For equal-yield comparison, both selectors exclude helpfulness ties. Within each split, Pareto selects all remaining pairs that pass its rule; the baseline selects the same number by descending absolute helpfulness gap. A fixed SHA-256 order breaks equal-gap ties without reading human labels. The summary also reports variation across twenty tie-breaking seeds, train and validation results separately, and Wilson intervals for nominal binomial variation.

At matched validation yield, each selector chooses 310 pairs. Pareto agrees with 281 of 286 strict human preferences; gap agrees with 277 of 282. The different denominators come from 24 versus 28 human ties. [Full counts and sensitivities](../results/human-preferences/summary.json).

HelpSteer2 and HelpSteer2-Preference are by NVIDIA, Scale AI and Zhilin Wang et al., under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The compact files omit response text and preserve scores, preference strengths and text hashes. HelpSteer3 lacks the matched numeric aspect ratings needed by this selector.

The [earlier UltraFeedback analysis](../results/public-preferences/) uses GPT-4 ratings on TruthfulQA. The [full-text disagreement notes](../results/disagreement-review/) are AI-assisted qualitative reviews of fixed samples.

## Equal-size DPO training experiment

### Data and selectors

[prepare_day_dpo.py](../tools/prepare_day_dpo.py) joins HelpSteer2 ratings and preferences by exact text hashes. The shared training pool requires:

- Train split and single-turn prompt/responses, with no `<extra_id_1>` markers.
- Nonzero helpfulness gap and nonzero human preference.
- Prompt length at most 4,000 characters and each response at most 8,000 characters.

There are 3,956 eligible pairs, of which 3,719 pass the correctness/coherence Pareto rule. Each training set contains 1,024 pairs and is reused across seeds:

| Selector | Pair selection | Chosen answer |
| --- | --- | --- |
| Pareto | Passing pairs in ascending SHA-256 order of `20261007:pair_id` | Pareto-dominant answer |
| Helpfulness gap | Largest absolute gaps; same hash order for ties | Higher helpfulness score |
| Human reference | Shared pool in the same hash order | Preferred human label |

Pareto and gap share 330 pairs; Pareto and human share 958. The experiment therefore compares pair selection and direction together.

Evaluation uses 192 distinct single-turn validation prompts, at most 4,000 characters, selected by the same hash rule on prompt hashes. Every evaluation hash is absent from all HelpSteer2 training rating rows. Validation responses and labels are unused.

[data.json.gz](../results/day-dpo/data.json.gz) contains full training texts and evaluation prompts. [data-metadata.json](../results/day-dpo/data-metadata.json) includes the pinned dataset revision, source hashes, selected IDs and overlaps. To rebuild from the upstream cache:

```bash
uv run python tools/compare_human_preferences.py --fetch --cache .cache/helpsteer2
uv run python tools/prepare_day_dpo.py --cache .cache/helpsteer2
```

### Model and training settings

The model is `Qwen/Qwen2.5-1.5B-Instruct`; [day_dpo_modal.py](../tools/day_dpo_modal.py) pins its revision and dependencies. Each condition starts from the same checkpoint with a new LoRA adapter, using matched seeds 1701, 1702 and 1703.

| Setting | Value |
| --- | --- |
| Epochs / optimizer steps | 1 / 64 |
| Learning rate | 0.00005 |
| DPO beta / loss | 0.1 / sigmoid |
| Per-device batch / accumulation | 2 / 8, effective batch 16 |
| Maximum prompt / full sequence tokens | 384 / 768 |
| Optimizer / weight decay | AdamW / 0.01 |
| LR schedule / warmup | Linear / 0 |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| LoRA targets | All linear layers |
| Precision / gradient checkpointing | bfloat16 / enabled |
| DPO reference | Starting checkpoint, adapter disabled |

Training and generation use the model's chat template. Labels apply to full source responses; TRL truncates long sequences. All nine runs completed 64 optimizer steps with finite losses and nonzero adapter changes. The `training-*.json` files contain settings, losses, step counts, data hashes and dependency versions.

A 64-pair pilot measured training and generation time before choosing 1.5B. It projected 181.92 aggregate L4 minutes, above the original 90-minute ceiling; I increased that ceiling before the complete experiment. Pilot outputs were unscored and the adapter was discarded. [Model choice](../results/day-dpo/model-choice.json) has the measurements. The [all-attempt compute estimate](../results/day-dpo/compute.json) is about $4.70, including interrupted attempts; this is an upper-bound estimate rather than an invoice.

### Generation and reward analysis

Each adapter and the starting model answer the same 192 prompts. Generation is greedy, with batch size 8, a 384-token prompt limit and at most 192 new tokens. There is one shared starting-model answer per prompt. Response length counts generated Qwen tokens through the first EOS, including EOS when present.

`OpenAssistant/reward-model-deberta-v3-large-v2` scores each prompt/answer pair with a raw scalar logit. Joint tokenization uses longest-first truncation to 512 tokens, affecting 117 of 1,920 inputs. Its [model card](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) lists WebGPT comparisons, summarization feedback, synthetic instruction pairs and Anthropic HH as training sources. These predate HelpSteer2; older-source prompt overlap and Qwen's instruction-training overlap are unknown.

The primary metric is mean `reward(Pareto) - reward(gap)` over prompts and matched seeds. The analysis makes 10,000 crossed paired bootstrap draws with seed 20261007: resample prompt IDs and seed IDs independently, reusing both draws for the two conditions. The 95% interval uses percentile endpoints with linear interpolation. An interval wholly above zero favors Pareto; wholly below zero favors gap; otherwise the comparison is inconclusive.

The observed difference is −0.11165175386213377 logits, with interval [−0.2567360822823427, 0.031770176391119544]. The per-seed means are −0.027171047372879304, −0.11790505394583306 and −0.18987916026768895. Secondary outputs include rewards, win/tie/loss counts against the starting model, and response lengths. No length adjustment is applied.

The `generations-*.jsonl.gz` files contain generated text and hashes; `generation-metadata-*.json` identifies checkpoints and evaluation sets. `records.jsonl.gz` adds reward scores. [analyze_day_dpo.py](../tools/analyze_day_dpo.py) rejects missing, duplicate, nonfinite or unbalanced observations and checks the exact evaluation prompt IDs.

```bash
uv run python tools/analyze_day_dpo.py --output-dir .cache/day-dpo-analysis
```

The [README](../README.md#scope) summarizes the sample-size, reward-model and truncation limits.

### Cloud execution

Training needs a configured Modal account and paid L4 GPUs. The Python 3.12 client runs no local model inference. The container pins PyTorch, Transformers, TRL, PEFT, datasets and Accelerate. Model downloads use a CPU container and the `verge-lab-day-hf` volume; adapters and completed outputs use `verge-lab-day-adapters`.

```bash
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode cache
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode pilot --size 1.5B
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode full --size 1.5B
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode score --size 1.5B
uv run python tools/analyze_day_dpo.py --output-dir .cache/day-dpo-analysis
```

Review the code and account costs first. Full training uses up to three parallel L4 containers with a 30-minute timeout per call; scoring uses one. Completed adapters, generations and scores are saved in the volume before returning to the client. Reruns recover completed outputs on CPU and skip their GPU work. Saved matching adapters can resume interrupted generation. Use a new adapter volume to retrain from scratch.

The Python 3.12 client and plain-string scheduler metadata avoid failures encountered in earlier attempts. [transport-recovery.json](../results/day-dpo/transport-recovery.json) describes those failures.

After saving results, remove downloaded base-model and scorer weights while keeping adapters and completed outputs:

```bash
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode cleanup
```

[cache-cleanup.json](../results/day-dpo/cache-cleanup.json) records the original cleanup. Offline analysis is unaffected; later generation or scoring requires `--mode cache` first.

## Earlier GPU smoke run

[modal_app.py](../modal_app.py) contains the earlier Qwen3-0.6B generation and LoRA DPO entrypoints. It uses a pinned checkpoint on an L4 with separate volumes for weights and adapters. These commands require a configured Modal account and paid GPU resources:

```bash
uv sync --extra modal --dev
uv run modal run modal_app.py --smoke
uv run modal run modal_app.py --train --pairs-jsonl artifacts/my-run.dpo.jsonl --adapter-name my-first-adapter --adapter-version v1
```

[modal-smoke.json](../artifacts/modal-smoke.json) contains the generation smoke result. [modal-training.json](../artifacts/modal-training.json) contains a two-pair, one-epoch run: 3.3742 seconds and training loss 0.6931471824645996, approximately ln 2. It checks the entrypoint; it has no held-out model evaluation.

## Checks

```bash
uv run pytest
uv run ruff check .
npm run check
```

Tests cover Pareto direction and boundary cases, abstention, stable ordering, artifact validation, mutation behavior, CLI specifications, export recomputation, agreement between `verge demo` and the browser artifact, and the human-preference and DPO analyses.
