# Workflows and assumptions

The browser imports `artifacts/demo-run.json`. It is a deterministic example built from `examples/candidates.json`, not a model evaluation. Candidate scores and judge confidences are authored inputs. Checkpoint reward, win-rate and loss curves, model metadata, GPU time and cost estimates are illustrative. Pair decisions and mutation checks are real computations on those example inputs.

## Local example and review UI

Python 3.11+ and uv are needed for the miner; Node.js 22+ and npm are needed for the UI.

```bash
uv sync --locked --dev
uv run verge demo --output artifacts/demo-run.json
npm ci
npm run dev
```

The UI has an overview, a candidate partial-order graph with a data table, a pair review view, and an example configuration/checkpoint view. I removed the decorative WebGL instruments so the explanation and pair decisions do not compete with unrelated graphics.

### GitHub Pages

The Vite base is `/verge-lab/` in `vite.config.ts`, so production scripts, styles and the favicon use the project prefix. To review the same production build locally:

```bash
npm ci
npm run build
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/verge-lab/`. The `pages.yml` workflow builds with the committed npm lockfile on pushes to `main` or manual dispatch, uploads `dist` with `actions/upload-pages-artifact`, and deploys with `actions/deploy-pages`. It does not enable Pages or change repository settings; the repository must already use GitHub Actions as its Pages source.

The npm lockfile pins the React UI and its build tools. The app does not load remote executable scripts or ThreeUI iframe scenes.

The older [browser smoke record](assets/pages-smoke.json) and [pair-review screenshot](assets/pages-pair-lab.png) describe the earlier UI. They are retained as historical checks, not verification of the current interface.

The current [Chromium assertions](../tools/check_demo.mjs) exercise keyboard navigation, filters, pair details and page overflow at desktop and mobile widths, and reject browser errors or console warnings. With a preview server already running, install the optional browser dependency without changing the lockfile and run:

```bash
npm install --no-save --package-lock=false puppeteer
node tools/check_demo.mjs http://127.0.0.1:4173/verge-lab/
```

Puppeteer downloads Chromium if needed. Close the preview server after the check; `npm ci` restores the locked dependency tree after this optional install.


## Analyze and export

```bash
uv run verge analyze examples/candidates.json --output artifacts/my-run.json
uv run verge export-dpo artifacts/my-run.json --output artifacts/my-run.dpo.jsonl
```

`analyze` trusts supplied scores and confidences. It does not verify a judge's identity, reliability, or calibration. The local package runs no model inference and executes no generated code. Built-in checkers inspect phrases, JSON fields, and regular expressions; built-in scorers use keywords, phrases, JSON fields, and word counts. These checks do not establish semantic correctness.

For each objective, `verge_lab/pareto.py` changes the sign of minimization scores so higher is better; it does not rescale objectives to equal ranges. The conservative margin is the directional score difference minus `uncertainty_scale * ((1 - left_confidence) + (1 - right_confidence))`. Defaults are uncertainty scale 0.05, minimum confidence 0.8, and strict margin epsilon 0.01, in the objective's native units. This is an assumed penalty, not a statistical confidence interval. A defended edge requires all margins to be nonnegative and at least one to exceed epsilon. Missing scores, low confidence, ties on every objective, or tradeoffs lead to abstention. With any nonzero penalty, an exact tie on one objective also blocks dominance.

`verge_lab/export.py` recomputes each comparison with the default thresholds and refuses artifacts whose stored evidence differs. Artifacts do not record thresholds, so an artifact mined with other settings is refused whenever those settings change a stored decision or margin. It exports only defended edges in TRL-compatible JSONL. This protects consistency, not the truth of externally supplied scores.

Artifacts contain run/model metadata, summary counts, prompt rubrics and constraints, candidate scores and projection coordinates, pair decisions, mutation results, and optional checkpoint inputs. Content hashes and deterministic ordering make repeat runs comparable; hashes are not proof of valid scoring.

## Human ratings and preferences

```bash
uv run python tools/compare_human_preferences.py
uv run python tools/compare_human_preferences.py --fetch --cache .cache/helpsteer2
```

The first command uses the committed compact source without the network. The second downloads the pinned original rating and preference files. `source-metadata.json` records their revision, hashes, join counts and exclusions. I join each human preference pair to both rating responses by the exact prompt and response text hashes, checking that the split agrees after mapping the preference file's `val` label to the rating file's `validation`. Identical duplicate rating keys are allowed only when the split and all scores agree; conflicting duplicates stop the analysis.

The primary Pareto rule maximizes **correctness and coherence**, rescaled from 0–4 to 0–1. I leave helpfulness out because it is the dataset's overall rating and supplies the baseline. Complexity and verbosity measure style, not universal quality: blindly maximizing them would reward sophistication and length regardless of the prompt. I also report the three quality aspects together and naive five-aspect maximization as sensitivities, without selecting the best variant.

The human reference is the separately collected `preference_strength`, not helpfulness and not an aspect average. Its sign is positive when response 2 is preferred. Human ties remain in selected counts; I separately report strict preference agreement, contradictions among all selected pairs, and tie counts. The ratings and preferences come from the same dataset population, so separate collection is not a claim of fully independent annotators or unbiased ground truth.

For equal-yield comparison, I exclude helpfulness ties from **both** selectors. Otherwise the gap baseline cannot choose a direction, and forcing a winner would be misleading. Within each split, Pareto selects all defended pairs in that common eligible pool; the baseline selects exactly as many pairs by descending absolute helpfulness gap and predicts the gap's sign. At equal gaps I use a label-blind SHA-256 ordering with a fixed seed, and report variation across twenty seeds. The baseline's agreement with helpfulness itself is tautological; its agreement with the separate human preferences is the useful test.

I report train and validation separately and do not tune on validation. Wilson intervals describe nominal binomial variation of agreement, not uncertainty in human annotations or a test of one selector's superiority. This label-agreement comparison uses no model inference or paid compute; the separate DPO experiment below tests trained models.

HelpSteer2 and HelpSteer2-Preference are by NVIDIA, Scale AI and Zhilin Wang et al., released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). I retain their card and source links; compact files omit text but preserve scores, strengths and identifying hashes. HelpSteer3 also declares CC-BY-4.0, but its preferences and free-text feedback do not supply matched numeric multi-aspect ratings. I do not invent aspect scores for it.


## Equal-size DPO training experiment

I fix the training sets, evaluation prompts and decision rule in [DPO_PLAN.md](DPO_PLAN.md). The original plan/data commit is `12c887e`; `3b062bd` records the compute-only amendment and selected 1.5B model before the complete experiment. The amendment explicitly acknowledges that the larger model's unscored pilot exceeded the original aggregate-minute ceiling. It changes no training or evaluation choice.

The committed `results/day-dpo/data.json.gz` contains full training texts for three 1,024-pair conditions and 192 held-out prompts. `data-metadata.json` records source hashes, eligibility, pair IDs, prompt IDs and overlaps. To rebuild those inputs from the pinned upstream cache:

```bash
uv run python tools/compare_human_preferences.py --fetch --cache .cache/helpsteer2
uv run python tools/prepare_day_dpo.py --cache .cache/helpsteer2
```

Training requires a configured Modal account and paid L4 GPUs. The client runs no local model inference. Its container image pins PyTorch, Transformers, TRL, PEFT, datasets and Accelerate. The cache command downloads both candidate models and the independent scorer using only a CPU container; GPU functions load locally from `verge-lab-day-hf`. Nine adapters are saved separately in `verge-lab-day-adapters`.

```bash
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode cache
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode pilot --size 1.5B
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode full --size 1.5B
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode score --size 1.5B
uv run python tools/analyze_day_dpo.py
```

Review the code and account costs first: complete training can use three parallel L4 containers, with a 30-minute timeout per call; scoring uses one. The pilot is compute-only and its adapter is not reused. I use a Python 3.12 client after encountering a Python 3.14 transport failure in the first attempt. Each completed unit is saved inside the adapter Volume before being returned; reruns recover completed generations and scores on CPU and skip their GPU work. An already saved matching adapter can also resume interrupted generation. To retrain rather than resume, use a new adapter volume in your own account.

`generations-*.jsonl.gz` retain every generated answer and text hash; `generation-metadata-*.json` identify the pinned checkpoint and evaluation set. `training-*.json` retain actual optimizer progress, loss, adapter changes and settings. `records.jsonl.gz` binds the independent reward to each response. Analysis rejects missing, duplicate, nonfinite or unbalanced records and checks the exact committed prompt IDs. `summary.json` and `reward-difference.svg` are computed from those raw scores with the planned crossed prompt/seed bootstrap. Response length counts generated Qwen tokens through the first EOS, including EOS when present; scorer-input truncation is reported rather than hidden.

The reward model is an older proxy trained before HelpSteer2, not a human judge. Its declared training sources exclude HelpSteer2, but older-source prompt overlap and shared pretraining cannot be ruled out. Three training seeds, 1,024 pairs, truncation and a 192-token generation cap limit what the result can establish.

After committing the complete generations, scores and summaries, I remove the downloaded base-model/scorer cache while retaining the learned adapters and completed result units:

```bash
uvx --python 3.12 --from modal==1.5.3 modal run tools/day_dpo_modal.py --mode cleanup
```

`cache-cleanup.json` records whether the cache existed before and after deletion. Offline analysis still works; a later generation or scoring run must first restore the pinned downloads with `--mode cache`.


## Historical GPU smoke run

`modal_app.py` retains the previous generation and LoRA DPO entrypoints. They need a configured Modal account and paid GPU resources, and are **not needed** for the CPU result or UI. No GPU work was performed for the public-data mining result.

```bash
uv sync --extra modal --dev
uv run modal run modal_app.py --smoke
uv run modal run modal_app.py --train --pairs-jsonl artifacts/my-run.dpo.jsonl --adapter-name my-first-adapter --adapter-version v1
```

The default model is Qwen/Qwen3-0.6B on an L4, pinned to a Hugging Face revision and safetensors. Weights and adapters use separate Modal volumes. Review the code and costs before using these commands; credentials must remain local.

`artifacts/modal-smoke.json` records a generation smoke test. `artifacts/modal-training.json` records one epoch on two DPO pairs: 3.3742 seconds and training loss 0.6931471824645996 (approximately ln 2). It demonstrates that the training entrypoint ran, not learning or improved model quality. There is no held-out evaluation supporting the illustrative UI curves.

## Checks

```bash
uv run pytest
uv run ruff check .
npm run check
```

The tests cover Pareto direction handling, tie, epsilon and confidence-penalty boundaries, abstention, stable pair ordering, artifact validation, mutation behavior, CLI specifications, export recomputation, that `artifacts/demo-run.json` matches `verge demo`, and the recorded human-preference and DPO analyses. They do not measure model performance.
