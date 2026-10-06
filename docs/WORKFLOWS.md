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

The UI has an overview, a candidate partial-order graph with a data table, a pair review view, and an example configuration/checkpoint view. ThreeUI provides ambient visual instruments, not model internals or measurements. Its upstream runtime assets run in opaque-origin `allow-scripts` iframes; candidate data is not passed into them. Deployments with stricter privacy requirements should replace those assets with vetted self-hosted ones.

### GitHub Pages

The Vite base is `/verge-lab/` in `vite.config.ts`, so production scripts, styles and the favicon use the project prefix. To review the same production build locally:

```bash
npm ci
npm run build
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/verge-lab/`. The `pages.yml` workflow builds with the committed npm lockfile on pushes to `main` or manual dispatch, uploads `dist` with `actions/upload-pages-artifact`, and deploys with `actions/deploy-pages`. It does not enable Pages or change repository settings; the repository must already use GitHub Actions as its Pages source.

The npm lockfile pins installed dependencies, including ThreeUI 0.3.0. Its three used iframe scenes reference Tailwind, Iconify, GSAP and Three.js. A Vite transform pins Tailwind to 3.4.17 and adds SHA-384 integrity checks to the already-versioned Iconify 1.0.7, GSAP 3.12.2 and Three.js r128 scripts. Tailwind's CDN endpoint does not send a CORS header, so it cannot support cross-origin SRI. Google Fonts stylesheets and decorative image endpoints remain remote assets, not pinned executable dependencies.

The checked-in [browser smoke record](assets/pages-smoke.json) and [pair-review screenshot](assets/pages-pair-lab.png) come from the prefixed production build in headless Chromium. The run visited every view, filtered 12 comparisons to 10 ambiguous pairs, opened pair details, exercised an empty candidate search, and restored the register. It recorded no console errors, uncaught page errors, failed requests or HTTP error responses.

## Analyze and export

```bash
uv run verge analyze examples/candidates.json --output artifacts/my-run.json
uv run verge export-dpo artifacts/my-run.json --output artifacts/my-run.dpo.jsonl
```

`analyze` trusts supplied scores and confidences. It does not verify a judge's identity, reliability, or calibration. The local package runs no model inference and executes no generated code. Built-in checkers inspect phrases, JSON fields, and regular expressions; built-in scorers use keywords, phrases, JSON fields, and word counts. These checks do not establish semantic correctness.

For each objective, `verge_lab/pareto.py` changes the sign of minimization scores so higher is better; it does not rescale objectives to equal ranges. The conservative margin is the directional score difference minus `uncertainty_scale * ((1 - left_confidence) + (1 - right_confidence))`. Defaults are uncertainty scale 0.05, minimum confidence 0.8, and strict margin epsilon 0.01, in the objective's native units. This is an assumed penalty, not a statistical confidence interval. A defended edge requires all margins to be nonnegative and at least one to exceed epsilon. Missing scores, low confidence, ties, or tradeoffs lead to abstention.

`verge_lab/export.py` recomputes each comparison and refuses artifacts whose stored evidence differs. It exports only defended edges in TRL-compatible JSONL. This protects consistency, not the truth of externally supplied scores.

Artifacts contain run/model metadata, summary counts, prompt rubrics and constraints, candidate scores and projection coordinates, pair decisions, mutation results, and optional checkpoint inputs. Content hashes and deterministic ordering make repeat runs comparable; hashes are not proof of valid scoring.

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

The tests cover Pareto direction handling, abstention, stable pair ordering, artifact validation, mutation behavior, CLI specifications, and export recomputation. They do not measure model performance.
