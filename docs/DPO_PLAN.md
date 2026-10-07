# Does Pareto pair selection improve DPO training?

I will compare equal-size training sets rather than treating score agreement as evidence of better trained models. I commit this plan before any training or reward results.

## Data and fixed selectors

I use pinned HelpSteer2 revision `990b2711a36180dd19d9c94b8627844866f8982a` (CC-BY-4.0), joining the recorded ratings and separately collected preferences by exact prompt and response hashes. All training conditions use the same eligible pool: train split, single-turn prompts (no `<extra_id_1>` markers), nonzero helpfulness gap, nonzero human preference, prompt at most 4,000 characters, and both responses at most 8,000 characters. Eligibility is applied before selection; the existing correctness/coherence Pareto rule is not changed.

I fix **N = 1,024 pairs per condition**, reused across seeds:

- **Pareto:** all defended correctness/coherence pairs in that pool, subsampled by ascending SHA-256 of `20261007:pair_id`; direction from Pareto dominance.
- **Gap:** largest absolute helpfulness gaps in the same pool, with the same hash order for equal gaps; direction from helpfulness.
- **Human reference:** a hash-ordered sample of the same pool; direction from the separate human pairwise preference sign.

I retain full original texts, selection IDs, source hashes, directions and set overlaps in `results/day-dpo/data.json.gz` and `data-metadata.json`. The human condition is a reference, not part of the primary superiority decision. Conditions need not select the same pairs; equal size is the constraint. This bounded experiment is not a rerun of the previous all-eligible agreement table.

I fix **192 distinct single-turn validation prompts**, at most 4,000 characters, ordered by the same hash rule on their prompt hashes. I exclude any prompt hash present in **any** HelpSteer2 train rating row, not merely the selected training sets. The data manifest commits the exact evaluation IDs before training. Evaluation uses prompts only, never validation responses or labels for selection or tuning.

## Starting model and compute-only pilot

The default is `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0). I will use `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, if a compute-only pilot estimates that the complete nine-run experiment and evaluation fit within **90 L4 minutes** and the **$4.50** allocation, leaving room for failures. This size decision depends on time/memory only, not reward scores.

The pilot trains the first 64 Pareto-selected pairs for one epoch with seed 1701 and the hyperparameters below, then measures greedy generation throughput on eight training prompts. Pilot adapters are discarded and never used as final models. I record training/generation rates and use them to estimate the full workload; I do not score pilot generations or inspect validation rewards. Both candidate models may be piloted. The selected size and measured rationale are written before the complete experiment begins.

I download model weights in a cheap CPU Modal container into `verge-lab-day-hf`; GPUs mount that volume and do not download weights. Final adapters are retained in `verge-lab-day-adapters`. Jobs use at most the booked container count, with explicit timeouts. No model runs occur on the laptop.

## Identical training settings

I use TRL DPO with one fresh LoRA adapter per condition/seed, always starting from the same pretrained checkpoint. Seeds are **1701, 1702, 1703**, paired across conditions. Fixed settings:

| Setting | Value |
| --- | --- |
| Epochs | 1 |
| Learning rate | 0.00005 |
| DPO beta / loss | 0.1 / sigmoid |
| Per-device batch / accumulation | 2 / 8 (effective 16) |
| Maximum prompt / full sequence tokens | 384 / 768 |
| Optimizer / weight decay | AdamW / 0.01 |
| LR schedule / warmup | linear / 0 |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| LoRA targets | all linear layers |
| Precision | bfloat16 |
| Gradient checkpointing | enabled |
| Reference | same starting checkpoint, adapter disabled |

I use the model's chat template for training and generation. Long sequences are truncated identically by TRL; labels were assigned to full responses, so truncation is a limitation. I record actual optimizer steps, losses, training-set hashes, dependency versions and checkpoint identifiers. No hyperparameter search or selection by reward is allowed.

## Independent reward evaluation

For each of the nine trained models and the untrained start, I generate on exactly the fixed 192 prompts: greedy decoding, no sampling, at most **192 new tokens**, generation batch size **8**, prompt limit **384** tokens. I record full generated text, response-token count and text hash. There is one common start generation per prompt, not nine duplicated reference observations.

My independent scorer is `OpenAssistant/reward-model-deberta-v3-large-v2`, revision `c355404efa9ad2ad069f3a197cae0523c14244fc` (MIT). Its [card](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2/blob/c355404efa9ad2ad069f3a197cae0523c14244fc/README.md) lists WebGPT comparisons, summarization feedback, synthetic instruction pairs and Anthropic HH. Its checkpoint was last modified in February 2023, before HelpSteer2's 2024 release, and its declared training data excludes HelpSteer2. I choose it because it is small enough for a bounded independent evaluation and accepts question/answer pairs directly. It is an older and imperfect reward proxy, not a human preference judge.

I score the raw prompt and generated answer as the card specifies, using a single scalar logit, evaluation mode, no gradients, and joint tokenization with longest-first truncation to **512 tokens**. I record truncation counts. Direct HelpSteer2 training overlap is excluded by the declared sources and chronology; exact or topical overlap with older source prompts, shared general-web pretraining and unknown Qwen instruction data cannot be ruled out. This is disclosed, not treated as clean independent human evaluation.

## Primary metric and decision

For each prompt and matched seed, I compute **reward(Pareto) − reward(gap)**. The primary estimate is the mean over all prompts and three seeds. I use **10,000 crossed paired bootstrap draws**, resampling the three seed indices and the 192 prompt indices independently, while reusing both index draws for the two conditions; bootstrap seed is **20261007**. The 95% interval is the percentile interval over those mean differences.

I call Pareto better only if the lower endpoint is strictly above zero; I call gap better only if the upper endpoint is strictly below zero. Otherwise the result is inconclusive. The decision concerns this reward proxy, model, sample and fixed training recipe only; it does not establish general alignment quality. Three seeds give weak information about training variance, which I disclose.

Secondary reports: per-seed paired differences; reward mean by condition; win/tie/loss rates against the common untrained start using raw reward comparisons; mean and median response length in tokens, with per-seed means and paired length differences. Length is descriptive, not a post-hoc correction or alternative primary metric. I commit compressed raw generations and scores, the analysis script and an SVG figure. I do not select checkpoints, prompts, seeds or a reward model after viewing results.
