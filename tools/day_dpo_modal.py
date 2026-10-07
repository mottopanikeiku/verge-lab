"""Equal-size DPO experiment; weights and adapters stay in named Modal Volumes."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "day-dpo" / "data.json.gz"
RESULTS = ROOT / "results" / "day-dpo"
MODELS = {
    "0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
}
REWARD = ("OpenAssistant/reward-model-deberta-v3-large-v2",
          "c355404efa9ad2ad069f3a197cae0523c14244fc")
SEEDS = (1701, 1702, 1703)
CONDITIONS = ("pareto", "gap", "human")
VERSIONS = {
    "torch": "2.8.0", "transformers": "4.56.2", "trl": "0.23.1",
    "peft": "0.17.1", "datasets": "4.1.1", "accelerate": "1.10.1",
}
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(*(f"{name}=={version}" for name, version in VERSIONS.items()),
                      "sentencepiece==0.2.1", "protobuf==6.32.1")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false",
               "OMP_NUM_THREADS": "2", "HF_HUB_DISABLE_TELEMETRY": "1"}))
app = modal.App("verge-lab-day-dpo")
weights = modal.Volume.from_name("verge-lab-day-hf", create_if_missing=True)
adapters = modal.Volume.from_name("verge-lab-day-adapters", create_if_missing=True)
VOLUMES = {"/cache": weights, "/adapters": adapters}


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_compressed(path: Path, rows: list[dict]) -> None:
    content = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
                       for row in rows)
    path.write_bytes(gzip.compress(content, mtime=0))


def save_object(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode()
    path.write_bytes(gzip.compress(content, mtime=0))


@app.function(image=image, cpu=2, memory=8192, timeout=300, max_containers=1,
              volumes={"/adapters": adapters})
def completed_results(payload: bytes, size: str) -> dict:
    """Recover completed units on CPU before allocating any new GPU work."""
    adapters.reload()
    data = json.loads(gzip.decompress(payload))
    models = []
    units = [("start", 1701), *((c, s) for c in CONDITIONS for s in SEEDS)]
    for condition, seed in units:
        path = Path(f"/adapters/{size}/results/{condition}-{seed}.json.gz")
        if not path.exists():
            continue
        result = json.loads(gzip.decompress(path.read_bytes()))
        if (
            result["model"] != list(MODELS[size])
            or result["evaluation_sha256"] != digest(data["evaluation"])
            or result["condition"] != condition
            or result["seed"] != (None if condition == "start" else seed)
        ):
            raise ValueError("Saved generations do not match the fixed experiment")
        if condition != "start":
            metadata = result["training"]
            if metadata["training_data_sha256"] != digest(data["train"][condition]):
                raise ValueError("Saved training set does not match the fixed experiment")
        models.append(result)
    score_path = Path(f"/adapters/{size}/results/scores.json.gz")
    score = json.loads(gzip.decompress(score_path.read_bytes())) if score_path.exists() else None
    return {"models": models, "score": score}


@app.function(image=image, cpu=2, memory=8192, timeout=60, max_containers=1,
              volumes={"/cache": weights})
def cleanup_weights() -> dict:
    """Remove downloaded weights after results are committed; retain all adapters."""
    import shutil

    weights.reload()
    root = Path("/cache/hf")
    present_before = root.exists()
    if present_before:
        shutil.rmtree(root)
    weights.commit()
    weights.reload()
    return {
        "volume": "verge-lab-day-hf", "cache_present_before": present_before,
        "cache_present_after": root.exists(), "adapters_untouched": True,
    }


@app.function(image=image, cpu=2, memory=8192, timeout=900, max_containers=1,
              volumes={"/cache": weights})
def cache_models() -> dict:
    """Download all candidate weights on CPU, never in a billed GPU function."""
    import inspect

    from huggingface_hub import snapshot_download
    from trl import DPOConfig, DPOTrainer

    files = {}
    for model, revision in (*MODELS.values(), REWARD):
        path = snapshot_download(
            model, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin",
                            "README.md", "LICENSE"],
        )
        files[model] = {"revision": revision, "cache_path": path,
                        "files": sorted(p.name for p in Path(path).iterdir())}
    weights.commit()
    return {"models": files, "versions": VERSIONS,
            "dpo_config_parameters": sorted(inspect.signature(DPOConfig).parameters),
            "dpo_trainer_parameters": sorted(inspect.signature(DPOTrainer).parameters)}


def tokenizer_for(size: str):
    from transformers import AutoTokenizer

    model, revision = MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def fresh_model(size: str):
    import torch
    from transformers import AutoModelForCausalLM

    model, revision = MODELS[size]
    return AutoModelForCausalLM.from_pretrained(
        model, revision=revision, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")


def train_adapter(records: list[dict], size: str, condition: str, seed: int,
                  output: str) -> tuple[object, object, dict]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(seed)
    torch.set_num_threads(2)
    tokenizer = tokenizer_for(size)
    tokenizer.padding_side = "right"
    model = fresh_model(size)
    dataset = Dataset.from_list([{
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    } for row in records])
    config = DPOConfig(
        output_dir=output, num_train_epochs=1, learning_rate=5e-5, beta=0.1,
        loss_type="sigmoid", per_device_train_batch_size=2, gradient_accumulation_steps=8,
        max_prompt_length=384, max_length=768, max_grad_norm=1.0,
        optim="adamw_torch", weight_decay=0.01, lr_scheduler_type="linear", warmup_ratio=0.0,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=8, save_strategy="no", report_to="none", seed=seed, data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = DPOTrainer(
        model=model, args=config, processing_class=tokenizer, train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                              task_type="CAUSAL_LM", target_modules="all-linear"),
    )
    before = [parameter.detach().float().cpu().clone() for name, parameter in
              trainer.model.named_parameters() if parameter.requires_grad and "lora_B" in name]
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = [parameter.detach().float().cpu() for name, parameter in
             trainer.model.named_parameters() if parameter.requires_grad and "lora_B" in name]
    changed = sum(
        float((left - right).abs().sum()) for left, right in zip(before, after, strict=True)
    )
    if not changed > 0:
        raise RuntimeError("Training did not change LoRA weights")
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    logs = [{key: value for key, value in log.items()
             if value is None or isinstance(value, (str, int, float, bool))}
            for log in trainer.state.log_history]
    metadata = {
        "condition": condition, "seed": seed, "size": size, "model": MODELS[size][0],
        "revision": MODELS[size][1], "pair_count": len(records),
        "training_data_sha256": digest(records),
        "optimizer_steps": trainer.state.global_step, "training_seconds": elapsed,
        "lora_weight_l1_change": changed, "metrics": result.metrics, "logs": logs,
        "versions": VERSIONS, "gpu": torch.cuda.get_device_name(0),
        "reference_mode": "Starting checkpoint with the LoRA adapter disabled",
        "hyperparameters": {
            key: getattr(config, key) for key in (
                "learning_rate", "beta", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "max_prompt_length", "max_length",
                "weight_decay", "warmup_ratio", "seed", "data_seed",
            )
        },
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "targets": "all-linear"},
    }
    metadata["hyperparameters"]["lr_scheduler_type"] = config.lr_scheduler_type.value
    Path(output, "experiment.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    trained = trainer.model
    del trainer
    return trained, tokenizer, metadata


def generate(model, tokenizer, prompts: list[dict], condition: str, seed: int | None) -> list[dict]:
    import torch

    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    output = []
    for offset in range(0, len(prompts), 8):
        group = prompts[offset:offset + 8]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True,
        ) for row in group]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=384,
                            return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=192,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        for row, tokens in zip(group, generated[:, encoded.input_ids.shape[1]:], strict=True):
            values = tokens.tolist()
            count = values.index(tokenizer.eos_token_id) + 1 \
                if tokenizer.eos_token_id in values else len(values)
            response = tokenizer.decode(values[:count], skip_special_tokens=True)
            output.append({
                "condition": condition, "seed": seed, "prompt_id": row["prompt_id"],
                "prompt": row["prompt"], "response": response, "response_tokens": count,
                "generation_sha256": hashlib.sha256(response.encode()).hexdigest(),
            })
    return output


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=300, max_containers=1,
              volumes=VOLUMES)
def pilot(payload: bytes, size: str) -> dict:
    import gc

    import torch

    weights.reload()
    data = json.loads(gzip.decompress(payload))
    rows = data["train"]["pareto"][:64]
    model, tokenizer, training = train_adapter(rows, size, "pilot", 1701, f"/tmp/pilot-{size}")
    prompts = [{"prompt_id": row["prompt_sha256"], "prompt": row["prompt"]} for row in rows[:8]]
    started = time.perf_counter()
    generated = generate(model, tokenizer, prompts, "pilot", 1701)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    estimated = training["training_seconds"] * 16 * 9 + elapsed * 24 * 10 + 300
    return {"kind": "compute_only_pilot", "size": size, "training": training,
            "generation_seconds_8_prompts": elapsed,
            "generated_tokens": sum(row["response_tokens"] for row in generated),
            "estimated_full_seconds_with_reward_allowance": estimated,
            "evaluated_reward": False, "gpu": "L4"}


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=1800, max_containers=3,
              volumes=VOLUMES)
def train_and_generate(payload: bytes, size: str, condition: str, seed: int) -> dict:
    import gc

    import torch
    from peft import PeftModel

    weights.reload()
    adapters.reload()
    data = json.loads(gzip.decompress(payload))
    destination = f"/adapters/{size}/{condition}/{seed}"
    marker = Path(destination, "experiment.json")
    if condition == "start":
        model, tokenizer, metadata = fresh_model(size), tokenizer_for(size), None
    elif marker.exists():
        metadata = json.loads(marker.read_text())
        if (
            metadata["training_data_sha256"] != digest(data["train"][condition])
            or metadata["revision"] != MODELS[size][1]
            or metadata["pair_count"] != 1024
        ):
            raise ValueError("Existing adapter does not match the fixed experiment")
        model = PeftModel.from_pretrained(fresh_model(size), destination)
        tokenizer = tokenizer_for(size)
    else:
        model, tokenizer, metadata = train_adapter(data["train"][condition], size, condition,
                                                   seed, destination)
        adapters.commit()
    rows = generate(model, tokenizer, data["evaluation"], condition,
                    None if condition == "start" else seed)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "training": metadata, "generations": rows, "model": MODELS[size],
        "condition": condition, "seed": None if condition == "start" else seed,
        "evaluation_sha256": digest(data["evaluation"]),
    }
    save_object(Path(f"/adapters/{size}/results/{condition}-{seed}.json.gz"), result)
    adapters.commit()
    return result


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=1800, max_containers=1,
              volumes=VOLUMES)
def score_generations(rows: list[dict], size: str) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    weights.reload()
    adapters.reload()
    input_sha256 = digest(rows)
    tokenizer = AutoTokenizer.from_pretrained(
        REWARD[0], revision=REWARD[1], local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD[0], revision=REWARD[1], local_files_only=True,
    ).to("cuda").eval()
    truncations = 0
    for offset in range(0, len(rows), 8):
        batch = rows[offset:offset + 8]
        questions, answers = [row["prompt"] for row in batch], [row["response"] for row in batch]
        lengths = tokenizer(questions, answers, truncation=False)["input_ids"]
        flags = [len(tokens) > 512 for tokens in lengths]
        truncations += sum(flags)
        encoded = tokenizer(questions, answers, padding=True, truncation="longest_first",
                            max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            scores = model(**encoded).logits[:, 0].float().cpu().tolist()
        for row, score, truncated in zip(batch, scores, flags, strict=True):
            row["reward"] = score
            row["reward_input_truncated"] = truncated
            row["reward_model"] = REWARD[0]
            row["reward_revision"] = REWARD[1]
    result = {
        "rows": rows, "reward_truncated_count": truncations, "reward_model": REWARD,
        "versions": VERSIONS, "generations_sha256": input_sha256,
    }
    save_object(Path(f"/adapters/{size}/results/scores.json.gz"), result)
    adapters.commit()
    return result


def store_training_result(result: dict) -> None:
    metadata = result["training"]
    name = "start" if metadata is None else f"{metadata['condition']}-{metadata['seed']}"
    save_compressed(RESULTS / f"generations-{name}.jsonl.gz", result["generations"])
    if metadata is not None:
        (RESULTS / f"training-{name}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    details = {key: result[key] for key in ("model", "condition", "seed", "evaluation_sha256")}
    (RESULTS / f"generation-metadata-{name}.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n")


@app.local_entrypoint()
async def main(
    mode: str = "cache", size: str = "0.5B", condition: str = "pareto", seed: int = 1701,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if mode == "cache":
        result = await cache_models.remote.aio()
        (RESULTS / "cache-models.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif mode == "pilot":
        result = await pilot.remote.aio(DATA.read_bytes(), size)
        (RESULTS / f"pilot-{size}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2))
    elif mode == "train":
        result = await train_and_generate.remote.aio(DATA.read_bytes(), size, condition, seed)
        store_training_result(result)
    elif mode == "full":
        payload = DATA.read_bytes()
        completed = await completed_results.remote.aio(payload, size)
        done = {(result["condition"], result["seed"]) for result in completed["models"]}
        for result in completed["models"]:
            store_training_result(result)
        if ("start", None) not in done:
            result = await train_and_generate.remote.aio(payload, size, "start", 1701)
            store_training_result(result)
        requests = [(payload, size, c, s) for c in CONDITIONS for s in SEEDS if (c, s) not in done]
        async for result in train_and_generate.starmap.aio(requests, order_outputs=False):
            store_training_result(result)
    elif mode == "score":
        rows = []
        for name in ("start", *(f"{c}-{s}" for c in CONDITIONS for s in SEEDS)):
            path = RESULTS / f"generations-{name}.jsonl.gz"
            rows.extend(
                json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
            )
        completed = await completed_results.remote.aio(DATA.read_bytes(), size)
        result = completed["score"]
        if result is None:
            result = await score_generations.remote.aio(rows, size)
        elif result["generations_sha256"] != digest(rows) or result["reward_model"] != list(REWARD):
            raise ValueError("Saved scores do not match these generations and reward model")
        save_compressed(RESULTS / "records.jsonl.gz", result.pop("rows"))
        (RESULTS / "reward-metadata.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif mode == "cleanup":
        result = await cleanup_weights.remote.aio()
        (RESULTS / "cache-cleanup.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        raise ValueError("mode must be cache, pilot, train, full, score or cleanup")
