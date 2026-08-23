"""Modal GPU entrypoints for Verge Lab candidate generation and DPO LoRA training.

The heavy ML stack exists only in ``gpu_image`` and is imported inside remote
functions, so importing this module locally does not import torch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import modal

APP_NAME = "verge-lab"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SEED = 1701
SUPPORTED_MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}
HF_CACHE_VOLUME_NAME = "verge-hf-cache"
ADAPTER_VOLUME_NAME = "verge-adapters"
HF_CACHE_DIR = "/cache/huggingface"
ADAPTER_ROOT = "/adapters"

MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 16_384
MAX_COMPLETION_CHARS = 32_768
MAX_SAMPLES_PER_PROMPT = 4
MAX_NEW_TOKENS = 256
MAX_TRAINING_PAIRS = 512
MAX_JSONL_BYTES = 64 * 1024 * 1024
MAX_SEQUENCE_LENGTH = 2_048

# These versions are intentionally locked together. GPU dependencies are not
# project dependencies: Modal installs them in the remote Python 3.11 image.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.15.1",
        "trl==1.10.0",
        "peft==0.20.0",
        "datasets==5.0.1",
        "accelerate==1.14.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": HF_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
adapter_volume = modal.Volume.from_name(ADAPTER_VOLUME_NAME, create_if_missing=True)


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _nonempty_text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds the {maximum:,}-character limit")
    return value


def _validate_model_id(model_id: Any) -> str:
    model_id = _nonempty_text("model_id", model_id, 256)
    if model_id not in SUPPORTED_MODEL_REVISIONS:
        supported = ", ".join(sorted(SUPPORTED_MODEL_REVISIONS))
        raise ValueError(f"model_id must be one of the pinned supported models: {supported}")
    return model_id


def _validate_generation_request(
    prompts: Any,
    model_id: Any,
    samples_per_prompt: Any,
    max_new_tokens: Any,
    seed: Any,
    temperature: Any,
) -> tuple[list[str], str, int, int, int, float]:
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError(f"prompts is limited to {MAX_PROMPTS} items per GPU call")
    clean_prompts = [
        _nonempty_text(f"prompts[{index}]", prompt, MAX_PROMPT_CHARS)
        for index, prompt in enumerate(prompts)
    ]
    clean_model_id = _validate_model_id(model_id)
    clean_samples = _bounded_int(
        "samples_per_prompt", samples_per_prompt, 1, MAX_SAMPLES_PER_PROMPT
    )
    clean_max_tokens = _bounded_int("max_new_tokens", max_new_tokens, 1, MAX_NEW_TOKENS)
    clean_seed = _bounded_int("seed", seed, 0, 2**31 - 1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")
    clean_temperature = float(temperature)
    if not math.isfinite(clean_temperature) or not 0.05 <= clean_temperature <= 2.0:
        raise ValueError("temperature must be finite and between 0.05 and 2.0")
    return (
        clean_prompts,
        clean_model_id,
        clean_samples,
        clean_max_tokens,
        clean_seed,
        clean_temperature,
    )


def _validate_pair_records(records: Any) -> list[dict[str, str]]:
    if not isinstance(records, list) or not records:
        raise ValueError("training data must contain at least one preference pair")
    if len(records) > MAX_TRAINING_PAIRS:
        raise ValueError(f"training data is limited to {MAX_TRAINING_PAIRS} pairs per run")

    clean_records: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index + 1} must be a JSON object")
        missing = [field for field in ("prompt", "chosen", "rejected") if field not in record]
        if missing:
            raise ValueError(f"record {index + 1} is missing: {', '.join(missing)}")
        prompt = _nonempty_text(f"record {index + 1} prompt", record["prompt"], MAX_PROMPT_CHARS)
        chosen = _nonempty_text(
            f"record {index + 1} chosen", record["chosen"], MAX_COMPLETION_CHARS
        )
        rejected = _nonempty_text(
            f"record {index + 1} rejected", record["rejected"], MAX_COMPLETION_CHARS
        )
        if chosen == rejected:
            raise ValueError(f"record {index + 1} chosen and rejected completions are identical")
        clean_records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return clean_records


def _read_pair_jsonl(path_value: str, max_pairs: int) -> list[dict[str, str]]:
    max_pairs = _bounded_int("max_pairs", max_pairs, 1, MAX_TRAINING_PAIRS)
    path = Path(_nonempty_text("pairs_jsonl", path_value, 4_096)).expanduser()
    if not path.is_file():
        raise ValueError(f"preference JSONL is not a file: {path}")
    if path.stat().st_size > MAX_JSONL_BYTES:
        raise ValueError(f"preference JSONL exceeds the {MAX_JSONL_BYTES:,}-byte limit")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(records) >= max_pairs:
                raise ValueError(
                    f"preference JSONL contains more than the configured {max_pairs} pairs"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record on line {line_number} must be a JSON object")
            records.append(record)
    return _validate_pair_records(records)


def _safe_component(name: str, value: Any, *, version: bool = False) -> str:
    value = _nonempty_text(name, value, 64)
    pattern = r"v[0-9]+(?:[._-][A-Za-z0-9]+)*" if version else r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if re.fullmatch(pattern, value) is None:
        if version:
            raise ValueError(f"{name} must look like v1 or v1-experiment")
        raise ValueError(
            f"{name} may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=15 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume},
)
def generate_candidates(
    prompts: list[str],
    model_id: str = DEFAULT_MODEL,
    samples_per_prompt: int = 2,
    max_new_tokens: int = 128,
    seed: int = DEFAULT_SEED,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate a small, deterministically ordered candidate set on one L4."""
    (
        prompts,
        model_id,
        samples_per_prompt,
        max_new_tokens,
        seed,
        temperature,
    ) = _validate_generation_request(
        prompts, model_id, samples_per_prompt, max_new_tokens, seed, temperature
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_revision = SUPPORTED_MODEL_REVISIONS[model_id]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=HF_CACHE_DIR,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=HF_CACHE_DIR,
        dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    candidates: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            template_options: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if model_id.startswith("Qwen/Qwen3"):
                template_options["enable_thinking"] = False
            rendered_prompt = tokenizer.apply_chat_template(messages, **template_options)
            model_inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
            input_length = model_inputs["input_ids"].shape[1]

            for sample_index in range(samples_per_prompt):
                sample_seed = seed + prompt_index * samples_per_prompt + sample_index
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                generated = model.generate(
                    **model_inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    generated[0, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                identity = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "completion": completion,
                }
                candidates.append(
                    {
                        "id": f"candidate-{_stable_digest(identity)[:16]}",
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "prompt": prompt,
                        "completion": completion,
                        "seed": sample_seed,
                    }
                )

    hf_cache_volume.commit()
    return {
        "kind": "candidate-generation",
        "model_id": model_id,
        "device": "L4",
        "model_revision": model_revision,
        "seed": seed,
        "prompt_count": len(prompts),
        "samples_per_prompt": samples_per_prompt,
        "max_new_tokens": max_new_tokens,
        "candidates": candidates,
    }


@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=2 * 60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume, ADAPTER_ROOT: adapter_volume},
)
def train_dpo(
    records: list[dict[str, Any]],
    model_id: str = DEFAULT_MODEL,
    adapter_name: str = "verge-qwen3-0.6b-dpo",
    adapter_version: str = "v1",
    seed: int = DEFAULT_SEED,
    epochs: int = 1,
    max_length: int = 1_024,
) -> dict[str, Any]:
    """Train a PEFT LoRA adapter from validated prompt/chosen/rejected pairs."""
    records = _validate_pair_records(records)
    model_id = _validate_model_id(model_id)
    adapter_name = _safe_component("adapter_name", adapter_name)
    adapter_version = _safe_component("adapter_version", adapter_version, version=True)
    seed = _bounded_int("seed", seed, 0, 2**31 - 1)
    epochs = _bounded_int("epochs", epochs, 1, 3)
    max_length = _bounded_int("max_length", max_length, 128, MAX_SEQUENCE_LENGTH)

    import os
    import random
    import shutil
    import tempfile

    model_revision = SUPPORTED_MODEL_REVISIONS[model_id]

    destination = os.path.join(ADAPTER_ROOT, adapter_name, adapter_version)
    adapter_volume.reload()
    if os.path.exists(destination):
        raise FileExistsError(
            f"adapter version already exists: modal-volume://{ADAPTER_VOLUME_NAME}/"
            f"{adapter_name}/{adapter_version}"
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    conversational_records = [
        {
            "prompt": [{"role": "user", "content": record["prompt"]}],
            "chosen": [{"role": "assistant", "content": record["chosen"]}],
            "rejected": [{"role": "assistant", "content": record["rejected"]}],
        }
        for record in records
    ]
    dataset = Dataset.from_list(conversational_records)
    dataset_digest = _stable_digest(records)

    with tempfile.TemporaryDirectory(prefix="verge-dpo-") as output_dir:
        config = DPOConfig(
            output_dir=output_dir,
            model_init_kwargs={
                "dtype": torch.bfloat16,
                "cache_dir": HF_CACHE_DIR,
                "revision": model_revision,
                "use_safetensors": True,
            },
            per_device_train_batch_size=1,
            gradient_accumulation_steps=min(8, len(records)),
            num_train_epochs=float(epochs),
            learning_rate=1e-5,
            max_length=max_length,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=seed,
            data_seed=seed,
            bf16=True,
            gradient_checkpointing=True,
        )
        trainer = DPOTrainer(
            model=model_id,
            args=config,
            train_dataset=dataset,
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        train_result = trainer.train()
        trainer.save_model(output_dir)
        trainer.processing_class.save_pretrained(output_dir)

        metrics: dict[str, Any] = {}
        for key, value in sorted(train_result.metrics.items()):
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if value is None or isinstance(value, (bool, int, float, str)):
                metrics[str(key)] = value

        files = sorted(
            os.path.relpath(os.path.join(root, filename), output_dir)
            for root, _, filenames in os.walk(output_dir)
            for filename in filenames
            if filename != "manifest.json"
        )
        files.append("manifest.json")
        files.sort()
        manifest: dict[str, Any] = {
            "model_revision": model_revision,
            "kind": "dpo-lora-training",
            "model_id": model_id,
            "device": "L4",
            "seed": seed,
            "pair_count": len(records),
            "dataset_sha256": dataset_digest,
            "adapter": {
                "name": adapter_name,
                "version": adapter_version,
                "volume": ADAPTER_VOLUME_NAME,
                "path": f"{adapter_name}/{adapter_version}",
                "uri": (f"modal-volume://{ADAPTER_VOLUME_NAME}/{adapter_name}/{adapter_version}"),
            },
            "training": {
                "epochs": epochs,
                "max_length": max_length,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": min(8, len(records)),
                "learning_rate": 1e-5,
                "lora_rank": 8,
                "lora_alpha": 16,
            },
            "metrics": metrics,
            "files": files,
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copytree(output_dir, destination)

    hf_cache_volume.commit()
    adapter_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    smoke: bool = False,
    train: bool = False,
    pairs_jsonl: str | None = None,
    model_id: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    max_pairs: int = 256,
    adapter_name: str = "verge-qwen3-0.6b-dpo",
    adapter_version: str = "v1",
    epochs: int = 1,
    max_length: int = 1_024,
) -> None:
    """Run exactly one paid GPU mode.

    ``--smoke`` performs one public-model prompt on an L4 (one short cold-start
    inference call). ``--train --pairs-jsonl PATH`` validates every local JSONL
    record before dispatch, then runs bounded DPO LoRA training on an L4 and
    persists a versioned adapter. Training is materially more expensive than
    smoke inference and is billed for image/model startup plus GPU runtime.
    Reusing the Hugging Face cache volume reduces later startup downloads.
    """
    if smoke == train:
        raise ValueError("choose exactly one mode: --smoke or --train")
    if smoke:
        if pairs_jsonl is not None:
            raise ValueError("--pairs-jsonl is valid only with --train")
        manifest = generate_candidates.remote(
            prompts=["In one sentence, explain why executable checks strengthen preference data."],
            model_id=model_id,
            samples_per_prompt=1,
            max_new_tokens=64,
            seed=seed,
            temperature=0.7,
        )
    else:
        if pairs_jsonl is None:
            raise ValueError("--train requires --pairs-jsonl PATH")
        records = _read_pair_jsonl(pairs_jsonl, max_pairs)
        manifest = train_dpo.remote(
            records=records,
            model_id=model_id,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            seed=seed,
            epochs=epochs,
            max_length=max_length,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
