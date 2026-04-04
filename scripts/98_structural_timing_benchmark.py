"""EXP-98: structural timing benchmark for GPT-2 head pruning.

This benchmark measures real wall-clock speedups from structurally pruning the
dead-head set and the dead-plus-redundant set saved by the EXP-98 pruning harness.
Unlike the reversible masking path, this script uses GPT-2's native head-pruning
support so the timing reflects actual compute reduction.
"""

from __future__ import annotations

import contextlib
import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

PUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]


def default_eval_path() -> Path:
    local = PUB_ROOT / "data" / "tinystories_val_excerpt.txt"
    if local.exists():
        return local
    return REPO_ROOT / "data" / "tinystories_corpus" / "val.txt"


@dataclass(frozen=True)
class BatchSpec:
    input_ids: torch.Tensor


@dataclass(frozen=True)
class VariantSummary:
    name: str
    mode: str
    pruned_heads: int
    parameter_count: int
    parameter_bytes: int


@dataclass(frozen=True)
class LengthVariantResult:
    name: str
    seq_len: int
    passes: int
    warmup_passes: int
    avg_ms: float
    tokens_per_second: float
    loss: float
    delta_loss: float
    speedup_vs_baseline: float


@dataclass(frozen=True)
class LengthBenchmarkResult:
    seq_len: int
    baseline_loss: float
    variants: list[LengthVariantResult]


@dataclass(frozen=True)
class TimingBenchmarkResult:
    model_name: str
    device: str
    eval_path: str
    eval_sequences: int
    batch_size: int
    passes: int
    warmup_passes: int
    seq_lengths: list[int]
    dead_heads: int
    dead_plus_redundant_heads: int
    variants: list[VariantSummary]
    lengths: list[LengthBenchmarkResult]


def model_hidden_size(model) -> int:
    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise AttributeError("Unsupported model config: hidden size not found")


def model_num_heads(model) -> int:
    if hasattr(model.config, "n_head"):
        return int(model.config.n_head)
    if hasattr(model.config, "num_attention_heads"):
        return int(model.config.num_attention_heads)
    raise AttributeError("Unsupported model config: attention head count not found")


def model_head_dim(model) -> int:
    return model_hidden_size(model) // model_num_heads(model)


def model_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Unsupported model architecture: decoder layers not found")


def layer_attention_module(layer):
    if hasattr(layer, "attn"):
        return layer.attn
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise AttributeError("Unsupported decoder layer: attention module not found")


def layer_output_projection(layer):
    attn = layer_attention_module(layer)
    if hasattr(attn, "c_proj"):
        return attn.c_proj
    if hasattr(attn, "o_proj"):
        return attn.o_proj
    raise AttributeError("Unsupported attention module: output projection not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural timing benchmark for EXP-98 head pruning.")
    parser.add_argument("--model-name", default="gpt2", help="HF model name or local path.")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=PUB_ROOT / "data" / "full_gpt2_head144_small_redundancy_v2_dimaware.json",
        help="EXP-98 result JSON containing dead and redundant head decisions.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=default_eval_path(),
        help="Evaluation corpus path.",
    )
    parser.add_argument(
        "--seq-lengths",
        default="128,256,512,1024",
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=16,
        help="Held-out sequences used for loss measurement at each sequence length.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=100,
        help="Measured forward passes per variant and sequence length.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=10,
        help="Unmeasured warmup passes before timing each variant.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size.")
    parser.add_argument(
        "--char-budget-scale",
        type=int,
        default=24,
        help="Approximate chars-per-token multiplier when reading corpus slices.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Model load dtype: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PUB_ROOT / "data" / "gpt2_structural_timing_v1.json",
        help="Where to write the timing JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_lengths(spec: str) -> list[int]:
    lengths: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            lengths.append(int(chunk))
    return sorted(set(length for length in lengths if length > 0))


def _read_text_budget(path: Path, seq_len: int, n_sequences: int, char_budget_scale: int) -> str:
    budget = max(250_000, seq_len * n_sequences * char_budget_scale)
    with path.open("r", encoding="utf-8") as handle:
        return handle.read(budget)


def build_batches(
    path: Path,
    tokenizer,
    seq_len: int,
    n_sequences: int,
    batch_size: int,
    seed: int,
    char_budget_scale: int,
) -> list[BatchSpec]:
    text = _read_text_budget(path, seq_len, n_sequences, char_budget_scale)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    total_blocks = len(token_ids) // seq_len
    if total_blocks <= 0:
        raise ValueError(f"Not enough tokens in {path} for seq_len={seq_len}.")
    usable_blocks = min(total_blocks, n_sequences)
    indices = list(range(total_blocks))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = sorted(indices[:usable_blocks])
    blocks = [
        token_ids[block_idx * seq_len : (block_idx + 1) * seq_len]
        for block_idx in chosen
    ]
    batches: list[BatchSpec] = []
    for start in range(0, len(blocks), batch_size):
        chunk = blocks[start : start + batch_size]
        if len(chunk) < batch_size:
            break
        batch = torch.tensor(chunk, dtype=torch.long)
        batches.append(BatchSpec(input_ids=batch))
    if not batches:
        raise ValueError(f"No complete batches produced for seq_len={seq_len}.")
    return batches


def resolve_torch_dtype(spec: str):
    normalized = spec.strip().lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {spec}")
    return mapping[normalized]


def load_model_and_tokenizer(model_name: str, torch_dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e12)
    dtype = resolve_torch_dtype(torch_dtype)
    load_kwargs = {"dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
    model.eval()
    return model, tokenizer


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def count_parameters(model) -> tuple[int, int]:
    params = sum(int(p.numel()) for p in model.parameters())
    bytes_total = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    return params, bytes_total


def load_prune_maps(result_json: Path) -> tuple[dict[int, set[int]], dict[int, set[int]], int, int]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    dead_map: dict[int, set[int]] = {}
    combined_map: dict[int, set[int]] = {}

    dead_units = [
        (row["layer"], row["head"])
        for row in payload["clr_theory"]["decisions"]
        if row["dead"]
    ]
    redundant_units = [
        (row["layer"], row["head"])
        for row in payload["redundancy_pass"]["decisions"]
        if row["redundant"]
    ]
    for layer_idx, head_idx in dead_units:
        dead_map.setdefault(layer_idx, set()).add(head_idx)
        combined_map.setdefault(layer_idx, set()).add(head_idx)
    for layer_idx, head_idx in redundant_units:
        combined_map.setdefault(layer_idx, set()).add(head_idx)
    return dead_map, combined_map, len(dead_units), len(set(dead_units + redundant_units))


@contextlib.contextmanager
def head_mask_context(model, masked_heads: dict[int, set[int]]):
    handles = []
    n_head = model_num_heads(model)
    head_dim = model_head_dim(model)

    for layer_idx, block in enumerate(model_layers(model)):
        layer_mask = masked_heads.get(layer_idx, set())
        if not layer_mask:
            continue
        output_proj = layer_output_projection(block)

        def _hook(module, inputs, layer_mask=layer_mask):
            x = inputs[0]
            batch, seq, width = x.shape
            head_view = x.reshape(batch, seq, n_head, head_dim)
            masked = head_view.clone()
            for head_idx in layer_mask:
                masked[:, :, head_idx, :] = 0.0
            return (masked.reshape(batch, seq, width),)

        handles.append(output_proj.register_forward_pre_hook(_hook))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def supports_native_head_prune(model) -> bool:
    try:
        layers = model_layers(model)
    except AttributeError:
        return False
    return all(hasattr(layer_attention_module(layer), "prune_heads") for layer in layers)


def structurally_prune_heads(model, prune_map: dict[int, set[int]]) -> None:
    for layer_idx, heads in sorted(prune_map.items()):
        if not heads:
            continue
        layer_attention_module(model_layers(model)[layer_idx]).prune_heads(set(heads))
    model.eval()


def evaluate_loss(
    model,
    batches: Sequence[BatchSpec],
    device: str,
    masked_heads: dict[int, set[int]] | None = None,
) -> float:
    losses: list[float] = []
    masked_heads = masked_heads or {}
    with torch.inference_mode():
        with head_mask_context(model, masked_heads=masked_heads):
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                losses.append(float(outputs.loss.detach().float().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def benchmark_forward(
    model,
    batches: Sequence[BatchSpec],
    device: str,
    passes: int,
    warmup_passes: int,
    masked_heads: dict[int, set[int]] | None = None,
) -> tuple[float, float]:
    total_tokens = 0
    masked_heads = masked_heads or {}
    with torch.inference_mode():
        with head_mask_context(model, masked_heads=masked_heads):
            for step in range(warmup_passes):
                batch = batches[step % len(batches)]
                input_ids = batch.input_ids.to(device)
                model(input_ids=input_ids, use_cache=False, return_dict=True)

            sync_device(device)
            start = time.perf_counter()
            for step in range(passes):
                batch = batches[step % len(batches)]
                input_ids = batch.input_ids.to(device)
                total_tokens += int(input_ids.numel())
                model(input_ids=input_ids, use_cache=False, return_dict=True)
            sync_device(device)
            elapsed = time.perf_counter() - start

    avg_ms = (elapsed / max(1, passes)) * 1000.0
    tokens_per_second = total_tokens / max(elapsed, 1e-12)
    return avg_ms, tokens_per_second


def build_variant_model(model_name: str, device: str, prune_map: dict[int, set[int]], torch_dtype: str):
    model, _tokenizer = load_model_and_tokenizer(model_name, torch_dtype=torch_dtype)
    mode = "structural" if supports_native_head_prune(model) else "masked_proxy"
    if prune_map and mode == "structural":
        structurally_prune_heads(model, prune_map)
    model.to(device)
    return model, mode


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    seq_lengths = parse_lengths(args.seq_lengths)
    dead_map, combined_map, dead_count, combined_count = load_prune_maps(args.result_json)

    tokenizer_model, tokenizer = load_model_and_tokenizer(args.model_name, torch_dtype=args.torch_dtype)
    del tokenizer_model

    variant_specs = [
        ("baseline", {}),
        ("dead_only", dead_map),
        ("dead_plus_redundant", combined_map),
    ]
    variant_models = {}
    variant_summaries: list[VariantSummary] = []
    for name, prune_map in variant_specs:
        model, mode = build_variant_model(args.model_name, device=device, prune_map=prune_map, torch_dtype=args.torch_dtype)
        params, param_bytes = count_parameters(model)
        pruned_heads = sum(len(heads) for heads in prune_map.values())
        variant_models[name] = {
            "model": model,
            "mode": mode,
            "masked_heads": prune_map if mode != "structural" else {},
        }
        variant_summaries.append(
            VariantSummary(
                name=name,
                mode=mode,
                pruned_heads=pruned_heads,
                parameter_count=params,
                parameter_bytes=param_bytes,
            )
        )
        print(
            f"[exp98-time] variant={name} mode={mode} pruned_heads={pruned_heads} "
            f"params={params} bytes={param_bytes}"
        )

    length_results: list[LengthBenchmarkResult] = []
    try:
        for seq_len in seq_lengths:
            batches = build_batches(
                path=args.eval_path,
                tokenizer=tokenizer,
                seq_len=seq_len,
                n_sequences=max(args.eval_sequences, args.passes * args.batch_size),
                batch_size=args.batch_size,
                seed=args.seed + seq_len,
                char_budget_scale=args.char_budget_scale,
            )
            loss_batches = batches[: max(1, args.eval_sequences // max(1, args.batch_size))]
            baseline_runtime = variant_models["baseline"]
            baseline_loss = evaluate_loss(
                baseline_runtime["model"],
                loss_batches,
                device=device,
                masked_heads=baseline_runtime["masked_heads"],
            )
            variant_rows: list[LengthVariantResult] = []
            baseline_avg_ms = 0.0
            for name, _prune_map in variant_specs:
                runtime = variant_models[name]
                model = runtime["model"]
                loss = evaluate_loss(
                    model,
                    loss_batches,
                    device=device,
                    masked_heads=runtime["masked_heads"],
                )
                avg_ms, tokens_per_second = benchmark_forward(
                    model,
                    batches=batches,
                    device=device,
                    passes=args.passes,
                    warmup_passes=args.warmup_passes,
                    masked_heads=runtime["masked_heads"],
                )
                if name == "baseline":
                    baseline_avg_ms = avg_ms
                speedup = (baseline_avg_ms / avg_ms - 1.0) if baseline_avg_ms > 0.0 else 0.0
                row = LengthVariantResult(
                    name=name,
                    seq_len=seq_len,
                    passes=args.passes,
                    warmup_passes=args.warmup_passes,
                    avg_ms=float(avg_ms),
                    tokens_per_second=float(tokens_per_second),
                    loss=float(loss),
                    delta_loss=float(loss - baseline_loss),
                    speedup_vs_baseline=float(speedup),
                )
                variant_rows.append(row)
                print(
                    f"[exp98-time] seq={seq_len} variant={name} avg_ms={avg_ms:.3f} "
                    f"tok_s={tokens_per_second:.1f} loss={loss:.5f} "
                    f"delta_loss={loss - baseline_loss:+.5f} "
                    f"speedup={speedup:+.3%}"
                )
            length_results.append(
                LengthBenchmarkResult(
                    seq_len=seq_len,
                    baseline_loss=float(baseline_loss),
                    variants=variant_rows,
                )
            )
    finally:
        for runtime in variant_models.values():
            runtime["model"].to("cpu")
            del runtime["model"]
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            with torch.no_grad():
                torch.mps.empty_cache()

    result = TimingBenchmarkResult(
        model_name=args.model_name,
        device=device,
        eval_path=str(args.eval_path),
        eval_sequences=args.eval_sequences,
        batch_size=args.batch_size,
        passes=args.passes,
        warmup_passes=args.warmup_passes,
        seq_lengths=seq_lengths,
        dead_heads=dead_count,
        dead_plus_redundant_heads=combined_count,
        variants=variant_summaries,
        lengths=length_results,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(f"[exp98-time] wrote results to {args.out}")


if __name__ == "__main__":
    main()
