"""EXP-98: true Level 2 KV-group compaction benchmark for Llama-style decoders.

This script takes the dead-head decisions from the EXP-98 CLR pass and performs
real grouped-query structural compaction on fully dead KV groups in a Llama-style
decoder. Unlike masked proxy timing, this rewrites q/k/v/o projections so the
model carries fewer query heads, fewer KV heads, and a smaller KV cache.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

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
class LayerCompaction:
    layer: int
    total_query_heads: int
    total_kv_heads: int
    group_size: int
    fully_dead_groups: list[int]
    kept_groups: list[int]
    removed_query_heads: list[int]
    kept_query_heads: list[int]


@dataclass(frozen=True)
class VariantSummary:
    name: str
    parameter_count: int
    parameter_bytes: int
    total_query_heads: int
    total_kv_heads: int
    kv_cache_bytes_per_token: int


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
    cache_bytes: int
    cache_reduction_vs_baseline: float
    speedup_vs_baseline: float


@dataclass(frozen=True)
class LengthBenchmarkResult:
    seq_len: int
    baseline_loss: float
    variants: list[LengthVariantResult]


@dataclass(frozen=True)
class KvGroupCompactionResult:
    model_name: str
    device: str
    result_json: str
    eval_path: str
    eval_sequences: int
    batch_size: int
    passes: int
    warmup_passes: int
    seq_lengths: list[int]
    total_query_heads: int
    dead_query_heads: int
    total_kv_groups: int
    fully_dead_kv_groups: int
    fully_dead_kv_fraction: float
    removed_query_heads: int
    removed_query_fraction: float
    variants: list[VariantSummary]
    layers: list[LayerCompaction]
    lengths: list[LengthBenchmarkResult]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true Level 2 KV-group compaction benchmark for a Llama-style decoder.",
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolLM2-360M",
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=PUB_ROOT / "data" / "smollm2_360m_head480_small_theory_redundancy_v1.json",
        help="EXP-98 result JSON containing CLR dead-head decisions.",
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
        default=8,
        help="Held-out sequences used for loss measurement at each sequence length.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=50,
        help="Measured forward passes per variant and sequence length.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=5,
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
        default=PUB_ROOT / "data" / "smollm2_level2_kv_compaction_v1.json",
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


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


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


def count_parameters(model) -> tuple[int, int]:
    params = sum(int(p.numel()) for p in model.parameters())
    bytes_total = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    return params, bytes_total


def layer_attention_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise AttributeError("Expected a decoder layer with self_attn.")


def model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Expected a decoder model with model.layers.")


def projection_dims(attn) -> tuple[int, int, int]:
    head_dim = int(attn.head_dim)
    num_q = int(attn.q_proj.weight.shape[0] // head_dim)
    num_kv = int(attn.k_proj.weight.shape[0] // head_dim)
    return num_q, num_kv, head_dim


def build_level2_plan(result_json: Path, model) -> tuple[list[LayerCompaction], int, int, int]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    dead_map: dict[int, set[int]] = {}
    for row in payload["clr_theory"]["decisions"]:
        if row["dead"]:
            dead_map.setdefault(int(row["layer"]), set()).add(int(row["head"]))

    layers = model_layers(model)
    layer_plans: list[LayerCompaction] = []
    total_query_heads = 0
    dead_query_heads = 0
    fully_dead_groups = 0

    for layer_idx, layer in enumerate(layers):
        attn = layer_attention_module(layer)
        num_q, num_kv, _head_dim = projection_dims(attn)
        group_size = int(num_q // num_kv)
        total_query_heads += num_q
        dead_heads = dead_map.get(layer_idx, set())
        dead_query_heads += len(dead_heads)
        dead_groups: list[int] = []
        kept_groups: list[int] = []
        removed_query_heads: list[int] = []
        kept_query_heads: list[int] = []

        for kv_idx in range(num_kv):
            group_heads = list(range(kv_idx * group_size, (kv_idx + 1) * group_size))
            if all(head in dead_heads for head in group_heads):
                dead_groups.append(kv_idx)
                removed_query_heads.extend(group_heads)
            else:
                kept_groups.append(kv_idx)
                kept_query_heads.extend(group_heads)

        fully_dead_groups += len(dead_groups)
        layer_plans.append(
            LayerCompaction(
                layer=layer_idx,
                total_query_heads=num_q,
                total_kv_heads=num_kv,
                group_size=group_size,
                fully_dead_groups=dead_groups,
                kept_groups=kept_groups,
                removed_query_heads=removed_query_heads,
                kept_query_heads=kept_query_heads,
            )
        )

    return layer_plans, total_query_heads, dead_query_heads, fully_dead_groups


def _slice_linear_out_features(linear: nn.Linear, keep_rows: list[int]) -> nn.Linear:
    new_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=len(keep_rows),
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[keep_rows, :])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias[keep_rows])
    return new_linear


def _slice_linear_in_features(linear: nn.Linear, keep_cols: list[int]) -> nn.Linear:
    new_linear = nn.Linear(
        in_features=len(keep_cols),
        out_features=linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[:, keep_cols])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def _head_rows(head_indices: list[int], head_dim: int) -> list[int]:
    rows: list[int] = []
    for head_idx in head_indices:
        start = head_idx * head_dim
        rows.extend(range(start, start + head_dim))
    return rows


def apply_level2_compaction(model, layer_plans: Sequence[LayerCompaction]) -> None:
    for plan in layer_plans:
        if not plan.fully_dead_groups:
            continue

        layer = model_layers(model)[plan.layer]
        attn = layer_attention_module(layer)
        num_q, num_kv, head_dim = projection_dims(attn)
        if num_q != plan.total_query_heads or num_kv != plan.total_kv_heads:
            raise ValueError(f"Layer {plan.layer}: attention shape mismatch before compaction.")
        if not plan.kept_groups:
            raise ValueError(f"Layer {plan.layer}: refusing to remove every KV group.")

        keep_q_rows = _head_rows(plan.kept_query_heads, head_dim)
        keep_kv_rows = _head_rows(plan.kept_groups, head_dim)

        attn.q_proj = _slice_linear_out_features(attn.q_proj, keep_q_rows)
        attn.k_proj = _slice_linear_out_features(attn.k_proj, keep_kv_rows)
        attn.v_proj = _slice_linear_out_features(attn.v_proj, keep_kv_rows)
        attn.o_proj = _slice_linear_in_features(attn.o_proj, keep_q_rows)

        new_num_q = len(plan.kept_query_heads)
        new_num_kv = len(plan.kept_groups)
        attn.num_heads = new_num_q
        attn.num_key_value_heads = new_num_kv
        attn.num_key_value_groups = new_num_q // new_num_kv
        attn.pruned_kv_groups = tuple(plan.fully_dead_groups)
        attn.pruned_query_heads = tuple(plan.removed_query_heads)

    model.eval()


def count_attention_heads(model) -> tuple[int, int]:
    total_q = 0
    total_kv = 0
    for layer in model_layers(model):
        attn = layer_attention_module(layer)
        num_q, num_kv, _head_dim = projection_dims(attn)
        total_q += num_q
        total_kv += num_kv
    return total_q, total_kv


def kv_cache_bytes_per_token(model, batch_size: int = 1) -> int:
    total = 0
    for layer in model_layers(model):
        attn = layer_attention_module(layer)
        _num_q, num_kv, head_dim = projection_dims(attn)
        dtype_bytes = int(attn.k_proj.weight.element_size())
        total += 2 * batch_size * num_kv * head_dim * dtype_bytes
    return int(total)


def evaluate_loss(model, batches: Sequence[BatchSpec], device: str) -> float:
    losses: list[float] = []
    with torch.inference_mode():
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


def benchmark_prefill(model, batches: Sequence[BatchSpec], device: str, passes: int, warmup_passes: int) -> tuple[float, float]:
    total_tokens = 0
    with torch.inference_mode():
        for step in range(warmup_passes):
            batch = batches[step % len(batches)]
            input_ids = batch.input_ids.to(device)
            model(input_ids=input_ids, use_cache=True, return_dict=True)

        sync_device(device)
        start = time.perf_counter()
        for step in range(passes):
            batch = batches[step % len(batches)]
            input_ids = batch.input_ids.to(device)
            total_tokens += int(input_ids.numel())
            model(input_ids=input_ids, use_cache=True, return_dict=True)
        sync_device(device)
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / max(1, passes)) * 1000.0
    tokens_per_second = total_tokens / max(elapsed, 1e-12)
    return avg_ms, tokens_per_second


def cache_bytes_from_forward(model, batch: BatchSpec, device: str) -> int:
    with torch.inference_mode():
        outputs = model(
            input_ids=batch.input_ids.to(device),
            use_cache=True,
            return_dict=True,
        )
    cache = outputs.past_key_values
    total = 0
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            total += int(layer.keys.numel() * layer.keys.element_size())
            total += int(layer.values.numel() * layer.values.element_size())
    else:
        for key, value in cache:
            total += int(key.numel() * key.element_size())
            total += int(value.numel() * value.element_size())
    return total


def build_variant(model_name: str, compact: bool, layer_plans: Sequence[LayerCompaction], device: str, torch_dtype: str):
    model, _tokenizer = load_model_and_tokenizer(model_name, torch_dtype=torch_dtype)
    if compact:
        apply_level2_compaction(model, layer_plans)
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    seq_lengths = parse_lengths(args.seq_lengths)

    plan_model, tokenizer = load_model_and_tokenizer(args.model_name, torch_dtype=args.torch_dtype)
    layer_plans, total_query_heads, dead_query_heads, fully_dead_groups = build_level2_plan(args.result_json, plan_model)
    total_kv_groups = sum(plan.total_kv_heads for plan in layer_plans)
    removed_query_heads = sum(len(plan.removed_query_heads) for plan in layer_plans)
    del plan_model

    baseline_model = build_variant(args.model_name, compact=False, layer_plans=layer_plans, device=device, torch_dtype=args.torch_dtype)
    compact_model = build_variant(args.model_name, compact=True, layer_plans=layer_plans, device=device, torch_dtype=args.torch_dtype)

    baseline_params, baseline_param_bytes = count_parameters(baseline_model)
    compact_params, compact_param_bytes = count_parameters(compact_model)
    baseline_q_heads, baseline_kv_heads = count_attention_heads(baseline_model)
    compact_q_heads, compact_kv_heads = count_attention_heads(compact_model)

    variants = [
        (
            "baseline",
            baseline_model,
            VariantSummary(
                name="baseline",
                parameter_count=baseline_params,
                parameter_bytes=baseline_param_bytes,
                total_query_heads=baseline_q_heads,
                total_kv_heads=baseline_kv_heads,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token(baseline_model, batch_size=args.batch_size),
            ),
        ),
        (
            "level2_full_dead_kv_groups",
            compact_model,
            VariantSummary(
                name="level2_full_dead_kv_groups",
                parameter_count=compact_params,
                parameter_bytes=compact_param_bytes,
                total_query_heads=compact_q_heads,
                total_kv_heads=compact_kv_heads,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token(compact_model, batch_size=args.batch_size),
            ),
        ),
    ]

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
            baseline_loss = evaluate_loss(baseline_model, loss_batches, device=device)
            baseline_cache_bytes = cache_bytes_from_forward(baseline_model, batches[0], device=device)

            rows: list[LengthVariantResult] = []
            baseline_avg_ms = 0.0
            for name, model, _summary in variants:
                loss = evaluate_loss(model, loss_batches, device=device)
                avg_ms, tokens_per_second = benchmark_prefill(
                    model,
                    batches=batches,
                    device=device,
                    passes=args.passes,
                    warmup_passes=args.warmup_passes,
                )
                cache_bytes = cache_bytes_from_forward(model, batches[0], device=device)
                if name == "baseline":
                    baseline_avg_ms = avg_ms
                speedup = (baseline_avg_ms / avg_ms - 1.0) if baseline_avg_ms > 0.0 else 0.0
                rows.append(
                    LengthVariantResult(
                        name=name,
                        seq_len=seq_len,
                        passes=args.passes,
                        warmup_passes=args.warmup_passes,
                        avg_ms=float(avg_ms),
                        tokens_per_second=float(tokens_per_second),
                        loss=float(loss),
                        delta_loss=float(loss - baseline_loss),
                        cache_bytes=int(cache_bytes),
                        cache_reduction_vs_baseline=float(1.0 - (cache_bytes / baseline_cache_bytes)),
                        speedup_vs_baseline=float(speedup),
                    )
                )
                print(
                    f"[exp98-smollm2-l2] seq={seq_len} variant={name} avg_ms={avg_ms:.3f} "
                    f"tok_s={tokens_per_second:.1f} loss={loss:.5f} "
                    f"delta_loss={loss - baseline_loss:+.5f} "
                    f"cache_reduction={1.0 - (cache_bytes / baseline_cache_bytes):+.3%} "
                    f"speedup={speedup:+.3%}"
                )
            length_results.append(
                LengthBenchmarkResult(
                    seq_len=seq_len,
                    baseline_loss=float(baseline_loss),
                    variants=rows,
                )
            )
    finally:
        baseline_model.to("cpu")
        compact_model.to("cpu")
        del baseline_model
        del compact_model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            with torch.no_grad():
                torch.mps.empty_cache()

    result = KvGroupCompactionResult(
        model_name=args.model_name,
        device=device,
        result_json=str(args.result_json),
        eval_path=str(args.eval_path),
        eval_sequences=args.eval_sequences,
        batch_size=args.batch_size,
        passes=args.passes,
        warmup_passes=args.warmup_passes,
        seq_lengths=seq_lengths,
        total_query_heads=total_query_heads,
        dead_query_heads=dead_query_heads,
        total_kv_groups=total_kv_groups,
        fully_dead_kv_groups=fully_dead_groups,
        fully_dead_kv_fraction=float(fully_dead_groups / max(1, total_kv_groups)),
        removed_query_heads=removed_query_heads,
        removed_query_fraction=float(removed_query_heads / max(1, total_query_heads)),
        variants=[summary for _name, _model, summary in variants],
        layers=layer_plans,
        lengths=length_results,
        notes=[
            "Level 2 removes only full dead KV groups from the CLR dead-head map.",
            "Timing uses cache-building prefill forwards (use_cache=True), not masked proxy timing.",
            "KV cache bytes are measured from the actual returned cache tensors.",
        ],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(f"[exp98-smollm2-l2] wrote results to {args.out}")


if __name__ == "__main__":
    main()
