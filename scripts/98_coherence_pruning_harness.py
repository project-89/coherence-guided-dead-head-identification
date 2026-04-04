"""EXP-98: coherence-guided structured pruning on frozen GPT-2.

This harness implements coherence-guided structured head pruning:

1. load a frozen causal LM
2. build fixed-length TinyStories calibration/eval blocks
3. score each attention head via reversible ablation
4. compare coherence-guided ranking against magnitude/activation/random baselines
5. export rankings and cumulative prune curves

Phase 1 is intentionally head-only. Channel and sublayer pruning can land on top of
the same evaluation contract once the first signal is measured.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coherence_lattice.analytics.trajectory import compute_meta_pas
from coherence_lattice.core import l2n, pas_ema
from coherence_lattice.metrics.canon_metrics import compute_B_fb

PUB_ROOT = CURRENT_FILE.parents[1]


def default_train_path() -> Path:
    local = PUB_ROOT / "data" / "tinystories_train_excerpt.txt"
    if local.exists():
        return local
    return REPO_ROOT / "data" / "tinystories_corpus" / "train.txt"


def default_eval_path() -> Path:
    local = PUB_ROOT / "data" / "tinystories_val_excerpt.txt"
    if local.exists():
        return local
    return REPO_ROOT / "data" / "tinystories_corpus" / "val.txt"


def model_hidden_size(model) -> int:
    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise AttributeError("Unsupported model config: hidden size not found")


def model_num_layers(model) -> int:
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    raise AttributeError("Unsupported model config: layer count not found")


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


def projection_weight_in_out(proj) -> torch.Tensor:
    weight = proj.weight.detach()
    if proj.__class__.__name__ == "Conv1D" or hasattr(proj, "nf"):
        return weight
    return weight.transpose(0, 1)


@dataclass(frozen=True)
class BatchSpec:
    input_ids: torch.Tensor


@dataclass(frozen=True)
class HeadResult:
    layer: int
    head: int
    delta_loss: float
    delta_coherence: float
    ablated_loss: float
    ablated_coherence: float
    magnitude_score: float
    activation_score: float
    structural_bandwidth: float
    structural_lambda2: float
    structural_bridge: float
    structural_score: float
    death_count: int
    death_max_streak: int
    death_persistence: float


@dataclass(frozen=True)
class PruneCurvePoint:
    method: str
    pruned_heads: int
    loss: float
    coherence: float
    delta_loss: float
    delta_coherence: float


@dataclass(frozen=True)
class EvalSummary:
    loss: float
    coherence: float
    per_layer_coherence: list[float]
    activation_scores: list[list[float]] | None = None


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    device: str
    seq_len: int
    calibration_sequences: int
    eval_sequences: int
    baseline_calibration: EvalSummary
    baseline_eval: EvalSummary
    head_results: list[HeadResult]
    prune_curves: list[PruneCurvePoint]
    clr_theory: ClrTheoryResult | None = None
    redundancy_pass: RedundancyPassResult | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeadScanTrace:
    layer: int
    head: int
    summary: EvalSummary
    batch_losses: list[float]
    batch_coherences: list[float]


@dataclass(frozen=True)
class ClrTheoryDecision:
    layer: int
    head: int
    mean_cosine: float
    min_cosine: float
    below_threshold_batches: int
    death_max_streak: int
    bridge_score: float
    bridge_centered: float
    boundary_protected: bool
    bridge_protected: bool
    protected: bool
    protection_reason: str
    dead: bool
    alive: bool
    safe_ground_truth: bool


@dataclass(frozen=True)
class ClrTheoryResult:
    death_threshold: float
    death_patience: int
    boundary_protection: str
    bridge_veto_basis: str
    ground_truth_loss_threshold: float
    dead_count: int
    protected_count: int
    boundary_protected_count: int
    bridge_protected_count: int
    alive_count: int
    dead_precision: float
    safe_recall: float
    dangerous_keep_rate: float
    true_positive_dead_safe: int
    false_positive_dead_unsafe: int
    false_negative_alive_safe: int
    true_negative_alive_unsafe: int
    protected_safe: int
    protected_unsafe: int
    eval_summary: EvalSummary
    decisions: list[ClrTheoryDecision]


@dataclass(frozen=True)
class RedundancyDecision:
    layer: int
    head: int
    reconstruction_ratio: float
    reconstruction_threshold: float
    reconstruction_significance: float
    max_pairwise_cosine: float
    pairwise_threshold: float
    pairwise_significance: float
    norm_similarity: float
    partner_head: int | None
    redundancy_score: float
    redundant: bool
    safe_ground_truth: bool


@dataclass(frozen=True)
class RedundancyCurvePoint:
    extra_pruned_heads: int
    loss: float
    coherence: float
    delta_loss: float
    delta_coherence: float


@dataclass(frozen=True)
class RedundancyPassResult:
    alive_pool_size: int
    hidden_size: int
    reconstruction_multiplier: float
    pairwise_threshold: float
    pairwise_sigma_multiplier: float
    norm_similarity_threshold: float
    redundant_count: int
    redundant_precision: float
    redundant_recall_within_alive: float
    true_positive_redundant_safe: int
    false_positive_redundant_unsafe: int
    false_negative_alive_safe: int
    true_negative_alive_unsafe: int
    eval_summary_dead_plus_redundant: EvalSummary
    decisions: list[RedundancyDecision]
    curves: list[RedundancyCurvePoint]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coherence-guided reversible head-ablation pruning on a frozen LM.",
    )
    parser.add_argument("--model-name", default="gpt2", help="HF model name or local path.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=default_train_path(),
        help="Calibration corpus path.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=default_eval_path(),
        help="Evaluation corpus path.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Fixed token block length.")
    parser.add_argument(
        "--calibration-sequences",
        type=int,
        default=16,
        help="Number of calibration sequences for head scoring.",
    )
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=16,
        help="Number of held-out sequences for prune curves.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Eval batch size.")
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=1,
        help="Calibration batch size for head scans and death-timer evidence.",
    )
    parser.add_argument(
        "--coherence-last-n-layers",
        type=int,
        default=4,
        help="Number of final transformer layers used in the coherence proxy.",
    )
    parser.add_argument(
        "--char-budget-scale",
        type=int,
        default=16,
        help="Approximate chars-per-token multiplier when reading corpus slices.",
    )
    parser.add_argument(
        "--attention-topk",
        type=int,
        default=8,
        help="Per-token top-k attention edges kept when building head structural graphs.",
    )
    parser.add_argument(
        "--limit-heads",
        type=int,
        default=0,
        help="Optional cap on number of heads scanned, for smoke testing.",
    )
    parser.add_argument(
        "--prune-counts",
        default="1,2,4,8,16",
        help="Comma-separated cumulative prune counts.",
    )
    parser.add_argument(
        "--safe-loss-frac",
        type=float,
        default=0.01,
        help="Relative calibration-loss threshold for safety-gated coherence ranking.",
    )
    parser.add_argument(
        "--safe-loss-abs",
        type=float,
        default=0.0,
        help="Optional absolute calibration-loss threshold overriding the relative floor when larger.",
    )
    parser.add_argument(
        "--death-threshold",
        type=float,
        default=0.679,
        help="Pure CLR death threshold on batch-mean head cosine.",
    )
    parser.add_argument(
        "--death-patience",
        type=int,
        default=3,
        help="Consecutive batches below threshold required before a non-protected head dies.",
    )
    parser.add_argument(
        "--bridge-veto-sigma",
        type=float,
        default=2.0,
        help="Bridge-veto threshold in layerwise standard deviations above the mean.",
    )
    parser.add_argument(
        "--bridge-veto-topk",
        type=int,
        default=3,
        help="A head must also be among the top-k bridge heads in its layer to be protected.",
    )
    parser.add_argument(
        "--boundary-protect-first-layers",
        type=int,
        default=1,
        help="Number of earliest layers treated as input boundary and protected in the theory test.",
    )
    parser.add_argument(
        "--boundary-protect-last-layers",
        type=int,
        default=1,
        help="Number of latest layers treated as output boundary and protected in the theory test.",
    )
    parser.add_argument(
        "--ground-truth-loss-frac",
        type=float,
        default=0.01,
        help="Relative delta-loss threshold for binarizing ablation ground truth as safe-to-remove.",
    )
    parser.add_argument(
        "--ground-truth-loss-abs",
        type=float,
        default=0.0,
        help="Optional absolute delta-loss threshold for ablation ground truth when larger.",
    )
    parser.add_argument(
        "--theory-only",
        action="store_true",
        help="Run the pure CLR death-threshold test and skip ranking-curve sweeps.",
    )
    parser.add_argument(
        "--run-redundancy-pass",
        action="store_true",
        help="Run a second-pass redundancy detector on the alive heads after the CLR theory pass.",
    )
    parser.add_argument(
        "--redundancy-reconstruction-threshold",
        type=float,
        default=0.0,
        help="Optional absolute override for the redundancy reconstruction threshold; <=0 uses a dimension-aware baseline.",
    )
    parser.add_argument(
        "--redundancy-pairwise-threshold",
        type=float,
        default=0.0,
        help="Optional absolute override for the redundancy pairwise-cosine threshold; <=0 uses a dimension-aware baseline.",
    )
    parser.add_argument(
        "--redundancy-reconstruction-multiplier",
        type=float,
        default=4.0,
        help="Multiplier on the random-subspace baseline m/d for alive-head reconstruction overlap.",
    )
    parser.add_argument(
        "--redundancy-pairwise-sigma-multiplier",
        type=float,
        default=4.0,
        help="Multiplier on the random pairwise overlap scale 1/sqrt(d) for alive-head redundancy.",
    )
    parser.add_argument(
        "--redundancy-norm-similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum min/max norm ratio required between a head and its strongest alive partner.",
    )
    parser.add_argument(
        "--redundancy-prune-counts",
        default="1,2,4,8,16,24,32",
        help="Comma-separated extra redundant-head prune counts layered on top of the dead set.",
    )
    parser.add_argument(
        "--reuse-results-json",
        type=Path,
        default=None,
        help="Optional previous result JSON to reuse head ablation scores and skip rescanning.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, mps.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Model load dtype: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PUB_ROOT / "data" / "results.json",
        help="Where to write the experiment JSON.",
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


def parse_prune_counts(spec: str) -> list[int]:
    counts: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        counts.append(int(chunk))
    return sorted(set(c for c in counts if c > 0))


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
        with contextlib.suppress(Exception):
            model.config._attn_implementation = "eager"
    model.eval()
    return model, tokenizer


def _read_text_budget(path: Path, seq_len: int, n_sequences: int, char_budget_scale: int) -> str:
    budget = max(200_000, seq_len * n_sequences * char_budget_scale)
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
        raise ValueError(f"Not enough tokens in {path} for seq_len={seq_len}")
    blocks = np.asarray(token_ids[: total_blocks * seq_len], dtype=np.int64).reshape(total_blocks, seq_len)
    rng = np.random.default_rng(seed)
    sample_count = min(n_sequences, total_blocks)
    indices = np.sort(rng.choice(total_blocks, size=sample_count, replace=False))
    sampled = blocks[indices]

    batches: list[BatchSpec] = []
    for start in range(0, len(sampled), batch_size):
        chunk = sampled[start : start + batch_size]
        batches.append(BatchSpec(input_ids=torch.tensor(chunk, dtype=torch.long)))
    return batches


def _hidden_coherence(hidden: np.ndarray, eps: float = 1e-6) -> float:
    if hidden.shape[0] < 3:
        return 0.0
    X = l2n(np.asarray(hidden, dtype=np.float32))
    pas, _, _ = pas_ema(X, alpha=0.90)
    mean_pas = float(np.mean(pas))
    meta_pas = float(compute_meta_pas(X))
    return max(eps, mean_pas) * max(eps, meta_pas)


def coherence_from_hidden_states(hidden_states: Sequence[torch.Tensor], last_n_layers: int) -> tuple[float, list[float]]:
    usable = hidden_states[1:]  # skip token embeddings
    if last_n_layers > 0:
        usable = usable[-last_n_layers:]
    per_layer_scores: list[float] = []
    for layer_hidden in usable:
        layer_np = layer_hidden.detach().float().cpu().numpy()
        sample_scores = [_hidden_coherence(sample_hidden) for sample_hidden in layer_np]
        per_layer_scores.append(float(np.mean(sample_scores)))
    if not per_layer_scores:
        return 0.0, []
    return float(np.mean(per_layer_scores)), per_layer_scores


@contextlib.contextmanager
def head_mask_context(
    model,
    masked_heads: dict[int, set[int]],
    activation_sums: np.ndarray | None = None,
    activation_counts: np.ndarray | None = None,
):
    handles = []
    n_head = model_num_heads(model)
    head_dim = model_head_dim(model)

    for layer_idx, block in enumerate(model_layers(model)):
        layer_mask = masked_heads.get(layer_idx, set())
        output_proj = layer_output_projection(block)

        def _hook(module, inputs, layer_idx=layer_idx, layer_mask=layer_mask):
            x = inputs[0]
            batch, seq, width = x.shape
            head_view = x.reshape(batch, seq, n_head, head_dim)

            if activation_sums is not None and activation_counts is not None:
                norms = torch.linalg.norm(head_view, dim=-1).mean(dim=(0, 1))
                activation_sums[layer_idx] += norms.detach().float().cpu().numpy()
                activation_counts[layer_idx] += 1.0

            if layer_mask:
                masked = head_view.clone()
                for head_idx in layer_mask:
                    masked[:, :, head_idx, :] = 0.0
                return (masked.reshape(batch, seq, width),)

            return None

        handles.append(output_proj.register_forward_pre_hook(_hook))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_model(
    model,
    batches: Sequence[BatchSpec],
    device: str,
    last_n_layers: int,
    masked_heads: dict[int, set[int]] | None = None,
    collect_activation: bool = False,
) -> EvalSummary:
    masked_heads = masked_heads or {}
    activation_sums = None
    activation_counts = None
    if collect_activation:
        n_layer = model_num_layers(model)
        n_head = model_num_heads(model)
        activation_sums = np.zeros((n_layer, n_head), dtype=np.float64)
        activation_counts = np.zeros((n_layer, n_head), dtype=np.float64)

    total_loss = 0.0
    total_coherence = 0.0
    total_batches = 0
    per_layer_accum: np.ndarray | None = None

    with torch.no_grad():
        with head_mask_context(
            model,
            masked_heads=masked_heads,
            activation_sums=activation_sums,
            activation_counts=activation_counts,
        ):
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                total_loss += float(outputs.loss.detach().float().cpu().item())
                coherence, per_layer = coherence_from_hidden_states(
                    outputs.hidden_states,
                    last_n_layers=last_n_layers,
                )
                total_coherence += coherence
                per_layer_np = np.asarray(per_layer, dtype=np.float64)
                if per_layer_accum is None:
                    per_layer_accum = np.zeros_like(per_layer_np)
                per_layer_accum += per_layer_np
                total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches evaluated.")

    per_layer_mean = (per_layer_accum / total_batches).tolist() if per_layer_accum is not None else []
    activation_scores = None
    if collect_activation and activation_sums is not None and activation_counts is not None:
        safe_counts = np.maximum(activation_counts, 1.0)
        activation_scores = (activation_sums / safe_counts).tolist()

    return EvalSummary(
        loss=total_loss / total_batches,
        coherence=total_coherence / total_batches,
        per_layer_coherence=per_layer_mean,
        activation_scores=activation_scores,
    )


def evaluate_model_trace(
    model,
    batches: Sequence[BatchSpec],
    device: str,
    last_n_layers: int,
    masked_heads: dict[int, set[int]] | None = None,
) -> tuple[EvalSummary, list[float], list[float]]:
    masked_heads = masked_heads or {}
    total_loss = 0.0
    total_coherence = 0.0
    total_batches = 0
    per_layer_accum: np.ndarray | None = None
    batch_losses: list[float] = []
    batch_coherences: list[float] = []

    with torch.no_grad():
        with head_mask_context(model, masked_heads=masked_heads):
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                batch_loss = float(outputs.loss.detach().float().cpu().item())
                coherence, per_layer = coherence_from_hidden_states(
                    outputs.hidden_states,
                    last_n_layers=last_n_layers,
                )
                batch_losses.append(batch_loss)
                batch_coherences.append(coherence)
                total_loss += batch_loss
                total_coherence += coherence
                per_layer_np = np.asarray(per_layer, dtype=np.float64)
                if per_layer_accum is None:
                    per_layer_accum = np.zeros_like(per_layer_np)
                per_layer_accum += per_layer_np
                total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches evaluated.")

    per_layer_mean = (per_layer_accum / total_batches).tolist() if per_layer_accum is not None else []
    summary = EvalSummary(
        loss=total_loss / total_batches,
        coherence=total_coherence / total_batches,
        per_layer_coherence=per_layer_mean,
        activation_scores=None,
    )
    return summary, batch_losses, batch_coherences


def _topk_sparsify(matrix: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0 or topk >= matrix.shape[1]:
        return matrix
    sparse = np.zeros_like(matrix)
    idx = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
    rows = np.arange(matrix.shape[0])[:, None]
    sparse[rows, idx] = matrix[rows, idx]
    return sparse


def attention_graph_metrics(attention: np.ndarray, topk: int) -> tuple[float, float, float]:
    W = np.asarray(attention, dtype=np.float64).copy()
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        return 0.0, 0.0, 0.0

    np.fill_diagonal(W, 0.0)
    W = _topk_sparsify(W, topk=topk)
    W = 0.5 * (W + W.T)
    total_weight = float(np.sum(W))
    if total_weight <= 1e-12:
        return 0.0, 0.0, 0.0

    adj = W > 0.0
    amplitudes = np.ones(W.shape[0], dtype=np.float64)
    bandwidth = compute_B_fb(W, adj=adj, a=amplitudes)

    degree = np.sum(W, axis=1)
    active = degree > 1e-12
    if np.count_nonzero(active) < 2:
        return float(bandwidth), 0.0, 0.0

    W_active = W[np.ix_(active, active)]
    degree_active = np.sum(W_active, axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree_active, 1e-12)))
    L_norm = np.eye(W_active.shape[0]) - inv_sqrt @ W_active @ inv_sqrt

    try:
        eigvals, eigvecs = np.linalg.eigh(L_norm)
    except np.linalg.LinAlgError:
        return float(bandwidth), 0.0, 0.0

    if eigvals.shape[0] < 2:
        return float(bandwidth), 0.0, 0.0

    lambda2 = float(np.clip(eigvals[1], 0.0, 2.0))
    v2 = eigvecs[:, 1]
    pivot = float(np.median(v2))
    left = v2 <= pivot
    right = ~left
    if not np.any(left) or not np.any(right):
        bridge = 0.0
    else:
        cut_mass = float(np.sum(W_active[np.ix_(left, right)]))
        balance = min(int(np.sum(left)), int(np.sum(right))) / max(int(np.sum(left)), int(np.sum(right)))
        bridge = balance * cut_mass / (float(np.sum(W_active)) + 1e-12)

    return float(bandwidth), lambda2, float(np.clip(bridge, 0.0, 1.0))


def collect_attention_structure_scores(
    model,
    batches: Sequence[BatchSpec],
    device: str,
    attention_topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    bandwidth = np.zeros((n_layer, n_head), dtype=np.float64)
    lambda2 = np.zeros((n_layer, n_head), dtype=np.float64)
    bridge = np.zeros((n_layer, n_head), dtype=np.float64)
    counts = np.zeros((n_layer, n_head), dtype=np.float64)

    with torch.no_grad():
        for batch in batches:
            input_ids = batch.input_ids.to(device)
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                if layer_attn is None:
                    continue
                attn_np = layer_attn.detach().float().cpu().numpy()
                for batch_idx in range(attn_np.shape[0]):
                    for head_idx in range(attn_np.shape[1]):
                        band, lam2, bridge_mass = attention_graph_metrics(
                            attn_np[batch_idx, head_idx],
                            topk=attention_topk,
                        )
                        bandwidth[layer_idx, head_idx] += band
                        lambda2[layer_idx, head_idx] += lam2
                        bridge[layer_idx, head_idx] += bridge_mass
                        counts[layer_idx, head_idx] += 1.0

    safe_counts = np.maximum(counts, 1.0)
    return bandwidth / safe_counts, lambda2 / safe_counts, bridge / safe_counts


def collect_head_phase_cosines(
    model,
    batches: Sequence[BatchSpec],
    device: str,
) -> np.ndarray:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    head_dim = model_head_dim(model)
    state: dict[str, object] = {"residual_inputs": {}, "current": None}
    handles = []

    for layer_idx, block in enumerate(model_layers(model)):
        output_proj = layer_output_projection(block)
        head_weights = projection_weight_in_out(output_proj).reshape(n_head, head_dim, -1)

        def _block_pre(module, inputs, layer_idx=layer_idx):
            residual_inputs = state["residual_inputs"]
            assert isinstance(residual_inputs, dict)
            residual_inputs[layer_idx] = inputs[0].detach()

        def _proj_pre(module, inputs, layer_idx=layer_idx, head_weights=head_weights):
            current = state["current"]
            residual_inputs = state["residual_inputs"]
            if current is None or not isinstance(residual_inputs, dict):
                return None
            residual_before = residual_inputs.get(layer_idx)
            if residual_before is None:
                return None

            x = inputs[0]
            batch, seq, width = x.shape
            head_view = x.reshape(batch, seq, n_head, head_dim)
            contrib = torch.einsum("bthd,hdo->btho", head_view, head_weights)
            # Compare each head's incoming write-back signal to the receiver's existing residual state.
            ref = residual_before.unsqueeze(2).expand(batch, seq, n_head, width)
            cos = F.cosine_similarity(contrib, ref, dim=-1, eps=1e-8)
            current[layer_idx] = cos.mean(dim=(0, 1)).detach().float().cpu().numpy()
            return None

        handles.append(block.register_forward_pre_hook(_block_pre))
        handles.append(output_proj.register_forward_pre_hook(_proj_pre))

    batch_cosines: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch in batches:
                state["current"] = np.zeros((n_layer, n_head), dtype=np.float64)
                residual_inputs = state["residual_inputs"]
                assert isinstance(residual_inputs, dict)
                residual_inputs.clear()
                input_ids = batch.input_ids.to(device)
                model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                current = state["current"]
                assert isinstance(current, np.ndarray)
                batch_cosines.append(current.copy())
    finally:
        for handle in handles:
            handle.remove()

    if not batch_cosines:
        return np.zeros((0, n_layer, n_head), dtype=np.float64)
    return np.stack(batch_cosines, axis=0)


def collect_head_contribution_gram(
    model,
    batches: Sequence[BatchSpec],
    device: str,
) -> np.ndarray:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    head_dim = model_head_dim(model)
    gram = np.zeros((n_layer, n_head, n_head), dtype=np.float64)
    handles = []

    for layer_idx, block in enumerate(model_layers(model)):
        output_proj = layer_output_projection(block)
        head_weights = projection_weight_in_out(output_proj).reshape(n_head, head_dim, -1)

        def _proj_pre(module, inputs, layer_idx=layer_idx, head_weights=head_weights):
            x = inputs[0]
            batch, seq, _width = x.shape
            head_view = x.reshape(batch, seq, n_head, head_dim)
            contrib = torch.einsum("bthd,hdo->btho", head_view, head_weights)
            flat = contrib.permute(2, 0, 1, 3).reshape(n_head, -1)
            gram[layer_idx] += flat.detach().float().cpu().numpy() @ flat.detach().float().cpu().numpy().T
            return None

        handles.append(output_proj.register_forward_pre_hook(_proj_pre))

    try:
        with torch.no_grad():
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    return gram


def compute_head_magnitude_scores(model) -> np.ndarray:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    head_dim = model_head_dim(model)
    scores = np.zeros((n_layer, n_head), dtype=np.float64)

    for layer_idx, block in enumerate(model_layers(model)):
        weight = projection_weight_in_out(layer_output_projection(block)).detach().float().cpu().numpy()
        for head_idx in range(n_head):
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            scores[layer_idx, head_idx] = float(np.linalg.norm(weight[start:end, :]))

    return scores


def all_head_units(model, limit_heads: int = 0) -> list[tuple[int, int]]:
    units = [
        (layer_idx, head_idx)
        for layer_idx in range(model_num_layers(model))
        for head_idx in range(model_num_heads(model))
    ]
    if limit_heads > 0:
        return units[:limit_heads]
    return units


def flatten_score_map(score_map: np.ndarray) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for layer_idx in range(score_map.shape[0]):
        for head_idx in range(score_map.shape[1]):
            out[(layer_idx, head_idx)] = float(score_map[layer_idx, head_idx])
    return out


def load_previous_results(path: Path) -> ExperimentResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline_calibration = EvalSummary(**payload["baseline_calibration"])
    baseline_eval = EvalSummary(**payload["baseline_eval"])
    head_results = [head_result_from_payload(row) for row in payload["head_results"]]
    prune_curves = [PruneCurvePoint(**row) for row in payload.get("prune_curves", [])]
    return ExperimentResult(
        model_name=payload["model_name"],
        device=payload["device"],
        seq_len=payload["seq_len"],
        calibration_sequences=payload["calibration_sequences"],
        eval_sequences=payload["eval_sequences"],
        baseline_calibration=baseline_calibration,
        baseline_eval=baseline_eval,
        head_results=head_results,
        prune_curves=prune_curves,
        notes=payload.get("notes", []),
    )


def head_result_from_payload(payload: dict) -> HeadResult:
    row = dict(payload)
    row.setdefault("structural_bandwidth", 0.0)
    row.setdefault("structural_lambda2", 0.0)
    row.setdefault("structural_bridge", 0.0)
    row.setdefault("structural_score", 0.0)
    row.setdefault("death_count", 0)
    row.setdefault("death_max_streak", 0)
    row.setdefault("death_persistence", 0.0)
    return HeadResult(**row)


def score_arrays_from_head_results(model, head_results: Sequence[HeadResult]) -> tuple[np.ndarray, np.ndarray]:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    magnitude_scores = np.zeros((n_layer, n_head), dtype=np.float64)
    activation_scores = np.zeros((n_layer, n_head), dtype=np.float64)
    for row in head_results:
        magnitude_scores[row.layer, row.head] = row.magnitude_score
        activation_scores[row.layer, row.head] = row.activation_score
    return magnitude_scores, activation_scores


def safe_loss_threshold(head_results: Sequence[HeadResult], baseline_loss: float, frac: float, abs_floor: float) -> float:
    positive_deltas = [max(0.0, row.delta_loss) for row in head_results]
    rel = max(0.0, baseline_loss * frac)
    q25 = float(np.quantile(positive_deltas, 0.25)) if positive_deltas else 0.0
    return max(abs_floor, rel, q25)


def pareto_front_indices(losses: np.ndarray, coherences: np.ndarray) -> list[int]:
    remaining = list(range(len(losses)))
    front: list[int] = []
    for i in remaining:
        dominated = False
        for j in remaining:
            if i == j:
                continue
            better_or_equal = losses[j] <= losses[i] and coherences[j] <= coherences[i]
            strictly_better = losses[j] < losses[i] or coherences[j] < coherences[i]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def pareto_rank(head_results: Sequence[HeadResult]) -> list[tuple[int, int]]:
    rows = list(head_results)
    losses = np.asarray([max(0.0, row.delta_loss) for row in rows], dtype=np.float64)
    coherences = np.asarray([row.delta_coherence for row in rows], dtype=np.float64)
    remaining = list(range(len(rows)))
    ordered: list[int] = []

    while remaining:
        sub_losses = losses[remaining]
        sub_coherences = coherences[remaining]
        local_front = pareto_front_indices(sub_losses, sub_coherences)
        front = [remaining[idx] for idx in local_front]
        front_sorted = sorted(front, key=lambda idx: (losses[idx], coherences[idx], rows[idx].layer, rows[idx].head))
        ordered.extend(front_sorted)
        front_set = set(front)
        remaining = [idx for idx in remaining if idx not in front_set]

    return [(rows[idx].layer, rows[idx].head) for idx in ordered]


def safe_partition(
    head_results: Sequence[HeadResult],
    safe_threshold: float,
) -> tuple[list[HeadResult], list[HeadResult]]:
    safe = [row for row in head_results if row.delta_loss <= safe_threshold]
    unsafe = [row for row in head_results if row.delta_loss > safe_threshold]
    return safe, unsafe


def shannon_safe_rank(head_results: Sequence[HeadResult], safe_threshold: float) -> list[tuple[int, int]]:
    safe, unsafe = safe_partition(head_results, safe_threshold=safe_threshold)
    safe_sorted = sorted(
        safe,
        key=lambda row: (
            row.structural_score,
            -row.death_persistence,
            row.delta_coherence,
            row.delta_loss,
            row.layer,
            row.head,
        ),
    )
    unsafe_sorted = sorted(
        unsafe,
        key=lambda row: (
            row.delta_loss,
            row.structural_score,
            -row.death_persistence,
            row.layer,
            row.head,
        ),
    )
    ordered = safe_sorted + unsafe_sorted
    return [(row.layer, row.head) for row in ordered]


def activation_safe_rank(head_results: Sequence[HeadResult], safe_threshold: float) -> list[tuple[int, int]]:
    safe, unsafe = safe_partition(head_results, safe_threshold=safe_threshold)
    safe_sorted = sorted(
        safe,
        key=lambda row: (row.activation_score, row.delta_coherence, row.delta_loss, row.layer, row.head),
    )
    unsafe_sorted = sorted(
        unsafe,
        key=lambda row: (row.delta_loss, row.activation_score, row.delta_coherence, row.layer, row.head),
    )
    ordered = safe_sorted + unsafe_sorted
    return [(row.layer, row.head) for row in ordered]


def hybrid_safe_rank(head_results: Sequence[HeadResult], safe_threshold: float) -> list[tuple[int, int]]:
    safe, unsafe = safe_partition(head_results, safe_threshold=safe_threshold)
    if not safe:
        return activation_safe_rank(head_results, safe_threshold=safe_threshold)

    safe_by_activation = sorted(safe, key=lambda row: (row.activation_score, row.delta_loss, row.layer, row.head))
    safe_by_coherence = sorted(safe, key=lambda row: (row.delta_coherence, row.delta_loss, row.layer, row.head))
    activation_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_activation)}
    coherence_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_coherence)}

    safe_sorted = sorted(
        safe,
        key=lambda row: (
            activation_rank[(row.layer, row.head)] + coherence_rank[(row.layer, row.head)],
            activation_rank[(row.layer, row.head)],
            coherence_rank[(row.layer, row.head)],
            row.delta_loss,
            row.layer,
            row.head,
        ),
    )
    unsafe_sorted = sorted(
        unsafe,
        key=lambda row: (row.delta_loss, row.activation_score, row.delta_coherence, row.layer, row.head),
    )
    ordered = safe_sorted + unsafe_sorted
    return [(row.layer, row.head) for row in ordered]


def dressed_hybrid_rank(head_results: Sequence[HeadResult], safe_threshold: float) -> list[tuple[int, int]]:
    safe, unsafe = safe_partition(head_results, safe_threshold=safe_threshold)
    if not safe:
        return hybrid_safe_rank(head_results, safe_threshold=safe_threshold)

    safe_by_activation = sorted(safe, key=lambda row: (row.activation_score, row.delta_loss, row.layer, row.head))
    safe_by_coherence = sorted(safe, key=lambda row: (row.delta_coherence, row.delta_loss, row.layer, row.head))
    safe_by_structure = sorted(
        safe,
        key=lambda row: (row.structural_score, row.delta_loss, row.delta_coherence, row.layer, row.head),
    )
    safe_by_death = sorted(
        safe,
        key=lambda row: (-row.death_persistence, row.structural_score, row.delta_loss, row.layer, row.head),
    )

    activation_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_activation)}
    coherence_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_coherence)}
    structure_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_structure)}
    death_rank = {(row.layer, row.head): idx for idx, row in enumerate(safe_by_death)}

    safe_sorted = sorted(
        safe,
        key=lambda row: (
            activation_rank[(row.layer, row.head)]
            + coherence_rank[(row.layer, row.head)]
            + structure_rank[(row.layer, row.head)]
            + death_rank[(row.layer, row.head)],
            structure_rank[(row.layer, row.head)],
            death_rank[(row.layer, row.head)],
            activation_rank[(row.layer, row.head)],
            coherence_rank[(row.layer, row.head)],
            row.delta_loss,
            row.layer,
            row.head,
        ),
    )
    unsafe_sorted = sorted(
        unsafe,
        key=lambda row: (
            row.delta_loss,
            row.structural_score,
            -row.death_persistence,
            row.activation_score,
            row.delta_coherence,
            row.layer,
            row.head,
        ),
    )
    ordered = safe_sorted + unsafe_sorted
    return [(row.layer, row.head) for row in ordered]


def rank_units_by_method(
    head_results: Sequence[HeadResult],
    magnitude_map: dict[tuple[int, int], float],
    activation_map: dict[tuple[int, int], float],
    method: str,
    seed: int,
    safe_threshold: float,
) -> list[tuple[int, int]]:
    if method == "coherence":
        ordered = sorted(head_results, key=lambda row: (row.delta_coherence, row.delta_loss, row.layer, row.head))
        return [(row.layer, row.head) for row in ordered]
    if method == "coherence_safe":
        ordered = sorted(
            head_results,
            key=lambda row: (
                row.delta_loss > safe_threshold,
                row.delta_coherence if row.delta_loss <= safe_threshold else row.delta_loss,
                row.delta_loss,
                row.layer,
                row.head,
            ),
        )
        return [(row.layer, row.head) for row in ordered]
    if method == "activation_safe":
        return activation_safe_rank(head_results, safe_threshold=safe_threshold)
    if method == "shannon_safe":
        return shannon_safe_rank(head_results, safe_threshold=safe_threshold)
    if method == "hybrid_safe":
        return hybrid_safe_rank(head_results, safe_threshold=safe_threshold)
    if method == "dressed_hybrid":
        return dressed_hybrid_rank(head_results, safe_threshold=safe_threshold)
    if method == "pareto":
        return pareto_rank(head_results)
    if method == "magnitude":
        return sorted(magnitude_map, key=lambda unit: (magnitude_map[unit], unit[0], unit[1]))
    if method == "activation":
        return sorted(activation_map, key=lambda unit: (activation_map[unit], unit[0], unit[1]))
    if method == "random":
        units = list(magnitude_map)
        rng = random.Random(seed)
        rng.shuffle(units)
        return units
    raise ValueError(f"Unknown ranking method: {method}")


def build_mask(units: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    mask: dict[int, set[int]] = {}
    for layer_idx, head_idx in units:
        mask.setdefault(layer_idx, set()).add(head_idx)
    return mask


def run_head_scan(
    model,
    calibration_batches: Sequence[BatchSpec],
    baseline_calibration: EvalSummary,
    baseline_batch_losses: Sequence[float],
    baseline_batch_coherences: Sequence[float],
    device: str,
    last_n_layers: int,
    magnitude_scores: np.ndarray,
    activation_scores: np.ndarray,
    structural_bandwidth: np.ndarray,
    structural_lambda2: np.ndarray,
    structural_bridge: np.ndarray,
    limit_heads: int,
    safe_loss_frac: float,
    safe_loss_abs: float,
) -> list[HeadResult]:
    units = all_head_units(model, limit_heads=limit_heads)
    traces: list[HeadScanTrace] = []
    composite_raw = (structural_bandwidth + 0.5 * structural_lambda2 + structural_bridge) / 3.0
    structural_mean = float(np.mean(composite_raw))

    for index, (layer_idx, head_idx) in enumerate(units, start=1):
        masked_heads = {layer_idx: {head_idx}}
        ablated, batch_losses, batch_coherences = evaluate_model_trace(
            model,
            calibration_batches,
            device=device,
            last_n_layers=last_n_layers,
            masked_heads=masked_heads,
        )
        traces.append(
            HeadScanTrace(
                layer=layer_idx,
                head=head_idx,
                summary=ablated,
                batch_losses=batch_losses,
                batch_coherences=batch_coherences,
            )
        )
        if index % 12 == 0 or index == len(units):
            print(f"[exp98] scanned {index}/{len(units)} heads")

    batch_loss_deltas = [
        max(0.0, trace.batch_losses[step] - baseline_batch_losses[step])
        for trace in traces
        for step in range(min(len(trace.batch_losses), len(baseline_batch_losses)))
    ]
    batch_coherence_deltas = [
        max(0.0, baseline_batch_coherences[step] - trace.batch_coherences[step])
        for trace in traces
        for step in range(min(len(trace.batch_coherences), len(baseline_batch_coherences)))
    ]
    local_loss_threshold = max(
        safe_loss_abs,
        baseline_calibration.loss * safe_loss_frac,
        float(np.quantile(batch_loss_deltas, 0.25)) if batch_loss_deltas else 0.0,
    )
    local_coherence_threshold = max(
        baseline_calibration.coherence * 0.05,
        float(np.quantile(batch_coherence_deltas, 0.25)) if batch_coherence_deltas else 0.0,
    )
    print(
        f"[exp98] death-timer thresholds loss<={local_loss_threshold:.5f} "
        f"coh<={local_coherence_threshold:.5f}"
    )

    head_results: list[HeadResult] = []
    for trace in traces:
        layer_idx = trace.layer
        head_idx = trace.head
        structural_raw = float(composite_raw[layer_idx, head_idx])
        structural_centered = structural_raw - structural_mean

        death_count = 0
        death_streak = 0
        death_max_streak = 0
        n_steps = min(len(trace.batch_losses), len(baseline_batch_losses), len(trace.batch_coherences), len(baseline_batch_coherences))
        for step in range(n_steps):
            delta_loss = trace.batch_losses[step] - baseline_batch_losses[step]
            delta_coherence = baseline_batch_coherences[step] - trace.batch_coherences[step]
            locally_dead = delta_loss <= local_loss_threshold and delta_coherence <= local_coherence_threshold
            structurally_weak = structural_centered <= 0.0
            if locally_dead and structurally_weak:
                death_count += 1
                death_streak += 1
                death_max_streak = max(death_max_streak, death_streak)
            elif delta_loss > local_loss_threshold or structural_centered > 0.0:
                death_streak = 0

        persistence = (death_count + death_max_streak) / max(1.0, 2.0 * n_steps)
        head_results.append(
            HeadResult(
                layer=layer_idx,
                head=head_idx,
                delta_loss=trace.summary.loss - baseline_calibration.loss,
                delta_coherence=baseline_calibration.coherence - trace.summary.coherence,
                ablated_loss=trace.summary.loss,
                ablated_coherence=trace.summary.coherence,
                magnitude_score=float(magnitude_scores[layer_idx, head_idx]),
                activation_score=float(activation_scores[layer_idx, head_idx]),
                structural_bandwidth=float(structural_bandwidth[layer_idx, head_idx]),
                structural_lambda2=float(structural_lambda2[layer_idx, head_idx]),
                structural_bridge=float(structural_bridge[layer_idx, head_idx]),
                structural_score=structural_centered,
                death_count=death_count,
                death_max_streak=death_max_streak,
                death_persistence=float(persistence),
            )
        )

    return head_results


def run_prune_curves(
    model,
    eval_batches: Sequence[BatchSpec],
    baseline_eval: EvalSummary,
    head_results: Sequence[HeadResult],
    magnitude_scores: np.ndarray,
    activation_scores: np.ndarray,
    prune_counts: Sequence[int],
    device: str,
    last_n_layers: int,
    seed: int,
    safe_threshold: float,
) -> list[PruneCurvePoint]:
    magnitude_map = flatten_score_map(magnitude_scores)
    activation_map = flatten_score_map(activation_scores)
    methods = [
        "coherence",
        "coherence_safe",
        "activation_safe",
        "shannon_safe",
        "hybrid_safe",
        "dressed_hybrid",
        "pareto",
        "magnitude",
        "activation",
        "random",
    ]
    curves: list[PruneCurvePoint] = []

    for method_idx, method in enumerate(methods):
        ordered_units = rank_units_by_method(
            head_results=head_results,
            magnitude_map=magnitude_map,
            activation_map=activation_map,
            method=method,
            seed=seed + method_idx,
            safe_threshold=safe_threshold,
        )
        for prune_count in prune_counts:
            masked = build_mask(ordered_units[:prune_count])
            summary = evaluate_model(
                model,
                eval_batches,
                device=device,
                last_n_layers=last_n_layers,
                masked_heads=masked,
                collect_activation=False,
            )
            curves.append(
                PruneCurvePoint(
                    method=method,
                    pruned_heads=prune_count,
                    loss=summary.loss,
                    coherence=summary.coherence,
                    delta_loss=summary.loss - baseline_eval.loss,
                    delta_coherence=baseline_eval.coherence - summary.coherence,
                )
            )
            print(
                f"[exp98] {method:>10} prune={prune_count:>3} "
                f"delta_loss={summary.loss - baseline_eval.loss:+.5f} "
                f"delta_coh={baseline_eval.coherence - summary.coherence:+.5f}"
            )

    return curves


def summarize_top_candidates(head_results: Sequence[HeadResult], top_k: int = 10) -> None:
    ordered = sorted(head_results, key=lambda row: (row.delta_coherence, row.delta_loss, row.layer, row.head))
    print("[exp98] top coherence-prune candidates")
    for row in ordered[:top_k]:
        print(
            f"  L{row.layer:02d}H{row.head:02d} "
            f"delta_coh={row.delta_coherence:+.5f} "
            f"delta_loss={row.delta_loss:+.5f} "
            f"mag={row.magnitude_score:.5f} "
            f"act={row.activation_score:.5f} "
            f"struct={row.structural_score:+.5f} "
            f"dead={row.death_persistence:.3f}"
        )


def summarize_safe_candidates(head_results: Sequence[HeadResult], safe_threshold: float, top_k: int = 10) -> None:
    ordered = sorted(
        head_results,
        key=lambda row: (
            row.delta_loss > safe_threshold,
            row.delta_coherence if row.delta_loss <= safe_threshold else row.delta_loss,
            row.delta_loss,
            row.layer,
            row.head,
        ),
    )
    print(f"[exp98] top safety-gated candidates (delta_loss <= {safe_threshold:.5f})")
    for row in ordered[:top_k]:
        print(
            f"  L{row.layer:02d}H{row.head:02d} "
            f"delta_coh={row.delta_coherence:+.5f} "
            f"delta_loss={row.delta_loss:+.5f} "
            f"struct={row.structural_score:+.5f} "
            f"dead={row.death_persistence:.3f}"
        )


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(a) != len(b):
        return 0.0
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if np.allclose(a_np.std(), 0.0) or np.allclose(b_np.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a_np, b_np)[0, 1])


def compute_ground_truth_loss_threshold(
    baseline_loss: float,
    frac: float,
    abs_floor: float,
) -> float:
    return max(abs_floor, baseline_loss * frac)


def run_clr_theory_test(
    model,
    eval_batches: Sequence[BatchSpec],
    baseline_eval: EvalSummary,
    head_results: Sequence[HeadResult],
    phase_cosines: np.ndarray,
    structural_bridge: np.ndarray,
    device: str,
    last_n_layers: int,
    death_threshold: float,
    death_patience: int,
    bridge_veto_sigma: float,
    bridge_veto_topk: int,
    boundary_protect_first_layers: int,
    boundary_protect_last_layers: int,
    ground_truth_loss_threshold: float,
) -> ClrTheoryResult:
    n_layer = model_num_layers(model)
    n_head = model_num_heads(model)
    layer_bridge_means = structural_bridge.mean(axis=1, keepdims=True)
    layer_bridge_stds = structural_bridge.std(axis=1, keepdims=True)
    layer_bridge_order = np.argsort(structural_bridge, axis=1)[:, ::-1]
    safe_map = {
        (row.layer, row.head): row.delta_loss <= ground_truth_loss_threshold
        for row in head_results
    }
    decisions: list[ClrTheoryDecision] = []
    dead_units: list[tuple[int, int]] = []
    tp = fp = fn = tn = 0
    protected_safe = 0
    protected_unsafe = 0
    boundary_protected_count = 0
    bridge_protected_count = 0

    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            trace = phase_cosines[:, layer_idx, head_idx] if phase_cosines.size else np.zeros((0,), dtype=np.float64)
            bridge_score = float(structural_bridge[layer_idx, head_idx])
            bridge_centered = bridge_score - float(layer_bridge_means[layer_idx, 0])
            bridge_threshold = float(layer_bridge_means[layer_idx, 0] + bridge_veto_sigma * layer_bridge_stds[layer_idx, 0])
            top_heads = set(int(h) for h in layer_bridge_order[layer_idx, : max(1, bridge_veto_topk)])
            boundary_protected = layer_idx < max(0, boundary_protect_first_layers) or layer_idx >= max(0, n_layer - boundary_protect_last_layers)
            bridge_protected = bridge_score > bridge_threshold and head_idx in top_heads
            protected = boundary_protected or bridge_protected
            if boundary_protected:
                boundary_protected_count += 1
            elif bridge_protected:
                bridge_protected_count += 1
            below = trace < death_threshold
            below_count = int(np.sum(below))
            streak = 0
            max_streak = 0
            for is_below in below.tolist():
                if is_below:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            dead = (not protected) and (max_streak >= death_patience)
            alive = not dead and not protected
            safe_ground_truth = safe_map.get((layer_idx, head_idx), False)

            if dead:
                dead_units.append((layer_idx, head_idx))
                if safe_ground_truth:
                    tp += 1
                else:
                    fp += 1
            else:
                if safe_ground_truth:
                    fn += 1
                    if protected:
                        protected_safe += 1
                else:
                    tn += 1
                    if protected:
                        protected_unsafe += 1

            decisions.append(
                ClrTheoryDecision(
                    layer=layer_idx,
                    head=head_idx,
                    mean_cosine=float(np.mean(trace)) if trace.size else 0.0,
                    min_cosine=float(np.min(trace)) if trace.size else 0.0,
                    below_threshold_batches=below_count,
                    death_max_streak=max_streak,
                    bridge_score=bridge_score,
                    bridge_centered=bridge_centered,
                    boundary_protected=boundary_protected,
                    bridge_protected=bridge_protected,
                    protected=protected,
                    protection_reason=(
                        "boundary"
                        if boundary_protected
                        else "bridge"
                        if bridge_protected
                        else "none"
                    ),
                    dead=dead,
                    alive=alive,
                    safe_ground_truth=safe_ground_truth,
                )
            )

    dead_precision = tp / max(1, tp + fp)
    safe_recall = tp / max(1, tp + fn)
    dangerous_keep_rate = tn / max(1, tn + fp)
    eval_summary = evaluate_model(
        model,
        eval_batches,
        device=device,
        last_n_layers=last_n_layers,
        masked_heads=build_mask(dead_units),
        collect_activation=False,
    )
    return ClrTheoryResult(
        death_threshold=death_threshold,
        death_patience=death_patience,
        boundary_protection=(
            f"first={max(0, boundary_protect_first_layers)}, "
            f"last={max(0, boundary_protect_last_layers)}"
        ),
        bridge_veto_basis=f"per-layer bridge > mean + {bridge_veto_sigma:.2f} sigma and top-{bridge_veto_topk}",
        ground_truth_loss_threshold=ground_truth_loss_threshold,
        dead_count=len(dead_units),
        protected_count=sum(1 for row in decisions if row.protected),
        boundary_protected_count=boundary_protected_count,
        bridge_protected_count=bridge_protected_count,
        alive_count=sum(1 for row in decisions if row.alive),
        dead_precision=float(dead_precision),
        safe_recall=float(safe_recall),
        dangerous_keep_rate=float(dangerous_keep_rate),
        true_positive_dead_safe=tp,
        false_positive_dead_unsafe=fp,
        false_negative_alive_safe=fn,
        true_negative_alive_unsafe=tn,
        protected_safe=protected_safe,
        protected_unsafe=protected_unsafe,
        eval_summary=eval_summary,
        decisions=decisions,
    )


def summarize_clr_theory(theory: ClrTheoryResult, baseline_eval: EvalSummary, top_k: int = 10) -> None:
    print(
        f"[exp98] CLR theory: dead={theory.dead_count} protected={theory.protected_count} "
        f"alive={theory.alive_count} threshold={theory.death_threshold:.3f} "
        f"patience={theory.death_patience}"
    )
    print(
        f"[exp98] protection: boundary={theory.boundary_protected_count} "
        f"bridge={theory.bridge_protected_count} "
        f"rule=({theory.boundary_protection}; {theory.bridge_veto_basis})"
    )
    print(
        f"[exp98] theory confusion: TP={theory.true_positive_dead_safe} "
        f"FP={theory.false_positive_dead_unsafe} FN={theory.false_negative_alive_safe} "
        f"TN={theory.true_negative_alive_unsafe} precision={theory.dead_precision:.3f} "
        f"recall={theory.safe_recall:.3f}"
    )
    print(
        f"[exp98] theory prune eval: delta_loss={theory.eval_summary.loss - baseline_eval.loss:+.5f} "
        f"delta_coh={baseline_eval.coherence - theory.eval_summary.coherence:+.5f}"
    )
    dead_rows = [row for row in theory.decisions if row.dead]
    dead_rows = sorted(dead_rows, key=lambda row: (row.mean_cosine, -row.death_max_streak, row.layer, row.head))
    print("[exp98] top CLR-dead heads")
    for row in dead_rows[:top_k]:
        print(
            f"  L{row.layer:02d}H{row.head:02d} cos={row.mean_cosine:+.4f} "
            f"min={row.min_cosine:+.4f} streak={row.death_max_streak} "
            f"bridge={row.bridge_centered:+.4f} safe={int(row.safe_ground_truth)}"
        )


def run_redundancy_pass(
    model,
    calibration_gram: np.ndarray,
    clr_theory: ClrTheoryResult,
    head_results: Sequence[HeadResult],
    eval_batches: Sequence[BatchSpec],
    baseline_eval: EvalSummary,
    device: str,
    last_n_layers: int,
    reconstruction_threshold: float,
    pairwise_threshold: float,
    reconstruction_multiplier: float,
    pairwise_sigma_multiplier: float,
    norm_similarity_threshold: float,
    prune_counts: Sequence[int],
) -> RedundancyPassResult:
    head_loss_map = {(row.layer, row.head): row.delta_loss for row in head_results}
    ground_truth_threshold = clr_theory.ground_truth_loss_threshold
    dead_units = [(row.layer, row.head) for row in clr_theory.decisions if row.dead]
    alive_decisions = [row for row in clr_theory.decisions if row.alive]
    alive_pool_size = len(alive_decisions)
    hidden_size = model_hidden_size(model)
    random_pairwise_scale = 1.0 / max(1.0, np.sqrt(float(hidden_size)))
    pairwise_cutoff = (
        pairwise_threshold
        if pairwise_threshold > 0.0
        else pairwise_sigma_multiplier * random_pairwise_scale
    )

    decisions: list[RedundancyDecision] = []
    tp = fp = fn = tn = 0

    for layer_idx in range(calibration_gram.shape[0]):
        layer_alive = [row for row in alive_decisions if row.layer == layer_idx]
        if len(layer_alive) < 2:
            for row in layer_alive:
                safe_ground_truth = bool(head_loss_map[(row.layer, row.head)] <= ground_truth_threshold)
                decisions.append(
                    RedundancyDecision(
                        layer=row.layer,
                        head=row.head,
                        reconstruction_ratio=0.0,
                        reconstruction_threshold=0.0,
                        reconstruction_significance=0.0,
                        max_pairwise_cosine=0.0,
                        pairwise_threshold=pairwise_cutoff,
                        pairwise_significance=0.0,
                        norm_similarity=0.0,
                        partner_head=None,
                        redundancy_score=0.0,
                        redundant=False,
                        safe_ground_truth=safe_ground_truth,
                    )
                )
                if safe_ground_truth:
                    fn += 1
                else:
                    tn += 1
            continue

        heads = [row.head for row in layer_alive]
        G = calibration_gram[layer_idx][np.ix_(heads, heads)].astype(np.float64)
        G = 0.5 * (G + G.T)
        diag = np.clip(np.diag(G), 1e-12, None)
        for idx, head_idx in enumerate(heads):
            other = [j for j in range(len(heads)) if j != idx]
            safe_ground_truth = bool(head_loss_map[(layer_idx, head_idx)] <= ground_truth_threshold)
            layer_reconstruction_baseline = len(other) / max(1, hidden_size)
            reconstruction_cutoff = (
                reconstruction_threshold
                if reconstruction_threshold > 0.0
                else min(1.0, reconstruction_multiplier * layer_reconstruction_baseline)
            )
            norm_similarity = 0.0
            partner_head: int | None = None
            if not other:
                reconstruction_ratio = 0.0
                max_pairwise_cosine = 0.0
            else:
                G_oo = G[np.ix_(other, other)] + 1e-8 * np.eye(len(other))
                g_oi = G[np.ix_(other, [idx])].reshape(len(other))
                try:
                    coeff = np.linalg.pinv(G_oo) @ g_oi
                    projected_sq = float(g_oi.T @ coeff)
                except np.linalg.LinAlgError:
                    projected_sq = 0.0
                reconstruction_ratio = float(np.clip(projected_sq / diag[idx], 0.0, 1.0))
                norm_i = float(np.sqrt(diag[idx]))
                pairwise_info: list[tuple[float, float, int]] = []
                for j in other:
                    if diag[j] <= 1e-12:
                        continue
                    pair = abs(float(G[idx, j] / np.sqrt(diag[idx] * diag[j])))
                    norm_j = float(np.sqrt(diag[j]))
                    sim = min(norm_i, norm_j) / max(norm_i, norm_j)
                    pairwise_info.append((pair, sim, heads[j]))
                if pairwise_info:
                    best_pair, best_norm_similarity, best_partner = max(
                        pairwise_info,
                        key=lambda item: (item[0] * item[1], item[0], item[1], -item[2]),
                    )
                    max_pairwise_cosine = float(best_pair)
                    norm_similarity = float(best_norm_similarity)
                    partner_head = int(best_partner)
                else:
                    max_pairwise_cosine = 0.0

            reconstruction_significance = (
                reconstruction_ratio / reconstruction_cutoff
                if reconstruction_cutoff > 1e-12
                else 0.0
            )
            pairwise_significance = (
                max_pairwise_cosine / pairwise_cutoff
                if pairwise_cutoff > 1e-12
                else 0.0
            )

            redundant = bool(
                reconstruction_ratio >= reconstruction_cutoff
                and max_pairwise_cosine >= pairwise_cutoff
                and norm_similarity >= norm_similarity_threshold
            )
            redundancy_score = reconstruction_significance * pairwise_significance * norm_similarity
            decisions.append(
                RedundancyDecision(
                    layer=layer_idx,
                    head=head_idx,
                    reconstruction_ratio=reconstruction_ratio,
                    reconstruction_threshold=float(reconstruction_cutoff),
                    reconstruction_significance=float(reconstruction_significance),
                    max_pairwise_cosine=max_pairwise_cosine,
                    pairwise_threshold=float(pairwise_cutoff),
                    pairwise_significance=float(pairwise_significance),
                    norm_similarity=float(norm_similarity),
                    partner_head=partner_head,
                    redundancy_score=redundancy_score,
                    redundant=redundant,
                    safe_ground_truth=safe_ground_truth,
                )
            )
            if redundant and safe_ground_truth:
                tp += 1
            elif redundant and not safe_ground_truth:
                fp += 1
            elif (not redundant) and safe_ground_truth:
                fn += 1
            else:
                tn += 1

    redundant_units = [(row.layer, row.head) for row in decisions if row.redundant]
    eval_summary_dead_plus_redundant = evaluate_model(
        model,
        eval_batches,
        device=device,
        last_n_layers=last_n_layers,
        masked_heads=build_mask(dead_units + redundant_units),
        collect_activation=False,
    )

    ranked_alive = sorted(
        decisions,
        key=lambda row: (
            row.redundancy_score,
            row.reconstruction_ratio,
            row.max_pairwise_cosine,
            row.layer,
            row.head,
        ),
        reverse=True,
    )
    curves: list[RedundancyCurvePoint] = []
    max_available = len(ranked_alive)
    for prune_count in [count for count in prune_counts if count <= max_available]:
        extra_units = [(row.layer, row.head) for row in ranked_alive[:prune_count]]
        summary = evaluate_model(
            model,
            eval_batches,
            device=device,
            last_n_layers=last_n_layers,
            masked_heads=build_mask(dead_units + extra_units),
            collect_activation=False,
        )
        curves.append(
            RedundancyCurvePoint(
                extra_pruned_heads=prune_count,
                loss=summary.loss,
                coherence=summary.coherence,
                delta_loss=summary.loss - baseline_eval.loss,
                delta_coherence=baseline_eval.coherence - summary.coherence,
            )
        )

    return RedundancyPassResult(
        alive_pool_size=alive_pool_size,
        hidden_size=hidden_size,
        reconstruction_multiplier=reconstruction_multiplier,
        pairwise_threshold=float(pairwise_cutoff),
        pairwise_sigma_multiplier=pairwise_sigma_multiplier,
        norm_similarity_threshold=norm_similarity_threshold,
        redundant_count=len(redundant_units),
        redundant_precision=float(tp / max(1, tp + fp)),
        redundant_recall_within_alive=float(tp / max(1, tp + fn)),
        true_positive_redundant_safe=tp,
        false_positive_redundant_unsafe=fp,
        false_negative_alive_safe=fn,
        true_negative_alive_unsafe=tn,
        eval_summary_dead_plus_redundant=eval_summary_dead_plus_redundant,
        decisions=decisions,
        curves=curves,
    )


def summarize_redundancy_pass(
    redundancy: RedundancyPassResult,
    clr_theory: ClrTheoryResult,
    baseline_eval: EvalSummary,
    top_k: int = 10,
) -> None:
    print(
        f"[exp98] redundancy pass: alive_pool={redundancy.alive_pool_size} "
        f"redundant={redundancy.redundant_count} "
        f"pair>={redundancy.pairwise_threshold:.3f} "
        f"pair_sigma={redundancy.pairwise_sigma_multiplier:.2f} "
        f"recon_mult={redundancy.reconstruction_multiplier:.2f} "
        f"norm>={redundancy.norm_similarity_threshold:.2f}"
    )
    print(
        f"[exp98] redundancy confusion (within alive): TP={redundancy.true_positive_redundant_safe} "
        f"FP={redundancy.false_positive_redundant_unsafe} "
        f"FN={redundancy.false_negative_alive_safe} "
        f"TN={redundancy.true_negative_alive_unsafe} "
        f"precision={redundancy.redundant_precision:.3f} "
        f"recall={redundancy.redundant_recall_within_alive:.3f}"
    )
    print(
        f"[exp98] dead+redundant eval: delta_loss={redundancy.eval_summary_dead_plus_redundant.loss - baseline_eval.loss:+.5f} "
        f"delta_coh={baseline_eval.coherence - redundancy.eval_summary_dead_plus_redundant.coherence:+.5f} "
        f"(dead-only delta_loss={clr_theory.eval_summary.loss - baseline_eval.loss:+.5f})"
    )
    ordered = sorted(
        redundancy.decisions,
        key=lambda row: (
            row.redundancy_score,
            row.reconstruction_ratio,
            row.max_pairwise_cosine,
            row.layer,
            row.head,
        ),
        reverse=True,
    )
    print("[exp98] top redundancy candidates")
    for row in ordered[:top_k]:
        print(
            f"  L{row.layer:02d}H{row.head:02d} "
            f"score={row.redundancy_score:.4f} "
            f"recon={row.reconstruction_ratio:.4f} "
            f"recon_thr={row.reconstruction_threshold:.4f} "
            f"pair={row.max_pairwise_cosine:.4f} "
            f"pair_thr={row.pairwise_threshold:.4f} "
            f"norm={row.norm_similarity:.3f} "
            f"peer={row.partner_head if row.partner_head is not None else -1:02d} "
            f"safe={int(row.safe_ground_truth)} "
            f"flag={int(row.redundant)}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.model_name, torch_dtype=args.torch_dtype)
    model.to(device)
    calibration_batches = build_batches(
        path=args.train_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        n_sequences=args.calibration_sequences,
        batch_size=args.scan_batch_size,
        seed=args.seed,
        char_budget_scale=args.char_budget_scale,
    )
    eval_batches = build_batches(
        path=args.eval_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        n_sequences=args.eval_sequences,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        char_budget_scale=args.char_budget_scale,
    )

    print(f"[exp98] model={args.model_name} device={device}")
    print(f"[exp98] eval_sequences={args.eval_sequences} seq_len={args.seq_len}")

    if args.reuse_results_json is not None:
        previous = load_previous_results(args.reuse_results_json)
        baseline_calibration = previous.baseline_calibration
        baseline_eval = previous.baseline_eval
        head_results = previous.head_results
        magnitude_scores, activation_scores = score_arrays_from_head_results(model, head_results)
        structural_bandwidth, structural_lambda2, structural_bridge = collect_attention_structure_scores(
            model,
            calibration_batches,
            device=device,
            attention_topk=args.attention_topk,
        )
        print(f"[exp98] reusing head scan from {args.reuse_results_json}")
    else:
        print(
            f"[exp98] calibration_sequences={args.calibration_sequences} "
            f"eval_sequences={args.eval_sequences} seq_len={args.seq_len}"
        )

        baseline_calibration = evaluate_model(
            model,
            calibration_batches,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            masked_heads=None,
            collect_activation=True,
        )
        _, baseline_batch_losses, baseline_batch_coherences = evaluate_model_trace(
            model,
            calibration_batches,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            masked_heads=None,
        )
        baseline_eval = evaluate_model(
            model,
            eval_batches,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            masked_heads=None,
            collect_activation=False,
        )
        structural_bandwidth, structural_lambda2, structural_bridge = collect_attention_structure_scores(
            model,
            calibration_batches,
            device=device,
            attention_topk=args.attention_topk,
        )

        if baseline_calibration.activation_scores is None:
            raise RuntimeError("Activation scores were not collected.")
        activation_scores = np.asarray(baseline_calibration.activation_scores, dtype=np.float64)
        magnitude_scores = compute_head_magnitude_scores(model)

        print(
            f"[exp98] baseline calibration loss={baseline_calibration.loss:.5f} "
            f"coherence={baseline_calibration.coherence:.5f}"
        )
        print(
            f"[exp98] baseline eval loss={baseline_eval.loss:.5f} "
            f"coherence={baseline_eval.coherence:.5f}"
        )

        head_results = run_head_scan(
            model=model,
            calibration_batches=calibration_batches,
            baseline_calibration=baseline_calibration,
            baseline_batch_losses=baseline_batch_losses,
            baseline_batch_coherences=baseline_batch_coherences,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            magnitude_scores=magnitude_scores,
            activation_scores=activation_scores,
            structural_bandwidth=structural_bandwidth,
            structural_lambda2=structural_lambda2,
            structural_bridge=structural_bridge,
            limit_heads=args.limit_heads,
            safe_loss_frac=args.safe_loss_frac,
            safe_loss_abs=args.safe_loss_abs,
        )

    ground_truth_loss_threshold = compute_ground_truth_loss_threshold(
        baseline_loss=baseline_calibration.loss,
        frac=args.ground_truth_loss_frac,
        abs_floor=args.ground_truth_loss_abs,
    )
    phase_cosines = collect_head_phase_cosines(
        model,
        calibration_batches,
        device=device,
    )
    clr_theory = run_clr_theory_test(
        model=model,
        eval_batches=eval_batches,
        baseline_eval=baseline_eval,
        head_results=head_results,
        phase_cosines=phase_cosines,
        structural_bridge=structural_bridge,
        device=device,
        last_n_layers=args.coherence_last_n_layers,
        death_threshold=args.death_threshold,
        death_patience=args.death_patience,
        bridge_veto_sigma=args.bridge_veto_sigma,
        bridge_veto_topk=args.bridge_veto_topk,
        boundary_protect_first_layers=args.boundary_protect_first_layers,
        boundary_protect_last_layers=args.boundary_protect_last_layers,
        ground_truth_loss_threshold=ground_truth_loss_threshold,
    )
    redundancy_pass = None
    if args.run_redundancy_pass:
        contribution_gram = collect_head_contribution_gram(
            model,
            calibration_batches,
            device=device,
        )
        redundancy_pass = run_redundancy_pass(
            model=model,
            calibration_gram=contribution_gram,
            clr_theory=clr_theory,
            head_results=head_results,
            eval_batches=eval_batches,
            baseline_eval=baseline_eval,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            reconstruction_threshold=args.redundancy_reconstruction_threshold,
            pairwise_threshold=args.redundancy_pairwise_threshold,
            reconstruction_multiplier=args.redundancy_reconstruction_multiplier,
            pairwise_sigma_multiplier=args.redundancy_pairwise_sigma_multiplier,
            norm_similarity_threshold=args.redundancy_norm_similarity_threshold,
            prune_counts=parse_prune_counts(args.redundancy_prune_counts),
        )
    print(
        f"[exp98] baseline calibration loss={baseline_calibration.loss:.5f} "
        f"coherence={baseline_calibration.coherence:.5f}"
    )
    print(
        f"[exp98] baseline eval loss={baseline_eval.loss:.5f} "
        f"coherence={baseline_eval.coherence:.5f}"
    )
    summarize_clr_theory(clr_theory, baseline_eval=baseline_eval)
    if redundancy_pass is not None:
        summarize_redundancy_pass(redundancy_pass, clr_theory=clr_theory, baseline_eval=baseline_eval)

    prune_curves: list[PruneCurvePoint] = []
    if not args.theory_only:
        summarize_top_candidates(head_results)
        safe_threshold = safe_loss_threshold(
            head_results=head_results,
            baseline_loss=baseline_calibration.loss,
            frac=args.safe_loss_frac,
            abs_floor=args.safe_loss_abs,
        )
        summarize_safe_candidates(head_results, safe_threshold=safe_threshold)

        print(
            "[exp98] corr(delta_coherence, delta_loss)="
            f"{correlation([row.delta_coherence for row in head_results], [row.delta_loss for row in head_results]):+.5f}"
        )
        print(f"[exp98] safety threshold delta_loss <= {safe_threshold:.5f}")

        max_available = len(head_results)
        prune_counts = [count for count in parse_prune_counts(args.prune_counts) if count <= max_available]
        prune_curves = run_prune_curves(
            model=model,
            eval_batches=eval_batches,
            baseline_eval=baseline_eval,
            head_results=head_results,
            magnitude_scores=magnitude_scores,
            activation_scores=activation_scores,
            prune_counts=prune_counts,
            device=device,
            last_n_layers=args.coherence_last_n_layers,
            seed=args.seed,
            safe_threshold=safe_threshold,
        )

    result = ExperimentResult(
        model_name=args.model_name,
        device=device,
        seq_len=args.seq_len,
        calibration_sequences=args.calibration_sequences,
        eval_sequences=args.eval_sequences,
        baseline_calibration=baseline_calibration,
        baseline_eval=baseline_eval,
        head_results=head_results,
        prune_curves=prune_curves,
        clr_theory=clr_theory,
        redundancy_pass=redundancy_pass,
        notes=[
            "Head-only phase-1 experiment on frozen GPT-2.",
            "Coherence proxy = mean over final layers of PAS(hidden trajectory) * meta_PAS(hidden trajectory).",
            "Calibration split scores individual heads; eval split checks cumulative pruning curves.",
            "CLR theory test uses per-head incoming write-back alignment against the receiver's pre-head residual state, plus boundary protection, a raw death threshold, a consecutive-batch death timer, and a rare-event bridge veto.",
            "Optional redundancy pass operates only on alive heads and uses dimension-aware overlap baselines plus same-scale partner checks on head-output Gram matrices.",
            "Attention-side structural proxy uses sparsified symmetrized head graphs with B_fb, normalized lambda_2, and Fiedler-cut bridge mass.",
            "Death timer increments only when a head is locally low-impact across calibration batches and its centered structural score is non-positive.",
            "Additional rankers include Shannon-safe and dressed-hybrid variants on top of the earlier safety-gated baselines.",
        ],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(f"[exp98] wrote results to {args.out}")


if __name__ == "__main__":
    main()
