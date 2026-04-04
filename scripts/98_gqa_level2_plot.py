"""EXP-98: focused GQA/Level-2 compaction figure.

This figure complements the five-model summary with:
1. full-dead and near-full KV-group fractions for Qwen2 and SmolLM2
2. measured SmolLM2 Level 2 cache-building prefill speedups
3. measured KV-cache bytes before and after Level 2 compaction
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PUB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PUB_ROOT / "data"
OUT_DIR = PUB_ROOT / "figures"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _group_stats(result_path: Path, query_heads: int, kv_heads: int) -> dict[str, float]:
    data = _load_json(result_path)
    group_size = query_heads // kv_heads
    total_groups = 0
    full = 0
    near = 0
    dead_by_layer: dict[int, set[int]] = {}
    for row in data["clr_theory"]["decisions"]:
        if row["dead"]:
            dead_by_layer.setdefault(int(row["layer"]), set()).add(int(row["head"]))
    layer_count = max(int(row["layer"]) for row in data["clr_theory"]["decisions"]) + 1
    for layer_idx in range(layer_count):
        dead_heads = dead_by_layer.get(layer_idx, set())
        for kv_idx in range(kv_heads):
            total_groups += 1
            group_heads = range(kv_idx * group_size, (kv_idx + 1) * group_size)
            dead_count = sum(head in dead_heads for head in group_heads)
            if dead_count == group_size:
                full += 1
            if dead_count >= group_size - 1:
                near += 1
    return {
        "total_groups": total_groups,
        "full_fraction": full / total_groups,
        "near_fraction": near / total_groups,
        "group_size": group_size,
    }


def main() -> None:
    qwen_stats = _group_stats(
        DATA_ROOT / "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json",
        query_heads=14,
        kv_heads=2,
    )
    smol_stats = _group_stats(
        DATA_ROOT / "smollm2_360m_head480_small_theory_redundancy_v1.json",
        query_heads=15,
        kv_heads=5,
    )
    smol_l2 = _load_json(DATA_ROOT / "smollm2_level2_kv_compaction_v1.json")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ax = axes[0]
    labels = ["Qwen2 0.5B\n(g=7)", "SmolLM2 360M\n(g=3)"]
    full_vals = [qwen_stats["full_fraction"] * 100.0, smol_stats["full_fraction"] * 100.0]
    near_only = [
        (qwen_stats["near_fraction"] - qwen_stats["full_fraction"]) * 100.0,
        (smol_stats["near_fraction"] - smol_stats["full_fraction"]) * 100.0,
    ]
    x = np.arange(len(labels))
    ax.bar(x, full_vals, color="#1f77b4", label="Fully dead KV groups")
    ax.bar(x, near_only, bottom=full_vals, color="#9ecae1", label="Near-full only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of KV groups (%)")
    ax.set_title("GQA Group Death Structure")
    ax.legend(loc="upper left", frameon=True)

    ax = axes[1]
    seqs = [row["seq_len"] for row in smol_l2["lengths"]]
    speedups = []
    for row in smol_l2["lengths"]:
        variant = next(v for v in row["variants"] if v["name"] == "level2_full_dead_kv_groups")
        speedups.append(variant["speedup_vs_baseline"] * 100.0)
    ax.plot(seqs, speedups, marker="o", linewidth=2.5, color="#2ca02c")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Prefill speedup (%)")
    ax.set_title("SmolLM2 Level 2 Speedup")

    ax = axes[2]
    baseline_cache = []
    compact_cache = []
    for row in smol_l2["lengths"]:
        baseline = next(v for v in row["variants"] if v["name"] == "baseline")
        compact = next(v for v in row["variants"] if v["name"] == "level2_full_dead_kv_groups")
        baseline_cache.append(baseline["cache_bytes"] / (1024 * 1024))
        compact_cache.append(compact["cache_bytes"] / (1024 * 1024))
    width = 0.35
    ax.bar(np.arange(len(seqs)) - width / 2, baseline_cache, width=width, color="#ff7f0e", label="Baseline")
    ax.bar(np.arange(len(seqs)) + width / 2, compact_cache, width=width, color="#fdd0a2", label="Level 2")
    ax.set_xticks(np.arange(len(seqs)))
    ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("KV cache (MiB)")
    ax.set_title("SmolLM2 Returned KV Cache")
    ax.legend(loc="upper left", frameon=True)

    figure_path = OUT_DIR / "exp98_gqa_level2_summary.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    print(f"[exp98-gqa-plot] wrote {figure_path}")


if __name__ == "__main__":
    main()
