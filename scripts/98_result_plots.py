"""EXP-98: generate summary figures from saved pruning and timing artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PUB_ROOT / "data"
FIGURE_ROOT = PUB_ROOT / "figures"


@dataclass(frozen=True)
class ModelSummary:
    label: str
    family: str
    hidden_size: int
    death_threshold: float
    dead_precision: float
    redundancy_precision: float
    dead_fraction: float
    redundancy_fraction: float
    combined_fraction: float
    combined_precision: float
    combined_delta_loss: float


PRUNE_FILES = {
    "gpt2": ("GPT-2", "GPT-2", 768, "full_gpt2_head144_small_redundancy_v2_dimaware.json"),
    "gpt2-medium": ("GPT-2 Medium", "GPT-2", 1024, "full_gpt2_medium_head384_small_theory_redundancy_v1.json"),
    "qwen25-0.5b": ("Qwen2.5 0.5B", "Qwen2", 896, "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json"),
    "smollm2-360m": ("SmolLM2 360M", "Llama", 960, "smollm2_360m_head480_small_theory_redundancy_v1.json"),
    "open-llama-7b": ("OpenLLaMA 7B", "Llama", 4096, "open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json"),
}

TIMING_FILES = {
    "gpt2": ("structural", "gpt2_structural_timing_v1.json", "dead_plus_redundant"),
    "gpt2-medium": ("structural", "gpt2_medium_structural_timing_v1.json", "dead_plus_redundant"),
    "qwen25-0.5b": ("masked proxy", "qwen25_05b_timing_proxy_v1.json", "dead_plus_redundant"),
    "smollm2-360m": ("masked proxy", "smollm2_360m_timing_proxy_v1.json", "dead_plus_redundant"),
    "open-llama-7b": ("structural level2", "open_llama_7b_level2_compaction_v1.json", "level2_full_dead_kv_groups"),
}


def load_model_summaries() -> list[ModelSummary]:
    summaries: list[ModelSummary] = []
    for key, (label, family, hidden_size, filename) in PRUNE_FILES.items():
        payload = json.loads((DATA_ROOT / filename).read_text())
        dead = payload["clr_theory"]
        red = payload["redundancy_pass"]
        dead_tp = dead["true_positive_dead_safe"]
        red_tp = red["true_positive_redundant_safe"]
        dead_fp = dead["false_positive_dead_unsafe"]
        red_fp = red["false_positive_redundant_unsafe"]
        total_heads = dead["dead_count"] + dead["protected_count"] + dead["alive_count"]
        summaries.append(
            ModelSummary(
                label=label,
                family=family,
                hidden_size=hidden_size,
                death_threshold=float(dead["death_threshold"]),
                dead_precision=float(dead["dead_precision"]),
                redundancy_precision=float(red["redundant_precision"]),
                dead_fraction=dead["dead_count"] / total_heads,
                redundancy_fraction=red["redundant_count"] / total_heads,
                combined_fraction=(dead["dead_count"] + red["redundant_count"]) / total_heads,
                combined_precision=(dead_tp + red_tp) / max(1, dead_tp + red_tp + dead_fp + red_fp),
                combined_delta_loss=float(
                    red["eval_summary_dead_plus_redundant"]["loss"] - payload["baseline_eval"]["loss"]
                ),
            )
        )
    return summaries


def load_timing_rows() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for key, (mode, filename, variant_name) in TIMING_FILES.items():
        payload = json.loads((DATA_ROOT / filename).read_text())
        points = []
        for row in payload["lengths"]:
            dd = next(item for item in row["variants"] if item["name"] == variant_name)
            points.append(
                {
                    "seq_len": int(row["seq_len"]),
                    "speedup_pct": 100.0 * float(dd["speedup_vs_baseline"]),
                    "delta_loss": float(dd["delta_loss"]),
                }
            )
        rows[key] = {"mode": mode, "points": points}
    return rows


def make_figure(summaries: list[ModelSummary], timing_rows: dict[str, dict[str, object]], out_path: Path) -> None:
    labels = [row.label for row in summaries]
    hidden = np.asarray([row.hidden_size for row in summaries], dtype=np.float64)
    thresholds = np.asarray([row.death_threshold for row in summaries], dtype=np.float64)
    dead_prec = 100.0 * np.asarray([row.dead_precision for row in summaries], dtype=np.float64)
    red_prec = 100.0 * np.asarray([row.redundancy_precision for row in summaries], dtype=np.float64)
    dead_frac = 100.0 * np.asarray([row.dead_fraction for row in summaries], dtype=np.float64)
    red_frac = 100.0 * np.asarray([row.redundancy_fraction for row in summaries], dtype=np.float64)
    combined_prec = 100.0 * np.asarray([row.combined_precision for row in summaries], dtype=np.float64)
    chi_values = thresholds * np.sqrt(hidden)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_threshold, ax_precision, ax_fraction, ax_timing = axes.ravel()

    # Threshold scaling curve.
    d_grid = np.linspace(min(hidden) * 0.9, max(hidden) * 1.05, 400)
    ax_threshold.plot(
        d_grid,
        0.96 / np.sqrt(d_grid),
        color="#1f4e79",
        linewidth=2.2,
        label=r"reference $\approx 0.96 / \sqrt{d}$",
    )
    ax_threshold.scatter(hidden, thresholds, color="#c0392b", s=60, zorder=3)
    for row in summaries:
        ax_threshold.annotate(row.label, (row.hidden_size, row.death_threshold), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax_threshold.text(
        0.98,
        0.06,
        rf"exact bundled $\chi_c \in [{min(chi_values):.5f}, {max(chi_values):.5f}]$",
        transform=ax_threshold.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax_threshold.set_title("Exact Frozen Thresholds vs. Rounded Reference Curve")
    ax_threshold.set_xlabel("Hidden Dimension d")
    ax_threshold.set_ylabel(r"$\tau_{death}$")
    ax_threshold.grid(alpha=0.25)
    ax_threshold.legend(frameon=False)

    # Precision bars.
    x = np.arange(len(labels))
    width = 0.24
    ax_precision.bar(x - width, dead_prec, width=width, label="Dead Precision", color="#2e86ab")
    ax_precision.bar(x, red_prec, width=width, label="Redundancy Precision", color="#27ae60")
    ax_precision.bar(x + width, combined_prec, width=width, label="Combined Precision", color="#e67e22")
    ax_precision.set_xticks(x)
    ax_precision.set_xticklabels(labels, rotation=20, ha="right")
    ax_precision.set_ylim(75, 101)
    ax_precision.set_ylabel("Precision (%)")
    ax_precision.set_title("Precision Across Models")
    ax_precision.grid(axis="y", alpha=0.25)
    ax_precision.legend(frameon=False, fontsize=9)

    # Removal fractions.
    ax_fraction.bar(x, dead_frac, label="Dead Removed", color="#34495e")
    ax_fraction.bar(x, red_frac, bottom=dead_frac, label="Redundant Removed", color="#95a5a6")
    ax_fraction.set_xticks(x)
    ax_fraction.set_xticklabels(labels, rotation=20, ha="right")
    ax_fraction.set_ylabel("Heads Removed (%)")
    ax_fraction.set_title("Two-Pass Sparsity by Model")
    ax_fraction.grid(axis="y", alpha=0.25)
    ax_fraction.legend(frameon=False, fontsize=9)

    # Timing curves.
    palette = {
        "gpt2": "#1f77b4",
        "gpt2-medium": "#ff7f0e",
        "qwen25-0.5b": "#2ca02c",
        "smollm2-360m": "#d62728",
        "open-llama-7b": "#9467bd",
    }
    for key, series in timing_rows.items():
        xs = [point["seq_len"] for point in series["points"]]
        ys = [point["speedup_pct"] for point in series["points"]]
        linestyle = "-" if series["mode"] == "structural" else "--"
        ax_timing.plot(xs, ys, marker="o", linewidth=2.0, linestyle=linestyle, color=palette[key], label=f"{PRUNE_FILES[key][0]} ({series['mode']})")
    ax_timing.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax_timing.set_xscale("log", base=2)
    ax_timing.set_xticks([128, 256, 512, 1024], labels=["128", "256", "512", "1024"])
    ax_timing.set_xlabel("Sequence Length")
    ax_timing.set_ylabel("Measured Speedup vs Baseline (%)")
    ax_timing.set_title("Timing at Increasing Context Length")
    ax_timing.grid(alpha=0.25)
    ax_timing.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("EXP-98: Dimension-Scaled Pruning Across Five Models", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summaries = load_model_summaries()
    timing_rows = load_timing_rows()
    out_path = FIGURE_ROOT / "exp98_five_model_summary.png"
    make_figure(summaries, timing_rows, out_path=out_path)
    print(f"[exp98-plot] wrote {out_path}")


if __name__ == "__main__":
    main()
