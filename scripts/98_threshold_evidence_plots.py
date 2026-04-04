"""EXP-98: head-level threshold evidence plots for the publication package.

This script generates two local publication figures from the bundled JSON artifacts:

1. a head-level threshold evidence figure with normalized-coupling scatter plots
   and categorical layer/head decision maps
2. a normalized-coupling collapse figure across the full five-model transfer set
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

PUB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PUB_ROOT / "data"
FIGURE_ROOT = PUB_ROOT / "figures"


STATUS_ORDER = ["dead", "boundary", "bridge", "redundant", "alive"]
STATUS_COLORS = {
    "dead": "#d73027",
    "boundary": "#4575b4",
    "bridge": "#6a3d9a",
    "redundant": "#fdae61",
    "alive": "#1a9850",
}
STATUS_LABELS = {
    "dead": "Dead",
    "boundary": "Boundary protected",
    "bridge": "Bridge protected",
    "redundant": "Redundant",
    "alive": "Alive",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    hidden_size: int
    filename: str


MODEL_SPECS = [
    ModelSpec("gpt2", "GPT-2", 768, "full_gpt2_head144_small_redundancy_v2_dimaware.json"),
    ModelSpec("gpt2-medium", "GPT-2 Medium", 1024, "full_gpt2_medium_head384_small_theory_redundancy_v1.json"),
    ModelSpec("qwen25-0.5b", "Qwen2.5 0.5B", 896, "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json"),
    ModelSpec("smollm2-360m", "SmolLM2 360M", 960, "smollm2_360m_head480_small_theory_redundancy_v1.json"),
    ModelSpec("open-llama-7b", "OpenLLaMA 7B", 4096, "open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json"),
]


def _load_payload(spec: ModelSpec) -> dict:
    with (DATA_ROOT / spec.filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _redundancy_map(payload: dict) -> dict[tuple[int, int], bool]:
    return {
        (int(row["layer"]), int(row["head"])): bool(row["redundant"])
        for row in payload["redundancy_pass"]["decisions"]
    }


def _status_for_head(row: dict, redundant: bool) -> str:
    if bool(row["dead"]):
        return "dead"
    if bool(row["boundary_protected"]):
        return "boundary"
    if bool(row["bridge_protected"]):
        return "bridge"
    if redundant:
        return "redundant"
    return "alive"


def _join_head_rows(spec: ModelSpec, payload: dict) -> list[dict]:
    redundancy = _redundancy_map(payload)
    head_results = {
        (int(row["layer"]), int(row["head"])): row
        for row in payload["head_results"]
    }
    rows: list[dict] = []
    for row in payload["clr_theory"]["decisions"]:
        key = (int(row["layer"]), int(row["head"]))
        result = head_results[key]
        redundant = redundancy.get(key, False)
        rows.append(
            {
                "layer": key[0],
                "head": key[1],
                "status": _status_for_head(row, redundant),
                "mean_cosine": float(row["mean_cosine"]),
                "normalized_coupling": math.sqrt(spec.hidden_size) * float(row["mean_cosine"]),
                "delta_loss": float(result["delta_loss"]),
                "safe_ground_truth": bool(row["safe_ground_truth"]),
                "protected": bool(row["protected"]),
                "dead": bool(row["dead"]),
            }
        )
    return rows


def _artifact_threshold(payload: dict) -> float:
    return float(payload["clr_theory"]["death_threshold"])


def _artifact_chi(spec: ModelSpec, payload: dict) -> float:
    return math.sqrt(spec.hidden_size) * _artifact_threshold(payload)


def _decision_matrix(rows: list[dict]) -> np.ndarray:
    n_layers = max(row["layer"] for row in rows) + 1
    n_heads = max(row["head"] for row in rows) + 1
    matrix = np.full((n_layers, n_heads), fill_value=-1, dtype=np.int64)
    code = {name: idx for idx, name in enumerate(STATUS_ORDER)}
    for row in rows:
        matrix[row["layer"], row["head"]] = code[row["status"]]
    return matrix


def _below_threshold_breakdown(spec: ModelSpec, payload: dict) -> dict[str, float]:
    threshold = _artifact_threshold(payload)
    counts = {"dead": 0, "protected": 0, "alive": 0}
    total = 0
    for row in payload["clr_theory"]["decisions"]:
        below = float(row["mean_cosine"]) < threshold
        if not below:
            continue
        total += 1
        if bool(row["dead"]):
            counts["dead"] += 1
        elif bool(row["protected"]):
            counts["protected"] += 1
        else:
            counts["alive"] += 1
    return {
        "total_below": total,
        "dead_below": counts["dead"],
        "protected_below": counts["protected"],
        "alive_below": counts["alive"],
        "fraction_below": total / len(payload["clr_theory"]["decisions"]),
    }


def make_head_threshold_evidence() -> Path:
    selected_specs = [MODEL_SPECS[0], MODEL_SPECS[-1]]
    payloads = {spec.key: _load_payload(spec) for spec in selected_specs}
    joined = {spec.key: _join_head_rows(spec, payloads[spec.key]) for spec in selected_specs}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=STATUS_COLORS[name], label=STATUS_LABELS[name], markersize=7)
        for name in STATUS_ORDER
    ]

    for ax, spec in zip(axes[0], selected_specs):
        rows = joined[spec.key]
        payload = payloads[spec.key]
        safe_line = float(payload["clr_theory"]["ground_truth_loss_threshold"])
        chi_threshold = _artifact_chi(spec, payload)
        for status in STATUS_ORDER:
            pts = [row for row in rows if row["status"] == status]
            if not pts:
                continue
            ax.scatter(
                [row["normalized_coupling"] for row in pts],
                [row["delta_loss"] for row in pts],
                s=26,
                alpha=0.82,
                c=STATUS_COLORS[status],
                edgecolors="none",
            )
        ax.axvline(chi_threshold, color="black", linestyle="--", linewidth=1.4, alpha=0.85)
        ax.axhline(safe_line, color="#444444", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel(r"Normalized coupling $\sqrt{d}\,\bar{c}_h$")
        ax.set_ylabel(r"One-head ablation $\Delta$loss")
        ax.set_title(f"{spec.label}: normalized coupling vs. ablation damage")
        ax.text(
            0.98,
            0.08,
            rf"$\chi_{{c,\mathrm{{artifact}}}} = {chi_threshold:.5f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        ax.text(0.98, 0.02, f"safe line = {safe_line:.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#444444")

    axes[0, 0].legend(handles=legend_handles, frameon=True, fontsize=9, loc="upper left")

    cmap = ListedColormap([STATUS_COLORS[name] for name in STATUS_ORDER])
    for ax, spec in zip(axes[1], selected_specs):
        rows = joined[spec.key]
        matrix = _decision_matrix(rows)
        ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=len(STATUS_ORDER) - 1)
        ax.set_title(f"{spec.label}: final head decisions")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        n_layers, n_heads = matrix.shape
        x_step = 1 if n_heads <= 16 else 4
        y_step = 1 if n_layers <= 16 else 4
        ax.set_xticks(np.arange(0, n_heads, x_step))
        ax.set_yticks(np.arange(0, n_layers, y_step))

    fig.suptitle("EXP-98: Head-Level Threshold Evidence", fontsize=15)
    out_path = FIGURE_ROOT / "exp98_head_threshold_evidence.png"
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_normalized_collapse() -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(-10.0, 20.0, 60)
    palette = {
        "gpt2": "#1f77b4",
        "gpt2-medium": "#ff7f0e",
        "qwen25-0.5b": "#2ca02c",
        "smollm2-360m": "#d62728",
        "open-llama-7b": "#9467bd",
    }
    breakdown_rows = []
    threshold_rows = []
    for spec in MODEL_SPECS:
        payload = _load_payload(spec)
        rows = _join_head_rows(spec, payload)
        values = [row["normalized_coupling"] for row in rows]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=palette[spec.key],
            label=spec.label,
        )
        breakdown_rows.append((spec.label, _below_threshold_breakdown(spec, payload), len(rows)))
        threshold_rows.append((spec.label, _artifact_chi(spec, payload), palette[spec.key]))
    for label, chi_threshold, color in threshold_rows:
        ax.axvline(chi_threshold, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
    chi_values = [row[1] for row in threshold_rows]
    chi_mid = 0.5 * (min(chi_values) + max(chi_values))
    ax.axvline(
        chi_mid,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label=rf"bundle exact $\chi_c \in [{min(chi_values):.5f}, {max(chi_values):.5f}]$",
    )
    ax.set_xlabel(r"Normalized coupling $\sqrt{d}\,\bar{c}_h$")
    ax.set_ylabel("Density")
    ax.set_title("Cross-model normalized coupling collapse")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1]
    labels = [row[0] for row in breakdown_rows]
    totals = [row[2] for row in breakdown_rows]
    dead = [100.0 * row[1]["dead_below"] / total for row, total in zip(breakdown_rows, totals)]
    prot = [100.0 * row[1]["protected_below"] / total for row, total in zip(breakdown_rows, totals)]
    alive = [100.0 * row[1]["alive_below"] / total for row, total in zip(breakdown_rows, totals)]
    x = np.arange(len(labels))
    ax.bar(x, dead, color=STATUS_COLORS["dead"], label="Dead below threshold")
    ax.bar(x, prot, bottom=dead, color=STATUS_COLORS["boundary"], label="Protected below threshold")
    ax.bar(x, alive, bottom=np.asarray(dead) + np.asarray(prot), color=STATUS_COLORS["alive"], label="Alive below threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Head fraction (%)")
    ax.set_title("Outcome of below-threshold heads")
    ax.legend(frameon=True, fontsize=8)

    fig.suptitle("EXP-98: Universal Threshold in Normalized Coordinates", fontsize=15)
    out_path = FIGURE_ROOT / "exp98_normalized_coupling_collapse.png"
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    evidence = make_head_threshold_evidence()
    collapse = make_normalized_collapse()
    print(f"[exp98-threshold-plot] wrote {evidence}")
    print(f"[exp98-threshold-plot] wrote {collapse}")


if __name__ == "__main__":
    main()
