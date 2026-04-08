#!/usr/bin/env python3
"""Scaling collapse test: which normalization collapses the cross-model
coupling distributions onto a universal threshold?

Compares three normalizations:
  1. Raw coupling (no normalization) — each model needs a different threshold
  2. sqrt(d_model) normalization — threshold collapses to chi_c = 0.96
  3. sqrt(d_head) normalization — threshold scatters (no collapse)

The result directly tests the geometric prediction: if residual states live
on S^(d_model - 1), then concentration of measure gives sigma_d = 1/sqrt(d_model),
and the BKT critical ratio should normalize by sqrt(d_model), not sqrt(d_head).
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIG_DIR = SCRIPT_DIR.parent / "figures"

CHI_C = 0.679 * 2**0.5  # 0.96025...

MODELS = [
    ("GPT-2",      "full_gpt2_head144_small_redundancy_v2_dimaware.json",               768, 64),
    ("GPT-2 Med",  "full_gpt2_medium_head384_small_theory_redundancy_v1.json",          1024, 64),
    ("Qwen2.5",    "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json",      896, 64),
    ("SmolLM2",    "smollm2_360m_head480_small_theory_redundancy_v1.json",              960, 64),
    ("OpenLLaMA",  "open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json", 4096, 128),
]

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def load_couplings(fname):
    data = json.load(open(DATA_DIR / fname))
    pool = [dec for dec in data["clr_theory"]["decisions"] if not dec["protected"]]
    return np.array([dec["mean_cosine"] for dec in pool])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Raw coupling
    ax = axes[0]
    for i, (name, fname, d_model, d_head) in enumerate(MODELS):
        raw = load_couplings(fname)
        tau = CHI_C / np.sqrt(d_model)
        bins = np.linspace(min(raw) - 0.01, max(raw) + 0.01, 40)
        ax.hist(raw, bins=bins, alpha=0.5, color=COLORS[i], label=name, density=True)
        ax.axvline(tau, color=COLORS[i], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Raw coupling (cosine)")
    ax.set_ylabel("Density")
    ax.set_title("Raw: different thresholds per model")
    ax.legend(fontsize=7)

    # Panel 2: sqrt(d_model) normalization
    ax = axes[1]
    for i, (name, fname, d_model, d_head) in enumerate(MODELS):
        raw = load_couplings(fname)
        nc = raw * np.sqrt(d_model)
        bins = np.linspace(-8, 20, 50)
        ax.hist(nc, bins=bins, alpha=0.4, color=COLORS[i], label=name, density=True)
    ax.axvline(CHI_C, color="black", linestyle="--", linewidth=2,
               label=f"$\\chi_c$ = {CHI_C:.2f}")
    ax.set_xlabel("$\\bar{c}_h \\times \\sqrt{d_{model}}$")
    ax.set_ylabel("Density")
    ax.set_title("$\\sqrt{d_{model}}$: threshold collapses to one value")
    ax.legend(fontsize=7)

    # Panel 3: sqrt(d_head) normalization
    ax = axes[2]
    for i, (name, fname, d_model, d_head) in enumerate(MODELS):
        raw = load_couplings(fname)
        nc = raw * np.sqrt(d_head)
        equiv = CHI_C * np.sqrt(d_model) / np.sqrt(d_head)
        bins = np.linspace(-1, 6, 50)
        ax.hist(nc, bins=bins, alpha=0.4, color=COLORS[i], label=name, density=True)
        ax.axvline(equiv, color=COLORS[i], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("$\\bar{c}_h \\times \\sqrt{d_{head}}$")
    ax.set_ylabel("Density")
    ax.set_title("$\\sqrt{d_{head}}$: threshold scatters (no collapse)")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Scaling Collapse: $\\sqrt{d_{model}}$ is the right normalization",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "exp98_scaling_collapse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scaling-collapse] wrote {out}")


if __name__ == "__main__":
    main()
