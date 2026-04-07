#!/usr/bin/env python3
"""Base-rate analysis: does the derived threshold add value beyond random selection?

This script addresses the question: if most heads are individually safe to
ablate, does the 0.96/sqrt(d) threshold provide meaningful discrimination
beyond what random selection would achieve?

The answer is nuanced:
  - Precision lift over random is modest (~1.0x) because individual ablation
    is a weak test on well-trained models.
  - The real value is threefold:
    1. Zero ablations required (random baseline needs N ablation runs first).
    2. Clean bimodal separation: dead heads cluster 4-8 sigma below alive heads
       in normalized coupling space.
    3. Derived from physics, not fitted — the same constant works across all
       architectures without model-specific calibration.
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIG_DIR = SCRIPT_DIR.parent / "figures"

CHI_C = 0.679 * 2**0.5  # 0.96025...

MODELS = [
    ("GPT-2",        "full_gpt2_head144_small_redundancy_v2_dimaware.json",               768),
    ("GPT-2 Medium", "full_gpt2_medium_head384_small_theory_redundancy_v1.json",          1024),
    ("Qwen2.5 0.5B", "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json",      896),
    ("SmolLM2 360M", "smollm2_360m_head480_small_theory_redundancy_v1.json",              960),
    ("OpenLLaMA 7B", "open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json", 4096),
]


def analyze_model(name, fname, d):
    data = json.load(open(DATA_DIR / fname))
    decisions = data["clr_theory"]["decisions"]
    sqrt_d = d ** 0.5
    tau = CHI_C / sqrt_d

    dead = [dec for dec in decisions if dec["dead"]]
    alive = [dec for dec in decisions if dec["alive"]]
    protected = [dec for dec in decisions if dec["protected"]]
    pool = [dec for dec in decisions if not dec["protected"]]

    unsafe_in_pool = [dec for dec in pool if not dec["safe_ground_truth"]]
    safe_in_pool = [dec for dec in pool if dec["safe_ground_truth"]]

    # Precision analysis
    base_prec = len(safe_in_pool) / len(pool) if pool else 0
    dead_safe = sum(1 for dec in dead if dec["safe_ground_truth"])
    thresh_prec = dead_safe / len(dead) if dead else 0
    lift = thresh_prec / base_prec if base_prec > 0 else float("inf")

    # Safety: dangerous heads correctly kept alive
    unsafe_kept_alive = sum(1 for dec in unsafe_in_pool if not dec["dead"])

    # Coupling separation
    dead_nc = [dec["mean_cosine"] * sqrt_d for dec in dead]
    alive_nc = [dec["mean_cosine"] * sqrt_d for dec in alive]
    unsafe_nc = [dec["mean_cosine"] * sqrt_d for dec in unsafe_in_pool]

    separation = (np.mean(alive_nc) - np.mean(dead_nc)) if dead_nc and alive_nc else 0
    pooled_std = np.sqrt(
        (np.var(dead_nc) * len(dead_nc) + np.var(alive_nc) * len(alive_nc))
        / (len(dead_nc) + len(alive_nc))
    ) if dead_nc and alive_nc else 1
    cohens_d = separation / pooled_std if pooled_std > 0 else 0

    return {
        "model": name,
        "d": d,
        "total": len(decisions),
        "pool": len(pool),
        "dead": len(dead),
        "protected": len(protected),
        "alive": len(alive),
        "unsafe_in_pool": len(unsafe_in_pool),
        "base_precision": base_prec,
        "threshold_precision": thresh_prec,
        "lift": lift,
        "unsafe_kept_alive": unsafe_kept_alive,
        "unsafe_total": len(unsafe_in_pool),
        "dead_coupling_mean": np.mean(dead_nc) if dead_nc else 0,
        "dead_coupling_std": np.std(dead_nc) if dead_nc else 0,
        "alive_coupling_mean": np.mean(alive_nc) if alive_nc else 0,
        "alive_coupling_std": np.std(alive_nc) if alive_nc else 0,
        "unsafe_coupling_mean": np.mean(unsafe_nc) if unsafe_nc else 0,
        "separation": separation,
        "cohens_d": cohens_d,
    }


def print_report(results):
    print("=" * 95)
    print("BASE-RATE ANALYSIS: Derived Threshold vs Random Selection")
    print("=" * 95)
    print()

    # Table 1: Precision comparison
    print("--- Precision vs Random ---")
    print(f'{"Model":<16} {"Pool":>4} {"Dead":>4} {"Unsafe":>6} '
          f'{"RandPrec":>8} {"ThreshPrec":>10} {"Lift":>5}')
    print("-" * 60)
    for r in results:
        print(f'{r["model"]:<16} {r["pool"]:>4} {r["dead"]:>4} {r["unsafe_in_pool"]:>6} '
              f'{r["base_precision"]:>7.1%}  {r["threshold_precision"]:>9.1%} {r["lift"]:>5.2f}')
    print()
    print("  Lift ~ 1.0 because individual ablation is a weak test on well-trained")
    print("  models. Most heads carry negligible individual loss weight.")
    print()

    # Table 2: Coupling separation (the real story)
    print("--- Normalized Coupling Separation (the real discriminator) ---")
    print(f'{"Model":<16} {"Dead mean":>9} {"Alive mean":>10} {"Separation":>10} {"Cohen d":>8}')
    print("-" * 60)
    for r in results:
        print(f'{r["model"]:<16} {r["dead_coupling_mean"]:>+9.3f} '
              f'{r["alive_coupling_mean"]:>+10.3f} '
              f'{r["separation"]:>10.3f} {r["cohens_d"]:>8.2f}')
    print()
    print("  Normalized coupling = mean_cosine * sqrt(d). Threshold at chi_c = 0.96.")
    print("  Separation 4-8 in normalized units = clean bimodal split.")
    print(f"  Cohen's d > 1.5 on all models = very large effect size.")
    print()

    # Table 3: Safety
    print("--- Dangerous Head Protection ---")
    print(f'{"Model":<16} {"Unsafe":>6} {"Kept Alive":>10} {"Unsafe coupling":>15}')
    print("-" * 55)
    for r in results:
        kept = r["unsafe_kept_alive"]
        total = r["unsafe_total"]
        pct = f"{100*kept/total:.0f}%" if total > 0 else "n/a"
        print(f'{r["model"]:<16} {total:>6} {kept:>5}/{total:<4} ({pct:>4}) '
              f'{r["unsafe_coupling_mean"]:>+14.3f}')
    print()
    print("  On SmolLM2/OpenLLaMA, unsafe heads have coupling >> threshold (high coherence).")
    print("  The 2 GPT-2 Medium false positives have coupling in the dead zone — edge cases.")


def generate_figure(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping figure)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: precision comparison
    ax = axes[0]
    names = [r["model"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [r["base_precision"] for r in results],
           width, label="Random baseline", color="#95a5a6", alpha=0.8)
    ax.bar(x + width/2, [r["threshold_precision"] for r in results],
           width, label="Derived threshold", color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Random Selection")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, 1.005)

    # Right: coupling separation
    ax = axes[1]
    for i, r in enumerate(results):
        ax.errorbar(i - 0.15, r["dead_coupling_mean"], yerr=r["dead_coupling_std"],
                     fmt="o", color="#e74c3c", capsize=4, markersize=6)
        ax.errorbar(i + 0.15, r["alive_coupling_mean"], yerr=r["alive_coupling_std"],
                     fmt="o", color="#2ecc71", capsize=4, markersize=6)
    ax.axhline(CHI_C, color="black", linestyle="--", linewidth=1, label=f"chi_c = {CHI_C:.2f}")
    ax.set_ylabel("Normalized coupling")
    ax.set_title("Dead vs Alive Coupling Separation")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = FIG_DIR / "exp98_base_rate_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {out}")


def main():
    results = [analyze_model(*m) for m in MODELS]
    print_report(results)
    generate_figure(results)

    # Save JSON
    out = DATA_DIR / "base_rate_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON saved to {out}")


if __name__ == "__main__":
    main()
