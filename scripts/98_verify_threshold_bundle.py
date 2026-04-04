"""Verify the dead-head threshold transfer bundle against frozen JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PUB_ROOT = Path(__file__).resolve().parents[1]
LATTICE_DEATH_POINT = 0.679
SIGMA_S1 = 1.0 / math.sqrt(2.0)
CHI_C_EXACT = LATTICE_DEATH_POINT / SIGMA_S1

CATALOG = [
    {
        "label": "GPT-2",
        "family": "GPT-2",
        "d_model": 768,
        "artifact": "full_gpt2_head144_small_redundancy_v2_dimaware.json",
    },
    {
        "label": "GPT-2 Medium",
        "family": "GPT-2",
        "d_model": 1024,
        "artifact": "full_gpt2_medium_head384_small_theory_redundancy_v1.json",
    },
    {
        "label": "Qwen2.5 0.5B",
        "family": "Qwen2",
        "d_model": 896,
        "artifact": "qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json",
    },
    {
        "label": "SmolLM2 360M",
        "family": "Llama",
        "d_model": 960,
        "artifact": "smollm2_360m_head480_small_theory_redundancy_v1.json",
    },
    {
        "label": "OpenLLaMA 7B",
        "family": "Llama",
        "d_model": 4096,
        "artifact": "open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the bundled EXP-98 threshold artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PUB_ROOT / "data",
        help="Folder containing the frozen JSON artifacts.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PUB_ROOT / "data" / "threshold_transfer_summary.json",
        help="Where to write the verified JSON summary.",
    )
    return parser.parse_args()


def verify_threshold(d_model: int, observed_tau: float, *, tol: float = 5e-4) -> tuple[float, float, float]:
    chi_artifact = observed_tau * math.sqrt(d_model)
    if abs(chi_artifact - CHI_C_EXACT) > tol:
        raise ValueError(
            "normalized threshold mismatch for "
            f"d={d_model}: chi_artifact={chi_artifact:.10f}, chi_exact={CHI_C_EXACT:.10f}"
        )
    tau_exact_transfer = CHI_C_EXACT / math.sqrt(d_model)
    tau_reference_rounded = 0.96 / math.sqrt(d_model)
    return chi_artifact, tau_exact_transfer, tau_reference_rounded


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_entry(data_dir: Path, spec: dict) -> dict:
    artifact_path = data_dir / spec["artifact"]
    payload = load_payload(artifact_path)
    theory = payload["clr_theory"]
    redundancy = payload["redundancy_pass"]
    decisions = theory["decisions"]
    total_heads = len(decisions)

    observed_tau = float(theory["death_threshold"])
    chi_artifact, tau_exact_transfer, tau_reference_rounded = verify_threshold(spec["d_model"], observed_tau)

    dead_count = int(theory["dead_count"])
    protected_count = int(theory["protected_count"])
    alive_count = int(theory["alive_count"])
    if dead_count + protected_count + alive_count != total_heads:
        raise ValueError(
            f"count mismatch for {spec['label']}: dead+protected+alive != total ({total_heads})"
        )

    redundant_count = int(redundancy["redundant_count"])
    combined_removed = dead_count + redundant_count
    combined_precision = (
        (float(theory["dead_precision"]) * dead_count) + (float(redundancy["redundant_precision"]) * redundant_count)
    ) / max(1, combined_removed)
    combined_delta_loss = float(redundancy["eval_summary_dead_plus_redundant"]["loss"]) - float(payload["baseline_eval"]["loss"])

    return {
        "model": spec["label"],
        "family": spec["family"],
        "artifact": spec["artifact"],
        "model_name": payload["model_name"],
        "d_model": spec["d_model"],
        "tau_artifact": round(observed_tau, 8),
        "tau_exact_transfer": round(tau_exact_transfer, 8),
        "tau_reference_rounded": round(tau_reference_rounded, 8),
        "chi_c_artifact": round(chi_artifact, 6),
        "dead_precision": round(float(theory["dead_precision"]), 4),
        "redundancy_precision": round(float(redundancy["redundant_precision"]), 4),
        "dead_heads": dead_count,
        "protected_heads": protected_count,
        "alive_heads": alive_count,
        "total_heads": total_heads,
        "heads_removed": combined_removed,
        "combined_precision": round(combined_precision, 4),
        "combined_delta_loss": round(combined_delta_loss, 5),
    }


def render_table(summary: dict) -> str:
    headers = [
        "Model",
        "d_model",
        "tau_artifact",
        "chi_artifact",
        "Dead Precision",
        "Red. Precision",
        "Heads Removed",
        "Combined Precision",
    ]
    rows = []
    for row in summary["models"]:
        rows.append(
            [
                row["model"],
                str(row["d_model"]),
                f"{row['tau_artifact']:.5f}",
                f"{row['chi_c_artifact']:.5f}",
                f"{row['dead_precision'] * 100:.1f}%",
                f"{row['redundancy_precision'] * 100:.1f}%",
                f"{row['heads_removed']} / {row['total_heads']}",
                f"{row['combined_precision'] * 100:.1f}%",
            ]
        )

    widths = []
    for col_idx, header in enumerate(headers):
        widths.append(max(len(header), max(len(row[col_idx]) for row in rows)))

    def fmt(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[idx]) for idx, part in enumerate(parts))

    lines = [
        fmt(headers),
        fmt(["-" * width for width in widths]),
    ]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    models = [summarize_entry(args.data_dir, spec) for spec in CATALOG]
    chi_values = [row["chi_c_artifact"] for row in models]
    summary = {
        "claim_scope": "Five ablation-validated decoder checkpoints in the dead-head transfer bundle.",
        "threshold_formula": "tau_death(d) = chi_c / sqrt(d_model)",
        "threshold_note": (
            "The frozen bundle uses the exact threshold stored in each artifact. "
            "Across the five checkpoints, chi_c_artifact spans "
            f"{min(chi_values):.5f} to {max(chi_values):.5f}; the paper cites chi_c ≈ 0.96 "
            "as the rounded normalized critical ratio."
        ),
        "chi_c_exact_transfer": round(CHI_C_EXACT, 6),
        "models": models,
    }

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(render_table(summary))
    print(f"\n[exp98-verify] wrote {args.out_json}")


if __name__ == "__main__":
    main()
