#!/usr/bin/env python3
"""Coherence Anatomy Scanner — standalone dead-head identification.

Scan any Hugging Face causal language model for dead attention heads using the
derived threshold from coupled-oscillator criticality:

    tau_death(d) = chi_c / sqrt(d_model),    chi_c = 0.96

No model-specific calibration is needed. The threshold is universal.

Usage:
    python coherence_anatomy_scan.py --model gpt2
    python coherence_anatomy_scan.py --model meta-llama/Llama-3.2-1B
    python coherence_anatomy_scan.py --model Qwen/Qwen2.5-0.5B --device cuda
    python coherence_anatomy_scan.py --model gpt2 --output anatomy.json
    python coherence_anatomy_scan.py --model gpt2 --report gpt2_anatomy.html

Requirements:
    pip install torch transformers numpy datasets matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Derived threshold constant
# ---------------------------------------------------------------------------
# CLR bond death on S^1:   cos(Delta theta)_death = 0.679
# Normalize by sigma_1:    chi_c = 0.679 / (1/sqrt(2)) = 0.96025
# Transfer to S^(d-1):     tau_death(d) = chi_c / sqrt(d)
CHI_C = 0.679 * (2 ** 0.5)  # 0.96025...


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------
def get_hidden_size(model) -> int:
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise AttributeError(f"Cannot determine hidden size from {type(model.config)}")


def get_num_layers(model) -> int:
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise AttributeError(f"Cannot determine layer count from {type(model.config)}")


def get_num_heads(model) -> int:
    for attr in ("n_head", "num_attention_heads"):
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise AttributeError(f"Cannot determine head count from {type(model.config)}")


def get_head_dim(model) -> int:
    if hasattr(model.config, "head_dim"):
        return int(model.config.head_dim)
    return get_hidden_size(model) // get_num_heads(model)


def get_layers(model):
    """Return the list of transformer blocks."""
    for path in (
        "transformer.h",        # GPT-2, GPT-J
        "model.layers",         # Llama, Qwen, Gemma, Mistral
        "gpt_neox.layers",      # GPT-NeoX, Pythia
        "model.decoder.layers", # OPT
    ):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise AttributeError(f"Cannot find transformer layers in {type(model)}")


def get_output_projection(block):
    """Return the attention output projection (W_O) for a block."""
    for path in (
        "attn.c_proj",          # GPT-2
        "self_attn.o_proj",     # Llama, Qwen, Gemma, Mistral
        "attention.dense",      # GPT-NeoX
        "self_attn.out_proj",   # OPT
    ):
        obj = block
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"Cannot find output projection in {type(block)}")


def get_proj_weight(proj, n_head: int, head_dim: int, hidden_size: int):
    """Get W_O as (n_head, head_dim, hidden_size) for write-back computation."""
    w = proj.weight.data
    if w.shape == (hidden_size, hidden_size):
        # Standard: (out, in) where in = n_head * head_dim
        return w.T.reshape(n_head, head_dim, hidden_size)
    elif w.shape == (hidden_size, n_head * head_dim):
        return w.T.reshape(n_head, head_dim, hidden_size)
    # Conv1D (GPT-2): weight is (in, out)
    if hasattr(proj, "nf"):
        return w.reshape(n_head, head_dim, hidden_size)
    raise ValueError(f"Unexpected W_O shape {w.shape} for n_head={n_head}, "
                     f"head_dim={head_dim}, hidden_size={hidden_size}")


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def build_calibration_batches(
    tokenizer, n_batches: int = 32, seq_len: int = 128, seed: int = 42
) -> list[torch.Tensor]:
    """Build calibration batches from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text)
    rng = np.random.RandomState(seed)
    batches = []
    for _ in range(n_batches):
        start = rng.randint(0, len(tokens) - seq_len - 1)
        ids = torch.tensor([tokens[start : start + seq_len]], dtype=torch.long)
        batches.append(ids)
    return batches


# ---------------------------------------------------------------------------
# Core: head coupling measurement
# ---------------------------------------------------------------------------
def measure_head_couplings(
    model,
    calibration_batches: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    """Measure per-head mean coupling to the residual stream.

    Returns array of shape (n_batches, n_layers, n_heads) where each entry
    is the position-averaged cosine between the head's write-back signal
    and the receiver's pre-head residual state.
    """
    n_layers = get_num_layers(model)
    n_heads = get_num_heads(model)
    head_dim = get_head_dim(model)
    hidden_size = get_hidden_size(model)

    state = {"residuals": {}, "current": None}
    handles = []
    layers = get_layers(model)

    for layer_idx, block in enumerate(layers):
        output_proj = get_output_projection(block)
        w_o = get_proj_weight(output_proj, n_heads, head_dim, hidden_size)

        def _block_pre(module, inputs, layer_idx=layer_idx):
            state["residuals"][layer_idx] = inputs[0].detach()

        def _proj_pre(module, inputs, layer_idx=layer_idx, w_o=w_o):
            current = state["current"]
            if current is None:
                return
            residual = state["residuals"].get(layer_idx)
            if residual is None:
                return
            x = inputs[0]  # (batch, seq, n_heads * head_dim)
            batch, seq, _ = x.shape
            head_view = x.reshape(batch, seq, n_heads, head_dim)
            # Compute each head's write-back: contribution to residual stream
            contrib = torch.einsum("bthd,hdo->btho", head_view, w_o)
            ref = residual.unsqueeze(2).expand(batch, seq, n_heads, hidden_size)
            cos = F.cosine_similarity(contrib, ref, dim=-1, eps=1e-8)
            current[layer_idx] = cos.mean(dim=(0, 1)).detach().float().cpu().numpy()

        handles.append(block.register_forward_pre_hook(_block_pre))
        handles.append(output_proj.register_forward_pre_hook(_proj_pre))

    batch_results = []
    try:
        with torch.no_grad():
            for ids in calibration_batches:
                state["current"] = np.zeros((n_layers, n_heads), dtype=np.float64)
                state["residuals"].clear()
                model(input_ids=ids.to(device), use_cache=False)
                batch_results.append(state["current"].copy())
    finally:
        for h in handles:
            h.remove()

    return np.stack(batch_results, axis=0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@dataclass
class HeadClassification:
    layer: int
    head: int
    mean_coupling: float
    normalized_coupling: float  # mean_coupling * sqrt(d) — dimensionless
    tau_death: float
    batch_death_frac: float  # fraction of batches where head was below threshold
    is_dead: bool
    is_protected: bool
    reason: str  # "dead", "alive", "protected_boundary"


def classify_heads(
    couplings: np.ndarray,
    hidden_size: int,
    chi_c: float = CHI_C,
    boundary_layers: int = 2,
    consistency_threshold: float = 0.5,
) -> list[HeadClassification]:
    """Classify heads as dead/alive using the derived threshold.

    A head is classified as dead only if:
    1. Its mean coupling across all batches is below tau_death, AND
    2. It was below threshold on at least `consistency_threshold` fraction
       of individual calibration batches (default: 50%).

    This batch-consistency check prevents transient fluctuations from
    producing false dead-head classifications.
    """
    tau = chi_c / (hidden_size ** 0.5)
    mean_per_head = couplings.mean(axis=0)  # (n_layers, n_heads)
    # Per-batch death: (n_batches, n_layers, n_heads) boolean
    per_batch_dead = couplings < tau
    batch_death_frac = per_batch_dead.mean(axis=0)  # (n_layers, n_heads)
    n_layers, n_heads = mean_per_head.shape
    sqrt_d = hidden_size ** 0.5

    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            c = float(mean_per_head[layer, head])
            bdf = float(batch_death_frac[layer, head])
            below_threshold = c < tau
            consistent = bdf >= consistency_threshold
            is_boundary = layer < boundary_layers or layer >= n_layers - 1

            if below_threshold and consistent and is_boundary:
                reason = "protected_boundary"
                is_dead, is_protected = False, True
            elif below_threshold and consistent:
                reason = "dead"
                is_dead, is_protected = True, False
            elif below_threshold and not consistent:
                reason = "alive"  # below mean but inconsistent across batches
                is_dead, is_protected = False, False
            else:
                reason = "alive"
                is_dead, is_protected = False, False

            results.append(HeadClassification(
                layer, head, c, c * sqrt_d, tau, bdf,
                is_dead, is_protected, reason))

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_anatomy(
    results: list[HeadClassification],
    model_name: str,
    hidden_size: int,
    n_layers: int,
    n_heads: int,
):
    tau = CHI_C / (hidden_size ** 0.5)
    dead = [r for r in results if r.is_dead]
    protected = [r for r in results if r.is_protected]
    alive = [r for r in results if not r.is_dead and not r.is_protected]
    total = len(results)

    print(f"\n{'='*70}")
    print(f"COHERENCE ANATOMY SCAN: {model_name}")
    print(f"{'='*70}")
    print(f"  Hidden dim:    {hidden_size}")
    print(f"  Layers:        {n_layers}")
    print(f"  Heads/layer:   {n_heads}")
    print(f"  Total heads:   {total}")
    print(f"  tau_death:     {tau:.5f}  (chi_c={CHI_C:.5f} / sqrt({hidden_size}))")
    print(f"  Dead:          {len(dead)} ({100*len(dead)/total:.1f}%)")
    print(f"  Protected:     {len(protected)} ({100*len(protected)/total:.1f}%)")
    print(f"  Alive:         {len(alive)} ({100*len(alive)/total:.1f}%)")
    print()

    # Layer-by-layer map
    print("  Layer map (D=dead, P=protected, .=alive):")
    print(f"  {'':>4}", end="")
    for h in range(n_heads):
        print(f" {h:>2}", end="")
    print()

    for layer in range(n_layers):
        layer_results = [r for r in results if r.layer == layer]
        print(f"  L{layer:02d} ", end="")
        for r in sorted(layer_results, key=lambda x: x.head):
            if r.is_dead:
                print(" D ", end="")
            elif r.is_protected:
                print(" P ", end="")
            else:
                print(" . ", end="")
        n_dead = sum(1 for r in layer_results if r.is_dead)
        print(f"  {n_dead}/{n_heads}")
    print()

    # Dead head detail
    if dead:
        print("  Dead heads (sorted by coupling):")
        print(f"    {'Head':>8}  {'coupling':>10}  {'normalized':>10}  {'batch_%dead':>11}")
        for r in sorted(dead, key=lambda x: x.mean_coupling):
            print(f"    L{r.layer:02d}H{r.head:02d}  {r.mean_coupling:+10.5f}"
                  f"  {r.normalized_coupling:+10.3f}"
                  f"  {r.batch_death_frac:10.0%}")
        print()


# ---------------------------------------------------------------------------
# Report generation (HTML with embedded figures)
# ---------------------------------------------------------------------------
def generate_report(
    results: list[HeadClassification],
    couplings: np.ndarray,
    model_name: str,
    hidden_size: int,
    n_layers: int,
    n_heads: int,
    report_path: str,
):
    """Generate a self-contained HTML report with embedded visualizations."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    tau = CHI_C / (hidden_size ** 0.5)
    sqrt_d = hidden_size ** 0.5
    dead = [r for r in results if r.is_dead]
    protected = [r for r in results if r.is_protected]
    alive = [r for r in results if not r.is_dead and not r.is_protected]
    mean_per_head = couplings.mean(axis=0)

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    figures = {}

    # --- Figure 1: Anatomy heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(4, n_layers * 0.3)))
    # Build status matrix: 0=alive, 1=dead, 2=protected
    status = np.zeros((n_layers, n_heads), dtype=int)
    for r in results:
        if r.is_dead:
            status[r.layer, r.head] = 1
        elif r.is_protected:
            status[r.layer, r.head] = 2
    cmap = ListedColormap(["#2ecc71", "#e74c3c", "#3498db"])  # alive, dead, protected
    im = ax.imshow(status, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Head Anatomy Map — {model_name}")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"Alive ({len(alive)})"),
        Patch(facecolor="#e74c3c", label=f"Dead ({len(dead)})"),
        Patch(facecolor="#3498db", label=f"Protected ({len(protected)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    figures["anatomy_map"] = fig_to_base64(fig)

    # --- Figure 2: Coupling heatmap (raw values) ---
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(4, n_layers * 0.3)))
    vmax = max(abs(mean_per_head.min()), abs(mean_per_head.max()))
    im = ax.imshow(mean_per_head, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Mean Coupling per Head — {model_name}")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    plt.colorbar(im, ax=ax, label="Mean coupling (cosine)")
    # Overlay threshold annotation
    ax.text(0.02, 0.02, f"tau = {tau:.5f}", transform=ax.transAxes,
            fontsize=8, color="black", backgroundcolor="white", alpha=0.8)
    figures["coupling_heatmap"] = fig_to_base64(fig)

    # --- Figure 3: Normalized coupling distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    norm_couplings = mean_per_head.flatten() * sqrt_d
    dead_norm = [r.normalized_coupling for r in dead]
    alive_norm = [r.normalized_coupling for r in alive]
    prot_norm = [r.normalized_coupling for r in protected]
    bins = np.linspace(min(norm_couplings) - 0.2, max(norm_couplings) + 0.2, 50)
    if dead_norm:
        ax.hist(dead_norm, bins=bins, alpha=0.7, color="#e74c3c", label="Dead")
    if alive_norm:
        ax.hist(alive_norm, bins=bins, alpha=0.7, color="#2ecc71", label="Alive")
    if prot_norm:
        ax.hist(prot_norm, bins=bins, alpha=0.7, color="#3498db", label="Protected")
    ax.axvline(CHI_C, color="black", linestyle="--", linewidth=2,
               label=f"chi_c = {CHI_C:.3f}")
    ax.set_xlabel("Normalized coupling (mean_coupling x sqrt(d))")
    ax.set_ylabel("Count")
    ax.set_title(f"Normalized Coupling Distribution — {model_name}")
    ax.legend()
    figures["coupling_distribution"] = fig_to_base64(fig)

    # --- Figure 4: Per-layer dead fraction ---
    fig, ax = plt.subplots(figsize=(8, 4))
    dead_per_layer = np.zeros(n_layers)
    for r in results:
        if r.is_dead:
            dead_per_layer[r.layer] += 1
    prot_per_layer = np.zeros(n_layers)
    for r in results:
        if r.is_protected:
            prot_per_layer[r.layer] += 1
    alive_per_layer = n_heads - dead_per_layer - prot_per_layer
    x = np.arange(n_layers)
    ax.bar(x, dead_per_layer, color="#e74c3c", label="Dead")
    ax.bar(x, prot_per_layer, bottom=dead_per_layer, color="#3498db", label="Protected")
    ax.bar(x, alive_per_layer, bottom=dead_per_layer + prot_per_layer,
           color="#2ecc71", label="Alive")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head count")
    ax.set_title(f"Per-Layer Head Classification — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=7)
    ax.legend(fontsize=8)
    figures["per_layer_bar"] = fig_to_base64(fig)

    # --- Figure 5: Batch consistency scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for category, color, label in [
        (dead, "#e74c3c", "Dead"),
        (alive, "#2ecc71", "Alive"),
        (protected, "#3498db", "Protected"),
    ]:
        if category:
            ax.scatter(
                [r.mean_coupling for r in category],
                [r.batch_death_frac for r in category],
                c=color, alpha=0.6, s=20, label=label, edgecolors="none")
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.5,
               label=f"tau = {tau:.5f}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5,
               label="50% consistency")
    ax.set_xlabel("Mean coupling (cosine)")
    ax.set_ylabel("Batch death fraction")
    ax.set_title(f"Coupling vs Batch Consistency — {model_name}")
    ax.legend(fontsize=8)
    figures["batch_consistency"] = fig_to_base64(fig)

    # --- Build HTML ---
    dead_rows_html = ""
    for r in sorted(dead, key=lambda x: x.mean_coupling):
        dead_rows_html += (
            f"<tr><td>L{r.layer:02d}H{r.head:02d}</td>"
            f"<td>{r.mean_coupling:+.5f}</td>"
            f"<td>{r.normalized_coupling:+.3f}</td>"
            f"<td>{r.batch_death_frac:.0%}</td></tr>\n")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coherence Anatomy Report — {model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px;
         color: #333; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 40px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px; margin: 20px 0; }}
  .stat {{ background: white; border-radius: 8px; padding: 15px;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
  .stat .value {{ font-size: 2em; font-weight: bold; }}
  .stat .label {{ font-size: 0.85em; color: #777; }}
  .stat.dead .value {{ color: #e74c3c; }}
  .stat.alive .value {{ color: #2ecc71; }}
  .stat.prot .value {{ color: #3498db; }}
  .formula {{ background: #2c3e50; color: #ecf0f1; padding: 20px;
              border-radius: 8px; font-family: monospace; font-size: 1.2em;
              text-align: center; margin: 20px 0; }}
  img {{ max-width: 100%; border-radius: 4px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin: 10px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #ddd;
            font-size: 0.9em; }}
  th {{ background: #2c3e50; color: white; }}
  tr:hover {{ background: #f5f5f5; }}
  .derivation {{ background: white; padding: 20px; border-radius: 8px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;
                 font-size: 0.95em; }}
  .derivation ol {{ line-height: 1.8; }}
  .footer {{ margin-top: 40px; padding: 20px; border-top: 1px solid #ddd;
             font-size: 0.85em; color: #999; text-align: center; }}
</style>
</head>
<body>

<h1>Coherence Anatomy Report</h1>
<p style="font-size:1.2em; color:#555;">{model_name}</p>

<div class="formula">
  tau_death(d) = chi_c / sqrt(d) = {CHI_C:.5f} / sqrt({hidden_size})
  = {tau:.5f}
</div>

<div class="stats">
  <div class="stat"><div class="value">{hidden_size}</div><div class="label">Hidden dim</div></div>
  <div class="stat"><div class="value">{n_layers}</div><div class="label">Layers</div></div>
  <div class="stat"><div class="value">{n_heads}</div><div class="label">Heads/layer</div></div>
  <div class="stat"><div class="value">{len(results)}</div><div class="label">Total heads</div></div>
  <div class="stat dead"><div class="value">{len(dead)} ({100*len(dead)/len(results):.1f}%)</div><div class="label">Dead</div></div>
  <div class="stat prot"><div class="value">{len(protected)}</div><div class="label">Protected</div></div>
  <div class="stat alive"><div class="value">{len(alive)}</div><div class="label">Alive</div></div>
</div>

<h2>Anatomy Map</h2>
<p>Red = dead, green = alive, blue = protected (boundary layers).</p>
<img src="data:image/png;base64,{figures['anatomy_map']}" alt="Anatomy map">

<h2>Mean Coupling Heatmap</h2>
<p>Per-head mean cosine between write-back signal and pre-head residual.
Blue = negative (anti-aligned), red = positive (aligned). The threshold
tau = {tau:.5f} is near zero in this scale.</p>
<img src="data:image/png;base64,{figures['coupling_heatmap']}" alt="Coupling heatmap">

<h2>Normalized Coupling Distribution</h2>
<p>Coupling rescaled by sqrt(d) so the threshold collapses to the universal
constant chi_c = {CHI_C:.3f} (dashed line). Heads to the left of the line
are in the dead regime.</p>
<img src="data:image/png;base64,{figures['coupling_distribution']}" alt="Coupling distribution">

<h2>Per-Layer Classification</h2>
<p>Stacked bar chart showing how dead/protected/alive heads distribute across layers.
Boundary layers (first {2} and last 1) are fully protected.</p>
<img src="data:image/png;base64,{figures['per_layer_bar']}" alt="Per-layer bar">

<h2>Coupling vs Batch Consistency</h2>
<p>Each dot is one head. X-axis: mean coupling. Y-axis: fraction of calibration
batches where the head was below threshold. Heads must be both below threshold
AND consistent across batches to be classified as dead.</p>
<img src="data:image/png;base64,{figures['batch_consistency']}" alt="Batch consistency">

<h2>Dead Head Table</h2>
<table>
<tr><th>Head</th><th>Coupling</th><th>Normalized</th><th>Batch % Dead</th></tr>
{dead_rows_html}
</table>

<div class="derivation">
<h3>Derivation</h3>
<ol>
<li>CLR bond death on S<sup>1</sup>: cos(Delta theta)<sub>death</sub> = 0.679</li>
<li>Normalize by S<sup>1</sup> fluctuation scale sigma<sub>1</sub> = 1/sqrt(2):
    chi_c = 0.679 / (1/sqrt(2)) = {CHI_C:.5f}</li>
<li>Transfer to S<sup>d-1</sup> by concentration of measure:
    tau(d) = chi_c / sqrt(d) = {CHI_C:.5f} / sqrt({hidden_size}) = {tau:.5f}</li>
</ol>
<p><em>Reference: Sharpe (2026), "Coherence-Guided Dead-Head Identification
in Frozen Transformers." No parameter is fitted.</em></p>
</div>

<div class="footer">
Generated by coherence_anatomy_scan.py | Coherence-Guided Dead-Head Identification
</div>

</body>
</html>"""

    Path(report_path).write_text(html)
    print(f"Report written to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Coherence anatomy scan: identify dead attention heads "
                    "using the derived threshold tau = 0.96/sqrt(d).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model gpt2
  %(prog)s --model meta-llama/Llama-3.2-1B --device cuda
  %(prog)s --model Qwen/Qwen2.5-0.5B --output anatomy.json
  %(prog)s --model gpt2 --report gpt2_anatomy.html
  %(prog)s --model gpt2 --n-cal 64 --seq-len 256
""")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--n-cal", type=int, default=32,
                        help="Number of calibration batches")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Calibration sequence length")
    parser.add_argument("--boundary-layers", type=int, default=2,
                        help="Number of boundary layers to protect")
    parser.add_argument("--consistency", type=float, default=0.5,
                        help="Fraction of batches a head must be below threshold "
                             "to be classified as dead (default: 0.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--report", type=str, default=None,
                        help="Generate HTML report with visualizations (e.g. report.html)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress visual output")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    hidden_size = get_hidden_size(model)
    n_layers = get_num_layers(model)
    n_heads = get_num_heads(model)

    print(f"Building calibration data ({args.n_cal} batches, seq_len={args.seq_len})...")
    cal_batches = build_calibration_batches(
        tokenizer, n_batches=args.n_cal, seq_len=args.seq_len
    )

    print("Measuring head couplings...")
    couplings = measure_head_couplings(model, cal_batches, args.device)

    results = classify_heads(
        couplings, hidden_size,
        boundary_layers=args.boundary_layers,
        consistency_threshold=args.consistency,
    )

    if not args.quiet:
        print_anatomy(results, args.model, hidden_size, n_layers, n_heads)

    if args.output:
        out_path = Path(args.output)
        out_data = {
            "model": args.model,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": get_head_dim(model),
            "chi_c": CHI_C,
            "tau_death": CHI_C / (hidden_size ** 0.5),
            "total_heads": len(results),
            "dead_heads": sum(1 for r in results if r.is_dead),
            "protected_heads": sum(1 for r in results if r.is_protected),
            "alive_heads": sum(1 for r in results
                               if not r.is_dead and not r.is_protected),
            "heads": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(out_data, indent=2))
        print(f"\nResults written to {out_path}")

    if args.report:
        generate_report(
            results, couplings, args.model,
            hidden_size, n_layers, n_heads, args.report)

    # Summary line
    n_dead = sum(1 for r in results if r.is_dead)
    total = len(results)
    print(f"\n  {n_dead}/{total} heads dead ({100*n_dead/total:.1f}%)"
          f"  |  tau={CHI_C / hidden_size**0.5:.5f}"
          f"  |  chi_c={CHI_C:.5f}")


if __name__ == "__main__":
    main()
