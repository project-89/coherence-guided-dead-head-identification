# Agent Briefing: Coherence-Guided Dead-Head Identification

## What This Is

A physics-derived dead-head detector for frozen transformers. Each attention
head writes a signal `s_h(i) = sum_j A_ij * V_h(x_j)` into the residual
stream. Measure the mean cosine alignment between that write-back and the
pre-head residual state `x_i`. If the mean coupling falls below a derived
threshold, the head is dead.

The threshold:

```
tau_death(d) = chi_c / sqrt(d_model),    chi_c = 0.96
```

is derived from the Berezinskii-Kosterlitz-Thouless (BKT) phase transition
on coupled oscillators (bond death at cos = 0.679, normalized by the S^1
fluctuation scale, transferred to S^(d-1) by concentration of measure).
No parameter is fitted. No model-specific calibration is needed.

## Why It Matters

Every existing pruning method requires fitting a threshold per model. This
one does not. The same formula, with the same constant, achieves 95--100%
precision across GPT-2, GPT-2 Medium, Qwen2.5 0.5B, SmolLM2 360M,
OpenLLaMA 7B, and Gemma 3 4B (d = 768 through 4096, four architecture
families). That makes it an identification law rather than a heuristic.

## How to Implement the Dead-Head Scan

```python
import torch

def dead_head_scan(model, calibration_tokens, chi_c=0.96):
    """
    Returns list of (layer, head, mean_coupling, is_dead) tuples.

    Steps:
    1. Run forward pass, hooking each attention layer.
    2. For each head h at each receiver position i:
       - extract pre-head residual x_i (before the attention block)
       - extract write-back s_h(i) = sum_j A_ij * V_h(x_j)
       - compute c_h(i) = cos(s_h(i), x_i)
    3. Average c_h(i) over positions and calibration sequences
       to get one scalar mean_coupling per head.
    4. Compare to tau_death = chi_c / sqrt(d_model).
    5. Apply protections:
       - Boundary: first 1-2 layers are input transducers; protect them.
       - Bridge veto: heads with unusually high bridge mass in the
         layerwise attention graph are protected.
    """
    d_model = model.config.hidden_size
    tau = chi_c / (d_model ** 0.5)
    # Full implementation: scripts/coherence_anatomy_scan.py
    # Publication harness: scripts/98_coherence_pruning_harness.py
```

## Derivation (for reference)

1. BKT bond death threshold on S^1: `cos(Delta theta)_death = 0.679`
2. Normalize by S^1 fluctuation scale `sigma_1 = 1/sqrt(2)`:
   `chi_c = 0.679 / (1/sqrt(2)) = 0.96025`
3. Transfer to S^(d-1) via concentration of measure (`sigma_d = 1/sqrt(d)`):
   `tau_death(d) = chi_c * sigma_d = 0.96 / sqrt(d)`

## What to Claim / What Not to Claim

**DO claim:**
- The threshold identifies dead heads at 95--100% precision across six
  models with zero fitted parameters
- The threshold is derived from coupled-oscillator criticality
- Dead heads are individually safe to ablate (validated by per-head ablation)
- The identification law transfers across architecture families

**DO NOT claim:**
- That simultaneously removing all dead heads is lossless (it is catastrophic)
- Theorem closure that transformers are exactly vector Kuramoto systems
- The full generalized BKT critical theory on S^(d-1)
- Coherence-during-training as already validated
- That naive deletion is the correct downstream operator

The identification law is this paper. Removal is a separate engineering
problem addressed in follow-on work.

## File Map

| File | Purpose |
|---|---|
| `paper.tex` | Formal manuscript (source of truth) |
| `paper.pdf` | Compiled manuscript |
| `references.bib` | Bibliography |
| `data/` | Frozen JSON artifacts for six models |
| `figures/` | Publication figures |
| `scripts/coherence_anatomy_scan.py` | **Standalone scanner** — scan any HF model |
| `scripts/98_coherence_pruning_harness.py` | Full experiment harness |
| `scripts/98_result_plots.py` | Regenerate summary figure |
| `scripts/98_threshold_evidence_plots.py` | Head-level threshold evidence |
| `scripts/98_verify_threshold_bundle.py` | Verify transfer rows |

## Editing Rules

- `paper.tex` is the source of truth. If you change the manuscript,
  recompile `paper.pdf`.
- Prefer bundled JSONs in `data/` over reaching into `out/`.
- If you change a figure script, regenerate the figure in `figures/`.
- Generalized BKT work goes in `papers/current/lattice_theory/`, not here.
