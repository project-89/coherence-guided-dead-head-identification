# Coherence-Guided Dead-Head Identification

**A zero-parameter dead-head threshold derived from coupled-oscillator
criticality, validated across six model families at 95--100% precision.**

## The Result

Attention heads in a frozen transformer whose mean coupling to the residual
stream falls below

```
tau_death(d) = 0.96 / sqrt(d_model)
```

are dead: individually safe to ablate, as confirmed by per-head ablation
ground truth. The threshold is derived from physics, not fitted. It transfers
from d=768 through d=4096 without recalibration.

| Model | Family | d | Dead Heads | Total | Dead Precision |
|---|---|---|---|---|---|
| GPT-2 | GPT-2 | 768 | 42 | 144 | 95.2% |
| GPT-2 Medium | GPT-2 | 1024 | 114 | 384 | 98.2% |
| Qwen2.5 0.5B | Qwen2 | 896 | 157 | 336 | 95.5% |
| SmolLM2 360M | Llama | 960 | 234 | 480 | 99.6% |
| OpenLLaMA 7B | Llama | 4096 | 286 | 1024 | 100.0% |
| Gemma 3 4B | Gemma | 2560 | 30 | 272 | 100.0% |

## Derivation in Three Lines

1. CLR bond death on S^1: `cos(Delta theta) = 0.679`
2. Normalize by S^1 fluctuation scale `1/sqrt(2)`: `chi_c = 0.96025`
3. Transfer to S^(d-1) by concentration of measure: `tau = chi_c / sqrt(d)`

## Quick Start: Scan Any Model

```bash
python scripts/coherence_anatomy_scan.py --model gpt2
python scripts/coherence_anatomy_scan.py --model meta-llama/Llama-3.2-1B
python scripts/coherence_anatomy_scan.py --model Qwen/Qwen2.5-0.5B
```

The scan script outputs per-head coupling values, dead/alive classification,
and a visual layer-by-layer anatomy map. See `scripts/coherence_anatomy_scan.py --help`
for all options.

## Package Contents

- `paper.tex` / `paper.pdf`: manuscript
- `data/`: frozen JSON artifacts (six model validations)
- `figures/`: publication figures
- `scripts/`: reproduction scripts and standalone anatomy scanner
- `references.bib`: bibliography
- `AGENTS.md`: agent briefing for this folder

## Build

```bash
make            # compile paper.pdf
make figures    # regenerate figures from bundled JSON
```

## Reproduction

### From Bundled Artifacts (no GPU needed)

```bash
python3 scripts/98_result_plots.py
python3 scripts/98_gqa_level2_plot.py
python3 scripts/98_threshold_evidence_plots.py
python3 scripts/98_verify_threshold_bundle.py
```

### Full Model Reruns

Requires `torch`, `transformers`, `numpy`, `matplotlib`. See script headers
for corpus path arguments. The scripts default to loading calibration data
from TinyStories via Hugging Face.

## Scope

This paper establishes an **identification law**: a physics-derived boundary
that separates dead heads from coherently participating ones. Naive
simultaneous removal of all dead heads is catastrophic (the paper documents
this). Coherence-aware removal procedures are the subject of follow-on work
and a companion open-source pipeline.
