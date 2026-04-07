# Coherence-Guided Dead-Head Identification

**A zero-parameter dead-head threshold derived from coupled-oscillator
criticality, validated across six model families at 95--100% precision.**

## The Key Insight: Transformers Are Coupled Oscillator Networks

Every pruning method in the literature fits its threshold from data.
Magnitude pruning, movement pruning, SparseGPT, Wanda, FLAP --- all
require model-specific calibration to decide what to cut.

This paper takes a different approach. It starts from a geometric
observation: **LayerNorm projects token states onto a sphere.** After
normalization, each token's hidden state lives approximately on
S^(d-1), the unit sphere in d dimensions. The transformer update
becomes a discrete-time dynamical system on that sphere:

```
x --> LayerNorm(x + attention_update + mlp_update)
       ^                                        ^
       on the sphere                  back on the sphere
```

In this picture, **attention heads are inter-oscillator couplings** ---
they transport information between token states on the sphere, exactly
like coupling terms in the Kuramoto model of synchronizing oscillators.
**MLP layers are intra-oscillator modes** --- they transform a single
token's state without coupling it to others.

This is not metaphor. It is geometry. And geometry has consequences.

## The Law

In coupled-oscillator physics, there is a critical coupling below which
an oscillator bond is dead --- it contributes nothing to coherent
information flow. The Coherence Learning Rule (CLR) on a lattice gives
this critical point as cos(Delta theta) = 0.679 on the circle S^1.

To transfer this to a transformer with hidden dimension d, two steps:

1. **Normalize** by the natural fluctuation scale on S^1 (which is
   1/sqrt(2)), giving a dimensionless critical ratio chi_c = 0.96.
2. **Rescale** by concentration of measure on S^(d-1): random unit
   vectors in d dimensions have dot products of order 1/sqrt(d).

The result is a universal dead-head threshold:

```
tau_death(d) = 0.96 / sqrt(d_model)
```

No parameter is fitted. No model-specific calibration is needed. You
plug in the hidden dimension and get the threshold.

## The Evidence

We measure each head's mean cosine alignment between its incoming
write-back signal and the receiver's pre-head residual state. Heads
below the threshold are classified as dead. We validate against
individual head ablation ground truth:

| Model | Family | d | Dead Heads | Total | Dead Precision |
|---|---|---|---|---|---|
| GPT-2 | GPT-2 | 768 | 42 | 144 | 95.2% |
| GPT-2 Medium | GPT-2 | 1024 | 114 | 384 | 98.2% |
| Qwen2.5 0.5B | Qwen2 | 896 | 157 | 336 | 95.5% |
| SmolLM2 360M | Llama | 960 | 234 | 480 | 99.6% |
| OpenLLaMA 7B | Llama | 4096 | 286 | 1024 | 100.0% |
| Gemma 3 4B | Gemma | 2560 | 30 | 272 | 100.0% |

The same constant, the same formula, 95--100% precision across six
models in four architecture families, from 768 to 4096 dimensions.

## Scan Any Model

```bash
pip install torch transformers numpy datasets matplotlib

# Terminal output with layer-by-layer anatomy map
python scripts/coherence_anatomy_scan.py --model gpt2

# Full HTML report with visualizations
python scripts/coherence_anatomy_scan.py --model gpt2 --report gpt2_anatomy.html

# JSON export for programmatic use
python scripts/coherence_anatomy_scan.py --model meta-llama/Llama-3.2-1B --output llama.json

# GPU acceleration
python scripts/coherence_anatomy_scan.py --model Qwen/Qwen2.5-0.5B --device cuda
```

The scanner outputs:
- Per-head coupling values and dead/alive classification
- Visual layer-by-layer anatomy map
- Batch consistency check (heads must be consistently dead, not just on average)
- Self-contained HTML report with 5 embedded visualizations

## Important: Identification, Not Pruning

This paper establishes an **identification law** --- a physics-derived
boundary that tells you which heads are dead. It does NOT claim that
naively deleting all dead heads produces a working model. In fact,
simultaneous removal is catastrophic (the paper documents this).

Why? A head can be individually inert yet still participate in
distributed basis rotations across layers. Removing many such heads at
once perturbs the residual-stream geometry cumulatively. Coherence-aware
removal --- using SVD spectral filtering, rotation compensation, and
mean-compensated pruning --- is a separate engineering problem addressed
in follow-on work.

## Convergent Evidence

Three independent research groups have recently arrived at observations
consistent with the geometric picture developed here:

- **SpectralQuant** (Gopinath, 2026): Key projection matrices have
  effective dimensionality d_eff ~ 4 out of 128, meaning ~97% of
  spectral energy is noise --- consistent with most attention operating
  near the random-alignment scale of S^(d-1).

- **TriAttention** (Mao et al., 2026): Q/K vectors concentrate around
  fixed directions in pre-RoPE space --- a signature of partial
  phase-locking in the oscillator picture.

- **Tensor-role asymmetry** (Turney, 2026): Attention tensors tolerate
  aggressive quantization while FFN write-back projections are
  quality-critical --- exactly the inter-oscillator vs. intra-oscillator
  distinction predicted by the coupled-oscillator geometry.

## Repository Contents

```
paper.tex / paper.pdf    Manuscript (source and compiled)
references.bib           Bibliography
data/                    Frozen JSON result artifacts (6 models)
figures/                 Publication figures
scripts/
  coherence_anatomy_scan.py         Standalone scanner (start here)
  98_coherence_pruning_harness.py   Full experiment harness
  98_result_plots.py                Regenerate summary figure
  98_threshold_evidence_plots.py    Head-level threshold evidence
  98_verify_threshold_bundle.py     Verify transfer rows
  98_gqa_level2_plot.py             GQA compaction figure
supporting/              Additional validation data
```

## Reproduce

```bash
# Regenerate all figures from bundled artifacts (no GPU needed)
python3 scripts/98_result_plots.py
python3 scripts/98_threshold_evidence_plots.py
python3 scripts/98_gqa_level2_plot.py
python3 scripts/98_verify_threshold_bundle.py

# Compile the paper
make
```

## Citation

```bibtex
@article{Sharpe2026,
  author = {Michael Sharpe},
  title  = {Coherence-Guided Dead-Head Identification in Frozen Transformers:
            A Zero-Parameter Geometric Threshold from Coupled-Oscillator Criticality},
  year   = {2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
