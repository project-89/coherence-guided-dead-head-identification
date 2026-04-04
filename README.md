---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-readme-001
version: 1.0.4
last_updated: 2026-04-02
status: draft
author: Codex
contributors: [parzival]

# DOCUMENT RELATIONSHIPS
parent_docs:
  - doc_id: coherence-guided-transformer-pruning-001
    relationship: packages
child_docs: []
related_docs:
  - doc_id: coherence-guided-transformer-pruning-plan-001
    relationship: operationalizes
  - doc_id: repository-guidelines-001
    relationship: informed_by

# CONTENT CLASSIFICATION
domain: publication
sub_domain: transformer_pruning
keywords: [publication, latex, pruning, coherence, reproducibility]

# SYNCHRONIZATION
last_sync: 2026-04-02
sync_notes: Clarified that the package proves a dead-head identification law, not a downstream pruning operator, while retaining the exact artifact-threshold workflow.
---

# Coherence-Guided Dead-Head Identification

This repository is the self-contained publication package for the five-model
dead-head identification paper. It includes the manuscript source, compiled PDF,
figures, frozen result artifacts, and the scripts used to verify and regenerate
the bundled evidence.

## Package Contents

- `paper.tex`: formal manuscript source
- `paper.pdf`: compiled manuscript
- `paper.md`: markdown narrative version of the manuscript
- `references.bib`: bibliography used by the TeX manuscript
- `LICENSE`: repository license and required notice
- `NOTICE`: copyright notice required with redistributed copies
- `data/`: frozen JSON result artifacts used for the figures and tables
- `figures/`: publication figures
- `scripts/`: reproduction and rerun scripts
- `AGENTS.md`: package-local guidance for agents working inside this folder
- `supporting/`: two small supplementary artifacts cited in the manuscript

## Build

From this folder:

```bash
make
```

To regenerate the bundled figures from the frozen local JSON artifacts:

```bash
make figures
```

To verify the bundled threshold-transfer table:

```bash
make verify
```

## Reproduction Modes

There are two supported reproduction modes.

### 1. Artifact-Level Reproduction

This package already includes the frozen JSON artifacts in `data/`. Running:

```bash
python3 scripts/98_result_plots.py
python3 scripts/98_gqa_level2_plot.py
python3 scripts/98_threshold_evidence_plots.py
python3 scripts/98_verify_threshold_bundle.py
```

rebuilds the publication figures directly from those bundled artifacts.
The verification script also writes `data/threshold_transfer_summary.json`, a
machine-readable summary of the five bundled threshold-transfer rows.

### 2. Full Model Reruns

The package also includes the executed experiment scripts:

- `scripts/98_coherence_pruning_harness.py`
- `scripts/98_verify_threshold_bundle.py`
- `scripts/98_structural_timing_benchmark.py`
- `scripts/98_smollm2_kv_group_compaction.py`
- `scripts/98_threshold_evidence_plots.py`

These rerun the actual model-side experiments. They require:

- Python environment with `torch`, `transformers`, `numpy`, `matplotlib`, and
  `sentencepiece`
- access to the relevant Hugging Face model checkpoints
- a calibration/evaluation corpus

The publication copies of the scripts are local-first:

- result JSON defaults point into this folder's `data/`
- figure outputs point into this folder's `figures/`
- corpus paths fall back to the main repo's TinyStories path if no local excerpt is
  supplied

If you want full reruns without relying on the repo-level TinyStories path, pass
explicit `--train-path` and `--eval-path` arguments.

## Result Scope

This package supports a narrow primary claim:

- a derived dead-head observable
- a zero-parameter dimension-scaled threshold `tau_death(d) = chi_c / sqrt(d_model)` with `chi_c approx 0.96`
- high-precision transfer of that dead-head identification criterion across five bundled decoder checkpoints
- validation against individual head ablation ground truth

The core claim is about identification, not removal. Downstream pruning operators,
including rotation-aware or SVD-compensated removal procedures, are separate
engineering layers and are not required to validate the geometric law in this
package.

The rounded analytic shorthand in the prose is `0.96 / sqrt(d_model)`, but the
publication figures and verification summary use the exact threshold stored in each
frozen JSON artifact. In the current bundle those normalized thresholds span
`chi_c = 0.96000` to `0.96025`.

## License

Copyright `Imaginal Media Inc.`

This repository is released under the PolyForm Noncommercial License 1.0.0.
Non-commercial use is permitted under the terms in [LICENSE](LICENSE). The
required copyright notice is in [NOTICE](NOTICE). This repository is
source-available, not an OSI open-source license.

It also contains secondary removal/compaction analyses, but those are not the core
scientific claim of the paper. In particular, this bundle does **not** claim that
simultaneously removing all identified dead heads is lossless, that naive deletion
is the correct downstream operator, or that a universal removal protocol is
already solved.

Included secondary analyses:

- five-model transfer rows:
  - GPT-2
  - GPT-2 Medium
  - Qwen2.5 0.5B
  - SmolLM2 360M
  - OpenLLaMA 7B
- exploratory structural timing where real compaction exists
- grouped-query Level 2 compaction on SmolLM2

Supplementary scan-only context, not part of the ablation-validated core claim:

- later CLR pipeline work on REAP-25B MoE also surfaced `407` dead heads
  (`26.5%` of heads), which is directionally consistent with the same dead-head
  observable at larger scale

It does not claim to close the generalized BKT-on-`S^(d-1)` theory or the
CLR-during-training program. Those are follow-on directions rather than part of
the validated result surface in this repository.
