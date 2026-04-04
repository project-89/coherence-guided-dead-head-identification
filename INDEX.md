---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-publication-index-001
version: 1.0.1
last_updated: 2026-03-21
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

# CONTENT CLASSIFICATION
domain: publication
sub_domain: packaging
keywords: [publication, pruning, transformers, coherence, reproducibility]

# SYNCHRONIZATION
last_sync: 2026-03-21
sync_notes: Updated the package index to mark dead-head identification as the primary publication claim and removal as secondary analysis.
---

# Coherence-Guided Dead-Head Identification Publication Bundle

This folder packages the final five-model dead-head identification paper as a formal publication
bundle.

## Contents

- `paper.tex`: formal manuscript source
- `paper.pdf`: compiled manuscript
- `paper.md`: markdown narrative version of the manuscript
- `README.md`: package build and reproduction guide
- `AGENTS.md`: package-local agent briefing
- `references.bib`: bibliography used by the TeX manuscript
- `data/`: local JSON mirrors used for figures and tables
- `scripts/`: exact EXP-98 scripts used to generate the results
- `figures/`: paper-ready figures
- `artifacts/`: compatibility copy of the original JSON outputs

## Included Model Results

- GPT-2
- GPT-2 Medium
- Qwen2.5 0.5B
- SmolLM2 360M
- OpenLLaMA 7B

## Claim Scope

Primary claim:

- the paper derives and validates a dead-head identification observable with
  threshold `tau_death(d) = chi_c / sqrt(d_model)` with `chi_c approx 0.96`
- validation is against individual head ablation ground truth

Secondary analyses included in the package:

- cumulative removal and redundancy rows
- structural timing and grouped-query compaction measurements

Those secondary analyses are packaged for traceability, but they are not the core
scientific claim of the publication.

## Key Figures

- `figures/exp98_five_model_summary.png`
- `figures/exp98_gqa_level2_summary.png`
- `figures/exp98_head_threshold_evidence.png`
- `figures/exp98_normalized_coupling_collapse.png`

## Reproducibility Notes

Primary scripts:

- `scripts/98_coherence_pruning_harness.py`
- `scripts/98_structural_timing_benchmark.py`
- `scripts/98_smollm2_kv_group_compaction.py`
- `scripts/98_result_plots.py`
- `scripts/98_gqa_level2_plot.py`
- `scripts/98_threshold_evidence_plots.py`

Key local data artifacts:

- `data/full_gpt2_head144_small_redundancy_v2_dimaware.json`
- `data/full_gpt2_medium_head384_small_theory_redundancy_v1.json`
- `data/qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json`
- `data/smollm2_360m_head480_small_theory_redundancy_v1.json`
- `data/smollm2_level2_kv_compaction_v1.json`
- `data/open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json`
- `data/open_llama_7b_level2_compaction_v1.json`
- `data/gpt2_structural_timing_v1.json`
- `data/gpt2_medium_structural_timing_v1.json`
- `data/qwen25_05b_timing_proxy_v1.json`
- `data/smollm2_360m_timing_proxy_v1.json`

## Final Status

The publication bundle now contains the formal manuscript, compiled PDF, local
figures, exact scripts, and local data mirrors needed to support the five-model
dead-head identification result set, including the final `d = 4096` OpenLLaMA
validation row.
