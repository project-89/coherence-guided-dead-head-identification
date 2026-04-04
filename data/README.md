---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-data-readme-001
version: 1.0.2
last_updated: 2026-04-02
status: draft
author: Codex
contributors: [parzival]

# DOCUMENT RELATIONSHIPS
parent_docs:
  - doc_id: coherence-guided-transformer-pruning-readme-001
    relationship: details
child_docs: []
related_docs:
  - doc_id: coherence-guided-transformer-pruning-001
    relationship: supports

# CONTENT CLASSIFICATION
domain: publication
sub_domain: data
keywords: [artifacts, json, results, pruning]

# SYNCHRONIZATION
last_sync: 2026-04-02
sync_notes: Documented the exact bundled threshold summary and removed review-only smoke outputs from the publication package.
---

# Data Folder

This folder contains the frozen JSON artifacts used by the publication figures and
result tables.

Included artifacts:

- `threshold_transfer_summary.json`
- `full_gpt2_head144_small_redundancy_v2_dimaware.json`
- `full_gpt2_medium_head384_small_theory_redundancy_v1.json`
- `gpt2_structural_timing_v1.json`
- `gpt2_medium_structural_timing_v1.json`
- `qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json`
- `qwen25_05b_timing_proxy_v1.json`
- `smollm2_360m_head480_small_theory_redundancy_v1.json`
- `smollm2_360m_timing_proxy_v1.json`
- `smollm2_level2_kv_compaction_v1.json`
- `open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json`
- `open_llama_7b_level2_compaction_v1.json`

These are the publication-side frozen artifacts. The plotting scripts in
`../scripts/` read from this folder directly.

`threshold_transfer_summary.json` is produced by
`../scripts/98_verify_threshold_bundle.py` and records the supported
five-checkpoint transfer table for the dimension-scaled
`tau_death(d) = chi_c / sqrt(d_model)` claim. It includes the exact
`tau_artifact` and `chi_c_artifact` values used by the frozen publication bundle.
