---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-figures-readme-001
version: 1.0.0
last_updated: 2026-03-21
status: draft
author: Codex
contributors: [parzival]

# DOCUMENT RELATIONSHIPS
parent_docs:
  - doc_id: coherence-guided-transformer-pruning-readme-001
    relationship: details
child_docs: []
related_docs:
  - doc_id: coherence-guided-transformer-pruning-data-readme-001
    relationship: driven_by

# CONTENT CLASSIFICATION
domain: publication
sub_domain: figures
keywords: [figures, plots, pruning, reproducibility]

# SYNCHRONIZATION
last_sync: 2026-03-21
sync_notes: Added figure-folder description for the pruning publication package.
---

# Figures Folder

This folder contains the publication figures:

- `exp98_five_model_summary.png`
- `exp98_gqa_level2_summary.png`
- `exp98_head_threshold_evidence.png`
- `exp98_normalized_coupling_collapse.png`

Regenerate them with:

```bash
python3 ../scripts/98_result_plots.py
python3 ../scripts/98_gqa_level2_plot.py
python3 ../scripts/98_threshold_evidence_plots.py
```

Both scripts read the frozen JSON artifacts from `../data/`.
