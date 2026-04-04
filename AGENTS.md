---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-agents-001
version: 1.0.0
last_updated: 2026-03-21
status: draft
author: Codex
contributors: [parzival]

# DOCUMENT RELATIONSHIPS
parent_docs:
  - doc_id: coherence-guided-transformer-pruning-readme-001
    relationship: supplements
child_docs: []
related_docs:
  - doc_id: coherence-guided-transformer-pruning-001
    relationship: governs
  - doc_id: coherence-guided-transformer-pruning-plan-001
    relationship: tracks

# CONTENT CLASSIFICATION
domain: publication
sub_domain: agent_guidance
keywords: [agents, publication, pruning, coherence, latex]

# SYNCHRONIZATION
last_sync: 2026-03-21
sync_notes: Added package-local agent briefing for the formal pruning publication bundle.
---

# Agent Guide

This folder is a publication package, not a sandbox.

## First Files to Read

1. `README.md`
2. `paper.tex`
3. `paper.md`

If you need to validate figures or tables, then inspect `data/`, `figures/`, and
`scripts/`.

## What This Package Claims

The paper's validated result surface is narrow and should stay narrow:

- the head-level observable is incoming write-back alignment with the pre-head
  residual stream
- the death threshold is `tau_death(d) = 0.96 / sqrt(d)`
- the threshold transfers across five completed model pilots
- structural speedups are claimed only where structural compaction was actually
  implemented and measured

Do not silently widen the claim to:

- a full generalized BKT theorem on `S^(d-1)`
- theorem closure that transformers are exactly vector Kuramoto systems
- CLR-during-training as an already-validated result

Those are legitimate follow-on programs, but not this paper.

## Local Source of Truth

- `paper.tex` is the formal manuscript source
- `references.bib` is the bibliography source
- `data/` contains the frozen JSON artifacts used to reproduce figures and tables
- `figures/` contains the publication figures
- `scripts/` contains the exact scripts used for artifact regeneration and model-side
  reruns
- `supporting/` contains two small supplementary artifacts referenced by the paper

## Editing Rules for This Folder

- keep paths local to this publication package whenever possible
- prefer reading bundled JSONs in `data/` over reaching back into `out/`
- if you change a figure script, regenerate the figure in `figures/`
- if you change the manuscript, recompile `paper.pdf`
- if you change the result surface, update `README.md` and `INDEX.md` as well

## Theory Hygiene

The physical derivation that belongs here is:

1. CLR bond death threshold `0.679` on `S^1`
2. normalize by the natural `S^1` fluctuation scale `1 / sqrt(2)` to get `0.96`
3. rescale by concentration of measure on `S^(d-1)` to get `0.96 / sqrt(d)`

That is the core derivation for this paper.

If you want to develop the generalized BKT critical-point program, put it in
`papers/current/lattice_theory/`, not here.
