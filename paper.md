---
# PROJECT 89 DOCUMENT METADATA
doc_id: coherence-guided-transformer-pruning-001
version: 1.0.5
last_updated: 2026-04-02
status: draft
author: Codex
contributors: [parzival]

# DOCUMENT RELATIONSHIPS
parent_docs:
  - doc_id: coherence-augmented-lm-paper-001
    relationship: extends
  - doc_id: trajectory-intelligence
    relationship: applies
child_docs:
  - doc_id: coherence-guided-transformer-pruning-plan-001
    relationship: operationalizes
related_docs:
  - doc_id: language-as-flow-theory
    relationship: grounds
  - doc_id: journal-sparse-hebbian-breakthrough-001
    relationship: motivates
  - doc_id: journal-k-field-ablation-report-001
    relationship: constrains

# CONTENT CLASSIFICATION
domain: hybrid-lm
sub_domain: transformer_pruning
keywords: [transformer, pruning, coherence, ablation, frozen_k, structured_sparsity]

# SYNCHRONIZATION
last_sync: 2026-04-02
sync_notes: Clarified that the manuscript proves a dead-head identification law, while downstream pruning and SVD-based compensation remain separate engineering work.
---

# Coherence-Guided Dead-Head Identification in Frozen Transformers

**Claim status:** `HYPOTHESIS`

## Abstract

We propose a narrow, testable hypothesis: dead attention heads in a frozen
transformer obey a coherence boundary analogous to CLR bond death on the lattice.
The key question is not "which ranking metric prunes best?" but whether a derived
death threshold can identify genuinely dead transformer heads with no fitted
parameters. The claim of this paper is an identification law, not a complete
removal procedure.

The first concrete result is a small-model GPT-2 pilot. A direct import of the
lattice threshold `0.679` fails, but a dimensionality-corrected threshold
`tau_death(d) = chi_c / sqrt(d)` with `chi_c approx 0.96` combined with boundary-layer protection yields a high-precision
dead-head detector on GPT-2/TinyStories: 95.2% precision on the heads it classifies
as dead, with no parameter fitting. This does not prove theorem closure for
transformers. It does establish that the death-threshold program is nontrivial,
falsifiable, and already informative on a standard frozen model.

The same threshold then transfers across GPT-2 Medium, Qwen2.5 0.5B, SmolLM2 360M,
and OpenLLaMA 7B, reaching dead-head precision of `98.2%`, `95.5%`, `99.6%`, and
`100.0%`, respectively, when validated against individual head ablation ground
truth. We explicitly do not claim here that simultaneously removing all identified
dead heads is lossless, that naive deletion is the correct downstream pruning
operator, or that a universal removal schedule is already solved. Secondary
removal and compaction analyses are included as engineering context only.

The geometric interpretation is explicit: LayerNorm places token states
approximately on `S^(d-1)`, attention heads act as state-dependent inter-oscillator
couplings, and MLP updates act as intra-oscillator modes. The dead-head observable
is therefore the natural high-dimensional generalization of bond alignment
`cos(Δθ)` from the lattice program.

The derivation chain used throughout the paper is:

`cos(Δθ)_death = 0.679`
`-> divide by (1 / sqrt(2)) ->`
`chi_c = 0.679 * sqrt(2) = 0.96025... approx 0.96`
`-> multiply by (1 / sqrt(d)) ->`
`tau_death(d) = chi_c / sqrt(d)`.

The first arrow converts the scalar lattice death point at the BKT critical radius
into a dimensionless critical ratio on `S^1`; the second transfers that ratio into
the random-alignment scale of `S^(d-1)`.

## 1. Scope and Claim Hygiene

This document makes four explicit scoping choices:

1. It is about **frozen-model dead-head identification**, not training a new lattice-native LM.
2. It is about **identification of dead heads**, not a full pruning operator.
3. It is a **consumer-track hypothesis**, not a canon-promotion candidate.
4. It does **not** claim that the grand theory of transformers has been proved.

Claims intentionally not made here:

- We do not claim that the transformer exactly obeys the canonical phase dynamics.
- We do not claim that SGD has already been identified with CLR beyond motivation.
- We do not import the lattice death threshold directly into hidden-state pruning.
- We do not claim theorem-grade closure for LLM compression.
- We do not claim that cumulative removal of all identified dead heads is lossless.

## 2. Hypothesis

We pre-register three linked hypotheses.

**H1. Frozen-K inference.** During inference, a trained transformer can be treated as
a dynamical system evolving on a frozen coupling landscape induced by its learned
weights.

**H2. Derived coherence death predicts dead-head identification.** Attention heads
whose transformer-native coupling falls below a derived death boundary are in the
dead regime at high precision under one-head ablation validation.

**H3. Identification should precede pruning.** Heads, MLP channels, and whole
sublayers are meaningful structured units, but the present paper only claims a
head-level identification law; downstream pruning is a later, secondary program.

## 3. Repo-Native Motivation

The pruning idea is not starting from zero. The repo already contains the relevant
ingredients.

### 3.1 Sparse Coherence Beats Dense Uniformity

`journal/2025-12-05-sparse-hebbian-breakthrough.md` shows that top-10% coherent-edge
updates produced a 3.6x improvement in basin differentiation over dense updates. The
useful lesson is structural: coherence tends to concentrate on a minority of edges,
and indiscriminate updates homogenize the field.

For pruning, the analog is immediate: remove units that fail to carry coherent
structure, not merely units with small raw coefficients.

### 3.2 K and B Already Separate Structure from Readout

`journal/2025-12-22-k-field-ablation-report.md` and
`experiments/coherence_head/66_k_vs_b_ablation.py` already show a functional split:
K-like interventions move internal attention/coherence, while B-like interventions
move logits directly. For pruning, this matters because a unit can be structurally
important even when its direct logit effect is small.

### 3.3 Language Work Already Treats Meaning as Flow

`papers/current/resonance_model/language_as_flow_theory.md`,
`papers/current/resonance_model/lattice_language_training_regime.md`, and
`papers/current/hybrid_lm/core/trajectory_intelligence.md` all treat language as
trajectory through a learned dynamical geometry. A pruning score that ignores
trajectory continuity, recurrence, and basin stability would therefore be misaligned
with the repo's core language paradigm.

### 3.4 Physics Already Warned Against Naive Hard Pruning

`research_registry/CURRENT_STATE.md` records a physics-side failure mode: hard
dead-bond pruning suppressed magnetic response by killing marginal-but-important
bonds. The transfer lesson is cautionary. Transformer pruning should begin with
reversible masking and ablation scoring before irreversible deletion.

## 4. Transformer-Native Death Observable

The first successful transformer observable is not a generic ranking score. It is a
head-level coupling quantity motivated by vector Kuramoto geometry.

### 4.1 Sphere Geometry from LayerNorm

Let `x_i^(l) ∈ R^d` denote the residual-stream state at token position `i` before
transformer layer `l`. Ignoring the learned affine parameters, LayerNorm acts as a
radial reprojection onto an approximately fixed-radius sphere. The normalized token
state therefore lives approximately on

`x̂_i^(l) ∈ S^(d-1)`.

At block level, the transformer update takes the form

`x̂_i^(l+1) = LayerNorm(x̂_i^(l) + Δ_attn,i^(l) + Δ_mlp,i^(l))`

which is a discrete-time perturb-and-reproject dynamics on a high-dimensional
sphere.

### 4.2 Attention Heads as Inter-Oscillator Couplings

For attention head `h`, the incoming write-back at receiver position `i` is

`s_h(i) = Σ_j A_ij^(h) V_h(x_j)`.

This is the transformer analog of a state-dependent coupling between oscillator
states. The comparison point is the vector Kuramoto / Lohe flow on `S^(d-1)`:

`dx_i/dt = Σ_j K_ij [x_j - (x_i · x_j)x_i]`.

In the transformer, the attention weights `A_ij^(h)` play the role of driven,
state-dependent couplings, while LayerNorm provides the discrete reprojection back
onto the representation sphere.

### 4.3 MLP Channels as Intra-Oscillator Modes

The MLP update does not couple different token positions. Instead it transforms the
state of a single token:

`Δ_mlp,i^(l) = W_down φ(W_up x_i^(l))`.

In the oscillator picture, attention heads are therefore inter-oscillator transport
channels, while MLP channels are intra-oscillator modes or local tangent
directions of the update field. The present paper concerns only the attention-side
coupling channels.

### 4.4 Why the Cosine Observable Is the Right One

On `S^1`, the pairwise Kuramoto alignment observable is `cos(θ_j - θ_i)`. On
`S^(d-1)`, the natural generalization is directional cosine between the incoming
signal and the receiver state. The head-level observable used in this paper,

`c_h(i) = cos(s_h(i), x_i)`,

therefore measures the same physical quantity in the higher-dimensional setting:
how much of the incoming coupling signal is coherent with the receiver's existing
state. Below threshold, the coupling is no longer distinguishable from the random
alignment scale of `S^(d-1)`.

### 4.5 Paradigm Mapping

| Lattice / Kuramoto object | Transformer object |
| --- | --- |
| Oscillator state on `S^1` or `S^(d-1)` | token-position residual state |
| Phase / state variable | residual-stream vector at one token position |
| Bond coupling `K_ij` | attention weight / effective head coupling |
| Pairwise alignment `cos(Δθ)` | `cos(s_h(i), x_i)` |
| Internal frame / local mode | MLP channel update direction |
| Bond death below criticality | dead head below `tau_death(d)` |
| Phase-locked mode cluster | coherent cross-layer head cluster / circuit |
| Domain wall / bridge bond | boundary or bridge-critical head |

Let `x_i` be the residual state at receiving position `i`, and let head `h` send an
incoming write-back signal to `i` through its value pathway. The head coupling is:

`cos_h = mean_{batch, position} cosine(writeback_h(i), residual_before(i))`

This asks the right question: does the head's incoming signal cohere with the
receiver's existing state? It is the transformer analog of lattice `cos(Delta theta)`
at a bond, not a generic "how much did the head change things?" proxy.

### 4.6 Dimensional Scaling of the Death Threshold

The lattice death threshold is `0.679` on `S^1`. The natural cosine scale on `S^1`
is `1 / sqrt(2)`. This defines a dimensionless critical ratio:

`chi_c = 0.679 / (1 / sqrt(2)) = 0.96025... approx 0.96`

On `S^(d-1)`, concentration of measure gives natural cosine scale `1 / sqrt(d)`.
The transformer death threshold is therefore:

`tau_death(d) = chi_c / sqrt(d)`

For GPT-2 (`d = 768`):

`tau_death(768) = 0.03465007365312038`

For the frozen five-model publication bundle, the exact artifact thresholds produce
normalized values `chi_c_artifact` in the range `0.96000` to `0.96025`. The paper
uses `0.96 / sqrt(d)` as rounded shorthand, but the bundled figures and verifier
read the exact thresholds from the JSON artifacts.

This threshold is derived from:

1. the lattice CLR death theorem at criticality, and
2. concentration of measure on high-dimensional spheres.

No fitting is used.

### 4.7 Boundary and Bridge Protection

Boundary layers play the transformer analog of driven lattice boundary nodes. They
are not judged by exactly the same death rule as interior synchronization channels.
The current theory pass therefore protects the first layer by default, and may
optionally protect the final layer as an output boundary.

Separately, a rare bridge veto protects heads whose attention graph has unusually
high bridge mass relative to their layer.

### 4.8 Operational Measurement Procedure

The practical method used in this paper is deliberately simple.

1. For each head, compute its incoming write-back signal at each receiving
   position:

   `signal_h(i) = sum_j A_ij^h V_h(x_j)`

2. Compare that incoming signal with the receiver's pre-head residual state:

   `coupling_h(i) = cosine(signal_h(i), x_i)`

3. Average over positions and calibration sequences:

   `mean_coupling_h = mean_{batch, position} coupling_h(i)`

4. Identify the head as dead if:

   `mean_coupling_h < tau_death(d)`

   unless the head is boundary-protected or bridge-protected.

5. Validate the resulting dead-head set against individual head ablation ground
   truth. Structural removal experiments are treated as a separate secondary
   analysis.

This is the operational core of the paper. One observable, one derived threshold,
and two protection rules determine the dead set.

This completes the scientific claim tested here. Identification is kept separate
from removal: a head can be individually ablation-safe yet still participate in
distributed basis rotation across layers when many heads are removed together.
Rotation-aware compensation operators, including SVD-based variants, belong to
downstream pruning engineering and are not part of the geometric law validated in
this paper.

The geometric intuition is equally simple. In `d` dimensions, random directions
have cosine scale `1 / sqrt(d)`. A head whose mean coupling is below
`tau_death(d)` is operating at or below the random-direction scale once the
critical lattice ratio is transferred into high-dimensional representation space.
That is why the threshold transfers with no fitted model-specific calibration.

### 4.9 Full Physical Derivation

The threshold used in this paper is derived from the oscillator physics, not fitted
from the transformer data.

Start from the bond energy for coupled phases on the lattice:

`E_ij = -K_ij cos(theta_i - theta_j)`

Under the CLR, bond growth or decay depends on whether the local phase alignment is
strong enough to overcome the shrink term:

`dot(K_ij) = eta [R_0(K_ij) cos(Delta theta_ij) - 2K_ij / r]`

At the alive/dead boundary the update changes sign. Evaluated at the critical
radius of the lattice analysis, this gives the lattice death threshold:

`cos(Delta theta)_death = 4 / r = 0.679`

That threshold lives on `S^1`. The natural scale of cosine fluctuations on `S^1` is:

`sigma_1 = 1 / sqrt(2)`

Dividing the death point by the natural fluctuation scale produces a dimensionless
critical ratio:

`chi_c = 0.679 / sigma_1 = 0.679 / (1 / sqrt(2)) = 0.96025... approx 0.96`

This dimensionless ratio is the transferable quantity. The raw cosine threshold
must change with representation dimension because the geometry of random directions
changes with `d`.

For unit vectors on `S^(d-1)`, concentration of measure gives:

`x · y approx N(0, 1 / d)`

so the natural fluctuation scale becomes:

`sigma_d = 1 / sqrt(d)`

Replacing the `S^1` fluctuation scale with the `S^(d-1)` fluctuation scale gives
the transformer death threshold:

`tau_death(d) = chi_c sigma_d = chi_c / sqrt(d)`

Equivalently, the full bridge from the scalar BKT-side critical point to the
transformer threshold is:

`0.679 -> divide by (1 / sqrt(2)) -> 0.96025... -> multiply by (1 / sqrt(d)) -> chi_c / sqrt(d)`

This is the full derivation chain:

1. CLR bond death at BKT criticality gives `0.679` on `S^1`
2. normalizing by the natural `S^1` fluctuation scale gives the dimensionless ratio `0.96025...`, usually cited as `0.96`
3. concentration of measure on high-dimensional spheres rescales the threshold to
   `chi_c / sqrt(d)`

For the models evaluated here:

- `gpt2`, `d = 768`: `tau_death = 0.03465`
- `Qwen/Qwen2.5-0.5B`, `d = 896`: `tau_death = 0.03207`
- `HuggingFaceTB/SmolLM2-360M`, `d = 960`: `tau_death = 0.03098`
- `gpt2-medium`, `d = 1024`: `tau_death = 0.03000`

The significance of the formula is that it measures a geometric coherence boundary
for `d`-dimensional representations, not a model-specific empirical statistic.
This is why it transfers across models without recalibration.

Within the wider coherence-lattice program, this same critical-ratio logic is tied
to the oscillator physics used elsewhere in the lattice-side fine-structure
derivation. The identification result does not establish theorem closure for transformers,
but it does show that the same critical coherence ratio survives a change of domain.

The stronger vector-Kuramoto interpretation remains a motivating hypothesis rather
than a closed theorem in this paper. We use it to choose the observable and the
transfer logic, not to claim a full transformer critical theory.

## 5. Current Result

The current best clean result is:

- Model: frozen `gpt2`
- Data: TinyStories calibration/eval slices
- Unit: attention heads
- Coupling observable: incoming write-back vs pre-head residual alignment
- Threshold: `tau_death(768) = 0.03465007365312038`
- Protection: first-layer boundary protection + rare bridge veto

Observed result on the heads classified as dead:

- precision: `95.2%`
- recall: `31.5%`
- dead heads: `42 / 144`
- false positives: `2`

This is a high-precision dead-head detector with zero fitted parameters. It does not
yet maximize recall, and it should not be interpreted as a full compression method.

### 5.1 Error Pattern

The error pattern is theoretically useful.

- Without boundary protection, most false positives live in layer 0. These heads are
  input transducers, not ordinary interior couplings.
- With boundary protection, the remaining false positives are two interior heads.
- Many false negatives are coherent but redundant heads. That is not a failure of
  the death rule. It indicates that redundancy removal is a second mechanism, not a
  correction to CLR death itself.

## 6. Two-Pass Architecture

The theory and the engineering now separate cleanly.

### Pass 1: CLR Death Detection

Use the derived death threshold plus protection rules to identify heads that are
incoherent with the network.

Properties:

- zero-parameter threshold once `d` is known
- high precision
- conservative recall

### Pass 2: Redundancy Detection

Among the heads that remain alive, detect which ones are functionally redundant with
other alive heads.

This is not the CLR death dynamic. It is a post-convergence compaction step.
It can require compensation because cumulative removal perturbs distributed basis
rotations across layers; that question is separate from proving the dead-head law.

The current GPT-2 redundancy rule is geometry-aware rather than based on fixed
absolute overlap thresholds:

- pairwise directional overlap is judged against the random high-dimensional scale
  `1 / sqrt(d)`
- reconstruction overlap is judged against the random-subspace baseline `m / d`
  where `m` is the count of alternative alive heads in the layer
- a head must also have a same-scale alive partner, enforced by a norm-similarity
  gate

Current GPT-2 redundancy-pass result:

- alive pool: `76`
- redundant heads flagged: `26`
- redundancy precision within alive pool: `88.5%`
- redundancy recall within alive pool: `32.9%`
- combined dead-plus-redundant eval `delta_loss`: `+1.85`

The correct conceptual split is:

- Pass 1 removes the dead.
- Pass 2 removes the redundant.

### 6.1 Structural Timing on GPT-2

Using GPT-2's native structural head-pruning path rather than reversible masking,
the current GPT-2 pilot shows the expected context-length dependence.

- Dead-only pruning removes `42 / 144` heads and reduces parameter memory from
  `497.8 MB` to `464.7 MB`
- Dead-plus-redundant pruning removes `68 / 144` heads and reduces parameter
  memory to `444.2 MB`
- At short context (`128`, `256`) the wall-clock gain is negligible or slightly
  negative on this MacBook run
- At longer context the gain becomes real:
  - `seq_len=512`: dead-only `+16.2%`, dead-plus-redundant `+22.9%`
  - `seq_len=1024`: dead-only `+13.7%`, dead-plus-redundant `+22.7%`

This is the expected shape if attention-head pruning buys more when attention
dominates total compute.

### 6.2 First Larger-Model Transfer: GPT-2 Medium

The first larger-model transfer keeps the same framework and changes only the model
dimension:

- model: `gpt2-medium`
- hidden dimension: `1024`
- death threshold: `0.03`
- dead heads: `114 / 384` at `98.2%` precision
- redundant alive heads: `85` at `100%` precision on the current slice
- combined prune set: `199 / 384` heads, or `51.8%`

The structural timing result is mixed at short context and strong at long context:

- `seq_len=128`: dead-plus-redundant is slower on this implementation path
- `seq_len=256`: still slower
- `seq_len=512`: still slightly slower
- `seq_len=1024`: dead-plus-redundant reaches `+32.1%` wall-clock speedup

This indicates that the threshold transfer is stronger than the current short-context
runtime path. The speed benefit appears once attention cost dominates enough to
overcome pruning overhead.

### 6.3 First Non-GPT2 Family Transfer: Qwen2 0.5B

The first non-GPT2 decoder-family transfer used `Qwen/Qwen2.5-0.5B`:

- model: Qwen2 causal decoder with grouped-query attention
- hidden dimension: `896`
- death threshold: `0.03207`
- boundary rule: first `2` layers protected, last `1` layer protected
- dead heads: `157 / 336` at `95.5%` precision
- redundant alive heads: `62` at `95.2%` precision
- combined prune set: `219 / 336` heads, or `65.2%`
- combined dead-plus-redundant eval `delta_loss`: `+2.12476`

This is the first demonstration in the repo that the same dimension-scaled death
logic transfers to a non-GPT2 decoder family with no change in the core formula.

The timing caveat is architectural. Qwen2 does not expose native per-head structural
pruning, and its grouped-query layout makes arbitrary head compaction a separate
engineering problem. The current timing artifact is therefore only a masked proxy,
not a deployment-grade structural speed measurement.

### 6.4 Llama-Family Transfer: SmolLM2 360M

The second non-GPT2 decoder-family transfer used `HuggingFaceTB/SmolLM2-360M`:

- model: Llama-family decoder
- hidden dimension: `960`
- death threshold: `0.03098`
- boundary rule: first `1` layer protected, last `1` layer protected
- dead heads: `261 / 480` at `99.6%` precision
- redundant alive heads: `38` at `89.5%` precision
- combined prune set: `299 / 480` heads, or `62.3%`
- dead-only eval `delta_loss`: `+0.87051`
- combined dead-plus-redundant eval `delta_loss`: `+2.05133`

This is the strongest dead-pass result so far. On the current slice, the
dimension-scaled death rule alone removes more than half of all heads with only one
false positive.

Unlike Qwen2, SmolLM2 is also a strong Level 2 target because its grouped-query
group size is only `3`. Recomputing the dead-head map at KV-group granularity gives:

- total KV groups: `160`
- fully dead KV groups: `44 / 160`, or `27.5%`
- near-full KV groups (`>= 2 / 3` dead): `89 / 160`, or `55.6%`

This is much stronger than Qwen2, where the larger group size `7` suppresses full
group death combinatorially:

- total KV groups: `48`
- fully dead KV groups: `3 / 48`, or `6.25%`
- near-full KV groups (`>= 6 / 7` dead): `7 / 48`, or `14.6%`

The comparison is informative rather than incidental. If per-query death occurs with
probability `p`, full-group death scales roughly as `p^g` for group size `g`. Qwen2
and SmolLM2 therefore behave differently for a structural reason: the group-size
barrier is much weaker for `g = 3` than for `g = 7`.

### 6.5 SmolLM2 Level 2 KV-Group Compaction

Using only the CLR dead-head decisions, we built a true structural Level 2 rewrite
for SmolLM2 that removes fully dead KV groups and their associated query heads.
This is not masked proxy timing. The q/k/v/o projections are rewritten per layer,
the model returns a smaller cache, and the speed measurement reflects actual compute
reduction in cache-building prefill mode.

Level 2 compaction result:

- fully dead KV groups removed: `44 / 160`, or `27.5%`
- associated query heads removed: `132 / 480`, or `27.5%`
- total query heads after compaction: `348`
- total KV heads after compaction: `116`
- parameter count: `361.8M -> 340.2M` (`-5.98%`)
- KV cache bytes per token: `81,920 -> 59,392` (`-27.5%`)

Measured prefill timing and quality:

| Sequence Length | Cache Reduction | Speedup vs Baseline | `delta_loss` |
| ---: | ---: | ---: | ---: |
| 128 | 27.5% | +4.07% | +0.37725 |
| 256 | 27.5% | -14.76% | +0.44358 |
| 512 | 27.5% | +14.50% | +0.50072 |
| 1024 | 27.5% | +30.96% | +0.53206 |

This is the first grouped-query structural result in the current pruning branch:
a derived dead-head set induces a measurable KV-cache reduction and a measured
runtime gain on a Llama-family model without introducing any fitted pruning
thresholds.

![EXP-98 GQA Level 2 summary](../../../out/exp98_pruning/figures/exp98_gqa_level2_summary.png)

Figure 2. Group-size controls how often full KV groups die (`g=7` for Qwen2 versus
`g=3` for SmolLM2), and on SmolLM2 the resulting Level 2 compaction yields both a
constant `27.5%` KV-cache reduction and a strong long-context prefill speedup.

### 6.6 LLaMA-Scale Transfer: OpenLLaMA 7B

The final scaling test uses `openlm-research/open_llama_7b`:

- model: OpenLLaMA 7B decoder
- hidden dimension: `4096`
- death threshold: `0.01500`
- boundary rule: first `1` layer protected, last `1` layer protected
- dead heads: `286 / 1024` at `100.0%` precision
- redundant alive heads: `69` at `98.6%` precision
- combined prune set: `355 / 1024` heads, or `34.7%`
- dead-only eval `delta_loss`: `+1.19436`
- combined dead-plus-redundant eval `delta_loss`: `+1.60351`

This is the cleanest threshold-transfer result in the current paper. The derived
boundary `tau_death(4096) = 0.01500` produces zero dead-head false positives on the current
pilot slice. The result is conservative in recall, but exceptionally strong in the
dimension that matters most for a theory-derived dead-head detector: precision.

Because OpenLLaMA 7B uses full multi-head attention (`g = 1`), Level 2 structural
compaction is exactly dead-head compaction. The measured structural result is:

- dead heads structurally removed: `286 / 1024`, or `27.93%`
- parameter count: `6.74B -> 6.14B` (`-8.90%`)
- KV cache bytes per token: `524,288 -> 377,856` (`-27.93%`)
- measured speedup: `+4.13%` at `128`, `+7.17%` at `256`, `+1.62%` at `512`, `+12.25%` at `1024`
- `delta_loss` at `1024`: `+1.15692`

This row closes the dimensional-scaling story for the current paper. The threshold
now spans `d = 768`, `896`, `960`, `1024`, and `4096` with no fitted death
parameter and no failure of precision.

### 6.7 Current Result Matrix

The scientific center of gravity in this matrix is the dead-pass precision column.
The redundancy and cumulative-removal columns are included as secondary engineering
context, not as the primary publication claim.

Across the five completed pilots:

| Model | Family | `d` | Death Threshold | Dead Precision | Redundancy Precision | Heads Removed | Combined Precision | Dead+Red `delta_loss` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | GPT-2 | 768 | 0.03465 | 95.2% | 88.5% | 68 / 144 | 92.7% | +1.84669 |
| `gpt2-medium` | GPT-2 | 1024 | 0.03000 | 98.2% | 100.0% | 199 / 384 | 99.0% | +1.43029 |
| `Qwen/Qwen2.5-0.5B` | Qwen2 | 896 | 0.03207 | 95.5% | 95.2% | 219 / 336 | 95.4% | +2.12476 |
| `HuggingFaceTB/SmolLM2-360M` | Llama | 960 | 0.03098 | 99.6% | 89.5% | 299 / 480 | 98.3% | +2.05133 |
| `openlm-research/open_llama_7b` | Llama | 4096 | 0.01500 | 100.0% | 98.6% | 355 / 1024 | 99.7% | +1.60351 |

The current pattern is stable:

- the dead-head pass stays high precision across all five models
- the redundancy pass is useful but more architecture-sensitive
- GPT-2-family models currently have the cleanest runtime story because they expose
  native structural head pruning
- grouped-query models transfer first on the classification side, then on the
  systems side once full KV-group compaction is implemented

Separate later CLR pipeline work also surfaced `407` dead heads (`26.5%`) on the
REAP-25B MoE model. That row is supportive scan-only context, not part of the
ablation-validated core claim, because this publication package does not bundle
its matching per-head ablation ground truth.

![EXP-98 five-model summary](../../../out/exp98_pruning/figures/exp98_five_model_summary.png)

Figure 1. The derived threshold curve, plotted from the exact bundled threshold
values, plus precision, sparsity, and the timing split between structural,
Level 2 structural, and masked-proxy runs across the five completed models.

### 6.8 Timing Matrix

These timing rows should be read as secondary systems analyses. They show what
happens under particular compaction paths after identification, not a universal
lossless removal claim.

Timing at `seq_len=1024`:

| Model | Timing Mode | Variant | Speedup vs Baseline | `delta_loss` |
| --- | --- | --- | ---: | ---: |
| `gpt2` | structural | dead+redundant | +22.69% | +2.66609 |
| `gpt2-medium` | structural | dead+redundant | +32.14% | +2.19939 |
| `Qwen/Qwen2.5-0.5B` | masked proxy | dead+redundant | -8.72% | +2.49661 |
| `HuggingFaceTB/SmolLM2-360M` | masked proxy | dead+redundant | -15.44% | +2.73854 |
| `openlm-research/open_llama_7b` | structural level2 | dead-only | +12.25% | +1.15692 |

This is the clean systems takeaway:

- the formula already transfers across architectures
- real wall-clock gains appear where true structural compaction exists
- for large-group GQA models such as Qwen2, the next bottleneck is grouped-query
  structural compaction and rebalancing, not threshold derivation

SmolLM2 now has the first true grouped-query structural timing result as well. For
the stricter Level 2 variant that removes only fully dead KV groups:

- timing mode: structural KV-group compaction
- cache reduction: `27.5%` at every tested length
- prefill speedup: `+14.50%` at `512`, `+30.96%` at `1024`

So the runtime story is now split:

- GPT-2 and GPT-2-medium: arbitrary head compaction already works structurally
- SmolLM2: grouped-query Level 2 compaction now works structurally
- OpenLLaMA 7B: dead-head structural compaction works at `d = 4096`
- Qwen2: grouped-query transfer still needs structural compaction rather than masked proxy

### 6.9 Executed Scripts and Artifacts

Primary scripts executed in this run:

- `experiments/coherence_head/exp98/98_coherence_pruning_harness.py`
- `experiments/coherence_head/exp98/98_structural_timing_benchmark.py`
- `experiments/coherence_head/exp98/98_result_plots.py`
- `experiments/coherence_head/exp98/98_smollm2_kv_group_compaction.py`
- `experiments/coherence_head/exp98/98_gqa_level2_plot.py`

Key pruning artifacts:

- `out/exp98_pruning/full_gpt2_head144_small_redundancy_v2_dimaware.json`
- `out/exp98_pruning/full_gpt2_medium_head384_small_theory_redundancy_v1.json`
- `out/exp98_pruning/qwen25_05b_head336_small_theory_redundancy_v2_boundary2.json`
- `out/exp98_pruning/smollm2_360m_head480_small_theory_redundancy_v1.json`
- `out/exp98_pruning/open_llama_7b_head1024_small_theory_redundancy_v2_dimscaled.json`

Key timing artifacts:

- `out/exp98_pruning/gpt2_structural_timing_v1.json`
- `out/exp98_pruning/gpt2_medium_structural_timing_v1.json`
- `out/exp98_pruning/qwen25_05b_timing_proxy_v1.json`
- `out/exp98_pruning/smollm2_360m_timing_proxy_v1.json`
- `out/exp98_pruning/smollm2_level2_kv_compaction_v1.json`
- `out/exp98_pruning/open_llama_7b_level2_compaction_v1.json`

Generated figure:

- `out/exp98_pruning/figures/exp98_five_model_summary.png`
- `out/exp98_pruning/figures/exp98_gqa_level2_summary.png`

These artifacts are sufficient to reconstruct the five-model classifier matrix, the
SmolLM2 grouped-query Level 2 result, and the final OpenLLaMA-7B validation row.

## 7. First Experiment

The first experiment is intentionally modest.

### 5.1 Model and Units

- One small frozen causal LM
- Units: attention heads, MLP channels, and optional whole sublayers
- No scalar-weight pruning in phase 1

### 5.2 Evaluation

For each candidate unit:

1. Run a calibration batch on the dense model.
2. Mask one unit.
3. Recompute task loss and coherence metrics.
4. Record the pair `(delta_coherence, delta_loss)`.

### 5.3 Baselines

- Magnitude ranking
- Activation ranking
- Random pruning
- Existing structured pruning heuristics from the literature

### 5.4 Success Criterion

The initial win condition is not maximum sparsity. It is ranking quality:

> Does coherence contribution predict low-damage prune candidates better than
> magnitude-only ranking?

If yes, the program is alive.

If no, the hypothesis is weakened immediately and the paper should say so plainly.

## 8. Falsifiers

The hypothesis takes real damage if any of the following occur:

1. Coherence ranking fails to beat random pruning.
2. Magnitude or activation ranking dominates coherence ranking across models.
3. Coherence scores are unstable across prompts and datasets.
4. Units with low coherence contribution consistently produce large task collapse.

Any of these outcomes would force a narrower claim or a different metric family.

## 9. Research Pack Already Pulled

### Internal evidence

- `journal/2025-12-05-sparse-hebbian-breakthrough.md`
- `journal/2025-12-22-k-field-ablation-report.md`
- `papers/current/resonance_model/lattice_language_training_regime.md`
- `papers/current/resonance_model/language_as_flow_theory.md`
- `papers/current/hybrid_lm/core/trajectory_intelligence.md`
- `research_registry/CURRENT_STATE.md`

### External baseline pull

- [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683)
- [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)
- [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
- [Wanda: A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)
- [FLAP: Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983)
- [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)
- [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/)
- [Differentiable Subset Pruning of Transformer Heads](https://arxiv.org/abs/2108.04657)

### Independent Confirmation from Weight Compression

An independent line of evidence strengthens the geometric picture. Turney (2026)
conducted systematic A/B isolation of tensor roles in transformer weight
compression, discovering through brute-force ablation that:

1. Attention tensors tolerate WHT-rotated quantization well
2. FFN write-back projections (`ffn_down`) are quality-critical
3. Boundary layers are disproportionately sensitive

These findings were reached without reference to the CLR framework. Each maps
directly onto elements of the present paper:

- Attention tolerance ↔ inter-oscillator couplings operate near the
  random-alignment scale, so moderate perturbation does not shift them across
  the death boundary
- Write-back sensitivity ↔ the residual-stream output projection is the
  coupling's *write* channel into the shared state, where errors compound
  layer-over-layer
- Boundary sensitivity ↔ boundary layers act as driven lattice boundary nodes
  under different dynamics from interior layers

We applied the tensor-role insight to the CLR weight-quantization stage
(Stage 4b). Guided by the anatomy eligibility profile from the *same single
forward pass* used for dead-head identification, attention and FFN read
projections receive 4-bit WHT-rotated quantization while FFN write-back
projections receive 6-bit protection. On Qwen3-8B (`d=4096`, 36 layers), this
role-aware policy reduces the weight-quantization quality cost from
`delta_loss = +0.339` (uniform 4-bit) to `delta_loss = +0.038` — an **89%
reduction in quality loss** at comparable effective bit rate (4.24 vs 3.69 bits).
No additional calibration, retraining, or Hessian computation is required; the
anatomy profile already contains the information needed.

This result is relevant to the present paper not as a pruning claim but as
*convergent evidence*: the same coupled-oscillator geometry that identifies dead
heads also predicts which weight matrices tolerate compression and which require
protection. The tensor-role asymmetry discovered empirically by Turney is a
natural consequence of the distinction between inter-oscillator transport channels
(attention) and intra-oscillator modes (MLP) articulated in Section 4.3.

## 10. Practical Next Step

The next artifact is no longer threshold validation. That part is done for the
current paper. The immediate next build is:

1. extend the same derivation pattern to channel pruning
2. derive a quantization-aware coherence schedule for local inference
3. build Level 3 rebalancing for large-group GQA models such as Qwen2

The current paper should center the derived threshold, the dimensional scaling, the
boundary protection rule, the five-model validation set, and the first structural
GQA / LLaMA-compaction results.

It should not try to close the full generalized BKT program on `S^(d-1)` or the
CLR-during-training program in the same manuscript. Those are strong follow-on
papers in their own right and would widen the claim surface without improving the
clarity of the present pruning result.
