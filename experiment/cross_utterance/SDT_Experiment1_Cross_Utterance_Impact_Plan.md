# Experiment 1 — Cross-Utterance Predictive Impact Analysis (Frozen SDT)

> **Implementation brief for Codex.** This document is the requested implementation plan, not a claim that the experiment has been implemented. Start by inspecting the actual repository, configuration, dataset pipeline, and checkpoint loading code. Reuse existing components and naming conventions; do not assume paths or APIs exist merely because they appear below.

## 0. Research question and scope

**Question:** In an already-trained Multimodal Emotion Recognition in Conversation (MERC) model, does information from one source utterance and modality have a measurable, positive or negative effect on the emotion prediction of another target utterance?

For a conversation with valid utterances `0..T-1`, source utterance `j`, target utterance `i != j`, and modality `m ∈ {text, audio, visual}`, measure the change in the **target** utterance's ground-truth log-probability when the **source** modality feature is replaced.

**Strict Phase 1 constraints**

- Use the existing **fully trained SDT checkpoint** as-is; **no retraining, fine-tuning, optimizer, or parameter updates**.
- Use `model.eval()` and `torch.inference_mode()` (or equivalent). Verify the repeated full-context forward is deterministic.
- Analyze **zero replacement** first, then **training-split feature-mean replacement**.
- Apply replacement to the **raw/pre-contextualized utterance-level modality input** consumed by SDT, *before* Conv1D / positional and speaker embeddings / intra- and inter-modal Transformers (subject to the actual code structure). Do **not** substitute a post-Transformer embedding as if that erased source information.
- Change **exactly one source utterance and one modality** per intervention. Keep every other input, conversation metadata, label, and all model weights unchanged.
- Keep the **original dataset split, labels, feature preprocessing, inference semantics, and metric protocol**. Do not leak test labels into model inputs.
- Analyze `j != i`. Optionally compute `j == i` only in a clearly separated diagnostic file; never combine it with cross-utterance statistics.
- Do not modify the original SDT training/inference entry points in a way that changes regular predictions. Prefer a self-contained analysis script and small reusable helpers.

## 1. Metric definitions (implement exactly)

Let `P_full[i, k]` be the SDT predicted probability of emotion class `k` for target utterance `i` using the original unmodified conversation. Let `P_mask[j,m][i,k]` be the prediction for target `i` after intervening on source utterance `j`, modality `m`.

For ground-truth emotion `y_i`, define the **signed predictive impact**:

\[
I_{j\rightarrow i}^{m,b}
= \log P_{\mathrm{full}}[i,y_i]
- \log P_{\mathrm{mask}}^{j,m,b}[i,y_i]
= \mathrm{CE}_{\mathrm{mask}}(i) - \mathrm{CE}_{\mathrm{full}}(i),
\]

where `b ∈ {zero, train_mean}` is the replacement strategy. Use **log-softmax of logits** (or clamped probabilities only if logits are unavailable) for numerical stability.

- `impact > 0`: source information is **helpful** under this particular intervention: removing/replacing it increases target loss.
- `impact < 0`: source information is **harmful** under this intervention: removing/replacing it decreases target loss.
- `impact ≈ 0`: near-zero / indeterminate under the chosen tolerance.

Save also:

- `delta_true_class_prob = P_full[i,y_i] - P_mask[i,y_i]` (different scale from log impact).
- `pred_full`, `pred_mask`, `correct_full`, `correct_mask`, `loss_full`, `loss_mask`.
- `changed_prediction = (pred_full != pred_mask)`.
- `js_divergence` between full and masked prediction distributions as an **unsigned prediction-change** diagnostic (optional but useful; do not confuse with signed predictive utility).

**Important interpretation:** These are *model-specific, intervention-dependent* predictive contributions, **not** proven real-world causal effects or intrinsic modality reliability. Treat replacement artifacts as a potential confound.

## 2. Repository inspection — mandatory before coding

Codex should inspect and report these details first:

1. Repository layout, exact SDT model class, forward signature, checkpoint format, evaluation entry points, and supported datasets.
2. How IEMOCAP (preferred first) and optionally MELD are split and loaded; how a conversation is represented, padded/batched, and identified.
3. Tensor layout and shape for `text`, `audio`, `visual`: `[T,D]`, `[B,T,D]`, `[T,B,D]`, or another layout. Locate each modality's input tensor **before temporal convolution**.
4. Where the model introduces speaker IDs, position encodings, attention masks / valid utterance masks and padding.
5. Whether the model uses full (bidirectional) conversation context; if so, label results **full-conversation contextual impact**, not history-only/causal prediction.
6. How final fused emotion logits are produced. Use **final fused logits**, not auxiliary student head logits, for the primary impact metric.
7. Whether features are already standardized/centered and how to compute replacement means in the **same feature space** without including validation/test data.

If an assumption conflicts with the repository, adapt the implementation and document the discrepancy. Ask for genuinely missing external paths (checkpoint or data) rather than inventing them.

## 3. Required implementation

### Task A — Frozen baseline inference

- Add a script such as `experiments/context_impact/run_impact_analysis.py` (adapt to repo style).
- Load configuration + checkpoint using the repo's existing utilities.
- Set eval mode, ensure all gradients are disabled, seed random number generators for reproducibility, and avoid all stochastic augmentations.
- Iterate **whole conversations** from the chosen evaluation split; preserve stable `conversation_id`, `utterance_index`, `speaker_id`, `emotion_label`, and valid-length masks.
- Perform one unmodified inference per conversation; cache **all valid target logits/log-probabilities and predictions**.
- Check that this full inference reproduces the existing validation/test metric for the same checkpoint (within numerical tolerance). If not, halt and explain the mismatch before interpreting impacts.

### Task B — Zero replacement interventions

For every valid source utterance `j` and modality `m`:

1. Copy the **pre-encoder/pre-convolution input feature tensor** of modality `m` (do not mutate the cached unmodified batch).
2. Replace only the source row `j` with zeros in that modality's actual input feature space.
3. Keep sequence length, speaker IDs, position indices, label tensors, attention/padding masks, other modalities and all other utterances unchanged.
4. Run the **entire original SDT forward pass** under inference mode.
5. Compare outputs for **all valid target utterances `i != j`** with cached full outputs; create a row per `(conversation_id, source_j, target_i, modality, replacement_strategy)`.
6. Restore original inputs (use a fresh clone or safe restore) before the next intervention.

**Important SDT nuance:** Replacing only the raw source feature eliminates that specific feature vector, not all presence/metadata associated with the utterance. Speaker/position embeddings and neighboring signals may still affect results. Report the experiment as *source-modality feature replacement*, not deletion of the entire utterance.

### Task C — Training-mean replacement interventions

- Calculate `mean_feature[m]` across **valid utterances in the training split only**, after the exact preprocessing used by the SDT input pipeline. No labels are needed for computing means.
- Save the mean vector and its metadata (split, feature dimension, preprocessing, mean norm, zero fraction) to the experiment output directory.
- Repeat Task B replacing only source row `j` with `mean_feature[m]`.
- Check whether the learned/preprocessed features are approximately zero-centered. If `zero` and `train_mean` nearly coincide, **explicitly disclose that this is not an independent perturbation robustness check**; do not manufacture a difference.
- Preserve the same evaluated conversations and source–target pairs for paired comparisons across strategies.

### Task D — Robust data export

Write one main long-format CSV or Parquet file, e.g. `results/impact_pairs.csv`, with at least:

| Column | Meaning |
|---|---|
| `dataset`, `split`, `checkpoint_id`, `seed` | Provenance |
| `conversation_id`, `conversation_length` | Conversation identity and valid length |
| `source_idx`, `target_idx`, `source_speaker`, `target_speaker` | Pair identifiers |
| `same_speaker`, `signed_offset`, `absolute_distance`, `source_before_target` | Context relations |
| `modality`, `replacement_strategy` | Intervention |
| `source_emotion`, `target_emotion` | **Analysis-only** ground-truth labels, never estimator inputs |
| `loss_full`, `loss_mask`, `signed_impact` | Primary outcome |
| `prob_full_true`, `prob_mask_true`, `delta_true_class_prob` | Interpretability |
| `pred_full`, `pred_mask`, `correct_full`, `correct_mask`, `changed_prediction` | Prediction outcomes |
| `js_divergence` | Optional unsigned output change |

Use canonical zero-based indices unless the repo convention requires otherwise; document any conversion. The `target_emotion` is legal for offline analysis of a held-out split and **must not be used to select source utterances at inference**.

Expected number of **cross-utterance rows** per conversation of length `T` for both strategies and all 3 modalities:

`2 × 3 × T × (T - 1)`.

A full baseline inference plus **`2 × 3 × T` intervention forwards** per conversation is sufficient to produce all the cross-utterance rows, because every intervention yields predictions for all targets. Avoid an unnecessary separate forward for every `(i,j)`.

### Task E — Summary statistics and visualizations

Implement `experiments/context_impact/analyze_impacts.py` (name can differ), reading saved results rather than recomputing inference.

**Primary outputs:**

1. Signed impact **histograms/distributions** for text, audio, visual, shown separately by `zero` vs `train_mean`.
2. Per-modality fraction of **positive / negative / near-zero** impacts, with predeclared tolerance (`impact_eps`, configurable; report sensitivity to a few tolerances).
3. Signed impact vs **absolute/signed temporal distance**, with source-before/after-target clearly distinguished.
4. Impact by **same-speaker vs cross-speaker**.
5. Per-conversation or per-target variation in impact (avoid conclusions drawn only from pooling millions of correlated rows).
6. For the exact same source–target–modality pairs, compare zero and train-mean scores: **Pearson/Spearman**, sign agreement outside near-zero, and top-K overlap/ranking agreement.
7. Optional: emotion-label agreement (`source_emotion == target_emotion`) as an **offline analysis**, not an inference feature; impact of same emotion must **not** be assumed positive.
8. Optional: effect on predicted label changes (`correct_full` vs `correct_mask`): distinguish `correct→wrong`, `wrong→correct`, and unchanged; report absolute counts.

Report aggregate counts and means/medians/quantiles, plus **conversation-level bootstrap confidence intervals** for the key aggregate statistics if straightforward. Do not treat dependent source–target rows from the same conversation as independent observations in significance testing.

Preferred output filenames (adapt to repo):

- `results/impact_pairs.csv` (or Parquet if large)
- `results/summary_by_modality.csv`
- `results/summary_by_distance.csv`
- `results/summary_by_speaker_relation.csv`
- `results/masking_robustness.csv`
- `figures/impact_distribution_by_modality.png`
- `figures/impact_vs_distance.png`
- `figures/impact_by_speaker_relation.png`
- `figures/zero_vs_mean_scatter.png`
- `report.md` describing observations, limitations and interpretation
- `config_used.yaml` or JSON and `environment_info.txt`

Make plots readable in a research paper: consistent axis units (signed cross-entropy difference), separate strategies, clearly labeled zero line, figure legends and sample counts.

## 4. Execution order for Codex

- [ ] **0. Inspect repository** and identify actual SDT checkpoint path, feature input points, dataset loading, masks, logits and speaker IDs.
- [ ] **1. Reproduce baseline**: full checkpoint inference matches original evaluator on a small subset and then on the intended split.
- [ ] **2. Small smoke test**: 1–3 conversations, zero replacement, verify shapes and all expected `(j,m,i)` rows.
- [ ] **3. Compute means** from training-split features only and run a smoke test of mean replacement.
- [ ] **4. Full zero-mask impact generation** on the evaluation split.
- [ ] **5. Full train-mean impact generation** on exactly the same conversations.
- [ ] **6. Generate summaries/figures** from exported impact pairs.
- [ ] **7. Run validation checks and report** reproducible commands, counts, masking behavior, and limitations.

Do **not** begin with new architectures, new loss functions, modality trust networks, rectified flow, Shapley estimation, retraining, or optimization experiments. Those belong to subsequent phases.

## 5. Tests / acceptance criteria

### Mandatory correctness checks

- [ ] Baseline full-input checkpoint predictions match the repository's original evaluation path (within tolerance).
- [ ] All measured inputs use `model.eval()` and no gradients/parameter updates occur.
- [ ] The original conversation tensors are never changed permanently by an intervention.
- [ ] A zero or train-mean intervention changes **one** raw input row in **one** modality, and only for valid source positions.
- [ ] Speaker/position IDs and valid sequence masks remain unchanged by modality-feature replacement.
- [ ] Every impact row has `source_idx != target_idx` and refers to a valid non-padded utterance.
- [ ] For each conversation and strategy, valid row count is exactly `3*T*(T-1)` (or the correctly documented count for modalities available in the dataset).
- [ ] Numerically check `signed_impact == loss_mask - loss_full` on sampled rows.
- [ ] No `NaN`, `inf`, duplicate `(conversation, source, target, modality, strategy)` keys, or inconsistent source/target speakers.
- [ ] Mean replacement feature vectors are computed from **training** data only; verify dimension/dtype/device compatibility.
- [ ] Signed offsets and speaker relation groupings are correct for known toy examples.
- [ ] Zero vs train-mean analysis is joined by exact pair keys, not by accidental row order.

### Useful additional checks

- [ ] Running the script twice with identical settings produces identical outputs (or differences within numeric tolerance).
- [ ] If a model input branch is independently known to have no effect on a controlled toy model, its measured impact is zero.
- [ ] Baseline per-conversation logits are reused rather than recomputed for every source–target row.
- [ ] At least one unit test checks the sign convention: if masking reduces correct-class probability, impact is **positive**.
- [ ] Confirm results on an independent run/checkpoint if one is available, but do not require retraining to complete Phase 1.

## 6. Interpret results conservatively

Phase 1 supports the working hypothesis only if cross-utterance impact:

1. exhibits **meaningful variation** across source utterances/modalities (not just numerical noise);
2. includes **repeatable helpful and harmful contributions** under well-specified interventions;
3. has reasonably **stable distributions/rankings across zero and train-mean replacement**, acknowledging cases where both baselines are effectively identical; and
4. displays potentially meaningful relations with temporal distance, speaker relation, or modality, without claiming causality.

If zero and train-mean yield contradictory impact signs for many substantial-impact pairs, treat that as **measurement instability**, not proof that the hypothesis is false or true. Document the limitation and propose a *separate future* robustness study with mask-aware fine-tuning or more realistic interventions.

**Do not claim** that: negative impact means a modality is inherently unreliable; same-emotion neighbors must help; strong impact is the same as an actual emotional cause; or that a significant distribution alone proves impact is learnable at inference time.

## 7. Final delivery expected from Codex

At completion, provide:

1. A concise list of changed/added files and why.
2. Exact commands (with the repository's real CLI options and paths) to run the zero and train-mean analyses and regenerate figures.
3. Baseline evaluation consistency check and test results.
4. Paths of exported tables, plots and `report.md`.
5. A brief written summary of **observed** positive/negative impact patterns, masking robustness, and any important blockers.

**Do not invent experimental numbers.** If data or checkpoint paths are unavailable, implement what can be implemented, make the missing paths configurable, and clearly report what could not be executed.

---

## Codex task prompt (paste together with this file)

> Implement **Phase 1 — Frozen SDT Cross-Utterance Predictive Impact Analysis** exactly as specified in this Markdown plan. First inspect the actual repository and its SDT data flow; do not assume guessed file paths. Keep the trained checkpoint completely frozen. Add a reproducible analysis workflow that computes ground-truth signed cross-utterance impacts using zero and training-mean source-modality feature replacement, saves pairwise CSV/Parquet results, produces the requested summary tables and figures, and validates baseline prediction equivalence and intervention correctness. Use existing repository conventions. Run a small smoke test before a full evaluation. Do not train/fine-tune or change the SDT architecture. If the required checkpoint or dataset is unavailable, leave configurable paths and report the blocker rather than fabricating results.
