# Breast Histopathology Pruning: Objectives, Novelty, and End-to-End Experimental Guide

**Prepared:** 1 November 2026  
**Status:** Research proposal and execution guide; no completed scientific results are claimed.  
**Primary target:** ACM Transactions on Computing for Healthcare (HEALTH).  
**Stretch target:** ACM Transactions on Intelligent Systems and Technology (TIST), conditional on a distinctive method and broader evidence.

This document consolidates the substantive discussion in this chat, including the uploaded plan, literature assessment, revisions to the objectives, novelty concerns, experimental design, and decision gates. It is an organized synthesis rather than a verbatim transcript. Proposed additions below are recommendations, not already implemented capabilities or verified novel methods. The original uploaded proposal is reproduced in Appendix A for completeness.

## 1. Main objective

Develop a structured channel-pruning method that reduces model size and measured inference latency for breast histopathology classification while preserving malignant-case detection.

The scientific goal is to test whether explicitly protecting pruning-vulnerable malignant patients improves the detection–efficiency tradeoff beyond existing pruning, malignant-class weighting, and sensitivity-constrained model selection.

Publication is an outcome to pursue after the evidence exists. Neither using BreaKHis nor reducing parameters establishes a publishable contribution by itself.

## 2. Our three research objectives

| Objective | Research question | Required evidence |
| --- | --- | --- |
| O1: Characterize malignant vulnerability | Which layers/channels cause disproportionate deterioration for malignant patients, and do conventional importance measures detect this? | Patient-balanced sensitivity analysis; margin/loss changes; magnification and subtype analyses; stability across fits. |
| O2: Develop vulnerability-aware pruning | Can patient-balanced, malignant-vulnerability-aware channel selection and allocation improve the measured sensitivity–latency frontier? | Fully specified algorithm and comparisons isolating ranking, allocation, class weighting, and selection effects. |
| O3: Establish practical value and generalization | Does the benefit persist across architectures, datasets, and measured deployment conditions? | Frozen grouped evaluation, paired uncertainty estimates, compact-model comparisons, replication, and actual runtime measurements. |

### Working title

**Patient-Balanced Malignant-Aware Channel Pruning for Efficient Breast Histopathology Classification**

Use this only if the implemented method genuinely uses patient-balanced malignant information. If identities remain unverified filename groups, use “group-balanced” instead of “patient-balanced.” A neutral alternative is **Evaluating Detection–Latency Tradeoffs in Pruned Breast Histopathology Classifiers**.

## 3. Novelty assessment: what is and is not new

| Proposed element | Assessment |
| --- | --- |
| Layer-by-layer pruning sensitivity curves | Established analysis; useful but weak standalone novelty. |
| Accuracy–FLOPs or accuracy–latency Pareto curves | Necessary evidence, generally not a new method. |
| Nonuniform pruning ratios | Established general strategy. |
| Class-aware channel importance | Established; CATRO and other methods are relevant precedents. |
| Hardware-aware selection using measured latency | Established; NetAdapt is an important comparator. |
| Malignant weighting or a recall constraint | Potentially useful adaptation; likely incremental by itself. |
| Patient-separated evaluation, duplicates audit, repeated seeds | Scientific rigor; do not describe routine leakage prevention as an algorithmic invention. |
| Patient-balanced protection of vulnerable malignant cases with demonstrated benefit beyond existing alternatives | Candidate contribution; precise novelty remains unverified. |
| Generalizable findings about which patients pruning harms and why | Potential empirical contribution if robust and informative. |

### Important correction to earlier optimism

“Malignant-aware pruning” is a research direction, not an established novel contribution. A new name, weighted gradient score, or combination of known components does not prove originality. The method must be compared with the closest class-aware, sample-aware, and hardware-aware literature.

### The key novelty test

Could an established pruning method obtain the same benefit simply by using our validation constraints or malignant weighting? If yes, the contribution is primarily a selection/evaluation protocol. If our channel decisions improve the frontier after controlling those factors, a stronger methodological claim becomes possible.

## 4. Literature that shapes the plan

The following sources were identified in the internet research for this conversation. Publisher full text was blocked for some ACM pages; some assessments therefore rely on publisher-indexed passages, author code, or institutional records. This is not an exhaustive systematic review. Avoid “first” claims until the focused novelty review is complete.

| Work | Relevance and novelty boundary |
| --- | --- |
| EVC-Net, ACM TIST, 2026 | Hybrid EfficientNetV2S/ViT/capsule model for BreaKHis binary and eight-subtype classification. Publisher-indexed text reports five-fold patient-level evaluation, average binary accuracy 0.985 and AUC 0.994. These are reported results, not directly comparable benchmarks for our protocol. |
| Compressing Deep Neural Networks with Goal-Specific Pruning and Self-Distillation, ACM TKDD, 2025 | Retraining-Aware Pruning, NGGM filter pruning, and HAP self-distillation; demonstrates that goal-specific compression is already developed. |
| ChannelZip, ACM TOSN, 2025 | Verified recent journal paper on SLO-aware channel compression and task-adaptive serving; full methodological comparison still required. |
| Zhu et al., 2019 | Squeeze-Excitation-Pruning for compact breast histopathology CNNs; channel pruning in this domain is established. |
| Pouramin et al., IET Image Processing, 2026 | Architecture-aware block pruning across VGG, ResNet, and Inception under several training regimes on PatchCamelyon, not BreaKHis. |
| NetAdapt, ECCV 2018 | Empirical latency-aware progressive adaptation; close precedent for the original greedy allocation proposal. |
| AMC, ECCV 2018 | Automated nonuniform model-compression policy search. |
| CATRO, IEEE TNNLS | Class-aware channel pruning with joint channel effects and greedy optimization; important ranking comparator. |
| Class-Aware Pruning for Efficient Neural Networks, DATE 2024 | Further precedent for class-aware filter assessment; inspect code completeness before depending on reproduction. |
| How Does Pruning Impact Long-Tailed Multi-label Medical Image Classifiers? | Disease-dependent pruning effects; weakens claims that unequal medical-class damage is unexplored. |
| Attend Who is Weak | Pruning-sensitive medical examples and difficult cases; distinguish our contribution from existing sample-vulnerability work. |
| Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis, MICCAI 2024 | Group-dependent channel effects in medicine; demographic groups differ from our diagnostic patient/class grouping, but conceptual overlap matters. |

### Literature-review task

For every close paper record: exact citation/year/venue, publication status, task and dataset, grouping/split protocol, ranking score, allocation method, recovery budget, selection rule, runtime objective, available code, and precise difference from our proposal. Read full methods for the nearest competitors before choosing the final algorithm.

Search outside ACM too: novelty is judged against the entire relevant literature. An ACM-only review is insufficient.

## 5. Starting implementation: reported status versus work required

The uploaded proposal records an earlier repository inspection. These claims have not been independently rerun in this chat.

| Reported existing capability | Limitation to resolve |
| --- | --- |
| Canonical `breakhis/` package with manifest, training, pruning, evaluation | Do not treat legacy scripts as equivalent supported workflows. |
| Binary custom VGG and torchvision ResNet18 training | Eight-subtype training is not reported as implemented. |
| Deterministic layerwise L1 weight masks | Masks do not physically shrink dense tensors. |
| Physical channel surgery for custom eight-convolution VGG | This is not standard VGG16; broader architecture surgery requires implementation. |
| Per-layer ratio maps | Proposed adaptive search and sensitivity constraints are not implemented. |
| Validation-accuracy checkpoint selection | Must support the declared scientific selection policy. |
| Smoke/acceptance artifacts | The reported Stage 9 uses 12 synthetic images, one epoch, one timing repeat; not scientific performance evidence. |
| Historical local audit: 7,909 images, 7,784 decoded-unique images after 125 exclusions, 81 filename groups | Local findings require provenance reconciliation; groups are not independently verified clinical identities. |
| Retained 224×224 derivative | Restore/trace official source; original BreaKHis images are larger. |

## 6. Research questions and hypotheses

| ID | Question / hypothesis |
| --- | --- |
| RQ1 | Does conventional average channel importance hide disproportionately large effects on some malignant patients? |
| RQ2 | Does patient-balanced vulnerability assessment improve channel decisions beyond class weighting alone? |
| RQ3 | Does the mechanism improve sensitivity at matched latency and comparable specificity? |
| RQ4 | Do improvements persist across backbones, magnifications, and a second dataset? |
| RQ5 | Are measured runtime savings consistent with theoretical compute savings, and when do they diverge? |

Treat every hypothesis as falsifiable. A negative result should revise the method or paper claim, not be removed from the report.

## 7. Phase 1 — Freeze the scope and protocol

### Step 1: Define the task

Start with BreaKHis benign-versus-malignant classification. Use pooled magnifications as the main task and per-magnification analysis as secondary. Do not launch separate models for every magnification until the pilot supports the cost. Add subtype classification only with a separate question and adequate implementation.

### Step 2: Define the intended deployment workload

Choose a concrete image/ROI classification workload. Specify input resolution, batch size, runtime, precision, and device. If the motivation is microscope-field classification, do not imply the experiment validates whole-slide diagnosis or clinical deployment.

### Step 3: Predeclare scientific decisions

Write a versioned protocol containing primary comparisons, grouping unit, prediction aggregation, folds, seeds, thresholds, margins, search budgets, adaptation/final training budgets, hardware timing method, statistics, and stopping rules. Exploratory changes are allowed but must precede frozen confirmatory evaluation or be labeled exploratory.

**Exit gate:** the study can be described without undefined terms such as “patient-level accuracy” or “real-time.”

## 8. Phase 2 — Reconcile and audit the data

### Step 4: Establish source lineage

Obtain or trace the official source; record version/source URL, acquisition date, archive checksum, licensing/citation, and all preprocessing. Preserve raw files as read-only. Treat resized copies as derivatives, never independent data.

### Step 5: Build an image manifest

Suggested fields:

```text
image_id, relative_path, source_release, class_label, subtype,
filename_group_id, verified_patient_id_if_available,
magnification, original_width, original_height,
raw_sha256, decoded_pixel_sha256, duplicate_component_id,
included, exclusion_reason, preprocessing_version
```

Investigate exact duplicates and label conflicts. Exact hashes do not detect all near-duplicates; review suspected near-duplicates without automatically discarding genuinely different fields. Preserve the excluded-file list and reasons.

### Step 6: Define leakage-safe groups

Keep all magnifications and images from the same patient together. Combine groups linked by confirmed content duplication before splitting. If patient metadata cannot be verified, say “filename-group-separated,” not clinically verified patient-separated.

### Step 7: Produce the audit report

Report counts by class/subtype/magnification/group, duplicate exclusions, group-size distribution, missing/corrupt files, and differences from published counts. Explain the reported 81 versus nominal 82 identities rather than silently choosing one number.

**Exit gate:** traceable data, documented exclusions, and no cross-partition identity/content overlap.

## 9. Phase 3 — Construct the evaluation design

### Step 8: Create outer folds

Aim for stratified five-fold group-separated evaluation if class support permits. Stratify primarily by benign/malignant status and audit subtype support. A group has one split assignment across all magnifications. Preserve immutable split files and checksums.

### Step 9: Separate inner development roles

Within each outer training partition, reserve grouped development data for search/checkpoint selection. If feasible, reserve separate grouped data for shortlisted-model confirmation and threshold calibration. Small sample size may prevent all subdivisions; document the compromise rather than pretending independent evidence exists.

Do not repeatedly inspect outer test performance while developing the method. Develop on a designated pilot partition, freeze the method, then execute outer evaluation. Reusing an already inspected pilot test as independent confirmation is inappropriate; disclose any such use and use the second dataset as an additional untouched test.

### Step 10: Define aggregation

Suggested primary rule: average malignant probabilities across a group's images, then apply a development-selected threshold. Validate that pooling has a sensible interpretation for the data. If acquisition counts differ by magnification, specify whether averaging is image-weighted or first averaged within magnification.

Report image metrics and group-by-magnification metrics as secondary. Do not confuse mean within-patient image accuracy with one prediction per patient. Lock aggregation rules before testing.

### Step 11: Define thresholds and constraints

The original argmax/0.5 threshold remains a reference. Add a development-selected operating point with a specificity safeguard. Choose thresholds using development labels only and give every baseline the same opportunity.

For candidate M and dense reference D, an exploratory feasible set can require:

```text
sensitivity(M) >= sensitivity(D) - delta_sensitivity
specificity(M) >= specificity(D) - delta_specificity
balanced_accuracy(M) >= balanced_accuracy(D) - delta_balanced_accuracy
memory(M) <= declared_budget, if used
```

Choose tolerances before test evaluation. Two percentage points was discussed as an exploratory tolerance, not a clinically justified safety margin. Relative preservation of a poor dense model is insufficient; report absolute performance too. Avoid adding many arbitrary constraints merely to make the proposal look sophisticated.

**Small-sample caution:** with ten malignant test groups, one miss changes sensitivity by ten percentage points. With fifty pooled malignant groups, one miss is two points. Extra seeds do not create extra patients. Formal noninferiority needs a justified margin and adequate power; this study may only estimate tradeoffs with wide uncertainty.

## 10. Phase 4 — Establish competitive dense baselines

### Step 12: Choose models

Use custom VGG for development, but include a standard backbone such as ResNet18 and a compact comparator such as MobileNetV3. Broaden to a standard sequential backbone when ready. EfficientNet-B0 is another reasonable compact comparator; choose based on implementation and compute, not a desire for a large table.

### Step 13: Match training fairly

Record pretrained weights and version, resolution, augmentation, normalization, optimizer, learning-rate schedule, batch size, epochs, stopping rule, loss, checkpoint criterion, and training seed. Fit any stain normalization using training data only. Keep augmentation out of test evaluation. Do not tune one method extensively while leaving competitors at weak defaults.

Expose seeds for initialization, data loading, augmentation, and splits. Record determinism settings and unavoidable nondeterminism. Separate deployment timing sessions from deterministic-training choices.

### Step 14: Save reproducible baseline artifacts

For each fit save config, code commit, environment, split IDs, checkpoint, learning curves, development predictions, and runtime metadata. Test predictions are generated only in the frozen evaluation phase.

**Exit gate:** competitive dense performance, correct metrics, reproducible runs, and a measured epoch-time estimate.

## 11. Phase 5 — Validate conventional pruning before inventing a method

### Step 15: Verify physical channel surgery

Removing output channels requires propagating shape changes into downstream inputs, batch normalization, classifiers, and residual/dependency paths. Use a suitable dependency-aware implementation for branched models. Verify tensor shapes, forward/backward execution, serialization/reload, and that physical parameters and MAC dimensions actually fall.

Weight masks are a useful control but should not be advertised as dense-kernel speedup. Sparse storage formats and sparse kernels are separate deployment choices.

### Step 16: Run a bounded pilot

Explore a small set of compression settings, such as 25%, 50%, and 75% removal, on inner data only. These are starting points, not guaranteed feasible settings. Use equal adaptation budgets and measure real latency. Inspect both predictive deterioration and benefit over the compact dense model.

### Step 17: Build the minimum baseline set

| Baseline | Why it is needed |
| --- | --- |
| Dense parent | Reference quality and runtime. |
| L1 masked weights | Separate zeroing from structural compression. |
| Uniform L1 channel pruning | Simple structural baseline. |
| Random channels at matched layer widths | Determine whether ranking adds value. |
| Same narrow topology trained from initialization | Test pruning versus simply training a small network. |
| Compact dense network | Test the practical reason to prune the parent. |
| Established adaptive/hardware-aware pruning | Compare allocation/search fairly. |
| Class-aware pruning or a justified faithful adaptation | Compare malignant-aware ranking against related ideas. |

Choose a manageable set of strong reproducible competitors. NetAdapt-like allocation and CATRO-like class-aware ranking address different aspects; describe adaptations and deviations explicitly. ACM TKDD RAP/NGGM/HAP is relevant, but compare complete recovery pipelines fairly rather than silently giving one method additional distillation.

## 12. Phase 6 — Characterize malignant vulnerability (O1)

### Step 18: Measure layer/candidate perturbation effects

From a fixed dense checkpoint, temporarily ablate or physically remove small channel groups. Evaluate development-group changes before and after the same short adaptation budget. Log layer, retained widths, channel IDs, predictive effects, and latency.

For each malignant development group, measure change in mean loss, mean malignant probability, decision margin, and whether a previously correct group becomes incorrect. Negative changes can represent improvement; retain signed values for analysis.

### Step 19: Balance patients/groups

Aggregate within each group before averaging across groups. An individual with hundreds of images should not dominate the score merely because of image count. Apply analogous balancing to benign groups for specificity monitoring.

### Step 20: Look for a reproducible finding

Compare magnitude/gradient importance with observed group damage. Examine whether average importance hides concentrated harm and whether affected cases remain vulnerable across fits. Report subgroup support. Do not infer biological explanations from heatmaps alone; any pathology interpretation requires suitable expert evidence.

**Exit gate:** either a clear vulnerability pattern worth targeting or an honest finding that the proposed premise is unsupported.

## 13. Phase 7 — Develop the method (O2)

### Step 21: Start with the simplest useful candidate mechanism

A proposed implementation scaffold is:

1. Compute group-balanced malignant damage for each candidate channel removal.
2. Include a prespecified vulnerable-tail summary rather than only mean damage.
3. Monitor benign damage/specificity.
4. Measure positive latency improvement on the target device.
5. Rank promising removals and evaluate them after equal short adaptation.
6. Accept only development-feasible candidates.
7. Recompute scores after accepted pruning.
8. Fully fine-tune a bounded shortlist and freeze the chosen configuration.

This scaffold is **not a validated algorithm or novelty claim**. Hard-case/sample-aware and class-aware precedents must be examined before finalizing the score.

### Optional exploratory score

Let d_g(c) be the change in average loss for malignant group g after candidate removal c. Define:

```text
D_mean(c) = mean of d_g(c) across malignant groups
D_tail(c) = mean damage among a prespecified worst-damage fraction of groups
D_benign(c) = mean loss damage across benign groups
R(c) = [a*D_mean(c) + b*D_tail(c) + c_weight*D_benign(c)] / latency_gain(c)
```

Prefer smaller R only when latency gain is positive and reliably measured. The tail fraction and weights are development hyperparameters with a fixed tuning budget. Loss damage is a proxy, not sensitivity itself. Tiny/noisy denominators make ratios unstable; use a minimum meaningful latency gain or constrained ranking instead. Group size is small, so worst-case/tail estimates can be unstable. Treat this score as an ablation candidate rather than a required final method.

### Step 22: Specify the search fully

Define candidate width increments, minimum widths, dependency groups, device-friendly width choices where relevant, maximum evaluations, adaptation steps, candidate reset checkpoint, tie handling, infeasibility behavior, and termination. Reuse scores only when justified after topology changes. A one-layer-at-a-time search can miss interacting removals; evaluate a limited interaction extension only if the pilot demonstrates the issue.

Never use outer test labels, predictions, or test-run performance to tune the search. Device latency can be measured with synthetic inputs; predictive feasibility uses inner data.

### Conceptual pseudocode

```python
# Conceptual specification: these are not existing CLI functions.
current = dense_checkpoint
history = []
while search_budget_remains():
    candidates = propose_dependency_safe_removals(current)
    candidates = screen_with_group_balanced_damage_and_latency(candidates)
    evaluated = []
    for candidate in candidates:
        model = rebuild_from_same_current_checkpoint(current, candidate)
        model = adapt_on_inner_training(model, fixed_steps)
        quality = evaluate_on_inner_development(model)
        runtime = benchmark_on_search_device(model)
        history.append((candidate, quality, runtime))
        if satisfies_predeclared_constraints(quality, runtime):
            evaluated.append(model)
    if not evaluated:
        break
    current = choose_by_frozen_rule(evaluated)
shortlist = select_bounded_shortlist(history)
finalists = fully_finetune_with_equal_budget(shortlist)
chosen = select_using_inner_data_only(finalists)
```

## 14. Phase 8 — Isolate the actual contribution

### Step 23: Run the decisive factorial-style comparisons

| Variant | Ranking / allocation | Model selection | Interpretation |
| --- | --- | --- | --- |
| A | Conventional pruning | Accuracy-based | Ordinary reference. |
| B | Same as A | Sensitivity/specificity constrained | Benefit from selection only. |
| C | Malignant-weighted loss/importance baseline | Same constraints as B | Benefit from simple class weighting. |
| D | Proposed group-balanced vulnerable-case mechanism | Same constraints as B | Incremental mechanism benefit. |
| E | Strong established adaptive/class-aware method | Same constraints as B | Whether an existing approach already suffices. |

Match measured latency targets, candidate budgets, initialization policy, and adaptation/final training resources. Record failures to reach a target. Do not compare only equal pruning percentages.

### Step 24: Targeted ablations

- Image-weighted versus group-balanced importance.
- Mean malignant damage versus mean-plus-tail damage.
- Uniform versus adaptive allocation with ranking held constant.
- Fixed ranking versus iterative reassessment.
- Accuracy-only versus constrained selection.
- Short adaptation versus full recovery; assess whether search rankings survive full recovery.
- Inherited weights versus retraining the selected widths, under explicit pretraining policies.

Avoid exhaustive combinations without a scientific question. Start with the ablations that distinguish our mechanism from existing alternatives.

**Decision gate:** if D does not improve on B/C/E repeatably, revise the method or reframe the study as an empirical evaluation. More plots or more seeds do not fix missing novelty.

## 15. Phase 9 — Execute the frozen study and replication (O3)

### Step 25: Freeze before final tests

Lock code commit, data/split hashes, selection policy, threshold policy, search and recovery budgets, metrics, statistical comparisons, and hardware protocol. Run each method independently within each outer fold. Select compression entirely inside that fold's development data.

### Step 26: Generate out-of-fold predictions

Save probabilities, true labels, group IDs, magnification, subtype, fold, seed, threshold, architecture, method, selected widths, and model/config identifiers. A sample receives test predictions only from a model that excluded its entire group from training and selection.

### Step 27: Add independent replication

Use a second suitable dataset. BRACS ROIs are a candidate because official resources provide group/patient metadata and reference splits. It includes normal/benign, atypical, DCIS, and invasive categories; do not silently merge atypical tissue into malignant. State the task and justify its labels separately. Prefer supplied patient-separated splits when appropriate and audit them.

Replicating the pruning method after training on the second dataset is methodological replication. Training on BreaKHis and testing on another compatible source is external transfer validation. These are different experiments and must not be conflated.

If cross-magnification transfer is investigated, exclude all images of test groups from training at every magnification.

## 16. Outcomes, statistics, and honest interpretation

### Primary outcomes

Group malignant sensitivity and specificity/balanced accuracy at the locked operating point, together with batch-one median forward latency. Predeclare a small number of paired method comparisons and latency budgets. If multiple endpoints are designated primary, explain their decision rule rather than selecting whichever looks favorable.

### Secondary outcomes

Image metrics, macro-F1, ROC-AUC, malignant-class PR-AUC, Brier score/calibration, per-magnification effects, malignant-subtype errors with support, p95 latency, model bytes, physical parameters, MACs, peak inference memory, search time, recovery time, and measured end-to-end throughput.

### Statistical workflow

1. Pair method predictions on identical held-out groups.
2. Resample independent groups or duplicate-linked components, not individual images.
3. Preserve paired outcomes when calculating bootstrap differences.
4. Specify handling of resamples without class support; class-stratified group resampling may be appropriate for sensitivity/specificity estimation and should be disclosed.
5. Report confidence intervals for differences, denominators, and raw false-negative counts.
6. Treat OOF bootstrap uncertainty as conditional on fitted models; it does not capture all training/split uncertainty.
7. Report fold and seed variation separately. Fifteen fits from five folds and three seeds are not fifteen independent datasets.
8. Predefine how multiple seeds enter estimates: report per-seed OOF results or a declared aggregation; do not duplicate patients in a bootstrap as if independent.
9. Control a small set of confirmatory comparisons; label broader subgroup exploration and handle multiplicity appropriately.

Sensitivity can be raised by predicting malignant more frequently. Every “preserved sensitivity” claim needs accompanying specificity and threshold information. Uncertain evidence should be reported as uncertain, not equivalence or clinical safety.

## 17. Deployment measurement protocol

Record device model, CPU/RAM, GPU/driver, operating system, framework/runtime versions, precision, thread counts, input resolution, batch size, export settings, and device power/performance mode where relevant.

1. Use evaluation mode and inference/no-gradient mode.
2. Keep inputs resident for forward-only measurements.
3. Warm up until representative steady behavior; choose repeats from a pilot instead of copying an arbitrary number.
4. For asynchronous GPU execution use correct synchronization or device events.
5. Run independent sessions, randomize model order, and avoid competing workloads.
6. Report median and p95 with session variation; iterations within a session are not independent device replications.
7. Separately time preprocessing, transfer, inference, and aggregation for the declared end-to-end workload.
8. Separate inference artifact size from checkpoints containing optimizer state.
9. Measure exported artifacts if making export/deployment claims.
10. Define memory measurement precisely; GPU allocated/reserved memory and process/device peak memory are different quantities.

Report CPU and GPU if available; otherwise limit claims to measured hardware. RTX 5070 Ti 16 GB, if used, is training/development hardware unless it is the intended deployment target. Pilot epoch time and peak memory before scheduling the full study. Do not infer edge suitability from desktop-GPU latency alone.

Parameter reduction, MAC reduction, storage reduction, latency improvement, and energy savings are separate claims. Measure energy if claiming energy savings; measured FLOPs do not establish it.

## 18. Reproducibility structure and run records

Suggested layout (to create in the project repository; not claimed to exist):

```text
project/
  configs/             # training, pruning, search, hardware settings
  data_manifests/      # source lineage and exclusions
  splits/              # immutable group assignments
  src/                 # canonical implementation
  tests/               # surgery, leakage, metrics, checkpoint correctness
  experiments/pilot/
  experiments/frozen/
  predictions/         # development and held-out outputs kept distinct
  benchmarks/          # timing sessions and device metadata
  analysis/            # reproducible tables and figures
  protocol.md
  decision_log.md
  environment.lock
```

Minimum run record:

```yaml
run_id: unique_identifier
code_commit: recorded_commit
data_manifest_hash: recorded_hash
split_hash: recorded_hash
phase: pilot_or_frozen
dataset: breakhis
architecture: selected_backbone
seed: recorded_integer
outer_fold: recorded_fold
pretraining: exact_policy_and_weights
method: recorded_pruning_variant
ranking_rule: recorded_rule
allocation_rule: recorded_rule
search_budget: fixed_trial_or_time_budget
adaptation_steps: fixed_integer
final_training_budget: recorded_budget
selection_rule: frozen_rule
threshold_rule: inner_data_only
device_profile: recorded_profile
```

These are schema examples, not runnable configuration for the existing CLI. Write actual commands only after verifying supported arguments. Meaningful correctness checks include no group/hash leakage, metric correctness on hand-checkable examples, pruning shape/dependency correctness, and equivalent predictions after export/reload within expected numerical tolerance.

## 19. Compute budget and execution order

Start with a cheap pilot, then expand only after a positive scientific decision. Two architectures × five folds × three seeds already requires 30 dense fits; three compressed versions per fit add 90 recovery jobs, before adaptive search, ablations, and replication. Estimate from measured epoch/search timings.

| Stage | Deliverable | Proceed when |
| --- | --- | --- |
| 1: Literature/protocol | Nearest-neighbor comparison and frozen definitions | Distinction is clear enough to test. |
| 2: Data audit | Traceable manifest and grouped splits | No unresolved leakage/provenance issue. |
| 3: Dense models | Competitive baselines and cost estimate | Training/evaluation is trustworthy. |
| 4: Conventional pilot | Measured compression frontier | There is practical headroom beyond compact baselines. |
| 5: Vulnerability analysis | Evidence supporting or rejecting premise | Mechanism is motivated by observed behavior. |
| 6: New mechanism and decisive ablations | Benefit beyond selection/class weighting | Advantage is repeatable and worth replication. |
| 7: Frozen full study | OOF predictions and runtime sessions | No test-driven redesign. |
| 8: Replication/statistics | Generalization and uncertainty | Claims match evidence. |
| 9: Manuscript/artifact | Reproducible paper package | Submission gates pass. |

The uploaded plan estimated roughly 8–13 weeks across overlapping preparation, pilot, method, full experiments, and writing phases. This is only a planning estimate; compute time and implementation gaps can change it substantially.

## 20. Planned paper exhibits

| Exhibit | Purpose |
| --- | --- |
| Data audit and fold table | Show group/class support and exclusions. |
| Group vulnerability versus conventional importance | Motivate the mechanism. |
| Sensitivity–latency curves with specificity and uncertainty | Main practical result. |
| Main matched-latency comparison | Contrast strong competitors fairly. |
| Selection-only/class-weighting/mechanism ablations | Establish the source of benefit. |
| Retained layer widths | Explain allocation decisions. |
| Magnification/subtype error table | Expose concentrated failure with denominators. |
| CPU/GPU runtime and artifact table | Support deployment claims. |
| Second-dataset replication | Establish breadth and limits. |
| Search and recovery cost table | Make experimental/deployment tradeoffs transparent. |

## 21. Journal fit and submission gates

HEALTH is the most coherent initial target for a substantive computing contribution motivated by healthcare. TIST becomes more plausible with a distinctive method and broader evidence. TKDD is more natural if the work becomes a general compression algorithm. TOSN is relevant related literature but is not a natural target without a genuine sensor/networked-system contribution.

EVC-Net establishes topical precedent for BreaKHis in TIST; it does not establish that another BreaKHis paper will be accepted. A second dataset is our evidence-strengthening recommendation, not a universal journal rule. No acceptance probability, quartile, turnaround, or current fee claim is justified here. Check current official author guidance before submission.

Submission checklist:

- [ ] Source lineage and group definitions are defensible.
- [ ] No cross-fold group or confirmed duplicate leakage.
- [ ] Scientific experiments replace synthetic smoke results.
- [ ] Dense and compact baselines are competitive.
- [ ] Closest class/sample/hardware-aware methods are discussed fairly.
- [ ] Mechanism benefit is separated from selection and class weighting.
- [ ] Search and recovery budgets are reported and comparable.
- [ ] Sensitivity is interpreted with specificity and operating thresholds.
- [ ] Small-group uncertainty is visible; no unsupported noninferiority claim.
- [ ] Deployment benefits are measured on stated hardware/workloads.
- [ ] Replication supports the claimed scope.
- [ ] Clean-environment reproduction succeeds.
- [ ] Negative findings and infeasible budgets are disclosed.
- [ ] Abstract and contribution claims describe completed evidence only.

## 22. Immediate next actions

1. Finish the closest-literature matrix, particularly sample-vulnerability methods.
2. Resolve original-source and group-identity questions.
3. Freeze the pilot split, aggregation, operating point, and search budget.
4. Train one standard backbone and one compact comparator.
5. Validate physical pruning and record actual latency.
6. Compare ordinary selection, constrained selection, and malignant weighting.
7. Only then build the group-balanced vulnerable-case mechanism.

Our present judgment remains: strong experimental discipline, an important application question, and unproven methodological novelty. The experiment should decide whether there is a new algorithm, a useful empirical finding, or a need to redirect.

## 23. References and research links

1. [EVC-Net — ACM TIST](https://doi.org/10.1145/3800952).
2. [Compressing Deep Neural Networks with Goal-Specific Pruning and Self-Distillation — ACM TKDD](https://doi.org/10.1145/3721293); [institutional record](https://scholar.nycu.edu.tw/en/publications/compressing-deep-neural-networks-with-goal-specific-pruning-and-s/); [author code](https://github.com/kantlulu/NetworkCompression).
3. [ChannelZip — ACM TOSN](https://doi.org/10.1145/3729534). Full-text method comparison remains outstanding.
4. [Zhu et al., compact CNNs and channel pruning for breast histopathology](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805574/).
5. [Pouramin et al., architecture-aware block pruning, 2026](https://doi.org/10.1049/ipr2.70414).
6. [NetAdapt — author project and paper links](https://web.mit.edu/netadapt/).
7. [AMC — author project](https://hanlab.mit.edu/projects/amc).
8. [CATRO — author preprint](https://arxiv.org/abs/2110.10921); [journal DOI](https://doi.org/10.1109/TNNLS.2023.3262952). The preprint predates the journal issue; cite the version actually used.
9. [Class-Aware Pruning, DATE 2024 — author code](https://github.com/HWAI-TUDa/Class-Aware-Pruning).
10. [How Does Pruning Impact Long-Tailed Multi-label Medical Image Classifiers? — bibliographic record](https://pubmed.ncbi.nlm.nih.gov/37829549/).
11. [Attend Who is Weak — paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10089697/).
12. [Achieving Fairness Through Channel Pruning — MICCAI 2024 paper and reviews](https://papers.miccai.org/miccai-2024/040-Paper2133.html).
13. [BreaKHis — official dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
14. [BRACS — official data details](https://www.bracs.icar.cnr.it/details/); [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9575967/).
15. [MobileNetV3 breast histopathology classification study](https://doi.org/10.1016/j.jpi.2024.100377).
16. [ACM HEALTH](https://dl.acm.org/journal/health); [ACM TIST](https://dl.acm.org/journal/tist).

Additional sources surfaced during the discussion, not selected as core comparators: [Interpretability-Aware Pruning for Efficient Medical Image Analysis, preprint](https://arxiv.org/abs/2507.08330); [progressive pruning for brain MRI detection, JMIR 2026](https://medinform.jmir.org/2026/1/e84095); [latency-surrogate compression, CVPR Workshop 2024](https://openaccess.thecvf.com/content/CVPR2024W/MAI/html/Nasery_End-to-End_Neural_Network_Compression_via_l1l2_Regularized_Latency_Surrogates_CVPRW_2024_paper.html); [PriPrune, ACM journal paper on privacy-aware federated pruning](https://doi.org/10.1145/3702241). These address adjacent problems; do not imply direct malignant-sensitivity overlap without reading their methods.

---

## Appendix A — Original uploaded research proposal

The following is reproduced for provenance. It records an earlier source/repository review and planning assumptions. Where its emphasis differs, the revised objectives and protocol above express the later discussion. Its source-inspection statements remain reported historical findings, not independently verified results from this chat.

```text
Research paper plan: efficient breast histopathology classification
Prepared 31 October 2026. This is a research proposal grounded in repository inspection, not a report of completed scientific experiments.

Recommendation
Target ACM Transactions on Computing for Healthcare (HEALTH) first. Build the paper around compression under group-separated evaluation, preservation of malignant-case sensitivity, and measured deployment costs. Treat ACM Transactions on Intelligent Systems and Technology (TIST) as a more ambitious alternative if the work develops a general compression algorithm and validates it beyond this dataset and architecture.

Working title: Sensitivity-Constrained Channel Pruning for Efficient Breast Histopathology Classification under Group-Separated Evaluation.

Use this title only if the proposed method is implemented and supported by experiments. A descriptive alternative is Evaluating the Accuracy–Latency Tradeoff of Neural Network Pruning in Breast Histopathology Classification. The latter needs substantial empirical findings to justify a journal article.

What the project currently establishes
Component	Evidence inspected	Research implication
Dataset audit	breakhis/manifest.py, docs/guides/DATASETS.md, historical artifacts/data/audit.json	Filename metadata, raw/pixel hashes, duplicate exclusion and group isolation are implemented.
Training	breakhis/cli.py, breakhis/data.py, breakhis/training.py	Binary VGG and torchvision ResNet18 workflows, validation selection and resumable checkpoints exist.
Weight pruning	breakhis/pruning.py	Layerwise L1 masks with deterministic ties and mask enforcement during optimization.
Structural pruning	breakhis/pruning.py, breakhis/models/vgg.py	Physical output-channel removal for the custom eight-convolution VGG, including BN and classifier reconstruction. This model is not standard VGG16.
Evaluation	docs/guides/EVALUATION.md, reporting/metrics/comparison modules	Image and filename-group classification, size accounting, Conv/Linear MAC estimates and resident forward latency.
Evidence maturity	Stage 6–8 smoke artifacts and artifacts/stage9-acceptance/summary.json	Execution evidence exists; Stage 9 explicitly uses 12 synthetic images, one epoch and one timing repeat. These are not scientific performance results.
The canonical package is breakhis/. Legacy scripts, CIFAR tutorials and other architecture files must not be represented as equally supported experimental methods. Eight-subtype training is not implemented in the canonical workflow. This review inspected source and stored artifacts; it did not rerun the test suite or train new models.

The historical audit reports 7,909 local binary images, 7,784 unique decoded images after 125 exclusions, and 81 filename patient/slide groups. These are local audit findings, not independently verified clinical identities. The original-resolution local copy has been removed. The retained v1 images are 224×224, whereas the official dataset describes larger images. Reconcile provenance, transformations, exclusions and group counts before final experiments. The official webpage itself contains inconsistent headline/table counts, so cite the actual audited release and explain differences rather than copying a headline number.

Research gap and novelty boundary
Pruning breast histopathology models is established work. Zhu et al. (2019) already studied compact CNNs, channel pruning and patient/image-level BreaKHis evaluation. Pouramin et al. (2026) study architecture-aware block pruning on PatchCamelyon. Neither applying L1 pruning nor adding reproducible engineering is sufficient on its own to claim a new algorithm.

Proposed question: Can validation-guided layerwise channel allocation reduce measured inference cost while preserving malignant sensitivity and balanced classification performance on unseen filename groups?

Potential contributions, conditional on evidence:

A layerwise pruning-selection procedure that optimizes measured latency under sensitivity and balanced-accuracy constraints.
An empirical analysis of how compression changes group-level errors across magnifications and tumor subtypes.
A reproducible benchmark separating masked sparsity, physical channel reduction, model storage, theoretical compute and observed runtime.
The novelty search performed for this plan is preliminary. Expand it around sensitivity-constrained pruning, hardware-aware pruning, medical model compression and risk-constrained model selection before claiming the method is novel.

Proposed method: work to implement
Let r be a vector of layer pruning ratios and M(r) the physically rebuilt model after a fixed fine-tuning budget. Select the lowest measured validation latency L(M(r)) subject to:

malignant sensitivity S(M(r)) >= S(M_dense) − delta_S;
balanced accuracy B(M(r)) >= B(M_dense) − delta_B;
any declared memory budget.
Use group-level metrics as the primary constraints and image metrics as secondary checks. Declare delta values before looking at outer test results. For an initial exploratory study, a two-percentage-point sensitivity margin can be examined, but it is a research tolerance, not a clinically justified safety threshold. With roughly 81 groups, narrow margins may be statistically unresolvable; report feasibility and uncertainty honestly.

Start with a bounded greedy search: remove a small additional channel fraction in one layer, run the same short adaptation budget, measure validation quality and device latency, and retain feasible candidates. Use stable L1 ranking within each layer initially so the comparison isolates allocation strategy. Fully fine-tune shortlisted candidates under equal budgets. Fix search trials and log every candidate. No test predictions, test labels or test timing-based model selection enter the search. Use a separate calibration subset or inner split if probability calibration or operating-threshold selection is added.

This procedure is a proposed extension. The current implementation supports per-layer ratio maps but does not implement this search or these constraints. Existing checkpoint selection uses validation accuracy; introduce configurable selection consistent with the stated research objective.

Experimental protocol
Re-audit the available source into a new experiment directory. Preserve historical manifests. Trace the 224×224 derivative to its original release, or restore the official source and define a documented transformation. Never interpret resized and original copies as independent datasets.
Build stratified five-fold outer splits over duplicate-linked filename groups. Within each outer training partition, reserve group-separated validation data. Keep all magnifications and exact duplicate-linked groups together. Audit class/subtype support in every fold.
Expose independent training seeds throughout the CLI, model initialization, loaders and saved provenance. The current public model commands do not expose a seed option. Use three training seeds per outer fold if resources permit; distinguish fold variation from initialization variation.
Train dense baselines under matched recipes. Select compression settings using only inner validation; evaluate once per frozen configuration on the corresponding outer test fold.
Compare models through paired out-of-fold predictions. Include a pooled-magnification experiment and separate per-magnification analyses. Cross-magnification transfer must also exclude test-group images from training at other magnifications.
Add a second suitable breast histopathology dataset for the stronger journal submission. First verify label, acquisition and grouping compatibility. BACH four-class labels are not automatically equivalent to BreaKHis binary labels. If tasks differ, perform a separate replication rather than calling it direct external validation.
Baselines and ablations
Comparison	Purpose
Dense custom VGG	Exact starting model for current channel surgery.
Uniform L1 weight masks	Distinguish zero weights from structural compression; expect unchanged dense MAC dimensions.
Uniform L1 channel pruning	Main existing structural baseline.
Random channel removal at matched widths	Test whether channel ranking contributes beyond topology reduction.
Identically narrow network trained from scratch	Test whether pruning beats simply choosing a smaller architecture.
Modern compact architecture, e.g. MobileNetV3 or EfficientNet-B0	Test whether pruning a large network is worthwhile at all; requires a supported implementation and matched training.
Strong established structured-pruning comparator	Compare with a reproduced contemporary method selected through the literature review.
Proposed constrained allocation	Test the incremental research contribution.
Add ablations for uniform versus learned allocation; accuracy-only versus sensitivity-constrained selection; no adaptation versus short adaptation versus full fine-tuning; random versus magnitude ranking; and initialization/pretraining policy. Compare at matched compute or observed latency budgets, not merely the same channel-removal percentage.

Screen uniform ratios such as 0.25, 0.50 and 0.75 on inner validation. Use a smaller pilot before the full matrix. For illustration, two architectures × five folds × three seeds requires 30 dense fits; three pruning ratios for each adds 90 fine-tuning jobs before search, magnification-specific models and other baselines. Estimate costs from real epoch timings before scheduling the expanded study. Current channel pruning applies only to custom VGG: broader structural experiments require additional implementations.

Outcomes and statistics
Primary outcomes: filename-group malignant sensitivity and balanced accuracy, together with batch-one median forward latency. Secondary outcomes: specificity, macro-F1, ROC-AUC, PR-AUC, Brier score/calibration, image metrics, per-magnification results, subtype-stratified malignant errors, model-state bytes, physical parameters, MAC estimates, p95 latency and peak inference memory.

Several secondary metrics, confidence intervals and peak-memory measurements require implementation. Preserve per-image probabilities and group assignments. Use paired group-cluster bootstrap confidence intervals for model differences, maintaining duplicate-linked groups as resampling units. Do not bootstrap images as independent patients. Out-of-fold bootstrap intervals are conditional on fitted models; they do not capture all training uncertainty. Report fold/seed variability separately and do not count 15 overlapping fits as 15 independent datasets. Predeclare a small number of primary comparisons and adjust exploratory multiple comparisons where appropriate.

Compare false-negative patterns at both the fixed argmax rule and a validation-selected operating point. Lock thresholds before each test fold. Report group denominators and uncertainty; do not claim noninferiority from a small observed difference alone. A formal noninferiority claim requires a justified margin and an adequately powered design.

For latency, hold hardware, runtime, precision, thread counts, input shape and batch size constant. Use independent timing sessions with sufficient warmup and repeats, randomized model order, and no competing workload. Report both CPU and GPU if available, or limit conclusions to the measured device. Separate resident forward time from end-to-end loading/preprocessing/transfer time. Report deployment state sizes separately from training checkpoints with optimizer payloads. Measure actual exported inference artifacts if claiming deployment storage savings.

Paper structure and planned exhibits
Introduction: deployment problem, precise gap, research questions and conditional contributions.
Related work: histopathology classification; structured/unstructured compression; hardware-aware selection; evaluation leakage and uncertainty.
Data and protocol: source lineage, audit, group definition, duplicate linkage, folds and preprocessing.
Method: channel surgery, constrained allocation, selection budgets and complexity.
Experimental setup: architectures, matched training, baselines, metrics and benchmark devices.
Results: predictive-performance/latency Pareto curves, constraint feasibility, group errors, magnification effects and ablations.
Discussion: practical benefit, failure cases, small-group uncertainty, source limitations and generalizability.
Conclusion and artifact availability.
Planned figures: workflow and data isolation; sensitivity–latency Pareto curves with uncertainty; per-magnification changes; layerwise retained widths. Planned tables: audited data counts; training/search settings; main quality/size/runtime comparison; ablations; malignant subgroup errors with support. Do not manufacture result values or write a results-bearing abstract before experiments.

Execution sequence and submission gates
Phase 1, approximately 1–2 weeks: literature matrix, source reconciliation, frozen protocol, seed/fold infrastructure and required metrics.
Phase 2, approximately 1 week: one complete full-data pilot and hardware cost estimate. Verify a competitive dense baseline before exploring compression.
Phase 3, approximately 2–4 weeks: constrained search, comparator implementations and pilot ablations.
Phase 4, approximately 2–4 weeks: frozen outer-fold experiments, replication dataset and repeated hardware measurements. Time depends on compute and search budget.
Phase 5, approximately 2 weeks: uncertainty analysis, figures, manuscript and reproducibility package.
These are planning estimates, not measured completion predictions. Prioritize VGG pilot experiments before funding a broad run matrix.

Submission gates: traceable data; zero cross-fold group/content overlap; completed full-data experiments; strong compact-model baseline; fair search budgets; meaningful effect or a well-supported generalizable negative finding; sufficient uncertainty reporting; independent reproduction from a clean environment; and a clear novelty comparison. A failure to beat a small dense model is informative and should change the paper's conclusion rather than be omitted.

For HEALTH, emphasize a computing contribution motivated by the cost and error tradeoffs in histopathology. For TIST, require stronger algorithmic generality across architectures and datasets. Journal fit is an editorial judgment, not an acceptance prediction. ACM scope pages returned access errors during this review; verify current submission format, review anonymity, article length and publication charges directly before preparing the submission. No current quartile, acceptance-rate or turnaround claims are made here.

Sources checked
Official BreaKHis description and original dataset citation.
Zhu et al., 2019: Breast cancer histopathology image classification through assembling multiple compact CNNs.
Pouramin et al., 2026: Architecture-Aware Block-Level Pruning Across CNN Models for Breast Cancer Histopathology Image Classification. Experiments described in the abstract use PatchCamelyon; do not characterize this as a BreaKHis experiment.
ACM's introduction of HEALTH and its multidisciplinary remit.
ACM Publications Board report containing the TIST charter. This supports broad fit, not current submission rules.
HEALTH journal and TIST journal: check current author instructions at submission time.
```
