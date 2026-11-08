# University research pilot protocol

Decision date: 7 November 2026. Research decision owner: Sutanu.
Status: finalized working protocol for the university pilot. This document
records the user's scope decisions and implementation defaults; it does not
claim a completed experiment or change any previously archived G0 result.

## Dataset and prior work

Use the supplied BreaKHis binary dataset at
`D:/Sutanu_WorkSpace/BreakHis/Dataset/dataset_cancer_v1/classificacao_binaria`.
The researcher declares that it came from the online BreaKHis dataset site,
with no renaming, resizing, augmentation or folder reorganization performed
by them. Accept this declaration for this study; comparison with an original
archive is not required. This is an operator declaration, not a byte-verified
claim that the local files equal the original public release.

Prior experiments occurred. Start fresh: do not reuse earlier checkpoints,
splits, predictions or tuning results. Use a new manifest and run identities.
Describe evaluation as held-out evaluation for this study, not historically
untouched data or independent external validation. The multiclass arrangement
is the same underlying dataset and is not an independent replication set.

## Fixed evaluation and selection rules

- Task: binary benign/malignant classification; malignant is positive.
- Grouping: preserve the complete filename grouping identifier, including
  suffixes, across all magnifications. Link exact raw/pixel duplicates before
  splitting and keep each connected component within one role. Use filename
  group terminology; patient identity is not established by filenames alone.
- Primary group score: average malignant probabilities within each available
  magnification, then average those magnification means equally. Secondary:
  average all images in the group. Do not impute missing magnifications.
- Evaluation target: five outer group-separated folds, subject to M1.2 support
  checks. Run one fold first as a disclosed pilot, not a five-fold result.
  If coverage is inadequate, document a revised split before fitting; do not
  silently fall back to image-level splits or select a favorable fold.
- Within each outer development portion, target component-disjoint 70/15/15
  train/selection/calibration allocations, stratified by class where feasible.
  Preserve components; exact percentages may vary. Check both-class support
  in selection, calibration and outer test. Unsupported roles require a
  documented design revision; no automatic reuse of selection for calibration.
- Fit only on training. Choose the epoch by highest selection group balanced
  accuracy at threshold 0.5; ties use lower selection loss, then earlier epoch.
- Report the fixed threshold 0.5. Separately calibrate a threshold on calibration
  groups using grid 0.01 through 0.99 in steps of 0.01, maximizing balanced
  accuracy; ties prefer higher sensitivity, then closest to 0.5, then lower value.
- For pruning candidates, apply the same selection access to all methods.
  At threshold 0.5 require sensitivity and specificity each within 0.02 of the
  corresponding dense reference on selection groups. Among feasible candidates
  choose lowest measured CPU latency, then higher balanced accuracy, then stable
  candidate ID. If none qualifies, report no feasible candidate. These are
  exploratory engineering tolerances, not clinical noninferiority margins.
- Outer-test labels, metrics and predictions must not influence training,
  stopping, candidate selection, threshold calibration or compute reductions.
  Final evaluation runs only after the corresponding development choices freeze.

## Pilot implementation defaults

- Primary architecture: repository custom eight-convolution VGG, not VGG16.
  Start from fresh random initialization, with no pretrained weights. Dense
  comparator first, then magnitude-pruning baseline; additional mechanisms
  follow the roadmap. Additional architectures are conditional on compute.
- Input: RGB, 224 by 224; scale channels to [0,1], normalize using mean/std
  0.5 for each channel. Training augmentation: independent horizontal and
  vertical flips, probability 0.5 each. Selection/calibration/test are deterministic.
- Training: RTX 5070 Ti; CPU for correctness and primary deployment timing.
  Initial batch size 32, loader workers zero. Adam, learning rate 0.0001,
  weight decay 0.0001, ordinary binary-task cross-entropy, no scheduler.
- Pilot budgets: 20 dense epochs, 5 adaptation epochs, 5 recovery epochs,
  maximum 8 pruning candidates per method. Adaptation/recovery are separate
  phases; record actual optimizer steps and time. These are caps, not completed
  work or guaranteed sufficient convergence. Match budgets across comparators.
- Seeds: split 101, training 102, initialization 103, loader 104, augmentation
  105, pruning 106, search 107, bootstrap 108, timing 109. Start with one fit
  seed; extra training seeds are conditional on measured compute and reported
  separately. Require deterministic settings where supported and record exceptions.
- CPU timing: batch one, two threads, eval/no-grad, resident forward only;
  50 warmups and 200 measured forwards in each of three sessions. Report median
  and session spread. GPU timing is secondary and uses explicit synchronization.
  Report end-to-end timing separately. Measure achievable latency first; do not
  invent an absolute millisecond budget before device measurements.
- Day-three compute review: estimate full cost from measured runs. If the 7-9
  day window cannot accommodate the full design, reduce architectures/search
  scope or report the one-fold pilot. Any fold/seed change is recorded before
  final evaluation, independently of outer-test performance.
- Save weights under the parent `checkpoint/binary` directory, outside the
  codebase, with distinct run/fold/method identities. Keep manifests, histories,
  resolved settings and scientific reports under the research artifact roots.

## Reporting and remaining implementation

Report group sensitivity, specificity and balanced accuracy with denominators,
image metrics secondarily, parameter count and measured CPU/GPU latency.
Use component-preserving paired bootstrap uncertainty (2,000 resamples) for
method comparisons where both-class support permits; disclose unsupported
resamples. A one-fold/one-seed pilot does not estimate across-fold/seed variability.
Literature review proceeds alongside implementation and must precede novelty
claims. No clinical safety, patient-independent or external-dataset claims.

The existing M0 archives remain immutable historical evidence. Their restrictions
are not cleared by editing this document. Current M1.1 supports synthetic
engineering and real metadata diagnostics only; real pilot loaders/fits need
an explicit versioned pilot permission route, updated scope bindings and tests.
M1.2 must establish feasible fold/role support. Neither that route nor the
remaining M1 packages are claimed implemented by this protocol finalization.

This pilot document supersedes conflicting provisional choices in the initial
protocol narrative for the pilot only. `protocol.initial.json` and the archived
G0 inputs retain their historical draft meaning; they are not silently promoted
to `pilot_ready`. Full-study/confirmatory policy remains a separate decision.

## Quick reference table

These are planned pilot settings, not experimentally validated results. This
table summarizes the rules above; it does not enable real training or change
completed M0/M1.1 work.

| Area | Current protocol |
| --- | --- |
| Task | Binary benign/malignant classification; malignant is positive. |
| Dataset | Local `dataset_cancer_v1/classificacao_binaria`; accept the researcher's declaration of download from the BreaKHis website without modifications by them. |
| Previous experiments | Start fresh; do not reuse earlier checkpoints, splits, predictions or tuning results. |
| Grouping | Keep complete filename groups and exact-duplicate-linked components together across magnifications and partitions. |
| Evaluation design | Target five outer group-separated folds; run one fold first as a pilot. M1.2 must verify support. |
| Development split | Target component-disjoint 70% training, 15% selection and 15% calibration within each outer development portion; preserve groups and both-class support. |
| Test separation | Outer-test results cannot influence training, checkpoint selection, pruning choices, threshold calibration or compute reductions. |
| Architecture | Repository custom eight-convolution VGG, initialized from scratch; additional architectures depend on compute. |
| Input | RGB, 224 by 224; scale to [0,1], then normalize each channel with mean/std 0.5. |
| Augmentation | Training only: horizontal and vertical flips, each with probability 0.5. |
| Training defaults | RTX 5070 Ti; batch 32; workers zero; Adam; learning rate 0.0001; weight decay 0.0001; cross-entropy; no scheduler. |
| Pilot budgets | Caps: 20 dense epochs, 5 adaptation epochs, 5 recovery epochs and 8 pruning candidates per method; match comparator budgets. |
| Checkpoint selection | Highest selection group balanced accuracy at threshold 0.5; ties use lower selection loss, then earlier epoch. |
| Group prediction | Primary: mean malignant probability within magnification, then equal mean over available magnifications. Secondary: ordinary image mean. No imputation. |
| Thresholds | Report fixed 0.5; separately calibrate on calibration groups over 0.01-0.99 in steps of 0.01, maximizing balanced accuracy. Ties: higher sensitivity, closest to 0.5, then lower threshold. |
| Pruning selection | At threshold 0.5, selection sensitivity and specificity must each be within 0.02 of the dense reference. Choose lowest CPU latency, then higher balanced accuracy, then stable candidate ID. Report no feasible candidate if none qualifies. |
| Seeds | Split 101; training 102; initialization 103; loader 104; augmentation 105; pruning 106; search 107; bootstrap 108; timing 109. Start with one fit seed. |
| Primary measurements | Group sensitivity, specificity, balanced accuracy and CPU latency, with class/group denominators. |
| Secondary measurements | Image metrics, image-weighted group aggregation, parameter count and GPU latency; disclose unsupported estimates. |
| Timing | CPU batch one, two threads; 50 warmups and 200 measured forwards in each of three sessions; resident forward separated from end-to-end timing. Synchronize GPU measurements. |
| Uncertainty | Component-preserving paired bootstrap, 2,000 resamples where class support permits; disclose unsupported resamples and pilot limitations. |
| Compute limit | Day-three cost review against the 7-9 day window; reduce scope independently of outer-test performance and disclose pilot-only results where applicable. |
| Checkpoints | Parent `checkpoint/binary`, outside the codebase; distinct run/fold/method identities. |
| Claims | Held-out evaluation within this study; no historically untouched-data, verified-patient, clinical safety or external-dataset generalization claim. |
| Implementation boundary | M0 and M1.1 remain complete. Real pilot fitting requires a versioned, tested pilot permission route; M1.2 checks fold/role feasibility. |
