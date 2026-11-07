# Research implementation roadmap: BreaKHis pruning

Prepared: 7 November 2026.
Status: proposed implementation and experiment schedule; no new scientific results claimed.
Progress update, 7 November 2026: M0 engineering implementation and real local
evidence generation are complete; [acceptance and G0 restrictions](../specs/m0/M0_6_RUNNER_GATE.md#progress-evidence).
M1 interface development is eligible. Real fits and confirmation remain blocked
pending the reported lineage/history/policy/review decisions; M1-M7 are not completed.
Planning window: 7-9 elapsed days from the agreed start, subject to measured compute costs. This is a scoped sprint, not a publication or successful-novelty guarantee.

This is the time-bounded implementation companion to the [end-to-end research guide](Breast_Histopathology_Pruning_End_to_End_Research_Guide.md) and [operational experiment plan](RESEARCH_PAPER_PLAN.md). The guide defines scientific intent; the operational plan supplies P0/P1/P2 and E01-E14 identifiers. This roadmap adds dependencies, concrete implementation deliverables, acceptance checks, compute limits and scope cuts. The older [codebase repair roadmap](../archive/CODEBASE_FIX_ROADMAP.md) concerns engineering reliability; its nine stages do not establish research completion.

## 1. Agreed scope and decisions to freeze

| Item | Sprint decision / status |
| --- | --- |
| Data location | `D:\Sutanu_WorkSpace\BreakHis\Dataset\dataset_cancer_v1\classificacao_binaria` |
| Primary task | BreaKHis binary benign/malignant; pool all four magnifications |
| Data breadth | BreaKHis only for this sprint; independent-dataset replication deferred, not required to start |
| Identity language | Filename-group/slide-separated unless patient identity is independently verified |
| Training and adaptation | RTX 5070 Ti; CPU execution remains available for correctness checks |
| Primary deployment objective | CPU batch-one median forward latency; proposed CPU thread setting 2, freeze after the timing pilot |
| Secondary deployment measurements | RTX 5070 Ti with a separately recorded fixed protocol; not an independently GPU-optimized search |
| Resolution | Retained 224 x 224 derivative as the working input; source/transformation provenance remains an audit gate |
| Primary group prediction | Proposed: mean probability within each magnification, then equal mean over available magnifications within the group |
| Secondary aggregation | Ordinary mean over all group images; report as a prespecified robustness comparison |
| Primary development model | Custom eight-convolution VGG; never label it VGG16 |
| Architecture controls | Add MobileNetV3-small as the compact dense comparator; retain ResNet18 as a standard dense baseline subject to the compute gate |
| Structured surgery | VGG first; dependency-aware ResNet/MobileNet surgery is outside the minimum sprint |
| Evaluation design | Aim for five group-separated outer folds; run size and seeds are frozen after group support and timing are audited |
| Novelty | Hypothesis: vulnerable-group protection adds benefit beyond constrained selection and malignant weighting; unverified |

The aggregation choice is a recommendation awaiting protocol freeze, not a claim of better accuracy. It has negligible additional cost compared with inference and prevents image-count imbalance across magnifications from setting implicit weights. Within each magnification images have equal weight; across magnifications the weights are equal; across groups each group has one metric observation. Training sampling/loss is a separate intervention and initially remains unchanged.

If a group lacks magnifications, average only the available ones, log the missingness, and report support. Do not duplicate or impute probabilities. For per-magnification results use only groups with that magnification and state denominators. Every method receives identical aggregation and development-threshold opportunities.

Before starting scientific fits, record in a versioned protocol: source lineage, grouping, magnification policy, folds, all seeds, pretraining recipes, epoch/candidate selection, thresholds, specificity safeguards, constraint margins, candidate budgets, adaptation/final recovery, CPU settings, statistics and stop rules. No invented clinical safety margin is assumed. A two-percentage-point exploratory tolerance cannot be treated as evidence of clinical noninferiority.

## 2. Existing foundation and implementation gaps

Existing supported foundation: canonical package, audited hashes and duplicate links, a grouped 70/15/15 split, VGG/ResNet18 binary training, L1 masks, VGG L1 channel surgery, saved configurations and resumable weights, test reports and model comparisons, synthetic CPU regressions and CI configuration. Historical test counts and smoke artifacts are execution evidence only; rerun relevant checks at the sprint start.

| Gap | Intended implementation location | Dependency |
| --- | --- | --- |
| Explicit scientific protocol and run ledger | Proposed `configs/research/`, `docs/research/RESEARCH_PROTOCOL.md`, `artifacts/research/` | M0 |
| Outer folds and inner roles | Proposed `breakhis/research/splits.py`; existing `manifest.py`, `data.py` | Audited inventory |
| Public independent seeds and epoch history | `cli.py`, `data.py`, `training.py`, `checkpoints.py` | Protocol/run schema |
| Development-only prediction exports | `reporting.py`; proposed research reporting interface | Partition permissions |
| Magnification-balanced pooling and operating thresholds | Proposed `breakhis/research/aggregation.py`, `selection.py` | Prediction schema |
| Selection constraints and malignant-weighted loss | `training.py`, `cli.py`, proposed `selection.py` | Tested metrics |
| Compact constructors and specified-width training | `checkpoints.py`, `models/`, training CLI | Seed/provenance support |
| Explicit retained channel indices and iterative ancestry | `pruning.py`, `checkpoints.py` | Existing VGG surgery |
| Vulnerability scoring and bounded search | Proposed `breakhis/research/vulnerability.py`, `search.py` | M1-M5 gates |
| Paired OOF analysis and repeated timing sessions | Proposed `breakhis/research/statistics.py`, `benchmarks.py` | Frozen methods/predictions |

Module/config names above are design targets, not existing supported APIs. Keep `breakhis/` canonical rather than moving the project into a new `src/` tree. Preserve current CLI/checkpoint behavior through explicit compatibility adapters and schema versions. Do not pass proposed flags before implementation/help/tests exist.

## 3. Milestones, schedule and hard gates

| Milestone | Target | Maps to operational plan | Deliverable | Proceed only when |
| --- | --- | --- | --- | --- |
| M0: Protocol, preservation and data | Day 1 | P0, E01 | Source snapshot, literature matrix, audit, decision register | No unexplained label/group leakage; derivative provenance stated |
| M1: Scientific data/reporting infrastructure | Days 1-2 | P1 | Fold/seed/aggregation/threshold/prediction interfaces | Hand-checkable metrics and no development access to outer test |
| M2: Dense baselines and cost gate | Days 2-3 | E02, E03, part E05 | Credible parent, compact/standard baseline pilot, timing estimates | Absolute quality and learning curves justify pruning; plan fits remaining budget |
| M3: Conventional controls and surgery | Days 3-4 | E04, P2, E05, E06 | Uniform/random/scratch/compact controls, A/B/C selection controls | Safe retained indices, equal budgets and useful latency region |
| M4: Vulnerability characterization | Days 4-5 | E07 | Signed group damage and stability analysis | Measurable, repeatable premise or documented negative result |
| M5: Candidate mechanism and decisive ablations | Days 5-6 | E08-E10 | Bounded group-aware mechanism plus A/B/C/D/E comparison | Advantage survives strongest feasible controls or claim is revised |
| M6: Freeze and held-out evaluation | Days 6-8 | E11; E12 only if feasible | Frozen code/protocol, fold-local fitting, OOF predictions | Fits selected using inner data only; no post-test redesign |
| M7: Statistics, runtime confirmation and handoff | Days 8-9 | E14 | Tables/figures, timing sessions, uncertainty, reproducibility package | Claims match completed evidence and limitations |

This sequence can overlap implementation, literature review and queued training, but dependent gates cannot be skipped. If limited to seven days, finish through a well-audited pilot and bounded mechanism analysis rather than claiming an unfinished five-fold study. Days 8-9 prioritize replication across folds, uncertainty and verification, not adding new methods after tests.

## 4. M0: preserve, audit and freeze the initial protocol

Implementation and acceptance contract: [M0 specification](../specs/m0/M0_SPEC.md).

Implementation and records:

- Record Git commit and dirty diff; snapshot relevant untracked source separately. Never identify a dirty experiment by commit hash alone.
- Recheck CPU/CUDA environments and regressions. Preserve current manifests, checkpoints and earlier inspection history.
- Audit the actual binary source into a new research directory. Record raw/decoded hashes, exclusions, dimensions, class/subtype/magnification support and duplicate components.
- Trace derivative acquisition/source/resize history. Locate original backup if available; record unresolved provenance explicitly. Do not combine original and resized copies as independent observations.
- Reconcile historical group/count discrepancies from evidence; retain filename suffixes unless authoritative metadata supports merging.
- Record every real-image group whose test predictions have already been inspected, including earlier smoke runs. The synthetic smoke is separate; previous real-image evaluation is not untouched confirmation.
- Build the nearest-literature matrix: ranking, allocation, class/sample balance, grouping, thresholds, adaptation, selection, latency and released code. Finish the closest-method review before claiming originality.

Existing command, to use a new output location:

```powershell
python -m breakhis prepare-data --dataset-root ../Dataset --sources cancer_binary --output artifacts/research/audit/v1-20261107 --seed 0
```

Acceptance checks:

- [ ] Audit accounts for included/excluded files and produces deterministic hashes.
- [ ] No unresolved mixed-label duplicate component enters experiments.
- [ ] Filename versus verified-patient terminology is explicit.
- [ ] Prior test exposure is recorded; exposed groups are development-only or resulting evidence is explicitly exploratory.
- [ ] Dataset release/derivative limitations and nearest-literature questions have owners in the decision register.

Gate G0: unresolved source/identity issues block strong clinical/patient or novelty claims. A derivative-only exploratory pilot may proceed only with its scope documented; it does not resolve those issues.

## 5. M1: implement scientific infrastructure

Development and acceptance contract: [M1 specification](../specs/m1/M1_SPEC.md).
Delivery order and package checklists: [M1 tracker](../specs/m1/README.md).
These planning documents do not start implementation or clear current G0 scientific restrictions.

### Folds and partition boundaries

Create immutable stratified outer fold assignments over duplicate-linked filename components. All magnifications and subtype images in a component move together. Inside each outer training partition assign inner training/development roles. Reserve separate calibration/shortlist confirmation groups only if support permits; otherwise declare development reuse as a small-data compromise.

Develop on designated pilot groups excluded from untouched confirmatory tests. If protecting all prior/pilot-exposed groups leaves insufficient five-fold support, revise the confirmatory design before results are seen: retain those groups as training/development-only, or report the BreaKHis study as exploratory. Do not silently reuse inspected groups as independent confirmation. Ordinary untouched K-fold groups appear once in held-out OOF predictions per method/seed; training-only exposed groups are outside that OOF denominator.

Acceptance: zero group/component/raw/pixel overlap across roles; deterministic same-seed folds; audit benign/malignant and subtype support; store assignment hashes and exclusions; reject infeasible folds clearly. Creating a new split seed does not erase previous test exposure.

### Seeds and run identity

Expose initialization/training, loader/augmentation, split, pruning and search seeds independently. Start with workers zero. Seed before model initialization, store effective settings and preserve exact restored RNG state on resume. Persist epoch curves, selection metrics, chosen epoch, code/source snapshot and parent hashes.

Acceptance: repeated controlled CPU runs agree; changed seeds affect intended components; resume retains the original run configuration/RNG; naming a run `s01` does not substitute for setting seed 1.

### Development predictions, pooling and selection

Export one row per image: source/image/raw hash, group/component, subtype, magnification, partition, outer fold, seeds, method, checkpoint/config hashes, true label, logits/probabilities, threshold identifier and protocol version. Keep development and held-out outputs in different directories.

Add validation-only reporting and benchmark-only entry points. Existing test-only `evaluate`/`compare` must not become development shortcuts. Add magnification-balanced pooling, balanced accuracy, sensitivity/specificity, FN/FP/support and frozen-threshold evaluation. Add AUC/PR-AUC/Brier with explicit absent-class rules where used.

Proposed threshold policy: retain fixed 0.5 as a reference; for each model use the same finite development-threshold grid, maximize sensitivity subject to a frozen specificity floor, then break ties by specificity and threshold closest to 0.5. Freeze the floor/grid before method comparison. Mark infeasible policies instead of relaxing floors. Threshold calibration uses only declared inner labels. This is a proposed policy to finalize at M1, not a clinical rule.

Keep epoch selection and pruning-candidate selection distinct. Proposed epoch selection is development group balanced accuracy at the fixed reference threshold, with earliest-epoch ties; candidate selection applies the frozen constrained operating-point rule. If another policy is chosen, record the exact replacement and give every baseline the same opportunities.

Acceptance: hand-computed pooling/threshold/metric fixtures pass; missing magnifications and ties are deterministic; no test labels, predictions or image iteration are available to development routines; per-image rows align with labels and metadata after reload.

Gate G1: a synthetic development-only end-to-end run passes before any scientific comparison.

## 6. M2: dense models and measured compute gate

Start with a full-data one-epoch execution pilot on the designated development design. Use pooled magnifications, 224 inputs, batch size 16 initially and workers zero. Measure epoch, evaluation, checkpoint, memory and startup costs. This is execution testing, not a competitive result.

Then establish a bounded credible recipe:

- Custom VGG parent: initial budget 20 epochs; inspect convergence rather than assuming twenty is sufficient. Limit recipe trials to a declared small grid.
- MobileNetV3-small: add supported constructor, exact pretrained-weight identity, binary head, state reconstruction and reporting. Give it a credible bounded development recipe.
- ResNet18: standard-backbone dense control; do not imply its channel surgery exists. Keep as a pilot if full-fold cost does not fit.
- Specify inherited/pretrained versus random-initialization policy per architecture. Do not give a pretrained compact comparator and untrained parent different hidden tuning budgets.

Acceptance: finite full-data fits/reload, plausible absolute malignant sensitivity and specificity, learning curves and recorded recipes, no outer-test inspection, compact constructor/reload tests, complete provenance.

Compute projection at Day 3:

```text
estimated GPU hours =
  sum(dense fits * dense epochs * measured dense epoch seconds)
  + sum(recovery jobs * recovery epochs * measured recovery epoch seconds)
  + sum(search candidates * adaptation steps * measured step seconds)
  + measured scoring/checkpoint overhead

remaining GPU allowance <= 70% of remaining wall-clock hours
```

Reserve the remaining time for evaluation, CPU search timing, failures and analysis. Budget CPU benchmarks separately; coordinate timing sessions to avoid competing jobs. Record actual GPU availability rather than assuming uninterrupted access. If projected cost exceeds the allowance, reduce declared scope before frozen testing.

Gate G2: freeze models, seeds, epochs, search trials, recovery budgets and the final run matrix from measured costs. Do not launch an exhaustive matrix first.

## 7. M3: conventional controls and reconstruction interface

Refactor channel ranking from VGG reconstruction. Accept explicit retained-index lists with bounds/uniqueness/minimum-width checks. Preserve original-parent channel ancestry through iterative surgery. Save truthful ranking/configuration metadata; do not call random or vulnerability ranking `l1-output-channel` to satisfy an old loader.

Acceptance checks:

- [ ] Zero-removal predictions match within declared tolerances.
- [ ] Random and specified retained indices propagate through Conv/BN/classifier.
- [ ] Parent weights/topology remain unchanged; every candidate starts from the same current parent.
- [ ] Iterative original-index mappings, save/reload and forward/backward pass.
- [ ] Fresh optimizer and controlled training/data RNG prevent candidate-order advantages.
- [ ] Unsupported residual/depthwise surgery fails clearly.

Run a bounded development frontier at initial channel removals 0.25/0.50/0.75, each from the same dense parent, with equal recovery (initially five epochs, revised only at the cost gate). Include a weight-mask control, random channels at identical widths, narrow-from-initialization training, and the compact dense model. Record total training/search costs separately from deployment costs.

Choose at most two CPU latency regions after the conventional pilot and before evaluating the proposed method. Match actual latency within a declared tolerance; report unattained targets and Pareto curves rather than forcing unequal-budget winners.

Implement A/B/C controls first: A and B share exactly the same conventional candidate pool; B changes only selection constraints. C1 adds malignant-weighted recovery, with weight 1 reproducing the unweighted case. C2 is an explicitly specified class-aware score/adaptation, not ordinary magnitude renamed as class-aware. Keep recovery and threshold opportunities equal.

Gate G3: conventional compression has practical headroom, or record that compact/narrow models dominate and revise the claim. A pruning paper is not justified merely by fewer parameters.

## 8. M4: characterize vulnerability before adding the mechanism

From fixed dense parents, temporarily remove small bundles in individual layers and log development effects before and after equal adaptation. Initial bundle size around 10% is exploratory; freeze the scored candidates and adaptation budget before comparisons.

For each malignant group export signed changes in within-group image loss, malignant probability, operating margin and correctness. Under primary magnification-balanced pooling, also report the matching pooled probability/margin; distinguish this from image-mean loss and from loss of an averaged probability. Each group contributes equally to across-group damage summaries. Monitor benign groups under analogous summaries.

Compare ordinary magnitude/gradient rankings against observed damage. Assess agreement before versus after recovery and stability across a bounded number of independent pilot fits. Report subtype/magnification denominators and retain negative damage (improvement). Heatmaps do not establish biological explanations.

Acceptance: aligned paired predictions, reproducible candidate resets, equal adaptation steps, signed damage, group-size balancing fixtures, and support/stability summaries.

Gate G4: if no stable concentration of malignant harm appears, stop the vulnerable-tail method branch or reframe the result. Do not add a complex score solely because the proposal named one.

## 9. M5: candidate mechanism and decisive comparisons

Implement the simplest mechanism justified by M4. Start with group-balanced mean malignant damage. Add a prespecified worst-damage tail only as a tested ablation. Monitor benign damage and the same specificity safeguards used by competitors.

The guide's damage/latency ratio is optional. If used, freeze weights/tail fraction, require a minimum meaningful positive latency gain, and reject noisy or nonpositive denominators. A constrained ranking without division is acceptable if better justified. No final score is predetermined by this roadmap.

Search requirements: fixed candidate increments/minimum widths, maximum evaluations, adaptation steps, dense/current reset point, score recomputation, ties, termination, shortlist length, failure handling and full recovery budget. Log every evaluated and rejected candidate, timing session and feasibility reason. Do not silently expand trials only for the new method.

| Comparison | What must be held equal | Question answered |
| --- | --- | --- |
| A vs B | Same conventional candidates/weights | Does selection alone explain benefit? |
| B vs C1/C2 | Constraints, target latency, recovery/tuning budgets | Does simple weighting/class awareness suffice? |
| B/C vs D | Constraints, data, thresholds, latency and budgets | Does group/vulnerable-case mechanism add value? |
| D vs E | Closest feasible established adaptive/class-aware method with faithful documented implementation | Is the mechanism competitive beyond our own controls? |
| L1 vs random | Same retained widths and recovery | Does ranking matter? |
| Pruned vs scratch/compact | Credible recipes; measured cost and quality | Is pruning practically useful? |

Acceptance: no test access, budget assertions, complete candidate history, reproducible selections, matched recovery, and ablations separating group balancing, tail protection and allocation. If E cannot be reproduced within the sprint, record that omission and keep novelty claims provisional; do not present an unvalidated adaptation as the published method.

Gate G5: freeze a limited set of promising methods only if the observed benefit is repeatable against strongest simple controls. Otherwise deliver the empirical finding and a revised method decision. Nonpositive results remain valid deliverables.

## 10. M6: freeze and execute held-out evaluation

Create a freeze record with code/source snapshot, data/split hashes, method definitions, all training/search seeds, recipes, constraints, thresholds, latency regions, budgets, aggregation and statistical comparisons. Record any excluded/exposed groups.

For every outer fold, initialize the method independently; fit/search/select/calibrate entirely inside that fold's declared training/development partitions. Generate held-out predictions only after selection. Never reuse a pruning topology or threshold learned using another fold's labels when those labels belong to this fold's test groups.

Preferred bounded expansion: five outer folds and one fixed training seed for the small frozen method set, if group support/time allow. Additional seeds are a robustness extension, not extra independent patients. Pilot seeds and confirmatory seed counts must be recorded separately. A second/third seed cannot replace missing held-out groups or a missing baseline.

Acceptance: one OOF prediction per eligible image/group/method/seed; all required folds accounted for; no overlap or leaked adaptation; original manifests/configuration/checkpoint hashes saved; infeasible runs included. If folds or comparators are incomplete, label the result partial/exploratory and list coverage instead of pooling a convenient subset as the planned study.

## 11. M7: statistics, benchmark confirmation and handoff

Implement paired cluster bootstrap differences using independent duplicate-linked groups/components. Keep methods paired; predeclare stratification and absent-class handling. Report denominators, false negatives, specificity, balanced accuracy and intervals, plus per-seed OOF results or a declared aggregation. Never bootstrap images as independent patients or duplicate the same groups across seeds as extra samples. OOF intervals are conditional on fitted models; report fold/seed variability separately.

Measure CPU batch-one latency as primary and RTX GPU latency as secondary in independent timing sessions with randomized model order. Freeze CPU threads, device/runtime/driver, precision, inputs, warm-up/repeats and session counts after a short timing pilot. Use adequate repeats and save all samples; synthetic smoke's one-repeat results are not scientific measurements. Separate resident forward time from preprocessing/transfer/aggregation/end-to-end time, and inference-only exported bytes from training checkpoint bytes. Define peak memory (allocated/reserved/process) before claiming memory savings.

Acceptance: hand-checkable statistical fixtures, paired group resampling, matched targets with uncertainty, raw timing/session metadata, explicit compute scope, and consistent export/reload predictions where exports are claimed. Numerical speedup without session stability is not a strong practical claim.

Final sprint handoff:

- [ ] Versioned protocol and decision/change log.
- [ ] Data lineage, exclusions, exposure register and immutable split hashes.
- [ ] Tested research interfaces and compatibility/CPU regression report.
- [ ] Baseline recipes, learning curves, checkpoints and parent provenance.
- [ ] Candidate/search history, adaptation cost and feasibility records.
- [ ] Development and held-out prediction records stored separately.
- [ ] A/B/C/D/E coverage table with omissions and scope stated.
- [ ] Group sensitivity/specificity/BA and latency frontiers with uncertainty.
- [ ] Compact/random/scratch controls and supported negative findings.
- [ ] Reproducible analysis commands, tables/figures, environment and source snapshot.
- [ ] Completed/partial/deferred status per experiment; no fabricated result cells.

A journal-ready manuscript is conditional on passing scientific gates; engineering completion alone does not establish novelty, clinical safety or journal acceptance.

## 12. Budget reductions and stop rules

| Trigger | Scope action |
| --- | --- |
| Source lineage cannot be resolved quickly | Derivative-only exploratory pilot with explicit limitation; block stronger data claims |
| Prior exposure leaves too few untouched groups | Training-only exposed groups or explicitly exploratory evaluation; no fake independent confirmation |
| Dense model is weak/unconverged | Repair recipe/data first; postpone new pruning mechanism |
| CPU latency does not improve with structural reductions | Revise widths/target budget or report negative practical outcome |
| Compact/scratch model dominates | Narrow the claim; do not hide the comparator |
| Simple B/C control explains the gain | Stop unsupported new-mechanism claim; analyze selection/weighting finding |
| Vulnerability rankings are unstable | Limit tail claims; test a simpler mean-damage mechanism or stop |
| Search recovery rankings do not survive final recovery | Revise screening within pilot budget; freeze only after validation |
| Projected run matrix exceeds Day-3 allowance | Reduce architectures/seeds/latency regions before freezing; keep essential controls |
| Deadline reaches Day 7 without frozen complete runs | Deliver tested infrastructure and pilot report; use Days 8-9 for audit/analysis rather than inventing final results |

Defer independent-dataset replication (E13), broader dependency-aware channel surgery, multiclass training, elaborate interaction searches, energy claims and export/runtime frameworks not actually tested. Include these as extensions, not hidden claims of completion. Missing strong comparators or full-fold results may keep publication-level conclusions provisional.

## 13. Run ledger and initial work order

Suggested research artifacts remain under `artifacts/research/`: `protocol`, `audit`, `splits`, `configs`, `predictions/development`, `predictions/heldout`, `search`, `benchmarks`, `analysis`, and `logs`. Version source protocol/config templates in the repository. Keep all checkpoints under the agreed parent `checkpoint/binary/` task tree, using IDs that encode experiment/fold/model/seeds/method and a unique suffix. Store hashes as authority; names are descriptive only.

Every ledger entry records: status, code commit/dirty snapshot, command/config schema, dataset/manifest/split hashes, role permissions, fold/seeds, architecture/pretraining, parent checkpoint hash, retained widths/original indices, ranking/allocation/selection/threshold policy, budget/actual cost, failure reasons, prediction locations, and hardware/session records.

Immediate implementation order:

1. M0 preservation/audit/exposure register and protocol template.
2. Public seeds, per-epoch history, development-only predictions and both pooling rules.
3. Threshold/selection metrics and immutable grouped outer/inner fold infrastructure.
4. A synthetic no-test-access scientific pipeline and compact model/specified-width support.
5. One real full-data development pilot and Day-3 cost gate.
6. Explicit-index surgery, conventional controls, then O1 vulnerability analysis.
7. Implement the mechanism only after the preceding gates support it.

This roadmap does not launch training, freeze unresolved scientific choices, create new CLI capabilities, or commit/push the current worktree. Those are separate execution steps.
