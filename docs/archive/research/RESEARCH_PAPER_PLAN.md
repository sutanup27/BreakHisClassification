> Historical reference. The active plan is [simple experiment milestones](../../research/EXPERIMENT_ROADMAP.md).

# Step-by-step research execution plan

Updated: 2 November 2026. This replaces the old proposal with an operational plan based on the current code and the [End-to-End Research Guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md). Proposed features below are not implemented capabilities. No new scientific result or novelty claim is established.

## 1. Goal and sequence

**Question:** At matched inference latency and comparable specificity, does group-balanced malignant-vulnerability-aware channel pruning preserve malignant detection better than ordinary pruning with identical selection constraints?

Working title: *Group-Balanced Malignant-Aware Channel Pruning for Efficient Breast Histopathology Classification*. Use filename-group terminology until clinical identities are verified. ACM HEALTH is the intended target; a TIST direction needs broader methodological evidence. Recheck journal requirements at submission.

| Order | Experiment/prerequisite | Main decision |
| --- | --- | --- |
| P0 | Preserve environment/code; closest-literature review | Reproducible starting point? |
| E01 | Reconcile and audit source | Defensible data/grouping? |
| P1 | Seeds, validation reports, scientific controls | Development without test access? |
| E02 | Full-data execution pilot | Correct complete workflow? |
| E03 | Dense VGG baseline | Credible parent model? |
| E04 | Uniform pruning frontier | Useful compression region? |
| P2 | Retained-index surgery | Alternative rankings supported safely? |
| E05 | Random, scratch-width and compact controls | Practical reason to prune? |
| E06 | Selection and weighting controls | Do simple alternatives suffice? |
| E07 | Vulnerability analysis | Stable effect worth targeting? |
| E08 | Group-balanced mean-damage ranking | Group balance adds value? |
| E09 | Vulnerable-tail protection | Tail term adds value? |
| E10 | Adaptive allocation and strong comparators | Complete method adds value? |
| E11 | Frozen grouped outer evaluation | Internal generalization? |
| E12 | Standard-backbone replication | Architecture generality? |
| E13 | Independent-dataset replication | Dataset generality? |
| E14 | Deployment confirmation and paper | Which claims are supported? |

E02–E10 are exploratory development. Freeze the method before E11. Do not fund the full matrix before pilot gates pass.

## 2. Current implementation and gaps

Working directory: `D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification`.

| Code | Existing behavior | Needed work |
| --- | --- | --- |
| `breakhis/manifest.py` | Audit, hashes, duplicate linkage, one grouped 70/15/15 split | Separate fold generation |
| `breakhis/data.py` | Manifest datasets/transforms and loader seed parameter | Public independent training seeds |
| `breakhis/cli.py`, `training.py` | Binary VGG/ResNet18, SGD/cosine, validation-accuracy selection | Group-aware selection, weights, epoch history |
| `breakhis/pruning.py` | L1 masks; physical L1 custom-VGG channel removal | Alternative retained indices/scoring/search |
| `--layer-sparsities` | Complete ratio maps | Automatic allocation not implemented |
| `breakhis/reporting.py` | Prediction collection/group metrics/size/MAC/latency | Validation-only reports and probability exports |
| `evaluate`, `compare` | Reserved test split | Do not use during development |
| `prune.validation.json` | Validation accuracy/loss | Add group sensitivity/specificity/support |
| Checkpoints | Manifest/source/magnification/preprocessing checks | Fold-specific manifests and matching lineage |

Custom VGG has eight convolutions, not VGG16. ResNet18 physical channel surgery, compact CLI models and eight-class training are unsupported. Legacy scripts are not equivalent tested workflows. Smoke results demonstrate execution, not scientific accuracy.

Preserve worktree changes, images and historical artifacts. Use fresh IDs/directories. The Guide retains the earlier proposal in its historical appendix; this file is the operational plan.

## 3. Shared rules and records

1. Train/adapt on inner training data; rank/select/threshold using declared development data. Never tune with outer test labels/predictions.
2. Do not call test-only `evaluate`/`compare` in E02–E10. Time synthetic resident inputs or development images.
3. Keep all magnifications and duplicate-linked groups in one partition. Resample linked components together for statistics.
4. Start with pooled magnifications (`all`), 224 input and custom VGG. Analyze magnifications separately before training four models.
5. Start with mean image malignant probability per group, matching existing aggregation. Freeze image-weighted versus magnification-balanced pooling.
6. Report sensitivity alongside specificity, FN/FP, support and threshold. Give baselines equal threshold tuning; retain 0.5/argmax reference results.
7. Match actual latency and search/recovery budgets. Equal removal fractions are not equal compute.
8. Record infeasible candidates, slowdowns and all seeds. Small research tolerances are not clinical safety margins; extra seeds do not add patients.

Create proposed `artifacts/research/` directories: `protocol`, `audit`, `splits`, `configs`, `predictions`, `benchmarks`, `analysis`, `logs`. Checkpoints remain under supported `../checkpoint/binary/<magnification>/<architecture>/<experiment-id>/` paths. Example IDs: `e04-pilot-vgg-s00-ch50`, later `e11-f00-vgg-s00-methodB-budget1`.

For each run save code commit plus dirty/untracked-source snapshot, environment, command/config, data/split hashes, partition/fold/seeds, parent hash, original retained indices/widths, scoring/selection policy, epoch losses/metrics, threshold, adaptation/search/recovery cost, image probabilities with group/subtype/magnification metadata, and raw timing/device/size/MAC records.

## 4. P0 — Preserve the starting point

1. Inspect Git status; save modified/untracked source as well as commit identity.
2. Activate the existing environment and check device execution.
3. Run regressions before code changes; save tests, dependency/runtime versions, device identity and CLI help.
4. Build a closest-literature matrix from the Guide: ranking, allocation, class/sample weighting, grouping, thresholds, recovery budgets and code.
5. Read nearest hardware-aware and class/sample-aware methods fully; select manageable strong comparators. Novelty review must extend beyond ACM.

```powershell
Set-Location D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification
.\.venv313\Scripts\Activate.ps1
python --version
python tools/check_environment.py --device cuda
python -m unittest discover -s tests -v
python -m breakhis --help
```

Use CPU checking if CUDA is unavailable. **Outputs:** source/environment/test record and literature matrix. **Gate:** correct execution and passing relevant checks.

## 5. E01 — Audit the actual research data

**Question:** what images and independent groups exist?

1. Locate the retained binary source. Recheck historical 7,909 files, 7,784 decoded-unique images, 125 exclusions and 81 filename groups.
2. Trace the 224×224 derivative to its release/transformation. Restored originals require a separate audit/retraining; do not mix provenance.
3. Run a fresh audit:

```powershell
python -m breakhis prepare-data --dataset-root ../Dataset --sources cancer_binary --output artifacts/research/audit/v1 --seed 0
```

4. Inspect exclusions, conflicts, duplicate components, dimensions, classes/subtypes, group sizes and split/magnification support.
5. Investigate identity-count discrepancies; preserve filename suffixes unless authoritative metadata supports merging.
6. Record source/archive hash/acquisition date, citation/usage terms and transformations. Review suspected near-duplicates; exact hashes do not detect all of them.
7. Freeze inventory. Never treat resized/original copies as independent datasets.

**Outputs:** manifest, audit, exclusions and lineage note. **Gate:** traceable data and zero cross-partition group/content overlap. Existing assignments support a pilot, not five-fold evaluation.

## 6. P1 — Minimum scientific infrastructure

These are required code additions, not existing commands. Document and test before scientific comparisons.

### P1.1 Seeds and epoch history

1. Add a public training seed in `cli.py`; pass it to `get_split_loaders` and seed before initialization. Keep split seed separate.
2. Save seeds in configs/checkpoints and preserve restored RNG state on resume.
3. Start with zero workers; add worker/generator seeding if workers are introduced.
4. Persist losses, group/image development metrics, timing, selection criterion and selected epoch.
5. Test controlled CPU reproducibility, different seeds and resume behavior.

### P1.2 Development-only reporting

1. Reuse prediction/group reporting on validation rows, evaluation transforms and nonshuffled loaders.
2. Construct only the needed split where practical; verify no test iteration.
3. Export probabilities with image/group/component IDs, subtype and magnification.
4. Add balanced accuracy, locked-threshold sensitivity/specificity, FN/FP and support; secondary AUC/PR-AUC/Brier summaries need explicit absent-class handling.
5. Add benchmark-only reporting independent of test classification.
6. Test alignment, groups, uneven batches, ties, absent classes and no test access.

### P1.3 Selection and controls

1. Keep legacy accuracy selection; add group-balanced-accuracy epoch selection with fixed ties.
2. Separate epoch selection from candidate selection and log both.
3. Add candidate selection under sensitivity/specificity constraints against a frozen dense development reference. Report infeasible instead of silently relaxing constraints.
4. Add malignant training-loss weighting and verify weighted-loss normalization/reporting.
5. Add specified-width VGG training and compact-model creation/checkpoint reconstruction.
6. Expose group metadata to scoring without breaking training compatibility. Keep training sampling fixed initially; balanced scoring and balanced training are separate interventions.

**Output:** tested research interfaces. **Gate:** synthetic validation-only end-to-end report with correct predictions and no test access.

## 7. E02 — Full-data execution pilot

1. Use E01 manifest, pooled magnifications, batch size 16 and zero workers.
2. Train one epoch with a fresh ID; this checks execution, not quality.
3. Fresh-process reload and P1 development reporting.
4. Prune 50% channels, recover one epoch and reload/report again.
5. Verify finite outputs, BN/downstream/classifier compatibility, class/provenance preservation and lower physical parameters/MAC dimensions.
6. Verify zero-ratio surgery preserves predictions within existing tolerances; equivalence is not expected at 50% removal.
7. Record epoch time/resource use for budgeting.

Current supported execution commands:

```powershell
python -m breakhis train --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --model vgg --epochs 1 --batch-size 16 --num-workers 0 --device auto --experiment-id e02-pilot-vgg
python -m breakhis prune --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --checkpoint ../checkpoint/binary/all/vgg/e02-pilot-vgg/train.pth --method channel --sparsity 0.5 --experiment-id e02-pilot-ch50
python -m breakhis finetune --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --checkpoint ../checkpoint/binary/all/vgg/e02-pilot-ch50/prune.pth --epochs 1 --learning-rate 0.0001 --batch-size 16 --num-workers 0 --experiment-id e02-pilot-ch50
```

These use current default seed/selection behavior. Set P1 options explicitly for later scientific runs once implemented; do not append invented flags now. Do not call test evaluation here.

**Outputs:** checkpoint chain, development reports, reload checks, resource estimate. **Gate:** real-data execution correctness.

## 8. E03 — Competitive dense VGG baseline

**Question:** is the starting model strong enough for meaningful pruning research?

1. Freeze pilot input/augmentation/initialization, optimizer/scheduler, checkpoint criterion and group aggregation.
2. Start with the current SGD recipe and a planned 20 epochs. Inspect curves; 20 is an initial budget, not proof of convergence.
3. If needed, perform a bounded development recipe search, for example learning rates 0.01, 0.001 and 0.0001 with equal epoch budgets. Record all attempts and adjust the grid to measured compute resources.
4. Choose the recipe using development only. Run the chosen recipe with training seeds 0, 1 and 2 via P1; do not select the best seed.
5. Save image/group probabilities, FN/FP lists, support and per-magnification summaries.
6. Define a development-threshold rule with a specificity safeguard; retain 0.5 results. Select the safeguard before method comparisons, considering small benign-group support.
7. Freeze this recipe and dense references for E04–E10.

Current-CLI training template:

```powershell
python -m breakhis train --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --model vgg --epochs 20 --learning-rate 0.001 --batch-size 16 --num-workers 0 --device auto --experiment-id e03-pilot-vgg-s00
```

An `s00` name does not set a seed. Use P1's implemented seed/selection controls for scientific fits and save their resolved values.

**Outputs:** dense parents, learning curves and fixed recipe. **Gate:** credible absolute sensitivity/specificity and convergence. Repair training/data issues before inventing pruning.

## 9. E04 — Uniform channel-pruning frontier

**Question:** where does conventional pruning become harmful, and does it accelerate inference?

1. Independently prune each E03 dense parent at 0.25, 0.50 and 0.75 channel removal. Start every ratio from dense; do not chain ratios.
2. Generate P1 development reports immediately after surgery.
3. Fine-tune every candidate equally: initially five epochs at 0.0001 with identical schedule/selection.
4. Report pre/post-recovery group sensitivity, specificity, balanced accuracy, widths, parameters, state bytes, MACs and resident-forward timing.
5. Add one 50% unstructured-mask control from the same parent. Do not equate weight sparsity with channel-removal ratio or matched compute.
6. Plot quality versus latency. Choose at most two feasible target latency regions for mechanism experiments; retain unreachable targets in records.
7. If recovery is inadequate, perform one bounded recovery-budget pilot on representative candidates, then freeze an equal recovery budget for subsequent methods.

Example existing commands for one branch:

```powershell
python -m breakhis prune --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --checkpoint ../checkpoint/binary/all/vgg/e03-pilot-vgg-s00/train.pth --method channel --sparsity 0.5 --experiment-id e04-pilot-vgg-s00-ch50
python -m breakhis finetune --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --checkpoint ../checkpoint/binary/all/vgg/e04-pilot-vgg-s00-ch50/prune.pth --epochs 5 --learning-rate 0.0001 --batch-size 16 --num-workers 0 --experiment-id e04-pilot-vgg-s00-ch50
```

Repeat with fresh IDs/parent paths for each ratio/seed. Use P1 reporting; no test evaluation.

**Outputs:** conventional development frontier and selected budgets. **Gate:** useful compression regions or an explicit negative practical finding.

## 10. P2 — Safe retained-index pruning

Complete before E05's random ranking and E08. Current ratio maps cannot select arbitrary filters: `prune_vgg_channels` chooses L1 indices internally.

1. Factor ranking from reconstruction; accept validated retained-output index lists while preserving the existing L1 wrapper.
2. Validate every layer before rebuilding: no duplicate/out-of-range/empty indices, masks or unsupported architectures.
3. Reuse BN/downstream/classifier copying. Preserve original-parent index mappings through iterative pruning.
4. Extend versioned checkpoint metadata and loader validation. Current configuration assumes `l1-output-channel`; never label a new ranking as L1 just to pass checks.
5. Test zero-removal equivalence, random selections, iterative mappings, parent immutability, serialization and forward/backward/reload.
6. Reset each candidate from the identical current parent with fresh optimizer and controlled RNG/data stream. Candidate evaluation order must not confer inherited training.

**Outputs:** validated reconstruction interface and truthful metadata. **Gate:** arbitrary supported selections reconstruct safely.

## 11. E05 — Simpler small-model alternatives

**Question:** is pruning worthwhile beyond selecting a small model?

1. Remove random channels at exactly E04 layer widths, using explicit pruning seeds. Recover under the same budget.
2. Train the same narrow topology from initialization with the dense training budget. Report total training cost separately from inherited-weight recovery.
3. Integrate/train a compact baseline, initially MobileNetV3-small. Match source/resolution/threshold policy and give it a bounded credible recipe-tuning budget.
4. Compare at nearby measured latency budgets. Report mismatches/frontiers instead of declaring unequal-budget winners.
5. Repeat promising comparisons across training/pruning seeds; separate ranking variation from initialization variation.

**Outputs:** random-ranking, narrow-scratch and compact-model controls. **Gate:** establish when pruning is practically useful. If compact models dominate, narrow the claim or redirect.

## 12. E06 — Separate selection and malignant weighting

**Question:** can simple selection or weighting explain the benefit?

| Variant | Mechanism | Candidate selection |
| --- | --- | --- |
| A | Conventional magnitude ranking and fixed candidate pool | Group balanced accuracy at target budget |
| B | Exactly the A candidates | Sensitivity/specificity constraints at the same budget |
| C1 | B ranking/allocation; malignant-weighted recovery loss | Same constraints as B |
| C2 | Explicit class-weighted importance or faithful class-aware adaptation; unchanged recovery loss | Same constraints as B |

1. Freeze constraints relative to dense development performance, threshold policy, budget and tie handling; also report absolute quality.
2. Execute A/B on the identical candidate pool to isolate selection only.
3. Implement C1 first. A small pilot weight grid such as 1, 2 and 4 is sufficient initially; weight 1 must reproduce the control. Record tuning costs.
4. Add C2 with a specified published/adapted score; ordinary magnitude is not class-aware importance.
5. Compare at the E04 latency regions and equal recovery/threshold tuning opportunities.
6. If the proposed method later receives more trials, provide comparable baseline tuning or explicitly analyze the imbalance.

**Outputs:** A/B/C table and complete candidate history. **Gate:** identify the strongest simple control; the proposed mechanism must improve on it, not merely A.

## 13. E07 — Characterize group vulnerability

**Question:** do ordinary importance scores hide concentrated malignant-group harm?

1. From a fixed E03 parent, perturb one VGG layer at a time by removing a small bundle, initially around 10% of that layer, leaving other widths unchanged.
2. Reset each perturbation to the same parent and record original channel IDs.
3. Save development predictions before removal, immediately after and after an equal short adaptation budget, initially one epoch.
4. For each group calculate signed changes in mean image cross-entropy, mean malignant probability, margin and correctness at the locked threshold. Mean image loss and loss of mean probability are different quantities; use the former initially.
5. Retain negative damage values and inspect benign damage too.
6. Compare magnitude/other importance with observed damage. Repeat across pilot seeds to assess vulnerable-group/layer stability.
7. Summarize subtype/magnification patterns with denominators; heatmaps alone do not justify biological explanations.
8. Check whether immediate damage ranks predict damage after recovery. Poor agreement means cheap pre-recovery scoring may be unreliable.

**Outputs:** signed group-damage records, layer plots, stability and recovery-rank analysis. **Gate:** reproducible pattern worth targeting. If absent, reject the premise or reframe the study.

## 14. E08 — Group-balanced mean-damage ranking

**Question:** does equal group weighting improve channel decisions beyond class weighting?

1. Hold allocation fixed at an E04 width vector. Change ranking only so allocation cannot explain the first result.
2. Construct scoring/candidate pools from training data or a fixed training subset using evaluation transforms; reserve development for acceptance/selection. If scoring uses development labels, explicitly declare/log adaptive reuse and its budget.
3. Start with mean malignant-group damage and a fixed benign-damage safeguard. Do not add tail protection yet.
4. Compare image-weighted damage, class-balanced image damage and group-balanced damage with identical candidates/recovery.
5. Specify bundle-based greedy selection: bundle size, candidate proposal, minimum widths, ties and rescoring after accepted removal. Individually scored channels interact; do not assume their damage sums exactly.
6. Reach the fixed target widths and run equal full recovery; apply B/C's threshold and constraints.
7. Compare at matched observed latency and comparable specificity across pilot seeds.
8. Record scoring/search cost, not only final inference cost.

**Outputs:** first complete mechanism and weighting ablation. **Gate:** repeatable added value over E06's strongest control. If group balance alone suffices, keep it simple.

## 15. E09 — Vulnerable-tail protection

**Question:** does protecting the worst-damaged malignant groups add value beyond mean damage?

1. Define `d_g(c)` as signed change in mean image loss for group g after candidate removal c.
2. Define the mean damage and mean damage in the worst q fraction of malignant groups. Specify rounding and minimum support.
3. Pilot a tiny grid, for example q = 0.25 or 0.50 and tail coefficient 0 or 1, only when group counts support it. Coefficient zero must recover E08.
4. Hold benign safeguard, allocation, candidates, thresholding and recovery fixed.
5. Compare mean-only/mean-plus-tail quality, group errors, seed stability and scoring cost.
6. Inspect whether the tail repeatedly chases a few development cases or changes erratically across fits.
7. Reject the tail term if no stable benefit appears; complexity is not novelty.

Avoid an unguarded damage/latency ratio: tiny noisy timing gains create unstable rankings. Initially rank damage under a resource budget and require repeatably measurable speed improvement when accepting speed-based steps.

**Outputs:** tail ablation and retain/reject decision. **Gate:** reliable incremental value without excessive development overfitting.

## 16. E10 — Allocation and established comparators

1. Compare uniform/adaptive allocation while holding E08/E09 ranking fixed.
2. Use bounded layerwise search: propose safe widths, rebuild each candidate from the same current parent, adapt equally, evaluate development feasibility and measure latency.
3. Specify increments, minimum widths, dependency groups, candidate budget, resets, ties, stopping and infeasibility. Small width changes need not improve real kernels.
4. Log every rejected/accepted candidate. No outer-test tuning.
5. Reproduce/adapt a strong hardware-aware comparator and a class/sample-aware comparator selected in P0. State deviations and recovery/distillation differences.
6. Match candidate trials, adaptation/final steps, threshold tuning and hyperparameter budgets as far as possible; disclose residual cost differences.
7. Compare complete method against B/C and established controls at at most two initial latency regions.
8. Run focused ablations: fixed/adaptive allocation, image/group weighting, mean/tail, fixed/reassessed ranking. Avoid a large factorial grid without a question.

**Outputs:** specified final algorithm, development frontiers and cost table. **Gate:** proceed as an algorithm paper only if the mechanism adds stable value beyond weighting/selection; otherwise reframe or redirect.

## 17. E11 — Frozen grouped outer evaluation

**Question:** do findings hold on groups excluded from each fit's training and selection?

### E11.1 Generate compatible folds

1. Add a fold-generation tool using E01 accepted records. Split over duplicate-linked components, stratify binary label and audit subtype support.
2. Aim for five disjoint outer test folds if support permits. Reserve group-separated development data inside each outer training partition.
3. Write one immutable manifest per fold using existing `train`, `validation`, `test` roles so loaders remain reusable. Save parent inventory hash and generation config.
4. Keep all magnifications together. Test zero group/content overlap and each eligible component appearing once in outer test.
5. Use each fold's manifest hash throughout its checkpoint lineage. Do not load fold-0 checkpoints under fold-1 metadata.

### E11.2 Declare prior development exposure

Pilot development labels have influenced algorithm design. CV over that population is not a wholly untouched confirmation of design decisions. List exposed groups and describe E11 as internal validation conditional on a frozen protocol. Do not revise the method after inspecting fold tests. Preserve a genuinely untouched dataset for E13 where possible. If requiring a permanent holdout, reserve it before development and exclude it from all pilot work and CV design.

### E11.3 Freeze and run

1. Freeze source snapshot, folds, algorithm, training/checkpoint/threshold policies, margins, targets, search/recovery budgets and analysis.
2. Use three initialization seeds per fold if feasible: five folds × three seeds = 15 dense fits for one backbone before pruning/search.
3. Train each dense parent only on its fold training partition. Compare inherited-weight methods from the same parent, resetting candidate adaptation fairly.
4. Fit thresholds/calibration using that fold's development data only. If separate search/calibration subsets are infeasible because of group counts, disclose adaptive reuse rather than pretending independence.
5. Lock model/topology/threshold before generating test predictions. Evaluate frozen configurations once and retain all probabilities/metadata.
6. Existing reports use argmax; use an added threshold-aware report for the primary locked operating point.
7. Current `compare` is for a matching dense/pruned/fine-tuned chain and enforces topology/mask compatibility. Do not use it to rank arbitrary A/B/C/D topologies without deliberately extending its contract. Report each method and join by fold/group IDs.
8. Keep per-seed out-of-fold results. Repeated predictions of the same group are not extra patients. A seed ensemble is an additional method with additional inference cost.

### E11.4 Analyze uncertainty

1. Predeclare primary endpoints and a few paired method contrasts at matched budgets.
2. Pair predictions on identical held-out groups. Bootstrap groups or duplicate-linked components while preserving method pairing.
3. Define absent-class resample handling; class-stratified group resampling may be appropriate for sensitivity/specificity and must be disclosed.
4. Report differences, intervals, FN/FP counts and denominators. OOF bootstrap uncertainty is conditional on fitted models, not all training/split uncertainty.
5. Report fold and seed variation separately. Fifteen overlapping fits are not fifteen independent datasets.
6. Control confirmatory comparison multiplicity; label sparse subtype/magnification analyses exploratory.
7. Do not infer equivalence, noninferiority or clinical safety from nonsignificance or a small point estimate.

**Outputs:** frozen held-out predictions, paired estimates, uncertainty and complete run matrix. **Gate:** no post-test revision presented as confirmation; revised methods require a disclosed new evaluation cycle.

## 18. E12 — Replicate on a standard backbone

**Question:** is the mechanism specific to custom VGG?

1. Keep VGG as the first primary study. Add torchvision ResNet18 because dense training already exists.
2. Implement dependency-aware physical surgery across residual/downsample paths. Current ResNet18 channel pruning is rejected; masks are not a replacement for structural replication.
3. Add topology reconstruction/provenance and forward/backward/reload tests; confirm actual shape/parameter/MAC reduction.
4. Apply frozen ranking and strongest controls with a declared backbone-specific development budget. Do not redesign around held-out outcomes.
5. Match actual latency regions and training policies. Document pretraining; give competitors equivalent access rather than comparing pretrained against scratch unfairly.
6. Report positive and negative replication conditions. A standard sequential backbone is a lower-surgery-cost alternative, but still requires proper CLI/checkpoint integration.

**Outputs:** second-backbone frontier and held-out comparisons. **Gate:** architecture generality claims proportional to evidence.

## 19. E13 — Replicate on an independent dataset

**Question:** does the mechanism extend beyond BreaKHis?

1. Review a suitable accessible dataset such as BRACS: official grouping metadata, reference splits, acquisition and labels. This plan does not certify current access/download availability.
2. Write a separate task protocol. Do not silently merge atypical, in-situ, invasive and benign labels to mimic BreaKHis.
3. Add a source-specific manifest adapter and class mapping with tests. Audit official grouped partitions and preserve them when suitable.
4. Train the dataset's dense models, then apply the frozen mechanism and strongest controls. Predeclare any permitted development adaptation.
5. Evaluate untouched partitions after selection using the correct grouping/statistics.
6. Call retraining and repeating the pruning method methodological replication. Training on BreaKHis and directly testing another compatible source is external transfer validation, a different experiment requiring justified task/preprocessing alignment.

**Outputs:** independent audit, task mapping, replication results and limitations. **Gate:** broader claims depend on results; resized copies of one source are not independent validation.

## 20. E14 — Confirm deployment benefit and write the paper

### E14.1 Final benchmark experiment

1. Specify device/workload, initially batch-one field/ROI classification. Desktop-GPU measurements do not establish edge suitability or whole-slide diagnosis.
2. Fix input shape, precision, runtime, thread counts, device settings and export. Separate resident-forward time from preprocessing/transfer/end-to-end time.
3. Use a pilot to choose warmup/repeats; 20 warmups and 100 forwards are initial settings, not universal adequacy guarantees.
4. Run at least three independent timing sessions with randomized model order and no competing workload. Within-session iterations are not independent hardware replications.
5. Save every timing sample, median/p95 and session variability. Validate outputs outside the timed interval.
6. Separate model-state serialization from training checkpoints containing optimizer/RNG state. Measure actual exported files when claiming exported storage savings.
7. Define peak memory precisely. Measure energy if claiming energy savings; MAC reduction does not establish energy or latency benefit.

Current supported final-test syntax template:

```powershell
python -m breakhis evaluate --dataset breakhis --source cancer_binary --manifest artifacts/research/audit/v1/manifest.json --magnification all --checkpoint ../checkpoint/binary/all/vgg/e04-pilot-vgg-s00-ch50/finetune.pth --benchmark-batch-size 1 --benchmark-warmup 20 --benchmark-repeats 100 --output artifacts/research/analysis/example-final-test.json
```

This illustrates syntax; **do not test the pilot now**. For E11 replace manifest/checkpoint with matching frozen fold artifacts. Default metrics are argmax; add primary threshold-aware analysis. Set and record actual device/CPU threads for final timing sessions. Benchmark-only P1 support is preferable for repeated timing without repeated test classification.

### E14.2 Manuscript artifacts

1. Data table: lineage, exclusions, groups, classes, magnifications and split units.
2. Method diagram: training/scoring/development/test roles and candidate reset flow.
3. Main table: held-out sensitivity/specificity/balanced accuracy, FN/FP, uncertainty and matched latency.
4. Pareto curves including compact models and strongest pruning controls.
5. Ablations separating selection, weighting, group balance, tail and allocation.
6. Signed vulnerability/stability analysis with subtype/magnification denominators and failures.
7. Efficiency table: physical parameters, MACs, inference bytes, median/p95, memory and search/recovery costs.
8. Architecture/dataset replication tables distinguishing replication from transfer.
9. Limitations: identity surrogates, small independent sample, development exposure, hardware specificity, compute constraints and no clinical deployment validation.
10. Reproduction package: environment, source snapshot, permitted split/prediction artifacts, configs and executable analysis. Redistribute images only under source terms.

Write the results-bearing abstract last. Describe completed effects and uncertainty. Recheck ACM scope, submission format/anonymity, publication charges and policies directly at submission.

## 21. Budget and stopping rules

Estimate after E02/E03: **dense fits + candidate scoring + search adaptation + full recovery + compact/scratch controls + repeated timing**. Epoch count alone understates search cost.

Pilot: one source, pooled magnifications, custom VGG, one development split; one seed for debugging, then three seeds for promising comparisons. One-seed screening is exploratory, not robustness evidence.

Primary full study: five folds × three seeds × one backbone = 15 dense parents. Every method/target adds search/recovery per parent. Budget E12/E13 separately; do not immediately multiply by four magnifications and large grids.

Redirect when:

- Source/grouping cannot support the claim: repair provenance first.
- Dense quality is poor: fix baseline training before pruning research.
- Physical reduction does not accelerate the runtime: revise budget/topology/device or report that limitation.
- Compact/scratch controls dominate: narrow the practical claim.
- B/C explain all gains: do not credit a new ranking mechanism.
- Mean-only matches the tail: remove the tail term.
- Effects depend on one seed/few groups: report instability.
- Closest literature already implements the mechanism: reproduce, distinguish substantively or change the contribution.

## 22. Execution checklist

- [ ] P0: preserve source/environment/tests and closest-literature matrix.
- [ ] E01: reconcile source and freeze audit.
- [ ] P1: seeds, validation reports/exports, selection and control infrastructure.
- [ ] E02: real-data execution/reload pilot without test inspection.
- [ ] E03: credible dense recipe and pilot parents.
- [ ] E04: uniform pruning/recovery frontier and feasible budgets.
- [ ] P2: retained-index surgery and truthful versioned metadata.
- [ ] E05: random, narrow-scratch and compact controls.
- [ ] E06: isolate constrained selection and malignant weighting.
- [ ] E07: establish or reject vulnerability premise.
- [ ] E08: group-balanced mean-damage ranking at fixed allocation.
- [ ] E09: retain tail only if supported.
- [ ] E10: isolate allocation and compare strong methods fairly.
- [ ] E11: freeze protocol, execute grouped evaluation, disclose development exposure.
- [ ] E12: validated standard-backbone replication.
- [ ] E13: independent-source replication.
- [ ] E14: repeated deployment measurements, statistics and paper artifacts.

The first substantive milestone is **E04**, a credible dense-to-pruned frontier. The first test of the proposed contribution is **E08 against E06's strongest control**. Complete these before funding the broad study.
