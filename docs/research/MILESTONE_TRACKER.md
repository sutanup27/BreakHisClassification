# Research milestone tracker

Updated: 8 November 2026.
Plan: [experiment roadmap](EXPERIMENT_ROADMAP.md).
Scientific objectives: [research guide](Breast_Histopathology_Pruning_End_to_End_Research_Guide.md).

This is the progress record for the five current milestones. Existing software
capabilities are not completed scientific experiments or established novelty.

| Milestone | Status | Already available | Completion target | Next action |
| --- | --- | --- | --- | --- |
| 1. Reproducible experiment foundation | Complete; M1.1-M1.5 reviewed | Existing workflow plus [master spec and package tracker](../specs/m1/README.md); validation-only development and exact epoch recovery | Verify the complete contract, including development/final isolation and reproducible recovery | Implement M2.3 pruning controls |
| 2. Dense and conventional pruning baselines | In progress; M2.1-M2.2 reviewed, M2.3-M2.5/experiments pending | Dense VGG/ResNet18 training, VGG channel pruning, weight masks, fine-tuning and comparison exports | Competitive trained baselines, relevant compact/scratch controls, matched budgets and a development-set detection/latency frontier | Implement [M2.3 pruning controls](../specs/m2/M2_3_PRUNING_CONTROLS.md) |
| 3. Malignant-group vulnerability analysis (O1) | Not started | Group metadata and prediction reporting can be reused | Controlled pruning probes establish whether conventional importance misses repeatable malignant-group damage | Define perturbations, per-group damage measurements and seed-stability analysis |
| 4. Proposed method and decisive comparisons (O2) | Not started | Structural-pruning primitives can be reused | Implement the candidate mechanism and isolate its benefit from selection rules, class weighting, topology and training budget | Use milestone 3 findings to specify the method and decisive comparator/ablation matrix |
| 5. Final validation and reporting (O3) | Not started; basic reports available | Metrics, saved predictions and resident-forward timing | Frozen evaluation, paired uncertainty, selected seed/fold repeats, repeated timing and claims bounded by feasible replication | Finalize the evaluation matrix after development choices are established |

## Current step

**Milestone 1 is Complete. M2.1-M2.2 are reviewed; next: implement M2.3.**
The [master spec](../specs/m1/M1_SPEC.md), [five-package tracker](../specs/m1/README.md)
and [alignment review](../specs/m1/ALIGNMENT_REVIEW.md) agree with the implemented
public workflow. All 25 acceptance cases have evidence: development uses
validation, final explicitly uses bound weights and test only, and comparisons
read saved phases without inference. Fresh-process dense, mask and channel
pipelines, failure/resume behavior and documentation are verified. Completion
is for the experiment foundation, not converged models or research novelty.

The [M2 package](../specs/m2/README.md) defines 15 requirements/22 planned
acceptance cases across matrix/recipes, initialization/models, pruning controls,
execution/budgets and frontier/handoff. M2.1 now compiles validated plans and
draft configs without fitting or test pixels; later model/control execution,
pilot measurements and scientific fits have not run. M2.2 additionally implements
compact, fresh continuation/scratch initialization and immutable parent acquisition;
its generated-image phases and resume checks are verified.

## Existing verification

- The final M1 CPU suite ran 135 cases (133 passed, 2 CUDA-only skips). Four new
  integration cases and a retained public dense/mask/channel workflow passed,
  including actual CLI failures/resumes and unavailable-role files. See M1.5.
- M1.4 separately passed all 23 focused cases in the CUDA environment, including
  actual CUDA-fit/CPU-final portability and synchronized CUDA timing.
- The generated-image workflow completed dense and channel-pruned/fine-tuned
  runs, exported results and reproduced dense weights for matching seeds.
- The one-epoch real-data run `baseline-real-check-20261107` completed on 40X
  data with GPU training and CPU evaluation. It is an execution check, not a
  converged baseline or evidence of the proposed method's benefit.
- Full trained-baseline comparisons, vulnerability experiments, the proposed
  method, uncertainty analysis and final research conclusions remain pending.

## Progress updates

Use `Not started -> In progress -> Tests passed -> Complete`.
For an experiment milestone, tests alone are insufficient: its declared
scientific experiment outputs and analysis must also exist before completion.
Record a concise summary of actual work, test/experiment results and remaining
gaps here; link detailed checklist specs and ordinary run outputs when available.
Keep this tracker and each milestone's package checklist consistent.

| Date | Update | Evidence/result |
| --- | --- | --- |
| 8 November 2026 | Established the five-milestone tracker and recorded existing capabilities | 63 active CPU tests passed; synthetic workflow and one-epoch real-data execution check available; no research novelty claimed |
| 8 November 2026 | Re-specified milestone 1 against the current code and research plan | Master, five checklist packages and alignment review; 17 requirements/25 planned acceptance cases; no new code implementation or model fitting |
| 8 November 2026 | Implemented/reviewed M1.1 config, storage and run identity | 77 CPU tests passed; real config preparation/revalidation succeeded without fitting or inference; next M1.2 |
| 8 November 2026 | Implemented/reviewed M1.2 full-source split validation and requested-role loaders | 92 CPU tests passed; 15 new role/leakage tests; real preparation/revalidation passed without fitting; next M1.3 |
| 8 November 2026 | Implemented/reviewed M1.3 development execution and committed-epoch recovery | 108 CPU tests passed; fresh-process VGG dense/channel development and seed reproduction passed; no automatic test inference; next M1.4 |
| 8 November 2026 | Implemented/reviewed M1.4 bound final inference, prediction attribution and saved phase comparisons | CPU: 129 passed/2 CUDA-only skips; CUDA: 23 passed; fresh-process VGG dense/channel both phases and JSON/CSV comparisons passed; next M1.5 |
| 8 November 2026 | Completed/reviewed M1.5 and Milestone 1 | 135 CPU cases: 133 passed/2 CUDA-only skips; public dense/mask/channel both phases, comparisons and real retries passed; legacy records unchanged; docs/help/links/HTML verified; next milestone 2 specs |
| 8 November 2026 | Specified M2 dense/conventional baseline/control milestone | Master, five ordered checklist packages, tracker and alignment review; 15 requirements/22 planned acceptance cases; no implementation/fitting; next M2.1 |
| 8 November 2026 | Implemented/reviewed M2.1 study preparation, matrix and capacity planning | 20 focused tests passed; full CPU suite 153 passed/2 CUDA-only skips; real metadata-only preparation and hashes verified; non-runnable drafts only, no M2 scientific fitting; next M2.2 |
| 8 November 2026 | Implemented/reviewed M2.2 initialization, immutable parent reuse and compact model | CPU 171 passed/3 CUDA-only skips; CUDA 19 focused cases passed; public generated VGG/continuation/scratch/compact both phases passed; no real-data fits; next M2.3 |
