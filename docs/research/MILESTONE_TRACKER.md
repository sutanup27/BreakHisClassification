# Research milestone tracker

Updated: 8 November 2026.
Plan: [experiment roadmap](EXPERIMENT_ROADMAP.md).
Scientific objectives: [research guide](Breast_Histopathology_Pruning_End_to_End_Research_Guide.md).

This is the progress record for the five current milestones. Existing software
capabilities are not completed scientific experiments or established novelty.

| Milestone | Status | Already available | Completion target | Next action |
| --- | --- | --- | --- | --- |
| 1. Reproducible experiment foundation | In progress; specification complete, acceptance pending | Existing workflow plus [master spec and five checklists](../specs/m1/README.md) | Verify the complete contract, including development/final isolation and reproducible recovery | Start M1.1 config/storage/identity implementation and revalidation |
| 2. Dense and conventional pruning baselines | Not started; supporting tools available | Dense VGG/ResNet18 training, VGG channel pruning, weight masks, fine-tuning and comparison exports | Competitive trained baselines, relevant compact/scratch controls, matched budgets and a development-set detection/latency frontier | Start after milestone 1 acceptance; specify the baseline matrix and training recipe |
| 3. Malignant-group vulnerability analysis (O1) | Not started | Group metadata and prediction reporting can be reused | Controlled pruning probes establish whether conventional importance misses repeatable malignant-group damage | Define perturbations, per-group damage measurements and seed-stability analysis |
| 4. Proposed method and decisive comparisons (O2) | Not started | Structural-pruning primitives can be reused | Implement the candidate mechanism and isolate its benefit from selection rules, class weighting, topology and training budget | Use milestone 3 findings to specify the method and decisive comparator/ablation matrix |
| 5. Final validation and reporting (O3) | Not started; basic reports available | Metrics, saved predictions and resident-forward timing | Frozen evaluation, paired uncertainty, selected seed/fold repeats, repeated timing and claims bounded by feasible replication | Finalize the evaluation matrix after development choices are established |

## Current step

**Milestone 1: start M1.1 implementation/revalidation.** The
[master spec](../specs/m1/M1_SPEC.md), [five-package tracker](../specs/m1/README.md)
and [alignment review](../specs/m1/ALIGNMENT_REVIEW.md) are complete as planning
documents. All package acceptance checklists remain uncompleted. Existing code
is reused; completing specifications is not completing the milestone.

## Existing verification

- The current active CPU regression suite passed 63 tests after cleanup.
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
