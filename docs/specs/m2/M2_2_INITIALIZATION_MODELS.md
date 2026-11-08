# M2.2: Parent initialization, compact model and scratch topology

Implementation status: Reviewed. M2-R04-M2-R06 and M2-T05-M2-T09 are implemented and verified. Depends on: M2.1.
Master: [M2_SPEC.md](M2_SPEC.md), sections 3, 4.3, 5-6.
Owns: M2-R04, M2-R05, M2-R06. Acceptance: M2-T05-M2-T09.
Progress: [tracker](README.md).

## Goal and deliverables

Extend existing config/setup/checkpoint/model/fit interfaces to make every control's
initialization explicit. Targets: `breakhis/experiments.py`, `experiment_setup.py`,
`checkpoints.py`, `cli.py`, `models/vgg.py`, parent-reference verification and model/
resume tests. Add torchvision MobileNetV3-Small as the one compact comparator;
do not implement its physical pruning or download pretrained weights.

## Implementation checklist

- [x] Implement the master-defined experiment config-2 `baseline` object, validating all combinations before model creation. Keep config-1 readers/defaults unchanged.
- [x] Verify completed schema-2 parent selection, checkpoint, histories, classes, topology and data/recipe/seed bindings. Freeze exact origin paths/hashes/code/runtime in child inputs.
- [x] Allow a compatible verified parent as immutable initialization input to a new run across code revisions; preserve origin identity. Keep exact child resume/final source checks and legacy schema-1 phase refusal.
- [x] Bind only declared confined parent roots; record an acquisition stage with no child parent-fitting steps. Do not synthesize last checkpoints or fake inherited history.
- [x] Implement fresh dense continuation from the same selected parent, resetting optimizer/schedule/RNG for declared F; no continuation of its optimizer state.
- [x] Expose narrow VGG construction from the verified eight-width topology: fresh weights, classifier, BN state and optimizer; copy no numerical pruned state.
- [x] Add compact model creation and strict state-dictionary reconstruction/class mapping, architecture metadata, no-pretraining policy and finite inference checks.
- [x] Preserve committed-epoch journal, history and best/tie state for all new fits and exact child retries; record initialization kind separately from resume boundary.

## Verification checklist

- [x] M2-T05: config-1 normal/invalid cases still behave correctly; every config-2 mode accepts meaningful inputs and rejects contradictions; legacy files remain unchanged.
- [x] M2-T06: corrupt/wrong parent bindings/classes/seeds refuse before fitting; a valid compatible older-origin parent is attributed accurately and never mutated/retrained; changed child code still prevents exact child resume.
- [x] M2-T07: actual compact forward/backward and checkpoint reload preserve outputs/classes within declared tolerance; pretrained download tripwire never fires.
- [x] M2-T08: scratch source widths match exactly, but no parent tensor/BN buffer is inherited; same seed reproduces initialization, changed seed changes it.
- [x] M2-T09: continuation optimizer is fresh; interrupted child stochastic fits resume numerically like uninterrupted runs, including epoch zero and unchanged parent bytes.

## Completion and handoff

M2.3 receives reproducible standard/continuation/scratch constructors and trusted
parent bindings. No scientific baseline quality is claimed from these model tests.
Use generated fixtures and retain CPU portability of selected weights.

## Progress evidence

Implemented `breakhis/baseline_init.py`, strict config-2 parsing, immutable
initialization/parent acquisition records, declared external roots, narrow/compact
construction and continuation wiring through existing fit/boundary/report/final
primitives. Existing config-1 semantics remain supported. Parent pruning validates
and prepares, but its fitting correctly refuses until M2.3. Baseline-control saved
comparison extensions remain M2.5; individual phase reports are usable now.

Verification commands:

```powershell
python -m unittest discover -s tests
python -m unittest discover -s tests -p test_baseline_initialization.py
```

- Full CPU suite: 174 cases in 211.713 seconds, 171 passed and three CUDA-only
  skips. Log: `artifacts/m2-2-regression.log`.
- CUDA environment: all 19 focused initialization cases passed in 13.414 seconds,
  including actual MobileNetV3-Small CUDA training, portable save and CPU reload
  within `atol=1e-4`, `rtol=1e-3`. Log: `artifacts/m2-2-cuda-tests.log`.
- Fresh-process public generated-image check passed for config-2 full VGG parent,
  continuation, L1 topology, fresh narrow scratch and MobileNetV3-Small; child
  development and explicit final phases all validated, parent weights unchanged.
  Evidence: `artifacts/m2-2-initialization-smoke/summary.json`, command/per-command
  logs in that folder and `artifacts/m2-2-smoke.log`. These are execution checks,
  not converged real-data baselines or efficiency claims.

M2-T05 covers config versions, mode contradictions and deferred pruning refusal.
M2-T06 covers missing/drifted parent/selection/recipe/seed/data/root/initialization
bindings and new-child versus old-origin source identity. Parent acquisition
copies no weights or synthetic history and charges zero child parent-fit steps.
M2-T07 covers real compact forward/backward, no-download tripwires, class mapping,
checkpoint reconstruction, both high-level phases and CUDA-to-CPU portability.
M2-T08 covers exact source widths with fresh numerical/BN state, reproducible
initialization, changed-seed effects, bound topology failure and actual scratch
training. Old unbound topologies require regeneration with current parent hashes.
M2-T09 covers fresh continuation Adam state, no invented parent train history,
interrupted/epoch-zero exact child resume and test-only final without fitting
pixels/optimizer creation. Existing M1 numerical SGD/Adam/tie and publication
regressions also pass in the full suite.

The compact sample is `configs/experiments/mobilenet_baseline.json`; its 25-epoch
40X recipe is a starting example, not a quality claim. Guides/config fields,
checkpoint rules, current help, local links/HTML and trackers are updated.
No real BreaKHis model fit or Git commit/push was performed by this package.
Next: M2.3 conventional random/L1 controls and matched recovery integration.
