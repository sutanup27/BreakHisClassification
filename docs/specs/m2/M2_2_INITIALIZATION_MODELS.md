# M2.2: Parent initialization, compact model and scratch topology

Implementation status: Not started. Depends on: M2.1.
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

- [ ] Implement the master-defined experiment config-2 `baseline` object, validating all combinations before model creation. Keep config-1 readers/defaults unchanged.
- [ ] Verify completed schema-2 parent selection, checkpoint, histories, classes, topology and data/recipe/seed bindings. Freeze exact origin paths/hashes/code/runtime in child inputs.
- [ ] Allow a compatible verified parent as immutable initialization input to a new run across code revisions; preserve origin identity. Keep exact child resume/final source checks and legacy schema-1 phase refusal.
- [ ] Bind only declared confined parent roots; record an acquisition stage with no child parent-fitting steps. Do not synthesize last checkpoints or fake inherited history.
- [ ] Implement fresh dense continuation from the same selected parent, resetting optimizer/schedule/RNG for declared F; no continuation of its optimizer state.
- [ ] Expose narrow VGG construction from the verified eight-width topology: fresh weights, classifier, BN state and optimizer; copy no numerical pruned state.
- [ ] Add compact model creation and strict state-dictionary reconstruction/class mapping, architecture metadata, no-pretraining policy and finite inference checks.
- [ ] Preserve committed-epoch journal, history and best/tie state for all new fits and exact child retries; record initialization kind separately from resume boundary.

## Verification checklist

- [ ] M2-T05: config-1 normal/invalid cases still behave correctly; every config-2 mode accepts meaningful inputs and rejects contradictions; legacy files remain unchanged.
- [ ] M2-T06: corrupt/wrong parent bindings/classes/seeds refuse before fitting; a valid compatible older-origin parent is attributed accurately and never mutated/retrained; changed child code still prevents exact child resume.
- [ ] M2-T07: actual compact forward/backward and checkpoint reload preserve outputs/classes within declared tolerance; pretrained download tripwire never fires.
- [ ] M2-T08: scratch source widths match exactly, but no parent tensor/BN buffer is inherited; same seed reproduces initialization, changed seed changes it.
- [ ] M2-T09: continuation optimizer is fresh; interrupted child stochastic fits resume numerically like uninterrupted runs, including epoch zero and unchanged parent bytes.

## Completion and handoff

M2.3 receives reproducible standard/continuation/scratch constructors and trusted
parent bindings. No scientific baseline quality is claimed from these model tests.
Use generated fixtures and retain CPU portability of selected weights.

## Progress evidence

Not executed: specification only. Internal VGG width support and M1 checkpoint/
resume primitives exist; compact support and public control modes remain proposed.
