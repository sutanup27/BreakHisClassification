# M2.3: Conventional pruning and fair controls

Implementation status: Not started. Depends on: M2.1-M2.2.
Master: [M2_SPEC.md](M2_SPEC.md), sections 4.2-4.3 and 5-6.
Owns: M2-R07, M2-R08, M2-R09. Acceptance: M2-T10-M2-T13.
Progress: [tracker](README.md).

## Goal and deliverables

Reuse physical VGG surgery for both L1 and seeded-random retained channels. Complete
parent reuse, masked controls, scratch and continuation through phase artifacts.
Targets: `breakhis/pruning.py`, `checkpoints.py`, `development.py`,
`phase_artifacts.py`, `final_evaluation.py`, CLI adapters and pruning/control tests.
Random/L1 channel metadata must validate faithfully; do not tag random channels
as L1 or enable unsafe residual/depthwise surgery.

## Implementation checklist

- [ ] Separate retained-index selection from the checked copy/rebuild primitive; preserve existing stable L1 semantics and minimum-one-channel rounding.
- [ ] Implement seed-local random selection at exactly L1's per-layer retained counts, sorting retained indices and recording seed/IDs without consuming fitting RNG.
- [ ] Validate channel metadata for both rules, downstream tensor/BN/classifier dependencies, and reconstruct both from saved state dictionaries.
- [ ] Retain L1 masks and enforce masked weights/optimizer buffers through recovery; never claim masked zeros shrink dense kernels or MACs.
- [ ] Build every pruning child from the same verified seed's selected parent, reset optimizer for F, and run dense continuation from that identical checkpoint.
- [ ] Route narrow scratch to fresh B+F fitting, compact to B+F, parent to B. Preserve selected versus executed epochs and explicit baseline-kind attribution.
- [ ] Extend mode-specific expected stages, histories, selection roles and parent-root references without fake completed training. Bind fixed designated roles exactly as in the master.
- [ ] Keep real M2 execution development-only. Verify existing explicit final can load compact/control selections on generated data, including portable CUDA-origin parent references.

## Verification checklist

- [ ] M2-T10: tied toy weights and known seeds produce exact widths/IDs; L1 and random have identical physical shapes while choices can differ; source and global RNG stay unchanged.
- [ ] M2-T11: forward/backward/save/reload, BN state, masks/momentum and physical parameter/MAC accounting remain correct; reject mixed masks/channels and residual surgery.
- [ ] M2-T12: fixture budgets prove F versus B+F and no hidden inherited optimizer; changed batches or asymmetric adaptation cannot be called matched.
- [ ] M2-T13: generated dense/reference, scratch/compact and recovery roles export valid development selections and explicit final predictions; external artifact tampering refuses and no study stage opens real test data.

## Completion and handoff

M2.4 can execute the bounded matrix with consistent initialization, adaptation and
artifact semantics. Uniform L1/random are conventional controls; neither is the
malignant-vulnerability mechanism, sensitivity-constrained search or class weighting.

## Progress evidence

Not executed: specification only. Existing L1 mask/channel correctness tests are
regression inputs, not evidence that proposed random/control phase support exists.
