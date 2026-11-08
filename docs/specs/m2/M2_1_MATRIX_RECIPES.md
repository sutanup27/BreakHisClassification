# M2.1: Study matrix, recipes and resource plan

Implementation status: Not started. Depends on: completed M1 foundation.
Master: [M2_SPEC.md](M2_SPEC.md), sections 2, 4.1-4.2, 6.
Owns: M2-R01, M2-R02, M2-R03. Acceptance: M2-T01-M2-T04.
Progress: [tracker](README.md).

## Goal and deliverables

Create the small study-plan reader/compiler and proposed `--stage prepare`
interface. Targets: new baseline-plan module, `configs/baselines/m2_study.json`,
public study adapter and focused plan tests. These paths/interfaces are proposed;
they are not claimed to exist. Produce inspectable configs/job dependencies before
launching expensive fits. The precise schema/default matrix is in the master.

## Implementation checklist

- [ ] Add strict versioned study parsing and path/ID validation, separate from existing experiment config parsing.
- [ ] Preserve config-1 behavior; generate proposed config-2 baseline descriptors only through explicit mode expansion.
- [ ] Resolve accepted source/split, 40X pilot versus pooled main, fitting/ranking seeds and device profiles. Never create a new split or test-data job.
- [ ] Expand deterministic parent, compact, L1/random/mask, scratch and dense-continuation jobs with declared B/F budgets and unique new IDs.
- [ ] Deduplicate shared parents by exact seed/recipe/data/initialization identity, without treating arbitrary identical-looking IDs as reusable artifacts.
- [ ] Count 22 fitting jobs for the single-recipe required matrix; enumerate extra pilot, alternative-recipe and restarted-job charges separately.
- [ ] Implement cost-estimate inputs and capacity checks without pretending an unmeasured epoch estimate is known. Plan-only works before profiling; main launch requires measured estimates/available resources.
- [ ] Atomically save frozen study, jobs, configs and non-success states under `artifacts/baselines/<study-id>/`; never overwrite a prior revision.

## Verification checklist

- [ ] M2-T01: reject duplicate keys/jobs, nonfinite costs, unsafe paths, bool scalars, unknown fields and unsupported mode combinations before launching any subprocess.
- [ ] M2-T02: fixture expansion is stable and has the exact required cells/dependencies/counts; no hidden subset on capacity shortfall.
- [ ] M2-T03: fitting seed never changes manifest bytes; wrong split/phase/magnification/policy cannot be called matched; test factories/pixels tripwire throughout planning.
- [ ] M2-T04: hand-count alternative-recipe/retry costs and shared-parent deduplication; insufficient/unknown main capacity prevents main launch but not preparing a readable plan.

## Completion and handoff

M2.2 receives validated, frozen job descriptions and explicit initialization modes.
No new models need fitting to complete this software package. B/F, profile adequacy
and capacity are measured/frozen during M2.4; unknown values remain visible.

## Progress evidence

Not executed: specification only. Existing M1 inputs/config validation is reusable;
no study compiler, new config or completed baseline matrix is claimed.
