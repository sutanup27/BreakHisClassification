# M2.1: Study matrix, recipes and resource plan

Implementation status: Reviewed. M2-R01-M2-R03 and M2-T01-M2-T04 are implemented and verified. Depends on: completed M1 foundation.
Master: [M2_SPEC.md](M2_SPEC.md), sections 2, 4.1-4.2, 6.
Owns: M2-R01, M2-R02, M2-R03. Acceptance: M2-T01-M2-T04.
Progress: [tracker](README.md).

## Goal and deliverables

Create the small study-plan reader/compiler and proposed `--stage prepare`
interface. Targets: new baseline-plan module, `configs/baselines/m2_study.json`,
public study adapter and focused plan tests. The study-plan module, example config and public preparation adapter are now
implemented; model/control execution remains in later packages. Produce inspectable configs/job dependencies before
launching expensive fits. The precise schema/default matrix is in the master.

## Implementation checklist

- [x] Add strict versioned study parsing and path/ID validation, separate from existing experiment config parsing.
- [x] Preserve config-1 behavior; generate proposed config-2 baseline descriptors only through explicit mode expansion.
- [x] Resolve accepted source/split, 40X pilot versus pooled main, fitting/ranking seeds and device profiles. Never create a new split or test-data job.
- [x] Expand deterministic parent, compact, L1/random/mask, scratch and dense-continuation jobs with declared B/F budgets and unique new IDs.
- [x] Deduplicate shared parents by exact seed/recipe/data/initialization identity, without treating arbitrary identical-looking IDs as reusable artifacts.
- [x] Count 22 fitting jobs for the single-recipe required matrix; enumerate extra pilot, alternative-recipe and restarted-job charges separately.
- [x] Implement cost-estimate inputs and capacity checks without pretending an unmeasured epoch estimate is known. Plan-only works before profiling; main launch requires measured estimates/available resources.
- [x] Atomically save frozen study, jobs, configs and non-success states under `artifacts/baselines/<study-id>/`; never overwrite a prior revision.

## Verification checklist

- [x] M2-T01: reject duplicate keys/jobs, nonfinite costs, unsafe paths, bool scalars, unknown fields and unsupported mode combinations before launching any subprocess.
- [x] M2-T02: fixture expansion is stable and has the exact required cells/dependencies/counts; no hidden subset on capacity shortfall.
- [x] M2-T03: fitting seed never changes manifest bytes; wrong split/phase/magnification/policy cannot be called matched; test factories/pixels tripwire throughout planning.
- [x] M2-T04: hand-count alternative-recipe/retry costs and shared-parent deduplication; insufficient/unknown main capacity prevents main launch but not preparing a readable plan.

## Completion and handoff

M2.2 receives validated, frozen job descriptions and explicit initialization modes.
No new models need fitting to complete this software package. B/F, profile adequacy
and capacity are measured/frozen during M2.4; unknown values remain visible.

## Progress evidence

Implemented `breakhis/baseline_plan.py`, `run_baselines.py` and
`configs/baselines/m2_study.json`. No existing config-1 parsing/training behavior
was changed. Preparation uses shared full-source metadata/path validation and
atomic finite writers; it creates no loaders/models/checkpoints or CUDA query.

Focused verification: 20 tests in `tests/test_baseline_plan.py` passed in 3.848
seconds. Log: `artifacts/m2-1-tests.log`. M2-T01-M2-T04 cover strict scalar/JSON/
profile/recipe errors, graph cycles/unknown/duplicate/mismatched parents, stable
semantic recipe identity, complete counts/draft bindings, unavailable test pixels,
manifest and seed invariants, cost overflow, missing/short capacity and replay
charges. Publication failure and occupied/unsafe outputs never claim success.

Real metadata-only public preparation succeeded with study ID
`m2-1-planning-reviewed-20261108`; outputs are under `artifacts/baselines/` and use
all four magnifications' existing metadata. No real fitting/inference occurred.
Default plan counts are 22 required + 3 optional recipes + 6 pilot fits (31 maximum
before retries); all config envelopes are `runnable=false`. Missing costs stay
unknown, and even sufficient capacity reports `execution_supported=false`.

The optional cost JSON and pure capacity helpers are documented in the main config
guide. They accept supplied estimates and replay overhead; they do not fabricate
hardware measurements. New future execution stages and preparation resume refuse.
Every output has a hash in the final prepared status; failures retain non-success.

The final CPU suite ran 155 tests in 197.753 seconds: 153 passed and two
CUDA-only cases skipped. Log: `artifacts/m2-1-regression-reviewed.log`.
All retained real-plan graph/artifact/source hashes, current help, Markdown
links/headings, generated HTML anchors and `git diff --check` passed.
Long study names have distinct fit/selector IDs, including when prefixes match.

Real cost profiling,
compact/control execution and scientific baseline quality remain M2.2-M2.5 work.
