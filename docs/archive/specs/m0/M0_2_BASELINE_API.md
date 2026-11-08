> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.2 baseline preservation API

Implemented in `breakhis/research/m0.py`. This is an engineering baseline API;
the complete M0 runner, audit, governance and Gate G0 remain later packages.
Importing this module performs no capture, dependency import or training.

## Run a baseline

From the repository root, with the intended virtual environment activated:

```powershell
@'
from pathlib import Path
from datetime import datetime, timezone
from breakhis.research.records import create_run_directory
from breakhis.research.m0 import capture_baseline

repo = Path.cwd().resolve()
run_id = datetime.now(timezone.utc).strftime('baseline-%Y%m%dT%H%M%SZ')
run = create_run_directory(repo / 'artifacts/research/m0', run_id)
result = capture_baseline(repo, run, run_id, devices=['cpu'],
                          environment_seconds=120, regression_seconds=600)
print('Evidence:', run)
print('CPU engineering:', result['cpu_engineering_ready'])
print('GPU execution:', result['gpu_execution_ready'])
print('Restrictions:', result['restrictions'])
'@ | python -
```

Use `devices=['cpu', 'cuda']` to check both devices with a CUDA-capable environment.
CPU checks hide CUDA and use two OMP/MKL threads; the environment check performs
synthetic tensor, preprocessing and VGG forward checks. Regressions use synthetic
fixtures. These commands do not request scientific training or real-image test
inference. No dependencies, weights or datasets are downloaded.

Each invocation needs a fresh run directory. Failed/partial evidence is retained;
do not reuse or edit it. No commit, push, dataset audit or checkpoint is created.
The future M0 runner will provide configuration binding and final gate evaluation.

## Functions and outputs

| API | Contract |
| --- | --- |
| `capture_source(repo_root, run_dir, run_id)` | Capture source only; refuse an existing baseline, retry source movement once, then fail with retained attempts |
| `baseline_plan(python, devices, environment_seconds=120, regression_seconds=600)` | Inspect fixed allowlisted argv; no execution; positive integer timeouts |
| `capture_environment_checks(repo_root, run_dir, run_id, python=None, devices=None, environment_seconds=120, regression_seconds=600)` | Current checks and metadata; default interpreter is the calling Python; default device CPU |
| `capture_baseline(repo_root, run_dir, run_id, **check_options)` | Preferred combined API; additionally recheck source after checks, block eligibility on drift/unresolved source |

`repo_root` must be the Git top-level directory with a HEAD commit. `run_dir`
must already exist under repository `artifacts/research/m0/`; use the M0.1
directory helpers, not a dataset path. Filesystem containment follows M0.1's
trusted-root/static-path threat model. Git commands have a 60-second timeout.

Records use M0.1 envelopes and no-clobber writes. `checks.json` is progressively
updated within its owned run. UTF-8 stdout/stderr logs retain exact subprocess
bytes, sizes and SHA-256 references. Each operation records argv, cwd, device,
start/end UTC, elapsed time, timeout, return code, reason and source-record input
hashes when source capture exists. Failures, absent utilities and timeouts remain
failed observations. A timeout terminates the command tree; if the Windows
tree-termination utility fails, that restriction is retained in the reason.
Ctrl+C during a check records interruption and stops the remaining check plan.

### Source reconstruction

`baseline/git.json` records HEAD/ref, NUL-delimited porcelain status (JSON-escaped),
dirty state, index identity, policy, attempts, unresolved LFS/submodules and source
identities. `source_inventory.json` lists portable repository-relative paths,
tracked/untracked state, bytes, SHA-256 and capture status. Staged deletions have
null bytes/hash and status `deleted`; no file is fabricated for them.

`source_snapshot/` contains the included working-tree files at their original
relative paths. Copy these into an empty reconstruction directory, then compare
each captured row's hash/size; deleted rows must be absent. This reconstructs the
included source, not the entire repository or its Git history. `tracked.patch`
is the binary-capable included worktree diff against HEAD; `staged.patch` retains
the included index diff against HEAD. Excluded files never enter either patch.

Policy v1 includes canonical and compatibility Python source, tools/tests,
configs, selected root project/dependency files, workflows and Markdown/assets.
It excludes data/weights/environments/builds/generated outputs, generated HTML
and non-allowlisted personal folders. Ignored untracked files are excluded;
tracked files remain subject to the same capture policy. Keep sensitive source
material outside these public-ready source folders. LFS pointers are copied as
pointers, and submodule references are recorded without fetching their content;
either prevents self-contained-baseline eligibility.

The snapshot digest covers sorted present path/hash pairs and policy; the source
identity additionally covers HEAD/index/status metadata, including dirty state.
The commit alone never identifies a dirty baseline. Attempts are checked before
and after copying; source changes during checks also invalidate combined-baseline
eligibility. Failed attempt copies remain named `attempt-1` / `attempt-2` and
are not presented as the accepted snapshot. Concurrent hostile path replacement
is outside the M0.1 static-filesystem guarantee.

### Environment and checks

`environment.cpu.json` and `environment.cuda.json` contain per-device status,
actual interpreter/version/venv, platform/CPU/dependency metadata and synthetic
environment observations when available. The separate metadata command runs
even when scientific dependencies cannot import. Missing processor strings and
failed environment observations are unavailable, not inferred hardware facts.
`pip check`, `pip freeze --all` and `pip --version` outputs are separate logged
checks. NVIDIA driver identifiers come only from the logged optional
`nvidia-smi` query; absent utilities produce unknown driver metadata, not a
guessed driver version. CUDA runtime and driver versions are distinct facts.

The plan includes imports, all six supported CLI help commands, pip checks,
per-requested-device synthetic environment checks, and the current full unittest
suite. Skipped regressions or zero discovered tests cannot satisfy acceptance.
An unrequested GPU is `not_requested`, not validated or known unavailable.
GPU failure restricts GPU execution without negating passing CPU engineering
checks. Optional driver utility failure alone does not block CPU or GPU compute
checks. `baseline/consistency.json` records the final source recheck.

The returned booleans are baseline-only eligibility. Neither means M0/G0 passed,
nor establishes data integrity, clinical identity, untouched exposure, protocol
agreement, scientific accuracy or confirmatory readiness.

## Developer tests

```powershell
python -m unittest discover -s tests -p test_m0.py -v
python -m unittest discover -s tests -v
```

See [package checklist and evidence](M0_2_BASELINE_PRESERVATION.md).
