# Testing and operational acceptance

The project supplies CPU regressions, generated-image checks and a GitHub Actions
workflow. Synthetic checks establish execution correctness; full-data experimental
acceptance and model quality remain separate tasks.

## Run the regression suite

Install the pinned environment first, then run from the codebase directory:

```powershell
python tools/check_environment.py --device cpu --output artifacts/environment-cpu.json
python -m unittest discover -s tests -v
```

The suite requires no downloaded dataset, pretrained weights, network calls, or
GPU. The end-to-end test generates twelve small PNGs, audits them, and starts
fresh Python interpreters for actual VGG training, weight/channel pruning,
fine-tuning, and comparison. Each subprocess has a four-minute timeout; captured
logs identify failed steps. Use a CPU environment to independently verify the
CPU dependency selection. The GPU environment can also run these CPU tests.

The integration test creates temporary checkpoints inside the workspace parent
`checkpoint/binary/` to respect task storage policy, and removes those test-owned
temporary directories afterward. Other tests use temporary synthetic data and
exercise patient isolation, transforms, finite checks, uneven-batch losses,
zero-accuracy first checkpoints, callbacks, masks/momentum, channel surgery,
fresh-process reconstruction, exact completed-epoch CPU resume, metrics and
benchmark/report conventions.

## Retain a reproducible smoke workflow

```powershell
python tools/smoke_workflow.py --work-dir artifacts/my-smoke --threads 2
```

This executes both supported pruning methods and retains:

```text
artifacts/my-smoke/
  Dataset/                       # Generated synthetic PNGs, not patient images
  data/manifest.json
  data/audit.json
  logs/                          # Eight fresh-process command logs
  commands.json                  # Actual driver argv and equivalent CLI commands
  unstructured-comparison.json
  channel-comparison.json
  summary.json
../checkpoint/binary/smoke-<unique-id>/
  train.pth
  train-last.pth
  unstructured/prune.pth
  unstructured/finetune.pth
  channel/prune.pth
  channel/finetune.pth
```

Configurations, fine-tuning last checkpoints, and validation reports are also
retained beside weights. `summary.json` lists the actual checkpoint directory.
The work directory must be new or empty. The tool always selects CPU, workers
zero, deterministic settings, and one training/fine-tuning epoch. It runs one
timing repeat with no warm-up for an inexpensive execution check; these timings
are not a credible performance benchmark or accuracy experiment.

| Tool argument | Default | Meaning |
| --- | --- | --- |
| `--work-dir` | Unique `artifacts/smoke-<id>` | New/empty generated-data, report and log directory |
| `--methods` | `unstructured channel` | One or both supported methods, e.g. `--methods channel` |
| `--threads` | `2` | Positive CPU intra-op thread count for every subprocess; also sets OMP/MKL limits |

Commands and reports contain absolute paths for this execution. Repeat the tool
to recreate the workflow in a new workspace; do not expect CI temporary paths to
remain available after the job ends. Checkpoints can be large because resume
state and mask buffers are included; remove smoke-owned artifacts when finished.

## V1-only data preparation

Fresh checkouts can now build a manifest without restoring v2:

```powershell
python -m breakhis prepare-data --dataset-root ../Dataset --sources cancer_binary --output artifacts/v1-data --seed 0 --workers 8
python -m breakhis train --dataset breakhis --source cancer_binary --manifest artifacts/v1-data/manifest.json --model vgg --experiment-id v1-baseline
```

Without `--sources`, audit existing configured source folders automatically.
Explicitly selected missing sources fail clearly. Audit reports list skipped
sources and mark the original-v2/v1 comparison unavailable when both were not
selected. Empty/invalid selected sources fail. For a new binary experiment,
selecting `cancer_binary` avoids unnecessary multiclass auditing.

Write a new output directory when keeping old experiments: new manifest hashes
must not be substituted into existing checkpoints. Keep the original manifest
alongside every checkpoint lineage used for resume/evaluation.

## CPU CI

`.github/workflows/cpu-tests.yml` runs on push, pull request and manual dispatch.
It uses a Windows runner, CPython 3.13.14, the pinned CPU wheels/common requirements,
editable package installation, environment checks, the entire regression suite,
and a generated-documentation freshness check. Permissions are read-only for
repository contents. It uploads only generated synthetic evidence and environment
reports for seven days; it does not download BreaKHis or run full-data training.

The suite sets `BREAKHIS_SMOKE_EVIDENCE=artifacts/ci-smoke` in CI so generated data,
reports and logs survive test teardown. Checkpoint weights remain temporary and
are not uploaded. Hosted runner images and action major tags can change; versions
of Python/packages are pinned, and runtime details are recorded. The first remote
CI run still needs to execute after these changes are pushed; local success is
not a claim that GitHub CI has passed.

The workflow follows the official [GitHub Python testing guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python),
[setup-python](https://github.com/actions/setup-python),
[checkout](https://github.com/actions/checkout), and
[artifact action](https://github.com/actions/upload-artifact) interfaces.

## Supported surface and limits

Supported: binary VGG/ResNet18 training, saved patient/slide splits, unstructured
weight masks, sequential VGG channel pruning, validation selection, versioned
checkpoint reconstruction, workers-zero completed-epoch resume, and
evaluation/comparison reports. The baseline platform is Windows x64/Python 3.13;
other OS environments need separate verification.

Experimental/unsupported: multiclass training, residual/depthwise channel surgery,
mixed channel/mask models and historical architecture-specific helpers.
See [pruning limits](PRUNING.md).

Filename IDs are slide/patient surrogates, not independently verified clinical
identities. No full-dataset accuracy, clinical validity, end-to-end latency benefit
or statistically significant speedup is established by these smoke tests.

## Full-data acceptance still required

Before declaring a final experiment complete, install a fresh environment, audit
the chosen real source and saved splits, train a full baseline, select on validation,
reload in a fresh process, prune and verify invariants, fine-tune with validation,
then evaluate the selected baseline/pruned models once on reserved test data.
Save configurations, checkpoint hashes, original manifests, logs and fixed-protocol
accuracy/size/MAC/latency comparisons. The active checklist below tracks this
separately from implementation and synthetic regression acceptance.

## Full-data acceptance checklist

Status: full research experiments remain outstanding; the one-epoch real-data
execution check is recorded in the [current roadmap](../research/EXPERIMENT_ROADMAP.md).
Check items only when evidence for the actual declared experiment is recorded.

- [ ] Install dependencies into a fresh virtual environment.
- [ ] Validate the chosen real dataset and saved patient/slide splits.
- [ ] Train a full binary baseline and select its checkpoint using validation.
- [ ] Reload the baseline in a fresh process.
- [ ] Prune and verify structural or mask invariants.
- [ ] Fine-tune using validation selection, then save/reload the result.
- [ ] Evaluate selected baseline and pruned models on reserved test data.
- [ ] Generate the accuracy, size, compute and latency comparison report.
- [ ] Preserve configurations, split references, checkpoint hashes and logs.
