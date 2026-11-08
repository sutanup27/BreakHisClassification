# Evaluation and efficiency reporting

Stage 8 extends `evaluate` and adds `compare`. Reports use the selected manifest's
test split by default with deterministic validation/test preprocessing.
Use `evaluate --split validation` for development reporting; `compare` uses test.
Only the requested evaluation role is constructed. Training,
pruning-ratio decisions, and checkpoint selection still use validation. Avoid
using repeated test reports to choose models or ratios.

## Evaluate one checkpoint

```powershell
python -m breakhis evaluate --dataset breakhis --checkpoint ../checkpoint/binary/40X/vgg/baseline/train.pth --device auto --benchmark-batch-size 1 --benchmark-warmup 10 --benchmark-repeats 30 --output artifacts/baseline-evaluation.json
```

Supply the original `--manifest`, `--source`, and `--magnification` when they
differ from defaults. Checkpoint provenance must match. The report includes
image/patient metrics, size accounting, compute estimates, latency measurements,
hardware/runtime details, and the checkpoint SHA-256. Its resolved configuration
is saved beside it as `.config.json`. Existing wrapper scripts such as `Test.py`
delegate to the enhanced evaluation command.

## Compare three selected checkpoints

```powershell
python -m breakhis compare --dataset breakhis --dense ../checkpoint/binary/40X/vgg/baseline/train.pth --pruned ../checkpoint/binary/40X/vgg/baseline/prune.pth --finetuned ../checkpoint/binary/40X/vgg/baseline/finetune.pth --device auto --benchmark-batch-size 1 --benchmark-warmup 10 --benchmark-repeats 30 --cpu-threads 2 --output artifacts/comparison.json
```

For channel-pruning experiments, point `--pruned` and `--finetuned` at that
experiment's channel checkpoints and retain its original dense baseline.
Comparison verifies matching source, magnification, manifest hash, preprocessing,
class labels and baseline architecture. The dense role must be dense; the other
roles must share their pruning method, masks or channel selection, and topology.
Roles are operator-supplied: these checks do not independently establish that the
file labeled fine-tuned was actually trained. That provenance remains in saved
training configuration and checkpoint history.

All three models use the same test loader, device, resident benchmark batch,
input dimensions, warm-up/repeats, and CPU thread setting. They are loaded and
measured sequentially. The comparison writes all individual reports plus deltas
relative to the dense baseline; it never trains or iterates training/validation.

## Arguments

Both commands accept shared data options from the [README](../../README.md).

| Argument | Commands | Default and meaning |
| --- | --- | --- |
| `--checkpoint` | evaluate | Required input checkpoint |
| `--dense` | compare | Required dense baseline file |
| `--pruned` | compare | Required pruned-before-fine-tuning file |
| `--finetuned` | compare | Required selected fine-tuned file |
| `--benchmark-batch-size` | both | `1`; positive resident forward batch, independent of evaluation `--batch-size` |
| `--benchmark-warmup` | both | `10`; nonnegative untimed warm-up forwards, after one untimed MAC-counting forward |
| `--benchmark-repeats` | both | `30`; positive measured forward count; every sample is saved |
| `--cpu-threads` | both | Unchanged when omitted; positive PyTorch intra-op thread count when supplied; record also includes actual inter-op threads |
| `--experiment-id` | both | Generated unless supplied; controls default report directory |
| `--output` | both | Optional `.json` report path. Defaults to `checkpoint/binary/<magnification>/<architecture>/<experiment-id>/evaluate.json` or `compare.json` |

`--output` must be JSON and must not replace an input checkpoint. Reports and
configurations are written locally; report output can be outside the checkpoint
task folder. `compare` uses `--dense/--pruned/--finetuned`, not `--checkpoint`.

## Classification definitions

- `accuracy` is a percentage (0-100). Precision, recall, F1, sensitivity,
  specificity and sparsity use fractions (0-1).
- Confusion-matrix rows are true labels and columns predicted labels. Ordered
  classes and `class_to_idx` are explicit: benign 0, malignant 1.
- Per-class precision/recall/F1 and support are reported. Macro metrics give each
  declared class equal weight; weighted metrics use class support. Undefined
  divisions are zero, including absent classes. This convention can lower macro
  scores on a split containing only one class; support remains visible.
- Binary sensitivity is malignant recall. Specificity is benign recall.
- Loss is mean image cross-entropy, weighted by image count across uneven batches.

## Filename patient/slide metrics

Use manifest `patient_id` (YEAR-SLIDE_ID) as a grouping surrogate. Average softmax
probabilities over each group's images and select the highest mean; ties select
the first label. Each group then contributes one classification observation,
regardless of image count. Reports preserve group probabilities, labels, image
counts, and the same confusion/averaging conventions as image metrics.

These are **filename-group metrics**, not independently verified clinical
patient metrics. `all` magnifications aggregate all selected images within the
group. Mixed labels in a group, missing identifiers, dropped last batches, and
shuffled manifest loaders are rejected. Generic datasets without row metadata
report patient metrics as unavailable rather than inventing identities.

## Size accounting

| Field | Meaning |
| --- | --- |
| `logical_parameters` | Parameter elements, including masked positions; mask buffers are not counted as parameters |
| `effective_nonzero_parameters` | Nonzeros after applying masks; includes other parameters such as BN affine values and biases |
| `parameter_sparsity` | One minus effective nonzero/total parameter count; natural zeros count too |
| `masked_weight_entries`, `mask_entries` | Explicit masked positions and total eligible weight positions, distinct from natural zeros |
| `dense_parameter_bytes` | Logical parameter value storage based on actual dtype; excludes buffers/metadata |
| `ideal_nonzero_value_bytes` | Nonzero values only, an ideal lower-bound component, not a real file size |
| `flat_index_sparse_bytes_estimate` | Nonzero values plus one int64 flat index each; excludes shapes, buffers and format overhead |
| `checkpoint_file_bytes` | Actual complete input file size, including any optimizer/RNG/buffers/masks |
| `model_state_serialized_bytes` | Measured `torch.save(state_dict)` bytes, including buffers/masks, excluding optimizer/RNG |
| `mask_buffer_bytes` | Explicit mask tensor storage |

A sparse estimate may exceed dense value storage because indices cost space.
A training checkpoint may exceed a pruning checkpoint because optimizer/RNG
payloads differ. Neither difference alone establishes deployment model size.

## Compute and latency protocol

Input dimensions come from the first real preprocessed evaluation image (normally
3 x 224 x 224), not a CIFAR-sized placeholder. Benchmark inputs repeat that image
into a contiguous batch of `--benchmark-batch-size`, resident on the model device.
The report records exact NCHW shape and dtype.

Hooks count Conv2d/Linear multiply-accumulates from actual forward outputs,
including grouped convolution dimensions. This is explicitly a **Conv2d/Linear
compute estimate**, excluding bias additions, BN, activation, pooling, residual
adds and mask multiplication. The related FLOP estimate uses two FLOPs per MAC.
Unstructured zeros do not reduce dense MACs. Channel removal changes dimensions
and therefore changes this compute estimate.

After one untimed MAC-counting forward and the requested warm-up forwards,
record repeated `perf_counter_ns` wall-clock forwards in inference/eval mode.
CUDA synchronizes before and after each measured call. Timings include forward
dispatch, pruning hooks and synchronization overhead. They exclude disk IO,
transforms, host-to-device transfers, loss, and metric aggregation. Output-finite
checks occur outside the measured interval.

Report mean, median, p95 (linear interpolation), min/max, population standard
deviation, all samples, median-based throughput, and amortized per-image latency.
Amortized latency is not single-image latency when batch size exceeds one.
CPU threads, CPU identity/platform, GPU identity/memory/capability, PyTorch/Python,
CUDA/cuDNN runtime, determinism and TF32 settings are recorded.

Keep hardware, dtype, batch size, thread counts, backend settings and protocol
fixed. Close competing workloads and use adequate repeats for a real experiment.
Cold-start initialization and end-to-end data-pipeline latency are not measured.
Default timing runs are measurements, not a guarantee of statistical significance
or reproducibility across different devices/drivers/software.

## Reading comparisons

`relative_to_dense` contains accuracy deltas in percentage points, actual file and
model-state byte ratios, parameter/MAC ratios, observed median speedup, and signed
latency change. Speedup is dense median / candidate median: below one is slower.
Positive latency change means slower. Slowdowns and larger files are retained.
Do not infer a practical speed benefit from mask sparsity or parameter reduction.

Verification covers hand-checked metrics/MAC counts, uneven-batch losses, filename
group aggregation, mask-aware sizes, invalid settings/provenance, and no train/
validation iteration during comparisons. Existing real-v1 smoke checkpoints were
compared for both masks and channel removal. Those small reports verify execution,
not full-dataset accuracy or general hardware speedup claims.
