# ResNet18 full dependency-aware channel pruning

Specification: **RESNET18-CHANNEL-SPEC-001**, version 1.0, 8 November 2026.
Status: **Software implemented and reviewed; real scientific experiments pending.**
Parent milestone: [M2 master](M2_SPEC.md). Progress: [M2 tracker](README.md).
Research scope: [active roadmap](../../research/EXPERIMENT_ROADMAP.md) and
[research guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md).

## 1. Objective and completion boundary

Add physical, dependency-aware channel pruning for the currently supported
**torchvision ResNet18**, including residual-stream output channels, internal
block channels, stage-transition projection shortcuts and classifier inputs.
Support conventional L1 and seeded-random controls with identical topology widths,
portable checkpoints, fresh recovery, scratch controls and the existing
validation-development / explicit-test-final workflow.

This extends M2.3 as **M2.3-R**, before ResNet study execution in M2.4. It does not
replace the five research milestones or reopen completed VGG acceptance. Previous
M2 documents' residual-surgery exclusions describe the implemented boundary; this
spec authorizes extending that boundary only after its tests pass. Existing VGG
study configs remain unchanged until an explicit study revision adds ResNet jobs.

Software completion means every requirement and acceptance test below has passing
evidence, user docs match behavior, and the public generated-image workflow works.
It does not mean a converged ResNet baseline, actual latency improvement,
architecture-general research novelty, or completion of M2 scientific experiments.
No external approval roles, literature gates or evidence archive are introduced.

## 2. Supported architecture and explicit limits

Accept only the checked torchvision ResNet18 layout: BasicBlock [2,2,2,2], RGB
7x7 stem stride 2, ordinary groups=1 convolutions, affine tracked BatchNorm2d,
maxpool, adaptive average pool and linear classifier. Dense canonical widths are
64/128/256/512. Stage 1 has no projection; stages 2-4 begin with stride-2 projection
blocks. Preserve strides, padding, dilation, pooling, activation order and class
order. Accept the existing `resnet` / `resnet18` aliases consistently.

Reject unsupported layouts **before any mutation or output publication**:
Bottleneck/ResNet50, modified depths, grouped/depthwise/dilated convolutions,
nonstandard shortcuts or normalization, fused/quantized/compiled models, masked
weights, shared/tied parameters, nonfinite source state, and already physically
pruned sources. Re-pruning is intentionally unsupported in this version. Do not
silently treat an arbitrary ResNet object as supported just because its name fits.
Pretrained parents may be pruned as loaded weights; reconstruction must never
trigger downloads. Preserve the existing experiment initialization policy and do
not silently enable pretrained weights for the main study.

## 3. Exact dependency graph

A channel ID always means the coordinate in the original dense parent. Retained
IDs are sorted, unique, nonempty and copied in that order. Equal width alone does
not establish residual compatibility: both addition branches require equal IDs.

There are **12 independently selectable dependency groups**: four residual-stage
output groups S1-S4 and eight block-internal groups H10-H41. Layer names below
use the canonical state-dictionary layout.

| Group | Original width | Coupled producer outputs | Consumers using the same input IDs |
| --- | ---: | --- | --- |
| S1 | 64 | stem `conv1`/`bn1`; `layer1.0.conv2`/`bn2`; `layer1.1.conv2`/`bn2` | both layer1 block conv1 inputs; `layer2.0.conv1` and `layer2.0.downsample.0` inputs |
| S2 | 128 | both layer2 block conv2/BN2 outputs; `layer2.0.downsample.0`/`.1` outputs | `layer2.1.conv1`; `layer3.0.conv1` and projection inputs |
| S3 | 256 | both layer3 block conv2/BN2 outputs; `layer3.0.downsample.0`/`.1` outputs | `layer3.1.conv1`; `layer4.0.conv1` and projection inputs |
| S4 | 512 | both layer4 block conv2/BN2 outputs; `layer4.0.downsample.0`/`.1` outputs | `layer4.1.conv1`; `fc` input columns |
| Hs,b (eight groups) | width of stage s | `layer{s}.{b}.conv1` / BN1 outputs | same block's conv2 input columns |

For block (s,b):

- Input IDs are S(s-1) for the first block of stages 2-4; otherwise Ss.
- `conv1.weight = old[Hs,b][:, input_ids]`; BN1 uses Hs,b.
- `conv2.weight = old[Ss][:, Hs,b]`; BN2 uses Ss.
- Existing transition projection weight uses `old[Ss][:, S(s-1)]`;
  its BN uses Ss and its original stride is retained.
- Identity shortcuts remain identity. Never add a new learned projection to repair
  inconsistent IDs; reject the inconsistent specification instead.
- Stem weights select S1 outputs and retain all RGB inputs. Stem BN selects S1.
- Final classifier selects S4 input columns and retains every output row and bias.

Identity connections couple all block outputs inside a stage. Stage 1 also couples
stem outputs because `layer1.0` has no projection. Stage transitions remain
independent because their existing projections map between the two stage groups.
No channel padding, runtime gather/scatter, masks or zero-filled dense branches
may substitute for smaller convolution/BN/linear tensors in the resulting model.

## 4. Selection policy: conventional controls only

Given a uniform requested sparsity r, each group of original width C retains
`max(1, C - round(C*r))`. Ratios must be finite real numbers in [0,1), excluding
booleans. Apply the exact same count calculation to L1 and random. Resolve all
selections from the untouched parent before rebuilding; never let earlier surgery
change later importance scores.

**Dependency-group L1**, explicitly defined to avoid arbitrary branch choices:

1. For each producer convolution in the group's table row, compute each original
   output filter's mean absolute weight over input and spatial dimensions.
2. Average those vectors equally across that group's producer convolutions.
   Include the stem for S1 and the existing projection for S2-S4.
3. Retain highest scores, with stable ascending original-ID tie breaking.
4. Do not use BN values, input-consumer columns, labels, gradients or validation
   performance in this conventional ranking. Record score rule
   `mean-producer-filter-mean-abs-v1` so it cannot be confused with VGG's rule.

For each hidden group, the sole producer is that block's first convolution. Mean
versus sum does not change its within-layer ordering, but the explicit normalized
stage rule prevents large producer filters dominating merely through filter size.
This is a declared baseline heuristic, not the proposed malignant-aware method.

**Random**: local CPU torch.Generator with a uint32 ranking seed. In fixed order
`stage1,stage2,stage3,stage4,layer1.0.hidden,layer1.1.hidden,...,layer4.1.hidden`,
draw one randperm(C), retain first K and sort IDs. Consume a draw for every group,
including r=0. Repeated seed/topology/ratios give identical IDs. Different seeds
need not differ at zero pruning or degenerate width. Preserve Python, NumPy, global
CPU and every initialized CUDA fitting RNG. L1 has no ranking seed.

Per-layer `--layer-sparsities` cannot express coupled dependencies safely: reject
it for ResNet channel surgery. Provide a proposed new `--group-sparsities FILE`
option for a complete exact 12-key map using the group keys above. Uniform ratio
remains the default; unknown/missing groups refuse. Shared-parent conventional
matrix controls initially require uniform ratios, as VGG controls do today.
Group allocations are not a class-aware allocation/search implementation.

## 5. Rebuild and state preservation

Implement a configurable ResNet18 factory with four stage output widths and eight
hidden widths. Existing torchvision BasicBlock assumes equal internal/output
widths; do not silently use that constructor for arbitrary pruned layouts. A
small explicit BasicBlock implementation may keep canonical module/state keys
while allowing conv1 output width to differ from conv2 output width. Preserve
forward order, in-place ReLU behavior and shortcut semantics. The default dense
factory must remain compatible with existing dense ResNet checkpoints.

Validate the whole graph before construction. Construct temporary modules without
advancing fitting RNG, then copy selected values under no_grad. Preserve source
device, floating dtype and train/eval mode. Copy BN affine values, running mean,
variance, batch counter, eps and momentum; reject nonstandard BN options rather
than guessing. Preserve source weights/buffers and optimizer objects unchanged.
Child recovery creates its own optimizer after rebuilding, with no inherited
optimizer state. Retain meaningful parameter requires_grad settings; include any
unsupported mixed freezing policy in early validation rather than silently reset.

For r=0, dense and rebuilt logits must agree within the documented dtype/device
tolerance. For r>0, equality to the ungated dense model is **not** required.

## 6. Checkpoint and configuration contract

Extend current checkpoint format 2 with an architecture-specific optional field;
keep dense ResNet, MobileNet, VGG and legacy readers backward compatible.
Proposed ResNet `architecture_config`:

```json
{
  "classes": 2,
  "resnet_channels": {
    "version": 1,
    "stage_widths": [32, 64, 128, 256],
    "hidden_widths": [[32, 32], [64, 64], [128, 128], [256, 256]]
  }
}
```

These example widths are illustrative; use actual saved topology values for runs. Keep the existing
VGG `channels` list contract unchanged. Validate dimensions, positive exact ints,
class count and tensor shapes before load_state_dict(strict=True).

Pruning metadata retains `type=channel` and architecture-specific channel_config:

- `layout=resnet18-dependency-v1`, method `l1-output-channel` or
  `random-output-channel`, score_rule for L1, ranking_seed only for random;
- exact group order, original widths, requested ratios and retained original IDs;
- `batch_norm` map of BN module names to exact eps/momentum, validated and restored;
- resulting stage/hidden widths, matching architecture_config exactly;
- original dense parent checkpoint binding through existing config hash/path rules.

Freeze exact schema keys during package R1, not ad-hoc per caller. Dispatch
metadata validation by architecture/layout; never route ResNet metadata through
VGG's eight-convolution validator. Recompute random IDs from saved seed/order to
reject inconsistent metadata. L1 requires valid topology/IDs and parent provenance;
score replay requires the parent and is tested separately, not fabricated at
standalone checkpoint loading. Unknown layouts/rules fail loudly.

Only tensors + validated metadata are saved, not pickled module objects. Fresh
processes reconstruct physical topology with no parent object, custom closures,
network downloads or GPU availability. Parent bindings are still checked when
used through phase-aware development/final. Exact resume remains stricter than
using a compatible older dense parent as input to a new child.

## 7. CLI, phases and fair experiment integration

Supported public prune command:

```powershell
python -m breakhis prune --dataset breakhis --checkpoint ../checkpoint/binary/40X/resnet/PARENT/train.pth --method channel --ranking l1 --sparsity 0.5 --experiment-id RESNET_L1
```

Use the same source/manifest/magnification options as the parent. Random changes
only ranking and adds `--ranking-seed 202`. Architecture is read from the parent,
not guessed from a filename. Existing VGG command behavior stays unchanged.

Integrate through these boundaries:

| Surface | Required behavior |
| --- | --- |
| CLI / pruning dispatch | Architecture-specific selection, graph validation and physical rebuild; help/errors describe support precisely |
| Parent initialization | `prune_from_parent` accepts supported dense ResNet; verifies same parent/seed/recipe/data and freezes group policy |
| Scratch config | Extend baseline.channels by architecture: VGG retains list; ResNet uses the validated resnet_channels object above; topology_checkpoint binds the same parent's L1 topology |
| Fresh scratch | Constructs the same narrow ResNet shape with fresh conv/classifier/BN state, no copied parent numerical tensors; B+F own fit |
| Recovery / continuation | L1, random, masks and dense continuation each use fresh F fitting from the seed's identical selected parent; parent B and selected/executed epochs remain attributable |
| Phase artifacts | Preserve dense/pruned/finetuned roles, designated finetuned role, immutable hashes, actual histories and explicit budget summaries |
| Checkpoint location | Existing parent-directory checkpoint/binary/magnification/model/run layout; no new output tree |
| Development / final | Validation only during development; explicit final needs test pixels only, supports CPU loading of CUDA-origin weights |
| Study planner / M2.4 | Explicit revised ResNet job matrix, parents/topology bindings and cost buckets; no silent mutation of existing VGG plans or pretending existing draft jobs run |
| Saved comparison / M2.5 | Preserve current refusal boundaries until comparator is extended; do not bypass refusal just to display ResNet results |
| Results dashboard | Architecture/role/split labels and configurations remain attributable; no automatic winner across differing settings |

Keep the default research task binary, 40X for pilot, pooled magnifications for
main, unchanged manifest/leakage checks, aggregation and operating point. ResNet
adds an architecture stratum; equal sparsity across VGG and ResNet does not imply
equal MACs, latency, capacity or training cost. Compare ranking rules at identical
ResNet widths first. Report measured resources and failures honestly.

## 8. Ordered implementation checklist

R1-R5 are implemented. Software evidence is recorded below; real measured
experiments remain M2.4/M2.5 work, with no extra approval ceremony.

| Package | Deliverables | Acceptance tests | Status |
| --- | --- | --- | --- |
| R1: graph and topology schema | Supported-layout validator; 12 groups and dependencies; exact architecture/metadata schemas; configurable factory | T01-T04 | Reviewed |
| R2: ranking and physical surgery | L1/local-random selectors; coordinated copy/BN/shortcut/classifier rebuild; parameter/MAC checks | T05-T10 | Reviewed |
| R3: portable checkpoints and public CLI | Strict reconstruction; no-download load; ranking/group flags and early errors; old-format regressions | T11-T13 | Reviewed |
| R4: experiment controls and recovery | Parent pruning, fresh scratch, continuation, exact resume, phase references and budgets | T14-T17 | Reviewed |
| R5: public integration and documentation | Generated-data fresh-process CPU/CUDA checks; study integration handoff; docs/dashboard verification | T18-T20 | Reviewed |

Likely code targets: `breakhis/models/resnet.py` (new configurable architecture),
`pruning.py` or a focused `resnet_pruning.py`, `checkpoints.py`, `cli.py`,
`baseline_init.py`, `development.py`, `phase_artifacts.py`, configuration validation
and `baseline_plan.py` only where architecture-specific bindings need extension.
Reuse training, metrics, timing and final evaluation instead of parallel workflows.
Keep tests focused in new ResNet topology/surgery/integration modules plus existing
regression suites. Avoid adding a generic arbitrary-network graph framework.

## 9. Acceptance test matrix

Use tiny generated-image fixtures and small valid topology variants where possible;
include the actual canonical dense ResNet18 where shape assumptions matter.

| ID | Required verification and passing condition |
| --- | --- |
| T01 | Exact canonical graph: 4 stage and 8 hidden groups, all producer/consumer edges covered once as appropriate; stage1 includes stem, all 3 projections and fc input included |
| T02 | Known distinct IDs propagate across each identity addition; same-width mismatched IDs refuse; preserve original transitions even if pruned adjacent widths happen to match |
| T03 | Positive int width/schema checks; reject bool/float/zero/extra/missing fields, unsupported blocks/groups/dilation, masked/tied/re-pruned/nonfinite sources before mutation/publication |
| T04 | Dense configurable factory matches canonical torchvision state keys/shapes, strict dense load and logits at r=0; no download or RNG drift |
| T05 | Hand-computed producer-normalized L1 scores, deliberate ties and minimum-channel rounding match expected original IDs; selection is computed from untouched parent for all groups |
| T06 | Local random exact reproducibility/order, valid seed limits, same retained widths as L1; global Python/NumPy/CPU/CUDA RNG and parent tensors unchanged |
| T07 | Exact tensor-slice checks for stem, every conv axis, every BN field/counter, 3 projection paths, all classifier columns and bias; source mode/device/dtype respected |
| T08 | Independent dense gated-reference oracle equals rebuilt eval logits at nonzero pruning: gate all group coordinates after BN/at residual boundaries, independently of the surgery implementation; cover non-prefix different IDs, unequal stage/hidden widths and downsampling. Exercise input sizes 64 and 224 with batch 1 and 2 |
| T09 | Forward/backward/optimizer step produces finite outputs/gradients; hooks confirm equal branch shapes at every add and expected IDs through oracle; zero-ratio eval parity CPU atol=1e-6/rtol=1e-5, CUDA fp32 atol=1e-4/rtol=1e-3 unless measured justified numerical tolerance documented |
| T10 | All selected groups physically shrink at nonzero uniform ratio; actual parameter count and dense Conv/Linear MAC count decrease; no mask/gather/padding stand-in; no assumed latency speedup |
| T11 | L1 and random save/reload in fresh CPU process reproduces logits/topology; CUDA-origin can load on CPU without CUDA or parent model object; classifier class order unchanged |
| T12 | Mutated widths/group IDs/seed/ratios/layout/classifier shapes fail; random seed replay matches saved IDs; old dense ResNet, VGG channel/mask, compact and legacy fixtures still load as before |
| T13 | Public architecture dispatch and help tested; invalid per-layer/group settings fail before pixels/outputs, defaults remain L1 and no ranking seed; unsupported depthwise/Bottleneck still refuse |
| T14 | L1/random/mask children bind identical selected dense ResNet parent per seed, fresh optimizer/scheduler and same F; immutable parent survives child runs; changed recipe/batch/B/ratio binding refuses |
| T15 | Narrow scratch has exact matched topology and fresh conv/BN/classifier state; B+F own fit and separate topology-discovery cost; dense continuation uses identical parent and fresh F, no fake train history |
| T16 | Interrupt after a committed recovery epoch then resume: final weights, optimizer/scheduler, RNG, numerical history and designated selection match uninterrupted run under supported same-device deterministic settings |
| T17 | Incomplete prune publication retries only owned uncommitted products; completed checkpoint/parent/report tampering refuses; no parent mutation or stale-resume acceptance |
| T18 | Generated-image fresh-process parent + L1/random/mask/continuation/scratch development and explicit final succeeds; remove test pixels during development, remove fitting pixels during final, trip training calls in final; reports identify architecture/role/ranking/split |
| T19 | CUDA actual fit/prune/recover and CPU final portability; report CUDA-only skips explicitly on CPU environments. CPU timing and RTX timing remain separate; existing full CPU and relevant CUDA regressions pass |
| T20 | Docs/config examples/help/dashboard reflect physical support and exact limits; existing VGG frozen study is unchanged; revised ResNet plan has correct job/cost counts and explicit unresolved/resolved bindings as appropriate. No test pixels or training during plan-only preparation |

The gated oracle must not call the pruning rebuild or reuse its sliced tensor
implementation; independent masking of the original dense graph catches semantic
errors that shape-only forward tests miss. BN gating is essential: zeroing only
convolution weights leaves BN offsets/running statistics able to affect results.
On nonzero pruning compare to this gated reference, not to ordinary dense logits.

## 10. Verification commands and evidence to retain

Implemented focused suites: `test_resnet_channel_surgery.py`,
`test_resnet_channel_integration.py`, `test_resnet_channel_study.py`:

```powershell
python -m unittest discover -s tests -p "test_resnet_channel_*.py"
python -m unittest discover -s tests
python -m breakhis prune --help
python tools/build_docs.py
python tools/build_results.py
```

Run focused suites in both existing CPU and CUDA environments. Record actual
case counts, skipped reasons, device/runtime, commands and ordinary artifact/log
paths below after execution. Generated-image public commands should be run in
fresh processes, with no real-data training required for software acceptance.
Future measured ResNet cost pilot uses the accepted real dataset in M2.4; launch
full pooled runs only after cost estimates/job accounting are updated. Do not
invent a completion date or reuse VGG throughput as measured ResNet throughput.

### Progress evidence

- R1-R3: exact graph/metadata/factory, coordinated surgery, normalized L1 and local
  random, independent gated oracle at 64/224 pixels and batches 1/2, portable strict
  reconstruction including BN settings, public CLI/group validation.
- R4: same-parent controls, fresh scratch/continuation, F versus B+F accounting,
  exact interrupted recovery, failed publication retry, tampering and role isolation.
- R5: all 16 focused cases passed in the CUDA environment in 52.958s;
  `artifacts/resnet-cuda-reviewed.log`. Actual CUDA fitting and CPU final verified.
- CPU focused: surgery 9 passed/one CUDA-only skip (5.397s), integration 3 passed/
  one CUDA-only skip (41.622s), study two passed (1.680s). Logs:
  `artifacts/resnet-surgery-tests.log`, `resnet-integration-tests.log`,
  `resnet-study-tests.log` under the same artifacts directory.
- Full CPU regression: 201 cases in 280.079s, 195 passed/six CUDA-only skips;
  `artifacts/resnet-regression-reviewed.log`. Existing VGG/dense ResNet/MobileNet/mask
  and phase-contract regressions passed.
- Fresh-process generated-image workflow: parent, L1/random/mask, continuation,
  matched B+F scratch and compact controls; explicit final for all six children.
  Commands/logs/summary: `artifacts/resnet-channel-public-smoke-reviewed/`;
  driver `artifacts/resnet-channel-smoke-reviewed.py`, output
  `artifacts/resnet-public-reviewed.log`. Immutable parent/source references reverified.
- Explicit opt-in study schema 2 adds `pruning_models`, with example
  `configs/baselines/m2_study_resnet.json`. Real metadata-only preparation passed:
  `artifacts/baselines/m2-resnet-reviewed-003/`, log
  `artifacts/resnet-study-reviewed.log`. Counts: 41 required + 3 optional + 6 pilot,
  maximum 50. Original schema-1 defaults/counts remain 22 + 3 + 6 = 31.
- Final review additionally rejects shared BN parameter storage and masking of
  narrow ResNet topologies. The complete CPU/CUDA/public workflows were rerun
  after these changes. Nonuniform group-map public CLI also passed:
  `artifacts/resnet-group-cli.log`. Source and phase references verified again.
- T01-T20 have software evidence across these suites/workflows. No real ResNet
  training, converged baseline, speedup or O1/O2/O3 result is claimed. Study stages
  beyond prepare still refuse; cost estimates remain unmeasured and comparisons
  retain M2.5 boundaries.

## 11. Final alignment and handoff checks

- [x] Full residual-stream pruning implemented: stage1/stem, all stage outputs,
  transition projections, hidden channels and classifier inputs, not hidden-only.
- [x] Conventional ranking is fully specified and recorded; L1/random exact widths
  and parent/recovery policies match. This supplies controls, not O1/O2 novelty.
- [x] M1 role isolation, leakage checks, frozen configs, checkpoint references and
  explicit final evaluation remain intact.
- [x] VGG/MobileNet/dense ResNet behavior and existing study artifacts stay compatible.
- [x] M2.4 has explicit ResNet parent/control jobs and separate cost buckets;
  actual budget measurements still belong to M2.4;
  M2.5 empirical comparison work remains separately tracked.
- [x] Architecture-level conclusions await actual matched experiments; all four
  magnifications remain the planned main scope and 40X remains the pilot.

Software completion handoff: mark M2.3-R reviewed, then use it for the explicitly
expanded M2.4 baseline study. The next scientific goal remains credible trained
baselines, not an additional approval framework or a claim that more architectures
alone establishes the proposed research novelty.
