# M0 specification: preservation, protocol and data readiness

Specification: M0-SPEC-001, version 1.0. Prepared: 7 November 2026.
Status: specification for implementation; creating this document does not execute M0.

Parent: [research implementation roadmap](../../research/RESEARCH_IMPLEMENTATION_ROADMAP.md), M0/G0. Scientific intent: [research guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md). Experiment mapping: [operational plan](../../research/RESEARCH_PAPER_PLAN.md), P0/E01. Existing behavior: [dataset audit](../../guides/DATASETS.md) and [testing](../../guides/TESTING.md).

MUST denotes an acceptance requirement; SHOULD permits a justified documented exception; MAY is optional. Proposed files, schemas and commands below are development targets, not existing capabilities. This document governs M0 coding, review and testing. M1 implementation and scientific experiments are outside its execution scope.

## 1. Objective and completion boundary

M0 MUST produce an attributable code/environment baseline, auditable BreaKHis inventory, source/identity uncertainty records, historical test-exposure register, focused literature matrix, initial protocol/decision register and evidence-backed Gate G0.

The handoff must answer: which code/environment produced the evidence; which images/groups exist and are excluded; which provenance/identity facts are known; which groups have prior exposure; which decisions remain proposed; and which kinds of subsequent work are eligible.

Completing M0 does not prove clinical identity, novelty or scientific model quality. Unknowns MUST become named restrictions, not silently passing checks. M1 engineering readiness, exploratory-pilot readiness and untouched confirmatory-design readiness are separate outcomes.

## 2. Scope and inputs

| Input | Required interpretation |
| --- | --- |
| Repository | `D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification` |
| Data | `D:\Sutanu_WorkSpace\BreakHis\Dataset\dataset_cancer_v1\classificacao_binaria` |
| Source/task | `cancer_binary`; benign/malignant; all available 40X/100X/200X/400X images |
| Group identity | Full YEAR-SLIDE_ID including suffixes; filename/slide groups unless independently verified |
| Working images | Retained 224 x 224 derivative; measure dimensions rather than assume them |
| Hardware | CPU primary deployment target; RTX 5070 Ti training/secondary inference target |
| Historical inputs | Existing manifests, audit/checkpoint configurations, logs and prior evaluation metadata |
| Time box | Day 1 target; unresolved investigations retained with owners/deadlines |

Original-resolution backup location and transformation lineage remain unknown until evidence is supplied. The deleted v2 folder MUST NOT be assumed present. The multiclass folder is another label arrangement of BreaKHis, not an independent dataset.

Out of scope: research outer folds, training/search, candidate pruning, threshold fitting, vulnerability scoring, second-dataset downloads, manuscript result claims and automatic Git commit/push. M0 MUST NOT run test inference to reconstruct historical exposure.

## 3. Proposed development structure

| Target | Responsibility |
| --- | --- |
| `breakhis/research/records.py` | Record validation, canonical hashes, safe paths, atomic JSON writes |
| `breakhis/research/m0.py` | Source/environment capture, audit orchestration, exposure/protocol validation and gates |
| `tools/run_m0.py` | Thin runner and read-only evidence verifier |
| `configs/research/m0.example.json` | Documented input configuration with unknowns explicit |
| `configs/research/protocol.initial.json` | Initial declarations and pending scientific decisions |
| `docs/research/RESEARCH_PROTOCOL.md` | Human-readable protocol and restrictions |
| `tests/test_research_records.py` | Schema/path/hash unit tests |
| `tests/test_m0.py` | Synthetic integration, exposure propagation and gate failures |

Reuse existing manifest parsing, inventory, hash/duplicate/group logic, environment checks and regressions where semantics match. Do not duplicate the loader/pruner or reorganize the canonical package. Preserve existing CLI, historical manifests and checkpoint compatibility. Prefer existing dependencies/standard library; a new package needs justification and environment/CI updates.

## 4. Run lifecycle and filesystem contract

Each unique run writes under ignored `artifacts/research/m0/<run_id>/`:

```text
run.json
config.resolved.json
baseline/
  git.json
  tracked.patch
  source_inventory.json
  source_snapshot/
  environment.cpu.json
  environment.cuda.json
  checks.json
audit/
  manifest.json
  audit.json
  inventory.jsonl
  exclusions.jsonl
  duplicate_components.json
  group_summary.json
  lineage.json
  near_duplicate_review.json
governance/
  exposure_register.json
  protocol.initial.json
  decisions.json
  literature_matrix.json
logs/
acceptance/
  evidence_index.json
  gate_g0.json
  M0_REPORT.md
```

Requirements:

- Create a new run directory; refuse an existing nonempty destination. Resolve paths before IO. Reject traversal, dataset/output overlap, escaping symlinks/reparse points and recursively capturing outputs as source.
- Read images/historical artifacts without overwriting, moving or deleting them. A new audit never replaces an old experiment manifest. M0 creates no model checkpoints; later weights remain in parent `checkpoint/binary/`.
- Run states: `created`, `running`, `completed`, `blocked`, `failed`. `completed` means evidence/report generation completed; scientific readiness is determined separately.
- Log operation ID, start/end UTC, argv/cwd, input hashes, return code, elapsed time and output/log references. Keep partial evidence on failure; never manufacture empty successful outputs.
- Version 1 does not silently resume/overwrite partial runs. A new run may reference a failed predecessor.
- Write UTF-8 JSON with finite numbers and atomic sibling-temp replacement. Store UTC ISO-8601 timestamps; IDs derive from content/identity rather than row order. Unsupported schema versions fail clearly.

## 5. Record schemas and semantics

All generated JSON record roots MUST contain `schema_version`, `record_type`, `run_id`, `created_at_utc`; collection records use an object envelope with an `items` field. Pre-run templates have their own validated input type and are bound to a run when copied. The evidence index declares JSONL row schema versions/types. Null means unavailable with an accompanying reason, never an invented observation. Validators distinguish malformed records from valid records with unresolved scientific facts.

M0.1 schema binding: object-record payload fields described below live under
`data`; collections use `items`. The common envelope validator enforces those
exact roots, while later packages validate domain payloads and readiness. See the
[implemented records API](M0_1_RECORDS_API.md) for the concrete pre-run configuration
keys, hash policy and evidence-interface boundaries. This binding does not mark
M0.2-M0.6 implemented or establish Gate G0.

### 5.1 Run and source preservation

`run.json`: spec ID/version/hash, requested mode (`pilot` or `confirmatory_design`), invocation/config hash, resolved repo/data/output paths, predecessor if any, state/times, and gate/report references.

`baseline/git.json`: HEAD commit/ref, dirty state, staged/unstaged status and relevant tracked/untracked file inventory. Capture HEAD plus both staged and unstaged changes and relevant untracked source/protocols. A commit hash alone MUST NOT identify a dirty run.

Store a binary-capable tracked diff against HEAD and source copies/hashes sufficient to reconstruct the relevant tree. Include package/tools/tests, project/dependency files, workflows and selected docs/configs. Exclude `.git`, environments, datasets, weights/checkpoints, generated output and unrelated personal configuration. Record inclusion/exclusion rules and unresolved LFS/submodule references. Do not place raw clinical-identity documents in a public-ready source snapshot.

Inventory fields: repository-relative POSIX path, tracked/untracked status, bytes, SHA-256, capture status. Recheck status/hashes after capture; retry once or fail if source changes. Define `source_snapshot_sha256` over sorted path/hash pairs plus capture policy, excluding root paths/timestamps. The snapshot MUST NOT recursively include itself.

### 5.2 Environment and current baseline checks

Capture Python/interpreter/venv, dependency/build versions, pip consistency, OS/CPU/threads, and available GPU/CUDA/runtime/driver identifiers. Missing optional utilities produce `unknown`, not guessed versions. Reuse the existing actual synthetic tensor/VGG check where applicable.

Each `checks.json` item: check ID, argv/cwd, device, status/return code, elapsed time, log paths and hashes. Run current regressions and import/CLI-help checks; historical counts cannot substitute for this run. CPU checks MUST run without CUDA. A failed/unavailable GPU is an explicit GPU-readiness restriction; CPU-only M1 unit development may still proceed.

### 5.3 Complete image accounting and grouping

Every discovered supported image, including failures, appears in `inventory.jsonl`:

```text
schema_version, image_id, source, relative_path, discovered_bytes,
status(included|excluded), exclusion_reason_code, exclusion_detail,
raw_sha256, decoded_pixel_sha256, width, height,
class_label, subtype, magnification, filename_group_id,
verified_patient_id, identity_evidence_ref, duplicate_component_id,
representative_image_id
```

Unavailable parse/decode fields are null with reasons. Hash raw bytes even when decoding fails if the bytes are readable. Count/list ignored non-image files separately. Reject image paths escaping the source.

`image_id` derives from source plus normalized relative path; content hashes remain separate. Pixel hashing follows existing RGB dimension/content semantics. Component IDs derive deterministically from canonical membership/linkage; record grouping/hash-policy versions.

Link exact duplicates before partition analysis. Preserve all raw inventory rows, select representatives deterministically for within-source duplicate exclusions, and record every representative/reason. Conflicting labels in duplicate-linked/identity components block scientific readiness; never fix them by arbitrary ordering.

Preserve full filename suffixes and cross-magnification identity. Do not merge groups to force a published count. `verified_patient_id` requires authoritative evidence; otherwise use filename/slide terminology.

`group_summary.json`: discovered/included/excluded counts by class/subtype/magnification/group; group-size distribution; dimension distributions; duplicate/component sizes; per-group magnification support; missing/corrupt files; historical reconciliation and open discrepancies. Historical 7,909/7,784/125/81 values are references to investigate, not expected constants to force.

If reused preparation produces a 70/15/15 manifest, label it `legacy_pilot_assignment` and validate exact group/hash separation. It is NOT the research outer-fold design. M1 consumes frozen inventory/components to create research folds. A new split seed never makes exposed groups untouched.

### 5.4 Source lineage and near-duplicate review

`lineage.json`: source/release identifier, acquisition date/archive hash if known, original URL/citation/usage terms and review date, derivative parent, transformation parameters/code/history, backup location if supplied, linkage confidence, evidence references, reviewer and open questions.

Every material fact has status `verified`, `self_reported`, `unknown` or `disputed`. Unknown dates/transforms remain null and generate open decisions. A directory name is not evidence of an official release. Restored originals require a separate lineage/audit and are not independent samples to concatenate.

`near_duplicate_review.json` distinguishes exact-hash checks from perceptual similarity. Version 1 MAY use bounded deterministic nonlearned screening or explicitly record incomplete screening. Record algorithm/threshold, candidate pair IDs, review/evidence, approved linkage/exclusion and unresolved pairs. Never automatically remove similar but potentially distinct tissue fields. Unreviewed suspicious cross-group pairs restrict confirmatory design; incomplete screening is not proof of no near-duplicates.

### 5.5 Historical test-exposure register

Inspect only metadata/configuration/log material necessary to establish access. Do not rerun old test inference, recompute test metrics or inspect error/probability values to decide which groups count as exposed.

Each entry:

```text
exposure_id, artifact_ref, artifact_sha256, occurred_at_or_unknown,
evidence_source(config|log|manifest|operator_attestation),
manifest_ref_and_hash, source, selected_partition, magnification_filter,
access_type(train|selection|threshold|test_metric_inspection|
            test_prediction_inspection|synthetic_check|unknown),
image_ids, filename_group_ids, duplicate_component_ids,
coverage(exact|conservative_superset|unresolved), confidence,
restriction(development_only|synthetic_not_real|unresolved), reviewer_note
```

For real-image evaluate/compare smoke history, resolve exact historical manifests and selected test rows. Without per-image access logs, conservatively include the whole selected partition/filter as potentially inspected. Propagate restrictions across duplicate-linked components and all magnifications. A small filtered run need not expose unrelated groups; absent evidence cannot prove they were untouched.

Synthetic checks have separate source identity and MUST NOT quarantine real groups merely because generated filenames resemble BreaKHis. Operator recollections are self-reported, not automatically exact coverage.

Component eligibility: known exposed/development-only; exposure unknown/requires adjudication; or candidate untouched within a stated reviewed coverage boundary. No-log evidence alone cannot mean verified unexposed. Require a reviewed history-coverage declaration before confirmatory eligibility. Missing manifests, unresolved references and contradictions remain open blockers.

M0 records eligibility, not folds. It cannot select held-out groups using past metric quality. Confirmed equivalent identities/content across derivative copies carry the same exposure restrictions.

### 5.6 Initial protocol and decision register

Protocol lifecycle: `draft`, `pilot_ready`, `initial_design_reviewed`. M0 MUST NOT emit `confirmatory_frozen`; final method/outer evaluation freeze belongs to M6.

For each topic store value/null, status `agreed|proposed|deferred|blocked`, rationale/evidence, owner and freeze-by milestone. Mandatory topics:

- Task, derivative/source, magnifications, grouping/component policy, identity terminology and exposure restrictions.
- Proposed primary magnification-balanced pooling, secondary image-weighted pooling, available-only missing-magnification handling and no imputation.
- GPU training; CPU batch-one primary timing; GPU secondary timing; provisional CPU threads 2 pending timing pilot.
- Eight-convolution custom VGG, proposed MobileNetV3-small comparator and ResNet18 role subject to cost; no implied VGG16/broad channel surgery.
- Outer/inner roles, calibration/selection compromises, independent split/training/pruning/search seed namespaces.
- Separate epoch/candidate selection, threshold calibration, specificity safeguards, numerical margins and absolute-performance reporting.
- Dense/adaptation/search/recovery budgets, latency regions, matched comparison rules and Day-3 compute gate.
- O1/O2/O3 hypotheses; A/B/C/D/E comparisons; random/scratch/compact controls; negative-result path.
- Group-paired uncertainty, fold/seed reporting, primary/secondary endpoints and timing/session scope.
- Limits on clinical, noninferiority, novelty and generalization claims.

Unknown numerical thresholds, margins, trial counts and identity facts remain proposed/unresolved. No default silently becomes an agreed clinical/scientific decision. M2/M5 choices may remain open for M1 interface development if owners/freeze deadlines are explicit.

`decisions.json` preserves revision history: stable decision ID, revision, topic/status, rationale/evidence, owner role, deadline milestone, affected claims and supersedes reference. Roles such as `research_owner` are allowed when no person is assigned; critical unassigned reviews must be surfaced.

### 5.7 Focused literature matrix

Required columns: literature ID; exact title/authors/year/venue/publication status; DOI/URL; access/review date; access level `full_methods|abstract_only|code_only|unavailable`; task/dataset; grouping; ranking/allocation; class/sample balancing; selection/threshold rule; search/adaptation/recovery budgets; runtime target; code/license; comparator role; exact distinction/unresolved overlap; reviewer and open questions.

Cover the guide's closest hardware-aware, class-aware and medical sample-vulnerability methods as review targets. Abstract/code access cannot be marked full-method review. A released repository does not establish faithful reproduction. Central overlaps require full methods or explicit unresolved limits before final novelty claims. Novelty is not an automated boolean.

Blocked access may remain in M0 with named restrictions/owners. It need not block infrastructure development, but MUST block claims that the closest-method review is complete. Cite the version actually read and distinguish preprint from published status.

## 6. Integrity and reproducibility contract

`evidence_index.json` lists relative artifact path, type/schema, bytes, SHA-256, producing operation and status. File hashes cover actual bytes. Semantic source/inventory hashes use separately documented canonical serialization. Relocation/timestamps may change file hashes without changing semantic content; these digests are not interchangeable.

Verify references/hashes before gating. Avoid circular hashing: index stable input/evidence artifacts first, excluding the index itself; gate references index digest; human report references gate/index. An optional final receipt can hash completed gate/report afterward. Missing/failed artifacts cannot appear as empty successful evidence.

Detect changed/unreadable files during capture and source/file-set changes at audit boundaries. Hashes describe bytes actually read; stable timestamps alone do not prove immutability. M1 must reverify image identity or retain existing per-read hash checks. Original manifests/checkpoints stay intact; new run identity binds inventory, exclusions, grouping/hash policy and governance versions.

## 7. Proposed runner interface

Design target only; these commands MUST NOT be represented as currently runnable:

```text
python tools/run_m0.py --config configs/research/m0.example.json --run-id m0-<unique-id>
python tools/run_m0.py --config configs/research/m0.example.json --run-id m0-<existing-id> --verify-only
```

Normal flow: validate inputs; create new run; capture source/environment/current tests; audit selected images; ingest reviewed governance records; index evidence; evaluate G0; write human report. Verification-only validates existing evidence/gate consistency without rerunning tests/inference, modifying statuses or rewriting artifacts. It returns nonzero for tampered/missing evidence.

Configuration includes: schema/spec version; repo/data/output paths; selected sources (initially exactly `cancer_binary`); audit workers/seed; requested devices; baseline check identities; lineage/exposure/protocol/literature input record paths; requested mode and operation timeouts. Resolve relative config paths against the configuration file directory and record absolute resolutions. Existing CLI cwd semantics remain compatible.

Reject unknown config keys, unsupported versions, invalid IDs/numeric ranges and unsafe paths before side effects. Use an allowlisted implementation for checks; config MUST NOT execute arbitrary shell strings. Runner flags are `--config`, `--run-id`, `--verify-only` and standard help. Configuration validation errors exit 2; execution/integrity failures or blocked requested-mode gates exit 1; exit 0 requires complete engineering evidence and eligibility for the requested mode. Restricted pilot mode may exit 0 only with restrictions prominent. `confirmatory_design` checks preparation eligibility; it never launches/freezes the confirmatory study.

## 8. Gate G0 contract

Gate output contains requested mode, overall status, engineering readiness, exploratory-pilot readiness, confirmatory-design readiness, GPU readiness, novelty-review state, blocker IDs, evidence references, restrictions and M1 handoff eligibility. Each readiness dimension is `ready|restricted|blocked` with reason codes. Automated checks, reviewer attestations and unresolved facts must be distinguishable.

`overall_status` is `ready|restricted|blocked` for the requested mode. A blocker record contains stable ID, affected dimension, reason code, evidence references, owner and resolution condition. Suggested reason codes include `SOURCE_INCONSISTENT`, `BASELINE_FAILED`, `DATA_LABEL_CONFLICT`, `LINEAGE_UNKNOWN`, `EXPOSURE_UNRESOLVED`, `FOLD_SUPPORT_UNREVIEWED`, `LITERATURE_INCOMPLETE`, `GPU_UNAVAILABLE` and `EVIDENCE_INVALID`. Requested-mode hard failures take precedence over restrictions. Do not infer review completion from a missing blocker or empty reviewer field.

| Condition | M1 engineering | Exploratory pilot | Confirmatory design / claims |
| --- | --- | --- | --- |
| Inconsistent source, failed baseline regressions, missing/tampered evidence | Blocked | Blocked | Blocked |
| Missing/empty data, unresolved label conflicts, path/hash inconsistency | Data-dependent work blocked; unrelated unit work may be separate | Blocked | Blocked |
| Valid derivative, unknown original transform lineage | Allowed with limitation | Restricted derivative-only | Strong lineage/generalization claims blocked until reviewed; explicit scope decision required |
| Clinical identities unverified but filename grouping consistent | Filename-group terminology | Filename-group scope | Filename-group design may proceed; verified-patient claims blocked |
| Historical exposures resolved | Ready with restrictions handed off | Declared development groups only | Untouched support and history coverage must be reviewed |
| Exposure coverage unresolved | Synthetic/interface development allowed | Documented development-only scope | Blocked pending adjudication; reseeding is not a remedy |
| Too few eligible groups for planned folds | Interfaces can be implemented | Reduced pilot with support disclosed | Proposed design blocked until revised before test inspection |
| Central literature not fully reviewed | Infrastructure allowed | Novelty limitation explicit | Closest-method novelty/comparator freeze blocked |
| CPU passes, GPU unavailable/fails | CPU development allowed | GPU fits blocked pending repair/scope revision | Hardware plan requires explicit revision |
| Later numerical decisions proposed with owners/deadlines | Interfaces allowed | Comparisons wait for necessary decisions | Full protocol required before confirmatory freeze |

Pilot lineage exceptions require a recorded decision and stated claims, not an automatic downgrade that hides missing evidence. Candidate untouched groups are never automatically approved just because they are absent from the register. A ready engineering gate proves neither safety nor novelty. M1 engineering entry and M2 real-data scientific entry are distinct gates.

## 9. Implementation work packages

Execution is organized into six short checklists in the [M0 package tracker](README.md).
They map WP0-WP9 below into M0.1-M0.6 and track `Not started`, `In progress`,
`Tests passed`, and `Reviewed`. This master specification remains the authority
for schemas, semantics and acceptance; checklist review does not resolve unknown
scientific facts automatically.

| Order / ID | Work | Completion evidence |
| --- | --- | --- |
| WP0 | Versioned records, canonical hashes, path checks and atomic IO | Unit schema/path/hash fixtures |
| WP1 | Git/source snapshot and consistency checks | Reconstructed tracked/untracked source matches captured hashes |
| WP2 | Environment/test capture with timeouts | Real CPU check; logged failure/device-unavailable fixtures |
| WP3 | Audit adapters, complete inventory and components | Duplicate/conflict fixtures; every discovery/exclusion reconciled |
| WP4 | Lineage/near-duplicate review ingestion | Unknown/reviewed facts preserved without invention |
| WP5 | Exposure resolution and component restrictions | Exact/superset/unresolved/synthetic scenarios |
| WP6 | Protocol, decisions and literature validation | Mandatory topics; proposed/approved/reviewed distinctions |
| WP7 | Evidence index, G0 evaluator and report | Gate matrix and missing/tampered-evidence failures |
| WP8 | Runner, docs and isolated integration | Synthetic end-to-end/verify-only and existing regressions pass |
| WP9 | Real local M0 run after implementation authorization | Fresh evidence, actual baseline outcomes, G0 report and handoff |

Develop and review these in order. Code review MUST trace requirements to evidence rather than merely verify output filenames. Do not implement training/search as hidden M0 work.

## 10. Acceptance-test matrix

Coverage mapping: sections 4/5.1 map to T01-T05; 5.2 to T19-T20; 5.3 to T06-T10; 5.4 to T15; 5.5 to T11-T14; 5.6 to T16-T17; 5.7 to T18; section 6 to T21-T22; sections 7-8 to T23-T24/T26; compatibility to T25. Tests must assert gate consequences as well as record contents. An integration report lists requirement section, test ID, pass/fail/skip and evidence path; skipped required acceptance checks cannot yield an unrestricted ready gate.

| Test ID | Fixture / action | Required result |
| --- | --- | --- |
| M0-T01 | Clean temporary Git repo | HEAD/source identity reproducible; copies match hashes |
| M0-T02 | Staged, unstaged and relevant untracked files | All included; commit alone never describes dirty run |
| M0-T03 | Nearby data/weights/environments/generated outputs | Excluded by recorded snapshot policy; no recursion |
| M0-T04 | Source changes during capture | Retry/fail; never a falsely coherent snapshot |
| M0-T05 | Existing nonempty output, traversal or escaping symlink | Refusal; no historical/data overwrite |
| M0-T06 | Synthetic binary source across magnifications | Stable IDs and correct class/subtype/group/count summaries |
| M0-T07 | Same decoded pixels under alternate encoding/name/group | Pixel linkage; raw hashes distinguished; stable representatives |
| M0-T08 | Conflicting label in duplicate/identity component | Data gate blocked with evidence; no arbitrary correction |
| M0-T09 | Corrupt image, malformed name, folder mismatch, unreadable file | All discoveries accounted with reasons and available raw hashes |
| M0-T10 | Missing/empty source; absent magnification | Source blocks; missing magnification reported without imputation |
| M0-T11 | Old real test report plus exact manifest/filter | Selected components exposed; linked magnifications restricted |
| M0-T12 | Known partition but no per-image access log | Conservative selected-partition superset, explicitly labeled |
| M0-T13 | Missing manifest or contradictory exposure evidence | Unknown coverage blocks confirmatory eligibility |
| M0-T14 | Synthetic smoke with realistic filename IDs | No false quarantine of real data |
| M0-T15 | Unknown derivative transform/clinical identity | Null/open decision; truthful terminology and restrictions |
| M0-T16 | Proposed pooling/margins in protocol | Remain proposed; no silent agreement or premature freeze |
| M0-T17 | Missing critical protocol topic/owner/deadline | Validation issue and affected restriction |
| M0-T18 | Abstract-only/code-only literature access | Cannot count as full methods or established novelty |
| M0-T19 | CPU passes, GPU absent/fails | Per-device results; GPU launch restricted |
| M0-T20 | Failed/timed-out baseline command | Logs/return code retained; engineering-ready false |
| M0-T21 | Deleted/tampered artifact or unknown schema | Verification fails; no concealed recomputation |
| M0-T22 | Relocation/timestamp change versus content change | Documented distinct file/semantic hash behavior |
| M0-T23 | Valid isolated synthetic M0 run | Complete evidence/gate/report; no training/test evaluation |
| M0-T24 | Repeat verify-only on valid and invalid runs | Read-only deterministic outcome; invalid returns nonzero |
| M0-T25 | Existing CLI/checkpoint/CPU regressions | Compatibility intact; original manifests/images unchanged |
| M0-T26 | Real local execution after authorization | Fresh research-only evidence, measured outcomes and M1 handoff |

Unit/integration tests MUST use temporary repositories/data where possible and need no network/GPU except explicitly optional integrations. Literature/provenance facts require curated review; automated tests validate record semantics, not scientific truth. Test restricted/blocked gates as thoroughly as ready gates.

## 11. Definition of done and handoff

- [ ] Normative requirements implemented or covered by a justified versioned spec revision.
- [ ] Verifiable source/environment snapshot and current baseline check logs.
- [ ] Discovered/included/excluded counts and duplicate representatives reconcile.
- [ ] Lineage, identity, similarity and exposure limitations explicitly reviewed or unresolved.
- [ ] Proposed scientific decisions have owners/freeze deadlines; no premature confirmatory freeze.
- [ ] Literature matrix covers closest mechanisms with truthful access/review status.
- [ ] G0 dimensions, blockers and restrictions agree with indexed evidence.
- [ ] Synthetic acceptance and existing regressions pass; fixtures distinguished from real evidence.
- [ ] Human report lists operations, measured counts, restrictions and open questions without invented results.
- [ ] M1 receives source snapshot identity, inventory/component hashes, exposure eligibility, protocol/decision version and allowed research mode/claims.

M1 then implements folds, seeds, development predictions and aggregation against these contracts. Real scientific fits and confirmatory evaluation remain gated by unresolved decisions and later milestones. Creating this specification alone satisfies none of the execution checklist.
