> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.3 data readiness API

Implemented in `breakhis/research/data_readiness.py`, exposed through the lazy
`breakhis.research.m0.audit_data` adapter. Existing `prepare-data`, manifests,
training and checkpoints are unchanged. This package creates no folds or fits.
Real local M0 execution remains coordinated in M0.6; acceptance uses synthetic
temporary images, not the user's BreaKHis files.

## Calling the audit

```python
from pathlib import Path
from breakhis.research.records import create_run_directory
from breakhis.research.m0 import audit_data

repo = Path.cwd().resolve()  # repository root
data = repo.parent / 'Dataset/dataset_cancer_v1/classificacao_binaria'
run_id = 'data-audit-unique-id'  # choose a new ID for each invocation
run = create_run_directory(repo / 'artifacts/research/m0', run_id,
                           protected_roots=[data])
result = audit_data(data, run, run_id, repo_root=repo, workers=4)
print(result['counts'], result['blockers'], result['restrictions'])
```

This example performs a real read-only audit if explicitly run. Use a synthetic
fixture root during development. Images and old manifests are never rewritten.
All output goes into a new `audit/` below the supplied run; an existing audit
directory is refused. Paths must not overlap the source. Symlink/junction roots
and child directories are refused; escaping or missing image paths are excluded
and block data-integrity readiness. Containment follows M0.1's static filesystem
contract, not a hostile concurrent filesystem guarantee.

`workers` is a positive integer, default 1. `repo_root` defaults to the installed
source repository; explicitly set it when using temporary fixture repositories.
Optional `lineage` is a validated input object described below. Optional
`legacy_manifest` is a path to an existing, binary-only legacy v1 manifest; it
is read and hashed, never rewritten. Invalid reuse blocks data readiness.

## Records and accounting

| Output under `audit/` | Meaning |
| --- | --- |
| `inventory.jsonl` | One v1 `image_inventory_row` per supported image discovery, including failures |
| `exclusions.jsonl` | Full excluded rows, not just an aggregate count |
| `duplicate_components.json` | Stable exact-content/full-filename-group linked components; label conflicts explicit |
| `group_summary.json` | Counts, distributions, dimensions, group support, exclusions and historical discrepancies |
| `lineage.json` | Evidence-status declarations, reviewer, unknowns, owners and claim restrictions |
| `near_duplicate_review.json` | Explicitly incomplete perceptual screening; no similar-tissue auto-removal |
| `manifest.json` | Research inventory references; no train/validation/test assignment generated |
| `audit.json` | Source identity, execution metadata, counts, blockers, restrictions and optional legacy validation |

JSON records use the M0.1 envelope. JSONL row fields follow section 5.3 of the
[master spec](../../../specs/m0/M0_SPEC.md), with `record_type`, `issues` and `unavailable_reasons`
added for machine-readable failures. All supported discoveries appear even if
parsing, reading or decoding fails. Raw bytes are hashed independently of name
validation and decoding. Unknown fields remain null with reasons. Non-image
paths are listed and counted separately; linked directories are reported without
traversing them. Discovery errors, file-set changes and byte changes block data
integrity. Missing/empty/all-invalid sources produce explicit blocked records.

The existing canonical filename parser supplies binary label, subtype,
magnification and full YEAR-SLIDE_ID including suffixes. Supported non-PNG
extensions use the same filename stem grammar; decoding checks actual content.
Dimensions are measured. Decoded hashes reuse the existing
`SHA256(ASCII(width + 'x' + height + ':') + RGB bytes)` semantics.

Image IDs derive from source plus normalized relative path. Component IDs derive
from sorted image membership under versioned transitive raw-hash, decoded-hash
and full-filename-group linkage. Identity links span magnifications. Stable
pixel-equivalent representatives use the lexicographically smallest eligible
relative path. Every duplicate exclusion references that representative. Mixed
label components block readiness; their eligible members retain their labels
and are not silently discarded to resolve the conflict. Excluded rows also
participate in linkage whenever their hashes/group metadata are available.

Discovered = included + excluded. Every class/subtype/magnification/group bucket
reports these three counts, with unknown metadata in an explicit unknown bucket.
Group support lists available/missing retained magnifications; no imputation is
performed. Historical 7,909 / 7,784 / 125 / 81 figures are comparison references
with observed deltas requiring review, never enforced totals or identity proof.

## Lineage declarations

Input shape: `{"schema_version": 1, "reviewer": null, "facts": {...}}`.
Each fact has exactly `value`, `status`, `reason`, `evidence_refs`, following
M0.1 fact validation. Allowed fields:

```text
source_release, acquisition_date, archive_sha256, original_url, citation,
usage_terms, terms_review_date, derivative_parent, transformation_parameters,
transformation_code_history, backup_location, linkage_confidence,
clinical_identity
```

Omitted facts become unknown/null, with an open question owned by
`research_owner` and a review deadline of M1. Self-reported facts remain
self-reported. Verified/disputed declarations require a named reviewer; evidence
references are declarations for later review, not automatically authenticated
external sources. No download or browsing occurs. Unknown keys/versions/malformed
facts fail before audit output creation. The resolved lineage payload is hashed.

Clinical identity is never inferred from the directory name, a published group
count or even a general verified lineage statement. Image-specific clinical
identity mapping is not implemented here: `verified_patient_id` and
`identity_evidence_ref` remain null, and terminology remains filename/slide groups.

## Readiness boundaries

`data_integrity_ready` means only that this inventory passed its data consistency
and label checks. `status` remains restricted when those checks pass. Exploratory
and confirmatory readiness remain false: lineage scope, exclusions, exposure,
protocol, fold support and similarity review need downstream adjudication. A
lineage exception needs an explicit decision in later governance, not an
automatic scientific pass.

Perceptual near-duplicate screening is deliberately recorded as incomplete,
algorithm `not_run`, threshold null with a reason, and no approved links or
automatic similar-image exclusions. Exact hashes do not prove the absence of
near-duplicates. Supplied suspicious-pair adjudication is not implemented in this
version and cannot be represented as completed review.

Legacy reuse requires an exact retained file set and matching raw/pixel hashes,
labels, magnifications and full filename groups. Group, duplicate-component and
hash boundaries must stay within a single partition. Validated assignments are
always labeled `legacy_pilot_assignment`, never research outer folds; they make
no claim that groups are historically untouched. Reseeding is not an exposure fix.

## Developer validation

```powershell
python -m unittest discover -s tests -p test_m0_data.py -v
python -m unittest discover -s tests -v
```

See [package acceptance and evidence](M0_3_DATA_READINESS.md).
