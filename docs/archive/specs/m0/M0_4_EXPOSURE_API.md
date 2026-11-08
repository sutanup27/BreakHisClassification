> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.4 exposure governance API

`breakhis/research/exposure.py` resolves explicit historical metadata against an
existing M0.3 audit. `breakhis.research.m0.resolve_exposure` is the lazy adapter.
It does not discover all history automatically, read images/weights, run models,
collect predictions, compare metric quality, select held-out groups or freeze G0.

## Calling the resolver

```python
from pathlib import Path
from breakhis.research.records import read_json
from breakhis.research.exposure import inventory_scope
from breakhis.research.m0 import resolve_exposure

run = Path('artifacts/research/m0/existing-run-id').resolve()
history = Path('path/to/reviewed/metadata').resolve()
scope = inventory_scope(run, 'existing-run-id')  # read-only; not a review
print(scope)  # bind these identities into a separately reviewed declaration
result = resolve_exposure(
    run, 'existing-run-id', history_root=history,
    source_instance_id='deliberately-identified-current-derivative',
    events=read_json(history / 'events.json'),
    history_coverage=read_json(history / 'history_coverage.json'),
)
print(result['eligibility_counts'], result['blockers'])
```

This requires a current M0.3 audit, not a legacy manifest alone. Inventory bytes,
schema, component membership and linkage policy are checked before resolution.
All three output paths must be new; an existing register is never overwritten.
M0.6 will provide the full configuration/runner and evidence-graph verification.

`history_root` is the trusted directory anchoring references. Every reference is
exactly `{"path": "relative/metadata.json", "sha256": "<64 lowercase hex>"}`.
Paths follow M0.1 containment rules and cannot refer inside this run. Only
`.json`, `.jsonl`, `.log`, `.txt`, `.md`, `.yaml`, `.yml` metadata is accepted;
weights and images are not read. Opaque log/config evidence is hashed without
parsing scores or error/probability values. Historical manifests are read for
source, relative path, filename group, partition, magnification and raw/pixel
hashes only. References are rechecked before candidate statuses are assigned.

## Event schema

Every event has exactly these fields; unknown keys, including metric arrays,
are rejected:

| Field | Input |
| --- | --- |
| `event_id` | Unique M0.1 token/ID |
| `artifact` | Historical metadata reference with expected hash |
| `manifest` | Historical v1 manifest reference; may be null for synthetic checks |
| `source` | Source key used in the historical manifest |
| `source_instance_id` | Deliberately declared derivative/fixture identity, distinct from source key |
| `source_kind` | `real` or `synthetic` |
| `selected_partition` | `train`, `validation`, `test`, `all` |
| `magnification_filter` | Distinct available values `40X`, `100X`, `200X`, `400X`, or exactly `["all"]` |
| `access_type` | `train`, `selection`, `threshold`, `test_metric_inspection`, `test_prediction_inspection`, `synthetic_check`, `unknown` |
| `accessed_paths` | Distinct manifest-relative paths from per-image access logs, or null |
| `evidence_source` | `config`, `log`, `manifest`, `operator_attestation` |
| `occurred_at` | Attributable time string or null with generated unknown-time reason |
| `reviewer_note` | Scope/uncertainty text or null |

For real history, the partition/magnification filter is applied to the *historical*
manifest. Missing per-image access logs quarantine the entire selected filter as
`conservative_superset`. A nonempty consistent `accessed_paths` list with evidence
source `log` permits `exact` coverage; operator recollections remain self-reported
and conservative. Empty per-image lists, contradictory filters/content, missing
manifests, changed hashes, unknown access types and unresolved identities remain
unresolved. Known matched exposures are retained even when other evidence is
contradictory. Unknown history blocks candidate untouched claims globally.

Matching exact hashes or current-instance filename groups propagates access to
the entire linked component, including other magnifications and exact duplicates.
Unrelated groups are not exposed merely because they share a magnification.
Historical preparation seeds and new split generation cannot remove exposure.

`source_kind=synthetic` requires `access_type=synthetic_check` and a distinct
instance ID. Its valid artifact evidence produces `synthetic_not_real`, with no
real image/component mapping even if hashes and filenames resemble real data.
An identity collision is unresolved, not an excuse to approve real non-exposure.

## Derivative equivalence

Across different real source instances, filename similarity alone never proves
equivalence. Exact raw/decoded hashes can establish content linkage. Additional
reviewed identity bridges may be supplied using `equivalences=[...]`, each with:

```text
source_instance_id, filename_group_id, current_component_id,
reviewer, evidence_refs
```

The reviewer must be named, component must exist, and evidence references must
have matching hashes. These are attributable reviewer declarations, not
automatically authenticated clinical identity. They propagate development-only
restrictions; they cannot erase an existing exposure.

## Reviewed history boundary

`history_coverage=None` is valid unresolved input. It never means no previous
access. A declaration requires exactly:

```text
schema_version: 1
status: reviewed
reviewer: nonempty named reviewer
inventory_sha256: digest from inventory_scope
source_instance_id: current instance
component_ids: explicit reviewed coverage boundary
events_complete: true (explicit reviewer assertion)
evidence_refs: nonempty metadata references with hashes
reviewer_note: explicit scope, review limits and history completeness rationale
```

A stale inventory hash, missing reviewer/evidence, incomplete declaration or
unresolved event cannot approve candidates. Partial reviewed coverage may mark
unexposed components as candidates only within that boundary; uncovered components
remain unknown and the overall history review is blocked. Evidence quality and
history completeness require human review; hashing a statement does not prove
its truth. Coverage statements should enumerate the reviewed history scope and
known limitations, not infer completeness from an empty directory.

## Outputs and readiness limits

M0.1 v1 envelope types added by this package:

| Output under `governance/` | Type and meaning |
| --- | --- |
| `exposure_register.json` | `exposure_register/items`: resolved/unresolved events with actual evidence, confidence, image/group/component propagation and issues |
| `component_eligibility.json` | `component_eligibility/items`: development-only, unknown, or candidate untouched within a reviewed boundary |
| `exposure_summary.json` | `exposure_summary/data`: inventory scope, declarations, equivalences, blockers, counts and restrictions |

Eligibility precedence is known `development_only`, then
`candidate_untouched_with_reviewed_coverage` only with valid coverage and no
unresolved events/data blockers, otherwise `exposure_unknown`. Open items carry
owner `research_owner`, M1 review deadline and a resolution condition. Candidates
are not approved folds, verified patients or untouched confirmatory samples.

`history_review_ready` refers only to recorded history review consistency.
`confirmatory_design_ready` always remains false at this package boundary;
lineage/protocol, eligible fold support and final Gate G0 remain downstream.
Engineering package completion does not adjudicate the real study's past access.

## Tests

```powershell
python -m unittest discover -s tests -p test_m0_exposure.py -v
python -m unittest discover -s tests -v
```

See [package checklist/evidence](M0_4_EXPOSURE_GOVERNANCE.md).
