# M0.5 protocol, decisions and literature API

Implementation: `breakhis/research/protocol.py`; lazy binding adapter:
`breakhis.research.m0.write_protocol_records`. No training, threshold fitting,
literature downloads or inference occurs on import or validation.

## Templates

| File under `configs/research/` | Role |
| --- | --- |
| `protocol.initial.json` | Draft v1, 17 mandatory topic policies, all proposed, unresolved numbers null, initially unbound |
| `decisions.initial.json` | One stable revision-1 decision per topic, with owners/evidence/deadlines; preserve all later revisions |
| `literature.initial.json` | Seven primary-source review targets with actual access levels and explicit central gaps |
| `lineage.initial.json` | Valid unresolved lineage input; no invented original transform or backup facts |
| `exposure.initial.json` | Unresolved history scaffold; null instance/root and no coverage declaration do not mean untouched data |
| `m0.example.json` | M0.1-compatible pre-run configuration, relative paths based on its own directory; CPU checks default |

The [M0.6 runner](M0_6_RUNNER_API.md) now consumes the example config. Its optional
`decisions_path` selects the register, defaulting to `decisions.initial.json`
beside `protocol_path` for compatibility. Decisions are supplied separately to
this package's library API.
Do not treat exposure scaffold nulls as a valid reviewed resolver input.

The [human protocol](../../research/RESEARCH_PROTOCOL.md) explains policy values,
comparisons, unresolved decisions and scientific restrictions.

## Validate and bind

```python
from pathlib import Path
from breakhis.research.records import read_json
from breakhis.research.protocol import (
    validate_protocol, validate_decisions, validate_literature,
    write_protocol_records,
)

inputs = Path('configs/research')
protocol = read_json(inputs / 'protocol.initial.json')
decisions = read_json(inputs / 'decisions.initial.json')
literature = read_json(inputs / 'literature.initial.json')
print(validate_protocol(protocol))
print(validate_decisions(decisions, protocol))
print(validate_literature(literature))

# Requires a fresh attributable baseline + audit + exposure records in this run:
# result = write_protocol_records(run, run_id, protocol=protocol,
#                                 decisions=decisions, literature=literature)
```

Validation returns restrictions and never promotes choices. Malformed structure,
missing topics/subpolicies/owners/deadlines/evidence and unsupported schema/lifecycle
raise `RecordError`; inconsistent revisions or stale bindings fail. Numerical
margins are fractions in [0,1], budget/thread/fold counts have integer limits and
seed values, when provided, must match the independent named uint32 namespaces.
`agreed` values cannot conceal unresolved nulls. These checks do not validate
clinical appropriateness or authorize scientific use of a supplied number.

## Input schemas

Protocol roots contain exactly `schema_version`, `protocol_id`, `revision`,
`lifecycle`, `reviewer`, `bindings`, `topics`. Each topic contains exactly `topic`,
`value`, `status`, `rationale`, `evidence_refs`, `owner`, `freeze_by`. `TOPICS` in
the implementation defines mandatory subpolicy keys for all 17 master topics.
Statuses are `agreed|proposed|deferred|blocked`. Role owners are allowed but
unassigned `research_owner`/`reviewer` roles are surfaced as restrictions.

Allowed lifecycles are `draft`, `pilot_ready`, `initial_design_reviewed`.
`confirmatory_frozen` is forbidden. Non-draft declarations need a named reviewer,
complete bindings and agreed critical topics; initial design review requires all
topics agreed. Binding also refuses a non-draft lifecycle with unresolved history.
No lifecycle is an overall G0 or scientific authorization result.

Decision rows contain the same topic fields plus `decision_id`, `revision`,
`supersedes`, `affected_claims`. Revision 1 has null supersedes; later revisions
are consecutive and reference `decision_id@N`. A topic has one stable decision ID.
Its latest full policy must equal the corresponding protocol topic.

Literature inputs are a list with the columns in master section 5.7, concretely
declared by `LITERATURE_FIELDS`. Missing fields require null plus
`unknown_reasons[field]`; no guessed authors/year/venue/license. Additional fields
include central/category flags, version actually read, review status, owner and
deadline. Required central categories are hardware-aware, class-aware and medical
sample vulnerability. Abstract/code/unavailable access can never carry
`review_status=reviewed_methods`. Full methods review needs an attributable
citation, reviewer/date and method notes; remaining questions/unknowns still
restrict central-review completion. `novelty_claim` remains explicitly unverified.
No Boolean output establishes scientific novelty or faithful reproduction.

## Binding and outputs

Templates use null bindings. The writer fills a detached copy with:

- Semantic source snapshot identity from the completed baseline Git record.
- Exact-byte SHA-256 of `audit/inventory.jsonl` and `governance/exposure_register.json`.
- Domain-separated canonical hashes of the decision and literature input lists.

Supplied nonnull bindings must already match. Source/audit/exposure run IDs and
exposure inventory scope must agree. The protocol input is separately hashed in
the review record. This avoids circular hashes between protocol and its inputs.

The writer creates new M0.1 v1 envelopes: `protocol.initial/data`, `decisions/items`,
`literature_matrix/items`, and additive `protocol_review/data` under `governance/`.
Existing destinations are refused. Outputs preserve decisions, unknown values,
access levels and draft lifecycle. Full evidence graph/semantic gate verification
is provided by M0.6; this binding API alone cannot assert scientific validity.

## Developer checks

```powershell
python -m unittest discover -s tests -p test_m0_protocol.py -v
python -m unittest discover -s tests -v
```

See [package acceptance evidence](M0_5_PROTOCOL_LITERATURE.md).
