# M1.1 contracts and permissions API

Implemented in `breakhis/research/m1_records.py`, using the standard library only.
Importing it creates no files, imports no PyTorch and runs no experiment. This
package implements contracts/entry/guards, not folds, loaders, fits, prediction
export, selection, the M1 runner or Gate G1. Those remain M1.2-M1.6.

## Verify a parent and prepare an engineering config

The example `configs/research/m1.example.json` is an inactive engineering template.
Its numbers are synthetic fixtures, not real scientific agreements. Pin fields
are null with reasons; active entry/run creation refuses them until filled.

```python
from pathlib import Path
from breakhis.research import records as r
from breakhis.research import m1_records as m1

repo = Path.cwd().resolve()
archive = repo / 'artifacts/research/m0/CHOOSE_VERIFIED_RUN_ID'
config_file = repo / 'configs/research/m1.example.json'
config = r.read_json(config_file)
pins = m1.inspect_m0(archive)  # full read-only verification, not a Boolean check
config['m0_archive'] = str(archive)
for field in ('m0_index_sha256', 'm0_receipt_sha256',
              'protocol_sha256', 'decisions_sha256'):
    config[field] = pins[field]
resolved = m1.validate_config(config, config_file=config_file, require_pins=True)
entry = m1.open_entry(resolved)
print(entry.bindings, entry.snapshot['restrictions'])

# Optional contract scaffold only; use a fresh ID, not an earlier nonempty run:
# run, entry = m1.create_run(resolved, 'm1-unique-engineering-id')
```

`inspect_m0` returns discovered version/hash information only after full M0
verification; the dictionary itself grants no permission. `VerifiedEntry` requires
expected index/receipt hashes, verifies the complete archive/receipt/gate/handoff,
and binds source, inventory, components, exposure/eligibility, protocol and
decisions. A valid parent with scientific exit 1 can still permit engineering.
Entry snapshots are detached copies; modifying them cannot alter stored authority.
Operation boundaries reverify the archive, catching changes after construction.
Never call the guard once per image/batch; guard constructors and operation entry.

## Configuration binding

Config keys are exact: `schema_version`, `spec_id`, `spec_version`, `mode`,
`repo_root`, `data_root`, `output_root`, `m0_archive`, four pin fields shown above,
`source`, `component_ids`, `fold_policy`, `seeds`, `loader`, `device`,
`deterministic`, `prediction_policy`, `selection_policy`, `timeouts`,
`predecessor`, `unknown_reasons`. Unknown keys and M0 spec identities are rejected.

| Group | Contract |
| --- | --- |
| Version | Schema 1, `M1-SPEC-001`, spec version `1.0` |
| Mode/source | Engineering requires `source.kind=synthetic`; diagnostic/real-design require real metadata identity |
| Paths | Config-directory-relative; existing repo directory; outputs confined to `artifacts/research/m1`; no M0/data/checkpoint overlap or hidden output links |
| Data root | May be null; metadata diagnostics do not require original images or environments |
| Parent pins | Exact-byte index/receipt/protocol/decision SHA-256; absent pins need reasons and prohibit active entry |
| Component scope | Distinct SHA-256 component IDs; real IDs must exist and design candidates must be reviewed/eligible |
| Fold policy | Integer outer folds >=2; explicit inner roles; boolean calibration reuse; synthetic/proposed/agreed status |
| Seeds | Exactly split/training/initialization/loader/augmentation/pruning/search uint32 values |
| Loader | Positive batch size, workers zero at this package scope |
| Device | CPU or CUDA; deterministic flag is boolean; CUDA compute needs verified GPU capability |
| Predictions | Class order benign/malignant, development scope and explicit policy status |
| Selection | Floor/grid/tie tolerance are finite fractions or null with reasons; grids distinct/nonempty when supplied |
| Timeout/predecessor | Positive integer operation timeout; optional valid predecessor ID |

`validate_config` validates without writes or parent verification. `require_pins`
distinguishes an editable template from an active entry. `create_run` requires a
normalized config and validates it again, verifies pins, then writes only
`run.json`, `config.resolved.json`, `inputs/m0.entry.json`. State is **created**,
scientific authorization false, and current M1 source capture explicitly pending
the later runner. A mid-publication failure retains a failed run/failure record
where writes remain possible; nonempty destinations are never resumed.

Current M0 source identifies the parent, not new M1 implementation. M1.6 must
capture current M1 source/environment and create the full immutable graph/G1.

## Role/operation guard

```python
result = m1.authorize(
    entry, mode='engineering', source_kind='synthetic',
    source_instance_id='independent-m1-fixture',
    operation='construct_dataset', role='development_selection',
)

# Future adapters must check before construction/iteration/export:
# dataset = m1.guarded_call(entry, factory=build_synthetic_dataset, ...)
```

| Mode | Allowed operations at this boundary |
| --- | --- |
| Engineering | Synthetic metadata/planning, inner-role dataset/iteration, inner-train fits, development exports, calibration/selection, synthetic resident benchmarks and stored synthetic OOF validation |
| Diagnostic | Frozen metadata reads/support diagnostics only; no image iteration, fitting, active fold planning or prediction export |
| Real design | Metadata/planning only after verified real G0 design eligibility and a nonempty reviewed candidate boundary |

Outer-test dataset construction, image iteration and prediction export are denied
in every mode, including synthetic development flows. The separate stored
synthetic OOF validation operation does not permit inference. Excluded and generic
development-only labels cannot silently become active loader roles. Calibration
requires the calibration role; epoch selection requires development selection;
training uses inner train. Development export rejects inner-training predictions.

Synthetic instance/component IDs cannot collide with the bound M0 identity or
known real components. Real instances must match the verified M0 source; updated
fold choices must match the agreed parent count for real design. Diagnostic mode
cannot be upgraded by changing a directory name or copied flag. Real image
iteration/fitting remains forbidden throughout M1 even if a future parent permits
it; later milestones consume the contracts under their own scope.

`guarded_call` invokes the supplied adapter only after permission succeeds. The
checks prevent accidental pipeline misuse, not arbitrary Python from bypassing
APIs or fabricating inputs. Later packages still must validate actual assignment,
row, recipe, calibration, support and provenance semantics. A returned permission
does not implement those operations or establish clinical/scientific truth.

## Record/JSONL and identity foundation

M1 currently registers `m1.run`, `m1.config`, `m1.entry`, `m1.permissions`,
`m1.failure`, `m1.operations`, `m1.evidence_index`. V1 envelopes use the common
`data`/`items` roots. These are envelope contracts, not a full future evidence
index/G1 validator. No unprefixed M0 kind is accepted through M1 wrappers, and
M0's default registry does not accept M1 kinds.

The initial flat JSONL type `m1.input_row` has exactly: schema/version/type,
source instance, image ID, relative path, raw/decoded hashes, full filename group,
component, class/subtype/magnification, included/excluded status and unavailable
reasons (see `validate_input_row` for concrete field spelling). Included identities
require hashes/group/class/magnification. Nulls require reasons. Later assignment,
epoch-history and prediction schemas are deliberately not registered yet.

`identity_sha256(rows, policy)` covers complete sorted identities and the explicit
policy, not output roots or row order. Duplicate image IDs and case-colliding
paths fail. `write_input_rows` validates before atomic no-clobber JSONL publication;
`read_input_rows` checks strict UTF-8/JSON, finite values, type/version and identity
uniqueness. `config_identity` excludes only physical repo/data/output/archive
locators; pin/policy/seed/content changes still change the semantic digest.

Shared M0 helpers gained optional explicit registry arguments and `publish_bytes`.
Legacy calls retain their default registry, JSON suffix checks, finite encoding,
link-safe paths and no-clobber atomic behavior. Full source capture and other
M0-only APIs are not treated as M1 APIs. Containment retains the static trusted-root
model; adversarial concurrent filesystem replacement is not promised.

## Tests

```powershell
python -m unittest discover -s tests -p test_m1_records.py -v
python -m unittest discover -s tests -v
```

See [M1.1 acceptance evidence](M1_1_CONTRACTS_PERMISSIONS.md#progress-evidence).
