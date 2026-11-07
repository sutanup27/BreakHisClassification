"""M0.1 record envelopes, content identity and confined evidence IO.

Only envelope/config/integrity validation lives here. Scientific record semantics
and G0 readiness belong to later M0 packages; successful validation grants no
scientific eligibility. This module uses the standard library only.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

SCHEMA_VERSION = 1
SPEC_ID = 'M0-SPEC-001'
SPEC_VERSION = '1.0'
RECORD_SHAPES = {
    'run': 'data', 'config.resolved': 'data', 'git': 'data',
    'source_inventory': 'items', 'environment': 'data', 'checks': 'items',
    'manifest': 'data', 'audit': 'data', 'duplicate_components': 'items',
    'group_summary': 'data', 'lineage': 'data', 'near_duplicate_review': 'data',
    'exposure_register': 'items', 'protocol.initial': 'data', 'decisions': 'items',
    'literature_matrix': 'items', 'evidence_index': 'items', 'gate_g0': 'data',
    'completion_receipt': 'data',
    'component_eligibility': 'items', 'exposure_summary': 'data',
    'protocol_review': 'data',
}
_ENVELOPE = {'schema_version', 'record_type', 'run_id', 'created_at_utc'}
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
_TOKEN = re.compile(r'[A-Za-z][A-Za-z0-9_.-]{0,63}\Z')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}


class RecordError(ValueError):
    """Malformed configuration, record or portable path."""


class IntegrityError(RecordError):
    """Missing, changed or incomplete evidence; never repaired automatically."""


def validate_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise RecordError('ID must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit')
    return value


def _token(value):
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise RecordError('Expected a nonempty record/operation token')
    return value


def _json_value(value):
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        try:
            value.encode('utf-8')
        except UnicodeEncodeError as error:
            raise RecordError('JSON strings must be valid UTF-8') from error
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RecordError('JSON numbers must be finite')
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordError('JSON object keys must be strings')
            _json_value(key)
            _json_value(item)
        return
    raise RecordError(f'Unsupported JSON value type: {type(value).__name__}')


def canonical_json_bytes(value):
    """UTF-8 JSON, sorted object keys, no whitespace; list order is meaningful."""
    try:
        _json_value(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (ValueError, RecursionError) as error:
        if isinstance(error, RecordError):
            raise
        raise RecordError(f'Invalid JSON value: {error}') from error


def semantic_sha256(value, *, domain):
    """Domain-separated canonical content hash; not an RFC 8785 implementation."""
    _token(domain)
    return hashlib.sha256(canonical_json_bytes({'hash_version': 1, 'domain': domain, 'value': value})).hexdigest()


def content_id(namespace, identity):
    """Stable 64-character identifier from identity content, never row position."""
    return semantic_sha256(identity, domain=namespace)


def file_sha256(path):
    """Exact-byte identity; callers use safe_child when containment is required."""
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def portable_relative_path(value):
    """Require an unambiguous relative POSIX path portable to Windows."""
    if not isinstance(value, str) or not value or any(char in value for char in '\\:\0'):
        raise RecordError('Expected a relative POSIX path without backslashes, drives or streams')
    if PurePosixPath(value).is_absolute():
        raise RecordError('Absolute artifact paths are forbidden')
    for component in value.split('/'):
        if component in ('', '.', '..') or component.endswith((' ', '.')) or any(char in component for char in '<>"|?*'):
            raise RecordError('Ambiguous or traversing artifact path')
        if any(ord(char) < 32 for char in component) or component.split('.')[0].upper() in _RESERVED:
            raise RecordError('Reserved artifact path component')
    _json_value(value)
    return value


def _reject_link(path):
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise RecordError(f'Symlinks/junctions are not permitted below the trusted root: {path}')


def safe_child(root, relative, *, must_exist=False):
    """Resolve below a trusted root, refusing links even if they point inside it.

    The root must itself be trusted/config-validated. This is static containment,
    not a claim of protection from a hostile concurrent filesystem writer.
    """
    portable_relative_path(relative)
    root = Path(root).expanduser().absolute()
    _reject_link(root)
    base = root.resolve()
    current = base
    for component in relative.split('/'):
        current = current / component
        _reject_link(current)
    resolved = current.resolve(strict=must_exist)
    if not resolved.is_relative_to(base):
        raise RecordError('Artifact escapes trusted root')
    return resolved


def reject_overlap(output, protected_roots):
    output = Path(output).expanduser().resolve()
    for protected in protected_roots:
        protected = Path(protected).expanduser().resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise RecordError(f'Output overlaps protected input: {protected}')
    return output


def create_run_directory(output_root, run_id, *, protected_roots=()):
    """Create/refuse a run destination without deleting or emptying anything."""
    validate_id(run_id)
    root = reject_overlap(output_root, protected_roots)
    # Do not let resolving the configured root conceal a junction at that root.
    _reject_link(Path(output_root).expanduser().absolute())
    target = safe_child(root, run_id)
    reject_overlap(target, protected_roots)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise RecordError('Run destination must be new or empty')
    else:
        root.mkdir(parents=True, exist_ok=True)
        target.mkdir()  # No exist_ok: concurrent creation must not grant ownership.
    return target


def capture_relative_path(path, *, repo_root, output_root):
    """Guard individual snapshot inputs; inclusion/exclusion policy is M0.2."""
    base = Path(repo_root).resolve()
    lexical = Path(path).absolute()
    try:
        relative = lexical.relative_to(base).as_posix()
    except ValueError as error:
        raise RecordError('Snapshot input is outside repository') from error
    candidate = safe_child(base, relative, must_exist=True)
    if not candidate.is_file() or candidate.is_relative_to(Path(output_root).resolve()):
        raise RecordError('Snapshot cannot capture directories or its own output')
    return relative


def snapshot_sha256(entries, policy):
    """Hash sorted portable path/hash pairs plus capture policy, ignoring row metadata."""
    if not isinstance(policy, dict):
        raise RecordError('Capture policy must be an object')
    pairs, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RecordError('Snapshot entries must be objects')
        path = portable_relative_path(entry.get('path'))
        digest = entry.get('sha256')
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RecordError('Invalid SHA-256')
        if path.casefold() in seen:
            raise RecordError('Duplicate or case-colliding snapshot path')
        seen.add(path.casefold())
        pairs.append({'path': path, 'sha256': digest})
    return semantic_sha256({'files': sorted(pairs, key=lambda item: item['path']), 'policy': policy}, domain='m0.source_snapshot')


def validate_fact(fact):
    """An explicit knowledge-status value; this does not verify its scientific truth."""
    if not isinstance(fact, dict) or set(fact) != {'value', 'status', 'reason', 'evidence_refs'}:
        raise RecordError('Fact requires value, status, reason and evidence_refs')
    canonical_json_bytes(fact)
    status = fact['status']
    if status not in ('verified', 'self_reported', 'unknown', 'disputed'):
        raise RecordError('Invalid fact status')
    refs = fact['evidence_refs']
    if not isinstance(refs, list) or not all(isinstance(ref, str) and ref.strip() for ref in refs):
        raise RecordError('Evidence references must be nonempty strings')
    if status == 'unknown' and fact['value'] is not None:
        raise RecordError('Unknown facts must use null, not an invented value')
    if status in ('unknown', 'disputed') and not (isinstance(fact['reason'], str) and fact['reason'].strip()):
        raise RecordError('Unknown/disputed facts require a reason')
    if status in ('verified', 'self_reported') and (fact['value'] is None or not refs):
        raise RecordError('Known/reported facts require a value and evidence reference')
    if fact['reason'] is not None and not isinstance(fact['reason'], str):
        raise RecordError('Fact reason must be text or null')
    return fact


def _utc_timestamp(value):
    if not isinstance(value, str):
        raise RecordError('created_at_utc must be an ISO-8601 UTC string')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise RecordError('Invalid UTC timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0 or 'T' not in value:
        raise RecordError('Timestamp must be UTC with explicit timezone')


def validate_record(record, *, expected_type=None):
    """Validate the exact v1 envelope, not later-package scientific payload fields."""
    if not isinstance(record, dict):
        raise RecordError('Record must be an object')
    if type(record.get('schema_version')) is not int or record['schema_version'] != SCHEMA_VERSION:
        raise RecordError('Unsupported record schema version')
    kind = record.get('record_type')
    if not isinstance(kind, str) or kind not in RECORD_SHAPES or (expected_type is not None and kind != expected_type):
        raise RecordError('Unknown or unexpected record type')
    field = RECORD_SHAPES[kind]
    if set(record) != _ENVELOPE | {field}:
        raise RecordError(f'Record envelope requires exactly {sorted(_ENVELOPE | {field})}')
    validate_id(record['run_id'])
    _utc_timestamp(record['created_at_utc'])
    if field == 'data' and not isinstance(record[field], dict):
        raise RecordError('Object record payload must be a data object')
    if field == 'items' and (not isinstance(record[field], list) or not all(isinstance(item, dict) for item in record[field])):
        raise RecordError('Collection record payload must be a list of objects')
    canonical_json_bytes(record)
    return record


def new_record(record_type, run_id, payload, *, created_at_utc=None):
    if not isinstance(record_type, str) or record_type not in RECORD_SHAPES:
        raise RecordError('Unknown record type')
    result = dict(schema_version=SCHEMA_VERSION, record_type=record_type, run_id=run_id,
                  created_at_utc=created_at_utc if created_at_utc is not None else datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    result[RECORD_SHAPES[record_type]] = payload
    validate_record(result)
    # Detach caller-owned mutable structures using the same strict JSON contract.
    return json.loads(canonical_json_bytes(result))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecordError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    """Strict UTF-8 JSON; a BOM is accepted, duplicates and nonfinite values are not."""
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object)
        canonical_json_bytes(value)
        return value
    except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
        raise RecordError(f'Invalid JSON file: {path}') from error


def read_record(run_dir, relative, *, expected_type=None):
    path = safe_child(run_dir, relative, must_exist=True)
    return validate_record(read_json(path), expected_type=expected_type)


def write_record(run_dir, relative, record, *, overwrite=False):
    """Atomic publication in a trusted run; no-clobber unless explicitly updating.

    Temp and destination share a filesystem. No-clobber uses an atomic hard-link
    publication; unsupported filesystems fail rather than falling back to a race.
    Callers must create/config-validate the run first. No arbitrary dataset root
    should be passed as a run directory.
    """
    validate_record(record)
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    root = Path(run_dir)
    if not root.is_dir():
        raise RecordError('Create a validated run directory before writing records')
    path = safe_child(root, relative)
    if path.suffix.lower() != '.json':
        raise RecordError('Record destination must end in .json')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    # Validate each created parent, never traversing a link below the trusted root.
    parent_parts = PurePosixPath(relative).parts[:-1]
    for index in range(1, len(parent_parts) + 1):
        parent = safe_child(root, '/'.join(parent_parts[:index]))
        parent.mkdir(exist_ok=True)
        _reject_link(parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.' + path.name + '-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        safe_child(root, relative)  # Recheck static path state before publication.
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_ARTIFACT_FIELDS = {'path', 'record_type', 'schema_version', 'sha256', 'bytes', 'operation_id', 'status'}


def validate_artifact_entry(entry):
    if not isinstance(entry, dict) or set(entry) != _ARTIFACT_FIELDS:
        raise RecordError('Invalid artifact-entry fields')
    portable_relative_path(entry['path'])
    _token(entry['record_type'])
    _token(entry['operation_id'])
    version = entry['schema_version']
    if version is not None and (type(version) is not int or version != SCHEMA_VERSION or entry['record_type'] not in RECORD_SHAPES):
        raise RecordError('Unsupported artifact record schema/type')
    if not isinstance(entry['sha256'], str) or not _SHA256.fullmatch(entry['sha256']):
        raise RecordError('Invalid artifact SHA-256')
    if type(entry['bytes']) is not int or entry['bytes'] < 0:
        raise RecordError('Artifact bytes must be a nonnegative integer')
    if entry['status'] not in ('complete', 'partial', 'failed'):
        raise RecordError('Invalid artifact status')
    return entry


def artifact_entry(run_dir, relative, *, record_type, operation_id, schema_version=None, status='complete'):
    path = safe_child(run_dir, relative, must_exist=True)
    if not path.is_file():
        raise RecordError('Evidence artifact must be a regular file')
    result = dict(path=relative, record_type=record_type, schema_version=schema_version,
                  sha256=file_sha256(path), bytes=path.stat().st_size,
                  operation_id=operation_id, status=status)
    return validate_artifact_entry(result)


def verify_artifact(run_dir, entry, *, run_id=None, require_complete=True):
    """Read-only byte and optional envelope verification. Never regenerates evidence."""
    validate_artifact_entry(entry)
    if require_complete and entry['status'] != 'complete':
        raise IntegrityError('Artifact is not complete')
    try:
        path = safe_child(run_dir, entry['path'], must_exist=True)
        if not path.is_file() or path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
            raise IntegrityError(f'Artifact content differs: {entry["path"]}')
    except OSError as error:
        raise IntegrityError(f'Artifact missing or unreadable: {entry["path"]}') from error
    if entry['schema_version'] is not None:
        record = read_record(run_dir, entry['path'], expected_type=entry['record_type'])
        if run_id is not None and record['run_id'] != run_id:
            raise IntegrityError('Evidence belongs to a different run')
    return path


def verify_evidence_index(run_dir, index, *, index_path='acceptance/evidence_index.json', require_complete=True):
    """M0.6 interface foundation: verify entries, not G0 or an evidence graph."""
    validate_record(index, expected_type='evidence_index')
    portable_relative_path(index_path)
    seen = set()
    for entry in index['items']:
        validate_artifact_entry(entry)
        key = entry['path'].casefold()
        if key == index_path.casefold() or key in seen:
            raise IntegrityError('Duplicate, case-colliding or self-referential evidence entry')
        seen.add(key)
        verify_artifact(run_dir, entry, run_id=index['run_id'], require_complete=require_complete)
    return {'verified_artifacts': len(seen), 'run_id': index['run_id']}


def resolve_config_path(value, config_file, *, kind=None):
    if not isinstance(value, str) or not value.strip() or '\0' in value:
        raise RecordError('Config path must be a nonempty string')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_file).resolve().parent / path
    path = path.resolve()
    if kind == 'file' and not path.is_file():
        raise RecordError(f'Config input file missing: {path}')
    if kind == 'directory' and not path.is_dir():
        raise RecordError(f'Config input directory missing: {path}')
    if kind not in (None, 'file', 'directory'):
        raise RecordError('Unknown config path kind')
    return path


_CONFIG_FIELDS = {'schema_version', 'spec_id', 'spec_version', 'repo_root', 'data_root', 'output_root',
                  'sources', 'audit_workers', 'audit_seed', 'check_devices', 'baseline_checks',
                  'lineage_path', 'exposure_path', 'protocol_path', 'literature_path', 'mode', 'timeouts'}


def _choices(value, choices, field):
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise RecordError(f'{field} must be a nonempty list of strings')
    if len(set(value)) != len(value) or set(value) - set(choices):
        raise RecordError(f'Invalid or duplicate {field}')


def validate_m0_config(config, *, config_file):
    """Validate pre-run structure/paths without creating outputs or running checks.

    Governance paths are resolved but need not exist yet; the later runner must
    ingest and validate those records before granting requested-mode eligibility.
    """
    if not isinstance(config, dict) or set(config) != _CONFIG_FIELDS:
        raise RecordError('M0 config has missing or unknown fields')
    canonical_json_bytes(config)
    if type(config['schema_version']) is not int or config['schema_version'] != SCHEMA_VERSION:
        raise RecordError('Unsupported config schema version')
    if config['spec_id'] != SPEC_ID or config['spec_version'] != SPEC_VERSION:
        raise RecordError('Unsupported M0 spec identity/version')
    if config['mode'] not in ('pilot', 'confirmatory_design'):
        raise RecordError('Invalid requested mode')
    if config['sources'] != ['cancer_binary']:
        raise RecordError('M0 v1 selects exactly cancer_binary')
    if type(config['audit_workers']) is not int or config['audit_workers'] < 1:
        raise RecordError('Audit workers must be a positive integer')
    if type(config['audit_seed']) is not int or not 0 <= config['audit_seed'] < 2**32:
        raise RecordError('Audit seed must be a uint32 integer')
    _choices(config['check_devices'], ('cpu', 'cuda'), 'check_devices')
    if 'cpu' not in config['check_devices']:
        raise RecordError('CPU baseline check is required')
    required_checks = {'imports', 'cli_help', 'environment', 'regressions'}
    _choices(config['baseline_checks'], required_checks, 'baseline_checks')
    if set(config['baseline_checks']) != required_checks:
        raise RecordError('All baseline checks are required')
    timeouts = config['timeouts']
    if not isinstance(timeouts, dict) or set(timeouts) != {'environment_seconds', 'regression_seconds', 'audit_seconds'}:
        raise RecordError('Invalid timeout fields')
    if any(type(value) is not int or value < 1 for value in timeouts.values()):
        raise RecordError('Timeouts must be positive integer seconds')
    resolved = json.loads(canonical_json_bytes(config))
    for field in ('repo_root', 'data_root', 'output_root', 'lineage_path', 'exposure_path', 'protocol_path', 'literature_path'):
        kind = 'directory' if field in ('repo_root', 'data_root') else None
        resolved[field] = str(resolve_config_path(config[field], config_file, kind=kind))
    repo = Path(resolved['repo_root'])
    output = Path(resolved['output_root'])
    # Keep the lexical anchor: resolving it too would conceal an escaping junction.
    anchor = repo / 'artifacts/research/m0'
    if not output.is_relative_to(anchor):
        raise RecordError('M0 output must stay under repository artifacts/research/m0')
    reject_overlap(output, [resolved['data_root']])
    return resolved
