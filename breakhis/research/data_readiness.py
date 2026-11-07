"""M0.3 read-only image audit; no splits, fitting, clinical identity inference or G0.

Reuses canonical filename and RGB hash semantics without changing legacy manifests.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
import time

from . import records as r

SOURCE = 'cancer_binary'
LINEAGE_FIELDS = ('source_release', 'acquisition_date', 'archive_sha256', 'original_url',
                  'citation', 'usage_terms', 'terms_review_date', 'derivative_parent',
                  'transformation_parameters', 'transformation_code_history',
                  'backup_location', 'linkage_confidence', 'clinical_identity')


def _discover(root):
    from breakhis.manifest import IMAGE_EXTENSIONS
    images, ignored, issues = [], [], []
    if not root.is_dir():
        return images, ignored, ['SOURCE_MISSING']
    def onerror(error):
        issues.append('DIRECTORY_UNREADABLE: ' + str(error))
    for directory, directories, files in os.walk(root, followlinks=False, onerror=onerror):
        for name in sorted(directories[:]):
            path = Path(directory) / name
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                directories.remove(name)
                issues.append('LINK_DIRECTORY_REJECTED: ' + path.relative_to(root).as_posix())
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            (images if path.suffix.lower() in IMAGE_EXTENSIONS else ignored).append(relative)
    return sorted(images), sorted(ignored), sorted(issues)


def _inspect(root, relative):
    from PIL import Image
    from breakhis.manifest import parse_filename
    row = dict(schema_version=1, record_type='image_inventory_row',
               image_id=r.content_id('m0.image', {'source': SOURCE, 'relative_path': relative}),
               source=SOURCE, relative_path=relative, discovered_bytes=None,
               status='included', exclusion_reason_code=None, exclusion_detail=None,
               raw_sha256=None, decoded_pixel_sha256=None, width=None, height=None,
               class_label=None, subtype=None, magnification=None, filename_group_id=None,
               verified_patient_id=None, identity_evidence_ref=None, duplicate_component_id=None,
               representative_image_id=None, unavailable_reasons={}, issues=[])
    def issue(code, detail):
        row['issues'].append(dict(code=code, detail=str(detail)))
    try:
        path = r.safe_child(root, relative, must_exist=True)
        row['discovered_bytes'] = path.stat().st_size
    except (OSError, r.RecordError) as error:
        issue('UNSAFE_OR_MISSING_PATH', error)
        path = None
    raw = None
    if path is not None:
        try:
            raw = path.read_bytes()
            row['raw_sha256'] = hashlib.sha256(raw).hexdigest()
            if len(raw) != row['discovered_bytes']:
                issue('FILE_CHANGED', 'Byte count changed while reading')
        except OSError as error:
            issue('IMAGE_UNREADABLE', error)
    try:
        # The canonical regex describes PNG names; other supported encodings use
        # the identical stem grammar. Actual decoding never trusts the extension.
        metadata = parse_filename(Path(relative).stem + '.png')
        row.update(class_label=metadata['label'], subtype=metadata['subtype'],
                   magnification=metadata['magnification'], filename_group_id=metadata['patient_id'])
        parts = relative.split('/')
        if len(parts) != 3 or parts[0] != row['magnification'] or parts[1] != row['class_label']:
            issue('FOLDER_MISMATCH', 'Expected <magnification>/<binary label>/<filename>')
    except ValueError as error:
        issue('FILENAME_INVALID', error)
    if raw is not None:
        try:
            with Image.open(BytesIO(raw)) as image:
                image.load()
                rgb = image.convert('RGB')
                width, height = rgb.size
                row.update(width=width, height=height,
                           decoded_pixel_sha256=hashlib.sha256(f'{width}x{height}:'.encode()+rgb.tobytes()).hexdigest())
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issue('IMAGE_DECODE_FAILED', error)
    if row['issues']:
        row.update(status='excluded', exclusion_reason_code=row['issues'][0]['code'],
                   exclusion_detail=row['issues'][0]['detail'])
    for key, value in row.items():
        if value is None and key not in ('exclusion_reason_code', 'exclusion_detail'):
            row['unavailable_reasons'][key] = ('No authoritative image-specific identity mapping supplied'
                if key in ('verified_patient_id', 'identity_evidence_ref') else 'Unavailable; see image issues')
    return row


def _components(rows):
    parent = {row['image_id']: row['image_id'] for row in rows}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for field in ('raw_sha256', 'decoded_pixel_sha256', 'filename_group_id'):
        seen = {}
        for row in rows:
            value = row[field]
            if value is not None:
                previous = seen.setdefault(value, row['image_id'])
                a, b = find(previous), find(row['image_id'])
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['image_id'])].append(row)
    components = []
    for members in groups.values():
        ids = sorted(row['image_id'] for row in members)
        component_id = r.content_id('m0.component.v1', ids)
        labels = sorted({row['class_label'] for row in members if row['class_label']})
        for row in members:
            row['duplicate_component_id'] = component_id
            row['unavailable_reasons'].pop('duplicate_component_id', None)
        components.append(dict(component_id=component_id, image_ids=ids,
                               filename_group_ids=sorted({row['filename_group_id'] for row in members if row['filename_group_id']}),
                               class_labels=labels, label_conflict=len(labels)>1,
                               linkage_policy='v1: transitive raw/RGB-dimensions-pixels/full-filename-group'))
    # Deduplicate only eligible equal decoded pixels; never discard conflicting labels.
    by_pixels = defaultdict(list)
    for row in rows:
        if row['status']=='included':
            by_pixels[row['decoded_pixel_sha256']].append(row)
    conflicts = {c['component_id'] for c in components if c['label_conflict']}
    for members in by_pixels.values():
        representative = min(members, key=lambda row: row['relative_path'])
        for row in members:
            row['representative_image_id'] = representative['image_id']
            row['unavailable_reasons'].pop('representative_image_id', None)
            if row is not representative and row['duplicate_component_id'] not in conflicts:
                row.update(status='excluded', exclusion_reason_code='EXACT_PIXEL_DUPLICATE',
                           exclusion_detail='Deterministic lexicographic-path representative retained')
    return sorted(components, key=lambda component: component['component_id'])


def lineage_record(payload=None):
    """Validate operator-supplied fact declarations; never infer release/identity.

    Verified here means the supplied knowledge status, not automated adjudication
    of the cited evidence. Image-specific clinical mappings are not implemented.
    """
    if payload is None:
        payload = dict(schema_version=1, reviewer=None, facts={})
    if not isinstance(payload, dict) or set(payload) != {'schema_version', 'reviewer', 'facts'} or type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise r.RecordError('Lineage input requires v1 schema_version, reviewer and facts')
    if payload['reviewer'] is not None and (not isinstance(payload['reviewer'], str) or not payload['reviewer'].strip()):
        raise r.RecordError('Lineage reviewer must be nonempty text or null')
    if not isinstance(payload['facts'], dict) or set(payload['facts'])-set(LINEAGE_FIELDS):
        raise r.RecordError('Unknown lineage fact')
    facts, questions = {}, []
    for field in LINEAGE_FIELDS:
        fact = payload['facts'].get(field, dict(value=None, status='unknown',
            reason='No reviewed evidence supplied', evidence_refs=[]))
        r.validate_fact(fact)
        if fact['status'] in ('verified', 'disputed') and not payload['reviewer']:
            raise r.RecordError('Verified/disputed lineage declarations require a named reviewer')
        facts[field] = fact
        if fact['status'] != 'verified':
            questions.append(dict(question_id='lineage-'+field, topic=field, status=fact['status'],
                                  owner='research_owner', freeze_by='M1',
                                  resolution='Supply and review attributable evidence; agree permitted scope'))
    return dict(source=SOURCE, facts=facts, reviewer=payload['reviewer'],
                reviewed=bool(payload['reviewer']), open_questions=questions,
                identity_terminology='filename/slide group; clinical identities not mapped',
                claim_restrictions=['No verified-patient claims without image-specific authoritative mapping'])


def validate_legacy_manifest(manifest, rows):
    """Validate reuse only; never generate or silently repair a historical split."""
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or not isinstance(manifest.get('rows'), list):
        raise r.RecordError('Expected legacy manifest v1 with rows')
    selected = [item for item in manifest['rows'] if isinstance(item, dict) and item.get('source')==SOURCE]
    if len(selected) != len(manifest['rows']):
        raise r.RecordError('Research reuse requires an explicitly cancer_binary-only legacy manifest')
    included = {row['relative_path']: row for row in rows if row['status']=='included'}
    paths, partitions = set(), {field: defaultdict(set) for field in ('filename_group_id', 'duplicate_component_id', 'raw_sha256', 'decoded_pixel_sha256')}
    for item in selected:
        path = r.portable_relative_path(item.get('path'))
        if path in paths or path not in included:
            raise r.IntegrityError('Legacy file set differs from retained research inventory')
        paths.add(path)
        row = included[path]
        for old, new in (('sha256', 'raw_sha256'), ('pixel_sha256', 'decoded_pixel_sha256'),
                         ('patient_id', 'filename_group_id'), ('label', 'class_label'), ('magnification', 'magnification')):
            if item.get(old) != row[new]:
                raise r.IntegrityError('Legacy metadata/content mismatch: '+path)
        if item.get('split') not in ('train', 'validation', 'test'):
            raise r.RecordError('Invalid legacy partition')
        for field, values in partitions.items():
            values[row[field]].add(item['split'])
    if paths != set(included) or not paths:
        raise r.IntegrityError('Legacy manifest is incomplete or empty')
    if any(len(values)>1 for by_key in partitions.values() for values in by_key.values()):
        raise r.IntegrityError('Legacy group/component/hash boundary overlap')
    return dict(status='validated', assignment_kind='legacy_pilot_assignment',
                rows=len(selected), research_outer_folds=False)


def _write_jsonl(run, relative, rows):
    path = r.safe_child(run, relative)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.tmp') as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(r.canonical_json_bytes(row)+b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit_data(data_root, run_dir, run_id, *, repo_root=None, workers=1, lineage=None, legacy_manifest=None):
    """Audit one binary source into a fresh audit/ destination; no real audit on import.

    Missing/empty data yields explicit blocked records. Malformed configuration
    fails before output creation. Return values are data restrictions, not G0.
    """
    from breakhis.manifest import MAGNIFICATIONS
    r.validate_id(run_id)
    if type(workers) is not int or workers < 1:
        raise r.RecordError('Workers must be a positive integer')
    lexical = Path(data_root).expanduser().absolute()
    if lexical.is_symlink() or getattr(lexical, 'is_junction', lambda: False)():
        raise r.RecordError('Source root cannot be a link/junction')
    root, run = lexical.resolve(), Path(run_dir).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    r.reject_overlap(run, [root])
    if not run.is_dir() or not run.is_relative_to(repo / 'artifacts/research/m0'):
        raise r.RecordError('Expected existing research M0 run directory')
    lineage_data = lineage_record(lineage)
    if legacy_manifest is not None:
        legacy_path = Path(legacy_manifest).resolve()
        if legacy_path.is_relative_to(run):
            raise r.RecordError('Historical manifest cannot be generated in this run')
        legacy = r.read_json(legacy_path)
        legacy_hash = r.file_sha256(legacy_path)
    images, ignored, discovery_issues = _discover(root)
    started = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    tick = time.monotonic()
    audit_dir = r.safe_child(run, 'audit')
    audit_dir.mkdir()  # No overwrite/resume.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda relative: _inspect(root, relative), images))
    components = _components(rows)
    blockers = list(discovery_issues)
    if len({path.casefold() for path in images}) != len(images):
        blockers.append('SOURCE_CASE_COLLISION')
    if not rows or not any(row['status']=='included' for row in rows):
        blockers.append('SOURCE_EMPTY_OR_NO_INCLUDED_IMAGES')
    if any(c['label_conflict'] for c in components):
        blockers.append('DATA_LABEL_CONFLICT')
    # Recheck all discovered bytes, including invalid/corrupt images, and the file set.
    after_images, after_ignored, after_issues = _discover(root)
    if (images, ignored, discovery_issues) != (after_images, after_ignored, after_issues):
        blockers.append('SOURCE_FILE_SET_CHANGED')
    for row in rows:
        if row['raw_sha256'] is not None:
            try:
                if r.file_sha256(r.safe_child(root, row['relative_path'], must_exist=True)) != row['raw_sha256']:
                    blockers.append('SOURCE_CONTENT_CHANGED')
            except (OSError, r.RecordError):
                blockers.append('SOURCE_CONTENT_CHANGED')
        if any(issue['code'] in ('UNSAFE_OR_MISSING_PATH', 'FILE_CHANGED') for issue in row['issues']):
            blockers.append('SOURCE_PATH_OR_HASH_INCONSISTENT')
    legacy_result = dict(status='not_requested', assignment_kind=None, research_outer_folds=False)
    if legacy_manifest is not None:
        try:
            legacy_result = validate_legacy_manifest(legacy, rows)
            if r.file_sha256(legacy_path) != legacy_hash:
                raise r.IntegrityError('Legacy manifest changed during audit')
        except (OSError, r.RecordError) as error:
            blockers.append('LEGACY_MANIFEST_INVALID')
            legacy_result = dict(status='failed', reason=str(error), assignment_kind='legacy_pilot_assignment', research_outer_folds=False)
        legacy_result.update(path=str(legacy_path), sha256=legacy_hash)
    excluded = [row for row in rows if row['status']=='excluded']
    counts = dict(discovered=len(rows), included=len(rows)-len(excluded), excluded=len(excluded))
    distributions = {}
    for field in ('class_label', 'subtype', 'magnification', 'filename_group_id'):
        buckets = defaultdict(Counter)
        for row in rows:
            buckets[row[field] or 'unknown'][row['status']] += 1
            buckets[row[field] or 'unknown']['discovered'] += 1
        distributions[field] = {key: {status: value[status] for status in ('discovered','included','excluded')}
                                for key, value in sorted(buckets.items())}
    group_rows = defaultdict(list)
    for row in rows:
        if row['filename_group_id']:
            group_rows[row['filename_group_id']].append(row)
    observed_mags = {row['magnification'] for row in rows if row['status']=='included'}
    restrictions = ['CLINICAL_IDENTITIES_UNVERIFIED', 'NEAR_DUPLICATE_SCREENING_INCOMPLETE',
                    'EXPOSURE_AND_PROTOCOL_NOT_ASSESSED', 'FOLD_SUPPORT_UNREVIEWED']
    if lineage_data['open_questions']:
        restrictions.append('LINEAGE_UNRESOLVED_SCOPE_DECISION_REQUIRED')
    if excluded:
        restrictions.append('EXCLUSIONS_REQUIRE_REVIEW')
    missing_mags = sorted(set(MAGNIFICATIONS)-observed_mags)
    if missing_mags:
        restrictions.append('MAGNIFICATIONS_MISSING_NO_IMPUTATION')
    group_summary = dict(counts=counts, distributions=distributions,
        dimensions=dict(sorted(Counter(f"{row['width']}x{row['height']}" for row in rows if row['width'] is not None).items())),
        exclusion_reasons=dict(Counter(row['exclusion_reason_code'] for row in excluded)),
        filename_groups=len(group_rows), group_size_distribution=dict(sorted(Counter(str(len(v)) for v in group_rows.values()).items())),
        groups=[dict(filename_group_id=key, discovered=len(value), included=sum(row['status']=='included' for row in value),
                     available_magnifications=sorted({row['magnification'] for row in value if row['status']=='included'}),
                     missing_magnifications=sorted(set(MAGNIFICATIONS)-{row['magnification'] for row in value if row['status']=='included'}))
                for key, value in sorted(group_rows.items())],
        component_size_distribution=dict(Counter(str(len(c['image_ids'])) for c in components)),
        duplicate_pixel_sets=sum(n>1 for n in Counter(row['decoded_pixel_sha256'] for row in rows if row['decoded_pixel_sha256']).values()),
        missing_magnifications=missing_mags, ignored_non_images=ignored, ignored_non_image_count=len(ignored),
        historical_reconciliation=[dict(metric=key, reference=reference, observed=actual, delta=actual-reference,
                                        status='requires_review', note='Historical reference, not a forced acceptance constant')
            for key, reference, actual in [('discovered',7909,len(rows)), ('included',7784,counts['included']),
                                           ('excluded',125,len(excluded)), ('filename_groups',81,len(group_rows))]])
    near = dict(status='incomplete', algorithm='not_run', threshold=None, threshold_reason='No perceptual screening implemented in v1',
                exact_hash_checks='raw SHA-256 and dimension-prefixed RGB SHA-256 completed', candidates=[],
                unresolved_pairs=[], approved_linkages=[], automatic_similar_image_exclusions=False,
                reviewer=None, owner='research_owner', freeze_by='M1',
                restriction='Incomplete screening is not evidence of no near-duplicates; review required before confirmatory eligibility')
    result = dict(source=SOURCE, source_root=str(root), counts=counts, status='blocked' if blockers else 'restricted',
                  data_integrity_ready=not blockers, exploratory_ready=False, confirmatory_ready=False,
                  blockers=sorted(set(blockers)), restrictions=restrictions, legacy=legacy_result,
                  started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  operation_id='data-audit', workers=workers, grouping_policy='full YEAR-SLIDE_ID; transitive exact-duplicate linkage v1',
                  elapsed_seconds=time.monotonic()-tick, invocation={'function':'breakhis.research.m0.audit_data',
                      'data_root':str(root), 'run_dir':str(run), 'workers':workers},
                  lineage_input_sha256=r.semantic_sha256(lineage_data, domain='m0.lineage_input'),
                  pixel_hash_policy='v1 SHA256 ASCII(width x height :) + RGB bytes; matches canonical manifest',
                  input_snapshot_sha256=r.semantic_sha256([dict(path=row['relative_path'], sha256=row['raw_sha256']) for row in rows], domain='m0.data_inventory'))
    _write_jsonl(run, 'audit/inventory.jsonl', rows)
    _write_jsonl(run, 'audit/exclusions.jsonl', excluded)
    for relative, kind, payload in [('audit/audit.json', 'audit', result),
            ('audit/manifest.json', 'manifest', dict(source=SOURCE, assignment_kind='inventory_only', research_outer_folds=False,
                                                   image_ids=[row['image_id'] for row in rows if row['status']=='included'], legacy=legacy_result)),
            ('audit/duplicate_components.json', 'duplicate_components', components),
            ('audit/group_summary.json', 'group_summary', group_summary),
            ('audit/lineage.json', 'lineage', lineage_data), ('audit/near_duplicate_review.json', 'near_duplicate_review', near)]:
        r.write_record(run, relative, r.new_record(kind, run_id, payload))
    return result
