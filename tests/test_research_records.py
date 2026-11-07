"""M0.1 foundation coverage for M0-T05, M0-T21, M0-T22; standard library only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from breakhis.research import records as r

STAMP = '2026-11-07T00:00:00Z'


def record(kind='run', payload=None, run_id='m0-test'):
    if payload is None:
        payload = [] if r.RECORD_SHAPES[kind] == 'items' else {}
    return r.new_record(kind, run_id, payload, created_at_utc=STAMP)


class RecordTests(unittest.TestCase):
    def test_import_has_no_filesystem_or_training_dependency(self):
        code = ('import sys; from unittest.mock import patch; '
                'from pathlib import Path; '
                '\n'
                'with patch.object(Path,"mkdir",side_effect=AssertionError("mkdir on import")), '
                'patch.object(Path,"open",side_effect=AssertionError("open on import")):\n'
                ' import breakhis.research.records\n'
                'assert "torch" not in sys.modules\n')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_declared_envelopes_and_detached_payloads(self):
        for kind, shape in r.RECORD_SHAPES.items():
            payload = {'note': 'explicit payload'} if shape == 'data' else [{'note': 'item'}]
            result = record(kind, payload)
            self.assertIs(r.validate_record(result, expected_type=kind), result)
            payload.clear()
            self.assertTrue(result[shape])

    def test_unknown_versions_types_extra_fields_and_timestamps(self):
        for field, value in [('schema_version', 2), ('schema_version', True), ('schema_version', 1.0),
                             ('record_type', 'unregistered'), ('run_id', '../escape'),
                             ('created_at_utc', '2026-11-07'), ('created_at_utc', '2026-11-07T01:00:00+01:00')]:
            invalid = record()
            invalid[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(r.RecordError):
                r.validate_record(invalid)
        extra = record(); extra['readiness'] = 'ready'
        with self.assertRaises(r.RecordError): r.validate_record(extra)
        with self.assertRaises(r.RecordError): r.validate_record(record(), expected_type='audit')
        with self.assertRaises(r.RecordError): r.new_record([], 'm0-test', {})
        with self.assertRaises(r.RecordError): r.new_record('run', 'm0-test', {}, created_at_utc='')
        with self.assertRaises(r.RecordError): record('checks', [1])
        with self.assertRaises(r.RecordError): record('run', [])

    def test_nonfinite_and_non_json_payloads_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), {1: 'key'}, {'set'}, Path('a'), (1, 2), '\ud800'):
            with self.subTest(value=repr(value)), self.assertRaises(r.RecordError):
                record(payload={'value': value})
        cycle = []; cycle.append(cycle)
        with self.assertRaises(r.RecordError): r.canonical_json_bytes(cycle)

    def test_explicit_unknowns_and_evidence_status(self):
        unknown = dict(value=None, status='unknown', reason='Original transform not supplied', evidence_refs=[])
        r.validate_fact(unknown)
        for changed in ({'value': 'guessed resize'}, {'reason': None}, {'status': 'assumed'}):
            invalid = {**unknown, **changed}
            with self.assertRaises(r.RecordError): r.validate_fact(invalid)
        for status in ('verified', 'self_reported'):
            fact = dict(value='source note', status=status, reason=None, evidence_refs=['governance/source-note'])
            r.validate_fact(fact)
            fact['evidence_refs'] = []
            with self.assertRaises(r.RecordError): r.validate_fact(fact)

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_accepts_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.json'
            for content in ('{"schema_version":1,"schema_version":2}', '{"value":NaN}', '{"value":1e999}', '{bad}'):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(r.RecordError): r.read_json(path)
            path.write_text('{"note":"histology"}', encoding='utf-8-sig')
            self.assertEqual(r.read_json(path), {'note': 'histology'})


class HashTests(unittest.TestCase):
    def test_canonical_keys_domain_and_content_ids(self):
        self.assertEqual(r.semantic_sha256({'b': 2, 'a': 1}, domain='m0.example'),
                         r.semantic_sha256({'a': 1, 'b': 2}, domain='m0.example'))
        self.assertNotEqual(r.semantic_sha256({'a': 1}, domain='m0.first'),
                            r.semantic_sha256({'a': 1}, domain='m0.second'))
        self.assertEqual(r.content_id('image', {'path': '40X/a.png'}),
                         r.content_id('image', {'path': '40X/a.png'}))
        self.assertEqual(len(r.content_id('image', {})), 64)
        self.assertEqual(r.canonical_json_bytes({'z': 'café', 'a': [2, 1]}), '{"a":[2,1],"z":"café"}'.encode())

    def test_exact_file_bytes_and_semantic_json_are_different_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder)/'a.json', Path(folder)/'b.json'
            one.write_bytes(b'{"a":1,"b":2}')
            two.write_bytes(b'{\n "b": 2, "a": 1\n}')
            self.assertEqual(r.file_sha256(one), hashlib.sha256(one.read_bytes()).hexdigest())
            self.assertNotEqual(r.file_sha256(one), r.file_sha256(two))
            self.assertEqual(r.semantic_sha256(r.read_json(one), domain='json'), r.semantic_sha256(r.read_json(two), domain='json'))

    def test_snapshot_order_relocation_metadata_and_changes(self):
        entries = [{'path': 'b.py', 'sha256': 'b'*64, 'root': 'D:/old', 'timestamp': 'old'},
                   {'path': 'a.py', 'sha256': 'a'*64}]
        policy = {'version': 1, 'include': ['*.py']}
        expected = r.snapshot_sha256(entries, policy)
        relocated = list(reversed(copy.deepcopy(entries)))
        relocated[1].update(root='C:/new', timestamp='new')
        self.assertEqual(expected, r.snapshot_sha256(relocated, policy))
        relocated[0]['sha256'] = 'c'*64
        self.assertNotEqual(expected, r.snapshot_sha256(relocated, policy))
        self.assertNotEqual(expected, r.snapshot_sha256(entries, {'version': 2}))
        with self.assertRaises(r.RecordError): r.snapshot_sha256([entries[0], entries[0]], policy)
        with self.assertRaises(r.RecordError): r.snapshot_sha256([{'path':'B.py','sha256':'a'*64}, entries[0]], policy)


class PathTests(unittest.TestCase):
    def test_portable_paths_reject_traversal_drives_streams_and_aliases(self):
        for value in ('../escape', '/absolute', 'C:/absolute', 'C:relative', r'folder\escape',
                      '//server/file', 'name:stream', './file', 'a/../b', 'a//b', 'a/', 'NUL.json',
                      'COM1/file', 'trailing./x', 'trailing /x', 'a\0b', ''):
            with self.subTest(value=value), self.assertRaises(r.RecordError): r.portable_relative_path(value)
        self.assertEqual(r.portable_relative_path('baseline/source_inventory.json'), 'baseline/source_inventory.json')

    def test_run_destinations_protected_roots_and_no_self_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'Dataset'; data.mkdir()
            with self.assertRaises(r.RecordError): r.create_run_directory(data/'artifacts', 'run', protected_roots=[data])
            with self.assertRaises(r.RecordError): r.create_run_directory(root, 'run', protected_roots=[data])
            run = r.create_run_directory(root/'output', 'run', protected_roots=[data])
            self.assertEqual(run, r.create_run_directory(root/'output', 'run'))
            (run/'existing').write_text('original')
            with self.assertRaises(r.RecordError): r.create_run_directory(root/'output', 'run')
            self.assertEqual((run/'existing').read_text(), 'original')
            source = root/'code.py'; source.write_text('source')
            self.assertEqual(r.capture_relative_path(source, repo_root=root, output_root=run), 'code.py')
            with self.assertRaises(r.RecordError): r.capture_relative_path(run/'existing', repo_root=root, output_root=run)
            with self.assertRaises(r.RecordError): r.capture_relative_path(source, repo_root=data, output_root=run)

    def test_junction_detection_without_windows_symlink_privilege(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, 'is_junction', create=True, autospec=True,
                              side_effect=lambda path: path.name == 'alias'):
                with self.assertRaises(r.RecordError): r.safe_child(folder, 'alias/file.json')
            with patch.object(Path, 'is_symlink', autospec=True,
                              side_effect=lambda path: path.name == 'broken'):
                with self.assertRaises(r.RecordError): r.safe_child(folder, 'broken/file.json')

    def test_existing_symlink_and_broken_symlink_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); outside = root/'outside'; outside.mkdir()
            trusted = root/'trusted'; trusted.mkdir()
            alias = trusted/'alias'
            junction = False
            try:
                alias.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                if os.name != 'nt':
                    raise
                # Junction creation needs no symlink privilege. Both endpoints are
                # verified children of this test-owned temporary directory.
                self.assertTrue(alias.absolute().is_relative_to(root.resolve()))
                self.assertTrue(outside.resolve().is_relative_to(root.resolve()))
                result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(outside)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f'{error}\n{result.stderr}')
                junction = True
            try:
                with self.assertRaises(r.RecordError): r.safe_child(trusted, 'alias/evidence.json')
                if not junction:
                    (trusted/'broken').symlink_to(root/'missing')
                    with self.assertRaises(r.RecordError): r.safe_child(trusted, 'broken')
            finally:
                if junction:
                    os.rmdir(alias)  # Remove just the junction, never its target.
            self.assertTrue(outside.is_dir())


class AtomicEvidenceTests(unittest.TestCase):
    def test_utf8_roundtrip_no_clobber_and_explicit_state_update(self):
        with tempfile.TemporaryDirectory() as folder:
            first = record(payload={'note':'café', 'status':'created'})
            path = r.write_record(folder, 'baseline/run.json', first)
            self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(r.read_record(folder, 'baseline/run.json'), first)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError): r.write_record(folder, 'baseline/run.json', record())
            self.assertEqual(path.read_bytes(), original)
            updated = record(payload={'status':'running'})
            r.write_record(folder, 'baseline/run.json', updated, overwrite=True)
            self.assertEqual(r.read_record(folder, 'baseline/run.json'), updated)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_interrupted_replace_preserves_existing_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            path = r.write_record(folder, 'run.json', record(payload={'old':True}))
            before = path.read_bytes()
            with patch.object(r.os, 'replace', side_effect=OSError('simulated interrupted publication')):
                with self.assertRaises(OSError): r.write_record(folder, 'run.json', record(), overwrite=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_concurrent_no_clobber_and_failed_flush_leave_no_partial_record(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/'run.json'
            def competing_writer(source, target):
                destination.write_bytes(b'other writer')
                raise FileExistsError(target)
            with patch.object(r.os, 'link', side_effect=competing_writer):
                with self.assertRaises(FileExistsError): r.write_record(folder, 'run.json', record())
            self.assertEqual(destination.read_bytes(), b'other writer')
            with patch.object(r.os, 'fsync', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError): r.write_record(folder, 'new.json', record())
            self.assertFalse((Path(folder)/'new.json').exists())
            with patch.object(r.os, 'link', side_effect=OSError('filesystem does not support hard links')):
                with self.assertRaises(OSError): r.write_record(folder, 'unsupported.json', record())
            self.assertFalse((Path(folder)/'unsupported.json').exists())
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_evidence_tamper_missing_and_verify_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = r.write_record(folder, 'run.json', record())
            entry = r.artifact_entry(folder, 'run.json', record_type='run', operation_id='baseline.capture', schema_version=1)
            index = record('evidence_index', [entry])
            before = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(r.verify_evidence_index(folder, index)['verified_artifacts'], 1)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            path.write_bytes(path.read_bytes().replace(b'm0-test', b'm0-xxxx'))
            with self.assertRaises(r.IntegrityError): r.verify_evidence_index(folder, index)
            path.unlink()
            with self.assertRaises(r.IntegrityError): r.verify_evidence_index(folder, index)
            self.assertFalse(path.exists())

    def test_evidence_schema_run_status_duplicates_and_self_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            r.write_record(folder, 'run.json', record())
            entry = r.artifact_entry(folder, 'run.json', record_type='run', operation_id='capture', schema_version=1)
            for changed in ({'schema_version':2}, {'bytes':True}, {'sha256':'nohash'}):
                with self.assertRaises(r.RecordError): r.verify_artifact(folder, {**entry,**changed})
            with self.assertRaises(r.IntegrityError): r.verify_artifact(folder,entry,run_id='other')
            with self.assertRaises(r.IntegrityError): r.verify_artifact(folder,{**entry,'status':'partial'})
            r.verify_artifact(folder,{**entry,'status':'partial'},require_complete=False)
            with self.assertRaises(r.IntegrityError): r.verify_evidence_index(folder,record('evidence_index',[entry,entry]))
            self_ref = {**entry,'path':'acceptance/evidence_index.json'}
            with self.assertRaises(r.IntegrityError): r.verify_evidence_index(folder,record('evidence_index',[self_ref]))
            malformed = record(); malformed['schema_version']=99
            Path(folder,'invalid.json').write_text(json.dumps(malformed))
            invalid = r.artifact_entry(folder,'invalid.json',record_type='run',operation_id='capture',schema_version=1)
            with self.assertRaises(r.RecordError): r.verify_artifact(folder,invalid)


class ConfigTests(unittest.TestCase):
    def config(self, root):
        (root/'repo').mkdir()
        (root/'Dataset').mkdir()
        directory = root/'configuration'; directory.mkdir()
        return directory/'m0.json', {
            'schema_version':1, 'spec_id':'M0-SPEC-001', 'spec_version':'1.0',
            'repo_root':'../repo', 'data_root':'../Dataset',
            'output_root':'../repo/artifacts/research/m0', 'sources':['cancer_binary'],
            'audit_workers':2, 'audit_seed':0, 'check_devices':['cpu','cuda'],
            'baseline_checks':['imports','cli_help','environment','regressions'],
            'lineage_path':'lineage.json', 'exposure_path':'exposure.json',
            'protocol_path':'protocol.json', 'literature_path':'literature.json',
            'mode':'pilot', 'timeouts':{'environment_seconds':120,'regression_seconds':600,'audit_seconds':600}}

    def test_input_config_paths_use_config_directory_without_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); config_file, config=self.config(root)
            resolved=r.validate_m0_config(config,config_file=config_file)
            self.assertEqual(Path(resolved['repo_root']),root/'repo')
            self.assertEqual(Path(resolved['lineage_path']),config_file.parent/'lineage.json')
            self.assertFalse(Path(resolved['output_root']).exists())
            self.assertEqual(config['repo_root'],'../repo')
            with self.assertRaises(r.RecordError): r.resolve_config_path('missing.json',config_file,kind='file')

    def test_input_config_rejects_unknown_fields_versions_commands_and_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            config_file, config=self.config(Path(folder))
            variants=[{'arbitrary_shell':'do something'}, {'schema_version':True}, {'schema_version':2},
                      {'spec_version':'future'}, {'mode':'confirmatory_frozen'}, {'sources':['breakhis_v2']},
                      {'audit_workers':0}, {'audit_workers':True}, {'audit_seed':-1}, {'audit_seed':2**32},
                      {'check_devices':['cuda']}, {'check_devices':['cpu','cpu']},
                      {'baseline_checks':['environment']}, {'baseline_checks':['arbitrary_command']},
                      {'timeouts':{'environment_seconds':0,'regression_seconds':1,'audit_seconds':1}}]
            for values in variants:
                with self.subTest(values=values), self.assertRaises(r.RecordError):
                    r.validate_m0_config({**config,**values},config_file=config_file)
            missing=dict(config); missing.pop('lineage_path')
            with self.assertRaises(r.RecordError): r.validate_m0_config(missing,config_file=config_file)

    def test_config_rejects_output_escape_and_dataset_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); config_file, config=self.config(root)
            for output in ('../Dataset/output','../repo/artifacts/research/m0/../../elsewhere', '../elsewhere'):
                with self.assertRaises(r.RecordError): r.validate_m0_config({**config,'output_root':output},config_file=config_file)
            with self.assertRaises(r.RecordError):
                r.validate_m0_config({**config,'data_root':'../repo'},config_file=config_file)


if __name__ == '__main__': unittest.main()
