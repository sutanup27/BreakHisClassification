"""M0.3 synthetic acceptance for M0-T06--T10 and M0-T15; never uses real images."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis.manifest import inspect_image
from breakhis.research import data_readiness as d
from breakhis.research import records as r


class DataReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve() / 'repo'
        self.repo.mkdir()
        self.data = Path(self.tmp.name).resolve() / 'data'
        self.data.mkdir()
        self.run = r.create_run_directory(self.repo/'artifacts/research/m0', 'synthetic')

    def image(self, slide='100AB', mag='40', code='B_A', sequence='001', color=(10,20,30), extension='.png', **options):
        label = 'benign' if code.startswith('B') else 'malignant'
        path = self.data/f'{mag}X/{label}/SOB_{code}-14-{slide}-{mag}-{sequence}{extension}'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12,8), color).save(path, **options)
        return path

    def audit(self, **kwargs):
        return d.audit_data(self.data, self.run, 'synthetic', repo_root=self.repo, **kwargs)

    def rows(self):
        return [json.loads(line) for line in (self.run/'audit/inventory.jsonl').read_text(encoding='utf-8').splitlines()]

    def record(self, name):
        record = r.read_record(self.run, 'audit/'+name+'.json')
        return record[r.RECORD_SHAPES[record['record_type']]]

    def test_t06_full_suffix_all_magnifications_dimensions_and_input_unchanged(self):
        for index, mag in enumerate(('40','100','200','400')):
            self.image(mag=mag, color=(index,30,50))
        before = {p.relative_to(self.data).as_posix(): r.file_sha256(p) for p in self.data.rglob('*.png')}
        result = self.audit(workers=2)
        self.assertTrue(result['data_integrity_ready'])
        self.assertFalse(result['exploratory_ready'])
        self.assertEqual(result['counts'], dict(discovered=4, included=4, excluded=0))
        rows = self.rows()
        self.assertEqual({row['filename_group_id'] for row in rows}, {'14-100AB'})
        self.assertEqual(len({row['duplicate_component_id'] for row in rows}), 1)
        self.assertEqual(self.record('group_summary')['missing_magnifications'], [])
        self.assertEqual(self.record('group_summary')['dimensions'], {'12x8':4})
        self.assertEqual(before, {p.relative_to(self.data).as_posix():r.file_sha256(p) for p in self.data.rglob('*.png')})
        canonical = inspect_image(('cancer_binary', self.data, next(self.data.rglob('*.png'))))
        actual = next(row for row in rows if row['relative_path']==canonical['path'])
        self.assertEqual(actual['decoded_pixel_sha256'], canonical['pixel_sha256'])

    def test_t07_alternate_encoding_exact_link_and_stable_representative(self):
        self.image(slide='100AB', compress_level=0)
        self.image(slide='200CD', compress_level=9)
        result = self.audit()
        rows = self.rows()
        self.assertEqual(len({row['raw_sha256'] for row in rows}),2)
        self.assertEqual(len({row['decoded_pixel_sha256'] for row in rows}),1)
        self.assertEqual(len(self.record('duplicate_components')),1)
        self.assertEqual(result['counts'], dict(discovered=2,included=1,excluded=1))
        retained = next(row for row in rows if row['status']=='included')
        self.assertEqual({row['representative_image_id'] for row in rows}, {retained['image_id']})
        original = [(row['image_id'],row['duplicate_component_id'],row['representative_image_id']) for row in rows]
        second = r.create_run_directory(self.repo/'artifacts/research/m0', 'second')
        d.audit_data(self.data, second, 'second', repo_root=self.repo, workers=2)
        other = [json.loads(line) for line in (second/'audit/inventory.jsonl').read_text().splitlines()]
        self.assertEqual(original, [(row['image_id'],row['duplicate_component_id'],row['representative_image_id']) for row in other])

    def test_t08_duplicate_label_conflict_blocks_without_silent_label_fix(self):
        self.image()
        self.image(slide='200', code='M_DC')
        result = self.audit()
        self.assertFalse(result['data_integrity_ready'])
        self.assertIn('DATA_LABEL_CONFLICT', result['blockers'])
        self.assertEqual(result['counts']['included'], 2)
        self.assertTrue(self.record('duplicate_components')[0]['label_conflict'])

    def test_t08_same_filename_group_mixed_labels_blocks_even_without_pixel_match(self):
        self.image(color=(0,0,0))
        self.image(code='M_DC', mag='100', color=(255,0,0))
        self.assertIn('DATA_LABEL_CONFLICT', self.audit()['blockers'])

    def test_t09_corrupt_bad_name_folder_and_unreadable_are_all_accounted(self):
        self.image()
        corrupt = self.image(slide='201',color=(1,2,3))
        corrupt.write_bytes(b'bad image')
        malformed = self.data/'40X/benign/not-breakhis.png'
        Image.new('RGB',(3,2)).save(malformed)
        misplaced = self.image(slide='300', mag='100', color=(2,3,4))
        misplaced.rename(self.data/'40X/benign'/misplaced.name)
        unreadable = self.image(slide='400',color=(3,4,5))
        (self.data/'notes.txt').write_text('ignored')
        original = Path.read_bytes
        def read(path):
            if path == unreadable:
                raise PermissionError('fixture unreadable image')
            return original(path)
        with patch.object(Path, 'read_bytes', read):
            result = self.audit()
        self.assertEqual(result['counts'],dict(discovered=5,included=1,excluded=4))
        rows = {Path(row['relative_path']).name:row for row in self.rows()}
        self.assertEqual(rows[corrupt.name]['raw_sha256'],hashlib.sha256(b'bad image').hexdigest())
        self.assertIsNone(rows[corrupt.name]['decoded_pixel_sha256'])
        self.assertIsNotNone(rows[malformed.name]['decoded_pixel_sha256'])
        self.assertIsNone(rows[malformed.name]['filename_group_id'])
        self.assertEqual(rows[unreadable.name]['exclusion_reason_code'],'IMAGE_UNREADABLE')
        self.assertIn('raw_sha256',rows[unreadable.name]['unavailable_reasons'])
        self.assertEqual(self.record('group_summary')['ignored_non_images'],['notes.txt'])
        self.assertEqual(len((self.run/'audit/exclusions.jsonl').read_text().splitlines()),4)

    def test_supported_non_png_extension_uses_same_stem_parser(self):
        self.image(extension='.bmp')
        self.assertEqual(self.audit()['counts']['included'],1)

    def test_t10_missing_and_empty_sources_have_blocked_records(self):
        self.data.rmdir()
        result = self.audit()
        self.assertIn('SOURCE_MISSING',result['blockers'])
        self.assertEqual(result['counts']['discovered'],0)
        self.data.mkdir()
        second = r.create_run_directory(self.repo/'artifacts/research/m0','empty')
        result = d.audit_data(self.data,second,'empty',repo_root=self.repo)
        self.assertIn('SOURCE_EMPTY_OR_NO_INCLUDED_IMAGES',result['blockers'])

    def test_t10_absent_magnifications_reported_without_imputation(self):
        self.image()
        self.audit()
        self.assertEqual(self.record('group_summary')['missing_magnifications'],['100X','200X','400X'])
        self.assertEqual(self.record('group_summary')['counts']['discovered'],1)

    def test_t15_unknown_lineage_and_identity_stays_unknown(self):
        self.image()
        result = self.audit()
        lineage = self.record('lineage')
        self.assertTrue(all(f['value'] is None and f['status']=='unknown' for f in lineage['facts'].values()))
        self.assertTrue(all(q['owner'] and q['freeze_by'] for q in lineage['open_questions']))
        self.assertIsNone(self.rows()[0]['verified_patient_id'])
        self.assertIn('LINEAGE_UNRESOLVED_SCOPE_DECISION_REQUIRED',result['restrictions'])
        near = self.record('near_duplicate_review')
        self.assertEqual(near['status'],'incomplete')
        self.assertFalse(near['automatic_similar_image_exclusions'])
        self.assertFalse(result['confirmatory_ready'])

    def test_lineage_supplied_facts_are_validated_not_promoted(self):
        fact=dict(value='operator backup',status='self_reported',reason=None,evidence_refs=['operator:statement'])
        result=d.lineage_record(dict(schema_version=1,reviewer=None,facts={'backup_location':fact}))
        self.assertEqual(result['facts']['backup_location']['status'],'self_reported')
        fact['status']='verified'
        with self.assertRaises(r.RecordError):
            d.lineage_record(dict(schema_version=1,reviewer=None,facts={'backup_location':fact}))
        with self.assertRaises(r.RecordError):
            d.lineage_record(dict(schema_version=1,reviewer='reviewer',facts={'invented':fact}))

    def test_near_similar_tissue_is_not_automatically_excluded(self):
        self.image(color=(10,20,30))
        self.image(slide='200',color=(10,20,31))
        result=self.audit()
        self.assertEqual(result['counts']['included'],2)
        self.assertIn('NEAR_DUPLICATE_SCREENING_INCOMPLETE',result['restrictions'])

    def test_content_change_during_audit_blocks_readiness(self):
        path=self.image()
        original=d._components
        def changed(rows):
            path.write_bytes(b'changed after inspection')
            return original(rows)
        with patch.object(d,'_components',side_effect=changed):
            self.assertIn('SOURCE_CONTENT_CHANGED',self.audit()['blockers'])

    def test_file_set_change_during_audit_blocks_readiness(self):
        self.image()
        original=d._components
        def changed(rows):
            self.image(slide='999',color=(1,1,1))
            return original(rows)
        with patch.object(d,'_components',side_effect=changed):
            self.assertIn('SOURCE_FILE_SET_CHANGED',self.audit()['blockers'])

    def test_existing_output_overlap_and_bad_inputs_refused(self):
        with self.assertRaises(r.RecordError): self.audit(workers=True)
        self.assertFalse((self.run/'audit').exists())
        with self.assertRaises(r.RecordError):
            d.audit_data(self.run,self.run,'synthetic',repo_root=self.repo)
        self.image()
        self.audit()
        with self.assertRaises(FileExistsError): self.audit()

    def legacy_rows(self):
        return [dict(source=d.SOURCE,path=row['relative_path'],sha256=row['raw_sha256'],
                     pixel_sha256=row['decoded_pixel_sha256'],patient_id=row['filename_group_id'],
                     label=row['class_label'],magnification=row['magnification'],split='train')
                for row in self.rows() if row['status']=='included']

    def test_legacy_split_reuse_checks_hash_fileset_and_component_boundaries(self):
        self.image(color=(1,2,3))
        self.image(mag='100',color=(2,3,4))
        self.audit()
        rows=self.legacy_rows()
        manifest=dict(schema_version=1,rows=rows)
        result=d.validate_legacy_manifest(manifest,self.rows())
        self.assertEqual(result['assignment_kind'],'legacy_pilot_assignment')
        self.assertFalse(result['research_outer_folds'])
        rows[1]['split']='test'
        with self.assertRaises(r.IntegrityError): d.validate_legacy_manifest(manifest,self.rows())
        rows[1]['split']='train'
        rows[0]['sha256']='0'*64
        with self.assertRaises(r.IntegrityError): d.validate_legacy_manifest(manifest,self.rows())
        with self.assertRaises(r.IntegrityError): d.validate_legacy_manifest(dict(schema_version=1,rows=[]),self.rows())

    def test_invalid_historical_manifest_produces_blocked_reuse_evidence(self):
        self.image()
        path=self.repo/'old-manifest.json'
        path.write_text(json.dumps(dict(schema_version=1,rows=[])))
        digest=r.file_sha256(path)
        result=self.audit(legacy_manifest=path)
        self.assertIn('LEGACY_MANIFEST_INVALID',result['blockers'])
        self.assertEqual(r.file_sha256(path),digest)

    def test_valid_historical_manifest_is_labeled_legacy_not_research_folds(self):
        self.image()
        self.audit()
        path=self.repo/'old-manifest.json'
        path.write_text(json.dumps(dict(schema_version=1,rows=self.legacy_rows())))
        second=r.create_run_directory(self.repo/'artifacts/research/m0','legacy')
        result=d.audit_data(self.data,second,'legacy',repo_root=self.repo,legacy_manifest=path)
        self.assertTrue(result['data_integrity_ready'])
        self.assertEqual(result['legacy']['assignment_kind'],'legacy_pilot_assignment')
        self.assertFalse(result['legacy']['research_outer_folds'])

    def test_link_directory_is_not_traversed_or_hashed(self):
        outside=Path(self.tmp.name).resolve()/'outside'
        outside.mkdir()
        (outside/'SOB_B_A-14-100-40-001.png').write_bytes(b'outside sentinel')
        alias=self.data/'alias'
        junction=False
        try:
            alias.symlink_to(outside,target_is_directory=True)
        except OSError:
            if os.name!='nt': raise
            import subprocess
            self.assertTrue(alias.absolute().is_relative_to(Path(self.tmp.name).resolve()))
            self.assertTrue(outside.is_relative_to(Path(self.tmp.name).resolve()))
            result=subprocess.run(['cmd','/c','mklink','/J',str(alias),str(outside)],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            junction=True
        try:
            result=self.audit()
            self.assertFalse(result['data_integrity_ready'])
            self.assertTrue(any('LINK_DIRECTORY_REJECTED' in reason for reason in result['blockers']))
            self.assertEqual(self.rows(),[])
            self.assertEqual((outside/'SOB_B_A-14-100-40-001.png').read_bytes(),b'outside sentinel')
        finally:
            if junction: os.rmdir(alias)
            else: alias.unlink()

    def test_bad_lineage_is_rejected_before_output_creation(self):
        with self.assertRaises(r.RecordError):
            self.audit(lineage={'schema_version':9,'reviewer':None,'facts':{}})
        self.assertFalse((self.run/'audit').exists())


if __name__=='__main__': unittest.main()
