"""M0.4 synthetic metadata-only acceptance: M0-T11--T14 and coverage failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis.research import records as r
from breakhis.research import data_readiness as d
from breakhis.research import exposure as e


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name).resolve()/'repo'
        self.repo.mkdir()
        self.data=Path(self.tmp.name).resolve()/'data'
        self.history=Path(self.tmp.name).resolve()/'history'
        self.history.mkdir()
        self.run=r.create_run_directory(self.repo/'artifacts/research/m0','fixture')
        for slide,mag,color in [('100AB','40',(1,2,3)),('100AB','100',(2,3,4)),
                                ('200CD','40',(1,2,3)),('300EF','40',(3,4,5)),('400GH','100',(4,5,6))]:
            path=self.data/f'{mag}X/benign/SOB_B_A-14-{slide}-{mag}-001.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(12,8),color).save(path)
        d.audit_data(self.data,self.run,'fixture',repo_root=self.repo)
        self.rows=[json.loads(line) for line in (self.run/'audit/inventory.jsonl').read_text().splitlines()]
        self.manifest=dict(schema_version=1,rows=[dict(source='cancer_binary',path=row['relative_path'],
            patient_id=row['filename_group_id'],magnification=row['magnification'],
            sha256=row['raw_sha256'],pixel_sha256=row['decoded_pixel_sha256'],
            split='test' if row['filename_group_id'] in ('14-100AB','14-200CD') else 'train') for row in self.rows])
        self.manifest_ref=self.metadata('manifest.json',self.manifest)
        self.log_ref=self.metadata('access.log','opaque log is hashed, never parsed')

    def metadata(self,name,value):
        path=self.history/name
        path.write_text(json.dumps(value) if not isinstance(value,str) else value,encoding='utf-8')
        return dict(path=name,sha256=r.file_sha256(path))

    def event(self,**kwargs):
        event=dict(event_id='old-eval',artifact=self.log_ref,manifest=self.manifest_ref,
            source='cancer_binary',source_instance_id='current-derivative',source_kind='real',
            selected_partition='test',magnification_filter=['40X'],access_type='test_metric_inspection',
            accessed_paths=None,evidence_source='log',occurred_at=None,reviewer_note='fixture declaration')
        event.update(kwargs)
        return event

    def coverage(self,**kwargs):
        value=dict(schema_version=1,status='reviewed',reviewer='fixture reviewer',
            inventory_sha256=r.semantic_sha256(self.rows,domain='m0.exposure_inventory'),
            source_instance_id='current-derivative',component_ids=sorted({row['duplicate_component_id'] for row in self.rows}),
            events_complete=True,evidence_refs=[self.metadata('coverage.md','Reviewed fixture history scope; no other accesses declared')],
            reviewer_note='Synthetic test declaration, not a claim about real study history')
        value.update(kwargs)
        return value

    def resolve(self,events=None,**kwargs):
        return e.resolve_exposure(self.run,'fixture',history_root=self.history,
            source_instance_id='current-derivative',events=[] if events is None else events,**kwargs)

    def record(self,kind):
        record=r.read_record(self.run,'governance/'+kind+'.json')
        return record[r.RECORD_SHAPES[record['record_type']]]

    def test_t11_exact_log_filter_propagates_all_linked_magnifications(self):
        path=next(row['relative_path'] for row in self.rows if row['filename_group_id']=='14-100AB' and row['magnification']=='40X')
        self.resolve([self.event(accessed_paths=[path])],history_coverage=self.coverage())
        item=self.record('exposure_register')[0]
        self.assertEqual(item['coverage'],'exact')
        self.assertEqual(item['historical_paths'],[path])
        self.assertEqual(len(item['image_ids']),3)
        self.assertEqual(item['filename_group_ids'],['14-100AB','14-200CD'])
        statuses=self.record('component_eligibility')
        self.assertEqual(sum(x['status']=='development_only' for x in statuses),1)
        self.assertEqual(sum(x['status']=='candidate_untouched_with_reviewed_coverage' for x in statuses),2)

    def test_t12_no_per_image_log_is_conservative_selected_partition_superset(self):
        result=self.resolve([self.event()],history_coverage=self.coverage())
        item=self.record('exposure_register')[0]
        self.assertEqual(item['coverage'],'conservative_superset')
        self.assertEqual(len(item['historical_paths']),2)
        self.assertEqual(len(item['image_ids']),3)
        self.assertTrue(result['history_review_ready'])
        self.assertFalse(result['confirmatory_design_ready'])

    def test_operator_recollections_never_promoted_to_exact(self):
        path=next(row['relative_path'] for row in self.rows if row['filename_group_id']=='14-100AB' and row['magnification']=='40X')
        self.resolve([self.event(evidence_source='operator_attestation',accessed_paths=[path])])
        item=self.record('exposure_register')[0]
        self.assertEqual(item['coverage'],'conservative_superset')
        self.assertEqual(item['confidence'],'self_reported')

    def test_t13_missing_manifest_blocks_all_candidate_untouched_claims(self):
        result=self.resolve([self.event(manifest={'path':'missing.json','sha256':'0'*64})],history_coverage=self.coverage())
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['exposure_unknown'],3)
        self.assertEqual(self.record('exposure_register')[0]['coverage'],'unresolved')

    def test_t13_contradictory_filter_is_unresolved_with_known_exposure_retained(self):
        self.resolve([self.event(accessed_paths=['40X/benign/not-in-manifest.png'])],history_coverage=self.coverage())
        item=self.record('exposure_register')[0]
        self.assertIn('PER_IMAGE_LOG_CONTRADICTS_FILTER',item['issues'])
        self.assertTrue(item['duplicate_component_ids'])
        self.assertFalse(any(x['status'].startswith('candidate') for x in self.record('component_eligibility')))

    def test_t13_changed_manifest_hash_unresolved(self):
        (self.history/'manifest.json').write_text('{}')
        self.assertFalse(self.resolve([self.event()],history_coverage=self.coverage())['history_review_ready'])

    def test_t14_synthetic_realistic_names_do_not_quarantine_real_components(self):
        result=self.resolve([self.event(source_kind='synthetic',source_instance_id='synthetic-smoke',access_type='synthetic_check',manifest=None)],history_coverage=self.coverage())
        self.assertEqual(result['eligibility_counts']['development_only'],0)
        self.assertEqual(result['eligibility_counts']['candidate_untouched_with_reviewed_coverage'],3)
        item=self.record('exposure_register')[0]
        self.assertEqual(item['restriction'],'synthetic_not_real')
        self.assertEqual(item['image_ids'],[])

    def test_synthetic_identity_collision_is_unresolved_not_untouched(self):
        result=self.resolve([self.event(source_kind='synthetic',access_type='synthetic_check')],history_coverage=self.coverage())
        self.assertEqual(result['eligibility_counts']['candidate_untouched_with_reviewed_coverage'],0)
        self.assertFalse(result['history_review_ready'])

    def test_no_logs_or_coverage_cannot_mean_unexposed(self):
        result=self.resolve()
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['exposure_unknown'],3)

    def test_reviewed_partial_scope_cannot_approve_entire_history(self):
        coverage=self.coverage()
        coverage['component_ids']=coverage['component_ids'][:1]
        result=self.resolve(history_coverage=coverage)
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['candidate_untouched_with_reviewed_coverage'],1)
        self.assertEqual(result['eligibility_counts']['exposure_unknown'],2)

    def test_stale_inventory_coverage_cannot_approve_candidates(self):
        result=self.resolve(history_coverage=self.coverage(inventory_sha256='0'*64))
        self.assertEqual(result['eligibility_counts']['exposure_unknown'],3)

    def test_unreviewed_or_missing_evidence_coverage_blocked(self):
        result=self.resolve(history_coverage=self.coverage(events_complete=False))
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['exposure_unknown'],3)

    def test_equivalent_derivative_exact_hashes_propagate_without_name_assumptions(self):
        result=self.resolve([self.event(source_instance_id='other-derivative')],history_coverage=self.coverage())
        self.assertEqual(result['eligibility_counts']['development_only'],1)

    def test_unknown_derivative_name_alone_cannot_match_real_groups(self):
        for row in self.manifest['rows']:
            row.update(sha256='a'*64,pixel_sha256='b'*64)
        ref=self.metadata('changed-derivative.json',self.manifest)
        result=self.resolve([self.event(source_instance_id='other-derivative',manifest=ref)],history_coverage=self.coverage())
        self.assertEqual(result['eligibility_counts']['development_only'],0)
        self.assertFalse(result['history_review_ready'])

    def test_reviewed_derivative_equivalence_bridge_propagates_restrictions(self):
        for row in self.manifest['rows']:
            row.update(sha256='a'*64,pixel_sha256='b'*64)
        ref=self.metadata('changed-derivative.json',self.manifest)
        component=next(row['duplicate_component_id'] for row in self.rows if row['filename_group_id']=='14-100AB')
        bridges=[dict(source_instance_id='other-derivative',filename_group_id=group,current_component_id=component,
                      reviewer='fixture reviewer',evidence_refs=[self.metadata('bridge.md','Reviewed fixture equivalence')])
                 for group in ('14-100AB','14-200CD')]
        result=self.resolve([self.event(source_instance_id='other-derivative',manifest=ref)],history_coverage=self.coverage(),equivalences=bridges)
        self.assertEqual(result['eligibility_counts']['development_only'],1)
        self.assertTrue(result['history_review_ready'])

    def test_same_instance_content_contradiction_retains_development_restriction(self):
        for row in self.manifest['rows']:
            row['sha256']='a'*64
        ref=self.metadata('contradiction.json',self.manifest)
        result=self.resolve([self.event(manifest=ref)],history_coverage=self.coverage())
        self.assertEqual(result['eligibility_counts']['development_only'],1)
        self.assertFalse(result['history_review_ready'])

    def test_resolver_never_decodes_images_or_loads_predictions(self):
        event=self.event()
        with patch.object(Image,'open',side_effect=AssertionError('image decoding during governance')):
            self.resolve([event],history_coverage=self.coverage())
        self.assertEqual(self.record('exposure_register')[0]['coverage'],'conservative_superset')
        extra=self.event(probabilities=[0.1])
        with self.assertRaises(r.RecordError): e.validate_event(extra)

    def test_malformed_input_and_existing_records_refused(self):
        with self.assertRaises(r.RecordError): self.resolve([self.event(accessed_paths=[{}])])
        self.assertFalse((self.run/'governance').exists())
        self.resolve()
        before=r.file_sha256(self.run/'governance/exposure_register.json')
        with self.assertRaises(FileExistsError): self.resolve()
        self.assertEqual(r.file_sha256(self.run/'governance/exposure_register.json'),before)

    def test_inventory_tampering_rejected(self):
        path=self.run/'audit/inventory.jsonl'
        rows=copy.deepcopy(self.rows)
        rows[0]['raw_sha256']='0'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaises(r.IntegrityError): self.resolve()

    def test_moving_historical_metadata_blocks_candidates(self):
        original=e._manifest_rows
        def changed(path,event):
            result=original(path,event)
            path.write_text('{}')
            return result
        with patch.object(e,'_manifest_rows',side_effect=changed):
            result=self.resolve([self.event()],history_coverage=self.coverage())
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['candidate_untouched_with_reviewed_coverage'],0)
        self.assertIn('HISTORICAL_REFERENCE_UNAVAILABLE_OR_CHANGED',self.record('exposure_register')[0]['issues'])

    def test_empty_access_log_is_not_proof_of_no_inspection(self):
        result=self.resolve([self.event(accessed_paths=[])],history_coverage=self.coverage())
        self.assertFalse(result['history_review_ready'])
        self.assertIn('EMPTY_PER_IMAGE_ACCESS_LOG',self.record('exposure_register')[0]['issues'])

    def test_reseeding_does_not_erase_historical_exposure(self):
        coverage=self.coverage()
        self.resolve([self.event()],history_coverage=coverage)
        before=self.record('component_eligibility')
        self.manifest['seed']=9999
        ref=self.metadata('reseeded.json',self.manifest)
        new_run=r.create_run_directory(self.repo/'artifacts/research/m0','reseed')
        d.audit_data(self.data,new_run,'reseed',repo_root=self.repo)
        e.resolve_exposure(new_run,'reseed',history_root=self.history,source_instance_id='current-derivative',
                           events=[self.event(manifest=ref)],history_coverage=coverage)
        after=r.read_record(new_run,'governance/component_eligibility.json')['items']
        self.assertEqual(before,after)

    def test_moving_equivalence_evidence_blocks_candidates(self):
        component=self.rows[0]['duplicate_component_id']
        ref=self.metadata('bridge.md','Reviewed synthetic fixture only')
        bridge=dict(source_instance_id='other-derivative',filename_group_id='14-100AB',
                    current_component_id=component,reviewer='fixture reviewer',evidence_refs=[ref])
        original=e._manifest_rows
        def changed(path,event):
            result=original(path,event)
            (self.history/'bridge.md').write_text('changed review evidence')
            return result
        with patch.object(e,'_manifest_rows',side_effect=changed):
            result=self.resolve([self.event(source_instance_id='other-derivative')],
                                history_coverage=self.coverage(),equivalences=[bridge])
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(result['eligibility_counts']['candidate_untouched_with_reviewed_coverage'],0)
        self.assertIn('EQUIVALENCE_EVIDENCE_UNAVAILABLE_OR_CHANGED',self.record('exposure_register')[0]['issues'])


if __name__=='__main__': unittest.main()
