"""M1-T01--T04: preparation/identity checks without fitting or inference."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis import experiments as e, experiment_setup as setup, experiment_io as storage
from breakhis import paths

ROOT=Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.runs=self.root/'runs'; self.data=self.root/'data'; self.data.mkdir()
        self.enterContext(patch.object(e,'RUNS',self.runs))
        self.enterContext(patch.object(paths,'CHECKPOINT_ROOT',self.root/'checkpoint'))
        rows=[]
        for index,(split,label) in enumerate((s,l) for s in ('train','validation','test') for l in ('benign','malignant')):
            code='B_A' if label=='benign' else 'M_DC'
            relative=f'40X/{label}/SOB_{code}-14-{1000+index}-40-001.png'
            file=self.data/relative; file.parent.mkdir(parents=True,exist_ok=True)
            image=Image.new('RGB',(8,8),(index*30,10,20)); image.save(file)
            group=f'14-{1000+index}'
            rows.append(dict(source='cancer_binary',path=relative,filename=file.name,split=split,
                label=label,subtype='adenosis' if label=='benign' else 'ductal_carcinoma',
                patient_id=group,group_id=group,magnification='40X',width=8,height=8,
                sha256=e.digest(file),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(schema_version=1,seed=101,
            roots={'cancer_binary':str(self.data)},rows=rows)))
        self.config=e.read_json(ROOT/'configs/experiments/vgg_baseline.json')
        self.config['data'].update(manifest=str(self.manifest),data_root=str(self.data))
        self.config['training']['device']='cpu'
        self.file=self.root/'config.json'; self.save_config()

    def save_config(self): self.file.write_text(json.dumps(self.config))

    def prepare(self,run_id='prepared'):
        return setup.prepare_run(self.file,run_id=run_id)

    def test_t01_prepares_frozen_inputs_and_phase_records_without_execution(self):
        with patch.object(e,'run_experiment',side_effect=AssertionError('Training invoked')),contextlib.redirect_stdout(io.StringIO()):
            e.main(['--config',str(self.file),'--run-id','prepared','--phase','development','--prepare-only','--seed','77'])
        run=self.runs/'prepared'; state=e.read_json(run/'run.json'); inputs=e.read_json(run/'inputs.json')
        self.assertEqual(state['schema_version'],2); self.assertEqual(state['status'],'prepared')
        self.assertEqual(state['stages'],{})
        self.assertEqual(state['phases']['final']['status'],'not_requested')
        self.assertEqual(inputs['seeds'],dict(split=101,fitting=77))
        self.assertEqual((run/'manifest.json').read_bytes(),self.manifest.read_bytes())
        self.assertEqual(e.digest(run/'config.json'),inputs['identity']['config_sha256'])
        self.assertEqual(e.read_json(run/'results.json')['phases'],dict(development=None,final=None))
        self.assertTrue(Path(inputs['checkpoint_directory']).is_relative_to(self.root/'checkpoint/binary'))
        self.assertFalse(list((self.root/'checkpoint').rglob('*.pth')))

    def test_t01_phase_requests_reject_incompatible_arguments_and_unimplemented_execution(self):
        requests=[dict(config_file=None,run_id='x',phase='development'),
            dict(config_file=None,run_id=None,phase='final'),
            dict(config_file=self.file,run_id='x',phase='unknown'),
            dict(config_file=None,run_id='x',phase='final',seed=0),
            dict(config_file=None,run_id='x',phase='final',device='cpu')]
        for request in requests:
            with self.subTest(request=request),self.assertRaises(ValueError): setup.validate_request(**request)
        with patch.object(e,'run_experiment',side_effect=AssertionError('Legacy fallback')),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): e.main(['--config',str(self.file),'--run-id','x','--phase','development'])
        self.assertFalse(self.runs.exists())

    def test_t01_malformed_scalars_and_overflow_fail_before_output(self):
        for value in (None,True,12,[],{}):
            self.config['name']=value; self.save_config()
            with self.assertRaises(ValueError): self.prepare()
        self.config['name']='valid'; self.config['training']['epochs']=True; self.save_config()
        with self.assertRaises(ValueError): self.prepare()
        self.file.write_text('{"value":1e999}')
        with self.assertRaises(ValueError): e.read_json(self.file)
        self.assertFalse(self.runs.exists())

    def test_t02_existing_ids_and_source_overlap_refuse_without_clobber(self):
        result=self.prepare(); original=(result['run']/'config.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'exists'): self.prepare()
        self.assertEqual((result['run']/'config.json').read_bytes(),original)
        with self.assertRaises(ValueError): setup.prepare_run(self.file,run_id='../escape')
        with patch.object(e,'RUNS',self.data/'runs'):
            with self.assertRaisesRegex(ValueError,'protected'): self.prepare('overlap')
        self.assertFalse((self.data/'runs').exists())

    def test_t02_report_aliases_and_prediction_checkpoint_paths_are_protected(self):
        from breakhis.cli import report_paths
        report=self.root/'report.json'
        with self.assertRaises(ValueError): report_paths(self.manifest,protected=[self.manifest])
        with self.assertRaises(ValueError): report_paths(report,report)
        with self.assertRaises(ValueError): report_paths(report,report.with_suffix('.config.json'))
        with self.assertRaises(ValueError): report_paths(report,self.root/'weights.pth')
        with self.assertRaises(ValueError): report_paths(report,self.manifest,protected=[self.manifest])
        self.assertTrue(self.manifest.is_file()); self.assertFalse(report.exists())

    def test_t02_linked_output_cannot_be_hidden_by_resolution(self):
        self.runs.mkdir(); target=self.root/'elsewhere'; target.mkdir(); alias=self.runs/'alias'; junction=False
        try:
            alias.symlink_to(target,target_is_directory=True)
        except OSError:
            if os.name!='nt': raise
            self.assertTrue(alias.absolute().is_relative_to(self.root))
            self.assertTrue(target.is_relative_to(self.root))
            result=subprocess.run(['cmd','/c','mklink','/J',str(alias),str(target)],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr); junction=True
        try:
            with self.assertRaisesRegex(ValueError,'linked'): paths.output_path(alias/'escape.json',root=self.runs)
            self.assertFalse((target/'escape.json').exists())
        finally:
            if junction: os.rmdir(alias)
            else: alias.unlink()

    def test_t03_atomic_failure_preserves_old_bytes_and_cleans_temporary(self):
        file=self.root/'atomic.json'; storage.write_json(file,dict(original=True))
        original=file.read_bytes()
        with patch.object(storage.os,'replace',side_effect=OSError('injected replace failure')):
            with self.assertRaises(OSError): storage.write_json(file,dict(replacement=True))
        self.assertEqual(file.read_bytes(),original); self.assertFalse(list(self.root.glob('.*.tmp')))
        with self.assertRaises(FileExistsError): storage.write_json(file,{},overwrite=False)
        with self.assertRaises(ValueError): storage.write_json(self.root/'new/invalid.json',dict(x=float('nan')))
        self.assertFalse((self.root/'new').exists())

    def test_t03_failed_setup_retains_failed_phase_without_successful_inputs(self):
        original=storage.publish
        def injected(path,content,**kwargs):
            if Path(path).name=='config.json': raise OSError('injected config publication failure')
            return original(path,content,**kwargs)
        with patch.object(storage,'publish',side_effect=injected):
            with self.assertRaises(OSError): self.prepare()
        run=self.runs/'prepared'; state=e.read_json(run/'run.json')
        self.assertEqual(state['status'],'failed')
        self.assertEqual(state['phases']['development']['status'],'failed')
        self.assertEqual(state['stages'],{}); self.assertFalse((run/'config.json').exists())
        self.assertFalse((run/'inputs.json').exists())

    def test_t04_frozen_config_manifest_and_source_drift_rejected(self):
        for name in ('config.json','manifest.json'):
            result=self.prepare(name.replace('.','-')); file=result['run']/name
            file.write_bytes(file.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'Frozen'): setup.prepare_run(self.file,run_id=result['state']['run_id'],resume=True)
        result=self.prepare('source-drift'); changed=e.revision(); changed['source_sha256']={'changed':'0'*64}
        with patch.object(e,'revision',return_value=changed):
            with self.assertRaisesRegex(ValueError,'Implementation'): setup.prepare_run(self.file,run_id='source-drift',resume=True)

    def test_t04_effective_config_and_original_manifest_drift_rejected(self):
        self.prepare()
        self.config['training']['epochs']+=1; self.save_config()
        with self.assertRaisesRegex(ValueError,'Config differs'): setup.prepare_run(self.file,run_id='prepared',resume=True)
        self.config['training']['epochs']-=1; self.save_config()
        self.manifest.write_text(self.manifest.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Original manifest'): setup.prepare_run(self.file,run_id='prepared',resume=True)

    def test_t04_missing_git_is_unavailable_not_a_fake_clean_commit(self):
        with patch.object(e.subprocess,'run',side_effect=FileNotFoundError('git missing')):
            recorded=e.revision()
        self.assertIsNone(recorded['commit']); self.assertIsNone(recorded['dirty'])
        self.assertFalse(recorded['git_available']); self.assertTrue(recorded['source_sha256'])

    def test_t04_artifact_references_verify_bytes_and_roots(self):
        result=self.prepare(); run=result['run']
        reference=setup.artifact_reference(run/'config.json',root=run,kind='run')
        self.assertEqual(setup.verify_reference(reference,roots={'run':run}),run/'config.json')
        escaped=dict(reference,path='../config.json')
        with self.assertRaises(ValueError): setup.verify_reference(escaped,roots={'run':run})
        (run/'config.json').write_text('{}')
        with self.assertRaises(ValueError): setup.verify_reference(reference,roots={'run':run})

    def test_prepared_final_is_not_completed_development_and_never_writes(self):
        self.prepare(); run=self.runs/'prepared'; before={p.name:p.read_bytes() for p in run.iterdir() if p.is_file()}
        with self.assertRaisesRegex(ValueError,'completed development'): setup.prepare_run(run_id='prepared',phase='final')
        self.assertEqual(before,{p.name:p.read_bytes() for p in run.iterdir() if p.is_file()})

    def test_expected_stage_artifacts_include_histories_predictions_and_pruning(self):
        self.config['pruning']=dict(method='channel',sparsity=.5,finetune_epochs=1); self.save_config()
        result=self.prepare(); outputs=result['inputs']['expected_artifacts']
        self.assertTrue(any(p.endswith('train.history.json') for p in outputs['train']))
        self.assertTrue(any(p.endswith('prune.validation.json') for p in outputs['prune']))
        self.assertEqual(len(outputs['final-finetuned']),2)
        self.assertIn(str(result['run']/'train.history.json'),outputs['development-complete'])
        self.assertIn(str(result['run']/'finetune.history.json'),outputs['development-complete'])
        setup.prepare_run(self.file,run_id='prepared',resume=True)  # Read-only preparation validation.


if __name__=='__main__': unittest.main()
