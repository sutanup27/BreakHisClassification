"""M2-T01--T04: prospective plans, role isolation and bounded resource accounting."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis import baseline_plan as b, experiments as e, experiment_io as storage, paths
import test_experiments as fixtures

ROOT=Path(__file__).resolve().parents[1]


class BaselinePlanTests(unittest.TestCase):
    save=fixtures.ExperimentTests.save

    def setUp(self):
        fixtures.ExperimentTests.setUp(self)
        from breakhis.manifest import inspect_image
        original=list(self.rows)
        for mag in (100,200,400):
            for index,row in enumerate(original):
                filename=Path(row['path']).name.replace('-40-','-'+str(mag)+'-')
                file=self.images/f'{mag}X'/row['label']/filename; file.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(8,8),(index*30,mag%256,40)).save(file)
                value=inspect_image(('cancer_binary',self.images,file))
                self.rows.append(dict(value,group_id=row['group_id'],split=row['split']))
        self.save()
        self.study=e.read_json(ROOT/'configs/baselines/m2_study.json')
        self.study['experiment_template']=str(self.file)
        self.study_file=self.root/'study.json'; self.save_study()
        self.addCleanup(patch.stopall)
        patch.object(b,'STUDIES',self.root/'studies').start()
        patch.object(e,'RUNS',self.root/'runs').start()
        patch.object(paths,'CHECKPOINT_ROOT',self.root/'checkpoints').start()

    def save_study(self): self.study_file.write_text(json.dumps(self.study))
    def compile(self): return b.compile_plan(self.study_file)

    def costs(self,plan,epoch_seconds=2,fixed_seconds=3):
        return dict(schema_version=1,plan_sha256=b.sha(plan),estimates={j['cost_key']:
            dict(epoch_seconds=epoch_seconds,fixed_seconds=fixed_seconds,basis='Synthetic hand-counted test input, not a hardware claim')
            for j in plan['fit_jobs']})

    def test_t01_invalid_scalars_fields_profiles_and_recipes_refuse_before_output(self):
        original=copy.deepcopy(self.study)
        cases=[('schema_version',True),('primary_seed',True),('ranking_seed',-1),('repeat_seed',102),
            ('dense_epochs',0),('available_wall_hours',float('inf')),('max_fit_jobs',True),
            ('ratios',[.25,.25]),('ratios',[True]),('repeat_ratio',.9),('main_magnification','40X'),
            ('compact_model','unknown'),('study_id','../escape'),('study_id','CON'),('extra',1)]
        for key,value in cases:
            self.study=copy.deepcopy(original); self.study[key]=value; self.save_study()
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): b.prepare(self.study_file)
        self.study=copy.deepcopy(original)
        for mutation in ('duplicate_recipe','unknown_recipe_field','cpu_batch','duplicate_profile','bool_lr'):
            self.study=copy.deepcopy(original)
            if mutation=='duplicate_recipe': self.study['recipe_candidates'][1]=self.study['recipe_candidates'][0]
            if mutation=='unknown_recipe_field': self.study['recipe_candidates'][0]['seed']=99
            if mutation=='cpu_batch': self.study['evaluation_profiles'][0]['batch_size']=2
            if mutation=='duplicate_profile': self.study['evaluation_profiles'][1]['profile_id']='CPU-B1'
            if mutation=='bool_lr': self.study['recipe_candidates'][0]['learning_rate']=True
            self.save_study()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): b.prepare(self.study_file)
        self.assertFalse((self.root/'studies').exists())

    def test_t01_duplicate_json_keys_and_invalid_template_refuse(self):
        self.study_file.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.compile()
        self.save_study(); self.config['pruning']=dict(method='channel',sparsity=.5,finetune_epochs=1); self.save()
        with self.assertRaisesRegex(ValueError,'unpruned'): self.compile()

    def test_t01_graph_duplicate_unknown_cycle_and_parent_mismatch_refuse(self):
        plan=self.compile()[0]
        for change in ('duplicate','unknown','cycle','phase','seed','topology'):
            value=copy.deepcopy(plan)
            child=next(j for j in value['fit_jobs'] if j['kind']=='narrow_scratch')
            if change=='duplicate': value['fit_jobs'].append(copy.deepcopy(child))
            if change=='unknown': child['dependencies'].append('missing')
            if change=='cycle': child['dependencies'].append(child['job_id'])
            if change=='phase': child['phase']='final'
            if change=='seed': child['seed']+=10
            if change=='topology': child['planned_widths'][0]+=1
            with self.subTest(change=change),self.assertRaises(ValueError): b.validate_graph(value)

    def test_t02_deterministic_full_matrix_dependencies_and_draft_modes(self):
        plan,configs,_,_=self.compile(); again=self.compile()
        self.assertEqual(plan,again[0]); self.assertEqual(configs,again[1])
        self.assertEqual(plan['totals'],dict(required_main=22,optional_recipes=3,pilot=6,maximum_fit_jobs=31))
        self.assertEqual(len(plan['selections']),3)
        self.assertEqual(len({j['job_id'].casefold() for j in plan['fit_jobs']}),31)
        self.assertTrue(all(len(j['job_id'])<=64 and not j['runnable'] for j in plan['fit_jobs']))
        for j in plan['fit_jobs']:
            config=configs[j['job_id']]['config']
            self.assertEqual(config['data']['manifest'],str(self.root/'studies'/self.study['study_id']/'manifest.json'))
            self.assertEqual(config['schema_version'],2)
            if j['kind'] in ('l1_channel','random_channel','l1_mask'):
                self.assertEqual(j['fit_epochs'],5); self.assertEqual(config['training']['epochs'],20)
            if j['kind']=='narrow_scratch':
                self.assertEqual(j['fit_epochs'],25); self.assertIsNone(config['baseline']['topology_checkpoint'])
                self.assertIn('baseline.topology_checkpoint',j['requires_resolution'])
        random=next(j for j in plan['fit_jobs'] if j['kind']=='random_channel')
        l1=next(j for j in plan['fit_jobs'] if j['kind']=='l1_channel' and j['ratio']==random['ratio'] and j['seed']==random['seed'])
        self.assertEqual(random['planned_widths'],l1['planned_widths'])
        with self.assertRaisesRegex(ValueError,'experiment fields'):
            e.load_config(self._save_draft(next(iter(configs.values()))))

    def _save_draft(self,value):
        path=self.root/'draft.json'; e.write_json(path,value); return path

    def test_t02_single_recipe_and_custom_grid_have_exact_visible_counts(self):
        self.study['recipe_candidates']=self.study['recipe_candidates'][:1]
        self.study['ratios']=[.25,.5]; self.save_study()
        plan=self.compile()[0]
        self.assertEqual(plan['totals'],dict(required_main=18,optional_recipes=0,pilot=6,maximum_fit_jobs=24))
        self.assertEqual(sum(j['kind']=='parent' and j['scope']=='main' for j in plan['fit_jobs']),4)

    def test_t02_long_ids_and_recipe_identity_are_stable(self):
        self.study['study_id']='a'*64; self.save_study(); first=self.compile()[0]
        self.assertTrue(all(len(j['job_id'])<=64 for j in first['fit_jobs']))
        self.study['recipe_candidates'][0]['learning_rate']=.0003; self.save_study(); changed=self.compile()[0]
        before=[j['parent_identity_key'] for j in first['fit_jobs'] if j['kind']=='parent' and j['recipe_index']==0]
        after=[j['parent_identity_key'] for j in changed['fit_jobs'] if j['kind']=='parent' and j['recipe_index']==0]
        self.assertNotEqual(before,after)

    def test_t03_prepare_no_test_pixels_loaders_models_or_cuda_query(self):
        import torch
        for row in self.rows:
            if row['split']=='test': (self.images/row['path']).unlink()
        before=(self.root/'manifest.json').read_bytes()
        with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('loader constructed')), \
             patch('breakhis.checkpoints.create_model',side_effect=AssertionError('model constructed')), \
             patch('breakhis.training.train',side_effect=AssertionError('fitting')), \
             patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA query')):
            output=b.prepare(self.study_file)
        self.assertEqual(before,(self.root/'manifest.json').read_bytes())
        self.assertEqual(before,(output/'manifest.json').read_bytes())
        status=e.read_json(output/'status.json'); self.assertEqual(status['status'],'prepared')
        self.assertFalse(status['fitting_started']); self.assertFalse(list(self.root.rglob('*.pth')))
        self.assertEqual(len(list((output/'configs').glob('*.json'))),31)
        for name,checksum in status['artifacts'].items(): self.assertEqual(e.digest(output/name),checksum)
        self.assertEqual(e.read_json(output/'cost_estimate.json')['estimated_seconds'],None)

    def test_t03_seed_change_does_not_resplit_and_identity_tracks_actual_manifest(self):
        first=self.compile()[0]; before=(self.root/'manifest.json').read_bytes()
        self.study['primary_seed']=500; self.save_study(); second=self.compile()[0]
        self.assertEqual(before,(self.root/'manifest.json').read_bytes())
        self.assertEqual(first['identity'],second['identity'])
        (self.root/'manifest.json').write_bytes(before+b' '); changed=self.compile()[0]
        self.assertNotEqual(first['identity']['manifest_sha256'],changed['identity']['manifest_sha256'])
        self.assertNotEqual(b.sha(first),b.sha(changed))
        self.assertEqual(first['counts']['all']['train']['images'],8)
        self.assertEqual(first['counts']['40X']['train']['images'],2)

    def test_t03_hidden_magnification_leakage_and_missing_fitting_path_fail(self):
        self.rows[-1]['split']='train'; self.save()
        with self.assertRaises(ValueError): self.compile()
        self.rows[-1]['split']='test'; self.save()
        row=next(r for r in self.rows if r['split']=='validation' and r['magnification']=='400X')
        (self.images/row['path']).unlink()
        with self.assertRaisesRegex(ValueError,'validation'): self.compile()

    def test_t04_capacity_unknown_or_short_does_not_prevent_planning(self):
        plan=self.compile()[0]
        with self.assertRaisesRegex(ValueError,'Missing cost'): b.require_capacity(plan)
        self.assertEqual(b.estimate_cost(plan)['required_fit_jobs'],31)
        self.study.update(max_fit_jobs=1,available_wall_hours=.0001); self.save_study()
        output=b.prepare(self.study_file); cost=e.read_json(output/'cost_estimate.json')
        self.assertEqual(len(e.read_json(output/'jobs.json')['fit_jobs']),31)
        self.assertFalse(cost['ready_for_main_capacity'])

    def test_t04_measured_input_estimates_shared_counts_and_retry_overheads(self):
        self.study.update(max_fit_jobs=32,available_wall_hours=100); self.save_study()
        plan=self.compile()[0]; estimates=self.costs(plan)
        expected=sum(j['fit_epochs']*2+3 for j in plan['fit_jobs'])
        result=b.require_capacity(plan,estimates)
        self.assertEqual(result['estimated_seconds'],expected); self.assertEqual(result['required_fit_jobs'],31)
        self.assertFalse(result['execution_supported'])
        retries=[dict(job_id=plan['fit_jobs'][0]['job_id'],extra_fit_jobs=1,extra_seconds=7,basis='Recorded restart from epoch zero')]
        result=b.require_capacity(plan,estimates,retries)
        self.assertEqual(result['estimated_seconds'],expected+7); self.assertEqual(result['required_fit_jobs'],32)
        retry=dict(retries[0],extra_fit_jobs=0,extra_seconds=4,basis='Replayed partial epoch')
        self.assertEqual(b.require_capacity(plan,estimates,[retry])['required_fit_jobs'],31)
        self.assertEqual(sum(j['kind']=='parent' and j['scope']=='main' for j in plan['fit_jobs']),7)

    def test_t04_wrong_plan_nonfinite_unknown_and_incomplete_costs_refuse(self):
        self.study.update(max_fit_jobs=31,available_wall_hours=100); self.save_study(); plan=self.compile()[0]
        original=self.costs(plan); key=next(iter(original['estimates']))
        for mutation in ('identity','nonfinite','overflow','unknown','empty','bool','missing'):
            costs=copy.deepcopy(original)
            if mutation=='identity': costs['plan_sha256']='0'*64
            if mutation=='overflow': costs['estimates'][key]['epoch_seconds']=1e308
            if mutation=='nonfinite': costs['estimates'][key]['epoch_seconds']=float('nan')
            if mutation=='unknown': costs['estimates']['unknown']=costs['estimates'][key]
            if mutation=='empty': costs['estimates'][key]['basis']=''
            if mutation=='bool': costs['estimates'][key]['fixed_seconds']=True
            if mutation=='missing': costs['estimates'].pop(key)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): b.require_capacity(plan,costs)

    def test_t04_job_and_wall_shortfalls_never_trim_matrix(self):
        self.study.update(max_fit_jobs=30,available_wall_hours=100); self.save_study(); plan=self.compile()[0]
        with self.assertRaisesRegex(ValueError,'Fit-job'): b.require_capacity(plan,self.costs(plan))
        self.study.update(max_fit_jobs=31,available_wall_hours=.0001); self.save_study(); plan=self.compile()[0]
        with self.assertRaisesRegex(ValueError,'Wall-time'): b.require_capacity(plan,self.costs(plan))
        self.assertEqual(len(plan['fit_jobs']),31)

    def test_atomic_failure_and_occupied_study_never_claim_success(self):
        original=storage.write_json
        def fail(path,*args,**kwargs):
            if Path(path).name=='jobs.json': raise OSError('injected publication failure')
            return original(path,*args,**kwargs)
        with patch.object(storage,'write_json',side_effect=fail),self.assertRaises(OSError): b.prepare(self.study_file)
        root=self.root/'studies'/self.study['study_id']
        self.assertEqual(e.read_json(root/'status.json')['status'],'failed')
        before={p.name:p.read_bytes() for p in root.iterdir()}
        with self.assertRaisesRegex(ValueError,'exists'): b.prepare(self.study_file)
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_prepare_input_alias_refused_before_publication(self):
        # An output root under the data root would mix generated records with images.
        with patch.object(b,'STUDIES',self.images/'studies'),self.assertRaisesRegex(ValueError,'protected'):
            b.prepare(self.study_file)
        self.assertFalse((self.images/'studies').exists())

    def test_semantic_recipe_identity_ignores_key_order(self):
        first=self.compile()[0]
        self.study['recipe_candidates'][0]=dict(reversed(list(self.study['recipe_candidates'][0].items())))
        self.save_study(); second=self.compile()[0]
        self.assertEqual([j['job_id'] for j in first['fit_jobs']],[j['job_id'] for j in second['fit_jobs']])
        self.study['recipe_candidates'][1]=dict(reversed(list(self.study['recipe_candidates'][0].items())))
        self.save_study()
        with self.assertRaisesRegex(ValueError,'Duplicate recipe'): self.compile()

    def test_long_study_names_do_not_share_fit_or_selector_ids(self):
        self.study['study_id']='a'*63+'x'; self.save_study(); first=self.compile()[0]
        self.study['study_id']='a'*63+'y'; self.save_study(); second=self.compile()[0]
        self.assertFalse({j['job_id'] for j in first['fit_jobs']} & {j['job_id'] for j in second['fit_jobs']})
        self.assertFalse({j['selection_id'] for j in first['selections']} & {j['selection_id'] for j in second['selections']})

    def test_occupied_future_fit_outputs_are_not_overwritten_or_reserved(self):
        plan=self.compile()[0]; job=plan['fit_jobs'][0]
        for location in (e.RUNS/job['job_id'],Path(job['checkpoint_directory'])):
            location.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'already exists'): b.prepare(self.study_file)
            location.rmdir()
        self.assertFalse((self.root/'studies').exists())

    def test_plan_identity_cannot_switch_to_test_or_another_policy(self):
        plan=self.compile()[0]
        for key,value in [('selection_split','test'),('phase','final'),('aggregation','image-only'),('main_magnification','40X')]:
            altered=copy.deepcopy(plan); altered['identity'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity differs'): b.validate_graph(altered)

    def test_public_cli_refuses_unprepared_execution_and_prepare_resume(self):
        for args in (['--stage','pilot'],['--stage','prepare','--resume'],['--stage','parents'],['--stage','report']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                b.main(['--study',str(self.study_file),*args])
        self.assertFalse((self.root/'studies').exists())
        result=subprocess.run([sys.executable,str(ROOT/'run_baselines.py'),'--help'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0); self.assertIn('--cost-estimates',result.stdout)


if __name__=='__main__': unittest.main()
