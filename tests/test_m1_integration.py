"""M1-T22--T25: public fresh-process workflow and non-mutating legacy refusal."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]


class MilestoneIntegrationTests(unittest.TestCase):
    def test_t22_t23_public_dense_mask_channel_phases_and_real_retries(self):
        from tools.smoke_experiments import run_smoke
        from breakhis.experiments import RUNS
        from breakhis.paths import checkpoint_directory
        prefix='test-m1-'+uuid.uuid4().hex[:8]
        names=[prefix+'-'+kind for kind in ('dense','mask','channel')]
        checkpoints=checkpoint_directory('cancer_binary')
        def cleanup():
            for name in names:
                for path,root in [(RUNS/name,RUNS),(checkpoints/'40X/vgg'/name,checkpoints)]:
                    resolved=path.resolve()
                    assert resolved!=root.resolve() and resolved.is_relative_to(root.resolve()) and resolved.name==name
                    if resolved.exists(): shutil.rmtree(resolved)
        self.addCleanup(cleanup)
        with tempfile.TemporaryDirectory() as folder:
            import os
            work=Path(os.environ['BREAKHIS_PHASE_SMOKE_EVIDENCE']) if os.environ.get('BREAKHIS_PHASE_SMOKE_EVIDENCE') else Path(folder)/'smoke'
            with contextlib.redirect_stdout(io.StringIO()): summary=run_smoke(work,run_prefix=prefix)
            self.assertEqual(summary['status'],'passed')
            for flag in ('dense_weights_reproduced','test_pixels_unavailable_during_development',
                'fitting_pixels_unavailable_during_final','all_pixels_unavailable_during_final_comparison',
                'actual_training_failure_resumed','actual_final_failure_resumed','public_cli_only'):
                self.assertTrue(summary[flag],flag)
            self.assertEqual(summary['pruning_invariants'],dict(mask=True,channel=True))
            commands=json.loads(Path(summary['commands']).read_text())
            self.assertTrue(all(c['status']=='passed' for c in commands))
            self.assertEqual(sum(c['expected_failure'] for c in commands),3)
            for phase,path in summary['comparisons'].items():
                report=json.loads(Path(path).read_text())
                self.assertEqual(len(report['rows']),7)
                self.assertEqual({r['split'] for r in report['rows']},{'validation' if phase=='development' else 'test'})

    def test_t24_legacy_run_report_readable_but_not_phase_promoted_or_rewritten(self):
        from breakhis import experiments as e, experiment_setup as setup
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'RUNS',Path(folder)/'runs'):
            legacy=e.RUNS/'legacy'; legacy.mkdir(parents=True)
            e.write_json(legacy/'run.json',dict(schema_version=1,status='completed',stages={}))
            e.write_json(legacy/'results.json',dict(schema_version=1,final_metrics={'accuracy':75}))
            e.write_json(legacy/'dense.metrics.json',dict(schema_version=1,split='test',accuracy=75))
            before={p.name:p.read_bytes() for p in legacy.iterdir()}
            self.assertEqual(e.read_json(legacy/'results.json')['final_metrics']['accuracy'],75)
            for phase in ('development','final'):
                with self.subTest(phase=phase),self.assertRaisesRegex(ValueError,'Schema-1'):
                    setup.prepare_run('unused.json' if phase=='development' else None,
                        run_id='legacy',phase=phase,resume=True)
            other=e.RUNS/'other'; other.mkdir(); e.write_json(other/'run.json',dict(schema_version=1,status='completed'))
            with self.assertRaisesRegex(ValueError,'Schema-1'):
                e.compare_runs([legacy,other],Path(folder)/'comparison.json')
            self.assertEqual(before,{p.name:p.read_bytes() for p in legacy.iterdir()})
            self.assertFalse((Path(folder)/'comparison.json').exists())

    def test_t24_version_one_weights_load_for_inference_but_cannot_exact_resume(self):
        import torch
        from breakhis import checkpoints as c, cli
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'legacy.pth'; model=torch.nn.Linear(2,2)
            torch.save(dict(format_version=1,architecture='vgg',classes=['benign','malignant'],
                state_dict=model.state_dict(),metadata={},epoch=1),path)
            original=path.read_bytes()
            with patch.object(c,'create_model',side_effect=lambda *a,**k:torch.nn.Linear(2,2)):
                loaded,payload=c.load_checkpoint(path,'cpu')
            self.assertTrue(torch.equal(model(torch.ones(1,2)),loaded(torch.ones(1,2))))
            with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('data accessed')), \
                 contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                cli.main(['train','--dataset','breakhis','--resume',str(path)])
            self.assertEqual(original,path.read_bytes())
            self.assertIsNone(payload.get('training_state'))

    def test_t25_public_help_exposes_supported_workflow_flags(self):
        for args,flags in [(['run_experiment.py','--help'],['--phase','--resume','--prepare-only']),
                (['compare_experiments.py','--help'],['--phase','--vary']),
                (['-m','breakhis','evaluate','--help'],['--split','--experiment-config','--predictions-output']),
                (['-m','breakhis','train','--help'],['--resume','--selection-metric']),
                (['-m','breakhis','prune','--help'],['--method','--checkpoint']),
                (['-m','breakhis','finetune','--help'],['--resume','--checkpoint'])]:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            for flag in flags: self.assertIn(flag,result.stdout)


if __name__=='__main__': unittest.main()
