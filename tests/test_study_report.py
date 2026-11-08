"""Hand-computed descriptive report/accounting checks, without fitting or pixels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from breakhis import study_execution as x,experiments as e


class StudyReport(unittest.TestCase):
    def test_frontier_ties_tradeoffs_failures_and_shared_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=root/'runs';root.mkdir(exist_ok=True)
            (root/'status.json').write_text(json.dumps(dict(preparation_seconds=2)))
            plan=dict(fit_jobs=[],selections=[]);ledger=dict(jobs={},selections={})
            for key,sens,spec,latency in [('parent',.8,.8,10),('tie',.8,.8,10),('slow',.7,.7,15),('trade',.9,.6,8)]:
                job=dict(job_id=key,kind='parent' if key=='parent' else 'l1_channel',model='vgg',scope='main',seed=102,ratio=None if key=='parent' else .5,bindings={} if key=='parent' else {'baseline.parent_selection':'parent'},optional_recipe=False)
                plan['fit_jobs'].append(job);run=runs/key/'development';run.mkdir(parents=True)
                metrics=dict(image_metrics=dict(accuracy=75),group_metrics=dict(sensitivity=sens,specificity=spec,balanced_accuracy=(sens+spec)/2),benchmark=dict(latency_ms=dict(median=latency),runtime=dict(device='cpu')))
                (run/'results.json').write_text(json.dumps(dict(designated_metrics=metrics)))
                ledger['jobs'][key]=dict(status='completed',committed_epochs=1,actual_optimizer_steps=4,image_presentations=8,child_wall_seconds=10,resource_attempts=[],attempts=[dict(seconds=10)])
            plan['fit_jobs'].append(dict(job_id='failed',kind='random_channel',model='vgg',optional_recipe=False))
            ledger['jobs']['failed']=dict(status='failed',error={'message':'OOM'},attempts=[dict(seconds=3)])
            selection=dict(designated_role='dense',roles=dict(dense=dict(epoch=1)))
            with patch.object(e,'RUNS',runs),patch.object(x,'verified_job',return_value=({}, {}, {}, selection)):
                result=x.report(root,plan,ledger)
            self.assertEqual(set(result['descriptive_frontier']),{'parent','tie','trade'})
            self.assertEqual(result['study_actual_total_known_seconds'],45)
            self.assertFalse(result['complete_required']);self.assertEqual(result['diagnostics'][0]['status'],'failed')
            child=next(r for r in result['rows'] if r['job_id']=='tie')
            self.assertEqual(child['full_lifecycle_steps'],8);self.assertEqual(child['full_lifecycle_presentations'],16)
            self.assertTrue((root/'development/results.csv').exists())


if __name__=='__main__':unittest.main()
