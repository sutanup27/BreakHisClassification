"""T20: explicit architecture expansion without mutating the original study."""
import copy
import unittest
from breakhis import baseline_plan as b,experiments as e
import test_baseline_plan as fixtures


class ResNetStudy(unittest.TestCase):
    save=fixtures.BaselinePlanTests.save
    save_study=fixtures.BaselinePlanTests.save_study
    compile=fixtures.BaselinePlanTests.compile
    costs=fixtures.BaselinePlanTests.costs
    def setUp(self):fixtures.BaselinePlanTests.setUp(self)

    def test_t20_version2_counts_shapes_bindings_costs_and_no_test_pixels(self):
        original=self.compile()[0];self.assertEqual(original['totals']['maximum_fit_jobs'],31)
        self.study.update(schema_version=2,study_id='resnet-extended',pruning_models=['vgg','resnet18']);self.save_study()
        for row in self.rows:
            if row['split']=='test':(self.images/row['path']).unlink()
        plan,configs,*_=self.compile()
        self.assertEqual(plan['totals'],dict(required_main=41,optional_recipes=3,pilot=6,maximum_fit_jobs=50))
        resnet=[j for j in plan['fit_jobs'] if j['model']=='resnet18' and j['kind']=='l1_channel']
        self.assertEqual(len(resnet),4)
        for job in resnet:
            self.assertEqual(len(job['planned_widths']['stage_widths']),4)
            self.assertEqual(len(job['planned_widths']['hidden_widths']),4)
            self.assertFalse(job['runnable'])
            self.assertIn('resnet18',job['cost_key'])
        cost=b.estimate_cost(plan,self.costs(plan))
        self.assertEqual(cost['required_fit_jobs'],50);self.assertEqual(len(cost['job_seconds']),50)
        self.assertEqual(cost['estimated_seconds'],sum(j['fit_epochs']*2+3 for j in plan['fit_jobs']))
        self.assertFalse(plan['execution_supported'])
        bad=copy.deepcopy(plan);job=next(j for j in bad['fit_jobs'] if j['model']=='resnet18' and j['kind']=='narrow_scratch');job['model']='vgg'
        with self.assertRaises(ValueError):b.validate_graph(bad)

    def test_t20_invalid_optin_refuses(self):
        for models in ([],['resnet50'],['vgg','vgg'],[True],[[]]):
            self.study.update(schema_version=2,pruning_models=models);self.save_study()
            with self.subTest(models=models),self.assertRaises(ValueError):self.compile()


if __name__=='__main__':unittest.main()
