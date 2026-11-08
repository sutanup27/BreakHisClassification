import importlib.util
from pathlib import Path
import unittest
from breakhis import baseline_plan as b,study_execution as x,experiments as e,experiment_io as io
import test_baseline_plan as fixtures
s=importlib.util.spec_from_file_location('capacity_revision',Path(__file__).resolve().parents[1]/'tools/revise_study_capacity.py');revision=importlib.util.module_from_spec(s);s.loader.exec_module(revision)


class CapacityRevision(unittest.TestCase):
    save=fixtures.BaselinePlanTests.save
    save_study=fixtures.BaselinePlanTests.save_study
    costs=fixtures.BaselinePlanTests.costs
    def setUp(self):
        fixtures.BaselinePlanTests.setUp(self);self.study.update(max_fit_jobs=100,available_wall_hours=8);self.save_study()
        self.anchor=b.prepare(self.study_file);self.plan=e.read_json(self.anchor/'jobs.json')
        io.write_json(self.anchor/'execution/ledger.json',dict(plan_sha256=b.sha(self.plan),jobs={},selections={}),root=self.anchor)
        io.write_json(self.anchor/'execution/measured_cost_inputs.json',self.costs(self.plan),root=self.anchor)
    def test_resource_revision_preserves_jobs_and_frozen_artifacts(self):
        old=[e.digest(self.anchor/j['config_file']) for j in self.plan['fit_jobs']]
        result=revision.revise(self.study_file,16,'Authorized additional eight-hour session')
        root,plan=x.frozen(self.study_file);self.assertEqual(plan['fit_jobs'],self.plan['fit_jobs']);self.assertEqual(result['new_hours'],16)
        self.assertEqual(old,[e.digest(self.anchor/j['config_file']) for j in plan['fit_jobs']])
    def test_main_started_or_smaller_budget_refuses(self):
        key=next(j['job_id'] for j in self.plan['fit_jobs'] if j['kind']=='parent')
        path=self.anchor/'execution/ledger.json';ledger=e.read_json(path);ledger['jobs'][key]=dict(status='failed');io.write_json(path,ledger,root=self.anchor)
        with self.assertRaisesRegex(ValueError,'before any main'):revision.revise(self.study_file,16,'extra capacity')
        with self.assertRaises(ValueError):revision.revise(self.study_file,4,'smaller')


if __name__=='__main__':unittest.main()
