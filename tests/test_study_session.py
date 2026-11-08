"""Session worker honors explicit rest requests and expired continuous budgets."""
from datetime import datetime,timezone,timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('session_worker',Path(__file__).resolve().parents[1]/'tools/run_study_session.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class WorkerSessions(unittest.TestCase):
    def test_pause_file_prevents_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'execution').mkdir();(root/'execution/PAUSE').touch()
            plan=dict(study=dict(study_id='fixture'),fit_jobs=[])
            with patch.object(worker.execution,'frozen',return_value=(root,plan)),patch.object(worker.execution,'execute',side_effect=AssertionError('fitting while paused')),patch.object(sys,'argv',['worker','--study','fixture.json']):worker.main()
            status=json.loads((root/'execution/live_status.json').read_text());self.assertEqual(status['status'],'paused_at_job_boundary')

    def test_waiting_for_pilot_counts_towards_session_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'execution').mkdir()
            (root/'execution/ledger.json').write_text(json.dumps(dict(jobs={},sessions=[dict(status='running',started_at=(datetime.now(timezone.utc)-timedelta(hours=9)).isoformat())])))
            plan=dict(study=dict(study_id='fixture'),fit_jobs=[dict(job_id='pilot',kind='pilot')])
            with patch.object(worker.execution,'frozen',return_value=(root,plan)),patch.object(worker.execution,'execute',side_effect=AssertionError('over-budget fitting')),patch.object(worker.time,'sleep',side_effect=AssertionError('over-budget wait')),patch.object(sys,'argv',['worker','--study','fixture.json','--wait-for-pilot']):worker.main()
            self.assertEqual(json.loads((root/'execution/live_status.json').read_text())['status'],'paused_session_limit')


if __name__=='__main__':unittest.main()
