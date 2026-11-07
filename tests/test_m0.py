"""M0.2 acceptance: M0-T01--T04, M0-T19--T20, synthetic isolated repos."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from breakhis.research import m0
from breakhis.research import records as r


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve() / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.put('.gitignore', 'artifacts/\n.venv*/\n')
        self.put('breakhis/example.py', 'original = 1\n')
        self.put('tests/sample.py', '# sample\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture baseline')
        self.run = r.create_run_directory(self.repo / 'artifacts/research/m0', 'fixture')

    def put(self, path, text):
        path = self.repo / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def git(self, *argv):
        return subprocess.run(['git', '-C', str(self.repo), *argv], check=True,
                              capture_output=True).stdout

    def inventory(self):
        return r.read_record(self.run, 'baseline/source_inventory.json')['items']

    def test_t01_clean_reconstructible_and_stable_identity(self):
        result = m0.capture_source(self.repo, self.run, 'fixture')
        self.assertFalse(result['dirty'])
        for entry in self.inventory():
            restored = self.run / 'baseline/source_snapshot' / entry['path']
            self.assertEqual(restored.read_bytes(), (self.repo / entry['path']).read_bytes())
            self.assertEqual(r.file_sha256(restored), entry['sha256'])
        second = r.create_run_directory(self.repo / 'artifacts/research/m0', 'second')
        again = m0.capture_source(self.repo, second, 'second')
        self.assertEqual(result['source_snapshot_sha256'], again['source_snapshot_sha256'])
        self.assertEqual(result['source_identity'], again['source_identity'])
        with self.assertRaises(FileExistsError):
            m0.capture_source(self.repo, self.run, 'fixture')

    def test_t02_staged_unstaged_untracked_deleted_and_binary(self):
        self.put('breakhis/example.py', 'staged = 2\n')
        self.git('add', 'breakhis/example.py')
        self.put('breakhis/example.py', 'unstaged = 3\n')
        self.put('configs/new.json', '{"new": true}')
        self.git('rm', 'tests/sample.py')
        # A source binary diff must be reconstructible without including weights.
        (self.repo / 'breakhis/example.py').write_bytes(b'\0changed source\xff')
        result = m0.capture_source(self.repo, self.run, 'fixture')
        self.assertTrue(result['dirty'])
        rows = {e['path']: e for e in self.inventory()}
        self.assertEqual(rows['configs/new.json']['tracking'], 'untracked')
        self.assertEqual(rows['tests/sample.py']['capture_status'], 'deleted')
        self.assertIn(b'GIT binary patch', (self.run / 'baseline/tracked.patch').read_bytes())
        self.assertIn(b'staged = 2', (self.run / 'baseline/staged.patch').read_bytes())
        self.assertEqual((self.run / 'baseline/source_snapshot/breakhis/example.py').read_bytes(), b'\0changed source\xff')

    def test_t03_excludes_data_weights_environments_generated_and_personal(self):
        for path in ('Dataset/image.jpg', 'breakhis/weights/model.pth', '.venv-local/private.py',
                     'artifacts/result.py', 'docs/index.html', 'private/clinical_identity.json',
                     'tests/data/identity.json', 'docs/result.png'):
            self.put(path, 'excluded payload')
        self.git('add', '-f', 'Dataset', 'breakhis/weights', 'docs', 'private', 'tests/data')
        result = m0.capture_source(self.repo, self.run, 'fixture')
        paths = {e['path'] for e in self.inventory()}
        self.assertEqual(paths, {'.gitignore', 'breakhis/example.py', 'tests/sample.py'})
        self.assertIn('capture_policy', result)
        self.assertNotIn(b'excluded payload', (self.run / 'baseline/tracked.patch').read_bytes())

    def test_t04_movement_retries_once_then_succeeds(self):
        original = m0._source_state
        calls = []
        def moving(*args):
            calls.append(True)
            if len(calls) == 2:
                self.put('breakhis/example.py', 'changed during capture\n')
            return original(*args)
        with patch.object(m0, '_source_state', side_effect=moving):
            result = m0.capture_source(self.repo, self.run, 'fixture')
        self.assertEqual(len(calls), 4)
        self.assertEqual([a['status'] for a in result['attempts']], ['failed', 'passed'])
        self.assertTrue((self.run / 'baseline/attempt-1').is_dir())

    def test_t04_repeated_movement_fails_and_retains_evidence(self):
        original = m0._source_state
        calls = []
        def moving(*args):
            calls.append(True)
            self.put('breakhis/example.py', f'value = {len(calls)}\n')
            return original(*args)
        with patch.object(m0, '_source_state', side_effect=moving):
            with self.assertRaises(r.IntegrityError):
                m0.capture_source(self.repo, self.run, 'fixture')
        result = r.read_record(self.run, 'baseline/git.json')['data']
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['attempts']), 2)
        self.assertFalse((self.run / 'baseline/source_snapshot').exists())

    def test_detached_head_and_lfs_not_hydrated(self):
        self.git('checkout', '--detach', '-q')
        self.put('configs/pointer.json', 'version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 123\n')
        result = m0.capture_source(self.repo, self.run, 'fixture')
        self.assertEqual(result['ref'], 'HEAD')
        self.assertEqual(result['unresolved_lfs'], ['configs/pointer.json'])

    def test_plan_only_allowlisted_safe_commands(self):
        plan = m0.baseline_plan(sys.executable, ['cpu', 'cuda'])
        self.assertTrue(all(argv[0] == sys.executable for _, argv, _, _ in plan))
        self.assertEqual(len([p for p in plan if p[0].startswith('help-')]), 6)
        for invalid in ([], ['cuda'], ['cpu', 'cpu'], ['cpu', 'other']):
            with self.assertRaises(r.RecordError):
                m0.baseline_plan(sys.executable, invalid)
        with self.assertRaises(r.RecordError):
            m0.baseline_plan(sys.executable, ['cpu'], environment_seconds=True)

    def test_t20_command_failure_and_timeout_retain_hashed_logs(self):
        fail = m0._execute(self.run, 'failed', [sys.executable, '-c', 'import sys; print("evidence", flush=True); sys.exit(7)'], self.repo, 'cpu', 10, os.environ.copy())
        self.assertEqual((fail['status'], fail['return_code']), ('failed', 7))
        self.assertIn('evidence', r.verify_artifact(self.run, fail['stdout']).read_text())
        timeout = m0._execute(self.run, 'timeout', [sys.executable, '-c', 'import time; print("partial", flush=True); time.sleep(30)'], self.repo, 'cpu', 1, os.environ.copy())
        self.assertEqual(timeout['status'], 'timed_out')
        self.assertIsNotNone(timeout['return_code'])
        self.assertIn('partial', r.verify_artifact(self.run, timeout['stdout']).read_text())

    def test_missing_executable_is_failed_evidence(self):
        result = m0._execute(self.run, 'absent', [str(self.repo/'missing.exe')], self.repo, 'cpu', 2, os.environ.copy())
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['return_code'])
        self.assertTrue(r.verify_artifact(self.run, result['stderr']).stat().st_size)

    def fixture_plan(self, gpu=False, failure=False, malformed=False, skipped=False, timeout=False):
        code = 'import json; print(json.dumps({"python":"fixture"}))'
        plan = [('metadata', [sys.executable, '-c', code], 'cpu', 10),
                ('environment-cpu', [sys.executable, '-c', 'print("bad-json")' if malformed else code], 'cpu', 10),
                ('regressions', [sys.executable, '-c', 'import sys; print("OK (skipped=1)", file=sys.stderr)' if skipped else 'raise SystemExit(1)' if failure else 'print("passed fixture")'], 'cpu', 10)]
        if gpu:
            plan.append(('environment-cuda', [sys.executable, '-c', 'raise RuntimeError("CUDA unavailable fixture")'], 'cuda', 10))
        if timeout:
            plan[-1] = ('regressions', [sys.executable, '-c', 'import time; print("partial baseline", flush=True); time.sleep(30)'], 'cpu', 1)
        return plan

    def check_fixture(self, **kwargs):
        devices = ['cpu', 'cuda'] if kwargs.get('gpu') else ['cpu']
        with patch.object(m0, 'baseline_plan', return_value=self.fixture_plan(**kwargs)):
            return m0.capture_environment_checks(self.repo, self.run, 'fixture', devices=devices)

    def test_t19_cpu_pass_gpu_failure_separate_restrictions(self):
        result = self.check_fixture(gpu=True)
        self.assertTrue(result['cpu_engineering_ready'])
        self.assertFalse(result['gpu_execution_ready'])
        self.assertEqual(r.read_record(self.run, 'baseline/environment.cuda.json')['data']['status'], 'failed')
        with self.assertRaises(FileExistsError):
            self.check_fixture(gpu=True)

    def test_t19_gpu_not_requested_not_claimed_available(self):
        result = self.check_fixture()
        self.assertTrue(result['cpu_engineering_ready'])
        self.assertFalse(result['gpu_execution_ready'])
        self.assertEqual(r.read_record(self.run, 'baseline/environment.cuda.json')['data']['status'], 'not_requested')

    def test_t20_failed_regression_blocks_engineering(self):
        self.assertFalse(self.check_fixture(failure=True)['cpu_engineering_ready'])
        checks = r.read_record(self.run, 'baseline/checks.json')['items']
        self.assertEqual(next(c for c in checks if c['check_id']=='regressions')['status'], 'failed')

    def test_t20_timed_out_regression_blocks_engineering(self):
        self.assertFalse(self.check_fixture(timeout=True)['cpu_engineering_ready'])
        checks = r.read_record(self.run, 'baseline/checks.json')['items']
        result = next(c for c in checks if c['check_id']=='regressions')
        self.assertEqual(result['status'], 'timed_out')
        self.assertIn('partial baseline', r.verify_artifact(self.run, result['stdout']).read_text())

    def test_malformed_successful_probe_is_failed(self):
        self.assertFalse(self.check_fixture(malformed=True)['cpu_engineering_ready'])

    def test_skipped_regressions_do_not_satisfy_acceptance(self):
        self.assertFalse(self.check_fixture(skipped=True)['cpu_engineering_ready'])

    def test_source_change_during_checks_blocks_baseline(self):
        def changed(*args, **kwargs):
            self.put('breakhis/example.py', 'changed after capture\n')
            return dict(cpu_engineering_ready=True, gpu_execution_ready=True, restrictions=[], checks=[])
        with patch.object(m0, 'capture_environment_checks', side_effect=changed):
            result = m0.capture_baseline(self.repo, self.run, 'fixture')
        self.assertFalse(result['cpu_engineering_ready'])
        self.assertFalse(result['gpu_execution_ready'])
        self.assertEqual(r.read_record(self.run, 'baseline/consistency.json')['data']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
