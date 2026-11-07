"""M0.2 source preservation and current baseline evidence (no scientific fits).

Import-safe, standard-library orchestration. Full M0 runner and G0 are M0.6.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import records as r


SOURCE_POLICY = {
    'version': 1,
    'directories': ['breakhis', 'tools', 'tests', 'configs', 'docs', '.github',
                    'DataProcess', 'Models', 'Testing', 'Utills', 'experiments'],
    'root_extensions': ['.py', '.toml', '.txt', '.md', '.yml', '.yaml', '.json'],
    'root_names': ['.gitignore', '.gitattributes', '.gitmodules'],
    'excluded_components': ['.git', 'artifacts', 'dataset', 'datasets', 'data',
                            'checkpoint', 'checkpoints', 'weights', 'runs', 'output',
                            'outputs', 'build', 'dist', 'venv', 'env', '__pycache__',
                            '.pytest_cache', '.ipynb_checkpoints', 'site-packages'],
    'excluded_extensions': ['.pt', '.pth', '.ckpt', '.onnx', '.bin', '.png', '.jpg',
                            '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.npy', '.npz',
                            '.zip', '.7z', '.pdf', '.pyc', '.pyo', '.log'],
    'excluded_docs': ['docs/index.html'],
    'notes': 'Allowlisted source/config/docs only; no raw identity documents. '
             'LFS pointers and submodules are recorded, not hydrated. Ignored files excluded.',
}


def _now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def audit_data(*args, **kwargs):
    """Lazy M0.3 adapter; see data_readiness.audit_data for its contract."""
    from .data_readiness import audit_data as audit
    return audit(*args, **kwargs)


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60).stdout


def _included(path):
    parts = path.split('/')
    if any(p.lower() in SOURCE_POLICY['excluded_components'] or p.lower().startswith('.venv')
           or p.lower().endswith('.egg-info') for p in parts):
        return False
    if path in SOURCE_POLICY['excluded_docs'] or Path(path).suffix.lower() in SOURCE_POLICY['excluded_extensions']:
        return False
    if len(parts) == 1:
        return Path(path).suffix.lower() in SOURCE_POLICY['root_extensions'] or path in SOURCE_POLICY['root_names']
    return parts[0] in SOURCE_POLICY['directories']


def _source_state(repo, output):
    """NUL-delimited Git inventories preserve spaces and Unicode, including renames."""
    head = _git(repo, 'rev-parse', 'HEAD').decode().strip()
    ref = _git(repo, 'symbolic-ref', '-q', '--short', 'HEAD') if _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip() != b'HEAD' else b'HEAD'
    index = _git(repo, 'ls-files', '--stage', '-z')
    tracked = set(_git(repo, 'ls-files', '-z').decode('utf-8').rstrip('\0').split('\0')) - {''}
    tracked |= set(_git(repo, 'ls-tree', '-r', '--name-only', '-z', 'HEAD').decode('utf-8').rstrip('\0').split('\0')) - {''}
    untracked = set(_git(repo, 'ls-files', '--others', '--exclude-standard', '-z').decode('utf-8').rstrip('\0').split('\0')) - {''}
    paths = sorted(p for p in tracked | untracked if _included(p))
    entries, pointers = [], []
    submodules = [line.split('\t', 1)[1] for line in index.decode('utf-8').split('\0')
                  if line.startswith('160000 ') and '\t' in line]
    for relative in paths:
        r.portable_relative_path(relative)
        if relative in submodules:
            continue
        path = r.safe_child(repo, relative)
        if not path.exists():
            entries.append(dict(path=relative, tracking='tracked', bytes=None, sha256=None, capture_status='deleted'))
            continue
        r.capture_relative_path(path, repo_root=repo, output_root=output)
        with path.open('rb') as stream:
            prefix = stream.read(130)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointers.append(relative)
        entries.append(dict(path=relative, tracking='tracked' if relative in tracked else 'untracked',
                            bytes=path.stat().st_size, sha256=r.file_sha256(path), capture_status='captured'))
    # Restrict patches to included paths: excluded datasets/weights must never leak into evidence.
    patch = _git(repo, 'diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--', *paths) if paths else b''
    staged = _git(repo, 'diff', '--binary', '--no-ext-diff', '--no-textconv', '--cached', '--', *paths) if paths else b''
    status = _git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
    return dict(head=head, ref=ref.decode().strip(), status_porcelain_z=status.decode('utf-8'),
                dirty=bool(status), index_sha256=r.semantic_sha256(index.hex(), domain='m0.git_index'),
                entries=entries, patch=patch, staged_patch=staged,
                unresolved_lfs=pointers, unresolved_submodules=submodules)


def capture_source(repo_root, run_dir, run_id):
    """Capture worktree bytes, index diff and HEAD diff; retry once on movement.

    Success reconstructs the included *worktree* from source_snapshot alone.
    Deletions are inventory tombstones. staged.patch preserves index distinctions.
    Failed attempts remain explicit partial evidence. No existing baseline overwritten.
    """
    repo, run = Path(repo_root).resolve(), Path(run_dir).resolve()
    if Path(_git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        raise r.RecordError('repo_root must be the Git top-level directory')
    if not run.is_dir() or not run.is_relative_to(repo / 'artifacts/research/m0'):
        raise r.RecordError('Create a fresh run under repository artifacts/research/m0 first')
    baseline = r.safe_child(run, 'baseline')
    baseline.mkdir()  # Refuse recapture, including a failed prior attempt.
    attempts = []
    for attempt in (1, 2):
        started = _now()
        try:
            before = _source_state(repo, run)
            target = r.safe_child(run, f'baseline/attempt-{attempt}')
            target.mkdir()
            for entry in before['entries']:
                if entry['capture_status'] != 'captured':
                    continue
                source = r.safe_child(repo, entry['path'], must_exist=True)
                destination = r.safe_child(target, entry['path'])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if r.file_sha256(destination) != entry['sha256']:
                    raise r.IntegrityError('Source moved while copying')
            after = _source_state(repo, run)
            if before != after:
                raise r.IntegrityError('Source Git state or bytes changed during capture')
        except (OSError, r.RecordError, subprocess.SubprocessError) as error:
            attempts.append(dict(attempt=attempt, started_at_utc=started, ended_at_utc=_now(), status='failed', error=str(error)))
            if attempt == 1:
                continue
            r.write_record(run, 'baseline/git.json', r.new_record('git', run_id,
                           dict(status='failed', attempts=attempts, reason='No coherent source snapshot')))
            raise r.IntegrityError('Source capture failed twice; retained partial baseline') from error
        attempts.append(dict(attempt=attempt, started_at_utc=started, ended_at_utc=_now(), status='passed'))
        target.rename(r.safe_child(run, 'baseline/source_snapshot'))
        (baseline / 'tracked.patch').write_bytes(before.pop('patch'))
        (baseline / 'staged.patch').write_bytes(before.pop('staged_patch'))
        entries = before.pop('entries')
        present = [e for e in entries if e['capture_status'] == 'captured']
        identity = r.snapshot_sha256(present, SOURCE_POLICY)
        before.update(status='complete', attempts=attempts, capture_policy=SOURCE_POLICY,
                      source_snapshot_sha256=identity,
                      source_identity=r.semantic_sha256({'git': before, 'snapshot': identity}, domain='m0.source_identity'))
        r.write_record(run, 'baseline/source_inventory.json', r.new_record('source_inventory', run_id, entries))
        r.write_record(run, 'baseline/git.json', r.new_record('git', run_id, before))
        return before


def _execute(run, check_id, argv, cwd, device, timeout, env):
    """Internal command executor; only baseline_plan supplies production argv."""
    r.validate_id(check_id)
    if type(timeout) is not int or timeout < 1:
        raise r.RecordError('Check timeout must be positive integer seconds')
    logs = r.safe_child(run, 'logs')
    logs.mkdir(exist_ok=True)
    out_rel, err_rel = f'logs/{check_id}.stdout.log', f'logs/{check_id}.stderr.log'
    started, tick = _now(), time.monotonic()
    code, status, reason = None, 'failed', None
    input_hashes = {}
    for relative in ('baseline/git.json', 'baseline/source_inventory.json'):
        path = r.safe_child(run, relative)
        if path.exists():
            input_hashes[relative] = r.file_sha256(path)
    with r.safe_child(run, out_rel).open('xb') as out, r.safe_child(run, err_rel).open('xb') as err:
        try:
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=err, env=env, **options)
            try:
                code = process.wait(timeout=timeout)
                status = 'passed' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status, reason = 'timed_out', f'Exceeded {timeout} seconds'
                if os.name == 'nt':
                    try:
                        killed = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=err, stderr=err, timeout=15)
                        if killed.returncode:
                            reason += '; process-tree termination failed'
                            process.kill()
                    except (OSError, subprocess.TimeoutExpired):
                        reason += '; process-tree termination unavailable'
                        process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=15)
            except KeyboardInterrupt:
                # Keep evidence and avoid leaving a regression worker behind on Ctrl+C.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=err, stderr=err, timeout=15)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=15)
                status, reason = 'interrupted', 'Interrupted by operator; remaining checks not run'
        except OSError as error:
            reason = str(error)
            err.write(reason.encode('utf-8'))
    return dict(check_id=check_id, operation_id=check_id, argv=list(argv), cwd=str(cwd), device=device,
                started_at_utc=started, ended_at_utc=_now(), elapsed_seconds=time.monotonic()-tick,
                timeout_seconds=timeout, return_code=code, status=status, reason=reason,
                input_hashes=input_hashes,
                stdout=r.artifact_entry(run, out_rel, record_type='log', operation_id=check_id),
                stderr=r.artifact_entry(run, err_rel, record_type='log', operation_id=check_id))


def baseline_plan(python, devices, *, environment_seconds=120, regression_seconds=600):
    """Fixed commands: import-only, help-only, synthetic environment and regressions."""
    if not isinstance(devices, list) or 'cpu' not in devices or len(set(devices)) != len(devices) or set(devices)-{'cpu', 'cuda'}:
        raise r.RecordError('Select CPU and optionally CUDA')
    if any(type(t) is not int or t < 1 for t in (environment_seconds, regression_seconds)):
        raise r.RecordError('Timeouts must be positive integer seconds')
    metadata_code = ('import sys, platform, os, json, importlib.metadata as m; '
                     'print(json.dumps(dict(python=sys.version, executable=sys.executable, '
                     'prefix=sys.prefix, base_prefix=sys.base_prefix, virtual_environment=sys.prefix!=sys.base_prefix, '
                     'platform=platform.platform(), machine=platform.machine(), processor=platform.processor() or None, '
                     'logical_cpu_count=os.cpu_count(), dependencies=sorted([(d.metadata["Name"], d.version) '
                     'for d in m.distributions()]))))')
    plan = [('imports', [python, '-c', 'import breakhis.cli, breakhis.training, breakhis.checkpoints; import Train, Main, Test, VGG'], 'cpu', environment_seconds)]
    plan.append(('metadata', [python, '-c', metadata_code], 'cpu', environment_seconds))
    for command in ('prepare-data', 'train', 'evaluate', 'prune', 'finetune', 'compare'):
        plan.append((f'help-{command}', [python, '-m', 'breakhis', command, '--help'], 'cpu', environment_seconds))
    plan.extend([('pip-check', [python, '-m', 'pip', 'check'], 'cpu', environment_seconds),
                 ('pip-freeze', [python, '-m', 'pip', 'freeze', '--all'], 'cpu', environment_seconds),
                 ('pip-version', [python, '-m', 'pip', '--version'], 'cpu', environment_seconds)])
    for device in devices:
        plan.append((f'environment-{device}', [python, 'tools/check_environment.py', '--device', device], device, environment_seconds))
    plan.append(('regressions', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'cpu', regression_seconds))
    return plan


def capture_environment_checks(repo_root, run_dir, run_id, *, python=None, devices=None,
                               environment_seconds=120, regression_seconds=600):
    """Run actual checks, retain failures, and return package-level eligibility only.

    cwd stays the repository for compatibility. Environment reports are read from stdout.
    No dataset loading, training or test-set inference is requested.
    """
    repo, run = Path(repo_root).resolve(), Path(run_dir).resolve()
    if not run.is_dir() or not run.is_relative_to(repo / 'artifacts/research/m0'):
        raise r.RecordError('Expected existing confined M0 run')
    if r.safe_child(run, 'baseline/checks.json').exists():
        raise FileExistsError('Checks already captured; create a new run')
    devices = ['cpu'] if devices is None else devices
    python = str(Path(python or sys.executable).resolve())
    plan = baseline_plan(python, devices, environment_seconds=environment_seconds, regression_seconds=regression_seconds)
    env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    results = []
    for check_id, argv, device, timeout in plan:
        check_env = dict(env)
        if device == 'cpu':
            check_env['CUDA_VISIBLE_DEVICES'] = ''
        result = _execute(run, check_id, argv, repo, device, timeout, check_env)
        if check_id == 'regressions' and result['status'] == 'passed':
            stderr = r.safe_child(run, result['stderr']['path']).read_text(encoding='utf-8', errors='replace')
            if 'skipped=' in stderr or 'Ran 0 tests' in stderr:
                result['status'], result['reason'] = 'failed', 'Required regressions skipped or no tests discovered'
        results.append(result)
        # Publish progressively so failure/interruption never implies a completed suite.
        r.write_record(run, 'baseline/checks.json', r.new_record('checks', run_id, results), overwrite=True)
        if result['status'] == 'interrupted':
            return dict(cpu_engineering_ready=False, gpu_execution_ready=False,
                        restrictions=['Baseline interrupted; checks incomplete'], checks=results)
    driver = _execute(run, 'gpu-driver', ['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'], repo, 'cuda', environment_seconds, env)
    results.append(driver)
    metadata_check = next(item for item in results if item['check_id']=='metadata')
    metadata = None
    if metadata_check['status'] == 'passed':
        try:
            metadata = json.loads(r.safe_child(run, metadata_check['stdout']['path']).read_text(encoding='utf-8'))
        except ValueError:
            metadata_check['status'], metadata_check['reason'] = 'failed', 'Malformed interpreter metadata'
    for device in ('cpu', 'cuda'):
        selected = [item for item in results if item['device'] == device and item['check_id']!='gpu-driver']
        observed = next((item for item in selected if item['check_id'] == f'environment-{device}'), None)
        probe = None
        if observed and observed['status'] == 'passed':
            text = r.safe_child(run, observed['stdout']['path'], must_exist=True).read_text(encoding='utf-8')
            try:
                probe = json.loads(text[text.index('{'):])
            except (ValueError, json.JSONDecodeError):
                observed['status'], observed['reason'] = 'failed', 'Missing/malformed environment JSON'
        if metadata is not None:
            metadata['unavailable_reasons'] = {key: 'Interpreter/platform did not report this field'
                                              for key, value in metadata.items() if value is None}
        data = dict(device=device, status='not_requested' if not selected else ('passed' if all(x['status']=='passed' for x in selected) else 'failed'),
                    reason=('Device not requested; availability not asserted' if not selected else
                            'Environment observation unavailable; see check logs' if probe is None else None),
                    interpreter_unavailable_reason='Metadata command failed; see logs' if metadata is None else None,
                    observed=probe, check_ids=[item['check_id'] for item in selected],
                    interpreter=metadata, metadata_check_ref='metadata',
                    execution_environment={'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'cuda_hidden': device=='cpu'},
                    driver={'value': driver['stdout'] if driver['status']=='passed' else None,
                            'status': 'verified' if driver['status']=='passed' else 'unknown',
                            'evidence_refs': [driver['stdout']['path'], driver['stderr']['path']],
                            'reason': None if driver['status']=='passed' else 'nvidia-smi failed or unavailable; see logs'})
        r.write_record(run, f'baseline/environment.{device}.json', r.new_record('environment', run_id, data))
    r.write_record(run, 'baseline/checks.json', r.new_record('checks', run_id, results), overwrite=True)
    cpu_ready = all(x['status']=='passed' for x in results if x['device']=='cpu')
    gpu_ready = 'cuda' in devices and all(x['status']=='passed' for x in results if x['check_id']=='environment-cuda')
    return dict(cpu_engineering_ready=cpu_ready, gpu_execution_ready=gpu_ready,
                restrictions=[] if gpu_ready else ['GPU execution not validated'], checks=results)


def capture_baseline(repo_root, run_dir, run_id, **check_options):
    """Capture a fresh baseline and reject source drift during its checks.

    Returns local baseline eligibility, never a G0/scientific gate. Caller owns
    validated configuration and a fresh run destination. No automatic retries of
    regressions: if source moves during checks, preserve the run and start another.
    """
    source = capture_source(repo_root, run_dir, run_id)
    checks = capture_environment_checks(repo_root, run_dir, run_id, **check_options)
    reason = None
    try:
        after = _source_state(Path(repo_root).resolve(), Path(run_dir).resolve())
        inventory = r.read_record(run_dir, 'baseline/source_inventory.json')['items']
        unchanged = after['entries'] == inventory
        unchanged &= all(after[key] == source[key] for key in
                         ('head', 'ref', 'status_porcelain_z', 'index_sha256', 'unresolved_lfs', 'unresolved_submodules'))
        unchanged &= after['patch'] == r.safe_child(run_dir, 'baseline/tracked.patch').read_bytes()
        unchanged &= after['staged_patch'] == r.safe_child(run_dir, 'baseline/staged.patch').read_bytes()
        if not unchanged:
            reason = 'Source changed during baseline checks; repeat in a new run'
    except (OSError, r.RecordError, subprocess.SubprocessError) as error:
        reason = f'Source recheck failed: {error}'
    r.write_record(run_dir, 'baseline/consistency.json', r.new_record('git', run_id,
                   dict(status='passed' if reason is None else 'failed', reason=reason,
                        source_snapshot_sha256=source['source_snapshot_sha256'], checked_at_utc=_now())))
    if reason:
        checks['cpu_engineering_ready'] = checks['gpu_execution_ready'] = False
        checks['restrictions'].append(reason)
    if source['unresolved_lfs'] or source['unresolved_submodules']:
        checks['cpu_engineering_ready'] = checks['gpu_execution_ready'] = False
        checks['restrictions'].append('Unresolved LFS/submodule source; snapshot is not self-contained')
    return dict(source=source, **checks)
