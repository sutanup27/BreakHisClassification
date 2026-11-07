"""Run a real CPU CLI smoke workflow on generated images; no BreaKHis download."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_smoke(work_dir, checkpoint_dir, *, methods=('unstructured', 'channel'), threads=2):
    """Use actual subprocess commands and normal checkpoint policy; retain evidence."""
    from PIL import Image
    from breakhis.paths import checkpoint_directory
    work_dir, checkpoint_dir = Path(work_dir).resolve(), Path(checkpoint_dir).resolve()
    binary_root = checkpoint_directory('cancer_binary').resolve()
    if not checkpoint_dir.is_relative_to(binary_root) or checkpoint_dir == binary_root:
        raise ValueError(f'Smoke checkpoints must use a child directory of {binary_root}')
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError('Smoke work directory must be new or empty')
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError('Smoke checkpoint directory must be new or empty')
    if threads < 1 or not methods or len(set(methods)) != len(methods) or set(methods) - {'unstructured', 'channel'}:
        raise ValueError('Select supported methods and a positive thread count')
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = work_dir/'Dataset'
    for label, code, offset in [('benign', 'B_A', 1000), ('malignant', 'M_DC', 2000)]:
        directory = dataset_root/'dataset_cancer_v1/classificacao_binaria/40X'/label
        directory.mkdir(parents=True)
        for index in range(6):
            image = Image.new('RGB', (28, 20))
            image.putdata([((pixel*11 + index*17 + offset) % 256,
                            (pixel*3 + index*23 + offset//10) % 256,
                            (pixel*7 + index*31 + offset//20) % 256) for pixel in range(28*20)])
            image.save(directory/f'SOB_{code}-14-{offset+index}-40-001.png')
    logs = work_dir/'logs'
    logs.mkdir()
    commands = []
    def execute(label, arguments):
        # Every step starts a fresh interpreter. CLI/model/checkpoint code is not mocked.
        invocation = [sys.executable, '-m', 'breakhis', *map(str, arguments)]
        driver = ('import os,sys,torch; torch.set_num_threads(int(os.environ["BREAKHIS_SMOKE_THREADS"])); '
                  'from breakhis.cli import main; main(sys.argv[1:])')
        import os
        env = dict(os.environ, BREAKHIS_SMOKE_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        actual = [sys.executable, '-c', driver, *map(str, arguments)]
        commands.append({'step': label, 'argv': actual, 'equivalent_cli': invocation, 'threads': threads})
        (work_dir/'commands.json').write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8')
        try:
            result = subprocess.run(actual, cwd=ROOT,
                                    env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
        except subprocess.TimeoutExpired as error:
            def decode(value): return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
            (logs/f'{label}.log').write_text(decode(error.stdout)+'\n'+decode(error.stderr)+'\nTIMEOUT after 240 seconds', encoding='utf-8')
            raise RuntimeError(f'Smoke step {label} timed out; see {logs/label}.log') from error
        (logs/f'{label}.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Smoke step {label} failed ({result.returncode}); see {logs/label}.log\n{result.stderr[-2000:]}')
        print(f'PASS {label}', flush=True)
    manifest = work_dir/'data/manifest.json'
    execute('prepare', ['prepare-data', '--dataset-root', dataset_root, '--output', manifest.parent,
                        '--sources', 'cancer_binary', '--workers', '1'])
    data = ['--dataset', 'breakhis', '--manifest', manifest, '--source', 'cancer_binary',
            '--magnification', '40X', '--device', 'cpu', '--batch-size', '2', '--num-workers', '0', '--deterministic']
    dense = checkpoint_dir/'train.pth'
    execute('train', ['train', *data, '--model', 'vgg', '--epochs', '1', '--output', dense])
    reports = {}
    for method in methods:
        directory = checkpoint_dir/method
        pruned, finetuned = directory/'prune.pth', directory/'finetune.pth'
        execute(f'{method}-prune', ['prune', *data, '--checkpoint', dense, '--method', method,
                                   '--sparsity', '.5', '--output', pruned])
        execute(f'{method}-finetune', ['finetune', *data, '--checkpoint', pruned, '--epochs', '1', '--output', finetuned])
        report = work_dir/f'{method}-comparison.json'
        execute(f'{method}-compare', ['compare', *data, '--dense', dense, '--pruned', pruned, '--finetuned', finetuned,
                                     '--benchmark-warmup', '0', '--benchmark-repeats', '1', '--cpu-threads', threads, '--output', report])
        reports[method] = str(report)
    summary = dict(status='passed', synthetic=True, clinical_or_accuracy_validation=False,
                   dataset_images=12, device='cpu', threads=threads, checkpoint_dir=str(checkpoint_dir),
                   manifest=str(manifest), reports=reports, commands=str(work_dir/'commands.json'),
                   limitation='One-epoch synthetic CPU execution check; one timing repeat is not a performance benchmark.')
    (work_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, help='New/empty directory for generated data, logs and reports')
    parser.add_argument('--methods', nargs='+', choices=['unstructured', 'channel'], default=['unstructured','channel'])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    from breakhis.paths import checkpoint_directory
    identifier = 'smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    work = args.work_dir or ROOT/'artifacts'/identifier
    checkpoints = checkpoint_directory('cancer_binary')/identifier
    print(json.dumps(run_smoke(work, checkpoints, methods=args.methods, threads=args.threads), indent=2))


if __name__ == '__main__': main()
