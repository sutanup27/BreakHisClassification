"""Internal bounded M0 audit process; use tools/run_m0.py for normal operation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breakhis.research import records as r
from breakhis.research.data_readiness import audit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = r.read_record(run, 'config.resolved.json', expected_type='config.resolved')
    lineage = r.read_record(run, 'inputs/lineage.json', expected_type='lineage')
    if config['run_id'] != lineage['run_id']:
        raise r.IntegrityError('Input run mismatch')
    data = config['data']
    result = audit_data(data['data_root'], run, config['run_id'], repo_root=data['repo_root'],
                        workers=data['audit_workers'], lineage=lineage['data'])
    print(result['counts'], result['blockers'], flush=True)


if __name__ == '__main__':
    main()
