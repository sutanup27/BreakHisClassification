"""Reconcile known-zero fitting cost after a failure before any fit stage began."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from breakhis import study_execution as execution,experiments as e,experiment_io as io,experiment_setup as setup


def reconcile(study_file):
    root,plan=execution.frozen(study_file);corrected=[]
    with execution.lease(root):
        file=root/'execution/ledger.json';ledger=e.read_json(file)
        for key,record in ledger['jobs'].items():
            if not record.get('unknown_work') or record['status'] not in ('failed','interrupted'):continue
            if any('seconds' not in a or a.get('status')=='running' for a in record.get('attempts',[])):continue
            run=Path(record['run'])
            config,state,inputs=setup._read_prepared(run)
            # Persistent M1 stages are never erased on retries. Absence proves that
            # this child never entered a train/finetune command, across all attempts.
            if set(state['stages']) & {'train','finetune'}:continue
            if any((run/(name+'.history.json')).exists() or (Path(inputs['checkpoint_directory'])/(name+'.history.json')).exists() for name in ('train','finetune')):continue
            rows,refs=execution.resources(run)
            if rows:continue
            record.setdefault('cost_corrections',[]).append(dict(at=datetime.now(timezone.utc).isoformat(),
                run_state_sha256=e.digest(run/'run.json'),basis='Verified frozen prepared inputs and persistent stage ledger: no child fit stage ever entered; fitting callback cost is known zero. All failed attempt wall time remains charged.'))
            record.update(unknown_work=False,actual_optimizer_steps=0,image_presentations=0,replayed_optimizer_steps=0)
            corrected.append(key)
        io.write_json(file,ledger,root=root)
    return corrected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);args=parser.parse_args()
    print('Verified zero-fit accounting corrections:',reconcile(args.study))


if __name__=='__main__':main()
