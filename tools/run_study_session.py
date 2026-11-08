"""Run one bounded real-study session; pause between jobs and reuse frozen runs."""
import argparse
from datetime import datetime,timezone,timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from breakhis import study_execution as execution,experiment_io as io,experiments as e


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--session-hours',type=float,default=8)
    parser.add_argument('--wait-for-pilot',action='store_true')
    parser.add_argument('--second-recipes',action='store_true')
    args=parser.parse_args()
    root,plan=execution.frozen(args.study)
    worker=root/'execution/worker';worker.mkdir(parents=True,exist_ok=True)
    with execution.lease(worker):
        ledger_path=root/'execution/ledger.json'
        ledger=e.read_json(ledger_path) if ledger_path.exists() else {}
        # Waiting for the current pilot shares its continuous-session start.
        start=datetime.now(timezone.utc)
        if args.wait_for_pilot and ledger.get('sessions'):
            start=datetime.fromisoformat(ledger['sessions'][0]['started_at'])
        deadline=start+timedelta(hours=args.session_hours)
        status=dict(pid=os.getpid(),status='starting',study_id=plan['study']['study_id'],started_at=start.isoformat(),deadline=deadline.isoformat(),
            note='Bounded session; waits for any active pilot, then runs parents/candidates one job at a time. No final test inference. A PAUSE file requests a stop at the next job boundary.')
        def save(**fields):
            status.update(fields,updated_at=datetime.now(timezone.utc).isoformat());io.write_json(root/'execution/live_status.json',status,root=root)
        try:
            if args.wait_for_pilot:
                save(status='waiting_for_pilot')
                while True:
                    ledger=e.read_json(ledger_path)
                    pilot=[j for j in plan['fit_jobs'] if j['kind']=='pilot']
                    if any(ledger.get('jobs',{}).get(j['job_id'],{}).get('status') in ('failed','interrupted') for j in pilot):raise RuntimeError('Pilot failed/interrupted; inspect its ledger before continuing')
                    if all(ledger.get('jobs',{}).get(j['job_id'],{}).get('status')=='completed' for j in pilot) and ledger.get('sessions',[])[-1].get('status')=='completed':break
                    if datetime.now(timezone.utc)>=deadline:save(status='paused_session_limit');return
                    time.sleep(15)
            for stage in ('parents','candidates'):
                while True:
                    hours=(deadline-datetime.now(timezone.utc)).total_seconds()/3600
                    if hours<=0 or (root/'execution/PAUSE').exists():save(status='paused_at_job_boundary',stage=stage);return
                    save(status='running',stage=stage)
                    from tools.reconcile_study_accounting import reconcile
                    reconcile(args.study)
                    result=execution.execute(args.study,stage,resume=True,session_hours=hours,max_jobs=1,second_recipes=args.second_recipes)
                    ledger=e.read_json(ledger_path);session=ledger['sessions'][-1]
                    save(completed_jobs=sum(r['status']=='completed' for r in ledger['jobs'].values()),last_session=session)
                    # Refresh saved reports only, never inference/training from the builder.
                    subprocess.run([sys.executable,str(ROOT/'tools/build_results.py')],cwd=ROOT,check=True)
                    if session['status']=='completed':break
                    if session['seconds']<1 and session['status']=='paused_at_job_boundary':
                        save(status='paused_insufficient_session_headroom',stage=stage);return
                    # A real job can take longer than one second just in validation.
                    if session['status']=='paused_at_job_boundary' and not any(a.get('started_at','')>=session['started_at'] for r in ledger['jobs'].values() for a in r.get('attempts',[])):
                        save(status='paused_insufficient_session_headroom',stage=stage);return
            result=execution.execute(args.study,'report')
            save(status='execution_complete' if result['complete_required'] else 'incomplete',scientific_quality_review='M2.5 pending')
        except BaseException as error:
            save(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=dict(type=type(error).__name__,message=str(error)))
            raise


if __name__=='__main__':main()
