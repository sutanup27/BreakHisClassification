"""Increase a provisional resource allowance before main fitting, preserving jobs."""
import argparse
import copy
from datetime import datetime,timezone
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from breakhis import baseline_plan as b,study_execution as x,experiments as e,experiment_io as io


def revise(study_file,hours,reason):
    study_file=Path(study_file).resolve();root,old=x.frozen(study_file)
    if type(hours) not in (int,float) or not math.isfinite(hours) or hours<=old['study']['available_wall_hours'] or not reason.strip():raise ValueError('A larger finite allowance and explicit reason are required')
    with x.lease(root):
        root,old=x.frozen(study_file);ledger=e.read_json(root/'execution/ledger.json')
        kinds={j['job_id']:j['kind'] for j in old['fit_jobs']}
        if any(kinds[k]!='pilot' for k in ledger['jobs']):raise ValueError('Resource-only revision is allowed before any main job starts')
        measured_path=root/'execution/measured_cost_inputs.json'
        costs=e.read_json(measured_path)
        if costs['plan_sha256']!=b.sha(old):raise ValueError('Measured inputs belong to a different plan')
        files=[study_file,root/'jobs.json',root/'study.json',root/'status.json',root/'cost_estimate.json',root/'execution/ledger.json',measured_path]
        before={p:p.read_bytes() for p in files}
        original=e.read_json(study_file);original['available_wall_hours']=hours
        try:
            io.write_json(study_file,original,root=study_file.parent)
            new,configs,_,_=b.compile_plan(study_file)
            expected=copy.deepcopy(old);expected['study']['available_wall_hours']=hours;expected['inputs']['study_sha256']=e.digest(study_file)
            if new!=expected:raise ValueError('Resource revision would alter scientific jobs, code, data or settings')
            for key,value in configs.items():
                if e.read_json(root/'configs'/(key+'.json'))!=value:raise ValueError('Resolved draft changed')
            costs['plan_sha256']=b.sha(new);capacity=b.estimate_cost(new,costs)
            if not capacity['ready_for_main_capacity']:raise ValueError('New allowance still cannot cover measured plan: '+str(capacity['reasons']))
            revision=dict(at=datetime.now(timezone.utc).isoformat(),old_hours=old['study']['available_wall_hours'],new_hours=hours,reason=reason,
                old_plan_sha256=b.sha(old),new_plan_sha256=b.sha(new),unchanged_jobs_sha256=b.sha(old['fit_jobs']),
                note='Resource-only allocation amendment before main fitting; exact scientific jobs/configs/checkpoints unchanged. Prior failed capacity session retained.')
            ledger.setdefault('resource_revisions',[]).append(revision);ledger['plan_sha256']=b.sha(new)
            io.write_json(root/'jobs.json',new,root=root);io.write_json(root/'study.json',new['study'],root=root)
            io.write_json(root/'cost_estimate.json',capacity,root=root);io.write_json(measured_path,costs,root=root)
            io.write_json(root/'execution/ledger.json',ledger,root=root)
            status=e.read_json(root/'status.json');status.update(plan_sha256=b.sha(new),resource_revision_count=len(ledger['resource_revisions']))
            for name in status['artifacts']:status['artifacts'][name]=e.digest(root/name)
            io.write_json(root/'status.json',status,root=root)
            x.frozen(study_file)
        except BaseException:
            for path,content in before.items():io.publish(path,content,root=path.parent)
            raise
    return revision


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--hours',type=float,required=True);p.add_argument('--reason',required=True);a=p.parse_args()
    revision=revise(a.study,a.hours,a.reason);print('Resource allowance revised; scientific jobs unchanged: '+str(revision['new_hours'])+' hours')


if __name__=='__main__':main()
