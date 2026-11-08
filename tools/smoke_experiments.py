"""Exercise public experiment CLIs on generated images, including real retries."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


@contextmanager
def unavailable_images(root, rows, holding):
    """Temporarily hide only generated fixture images; restore even after failure."""
    root,holding=Path(root).resolve(),Path(holding).resolve()
    moved=[]
    try:
        for row in rows:
            source=(root/row['path']).resolve(); target=(holding/row['path']).resolve()
            if not source.is_relative_to(root) or not target.is_relative_to(holding):
                raise ValueError('Fixture image path escaped its directory')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise ValueError('Fixture holding destination already exists')
            source.rename(target); moved.append((source,target))
        yield
    finally:
        for source,target in reversed(moved): target.rename(source)


def run_smoke(work_dir, *, run_prefix=None):
    from PIL import Image
    import torch
    from breakhis.experiments import RUNS, load_config, read_json, write_json, digest
    from breakhis.checkpoints import read_checkpoint, load_checkpoint
    from breakhis.paths import checkpoint_directory, validate_experiment_id
    from breakhis.phase_artifacts import development, verify_stage, validate_report
    work=Path(work_dir).resolve()
    if work.exists(): raise ValueError('Choose a new work directory')
    identifier=validate_experiment_id(run_prefix or 'phase-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:4])
    ids=[validate_experiment_id(identifier+'-'+kind) for kind in ('dense','mask','channel')]
    for name in ids:
        if (RUNS/name).exists() or (checkpoint_directory('cancer_binary')/'40X/vgg'/name).exists():
            raise ValueError('Smoke run ID already exists')
    work.mkdir(parents=True); logs=work/'logs'; logs.mkdir(); commands=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')

    def execute(label,args,*,failure=False):
        argv=[sys.executable,*map(str,args)]
        record=dict(step=label,argv=argv,expected_failure=failure,log=str(logs/f'{label}.log'))
        commands.append(record); write_json(work/'commands.json',commands)
        try:
            result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,
                encoding='utf-8',errors='replace',timeout=240)
        except subprocess.TimeoutExpired as error:
            record['status']='timeout'; write_json(work/'commands.json',commands)
            (logs/f'{label}.log').write_text('TIMEOUT after 240 seconds',encoding='utf-8')
            raise RuntimeError('Smoke command timed out: '+label) from error
        record.update(return_code=result.returncode,status='passed' if bool(result.returncode)==failure else 'failed')
        write_json(work/'commands.json',commands)
        (logs/f'{label}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if record['status']!='passed': raise RuntimeError('Smoke command failed: '+label+'; see '+record['log'])
        print('PASS '+label,flush=True)
        return result

    data=work/'Dataset'; images=data/'dataset_cancer_v1/classificacao_binaria'
    for label,code,offset in [('benign','B_A',1000),('malignant','M_DC',2000)]:
        directory=images/'40X'/label; directory.mkdir(parents=True)
        for index in range(6):
            image=Image.new('RGB',(28,20))
            image.putdata([((pixel*11+index*17+offset)%256,(pixel*3+index*23+offset//10)%256,
                (pixel*7+index*31+offset//20)%256) for pixel in range(28*20)])
            image.save(directory/f'SOB_{code}-14-{offset+index}-40-001.png')
    manifest=work/'data/manifest.json'
    execute('prepare',['-m','breakhis','prepare-data','--dataset-root',data,'--sources','cancer_binary',
        '--output',manifest.parent,'--seed','101','--workers','1'])
    rows=read_json(manifest)['rows']; frozen=manifest.read_bytes(); runs=[]; configs=[]
    test=[row for row in rows if row['split']=='test']; fitting=[row for row in rows if row['split']!='test']
    with unavailable_images(images,test,work/'held-test'):
        for kind,name in zip(('dense','mask','channel'),ids):
            config=load_config(ROOT/f'configs/experiments/{"vgg_baseline" if kind=="dense" else "vgg_channel"}.json',device='cpu')
            config['data'].update(manifest=str(manifest),data_root=str(images))
            config['training'].update(epochs=1,batch_size=2)
            config['evaluation'].update(warmup=0,repeats=2)
            if config['pruning']:
                config['pruning'].update(method='unstructured' if kind=='mask' else 'channel',finetune_epochs=1)
            file=work/f'{kind}.json'; write_json(file,config); configs.append(file)
            args=['run_experiment.py','--config',file,'--run-id',name]
            run=RUNS/name
            if kind=='dense':
                # Real child failure after epoch-zero checkpoint creation; valid retry
                # restores the same original input bytes and resumes that boundary.
                broken=images/next(r['path'] for r in rows if r['split']=='train')
                original=broken.read_bytes()
                try:
                    broken.write_bytes(b'corrupt generated training fixture')
                    execute('development-failure',args,failure=True)
                finally: broken.write_bytes(original)
                failed=read_json(run/'run.json'); assert failed['phases']['development']['status']=='failed'
                checkpoint=Path(read_json(run/'inputs.json')['checkpoint_directory'])
                assert read_checkpoint(checkpoint/'train-last.pth')['epoch']==0
                assert 'changed since audit' in (run/'logs/train.log').read_text(encoding='utf-8').lower()
                execute('development-resume',[*args,'--resume'])
            else: execute('development-'+kind,args)
            config,state,inputs,selection=development(run)
            assert (run/'manifest.json').read_bytes()==frozen
            assert read_json(run/'config.json')==config
            assert state['phases']['final']['status']=='not_requested' and not (run/'final').exists()
            assert read_json(run/'results.json')['phases']['final'] is None
            for role in selection['roles']: validate_report(run,config,inputs,selection,role,'development')
            for command in ('train','finetune') if config['pruning'] else ('train',):
                history=read_json(run/f'{command}.history.json')['epochs']
                assert len(history)==1 and history[0]['epoch']==1 and history[0]['optimizer_steps']>0
            runs.append(run)
    dense=[]; invariants={}
    for kind,run in zip(('dense','mask','channel'),runs):
        inputs=read_json(run/'inputs.json'); checkpoint=Path(inputs['checkpoint_directory'])
        assert checkpoint.is_relative_to(checkpoint_directory('cancer_binary')) and not checkpoint.is_relative_to(ROOT)
        dense.append(read_checkpoint(checkpoint/'train.pth')['state_dict'])
        if kind!='dense':
            before=read_checkpoint(checkpoint/'prune.pth'); after=read_checkpoint(checkpoint/'finetune.pth')
            model,payload=load_checkpoint(checkpoint/'finetune.pth','cpu')
            assert payload['pruning']==before['pruning']
            if kind=='mask':
                masks=[k for k in before['state_dict'] if k.endswith('.weight_mask')]; assert masks
                for key in masks:
                    assert torch.equal(before['state_dict'][key],after['state_dict'][key])
                    assert torch.all(after['state_dict'][key.removesuffix('_mask')+'_orig'][after['state_dict'][key]==0]==0)
            else:
                assert before['architecture_config']==after['architecture_config']
                assert sum(p.numel() for p in model.parameters())<sum(v.numel() for v in dense[0].values())
            invariants[kind]=True
            del model,payload,before,after
    assert all(a.keys()==dense[0].keys() and all(torch.equal(a[k],dense[0][k]) for k in a) for a in dense[1:])
    del dense
    with unavailable_images(images,fitting,work/'held-fitting'):
        for index,run in enumerate(runs):
            if index==0:
                broken=images/test[0]['path']; original=broken.read_bytes()
                dev_hashes={p.name:digest(p) for p in (run/'development').iterdir()}
                try:
                    broken.write_bytes(b'corrupt generated final fixture')
                    execute('final-failure',['run_experiment.py','--run-id',run.name,'--phase','final'],failure=True)
                finally: broken.write_bytes(original)
                assert read_json(run/'run.json')['phases']['final']['status']=='failed'
                assert dev_hashes=={p.name:digest(p) for p in (run/'development').iterdir()}
                execute('development-comparison-after-final-failure',['compare_experiments.py','--runs',*ids,
                    '--output',work/'development.comparison.json'])
                execute('final-resume',['run_experiment.py','--run-id',run.name,'--phase','final','--resume'])
            else: execute('final-'+str(index),['run_experiment.py','--run-id',run.name,'--phase','final'])
            config,state,inputs,selection=development(run)
            assert state['phases']['final']['status']=='completed'
            for role in selection['roles']:
                verify_stage(run,inputs,state,'final-'+role)
                validate_report(run,config,inputs,selection,role,'final')
            verify_stage(run,inputs,state,'final-complete')
        # Comparisons must work with every fixture pixel file unavailable.
        with unavailable_images(images,test,work/'held-test'):
            execute('final-comparison',['compare_experiments.py','--runs',*ids,'--phase','final',
                '--output',work/'final.comparison.json'])
    for phase in ('development','final'):
        comparison=read_json(work/f'{phase}.comparison.json')
        assert len(comparison['rows'])==7 and comparison['phase']==phase
        assert (work/f'{phase}.comparison.csv').is_file()
    rejected=execute('completed-final-refusal',['run_experiment.py','--run-id',ids[0],'--phase','final'],failure=True)
    assert 'already completed' in rejected.stderr.lower()
    summary=dict(status='passed',synthetic=True,accuracy_claim=False,images=12,runs=list(map(str,runs)),
        phases=['development','final'],methods=['dense','unstructured','channel'],
        comparisons={p:str(work/f'{p}.comparison.json') for p in ('development','final')},
        dense_weights_reproduced=True,pruning_invariants=invariants,test_pixels_unavailable_during_development=True,
        fitting_pixels_unavailable_during_final=True,all_pixels_unavailable_during_final_comparison=True,
        actual_training_failure_resumed=True,actual_final_failure_resumed=True,public_cli_only=True,
        commands=str(work/'commands.json'),
        limitation='One-epoch generated-image execution checks; two timing repeats do not establish research quality or performance.')
    write_json(work/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path)
    args=parser.parse_args()
    from breakhis.experiments import write_json
    identifier='m1-acceptance-'+uuid.uuid4().hex[:8]
    try:
        summary=run_smoke(args.work_dir or ROOT/'artifacts'/identifier,run_prefix=identifier)
    except (ValueError,OSError,RuntimeError,AssertionError) as error: parser.exit(1,str(error)+'\n')
    import json
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
