"""Verify the simple experiment workflow on generated images, not research accuracy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path)
    args=parser.parse_args()
    from PIL import Image
    from breakhis.manifest import prepare_dataset
    from breakhis.experiments import load_config,read_json,write_json,run_experiment,compare_runs
    from breakhis.checkpoints import read_checkpoint
    import torch
    identifier='simple-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:4]
    work=(args.work_dir or ROOT/'artifacts'/identifier).resolve()
    if work.exists(): parser.error('Choose a new work directory')
    data=work/'Dataset'
    for label,code,offset in [('benign','B_A',1000),('malignant','M_DC',2000)]:
        directory=data/'dataset_cancer_v1/classificacao_binaria/40X'/label
        directory.mkdir(parents=True)
        for index in range(6):
            image=Image.new('RGB',(28,20))
            image.putdata([((pixel*11+index*17+offset)%256,(pixel*3+index*23+offset//10)%256,
                (pixel*7+index*31+offset//20)%256) for pixel in range(28*20)])
            image.save(directory/f'SOB_{code}-14-{offset+index}-40-001.png')
    prepare_dataset(data,work/'data',seed=101,workers=1,sources=['cancer_binary'])
    runs=[]
    for name in ('vgg_baseline','vgg_channel'):
        config=load_config(ROOT/f'configs/experiments/{name}.json',device='cpu')
        config['data'].update(manifest=str(work/'data/manifest.json'),data_root=str(data/'dataset_cancer_v1/classificacao_binaria'))
        config['training'].update(epochs=1,batch_size=2)
        config['evaluation'].update(warmup=0,repeats=2)
        if config['pruning']: config['pruning']['finetune_epochs']=1
        file=work/f'{name}.json'; write_json(file,config)
        run=run_experiment(file,run_id=identifier+'-'+('dense' if config['pruning'] is None else 'channel'))
        assert read_json(run/'run.json')['status']=='development_completed'
        assert read_json(run/'train.history.json')['epochs']
        predictions=read_json(run/'development/dense.predictions.json')
        assert len(predictions['rows'])==2
        assert read_json(run/'development/dense.metrics.json')['group_metrics']['available']
        runs.append(run)
    dense=[]
    for run in runs:
        state=read_json(run/'run.json')
        from breakhis.experiment_setup import verify_reference
        inputs=read_json(run/'inputs.json')
        checkpoint=verify_reference(state['stages']['train']['artifacts'][0],
            roots={'checkpoint':Path(inputs['checkpoint_directory']),'run':run})
        dense.append(read_checkpoint(checkpoint)['state_dict'])
        assert state['phases']['final']['status']=='not_requested'
        assert not (run/'final').exists()
        assert read_json(run/'development/dense.predictions.json')['split']=='validation'
    assert dense[0].keys()==dense[1].keys()
    assert all(torch.equal(dense[0][key],dense[1][key]) for key in dense[0]), 'Identical fresh seeds/configs must reproduce the dense model'
    from breakhis.final_evaluation import run_final
    for run in runs:
        run_final(run_id=run.name)
        assert read_json(run/'run.json')['phases']['final']['status']=='completed'
        assert read_json(run/'final/dense.predictions.json')['split']=='test'
    comparisons={phase:str(compare_runs(runs,work/f'{phase}.comparison.json',phase=phase))
        for phase in ('development','final')}
    summary=dict(status='passed',synthetic=True,accuracy_claim=False,images=12,
        runs=list(map(str,runs)),phases=['development','final'],comparisons=comparisons,dense_weights_reproduced=True,
        limitation='One-epoch generated-image execution check; two timing repeats do not establish performance.')
    write_json(work/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
