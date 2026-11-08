"""Compare verified saved phases only; never load models or run inference."""
import csv
import io
from pathlib import Path
from . import experiments as e, experiment_setup as setup
from .phase_artifacts import development, verify_stage, validate_report
from .experiment_io import publish, json_bytes


def compare_saved(runs,output,*,phase='development',vary=()):
    if phase not in ('development','final'): raise ValueError('Unknown comparison phase')
    if len(runs)<2: raise ValueError('Compare at least two distinct runs')
    if set(vary)-{'training.selection_metric'} or len(set(vary))!=len(vary):
        raise ValueError('Only --vary training.selection_metric is supported; data/measurement invariants cannot vary')
    directories=[(Path(v) if Path(v).is_dir() else e.RUNS/v).resolve() for v in runs]
    if len(set(directories))!=len(directories): raise ValueError('Compare distinct runs')
    records=[]; diagnostics=[]; protected=list(directories)
    for run in directories:
        state=e.read_json(run/'run.json')
        if state.get('schema_version')!=2:
            raise ValueError('Schema-1 runs are readable but not eligible for phase-aware comparison; use new runs')
        if state.get('phases',{}).get(phase,{}).get('status')!='completed':
            diagnostics.append(dict(run=str(run),phase=phase,status=state.get('phases',{}).get(phase,{}).get('status'),
                error=state.get('phases',{}).get(phase,{}).get('error')))
            continue
        config,state,inputs,selection=development(run)
        if phase=='final':
            for role in selection['roles']: verify_stage(run,inputs,state,'final-'+role)
            verify_stage(run,inputs,state,'final-complete')
        result=e.read_json(run/phase/'results.json')
        reports={role:validate_report(run,config,inputs,selection,role,phase)[0] for role in selection['roles']}
        expected_reports={r:str(run/phase/f'{r}.metrics.json') for r in reports}
        expected_predictions={r:str(run/phase/f'{r}.predictions.json') for r in reports}
        if (result.get('schema_version')!=2 or result.get('run_id')!=state['run_id'] or result.get('phase')!=phase or result.get('split')!=('validation' if phase=='development' else 'test') or
            result.get('reports')!=expected_reports or result.get('predictions')!=expected_predictions or
            result.get('designated_role')!=selection['designated_role'] or result.get('config')!=config or
            result.get('identity')!=inputs['identity'] or
            result.get('designated_metrics')!=reports[selection['designated_role']]):
            raise ValueError('Phase results differ from verified reports/config/selection')
        if phase=='final':
            bound=setup.verify_reference(result['selection'],roots={'run':run})
            if bound!=run/'development/selection.json': raise ValueError('Final selection binding differs')
        protected.extend([run,Path(inputs['checkpoint_directory']),Path(config['data']['data_root']),
            Path(config['data']['manifest']),Path(inputs['config_source'])])
        records.append((run,config,state,inputs,selection,reports))
    if len(records)<2:
        raise ValueError('At least two completed requested phases are required; incomplete phases: '+str(diagnostics))
    def measurement(report):
        b=report['benchmark']
        return {k:b[k] for k in ('protocol','input_shape','dtype','input_seed','input_sha256','warmup','repeats',
            'timing_method','excludes','includes','runtime')}
    def invariants(record):
        run,config,state,inputs,selection,reports=record
        recipe={k:v for k,v in config['training'].items() if k not in ('model','seed','epochs','selection_metric')}
        if 'training.selection_metric' not in vary: recipe['selection_metric']=config['training']['selection_metric']
        return dict(phase=phase,split='validation' if phase=='development' else 'test',
            manifest_sha256=inputs['identity']['manifest_sha256'],source=config['data']['source'],
            magnification=config['data']['magnification'],preprocessing=selection['preprocessing'],
            classes=selection['classes'],class_to_idx=selection['class_to_idx'],
            aggregation=selection['aggregation'],threshold=selection['threshold'],
            training_recipe=recipe,training_runtime=inputs['identity']['runtime'],
            resolved_training_device=inputs['identity']['resolved_device'],evaluation=config['evaluation'],
            benchmark=measurement(next(iter(reports.values()))))
    protocol=invariants(records[0])
    for record in records:
        if invariants(record)!=protocol: raise ValueError('Comparison data/measurement/training invariants differ; selection changes require --vary training.selection_metric')
        if any(measurement(report)!=protocol['benchmark'] for report in record[-1].values()):
            raise ValueError('Role benchmark measurement invariants differ')
    factors={}
    for field in ('model','seed','epochs','selection_metric'):
        values={r[1]['name']+':'+r[2]['run_id']:r[1]['training'][field] for r in records}
        if len({str(v) for v in values.values()})>1: factors['training.'+field]=values
    pruning={r[2]['run_id']:r[1]['pruning'] for r in records}
    if len({str(v) for v in pruning.values()})>1: factors['pruning']=pruning
    table=[]
    for run,config,state,inputs,selection,reports in records:
        histories={command:e.read_json(run/f'{command}.history.json')['epochs']
            for command in (('train','finetune') if config['pruning'] else ('train',))}
        def budget(command):
            rows=histories.get(command,[])
            return dict(epochs=len(rows),steps=sum(r['optimizer_steps'] for r in rows),
                train_seconds=sum(r['training_seconds'] for r in rows),
                validation_seconds=sum(r['validation_seconds'] for r in rows))
        dense=budget('train'); fine=budget('finetune')
        for role,report in reports.items():
            g=report['group_metrics']; image=report['image_metrics']; b=report['benchmark']
            row=dict(run_id=state['run_id'],phase=phase,split=protocol['split'],role=role,
                designated=role==selection['designated_role'],checkpoint_sha256=selection['roles'][role]['checkpoint']['sha256'],
                selected_epoch=selection['roles'][role]['epoch'],model=config['training']['model'],seed=config['training']['seed'],
                training_device=inputs['identity']['resolved_device'],
                method='dense' if role=='dense' else config['pruning']['method'],
                sparsity=None if role=='dense' else config['pruning']['sparsity'],selection_metric=config['training']['selection_metric'],
                groups=g['samples'],benign_groups=g['per_class']['benign']['support'],malignant_groups=g['per_class']['malignant']['support'],
                group_balanced_accuracy=g['balanced_accuracy'],group_sensitivity=g['sensitivity'],group_specificity=g['specificity'],
                image_accuracy_percent=image['accuracy'],parameters=report['size']['logical_parameters'],
                conv_linear_macs=b['conv_linear_macs_per_image'],median_latency_ms=b['latency_ms']['median'],
                latency_change_vs_dense_percent=100*(b['latency_ms']['median']/reports['dense']['benchmark']['latency_ms']['median']-1),
                observed_speedup_vs_dense=reports['dense']['benchmark']['latency_ms']['median']/b['latency_ms']['median'],
                planned_dense_epochs=config['training']['epochs'],completed_dense_epochs=dense['epochs'],dense_optimizer_steps=dense['steps'],
                dense_training_seconds=dense['train_seconds'],dense_validation_seconds=dense['validation_seconds'],
                planned_finetune_epochs=config['pruning']['finetune_epochs'] if role=='finetuned' else 0,
                completed_finetune_epochs=fine['epochs'] if role=='finetuned' else 0,
                finetune_optimizer_steps=fine['steps'] if role=='finetuned' else 0,
                finetune_training_seconds=fine['train_seconds'] if role=='finetuned' else 0,
                finetune_validation_seconds=fine['validation_seconds'] if role=='finetuned' else 0,
                phase_stage_seconds=sum(r.get('seconds',0) for k,r in state['stages'].items()
                    if k.startswith(phase+'-') or phase=='development' and k in ('train','prune','finetune')))
            table.append(row)
    output=Path(output).absolute()
    from .paths import output_path
    output=output_path(output,root=output.parent,protected=protected)
    csv_path=output_path(output.with_suffix('.csv'),root=output.parent,protected=protected)
    if output.suffix!='.json' or output.exists() or csv_path.exists(): raise ValueError('Choose a new .json comparison output')
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    document=dict(schema_version=2,phase=phase,comparison_protocol=protocol,varied_factors=factors,
        declared_variations=list(vary),rows=table,diagnostics=diagnostics,
        note='Descriptive comparison, not a fairness verdict. Extra epochs/steps are disclosed, not equalized. Seeds are not independent patient samples. Speedups below one are slowdowns; one timing session does not establish significance.')
    # JSON is the completion record; publish it last. A failed pair is never complete.
    publish(csv_path,stream.getvalue().encode('utf-8'),overwrite=False)
    try: publish(output,json_bytes(document),overwrite=False)
    except BaseException:
        csv_path.unlink(missing_ok=True)
        raise
    return output
