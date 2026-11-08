"""Classification metrics and explicitly scoped model efficiency measurements."""
import hashlib
import io
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn
from torch.utils.data import SequentialSampler, BatchSampler

from .training import model_device

AGGREGATION = 'filename-group/equal-magnification/mean-probability'
THRESHOLD = 'argmax'
BENCHMARK_PROTOCOL = 'resident-forward-synthetic-v1'
CLASSES = ['benign', 'malignant']


def classification_metrics(targets, predictions, classes):
    if not targets or len(targets) != len(predictions):
        raise ValueError('Metrics require equal nonempty target/prediction sequences')
    count = len(classes)
    matrix = [[0] * count for _ in classes]
    for target, prediction in zip(targets, predictions):
        if not 0 <= target < count or not 0 <= prediction < count:
            raise ValueError('Class index is outside the label mapping')
        matrix[target][prediction] += 1
    per_class = {}
    for index, label in enumerate(classes):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = 2 * true_positive / (support + predicted) if support + predicted else None
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support, predicted=predicted,
            undefined={k:v for k,v in dict(precision='No predicted positives' if not predicted else None,
                recall='No true class support' if not support else None,
                f1='No true or predicted support' if not support+predicted else None).items() if v})
    total = len(targets)
    def average(metric, weighted=False):
        values=[(v[metric],v['support'] if weighted else 1) for v in per_class.values()
                if not weighted or v['support']]
        if any(v is None for v,w in values): return None
        return sum(v*w for v,w in values)/sum(w for v,w in values)
    macro = {m:average(m) for m in ('precision','recall','f1')}
    weighted = {m:average(m,True) for m in ('precision','recall','f1')}
    result = dict(samples=total, classes=list(classes), class_to_idx={label: index for index, label in enumerate(classes)},
                  accuracy=100 * sum(matrix[index][index] for index in range(count)) / total,
                  accuracy_unit='percent', rate_unit='fraction', per_class=per_class,
                  macro=macro, weighted=weighted, confusion_matrix=matrix,
                  confusion_matrix_axes={'rows': 'true', 'columns': 'predicted'}, zero_division=None, balanced_accuracy=macro['recall'],
                  balanced_accuracy_reason='Missing true class support' if macro['recall'] is None else None)
    if classes == ['benign', 'malignant']:
        result.update(positive_class='malignant', sensitivity=per_class['malignant']['recall'],
                      specificity=per_class['benign']['recall'])
    return result


def grouped_patient_metrics(rows, targets, probabilities, classes):
    if rows is None:
        return {'available': False, 'reason': 'Dataset has no ordered filename patient metadata'}
    validate_probabilities(targets, probabilities, classes)
    if len(rows) != len(targets):
        raise ValueError('Patient metadata and predictions differ in length')
    groups = {}
    for row, target, probability in zip(rows, targets, probabilities):
        patient = row.get('patient_id')
        if not patient:
            raise ValueError('Missing filename patient/slide identifier')
        group = groups.setdefault(patient, {'target': target, 'scores': [0.0] * len(classes), 'images': 0})
        if group['target'] != target:
            raise ValueError(f'Conflicting labels in filename patient/slide group {patient}')
        group['images'] += 1
        group['scores'] = [left + right for left, right in zip(group['scores'], probability)]
    records = []
    for patient, group in sorted(groups.items()):
        scores = [value / group['images'] for value in group['scores']]
        records.append(dict(patient_id=patient, images=group['images'], target=group['target'],
                            prediction=max(range(len(classes)), key=lambda index: scores[index]), probabilities=scores))
    attach_group_identity(records, rows, 'patient_id')
    result = classification_metrics([row['target'] for row in records], [row['prediction'] for row in records], classes)
    result.update(available=True, aggregation='Unweighted mean softmax probability per filename patient/slide ID; argmax with first-label tie break',
                  grouping='YEAR-SLIDE_ID; a filename surrogate, not independently verified patient identity', groups=records)
    return result


def magnification_balanced_metrics(rows, targets, probabilities, classes):
    """One vote per available magnification, then one vote per filename group."""
    if rows is None or any(not row.get('magnification') for row in rows):
        return dict(available=False, reason='Ordered magnification metadata unavailable')
    if len(rows) != len(targets) or len(targets) != len(probabilities):
        raise ValueError('Group prediction metadata length differs')
    validate_probabilities(targets, probabilities, classes)
    groups = {}
    for row, target, scores in zip(rows, targets, probabilities):
        group = groups.setdefault(row['patient_id'], dict(target=target, magnifications={}))
        if group['target'] != target:
            raise ValueError('Conflicting labels in filename group')
        group['magnifications'].setdefault(row['magnification'], []).append(scores)
    records = []
    for name, group in sorted(groups.items()):
        means = [[sum(v[i] for v in values)/len(values) for i in range(len(classes))]
                 for values in group['magnifications'].values()]
        scores = [sum(v[i] for v in means)/len(means) for i in range(len(classes))]
        records.append(dict(group_id=name, target=group['target'], probabilities=scores,
            prediction=max(range(len(classes)), key=lambda i:scores[i]),
            magnifications=sorted(group['magnifications'])))
    attach_group_identity(records, rows, 'group_id')
    result = classification_metrics([v['target'] for v in records], [v['prediction'] for v in records], classes)
    result.update(available=True, balanced_accuracy=result['macro']['recall'], groups=records,
        aggregation='Mean probability within magnification, then equal mean over available magnifications',
        grouping='Filename group; not independently verified patient identity', missing_magnifications='available-only')
    return result


def validate_probabilities(targets, probabilities, classes):
    if len(targets)!=len(probabilities): raise ValueError('Probability coverage differs')
    for scores in probabilities:
        if (len(scores)!=len(classes) or any(not math.isfinite(v) or not 0<=v<=1 for v in scores)
                or not math.isclose(sum(scores),1.,abs_tol=1e-5)):
            raise ValueError('Invalid class-ordered probabilities')


def attach_group_identity(records, rows, key):
    for record in records:
        patient=record[key]; selected=[r for r in rows if r['patient_id']==patient]
        components={r['group_id'] for r in selected if 'group_id' in r}
        if len(components)>1: raise ValueError('Inconsistent component within filename group')
        record.update(patient_id=patient,component_id=next(iter(components),None),
            subtypes=sorted({r['subtype'] for r in selected if 'subtype' in r}),
            images=len(selected),magnifications=sorted({r['magnification'] for r in selected if 'magnification' in r}))


@torch.inference_mode()
def collect_predictions(model, loader, *, prediction_rows=None):
    if getattr(loader, 'drop_last', False):
        raise ValueError('Evaluation cannot drop the last batch')
    rows = getattr(loader.dataset, 'rows', None)
    if rows is not None and not isinstance(loader.sampler, SequentialSampler):
        raise ValueError('Patient reporting requires an ordered sequential evaluation loader')
    if type(loader.batch_sampler) is not BatchSampler or list(loader.sampler)!=list(range(len(loader.dataset))):
        raise ValueError('Evaluation requires complete sequential batch coverage')
    if rows is not None:
        if len(rows)!=len(loader.dataset): raise ValueError('Metadata coverage differs from dataset')
        paths=[r['path'].casefold() for r in rows if 'path' in r]
        if len(set(paths))!=len(paths): raise ValueError('Duplicate prediction image identity')
    device = model_device(model)
    model.eval()
    targets, predictions, probabilities = [], [], []
    loss_sum, first_input, shape = 0.0, None, None
    classes = list(loader.dataset.classes)
    for inputs, labels in loader:
        if inputs.size(0) == 0:
            continue
        if not torch.isfinite(inputs).all():
            raise ValueError('Evaluation contains nonfinite inputs')
        if shape is None:
            shape = list(inputs.shape[1:])
            first_input = inputs[:1].detach().clone()
        if list(inputs.shape[1:]) != shape:
            raise ValueError('Evaluation images must have a fixed input shape')
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        if logits.shape != (labels.size(0), len(classes)) or not torch.isfinite(logits).all():
            raise ValueError('Invalid or nonfinite evaluation logits')
        loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite evaluation loss')
        loss_sum += loss.item()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(logits.softmax(1).cpu().tolist())
    if len(targets)!=len(loader.dataset): raise ValueError('Missing or duplicated evaluation rows')
    if rows is not None and any('label' in r and classes[t]!=r['label'] for r,t in zip(rows,targets)):
        raise ValueError('Prediction target differs from manifest label')
    metrics = classification_metrics(targets, predictions, classes)
    metrics['loss'] = loss_sum / len(targets)
    metrics['loss_definition'] = 'Mean image cross-entropy'
    patients = grouped_patient_metrics(rows, targets, probabilities, classes)
    if prediction_rows is not None:
        for index, (target, prediction, scores) in enumerate(zip(targets, predictions, probabilities)):
            identity = {key: rows[index][key] for key in ('path','sha256','pixel_sha256','patient_id','group_id','magnification','subtype','split')
                        if rows is not None and key in rows[index]}
            prediction_rows.append(dict(**identity, target=target, prediction=prediction, probabilities=scores))
    return metrics, patients, first_input


def effective_parameters(model):
    seen = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield parameter * module.weight_mask if name == 'weight_orig' and hasattr(module, 'weight_mask') else parameter


def size_report(model, checkpoint):
    total, nonzero, dense_bytes, value_bytes = 0, 0, 0, 0
    for parameter in effective_parameters(model):
        count = parameter.numel()
        retained = int(parameter.count_nonzero())
        total += count
        nonzero += retained
        dense_bytes += count * parameter.element_size()
        value_bytes += retained * parameter.element_size()
    masks = [module.weight_mask for module in model.modules() if hasattr(module, 'weight_mask')]
    from .checkpoints import cpu_state
    stream = io.BytesIO()
    torch.save(cpu_state(model.state_dict()), stream)
    return dict(logical_parameters=total, effective_nonzero_parameters=nonzero,
                parameter_sparsity=1 - nonzero / total if total else 0.0,
                masked_weight_entries=sum(int((mask == 0).sum()) for mask in masks),
                mask_entries=sum(mask.numel() for mask in masks),
                dense_parameter_bytes=dense_bytes, ideal_nonzero_value_bytes=value_bytes,
                flat_index_sparse_bytes_estimate=value_bytes + 8 * nonzero,
                sparse_estimate_definition='Effective nonzero parameter values plus one int64 flat index per value; excludes buffers, shapes, metadata and format overhead',
                checkpoint_file_bytes=Path(checkpoint).stat().st_size,
                checkpoint_size_note='Actual entire file, including any optimizer, RNG, buffers and masks; not model-only storage',
                model_state_serialized_bytes=stream.tell(),
                model_state_size_note='Measured torch.save(state_dict) bytes, includes mask buffers but excludes optimizer/RNG',
                mask_buffer_bytes=sum(mask.numel() * mask.element_size() for mask in masks))


@torch.inference_mode()
def convolution_linear_macs(model, inputs):
    total = 0
    handles = []
    def hook(module, arguments, output):
        nonlocal total
        operations = module.in_features if isinstance(module, nn.Linear) else (
            module.in_channels // module.groups * math.prod(module.kernel_size))
        total += output.numel() * operations
    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook))
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return total


def runtime_report(device):
    result = dict(python=platform.python_version(), pytorch=str(torch.__version__),
                  cuda_runtime=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version() if device.type == 'cuda' else None,
                  cudnn_reason=None if device.type == 'cuda' else 'Not queried for CPU execution',
                  device=str(device), cpu=platform.processor(), platform=platform.platform(),
                  logical_cpu_count=os.cpu_count(), torch_cpu_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      compute_capability=[properties.major, properties.minor])
    return result


@torch.inference_mode()
def benchmark_model(model, first_input, batch_size=1, warmup=10, repeats=30):
    if batch_size < 1 or warmup < 0 or repeats < 1:
        raise ValueError('Benchmark batch size/repeats must be positive; warmup nonnegative')
    device = model_device(model)
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(0)
    synthetic = torch.randn((batch_size,*first_input.shape[1:]),generator=generator,dtype=first_input.dtype)
    input_sha256 = hashlib.sha256(synthetic.numpy().tobytes()).hexdigest()
    inputs = synthetic.to(device)
    def synchronize():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    macs = convolution_linear_macs(model, inputs)
    for _ in range(warmup): model(inputs)
    synchronize()
    timings = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        outputs = model(inputs)
        synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if not torch.isfinite(outputs).all():
            raise ValueError('Nonfinite benchmark outputs')
        if not math.isfinite(elapsed) or elapsed<=0: raise ValueError('Invalid benchmark duration')
        timings.append(elapsed)
    ordered = sorted(timings)
    position = .95 * (len(ordered) - 1)
    lower = int(position)
    p95 = ordered[lower] + (ordered[min(lower+1, len(ordered)-1)] - ordered[lower]) * (position-lower)
    median = statistics.median(timings)
    return dict(input_shape=list(inputs.shape), dtype=str(inputs.dtype), warmup=warmup, repeats=repeats,
                protocol=BENCHMARK_PROTOCOL, input_seed=0, input_sha256=input_sha256,
                input_policy='Local CPU RNG standard-normal synthetic resident input; independent of held-out image values',
                pre_measurement_forwards={'mac_counting': 1, 'warmup': warmup},
                timing_method='Device-synchronized wall-clock forward latency; GPU sync before and after every measured call',
                excludes=['disk IO', 'image transforms', 'host-to-device copy', 'loss', 'metrics'],
                includes=['model forward', 'Python dispatch', 'pruning forward hooks', 'device synchronization overhead'],
                latency_ms=dict(mean=statistics.mean(timings), median=median, p95=p95,
                                minimum=min(timings), maximum=max(timings), stddev=statistics.pstdev(timings)),
                samples_ms=timings, throughput_images_per_second=batch_size * 1000 / median,
                amortized_ms_per_image=median / batch_size,
                conv_linear_macs_per_image=macs // batch_size,
                conv_linear_flops_estimate_per_image=2 * macs // batch_size,
                compute_scope='Dense Conv2d/Linear multiply-accumulates only; excludes bias, BN, pooling, activations, residual adds and mask multiplication. Two FLOPs per MAC; zeros do not reduce dense MACs.',
                runtime=runtime_report(device))


def checkpoint_report(model, loader, checkpoint, *, batch_size=1, warmup=10, repeats=30, prediction_rows=None):
    predictions = []
    image, patient, first_input = collect_predictions(model, loader, prediction_rows=predictions)
    if prediction_rows is not None:
        prediction_rows.extend(predictions)
    groups = magnification_balanced_metrics(getattr(loader.dataset, 'rows', None),
        [v['target'] for v in predictions], [v['probabilities'] for v in predictions], loader.dataset.classes)
    with Path(checkpoint).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest,
                accuracy=image['accuracy'], accuracy_unit='percent', loss=image['loss'],
                image_metrics=image, patient_metrics=patient, group_metrics=groups,
                size=size_report(model, checkpoint),
                benchmark=benchmark_model(model, first_input, batch_size, warmup, repeats))
