"""Reporting units, mask-aware accounting and fixed-protocol comparison."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class ReportingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def test_cpu_runtime_never_initializes_cudnn_with_hidden_cuda(self):
        from breakhis.reporting import runtime_report
        with patch.object(self.t.backends.cudnn,'version',side_effect=ValueError('no visible CUDA devices')):
            result=runtime_report(self.t.device('cpu'))
        self.assertIsNone(result['cudnn'])
        self.assertEqual(result['cudnn_reason'],'Not queried for CPU execution')

    def test_cpu_environment_probe_avoids_gpu_metadata_when_availability_is_true(self):
        import tools.check_environment as check
        with patch.object(check.sys,'argv',['check_environment.py','--device','cpu']), \
             patch.object(self.t.cuda,'is_available',return_value=True), \
             patch.object(self.t.cuda,'get_device_name',side_effect=AssertionError('CPU probe queried GPU name')), \
             patch.object(self.t.backends.cudnn,'version',side_effect=ValueError('No visible CUDA devices')), \
             patch.object(check.subprocess,'run'):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): check.main()
        result=json.loads(stream.getvalue())
        self.assertIsNone(result['cuda_available'])
        self.assertIsNone(result['gpu'])
        self.assertIn('Not queried',result['gpu_reason'])

    def test_confusion_precision_recall_f1_and_absent_class(self):
        from breakhis.reporting import classification_metrics
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ['benign', 'malignant'])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(result['accuracy'], 75)
        self.assertAlmostEqual(result['per_class']['malignant']['precision'], 2/3)
        self.assertAlmostEqual(result['per_class']['malignant']['f1'], .8)
        self.assertEqual(result['sensitivity'], 1)
        self.assertEqual(result['specificity'], .5)
        self.assertAlmostEqual(result['macro']['recall'], .75)
        absent = classification_metrics([0], [0], ['benign', 'malignant'])
        self.assertIsNone(absent['per_class']['malignant']['f1'])
        self.assertIsNone(absent['macro']['f1'])

    def test_patient_probability_mean_gives_each_group_one_vote(self):
        from breakhis.reporting import grouped_patient_metrics
        rows = [{'patient_id': value} for value in ('a', 'a', 'a', 'b')]
        report = grouped_patient_metrics(rows, [0, 0, 0, 1], [[.1,.9],[.9,.1],[.9,.1],[.1,.9]], ['benign','malignant'])
        self.assertEqual(report['samples'], 2)
        self.assertEqual(report['accuracy'], 100)
        with self.assertRaisesRegex(ValueError, 'Conflicting labels'):
            grouped_patient_metrics(rows, [0,1,0,1], [[.5,.5]]*4, ['benign','malignant'])

    def test_size_counts_effective_masks_not_hidden_originals(self):
        from torch.nn.utils import prune
        from breakhis.reporting import size_report
        from breakhis.metrics import get_num_parameters, get_model_sparsity
        t = self.t
        model = t.nn.Linear(2, 2, bias=False)
        with t.no_grad(): model.weight.fill_(1)
        prune.custom_from_mask(model, 'weight', t.tensor([[0., 1.], [0., 1.]]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.pth'
            t.save(model.state_dict(), path)
            report = size_report(model, path)
            self.assertEqual(report['logical_parameters'], 4)
            self.assertEqual(report['effective_nonzero_parameters'], 2)
            self.assertEqual(report['parameter_sparsity'], .5)
            self.assertEqual(report['dense_parameter_bytes'], 16)
            self.assertEqual(report['ideal_nonzero_value_bytes'], 8)
            self.assertEqual(report['flat_index_sparse_bytes_estimate'], 24)
            self.assertEqual(report['checkpoint_file_bytes'], path.stat().st_size)
            self.assertEqual(get_num_parameters(model, True), 2)
            self.assertEqual(get_model_sparsity(model), .5)

    def test_actual_shapes_macs_and_benchmark_protocol(self):
        from breakhis.reporting import convolution_linear_macs, benchmark_model
        from breakhis.pruning import prune_weights
        t = self.t
        model = t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1,bias=False),t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,2,bias=False)).eval()
        inputs = t.ones(1,3,8,8)
        expected = 4*8*8*3*3*3 + 4*2
        self.assertEqual(convolution_linear_macs(model, inputs), expected)
        prune_weights(model, .5)
        self.assertEqual(convolution_linear_macs(model, inputs), expected)
        report = benchmark_model(model, inputs, batch_size=2, warmup=1, repeats=3)
        self.assertEqual(report['input_shape'], [2,3,8,8])
        self.assertEqual(report['conv_linear_macs_per_image'], expected)
        self.assertEqual(len(report['samples_ms']), 3)
        self.assertGreater(report['latency_ms']['median'], 0)
        self.assertEqual(report['runtime']['torch_cpu_threads'], 2)
        self.assertFalse(any(module._forward_hooks for module in model.modules()))
        with self.assertRaises(ValueError): benchmark_model(model, inputs, repeats=0)

    def test_evaluation_partial_batch_loss_and_patient_alignment(self):
        from torch.utils.data import DataLoader, TensorDataset
        from breakhis.reporting import collect_predictions
        t = self.t
        inputs = t.tensor([[4.,0.],[4.,0.],[0.,4.]])
        labels = t.zeros(3,dtype=t.long)
        dataset = TensorDataset(inputs, labels)
        dataset.classes = ['benign','malignant']
        dataset.rows = [{'patient_id': 'a'}, {'patient_id': 'a'}, {'patient_id': 'b'}]
        image, patients, first = collect_predictions(t.nn.Identity(), DataLoader(dataset,batch_size=2))
        self.assertAlmostEqual(image['loss'], t.nn.functional.cross_entropy(inputs,labels).item(), places=6)
        self.assertAlmostEqual(image['accuracy'], 200/3)
        self.assertEqual(patients['samples'], 2)
        self.assertEqual(tuple(first.shape), (1,2))
        with self.assertRaisesRegex(ValueError, 'ordered sequential'):
            collect_predictions(t.nn.Identity(), DataLoader(dataset,batch_size=2,shuffle=True))

    def test_comparison_keeps_slowdown_and_cli_validates_benchmark_settings(self):
        from breakhis.comparison import compare_deltas
        from breakhis.cli import main
        baseline = dict(accuracy=90, size=dict(logical_parameters=10,checkpoint_file_bytes=100,model_state_serialized_bytes=50),
                        benchmark=dict(latency_ms={'median':2},conv_linear_macs_per_image=10))
        slower = dict(accuracy=85, size=dict(logical_parameters=10,checkpoint_file_bytes=110,model_state_serialized_bytes=60),
                      benchmark=dict(latency_ms={'median':4},conv_linear_macs_per_image=10))
        delta = compare_deltas(baseline,slower)
        self.assertEqual(delta['observed_median_speedup'], .5)
        self.assertEqual(delta['median_latency_change_percent'],100)
        self.assertEqual(delta['accuracy_delta_percentage_points'],-5)
        for value in ('--benchmark-repeats','--benchmark-batch-size','--cpu-threads'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['evaluate','--dataset','breakhis','--checkpoint','unused.pth',value,'0'])

    def test_cli_three_checkpoint_comparison_without_train_or_validation_iteration(self):
        from torch.utils.data import DataLoader, TensorDataset
        from breakhis.cli import main
        from breakhis.checkpoints import save_checkpoint
        from breakhis.pruning import prune_weights
        t = self.t
        def create(*args, **kwargs): return t.nn.Sequential(t.nn.Linear(2,2))
        dataset = TensorDataset(t.tensor([[1.,0.],[0.,1.]]),t.tensor([0,1]))
        dataset.classes=['benign','malignant']
        dataset.rows=[{'patient_id':'a'},{'patient_id':'b'}]
        class ReservedLoader(DataLoader):
            def __iter__(self): raise AssertionError('Comparison iterated a training/validation loader')
        loaders={split:ReservedLoader(dataset) for split in ('train','validation')}
        loaders['test']=DataLoader(dataset,batch_size=1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(schema_version=1,roots={'cancer_binary':folder},rows=[])))
            metadata=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),source='cancer_binary',magnification='40X',preprocessing='resize-short-edge-256/crop224/ImageNet-normalization')
            model=create()
            save_checkpoint(root/'dense.pth',model,'vgg',metadata)
            prune_weights(model,.5)
            save_checkpoint(root/'pruned.pth',model,'vgg',metadata)
            save_checkpoint(root/'finetuned.pth',model,'vgg',metadata)
            output=root/'comparison.json'
            with patch('breakhis.data.get_split_loaders',return_value=loaders),patch('breakhis.checkpoints.create_model',side_effect=create),contextlib.redirect_stdout(io.StringIO()):
                main(['compare','--dataset','breakhis','--manifest',str(manifest),'--device','cpu',
                      '--dense',str(root/'dense.pth'),'--pruned',str(root/'pruned.pth'),'--finetuned',str(root/'finetuned.pth'),
                      '--output',str(output),'--benchmark-warmup','0','--benchmark-repeats','2'])
            report=json.loads(output.read_text())
            self.assertEqual(set(report['models']),{'dense','pruned','finetuned'})
            self.assertEqual(report['split'],'test')
            self.assertEqual(report['models']['dense']['patient_metrics']['samples'],2)
            self.assertTrue(output.with_suffix('.config.json').is_file())
            # Version-1 state-dictionary masks remain comparable to new fine-tuned files.
            old = t.load(root/'pruned.pth', weights_only=True)
            old['format_version'] = 1
            for key in ('architecture_config', 'pruning', 'config'):
                old.pop(key)
            t.save(old, root/'pruned.pth')
            with patch('breakhis.data.get_split_loaders',return_value=loaders),patch('breakhis.checkpoints.create_model',side_effect=create),contextlib.redirect_stdout(io.StringIO()):
                main(['compare','--dataset','breakhis','--manifest',str(manifest),'--device','cpu',
                      '--dense',str(root/'dense.pth'),'--pruned',str(root/'pruned.pth'),'--finetuned',str(root/'finetuned.pth'),
                      '--output',str(output),'--benchmark-warmup','0','--benchmark-repeats','1'])
            self.assertEqual(json.loads(output.read_text())['models']['pruned']['pruning']['type'],'unstructured')
            # Incompatible data is rejected before any loader/model computation.
            bad = t.load(root/'finetuned.pth', weights_only=True)
            bad['metadata']['magnification'] = '100X'
            t.save(bad, root/'finetuned.pth')
            with patch('breakhis.data.get_split_loaders') as loader, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['compare','--dataset','breakhis','--manifest',str(manifest),
                      '--dense',str(root/'dense.pth'),'--pruned',str(root/'pruned.pth'),'--finetuned',str(root/'finetuned.pth')])
            loader.assert_not_called()

    def test_evaluate_cli_writes_full_report_and_keeps_percent_accuracy(self):
        from torch.utils.data import DataLoader, TensorDataset
        from breakhis.cli import main
        from breakhis.checkpoints import save_checkpoint
        t = self.t
        def create(*args, **kwargs): return t.nn.Linear(2,2)
        dataset=TensorDataset(t.ones(3,2),t.zeros(3,dtype=t.long))
        dataset.classes=['benign','malignant']
        dataset.rows=[{'patient_id':'a'}]*3
        loaders={split:DataLoader(dataset,batch_size=2) for split in ('train','validation','test')}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); manifest=root/'manifest.json'
            images=root/'images'; images.mkdir()
            manifest.write_text(json.dumps(dict(schema_version=1,roots={'cancer_binary':str(images)},rows=[])))
            model=create()
            with t.no_grad(): model.weight.zero_(); model.bias.copy_(t.tensor([1.,0.]))
            metadata=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),source='cancer_binary',magnification='40X',preprocessing='resize-short-edge-256/crop224/ImageNet-normalization')
            checkpoint=root/'model.pth'; save_checkpoint(checkpoint,model,'vgg',metadata)
            output=root/'report.json'
            with patch('breakhis.data.get_split_loaders',return_value=loaders),patch('breakhis.checkpoints.create_model',side_effect=create),contextlib.redirect_stdout(io.StringIO()):
                main(['evaluate','--dataset','breakhis','--manifest',str(manifest),'--checkpoint',str(checkpoint),
                      '--device','cpu','--benchmark-warmup','0','--benchmark-repeats','2','--output',str(output)])
            report=json.loads(output.read_text())
            self.assertEqual(report['accuracy'],100)
            self.assertEqual(report['accuracy_unit'],'percent')
            self.assertEqual(report['image_metrics']['samples'],3)
            self.assertEqual(report['patient_metrics']['samples'],1)


if __name__ == '__main__': unittest.main()
