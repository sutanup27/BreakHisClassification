"""Training regressions: uneven batches, invalid values, device and selection."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def loader(self, inputs, targets, batch_size=2):
        from torch.utils.data import DataLoader, TensorDataset
        return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)

    def test_uneven_batches_have_sample_weighted_losses(self):
        from breakhis.training import train, evaluate
        t = self.torch
        inputs = t.tensor([[4., 0.], [4., 0.], [0., 4.]])
        targets = t.zeros(3, dtype=t.long)
        for reduction in ('mean', 'sum'):
            model = t.nn.Linear(2, 2, bias=False)
            with t.no_grad(): model.weight.copy_(t.eye(2))
            criterion = t.nn.CrossEntropyLoss(reduction=reduction)
            expected = t.nn.functional.cross_entropy(inputs, targets).item()
            actual = train(model, self.loader(inputs, targets), criterion,
                           t.optim.SGD(model.parameters(), lr=0))
            accuracy, loss = evaluate(model, self.loader(inputs, targets), criterion, verbose=False)
            self.assertAlmostEqual(actual, expected, places=6)
            self.assertAlmostEqual(loss, expected, places=6)
            self.assertAlmostEqual(accuracy, 200 / 3, places=4)

    def test_empty_loaders_are_rejected(self):
        from breakhis.training import train, evaluate
        t = self.torch
        model = t.nn.Linear(2, 2)
        empty = self.loader(t.empty(0, 2), t.empty(0, dtype=t.long))
        with self.assertRaisesRegex(ValueError, 'empty'):
            train(model, empty, t.nn.CrossEntropyLoss(), t.optim.SGD(model.parameters(), lr=.1))
        with self.assertRaisesRegex(ValueError, 'empty'):
            evaluate(model, empty, t.nn.CrossEntropyLoss(), verbose=False)

    def test_nonfinite_inputs_outputs_and_loss_are_rejected(self):
        from breakhis.training import train, evaluate
        t = self.torch
        targets = t.zeros(2, dtype=t.long)
        for phase in ('train', 'evaluate'):
            for fault in ('inputs', 'outputs', 'loss'):
                model = t.nn.Linear(2, 2)
                inputs = t.ones(2, 2)
                criterion = t.nn.CrossEntropyLoss()
                if fault == 'inputs': inputs[0, 0] = float('nan')
                if fault == 'outputs':
                    with t.no_grad(): model.weight.fill_(float('inf'))
                if fault == 'loss':
                    class InvalidLoss(t.nn.Module):
                        def forward(self, outputs, targets):
                            return outputs.sum() * float('nan')
                    criterion = InvalidLoss()
                with self.subTest(phase=phase, fault=fault), self.assertRaisesRegex(ValueError, 'nonfinite'):
                    if phase == 'train':
                        train(model, self.loader(inputs, targets), criterion,
                              t.optim.SGD(model.parameters(), lr=.1))
                    else:
                        evaluate(model, self.loader(inputs, targets), criterion, verbose=False)

    def test_nonfinite_gradients_do_not_update_weights(self):
        from breakhis.training import train
        t = self.torch
        model = t.nn.Linear(2, 2)
        before = model.weight.detach().clone()
        model.weight.register_hook(lambda gradient: t.full_like(gradient, float('nan')))
        with self.assertRaisesRegex(ValueError, 'gradient'):
            train(model, self.loader(t.ones(2, 2), t.zeros(2, dtype=t.long)),
                  t.nn.CrossEntropyLoss(), t.optim.SGD(model.parameters(), lr=.1))
        self.assertTrue(t.equal(before, model.weight))

    def test_cpu_model_overrides_global_device_and_zero_accuracy_is_retained(self):
        from breakhis.training import Training, predict
        t = self.torch
        model = t.nn.Linear(2, 2)
        with t.no_grad():
            model.weight.zero_()
            model.bias.copy_(t.tensor([1., 0.]))
        loader = self.loader(t.ones(2, 2), t.ones(2, dtype=t.long))
        with patch('breakhis.training.device', t.device('cuda')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(predict(model, t.ones(2)), 0)
            best, _, _, accuracies = Training(model, loader, loader, t.nn.CrossEntropyLoss(),
                                             t.optim.SGD(model.parameters(), lr=0), num_epochs=1)
        self.assertIsNotNone(best)
        self.assertEqual(accuracies, [0.0])
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            Training(model, loader, loader, t.nn.CrossEntropyLoss(),
                     t.optim.SGD(model.parameters(), lr=0), num_epochs=0)

    def test_cli_rejects_invalid_training_arguments_before_loading_data(self):
        from breakhis.cli import main
        for args in (['--epochs', '0'], ['--epochs', '-1'], ['--learning-rate', 'nan'],
                     ['--learning-rate', 'inf'], ['--learning-rate', '0'],
                     ['--batch-size', '0'], ['--num-workers', '-1']):
            with self.subTest(args=args), patch('breakhis.data.get_split_loaders') as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(['train', '--dataset', 'breakhis', *args])
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_cli_saves_first_zero_accuracy_epoch_without_iterating_test_split(self):
        from breakhis.cli import main
        from torch.utils.data import DataLoader, TensorDataset
        t = self.torch
        model = t.nn.Linear(2, 2)
        with t.no_grad():
            model.weight.zero_()
            model.bias.copy_(t.tensor([100., -100.]))
        dataset = TensorDataset(t.ones(3, 2), t.ones(3, dtype=t.long))
        dataset.classes = ['benign', 'malignant']
        class ReservedTestLoader(DataLoader):
            def __iter__(self):
                raise AssertionError('Training iterated the reserved test split')
        loaders = dict(train=DataLoader(dataset, batch_size=2),
                       validation=DataLoader(dataset, batch_size=2),
                       test=ReservedTestLoader(dataset, batch_size=2))
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'roots': {'cancer_binary': folder}, 'rows': []}))
            root = Path(folder) / 'checkpoint'
            with patch('breakhis.paths.CHECKPOINT_ROOT', root), \
                 patch('breakhis.data.get_split_loaders', return_value=loaders), \
                 patch('breakhis.checkpoints.create_model', return_value=model), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(['train', '--dataset', 'breakhis', '--manifest', str(manifest),
                      '--device', 'cpu', '--epochs', '2', '--experiment-id', 'zero-accuracy'])
            checkpoint = t.load(root / 'binary/40X/vgg/zero-accuracy/train.pth', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['metadata']['source'], 'cancer_binary')
            self.assertTrue(all(t.isfinite(value).all() for value in checkpoint['state_dict'].values()))


if __name__ == '__main__':
    unittest.main()
