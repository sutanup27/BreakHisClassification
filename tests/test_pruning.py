"""Pruning invariants, edge cases, fine-tuning, and reconstruction."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class PruningTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def test_exact_counts_ties_tiny_tensors_and_boundaries(self):
        from breakhis.pruning import magnitude_mask
        t = self.t
        for count in (1, 2, 3, 11):
            for ratio in (0, .01, .5, .9, 1):
                mask = magnitude_mask(t.ones(count), ratio)
                self.assertEqual(int((mask == 0).sum()), round(count * ratio))
                self.assertTrue(t.equal(mask, magnitude_mask(t.ones(count), ratio)))
        self.assertEqual(int((magnitude_mask(t.ones(2, 3).T, .5) == 0).sum()), 3)
        for ratio in (-.1, 1.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError): magnitude_mask(t.ones(1), ratio)
        with self.assertRaises(ValueError): magnitude_mask(t.empty(0), .5)

    def test_missing_settings_tied_weights_and_repeated_pruning(self):
        from breakhis.pruning import prune_weights
        t = self.t
        model = t.nn.Sequential(t.nn.Linear(2, 2), t.nn.Linear(2, 2))
        with self.assertRaisesRegex(ValueError, 'every prunable layer'):
            prune_weights(model, settings={'0': .5})
        self.assertFalse(hasattr(model[0], 'weight_mask'))
        model[1].weight = model[0].weight
        with self.assertRaisesRegex(ValueError, 'Tied/shared'): prune_weights(model)
        model = t.nn.Linear(2, 2)
        prune_weights(model, .5)
        prune_weights(model, .75)
        self.assertEqual(int((model.weight_mask == 0).sum()), 3)
        with self.assertRaisesRegex(ValueError, 'reduce existing sparsity'): prune_weights(model, .5)

    def test_masks_originals_and_momentum_stay_zero_after_updates_and_reload(self):
        from torch.utils.data import DataLoader, TensorDataset
        from breakhis.pruning import prune_weights
        from breakhis.training import train
        from breakhis.checkpoints import save_checkpoint, load_checkpoint
        t = self.t
        model = t.nn.Sequential(t.nn.Linear(2, 2))
        optimizer = t.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.1)
        loader = DataLoader(TensorDataset(t.ones(5, 2), t.ones(5, dtype=t.long)), batch_size=2)
        train(model, loader, t.nn.CrossEntropyLoss(), optimizer)
        prune_weights(model, .5)
        callback_counts = []
        for epoch in range(2):
            train(model, loader, t.nn.CrossEntropyLoss(), optimizer, epoch=epoch+1,
                  callbacks=[lambda: callback_counts.append(1)])
            module = model[0]
            removed = module.weight_mask == 0
            self.assertTrue(t.all(module.weight_orig[removed] == 0))
            self.assertTrue(t.all(module.weight[removed] == 0))
            self.assertTrue(t.all(optimizer.state[module.weight_orig]['momentum_buffer'][removed] == 0))
        self.assertEqual(len(callback_counts), 6)
        with tempfile.TemporaryDirectory() as folder, patch('breakhis.checkpoints.create_model',
                side_effect=lambda *args, **kwargs: t.nn.Sequential(t.nn.Linear(2, 2))):
            path = Path(folder) / 'model.pth'
            save_checkpoint(path, model, 'vgg', {})
            restored, _ = load_checkpoint(path)
            self.assertTrue(t.equal(restored[0].weight_mask, model[0].weight_mask))
            self.assertTrue(t.allclose(restored(t.ones(2, 2)), model(t.ones(2, 2))))

    def test_legacy_finetuning_uses_callbacks_epoch_and_supplied_scheduler(self):
        from torch.utils.data import DataLoader, TensorDataset
        from breakhis.training import TrainingPrunned
        from breakhis.legacy_utils.PrunUtillFGP import FineGrainedPruner
        t = self.t
        model = t.nn.Linear(2, 2)
        pruner = FineGrainedPruner(model, {'': .5})
        optimizer = t.optim.SGD(model.parameters(), lr=.1)
        scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        loader = DataLoader(TensorDataset(t.ones(2, 2), t.ones(2, dtype=t.long)))
        with patch.object(pruner, 'apply', wraps=pruner.apply) as callback, contextlib.redirect_stdout(io.StringIO()):
            accuracy, best, history = TrainingPrunned(model, loader, loader, t.nn.CrossEntropyLoss(),
                                                    optimizer, pruner, scheduler, num_finetune_epochs=2)
        self.assertEqual(callback.call_count, 4)
        self.assertIsNotNone(best)
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .025)
        self.assertEqual(scheduler.last_epoch, 2)

    def test_channel_shapes_batchnorm_classifier_and_zero_ratio_identity(self):
        from breakhis.models.vgg import VGG
        from breakhis.pruning import prune_vgg_channels
        t = self.t
        model = VGG(channels=[4]*8).eval()
        for index in range(8):
            bn = model.backbone.get_submodule(f'bn{index}')
            with t.no_grad():
                bn.running_mean.copy_(t.arange(4).float())
                bn.running_var.copy_(t.arange(4).float()+1)
                bn.num_batches_tracked.fill_(7)
        inputs = t.randn(2, 3, 32, 32)
        unchanged = prune_vgg_channels(model, 0)
        self.assertTrue(t.allclose(model(inputs), unchanged(inputs), atol=1e-6, rtol=1e-5))
        pruned = prune_vgg_channels(model, .5)
        self.assertEqual(pruned.channels, [2]*8)
        self.assertLess(sum(p.numel() for p in pruned.parameters()), sum(p.numel() for p in model.parameters()))
        for index in range(8):
            conv = pruned.backbone.get_submodule(f'conv{index}')
            bn = pruned.backbone.get_submodule(f'bn{index}')
            retained = pruned.channel_pruning['layers'][f'backbone.conv{index}']['retained_indices']
            self.assertEqual(conv.in_channels, 3 if index == 0 else 2)
            self.assertEqual(bn.num_features, 2)
            self.assertTrue(t.equal(bn.running_mean, model.backbone.get_submodule(f'bn{index}').running_mean[retained]))
            self.assertEqual(bn.num_batches_tracked.item(), 7)
        self.assertEqual(pruned.fc2.in_features, 2)
        pruned.train()
        loss = t.nn.functional.cross_entropy(pruned(inputs), t.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(t.isfinite(p.grad).all() for p in pruned.parameters() if p.grad is not None))

    def test_channel_checkpoint_fresh_process_reload(self):
        from breakhis.models.vgg import VGG
        from breakhis.pruning import prune_vgg_channels
        from breakhis.checkpoints import save_checkpoint, load_checkpoint
        t = self.t
        model = prune_vgg_channels(VGG(channels=[4]*8).eval(), .5)
        inputs = t.randn(2, 3, 32, 32)
        with t.inference_mode(): expected = model(inputs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_checkpoint(root / 'model.pth', model, 'vgg', {})
            t.save(inputs, root / 'inputs.pt')
            script = ('import sys,torch; from pathlib import Path; from breakhis.checkpoints import load_checkpoint; '
                      'torch.set_num_threads(2); root=Path(sys.argv[1]); '
                      'model,payload=load_checkpoint(root/"model.pth"); model.eval(); '
                      'torch.save(model(torch.load(root/"inputs.pt",weights_only=True)).detach(),root/"outputs.pt")')
            result = subprocess.run([sys.executable, '-c', script, str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(t.allclose(t.load(root/'outputs.pt', weights_only=True), expected, atol=1e-6, rtol=1e-5))
            restored, payload = load_checkpoint(root/'model.pth')
            self.assertEqual(restored.channel_pruning, model.channel_pruning)
            self.assertEqual(payload['architecture_config']['channels'], [2]*8)
            payload['pruning']['channel_config']['layers']['backbone.conv0']['retained_indices'] = [99, 100]
            t.save(payload, root/'invalid.pth')
            from breakhis.checkpoints import read_checkpoint
            with self.assertRaisesRegex(ValueError, 'retained channel indices'):
                read_checkpoint(root/'invalid.pth')

    def test_unsupported_channel_models_masks_and_ratios_fail(self):
        from breakhis.models.vgg import VGG
        from breakhis.pruning import prune_vgg_channels, prune_weights
        from breakhis.checkpoints import create_model
        for architecture in ('resnet18',):
            model = create_model(architecture)
            with self.assertRaisesRegex(ValueError, 'residual/depthwise'): prune_vgg_channels(model, .5)
            prune_weights(model, .1)
            logits = model(self.t.randn(2, 3, 32, 32))
            logits.sum().backward()
            self.assertEqual(tuple(logits.shape), (2, 2))
        tiny = VGG(channels=[1]*8)
        with self.assertRaises(ValueError): prune_vgg_channels(tiny, 1)
        pruned = prune_vgg_channels(tiny, .9)
        self.assertEqual(pruned.channels, [1]*8)
        with self.assertRaisesRegex(ValueError, 'Combining channel'): prune_weights(pruned)
        prune_weights(tiny, .5)
        with self.assertRaisesRegex(ValueError, 'unmasked'): prune_vgg_channels(tiny)


if __name__ == '__main__':
    unittest.main()
