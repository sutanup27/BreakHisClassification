"""Epoch-boundary resume must match an uninterrupted stochastic training run."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class ResumeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def assert_state_equal(self, left, right):
        if isinstance(left, self.t.Tensor):
            self.assertTrue(self.t.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_dense_and_pruned_resume_match_uninterrupted_weights_optimizer_scheduler_and_rng(self):
        import numpy as np
        from torch.utils.data import DataLoader, Dataset
        from breakhis.cli import main
        from breakhis.data import set_seed
        from breakhis.training import train
        from breakhis.checkpoints import save_checkpoint, read_checkpoint
        t = self.t
        class StochasticDataset(Dataset):
            classes = ['benign', 'malignant']
            def __len__(self): return 7
            def __getitem__(self, index):
                return t.tensor([index / 7, random.random(), float(np.random.rand()), t.rand(()).item()]), index % 2
        def loaders(*args, **kwargs):
            set_seed(0, deterministic=True)
            return {split: DataLoader(StochasticDataset(), batch_size=3, shuffle=split == 'train')
                    for split in ('train', 'validation', 'test')}
        def create(*args, **kwargs):
            return t.nn.Sequential(t.nn.Linear(4, 4), t.nn.Dropout(.4), t.nn.Linear(4, 2))
        def interrupted_train(*args, **kwargs):
            value = train(*args, **kwargs)
            if kwargs.get('epoch') == 2:
                raise KeyboardInterrupt
            return value
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'roots': {'cancer_binary': folder}, 'rows': []}))
            metadata = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            source='cancer_binary', magnification='40X',
                            preprocessing='resize-short-edge-256/crop224/ImageNet-normalization')
            for pruned in (False, True):
                with self.subTest(pruned=pruned), \
                     patch('breakhis.paths.CHECKPOINT_ROOT', root / 'checkpoint'), \
                     patch('breakhis.data.get_split_loaders', side_effect=loaders), \
                     patch('breakhis.checkpoints.create_model', side_effect=create), \
                     contextlib.redirect_stdout(io.StringIO()):
                    command = 'finetune' if pruned else 'train'
                    extra = []
                    if pruned:
                        set_seed(0)
                        model = create()
                        from torch.nn.utils import prune
                        prune.l1_unstructured(model[0], 'weight', amount=.5)
                        initial = root / 'pruned.pth'
                        save_checkpoint(initial, model, 'vgg', metadata)
                        extra = ['--checkpoint', str(initial)]
                    common = [command, '--dataset', 'breakhis', '--manifest', str(manifest),
                              '--device', 'cpu', '--epochs', '3', '--batch-size', '3', '--deterministic']
                    main([*common, *extra, '--experiment-id', f'full-{command}'])
                    with patch('breakhis.training.train', side_effect=interrupted_train):
                        main([*common, *extra, '--experiment-id', f'resume-{command}'])
                    directory = root / f'checkpoint/binary/40X/vgg/resume-{command}'
                    latest = directory / f'{command}-last.pth'
                    boundary = read_checkpoint(latest)
                    self.assertEqual(boundary['epoch'], 1)
                    partial = read_checkpoint(directory / f'{command}-interrupted.pth')
                    self.assertIsNone(partial['training_state'])
                    # Omitted options must inherit the saved effective configuration.
                    main([command, '--dataset', 'breakhis', '--resume', str(latest)])
                    resumed = read_checkpoint(latest)
                    full = read_checkpoint(root / f'checkpoint/binary/40X/vgg/full-{command}/{command}-last.pth')
                    self.assertEqual(resumed['epoch'], 3)
                    self.assert_state_equal(full['state_dict'], resumed['state_dict'])
                    self.assert_state_equal(full['training_state'], resumed['training_state'])
                    if pruned:
                        self.assertEqual(resumed['pruning']['masked_modules'], ['0'])
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([command, '--dataset', 'breakhis', '--resume', str(latest), '--epochs', '4'])

    def test_missing_checkpoint_and_invalid_experiment_fail_before_data_loading(self):
        from breakhis.cli import main
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            manifest.write_text('{}')
            for args in (['evaluate', '--checkpoint', str(Path(folder) / 'missing.pth')],
                         ['train', '--manifest', str(manifest), '--experiment-id', '../escape']):
                with self.subTest(args=args), patch('breakhis.data.get_split_loaders') as loader:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([*args, '--dataset', 'breakhis'])
                    loader.assert_not_called()

    def test_format_one_remains_readable_but_not_resumable(self):
        from breakhis.checkpoints import read_checkpoint, restore_training_state
        t = self.t
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.pth'
            t.save(dict(format_version=1, architecture='vgg', classes=['benign', 'malignant'],
                        state_dict={'weight': t.ones(1)}, metadata={}, epoch=1), path)
            payload = read_checkpoint(path)
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                restore_training_state(payload, None, None, {})

    def test_invalid_label_mapping_and_nonfinite_checkpoint_weights_rejected(self):
        from breakhis.checkpoints import save_checkpoint, read_checkpoint
        t = self.t
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pth'
            save_checkpoint(path, t.nn.Linear(2, 2), 'vgg', {})
            payload = t.load(path, weights_only=True)
            payload['class_to_idx'] = {'benign': 1, 'malignant': 0}
            t.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'label mapping'):
                read_checkpoint(path)
            payload['class_to_idx'] = {'benign': 0, 'malignant': 1}
            payload['state_dict']['weight'][0, 0] = float('nan')
            t.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'nonfinite weights'):
                read_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
