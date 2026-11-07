"""Actual generated-image, fresh-process CPU acceptance regression."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires the CPU training environment')
class EndToEndTests(unittest.TestCase):
    def test_generated_images_through_audit_training_pruning_finetuning_and_reports(self):
        import torch
        from tools.smoke_workflow import run_smoke
        from breakhis.paths import checkpoint_directory
        from breakhis.checkpoints import read_checkpoint
        from breakhis.manifest import validate_splits
        root = checkpoint_directory('cancer_binary')
        root.mkdir(parents=True, exist_ok=True)
        # Temporary checkpoints still obey the workspace's parent/task storage policy.
        with tempfile.TemporaryDirectory() as images, tempfile.TemporaryDirectory(prefix='test-smoke-', dir=root) as checkpoints:
            work = Path(os.environ['BREAKHIS_SMOKE_EVIDENCE']) if os.environ.get('BREAKHIS_SMOKE_EVIDENCE') else Path(images)/'smoke'
            summary = run_smoke(work, Path(checkpoints))
            self.assertEqual(summary['status'], 'passed')
            manifest = json.loads(Path(summary['manifest']).read_text())
            self.assertEqual(set(manifest['roots']), {'cancer_binary'})
            self.assertEqual(validate_splits(manifest['rows'])['patient_overlap'], 0)
            self.assertEqual({row['split'] for row in manifest['rows']}, {'train','validation','test'})
            dense = read_checkpoint(Path(checkpoints)/'train.pth')
            self.assertEqual(dense['epoch'], 1)
            for method, filename in summary['reports'].items():
                report = json.loads(Path(filename).read_text())
                self.assertEqual(report['split'], 'test')
                self.assertEqual(report['models']['dense']['benchmark']['input_shape'], [1,3,224,224])
                self.assertEqual(report['models']['finetuned']['image_metrics']['samples'], 2)
                self.assertEqual(report['models']['finetuned']['patient_metrics']['samples'], 2)
                self.assertEqual(report['models']['dense']['benchmark']['runtime']['device'], 'cpu')
                before = read_checkpoint(Path(checkpoints)/method/'prune.pth')
                after = read_checkpoint(Path(checkpoints)/method/'finetune.pth')
                if method == 'unstructured':
                    for key, mask in before['state_dict'].items():
                        if key.endswith('.weight_mask'):
                            self.assertTrue(torch.equal(mask, after['state_dict'][key]))
                            self.assertTrue(torch.all(after['state_dict'][key.removesuffix('_mask')+'_orig'][mask == 0] == 0))
                else:
                    self.assertEqual(before['pruning'], after['pruning'])
                    self.assertLess(report['models']['pruned']['size']['logical_parameters'],report['models']['dense']['size']['logical_parameters'])
            self.assertEqual(len(list((work/'logs').glob('*.log'))), 8)


if __name__ == '__main__': unittest.main()
