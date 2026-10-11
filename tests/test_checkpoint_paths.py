from pathlib import Path
import unittest

from breakhis.paths import CHECKPOINT_ROOT, checkpoint_directory, checkpoint_output


class CheckpointPathTests(unittest.TestCase):
    def test_defaults_are_outside_codebase_and_separate_tasks(self):
        workspace = Path(__file__).resolve().parents[2]
        self.assertEqual(CHECKPOINT_ROOT, workspace / 'checkpoint')
        for source, task in [('breakhis_v2', 'binary'), ('cancer_binary', 'binary'),
                             ('cancer_multiclass', 'multiclass')]:
            self.assertEqual(checkpoint_directory(source), workspace / 'checkpoint' / task)
            for command in ('train', 'prune', 'finetune'):
                self.assertEqual(checkpoint_output(source, '40X', 'vgg', command),
                                 workspace / 'checkpoint' / task / '40X' / 'vgg' / f'{command}.pth')

    def test_custom_filename_and_absolute_path(self):
        root = checkpoint_directory('cancer_binary').resolve()
        for output in ('baseline.pth', root / 'baseline.pth'):
            self.assertEqual(checkpoint_output('cancer_binary', '40X', 'vgg', 'train', output),
                             root / 'baseline.pth')

    def test_rejects_outputs_outside_task_and_non_checkpoint_files(self):
        for output in (CHECKPOINT_ROOT / 'multiclass' / 'wrong.pth',
                       CHECKPOINT_ROOT / 'binary' / '..' / 'wrong.pth', 'wrong.json'):
            with self.assertRaises(ValueError):
                checkpoint_output('cancer_binary', '40X', 'vgg', 'train', output)


if __name__ == '__main__':
    unittest.main()
