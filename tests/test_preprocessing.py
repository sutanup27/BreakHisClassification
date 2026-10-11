"""Actual image-transform checks; skipped when training dependencies are absent."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'torchvision', 'PIL'))


@unittest.skipUnless(AVAILABLE, 'Requires PyTorch, torchvision, and Pillow')
class PreprocessingTests(unittest.TestCase):
    def test_output_size_for_both_transforms(self):
        from PIL import Image
        from DataPreprocessing import train_transform, test_transform
        for size in [(700, 460), (460, 700), (128, 128)]:
            image = Image.new('RGB', size, color=(100, 150, 200))
            for transform in (train_transform, test_transform):
                self.assertEqual(tuple(transform(image).shape), (3, 224, 224))

    def test_evaluation_is_deterministic_and_resize_preserves_ratio(self):
        from PIL import Image
        import torch
        from DataPreprocessing import test_transform
        image = Image.new('RGB', (700, 460), color=(100, 150, 200))
        self.assertTrue(torch.equal(test_transform(image), test_transform(image)))
        resized = test_transform.transforms[0](image)
        self.assertEqual(min(resized.size), 256)
        self.assertAlmostEqual(resized.width / resized.height, 700 / 460, delta=0.01)


if __name__ == '__main__':
    unittest.main()
