import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

AVAILABLE=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(AVAILABLE,'Requires PyTorch')
class LoadingTests(unittest.TestCase):
    def test_manifest_reuse_and_changed_images_rejected(self):
        from breakhis.data import get_split_datasets
        from breakhis.manifest import inventory,assign_splits
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'images'
            for label,code in [('benign','B_A'),('malignant','M_DC')]:
                directory=root/'40X'/label
                directory.mkdir(parents=True)
                for n in range(6):
                    Image.new('RGB',(20,16),(n*25,50 if label=='benign' else 150,200)).save(
                        directory/f'SOB_{code}-14-{100+n if label=="benign" else 200+n}-40-001.png')
            rows,_=inventory('breakhis_v2',root,workers=1)
            rows=assign_splits(rows)
            manifest=Path(folder)/'manifest.json'
            manifest.write_text(json.dumps(dict(schema_version=1,roots={'breakhis_v2':str(root)},rows=rows)))
            first=get_split_datasets(manifest)
            second=get_split_datasets(manifest)
            self.assertEqual(first['train'].rows,second['train'].rows)
            self.assertEqual(first['train'][0][0].shape,(3,224,224))
            changed=root/first['train'].rows[0]['path']
            changed.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed since audit'): first['train'][0]

    def test_missing_manifest_does_not_fall_back_to_random_split(self):
        from breakhis.data import get_split_datasets
        with self.assertRaisesRegex(ValueError,'prepare-data'):
            get_split_datasets('missing-patient-manifest.json')

    def test_pruned_checkpoint_roundtrip(self):
        import torch
        from torch.nn.utils import prune
        from breakhis.checkpoints import create_model,load_checkpoint,save_checkpoint
        previous_threads=torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,previous_threads)
        model=create_model('vgg').eval()
        prune.l1_unstructured(model.backbone.conv0,'weight',amount=.5)
        inputs=torch.randn(1,3,32,32)
        with torch.inference_mode(): expected=model(inputs)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pth'
            save_checkpoint(path,model,'vgg',{'source':'synthetic'})
            restored,_=load_checkpoint(path)
            restored.eval()
            with torch.inference_mode(): actual=restored(inputs)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
            self.assertTrue(torch.equal(restored.backbone.conv0.weight_mask,model.backbone.conv0.weight_mask))


if __name__=='__main__': unittest.main()
