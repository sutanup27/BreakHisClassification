"""Experiment contract, leakage and aggregation checks; real pipeline smoke is separate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from breakhis import experiments as e

ROOT=Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=e.read_json(ROOT/'configs/experiments/vgg_baseline.json')
        self.images=self.root/'images'; self.images.mkdir()
        self.config['data'].update(manifest=str(self.root/'manifest.json'),data_root=str(self.images))
        self.file=self.root/'config.json'
        self.rows=[]
        for split in ('train','validation','test'):
            for label in ('benign','malignant'):
                from breakhis.manifest import inspect_image
                code='B_A' if label=='benign' else 'M_DC'
                file=self.images/f'40X/{label}/SOB_{code}-14-{1000+len(self.rows)}-40-001.png'
                file.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(8,8),(len(self.rows)*30,10,20)).save(file)
                row=inspect_image(('cancer_binary',self.images,file))
                self.rows.append(dict(row,group_id=row['patient_id'],split=split))
        self.save()

    def save(self):
        self.file.write_text(json.dumps(self.config))
        (self.root/'manifest.json').write_text(json.dumps(dict(schema_version=1,
            roots={'cancer_binary':str(self.images)},rows=self.rows)))

    def test_config_paths_and_overrides_are_resolved_and_recordable(self):
        config=e.load_config(self.file,seed=77,device='cpu')
        self.assertEqual(config['training']['seed'],77)
        self.assertEqual(config['training']['device'],'cpu')
        self.assertEqual(config['data']['manifest'],str(self.root/'manifest.json'))
        _,counts=e.validate_data(config)
        self.assertEqual(counts['test']['images'],2)

    def test_unknown_fields_bad_types_and_nonfinite_refuse_before_run(self):
        for section,field,value in [('training','seed',True),('training','num_workers',2),
            ('training','learning_rate',float('nan')),('evaluation','repeats',0),('training','typo',1)]:
            with self.subTest(field=field):
                config=copy.deepcopy(self.config); config[section][field]=value
                self.file.write_text(json.dumps(config))
                with self.assertRaises(ValueError): e.load_config(self.file)

    def test_duplicate_json_keys_rejected(self):
        self.file.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): e.read_json(self.file)

    def test_leakage_and_missing_class_fail_before_training(self):
        self.rows[2]['patient_id']=self.rows[0]['patient_id']; self.save()
        with self.assertRaisesRegex(ValueError,'overlap'): e.validate_data(e.load_config(self.file))
        self.rows[2]['patient_id']='other'; self.rows[2]['pixel_sha256']=self.rows[0]['pixel_sha256']; self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate'): e.validate_data(e.load_config(self.file))

    def test_missing_selected_class_cannot_silently_get_zero_recall(self):
        self.rows.pop(); self.save()
        with self.assertRaisesRegex(ValueError,'Both classes'): e.validate_data(e.load_config(self.file))

    def test_group_link_leakage_rejected(self):
        self.rows[2]['group_id']=self.rows[0]['group_id']; self.save()
        with self.assertRaisesRegex(ValueError,'Linked group'): e.validate_data(e.load_config(self.file))

    def test_paths_outside_data_root_rejected(self):
        self.rows[0]['path']='../escape.png'; self.save()
        with self.assertRaisesRegex(ValueError,'Unsafe'): e.validate_data(e.load_config(self.file))

    def test_channel_pruning_rejects_unsupported_models(self):
        self.config['training']['model']='resnet18'
        self.config['pruning']=dict(method='channel',sparsity=.5,finetune_epochs=1); self.save()
        with self.assertRaisesRegex(ValueError,'VGG'): e.load_config(self.file)

    def test_magnification_balance_is_not_image_weighted(self):
        from breakhis.reporting import magnification_balanced_metrics,grouped_patient_metrics
        rows=[dict(patient_id='a',magnification='40X') for _ in range(3)]+[dict(patient_id='a',magnification='100X')]
        probabilities=[[.9,.1]]*3+[[0.,1.]]
        primary=magnification_balanced_metrics(rows,[1]*4,probabilities,['benign','malignant'])
        secondary=grouped_patient_metrics(rows,[1]*4,probabilities,['benign','malignant'])
        self.assertEqual(primary['sensitivity'],1.)
        self.assertEqual(secondary['sensitivity'],0.)
        self.assertAlmostEqual(primary['groups'][0]['probabilities'][1],.55)

    def test_existing_run_refuses_instead_of_overwrite(self):
        with patch.object(e,'RUNS',self.root/'runs'):
            (self.root/'runs/occupied').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'exists'): e.run_experiment(self.file,run_id='occupied',device='cpu')

    def test_resume_requires_original_id(self):
        with self.assertRaisesRegex(ValueError,'original'): e.run_experiment(self.file,resume=True,device='cpu')

    def test_prepare_refuses_overwriting_saved_split(self):
        from breakhis.manifest import prepare_dataset
        with self.assertRaisesRegex(ValueError,'already exists'): prepare_dataset(self.root,self.root,workers=1)


if __name__=='__main__': unittest.main()
