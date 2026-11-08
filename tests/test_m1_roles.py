"""M1-T05--T08: full-source linkage and role-isolated real image access."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis import manifest as m, data as d, experiments as e, paths

ROOT=Path(__file__).resolve().parents[1]


class RoleTests(unittest.TestCase):
    def setUp(self):
        import torch
        threads=torch.get_num_threads(); torch.set_num_threads(2); self.addCleanup(torch.set_num_threads,threads)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.images=self.root/'images'
        rows=[]
        for label,code,offset in [('benign','B_A',1000),('malignant','M_DC',2000)]:
            for index in range(6):
                slide=str(offset+index)
                if label=='benign' and index<2: slide='1000'+('A' if index==0 else 'B')
                for mag in ('40','100'):
                    file=self.images/f'{mag}X/{label}/SOB_{code}-14-{slide}-{mag}-001.png'
                    file.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(10,8),((offset//10+index*13)%256,int(mag),index*31)).save(file)
                    rows.append(m.inspect_image(('cancer_binary',self.images,file)))
        self.rows=m.assign_splits(rows,seed=101)
        self.file=self.root/'manifest.json'; self.save()
        self.config=e.load_config(ROOT/'configs/experiments/vgg_baseline.json',device='cpu')
        self.config['data'].update(manifest=str(self.file),data_root=str(self.images))

    def save(self):
        self.file.write_text(json.dumps(dict(schema_version=1,seed=101,
            roots={'cancer_binary':str(self.images)},rows=self.rows)))

    def pick(self,split,label='benign',mag='40X'):
        return next(row for row in self.rows if row['split']==split and row['label']==label and row['magnification']==mag)

    def test_t05_seed_reproducibility_suffixes_and_manifest_reuse(self):
        first=m.assign_splits(copy.deepcopy(self.rows),seed=101)
        self.assertEqual(first,m.assign_splits(copy.deepcopy(self.rows),seed=101))
        self.assertIn('14-1000A',{r['patient_id'] for r in first})
        self.assertIn('14-1000B',{r['patient_id'] for r in first})
        before=self.file.read_bytes()
        config=copy.deepcopy(self.config); config['training']['seed']=999
        e.validate_data(config,splits=('train','validation'))
        self.assertEqual(self.file.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'already exists'): m.prepare_dataset(self.root,self.root,workers=1)

    def test_t06_hidden_magnification_leakage_refused_before_selection(self):
        row=self.pick('train',mag='100X'); row['split']='test'; row['group_id']='fake'
        self.save()
        with self.assertRaisesRegex(ValueError,'overlap'): d.get_split_datasets(self.file,'cancer_binary','40X',splits=('train',))

    def test_t06_raw_and_pixel_leakage_cannot_hide_behind_group_ids(self):
        original=copy.deepcopy(self.rows)
        for field in ('sha256','pixel_sha256'):
            self.rows=copy.deepcopy(original)
            self.pick('validation')[field]=self.pick('train')[field]; self.save()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Duplicate'):
                e.validate_data(self.config,splits=('train','validation'))

    def test_t06_transitive_component_membership_is_recomputed(self):
        # Link two groups through separate raw/pixel edges inside the same role.
        train=[r for r in self.rows if r['split']=='train' and r['label']=='benign']
        groups=sorted({r['patient_id'] for r in train})[:3]
        a=next(r for r in train if r['patient_id']==groups[0] and r['magnification']=='40X')
        b=next(r for r in train if r['patient_id']==groups[1] and r['magnification']=='40X')
        c=next(r for r in train if r['patient_id']==groups[2] and r['magnification']=='40X')
        b['pixel_sha256']=a['pixel_sha256']; c['sha256']=b['sha256']; self.save()
        with self.assertRaisesRegex(ValueError,'membership'): e.validate_data(self.config,splits=('train',))

    def test_t06_false_component_merge_rejected(self):
        groups=sorted({r['patient_id'] for r in self.rows if r['split']=='train' and r['label']=='benign'})[:2]
        for row in self.rows:
            if row['patient_id'] in groups: row['group_id']='false-merged-label'
        self.save()
        with self.assertRaisesRegex(ValueError,'membership'): e.validate_data(self.config,splits=('train',))

    def test_t06_component_alias_can_survive_duplicate_representative_exclusion(self):
        # Stored labels need not equal the minimum *retained* filename: compare partitions.
        for row in self.rows:
            if row['patient_id']=='14-1000A': row['group_id']='14-0999'
        self.save(); e.validate_data(self.config,splits=('train','validation'))

    def test_t07_bad_hash_or_filename_metadata_in_other_magnification_rejected(self):
        original=copy.deepcopy(self.rows)
        for field,value in [('sha256','bad'),('subtype','wrong'),('patient_id','14-WRONG'),('width',True)]:
            self.rows=copy.deepcopy(original); self.pick('train',mag='100X')[field]=value; self.save()
            with self.subTest(field=field),self.assertRaises(ValueError): e.validate_data(self.config,splits=('train',))

    def test_t07_case_collision_and_unsafe_path_rejected(self):
        row=copy.deepcopy(self.rows[0]); row['path']=row['path'].upper(); self.rows.append(row); self.save()
        with self.assertRaisesRegex(ValueError,'case-colliding'): e.validate_data(self.config)
        self.rows.pop(); self.rows[0]['path']='../outside.png'; self.save()
        with self.assertRaisesRegex(ValueError,'Unsafe'): e.validate_data(self.config)

    def test_t07_missing_test_files_do_not_block_development(self):
        for row in self.rows:
            if row['split']=='test': (self.images/row['path']).unlink()
        e.validate_data(self.config,splits=('train','validation'))
        loaders=d.get_split_loaders(self.file,'cancer_binary','40X',batch_size=2,splits=('train','validation'))
        for loader in loaders.values(): list(loader)
        with self.assertRaisesRegex(ValueError,'missing image in test'):
            d.get_split_datasets(self.file,'cancer_binary',splits=('test',))

    def test_t07_missing_class_is_checked_for_requested_role_only(self):
        self.rows=[r for r in self.rows if not (r['split']=='test' and r['label']=='malignant')]; self.save()
        e.validate_data(self.config,splits=('train','validation'))
        with self.assertRaisesRegex(ValueError,'Both classes.*test'): e.validate_data(self.config,splits=('test',))

    def test_t07_changed_corrupt_and_wrong_pixel_bytes_name_the_role(self):
        dataset=d.get_split_datasets(self.file,'cancer_binary',splits=('validation',))['validation']
        row=dataset.rows[0]; file=self.images/row['path']; original=file.read_bytes()
        file.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'validation: Image changed'): dataset[0]
        row['sha256']=hashlib.sha256(b'changed').hexdigest()
        with self.assertRaisesRegex(ValueError,'validation: cannot decode'): dataset[0]
        file.write_bytes(original); row['sha256']=hashlib.sha256(original).hexdigest(); row['pixel_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'validation: decoded pixels'): dataset[0]

    def test_t08_development_factories_and_pixels_never_touch_test(self):
        original=d.ManifestDataset; seen=[]
        def factory(root,rows,transform):
            role=rows[0]['split']; self.assertNotEqual(role,'test'); seen.append(role)
            return original(root,rows,transform)
        with patch.object(d,'ManifestDataset',side_effect=factory):
            loaders=d.get_split_loaders(self.file,'cancer_binary',splits=('train','validation'),batch_size=4)
            for loader in loaders.values(): list(loader)
        self.assertEqual(set(seen),{'train','validation'})

    def test_t08_final_constructs_test_only_without_training_files(self):
        for row in self.rows:
            if row['split']!='test': (self.images/row['path']).unlink()
        original=d.ManifestDataset; seen=[]
        def factory(root,rows,transform):
            role=rows[0]['split']; self.assertEqual(role,'test'); seen.append(role)
            return original(root,rows,transform)
        with patch.object(d,'ManifestDataset',side_effect=factory):
            loaders=d.get_split_loaders(self.file,'cancer_binary',splits=('test',),batch_size=4)
            list(loaders['test'])
        self.assertEqual(seen,['test'])

    def test_t08_invalid_requested_roles_rejected(self):
        for roles in ((),('test','test'),('outer_test',),'train',(True,)):
            with self.subTest(roles=roles),self.assertRaises(ValueError):
                d.get_split_datasets(self.file,'cancer_binary',splits=roles)

    def test_t08_actual_cli_train_and_evaluate_route_only_requested_roles(self):
        import torch
        from breakhis.cli import main
        from breakhis.checkpoints import read_checkpoint
        def model(): return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,2))
        original=d.ManifestDataset; seen=[]
        def factory(root,rows,transform):
            seen.append(rows[0]['split']); return original(root,rows,transform)
        common=['--dataset','breakhis','--manifest',str(self.file),'--source','cancer_binary','--device','cpu','--batch-size','4']
        with patch.object(paths,'CHECKPOINT_ROOT',self.root/'checkpoint'),patch.object(d,'ManifestDataset',side_effect=factory),contextlib.redirect_stdout(io.StringIO()):
            with patch('breakhis.checkpoints.create_model',side_effect=lambda *a,**k:model()):
                main(['train',*common,'--epochs','1','--experiment-id','routing'])
            self.assertEqual(set(seen),{'train','validation'})
            checkpoint=self.root/'checkpoint/binary/40X/vgg/routing/train.pth'; payload=read_checkpoint(checkpoint)
            for split in ('validation','test'):
                seen.clear(); output=self.root/f'{split}.json'
                with patch('breakhis.checkpoints.load_checkpoint',return_value=(model(),payload)):
                    main(['evaluate',*common,'--checkpoint',str(checkpoint),'--split',split,'--output',str(output),'--benchmark-warmup','0','--benchmark-repeats','1'])
                self.assertEqual(seen,[split]); self.assertEqual(json.loads(output.read_text())['split'],split)


if __name__=='__main__': unittest.main()
