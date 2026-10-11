from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from breakhis.manifest import assign_splits, compare_sources, inventory, parse_filename, validate_splits


def sample_rows():
    return [dict(source='one',path=f'{mag}/{label}/{patient}.png',filename=f'{patient}-{mag}.png',
                 label=label,magnification=mag,patient_id=patient,
                 sha256=f'{patient}-{mag}',pixel_sha256=f'{patient}-{mag}')
            for label in ('benign','malignant') for n in range(6)
            for patient in [f'{label}{n}'] for mag in ('40X','100X')]


class ManifestTests(unittest.TestCase):
    def test_official_filename_identifier_preserves_suffix(self):
        a=parse_filename('SOB_B_A-14-22549AB-40-001.png')
        b=parse_filename('SOB_B_A-14-22549AB-400-019.png')
        self.assertEqual(a['patient_id'],'14-22549AB')
        self.assertEqual(a['patient_id'],b['patient_id'])
        self.assertEqual(a['label'],'benign')
        with self.assertRaises(ValueError):
            parse_filename('SOB_M_A-14-22549AB-40-001.png')

    def test_splits_reproducible_and_patient_isolated(self):
        rows=sample_rows()
        a=assign_splits(rows,seed=12)
        self.assertEqual(a,assign_splits(rows,seed=12))
        self.assertNotEqual([r['split'] for r in a],[r['split'] for r in assign_splits(rows,seed=13)])
        self.assertEqual(validate_splits(a)['patient_overlap'],0)
        for split in ('train','validation','test'):
            self.assertEqual({r['label'] for r in a if r['split']==split},{'benign','malignant'})

    def test_identical_pixels_link_different_patient_ids(self):
        rows=sample_rows()
        rows[2]['pixel_sha256']=rows[0]['pixel_sha256']
        result=assign_splits(rows,seed=1)
        self.assertEqual(result[0]['group_id'],result[2]['group_id'])
        self.assertEqual(result[0]['split'],result[2]['split'])
        validate_splits(result)

    def test_overlap_and_conflicting_labels_rejected(self):
        rows=assign_splits(sample_rows())
        rows[1]['split']='test' if rows[0]['split']!='test' else 'train'
        with self.assertRaises(ValueError): validate_splits(rows)
        rows=sample_rows()
        rows[-1]['pixel_sha256']=rows[0]['pixel_sha256']
        with self.assertRaises(ValueError): assign_splits(rows)

    def test_inventory_reports_corrupt_images_and_folder_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            images=root/'40X/benign'
            images.mkdir(parents=True)
            Image.new('RGB',(12,8)).save(images/'SOB_B_A-14-111-40-001.png')
            (images/'SOB_B_A-14-112-40-001.png').write_bytes(b'not an image')
            Image.new('RGB',(12,8)).save(images/'SOB_M_DC-14-113-40-001.png')
            (root/'note.txt').write_text('metadata')
            rows,audit=inventory('one',root,workers=1)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(audit['excluded']),2)
            self.assertEqual(audit['ignored'],['note.txt'])

    def test_source_comparison_detects_changed_content(self):
        row=sample_rows()[0]
        result=compare_sources([dict(row,source='a'),dict(row,source='b',sha256='different',pixel_sha256='different')],'a','b')
        self.assertEqual(result['shared_names'],1)
        self.assertEqual(result['identical_pixels'],0)


if __name__=='__main__': unittest.main()
