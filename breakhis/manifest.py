"""Decode, fingerprint, and split BreaKHis images by filename patient/slide ID."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image

MAGNIFICATIONS = ('40X', '100X', '200X', '400X')
SUBTYPES = {'A': 'adenosis', 'F': 'fibroadenoma', 'PT': 'phyllodes_tumor',
            'TA': 'tubular_adenoma', 'DC': 'ductal_carcinoma', 'LC': 'lobular_carcinoma',
            'MC': 'mucinous_carcinoma', 'PC': 'papillary_carcinoma'}
PATTERN = re.compile(r'^SOB_(B|M)_([A-Z]+)-(\d{2})-([0-9]+[A-Z]*)-(40|100|200|400)-(\d+)\.png$', re.I)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_filename(name):
    match = PATTERN.fullmatch(name)
    if not match or match[2].upper() not in SUBTYPES:
        raise ValueError(f'Unrecognized BreaKHis filename: {name}')
    kind, subtype, year, slide, mag, sequence = match.groups()
    if (kind.upper()=='B') != (subtype.upper() in ('A','F','PT','TA')):
        raise ValueError('Filename tumor class and subtype disagree')
    return dict(label='benign' if kind.upper() == 'B' else 'malignant',
                subtype=SUBTYPES[subtype.upper()], patient_id=f'{year}-{slide.upper()}',
                magnification=f'{mag}X', sequence=sequence)


def inspect_image(item):
    source, root, path = item
    relative = path.relative_to(root).as_posix()
    try:
        metadata = parse_filename(path.name)
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] != metadata['magnification']:
            raise ValueError('Image location and filename magnification disagree')
        expected_folder = metadata['subtype'] if source=='cancer_multiclass' else metadata['label']
        if parts[1] != expected_folder:
            raise ValueError('Image folder and filename label disagree')
        raw = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            rgb = image.convert('RGB')
            width, height = rgb.size
            pixel_hash = hashlib.sha256(f'{width}x{height}:'.encode() + rgb.tobytes()).hexdigest()
        return dict(source=source, path=relative, filename=path.name, **metadata,
                    sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel_hash,
                    width=width, height=height, bytes=len(raw))
    except Exception as error:
        return dict(source=source, path=relative, excluded=True, reason=str(error))


def inventory(source, root, workers=8):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    files = sorted(path for path in root.rglob('*') if path.is_file())
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    image_set = set(images)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(inspect_image, ((source, root, path) for path in images)))
    accepted = [row for row in results if not row.get('excluded')]
    excluded = [row for row in results if row.get('excluded')]
    return accepted, dict(root=str(root), files=len(files), images=len(images),
                         accepted=len(accepted), excluded=excluded,
                         ignored=[p.relative_to(root).as_posix() for p in files if p not in image_set])


def assign_splits(rows, seed=0, fractions=(0.70, 0.15, 0.15)):
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions)-1) > 1e-8:
        raise ValueError('Three positive split fractions must sum to one')
    # Keep exact decoded-image duplicates in the same group, even when names differ.
    parent = {row['patient_id']: row['patient_id'] for row in rows}
    def find(patient):
        while parent[patient] != patient:
            parent[patient] = parent[parent[patient]]
            patient = parent[patient]
        return patient
    by_hash = defaultdict(set)
    for row in rows:
        by_hash[row['pixel_sha256']].add(row['patient_id'])
    for patients in by_hash.values():
        representative = min(patients)
        for patient in patients:
            a, b = find(representative), find(patient)
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['patient_id'])].append(row)
    strata = defaultdict(list)
    for group, images in groups.items():
        labels = {row['label'] for row in images}
        if len(labels) != 1:
            raise ValueError(f'Conflicting binary labels in linked patient group {group}')
        strata[next(iter(labels))].append(group)
    assignments = {}
    rng = random.Random(seed)
    for label, patients in sorted(strata.items()):
        patients.sort()
        rng.shuffle(patients)
        if len(patients) < 3:
            raise ValueError(f'Need at least three independent groups for {label}')
        n_train = max(1, min(len(patients)-2, round(fractions[0]*len(patients))))
        n_val = max(1, min(len(patients)-n_train-1, round(fractions[1]*len(patients))))
        for index, patient in enumerate(patients):
            assignments[patient] = 'train' if index < n_train else 'validation' if index < n_train+n_val else 'test'
    return [dict(row, group_id=find(row['patient_id']), split=assignments[find(row['patient_id'])]) for row in rows]


def validate_splits(rows):
    patient_splits, hash_splits = defaultdict(set), defaultdict(set)
    for row in rows:
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        patient_splits[row['patient_id']].add(row['split'])
        hash_splits[row['pixel_sha256']].add(row['split'])
    if any(len(values)>1 for values in patient_splits.values()):
        raise ValueError('Patient overlap across splits')
    if any(len(values)>1 for values in hash_splits.values()):
        raise ValueError('Duplicate image content crosses splits')
    return dict(patient_overlap=0, duplicate_content_overlap=0)


def summarize(rows):
    counts = Counter((row['source'], row['magnification'], row['split'], row['label']) for row in rows)
    return [dict(source=source, magnification=mag, split=split, label=label, images=count,
                 patients=len({r['patient_id'] for r in rows if (r['source'],r['magnification'],r['split'],r['label']) == (source,mag,split,label)}))
            for (source, mag, split, label), count in sorted(counts.items())]


def compare_sources(rows, first='breakhis_v2', second='cancer_binary'):
    left = {r['filename']:r for r in rows if r['source']==first}
    right = {r['filename']:r for r in rows if r['source']==second}
    shared = sorted(left.keys() & right.keys())
    return dict(shared_names=len(shared), identical_bytes=sum(left[n]['sha256']==right[n]['sha256'] for n in shared),
                identical_pixels=sum(left[n]['pixel_sha256']==right[n]['pixel_sha256'] for n in shared),
                changed_content=[n for n in shared if left[n]['pixel_sha256']!=right[n]['pixel_sha256']],
                only_first=sorted(left.keys()-right.keys()), only_second=sorted(right.keys()-left.keys()))


def prepare_dataset(dataset_root, output, seed=0, workers=8):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    sources = {'breakhis_v2':dataset_root/'BreaKHis_v2',
               'cancer_binary':dataset_root/'dataset_cancer_v1'/'classificacao_binaria',
               'cancer_multiclass':dataset_root/'dataset_cancer_v1'/'classificacao_multiclasse'}
    rows, audits = [], {}
    for source, root in sources.items():
        print(f'Auditing {source}: {root}', flush=True)
        accepted, audit = inventory(source, root, workers)
        rows.extend(accepted)
        audits[source] = audit
        print(f'  accepted {len(accepted)}; excluded {len(audit["excluded"])}', flush=True)
    rows = assign_splits(rows, seed)
    integrity = validate_splits(rows)
    # Preserve every cross-source entry, but omit repeated decoded images within each source.
    seen, unique, duplicates = set(), [], []
    for row in rows:
        key = (row['source'],row['pixel_sha256'])
        if key in seen:
            duplicates.append(dict(source=row['source'],path=row['path'],reason='duplicate decoded content'))
        else:
            seen.add(key)
            unique.append(row)
    comparison = compare_sources(rows)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, seed=seed, fractions=[.70,.15,.15],
                    patient_rule='YEAR-SLIDE_ID from official filename; subtype and magnification excluded',
                    roots={key:str(value) for key,value in sources.items()}, rows=unique)
    report = dict(audits=audits, comparison=comparison, duplicate_exclusions=duplicates,
                  integrity=integrity, patient_groups=len({r['patient_id'] for r in rows}),
                  split_groups=len({r['group_id'] for r in rows}), distributions=summarize(unique))
    report['dimensions'] = {source:{f'{width}x{height}':count for (width,height),count in
                           Counter((r['width'],r['height']) for r in rows if r['source']==source).items()}
                           for source in sources}
    report['patient_distributions'] = {split:dict(Counter(label for _,label in
        {(r['patient_id'],r['label']) for r in unique if r['split']==split}))
        for split in ('train','validation','test')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(integrity=integrity, shared_names=comparison['shared_names'],
                          changed_content=len(comparison['changed_content']),
                          only_second=len(comparison['only_second']),
                          patient_groups=report['patient_groups']),indent=2), flush=True)
    return manifest, report
