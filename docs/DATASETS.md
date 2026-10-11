# Dataset audit and patient-separated splits

## Audited local sources

The audit decodes every supported image, checks filename metadata against its
folder, computes SHA-256 hashes for the original file and decoded RGB pixels,
and records the accepted images in a saved manifest. It does not edit dataset files.

| Source argument | Local dataset root | Images decoded | Unique decoded images used |
| --- | --- | ---: | ---: |
| `breakhis_v2` | `../Dataset/BreaKHis_v2` | 7,900 | 7,775 |
| `cancer_binary` | `../Dataset/dataset_cancer_v1/classificacao_binaria` | 7,909 | 7,784 |
| Audit only | `../Dataset/dataset_cancer_v1/classificacao_multiclasse` | 7,909 | 7,784 |

No decoding failures were found in this local audit. Each source contains 125
repeated decoded images; the manifest excludes subsequent occurrences within
that source. Cross-source entries are retained because source selection is
explicit, and the sources are not concatenated for primary training.

The binary roots have 7,900 filenames in common, but **none of those shared images
have identical decoded pixels**. `dataset_cancer_v1` contains 224×224 versions;
`BreaKHis_v2` has mostly 700×460 images, with a smaller set at 700×456.
The cancer dataset has nine additional named images. Their names and all changed
content entries are saved in `artifacts/data/audit.json`.

## Identifier rule

The [official BreaKHis dataset description](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
documents procedure, tumor class, subtype, year, slide identifier, magnification,
and sequence in the image filename. The grouping rule uses the full `YEAR-SLIDE_ID`:

```text
SOB_B_A-14-22549AB-40-001.png
        └─ 14-22549AB is the patient/slide group
```

The subtype, magnification, and image sequence are excluded from the group key.
Letter suffixes are preserved. The same identifier is grouped together even when
it appears under more than one subtype. Local filenames yield **81 patient/slide
groups**; this is the local filename census, not an independently verified count
of distinct people. No external patient-to-slide metadata is present in the repo.

Exact decoded-image duplicates also link groups before splitting, so matching
content cannot appear in different splits. Conflicting binary labels within
linked groups fail the audit rather than silently choosing a label.

## Split policy

The saved seed defaults to zero. Groups are stratified by binary label, shuffled
with a local seeded generator, and allocated approximately **70% training,
15% validation, and 15% test by group count**. Image proportions differ because
patients have different numbers of images. Class/image distributions are reported
for each source, magnification, and split.

All magnifications and all local source copies use consistent assignments for a
patient/slide identifier. `--magnification all` therefore preserves group isolation.
The audit validates zero patient overlap and zero exact decoded-content overlap
across splits. It does not establish that visually similar, nonidentical images
are unrelated; the filename identifier remains the patient-group reference.

## Prepare and reuse

From the repository directory, in the activated environment:

```powershell
python -m breakhis prepare-data --seed 0
```

Outputs:

- `artifacts/data/manifest.json`: source roots, seed, split fractions, patient-group
  rule, image paths, labels, magnifications, file/pixel hashes, and split assignments.
- `artifacts/data/audit.json`: inventory results, exclusions, source comparison,
  dimensions, and class/patient distributions.

These generated files are ignored by Git. Keep the manifest with your experiment
artifacts to reproduce its split. Rebuilding is deterministic for the same data
and seed, but changed data creates a new experiment manifest.

```powershell
python -m breakhis train --dataset breakhis --source breakhis_v2 --magnification 40X
python -m breakhis train --dataset breakhis --source cancer_binary --magnification 40X
python -m breakhis train --dataset breakhis --source breakhis_v2 --magnification all
```

Use `--manifest` for a different saved manifest. `--data-root` relocates the selected
source without changing its split. The loader verifies each image's file hash
against the audit while loading; altered files fail with an instruction to rebuild.
Missing manifests fail clearly rather than falling back to image-level random splits.

Training and fine-tuning select weights using **validation**. The `evaluate`
command alone reads the reserved test split for final metrics. Repeated use of
test results to tune experiments would still compromise that independence.

The eight-class inventory is audited for comparison; primary commands remain
binary. Its subtype field is preserved for a future eight-class extension.
