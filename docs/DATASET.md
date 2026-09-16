# SimoLFA Dataset Release Record

## Dataset identity

- Dataset name: SimoLFA Dataset
- Internal source label: POCTDataset_rescreened_v1
- Data unit: image patch
- Classes: `UT = 0`, `T = 1`
- Image format: PNG, RGB
- Model input size: 224 × 224

## Current composition

| Partition | UT | T | Total |
|---|---:|---:|---:|
| Train | 12,584 | 12,584 | 25,168 |
| Validation | 2,097 | 2,097 | 4,194 |
| Test | 2,027 | 2,027 | 4,054 |
| Total | 16,708 | 16,708 | 33,416 |

## Image processing

The source workflow creates patches using a 5 × 5 grid. The final row and column receive any remainder pixels when the source dimensions are not divisible by five. RGB inputs are resized with bilinear interpolation and normalized using ImageNet statistics. Reference maps are generated from grayscale images using `mean - K × standard_deviation`, binary inversion through `THRESH_BINARY_INV`, and nearest-neighbor resizing.

## Integrity audit

The post-replacement audit on 2026-09-08 decoded all 33,416 images successfully. It found:

- zero unreadable images;
- zero copy-suffixed filenames;
- zero unmatched patch filenames;
- zero duplicate SHA-256 groups anywhere in the current dataset;
- zero cross-partition exact duplicate groups;
- zero cross-partition equal difference-hash candidate groups.

The detailed audit is in the research workspace result directory and contains internal filenames. It is not automatically public data.

## Sampling and partitioning status

The current records establish fixed directory partitions and balanced class counts. They do not provide sufficient evidence to claim documented stratified random sampling, patient-level independence, or source-image-level independence. The filename-derived source-group analysis is a provenance heuristic until the data-generation records define the meaning of the filename prefix.

## Access and publication status

The image files are included in this repository-preparation package. Public deposition requires confirmation of:

1. ethics approval and approval number;
2. consent or an approved waiver;
3. removal of identifying metadata and filename content;
4. institutional ownership and distribution permission;
5. an open or controlled-access licence;
6. an appropriate repository with a persistent identifier;
7. anonymous peer-review access where required.

Clinical 920, Clinical Legacy, and Clinical 2026-07-25 are separate restricted assets and are excluded from the planned public dataset release.

## Reuse conditions

Until the access conditions and licence are finalized, the presence of local image files must not be interpreted as authorization for redistribution or public reuse.
