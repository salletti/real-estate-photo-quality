# Dataset Pipeline

The dataset is built from real estate photos annotated with binary quality labels.

There are two supported workflows:

- Manual annotation of real photos
- Synthetic generation from clean source images

## Directory Layout

```text
backend/data/
├── raw_images/
├── source_images/
├── images/
├── dataset.csv
└── model.pth
```

`data/images/` and `data/dataset.csv` are used by the training script.

## Label Format

`backend/data/dataset.csv` uses one row per image:

```csv
image_name,room_type,blurry,low_light,cluttered,bad_framing,tilted,poor_space_visibility,watermark
abc123.jpg,living_room,0,0,1,0,0,0,0
def456.jpg,bedroom,1,1,0,0,0,0,0
```

Each issue column is binary. Multiple columns can be `1` for the same image.

Supported issue labels:

```text
blurry
low_light
cluttered
bad_framing
tilted
poor_space_visibility
watermark
```

Supported room types:

```text
living_room
bedroom
kitchen
bathroom
exterior
garden
pool
attic
```

The API also accepts `other` as a room type for scoring.

## Manual Annotation

Place images in `backend/data/source_images/`, then run:

```bash
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type bedroom \
  --issues blurry low_light \
  --move
```

Arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--room_type` | No | `living_room` | Room type for all images in the folder |
| `--issues` | No | None | Space-separated defect labels |
| `--move` | No | Disabled | Move files to `data/images/` after annotation |

Examples:

```bash
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type kitchen
```

```bash
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type exterior \
  --issues bad_framing \
  --move
```

The script reads `.jpg`, `.jpeg`, `.webp`, and `.avif` files from `data/source_images/`. AVIF files are converted to JPEG before annotation. Existing rows are updated in place and new images are appended.

## Synthetic Dataset Generation

Synthetic generation applies controlled visual defects to source images.

```bash
docker compose run --rm backend python scripts/create_issue.py
docker compose run --rm backend python scripts/generate_dataset.py
docker compose run --rm backend python scripts/annotate_folder.py --room_type living_room --move
```

For each source image, `generate_dataset.py` creates:

- 1 original version
- 7 variants, each with one defect applied

This creates a balanced synthetic dataset, but it does not fully represent real-world image variability.

## Defect Simulation

Implemented in `backend/scripts/image_transforms.py`.

| Function | Simulation | Label |
|---|---|---|
| `apply_blur` | Gaussian blur | `blurry` |
| `apply_low_light` | Brightness reduction | `low_light` |
| `apply_tilt` | Rotation | `tilted` |
| `apply_bad_framing` | Cropping and resize | `bad_framing` |
| `apply_poor_space` | Tight crop | `poor_space_visibility` |
| `apply_cluttered` | Center crop and resize | `cluttered` |

`unappealing_composition` was removed from the first version because its synthetic simulation was too similar to `bad_framing`, creating noisy labels.

## Dataset Limitations

- Synthetic defects are simpler than real defects
- Synthetic variants usually contain a single defect
- Real photos often contain several simultaneous issues
- The current dataset is too small for robust generalization
- More real annotated examples are needed before adding model complexity
