# BrickFinder
Finding LEGO bricks from a pile

You give it a picture of a pile of LEGO and the piece you want, and it draws boxes around that piece.
See `CSC 4888 Project Proposal.pdf` for the full plan.

## First model

YOLO11s (pretrained on COCO) fine-tuned on the B200 LEGO detection dataset. That dataset is 2000 computer
rendered images (2048x2048) with 400 pieces each. For now we only train on 10 pieces; all the other parts are just background:

| Part | Piece | Part | Piece |
|---|---|---|---|
| 3001 | Brick 2x4 | 3022 | Plate 2x2 |
| 3003 | Brick 2x2 | 3023 | Plate 1x2 |
| 3004 | Brick 1x2 | 3039 | Slope 45 2x2 |
| 3005 | Brick 1x1 | 3069b | Tile 1x2 |
| 3020 | Plate 2x4 | 3062b | Round Brick 1x1 |

- No data augmentation yet. When we add it, don't flip the images: the dataset readme says some pieces
  have mirror image versions, so a flipped piece looks like a different piece.
- Images are shrunk to 1024 for training.
- Split is 1600 train / 200 val / 200 test images, saved in `data/splits.json`.
- The model only uses shape for now. The `color_id` values (`m0`-`m39`) match real LEGO colors in
  `data/raw/color-scheme.png` (for example `m2` is Bright Red), so we can add color later.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

## Run

```bash
bash scripts/download_data.sh            # downloads the images into data/raw (about 14 GB)
python scripts/prepare_yolo_dataset.py   # makes the YOLO labels and split in data/yolo
```

Then open `yolo_model.ipynb` with the `.venv` kernel. It trains the model, tests it on the test images,
and has a `find_piece()` function that draws boxes around a piece in an image. Training takes about
3 minutes per epoch on an M2 Max. The test results are saved to `results/initial_metrics.csv`.

## Licenses

- B200 LEGO detection dataset (Ronan Pickell): CC BY-SA 4.0
- Ultralytics YOLO: AGPL-3.0
