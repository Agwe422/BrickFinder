# Makes the YOLO dataset for our 10 lego pieces out of Briggs's cleaned annotations.
# Run this once from the repo folder after downloading the images:
#   python scripts/prepare_yolo_dataset.py

import json
import os
import random

import pandas as pd

# the 10 pieces we're looking for. everything else in the pile is just background
PIECES = ["3001", "3003", "3004", "3005", "3020", "3022", "3023", "3039", "3069b", "3062b"]
CLASS_NAMES = [
    "3001_brick_2x4",
    "3003_brick_2x2",
    "3004_brick_1x2",
    "3005_brick_1x1",
    "3020_plate_2x4",
    "3022_plate_2x2",
    "3023_plate_1x2",
    "3039_slope_45_2x2",
    "3069b_tile_1x2",
    "3062b_round_brick_1x1",
]
IMG_SIZE = 2048  # all the B200 images are 2048x2048

df = pd.read_parquet("data/processed/lego_annotations.parquet")
boxes = df[df["part_id"].isin(PIECES)]
print("images:", df["image_filename"].nunique())
print("boxes for our 10 pieces:", len(boxes))

# split the images 80/10/10. save the split so we always use the same one
if os.path.exists("data/splits.json"):
    with open("data/splits.json") as f:
        splits = json.load(f)
else:
    images = sorted(df["image_filename"].unique(), key=lambda name: int(name[:-4]))
    random.seed(0)
    random.shuffle(images)
    splits = {
        "train": sorted(images[:1600], key=lambda name: int(name[:-4])),
        "val": sorted(images[1600:1800], key=lambda name: int(name[:-4])),
        "test": sorted(images[1800:], key=lambda name: int(name[:-4])),
    }
    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=1)

for split in ["train", "val", "test"]:
    os.makedirs(f"data/yolo/images/{split}", exist_ok=True)
    os.makedirs(f"data/yolo/labels/{split}", exist_ok=True)

    for name in splits[split]:
        # link to the image instead of copying it since the images are 14 GB
        link = f"data/yolo/images/{split}/{name}"
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f"data/raw/images/{name}"), link)

        # YOLO label file: one line per piece -> class x_center y_center width height (0 to 1)
        lines = []
        for row in boxes[boxes["image_filename"] == name].itertuples():
            c = PIECES.index(row.part_id)
            x = (row.xmin + row.xmax) / 2 / IMG_SIZE
            y = (row.ymin + row.ymax) / 2 / IMG_SIZE
            w = (row.xmax - row.xmin) / IMG_SIZE
            h = (row.ymax - row.ymin) / IMG_SIZE
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        with open(f"data/yolo/labels/{split}/{name[:-4]}.txt", "w") as f:
            f.writelines(lines)

    print(split, len(splits[split]), "images")

# config file that tells YOLO where everything is
with open("data/yolo/dataset.yaml", "w") as f:
    f.write(f"path: {os.path.abspath('data/yolo')}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")

print(boxes["part_id"].value_counts())
