"""
Step 2: Organize the downloaded Freiburg Groceries images into a
train/val folder structure that YOLOv8's classification mode (and any
standard PyTorch ImageFolder-based trainer) expects:

    dataset_split/
        train/
            BEANS/   img1.png ...
            CANDY/   ...
            ...
        val/
            BEANS/   ...
            ...

Freiburg Groceries ships as: images/<CLASS_NAME>/<file>.png (25 class folders).
This script does an 80/20 split per class, copying files (originals untouched).

Usage:
    python prepare_split.py --src /path/to/freiburg_groceries_dataset/images --dst ./dataset_split
"""
import argparse
import random
import shutil
from pathlib import Path

random.seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the downloaded 'images' folder (contains one subfolder per class)")
    ap.add_argument("--dst", default="./dataset_split", help="Where to write train/val folders")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    classes = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders found under {src} -- check the path.")

    total_train, total_val = 0, 0
    for cls_dir in classes:
        files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split_name, split_files in (("train", train_files), ("val", val_files)):
            out_dir = dst / split_name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"{cls_dir.name:20s} train={len(train_files):4d}  val={len(val_files):4d}")

    print(f"\nDone. Total train={total_train}, val={total_val}")
    print(f"Split written to: {dst.resolve()}")


if __name__ == "__main__":
    main()
