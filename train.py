"""
Step 4: Train a YOLO26n classifier on the split dataset.

YOLO26 is Ultralytics' current SOTA model family (released Jan 2026),
replacing YOLOv8/v9/v10/v11 for new projects.

Requires: pip install -U ultralytics

Usage:
    python train.py --data ./dataset_split --epochs 20
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset_split", help="Path to folder containing train/ and val/ subfolders")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # yolo26n-cls.pt = smallest/fastest YOLO26 classification model, CPU-friendly
    model = YOLO("yolo26n-cls.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",  # change to 0 if you have a CUDA GPU available
    )

    # Validate on the val split and print accuracy
    metrics = model.val()
    print("Top-1 accuracy:", metrics.top1)
    print("Top-5 accuracy:", metrics.top5)

    # Weights + training curves are saved automatically under runs/classify/train*/


if __name__ == "__main__":
    main()
