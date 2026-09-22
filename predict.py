"""
Quick inference test: point the trained model at one or more images
and see its top predictions -- good for a live demo during your review.

Usage (single image):
    python predict.py --weights runs/classify/train-2/weights/best.pt --source path/to/image.jpg

Usage (a whole folder of images):
    python predict.py --weights runs/classify/train-2/weights/best.pt --source path/to/folder
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pt from training")
    ap.add_argument("--source", required=True, help="Path to a single image OR a folder of images")
    ap.add_argument("--topk", type=int, default=5, help="How many top predictions to show per image")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model(args.source)

    for r in results:
        print(f"\nImage: {r.path}")
        probs = r.probs
        top_indices = probs.top5[: args.topk]
        top_confs = probs.top5conf[: args.topk]
        for rank, (idx, conf) in enumerate(zip(top_indices, top_confs), start=1):
            class_name = r.names[int(idx)]
            print(f"  {rank}. {class_name:15s}  confidence={float(conf) * 100:.1f}%")


if __name__ == "__main__":
    main()
