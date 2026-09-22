"""
Real-time object detection from a webcam using a YOLO26 detection model.

Stage 1: uses YOLO26n pretrained on COCO (generic objects) so you have a
working real-time detection pipeline immediately -- swap in your own
fine-tuned weights later (see stage 2 in the README notes below).

Requires: pip install -U ultralytics
Needs a webcam connected to the machine you run this on.

Usage:
    python realtime_detect.py
    (press 'q' in the video window to quit)
"""
import cv2
from ultralytics import YOLO

# Stage 1: generic pretrained detector (downloads automatically, ~5-6MB)
model = YOLO("yolo26n.pt")

# source=0 -> default webcam; show=True opens a live annotated video window.
# With stream=True, predict() returns a lazy generator -- it does nothing
# until you iterate over it, so we loop over results to actually run frames.
results = model.predict(source=0, show=True, conf=0.4, stream=True)

for r in results:
    # 'q' closes the window (checked once per frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
