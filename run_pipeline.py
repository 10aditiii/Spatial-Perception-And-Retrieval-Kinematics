"""
Full scan run: the simulated robot visits each waypoint (aisle/zone),
captures a real webcam frame at each stop, runs the detect+classify
pipeline on it, and remembers what it saw and where.

This is the "exploration" phase -- run this first to populate memory,
then use query_item.py to ask "where is X".

Requires: pip install -U ultralytics opencv-python
Needs a webcam. At each waypoint the script pauses so you can show the
camera a different item (simulates the robot seeing different products
in different aisles), then press any key to continue.

Usage:
    python run_pipeline.py --cls_weights "runs/classify/train-2/weights/best.pt"
"""
import argparse
import cv2
from ultralytics import YOLO

from cognitive_memory import CognitiveMemory
from sim_world import Robot, WAYPOINTS


def scan_here(detector, classifier, robot, memory, conf_thresh, camera):
    """Capture one frame from the webcam, run detect+classify, remember
    anything found at the robot's current position."""
    cap = cv2.VideoCapture(camera)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  [warn] could not read from camera, skipping this stop")
        return

    det_results = detector.predict(frame, conf=conf_thresh, verbose=False)[0]
    found_any = False
    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_result = classifier.predict(crop, verbose=False)[0]
        top1_idx = int(cls_result.probs.top1)
        top1_conf = float(cls_result.probs.top1conf)
        label = cls_result.names[top1_idx]

        memory.remember(label, x=robot.x, y=robot.y, confidence=top1_conf, zone=robot.current_zone)
        print(f"  [seen] {label} (confidence {top1_conf*100:.0f}%) -> remembered at "
              f"{robot.current_zone} ({robot.x:.1f}, {robot.y:.1f})")
        found_any = True

    if not found_any:
        print("  [seen] nothing recognized at this stop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", default="yolo26n.pt")
    ap.add_argument("--cls_weights", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    detector = YOLO(args.det_weights)
    classifier = YOLO(args.cls_weights)
    memory = CognitiveMemory()
    robot = Robot()

    print(f"Starting scan of {len(WAYPOINTS)} waypoints.\n"
          "At each stop, show the webcam an item, then press Enter to continue.\n")

    for name in WAYPOINTS:
        robot.move_to_waypoint(name)
        input(f"  -> Show an item to the camera for '{name}', then press Enter...")
        scan_here(detector, classifier, robot, memory, args.conf, args.camera)
        print()

    print("Scan complete. Memory saved to spatial_memory.json")
    print(f"Items remembered: {[e.item for e in memory.all_items()]}")


if __name__ == "__main__":
    main()
