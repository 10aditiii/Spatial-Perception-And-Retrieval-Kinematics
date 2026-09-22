"""
Ask the robot where an item is. It:
  1. Checks memory for the item's last known location
  2. Moves there (simulated) and re-scans with the real camera to verify
  3. If NOT found: smart-searches other likely locations (closest +
     most-recently-active first) instead of a blind full sweep
  4. Once found (wherever that turned out to be), updates memory with the
     new confirmed location

Requires: pip install -U ultralytics opencv-python

Usage:
    python query_item.py --cls_weights "runs/classify/train-2/weights/best.pt" --item "Water"
"""
import argparse
import cv2
from ultralytics import YOLO

from cognitive_memory import CognitiveMemory
from sim_world import Robot


def scan_for_item(detector, classifier, camera, conf_thresh, target_item):
    """Capture one frame, run detect+classify, return (found: bool, confidence)."""
    cap = cv2.VideoCapture(camera)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False, 0.0

    det_results = detector.predict(frame, conf=conf_thresh, verbose=False)[0]
    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cls_result = classifier.predict(crop, verbose=False)[0]
        label = cls_result.names[int(cls_result.probs.top1)]
        conf = float(cls_result.probs.top1conf)
        if label.lower() == target_item.lower():
            return True, conf
    return False, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", default="yolo26n.pt")
    ap.add_argument("--cls_weights", required=True)
    ap.add_argument("--item", required=True, help="Item to find, e.g. 'Water'")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    detector = YOLO(args.det_weights)
    classifier = YOLO(args.cls_weights)
    memory = CognitiveMemory()
    robot = Robot()

    entry = memory.recall(args.item)
    if entry is None:
        print(f"I have never seen '{args.item}' before. Run run_pipeline.py first to scan the store.")
        return

    bearing, distance = robot.direction_and_distance_to(entry.x, entry.y)
    print(f"'{args.item}' was last seen in {entry.zone} at ({entry.x:.1f}, {entry.y:.1f}), "
          f"{distance:.1f} units away, bearing {bearing:.0f} deg. Heading there now...\n")

    robot.move_to(entry.x, entry.y, zone_name=entry.zone)
    input(f"Arrived. Show the camera what's actually here (or show nothing if it's gone), then press Enter...")

    found, conf = scan_for_item(detector, classifier, args.camera, args.conf, args.item)
    if found:
        memory.remember(args.item, x=robot.x, y=robot.y, confidence=conf, zone=robot.current_zone)
        print(f"Confirmed: '{args.item}' is here. Memory updated (confidence {conf*100:.0f}%).")
        return

    print(f"'{args.item}' is NOT where memory said. Marking as missing and searching nearby likely spots...\n")
    memory.mark_missing(args.item)

    search_order = memory.smart_search_order(args.item, robot.x, robot.y)
    if not search_order:
        print("No other known locations to check. Full-area search needed (not automated here).")
        return

    for candidate in search_order:
        print(f"Checking {candidate.zone} ({candidate.x:.1f}, {candidate.y:.1f}) -- "
              f"last active seeing '{candidate.item}'...")
        robot.move_to(candidate.x, candidate.y, zone_name=candidate.zone)
        input("  Show the camera what's here, then press Enter...")
        found, conf = scan_for_item(detector, classifier, args.camera, args.conf, args.item)
        if found:
            memory.remember(args.item, x=robot.x, y=robot.y, confidence=conf, zone=robot.current_zone)
            print(f"\nFound '{args.item}' at {robot.current_zone}! Memory updated (confidence {conf*100:.0f}%).")
            return
        print("  Not here either.\n")

    print(f"Checked {len(search_order)} likely locations, '{args.item}' not found. "
          "Would need a full-area sweep next.")


if __name__ == "__main__":
    main()
