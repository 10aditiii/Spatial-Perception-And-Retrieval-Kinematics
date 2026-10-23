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

from vision_system import VisionSystem
from cognitive_memory import CognitiveMemory
from sim_world import Robot


def main():
    import json
    with open("config.json", "r") as f:
        config = json.load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", default=config["models"]["detector_weights"])
    ap.add_argument("--cls_weights", default=config["models"]["classifier_weights"])
    ap.add_argument("--item", required=True, help="Item to find, e.g. 'Water'")
    ap.add_argument("--conf", type=float, default=config["models"]["confidence_threshold"])
    ap.add_argument("--camera", type=int, default=config["camera"]["index"])
    args = ap.parse_args()

    vision = VisionSystem(
        args.det_weights, 
        args.cls_weights, 
        args.camera, 
        args.conf,
        required_frames=config["camera"].get("required_frames", 3),
        max_attempts=config["camera"].get("max_attempts", 10)
    )
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

    found, conf = vision.find_specific_item(args.item)
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
        found, conf = vision.find_specific_item(args.item)
        if found:
            memory.remember(args.item, x=robot.x, y=robot.y, confidence=conf, zone=robot.current_zone)
            print(f"\nFound '{args.item}' at {robot.current_zone}! Memory updated (confidence {conf*100:.0f}%).")
            return
        print("  Not here either.\n")

    print(f"Checked {len(search_order)} likely locations, '{args.item}' not found. "
          "Would need a full-area sweep next.")


if __name__ == "__main__":
    main()
