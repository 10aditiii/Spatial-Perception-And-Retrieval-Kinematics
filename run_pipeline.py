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

from vision_system import VisionSystem
from cognitive_memory import CognitiveMemory
from sim_world import Robot, WAYPOINTS


def scan_here(vision, robot, memory):
    """Capture one frame from the webcam, run detect+classify, remember
    anything found at the robot's current position."""
    results, frame = vision.capture_and_analyze()

    if not results:
        print("  [seen] nothing recognized at this stop")
        return

    for res in results:
        memory.remember(res.label, x=robot.x, y=robot.y, confidence=res.confidence, zone=robot.current_zone)
        print(f"  [seen] {res.label} (confidence {res.confidence*100:.0f}%) -> remembered at "
              f"{robot.current_zone} ({robot.x:.1f}, {robot.y:.1f})")


def main():
    import json
    with open("config.json", "r") as f:
        config = json.load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", default=config["models"]["detector_weights"])
    ap.add_argument("--cls_weights", default=config["models"]["classifier_weights"])
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

    print(f"Starting scan of {len(WAYPOINTS)} waypoints.\n"
          "At each stop, show the webcam an item, then press Enter to continue.\n")

    for name in WAYPOINTS:
        robot.move_to_waypoint(name)
        input(f"  -> Show an item to the camera for '{name}', then press Enter...")
        scan_here(vision, robot, memory)
        print()

    print("Scan complete. Memory saved to spatial_memory.db")
    print(f"Items remembered: {[e.item for e in memory.all_items()]}")


if __name__ == "__main__":
    main()
