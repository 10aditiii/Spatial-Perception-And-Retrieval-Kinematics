# Spatial Perception And Retrieval Kinematics (SPARK)

A cognitive robotics pipeline that detects and classifies items in real time,
builds a spatial memory of where each item was last seen, and can navigate
back to retrieve an item on request — re-searching nearby likely locations
if the item has moved. Built and validated end-to-end on a laptop CPU, with
a lightweight simulated environment standing in for a physical robot.

Final-year B.Tech (CSE, AI & Robotics) project.

## What it actually does right now

- **Detection**: locates objects in a live camera frame (YOLO26n, pretrained
  on COCO's general object classes)
- **Classification**: identifies which of 25 supermarket product categories
  a detected item belongs to (YOLO26n-cls, fine-tuned on the Freiburg
  Groceries Dataset — 89.7% Top-1 / 98.6% Top-5 accuracy on the val split)
- **Spatial memory**: remembers `(item, location, last-seen time, confidence)`
  every time something is detected and classified, persisted to
  `spatial_memory.json`
- **Retrieval + smart search**: given "where is X", recalls its last known
  location, moves there (simulated) and re-verifies with the camera; if it's
  not there anymore, checks the closest / most-recently-active other known
  locations before giving up, rather than searching randomly
- **Simulated environment**: a simple 2D waypoint-based "store" (`sim_world.py`)
  stands in for real robot localization/navigation for this proof-of-concept

## Honest scope / limitations

- The detector (`yolo26n.pt`) is a general-purpose pretrained model, not
  fine-tuned on supermarket products specifically — only the classifier is
  trained on project data. Fine-tuning the detector needs a bounding-box
  labeled dataset (e.g. HoloSelecta), which is a planned next step.
- Localization is simulated via fixed, hardcoded waypoints, not real SLAM.
  This is a deliberate simplification for a laptop-only proof-of-concept,
  not a claim of physical robot deployment.
- The classifier is trained and evaluated on the Freiburg Groceries Dataset
  (public benchmark images), not on real footage from the eventual physical
  deployment environment.

## Architecture

```
Camera frame
     |
     v
[Detector: yolo26n.pt] -- finds WHERE objects are (bounding boxes)
     |
     v
[Classifier: yolo26n-cls best.pt] -- finds WHAT each object is
     |
     v
[Cognitive Memory: cognitive_memory.py] -- remembers (item, location, time)
     |
     v
[Query + Retrieval: query_item.py] -- recall -> navigate -> verify -> smart
                                       re-search if missing -> update memory
```

## Setup

```bash
pip install -U ultralytics opencv-python
```

## Reproducing from scratch

1. **Get the dataset:**
   ```bash
   git clone https://github.com/PhilJd/freiburg_groceries_dataset.git
   cd freiburg_groceries_dataset/src
   python download_dataset.py
   cd ../..
   ```
2. **Split into train/val:**
   ```bash
   python prepare_split.py --src freiburg_groceries_dataset/images --dst ./dataset_split
   ```
3. **Train the classifier:**
   ```bash
   python train.py --data ./dataset_split --epochs 20
   ```
   Produces `runs/classify/train/weights/best.pt`.

## Running the full pipeline

**Scan phase** (populate spatial memory by visiting each simulated waypoint):
```bash
python run_pipeline.py --cls_weights "runs/classify/train/weights/best.pt"
```

**Query phase** (ask where an item is):
```bash
python query_item.py --cls_weights "runs/classify/train/weights/best.pt" --item "Water"
```

**Other scripts:**
- `predict.py` — quick single-image/folder inference test on the trained classifier
- `realtime_detect.py` — live webcam detection only (generic COCO classes)
- `sim_world.py` — the simulated 2D environment (waypoints, robot pose, navigation)

## Tech stack

Python, Ultralytics YOLO26 (detection + classification), OpenCV, Tesseract-style
OCR groundwork from earlier project iterations.

## Next steps

- Fine-tune the detection stage on a bounding-box-labeled retail dataset
  (HoloSelecta) so detection is supermarket-specific, not generic
- Replace simulated waypoint localization with a more realistic
  localization/mapping approach
- Validate against real (non-benchmark) camera footage from the target
  deployment environment
