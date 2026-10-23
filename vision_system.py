"""
Vision System for SPARK.
Handles webcam capture, YOLO object detection, and classification.
"""
import cv2
import time
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

# Freiburg Groceries Dataset classes. Used as a sanity check at startup so
# the app refuses to silently run on the wrong classifier (e.g. the raw
# ImageNet-pretrained checkpoint instead of our fine-tuned one). This is
# exactly the bug that caused the "nipple / goldfish / sunscreen" labels
# during the last review -- the wrong .pt got loaded and nobody noticed
# until it was on screen.
EXPECTED_CLASSES = {
    "beans", "cake", "candy", "cereal", "chips", "chocolate", "coffee", "corn",
    "fish", "flour", "honey", "jam", "juice", "milk", "nuts", "oil", "pasta",
    "rice", "soda", "spices", "sugar", "tea", "tomato_sauce", "vinegar", "water",
}


class CameraError(Exception):
    """Custom exception for hardware-level camera failures."""
    pass


class WrongModelError(Exception):
    """Raised when the loaded classifier's class names don't match the
    grocery dataset -- i.e. the wrong weights file got loaded."""
    pass


@dataclass
class DetectionResult:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


def _box_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _iou(a, b):
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment_ratio(a, b):
    """How much of the SMALLER box is covered by the intersection.
    Standard IoU misses this: a small box fully nested inside a much
    bigger one (e.g. a bottle cap region inside the whole-bottle box)
    has low IoU but is clearly the same physical object."""
    inter = _intersection_area(a, b)
    smaller = min(_box_area(a), _box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def dedupe_boxes(results: List[DetectionResult], iou_thresh: float = 0.35,
                  containment_thresh: float = 0.75) -> List[DetectionResult]:
    """Collapses near-duplicate / nested boxes that the detector's own NMS
    didn't catch (common when the detector is a generic, non-fine-tuned
    model producing several candidate boxes per physical item). Keeps the
    highest-confidence box in each overlapping cluster."""
    if not results:
        return results

    kept: List[DetectionResult] = []
    # Highest confidence first, so a strong detection "wins" its cluster.
    for res in sorted(results, key=lambda r: r.confidence, reverse=True):
        duplicate = False
        for k in kept:
            if _iou(res.box, k.box) >= iou_thresh or _containment_ratio(res.box, k.box) >= containment_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(res)
    return kept


class VisionSystem:
    def __init__(self, det_weights: str, cls_weights: str, camera_index: int = 0,
                 det_conf_thresh: float = 0.4, cls_conf_thresh: float = 0.6,
                 required_frames: int = 3, max_attempts: int = 10,
                 verify_classes: bool = True):
        self.detector = YOLO(det_weights)
        self.classifier = YOLO(cls_weights)
        self.camera_index = camera_index
        self.det_conf_thresh = det_conf_thresh
        self.cls_conf_thresh = cls_conf_thresh
        self.required_frames = required_frames
        self.max_attempts = max_attempts
        self.full_frame_priority = 0.85

        if verify_classes:
            self._verify_classifier()

    def _verify_classifier(self):
        """Fail loudly, at startup, if the classifier isn't the grocery
        model. This is the check that would have caught last review's bug
        before it ever reached the demo."""
        names = set(n.lower() for n in self.classifier.names.values())
        overlap = names & EXPECTED_CLASSES
        if len(overlap) < 10:  # fine-tuned model should match almost all 25
            raise WrongModelError(
                f"Loaded classifier has {len(names)} classes and only "
                f"{len(overlap)} match the grocery dataset "
                f"(e.g. found: {list(names)[:6]}...). "
                f"This looks like the wrong weights file -- check "
                f"config.json's 'classifier_weights' path. Expected the "
                f"fine-tuned grocery model, not a generic pretrained checkpoint."
            )

    def analyze_frame(self, frame) -> Tuple[List[DetectionResult], any]:
        """Runs detection and classification on a given image frame."""
        results = []
        if frame is None or frame.size == 0:
            return results, frame

        # Run detection with class-agnostic NMS to prevent overlapping boxes on the same object
        det_results = self.detector.predict(
            frame, conf=self.det_conf_thresh, iou=0.4, agnostic_nms=True, verbose=False
        )[0]

        boxes = list(det_results.boxes)

        def classify_full_frame():
            """Fallback: classify the whole frame as one item. Used when the
            generic COCO detector finds no box at all, or finds only small /
            badly-cropped boxes that starve the classifier of context (e.g.
            a 4%-of-frame sliver misread as 'book'). The classifier was
            trained on frame-filling product photos, so for a single-image
            Camera Override upload this is often more reliable than trusting
            an un-tuned detector's box shape."""
            h, w = frame.shape[:2]
            cls_result = self.classifier.predict(frame, verbose=False)[0]
            top1_idx = int(cls_result.probs.top1)
            top1_conf = float(cls_result.probs.top1conf)
            if top1_conf >= self.cls_conf_thresh:
                return DetectionResult(label=cls_result.names[top1_idx], confidence=top1_conf, box=(0, 0, w, h))
            return None

        # Whole-frame-first: the classifier was trained on frame-filling product
        # photos. If it is highly confident about the entire frame, trust that
        # over the generic detector's sub-crops (which cut bottles into slivers
        # and made VINEGAR read as CANDY/SODA and JUICE read as SODA).
        full = classify_full_frame()
        if full and full.confidence >= self.full_frame_priority:
            return [full], frame

        if not boxes:
            fallback = full
            if fallback:
                results.append(fallback)
            return results, frame

        # Crop and classify each detected object
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls_result = self.classifier.predict(crop, verbose=False)[0]
            top1_idx = int(cls_result.probs.top1)
            top1_conf = float(cls_result.probs.top1conf)

            # Reject low-confidence classifications (background objects)
            if top1_conf < self.cls_conf_thresh:
                continue

            label = cls_result.names[top1_idx]

            results.append(DetectionResult(
                label=label,
                confidence=top1_conf,
                box=(x1, y1, x2, y2)
            ))

        # Collapse near-duplicate / nested boxes the detector's NMS missed
        # (the main source of the cluttered, overlapping-label mess).
        results = dedupe_boxes(results)

        # If the detector produced boxes but NONE of them survived
        # classification, the boxes were likely too small/poorly cropped
        # for the classifier to work with (not necessarily that nothing is
        # there) -- try the whole frame once before giving up.
        if not results:
            fallback = classify_full_frame()
            if fallback:
                results.append(fallback)

        return results, frame

    def capture_and_analyze(self) -> Tuple[List[DetectionResult], any]:
        """
        Captures multiple frames from the camera and returns only objects
        that were consistently detected across `required_frames` consecutive frames.
        Returns the final frame used.
        """
        cap = None
        final_frame = None

        # Maps label -> { "count": int, "latest_res": DetectionResult }
        consecutive_counts = {}
        confirmed_results = {}

        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise CameraError(f"Camera index {self.camera_index} could not be opened. Check connection.")

            # Brief pause to allow the camera sensor to warm up
            time.sleep(0.1)

            for _ in range(self.max_attempts):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                final_frame = frame
                results, _ = self.analyze_frame(frame)

                # Keep track of what we saw in THIS frame
                seen_this_frame = {}
                for res in results:
                    if res.label not in seen_this_frame or res.confidence > seen_this_frame[res.label].confidence:
                        seen_this_frame[res.label] = res

                # Update consecutive counts
                for label, res in seen_this_frame.items():
                    if label in consecutive_counts:
                        consecutive_counts[label]['count'] += 1
                        consecutive_counts[label]['latest_res'] = res
                    else:
                        consecutive_counts[label] = {'count': 1, 'latest_res': res}

                # Reset counts for things NOT seen in this frame
                for label in list(consecutive_counts.keys()):
                    if label not in seen_this_frame:
                        consecutive_counts[label]['count'] = 0

                # Check if anything hit the threshold
                for label, data in consecutive_counts.items():
                    if data['count'] >= self.required_frames:
                        confirmed_results[label] = data['latest_res']

                time.sleep(0.05)  # small delay between frames (approx 20fps logic speed)

        except Exception as e:
            print(f"  [Error] VisionSystem Hardware failure: {e}")
            return [], None
        finally:
            # Crucial for robotics: ALWAYS release hardware resources
            if cap is not None and cap.isOpened():
                cap.release()

        return list(confirmed_results.values()), final_frame

    def find_specific_item(self, target_item: str) -> Tuple[bool, float]:
        """Captures a frame and checks if a specific item is present."""
        results, _ = self.capture_and_analyze()
        for res in results:
            if res.label.lower() == target_item.lower():
                return True, res.confidence
        return False, 0.0
