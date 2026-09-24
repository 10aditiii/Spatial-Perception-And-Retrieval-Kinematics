"""
Vision System for SPARK.
Handles webcam capture, YOLO object detection, and classification.
"""
import cv2
import time
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

class CameraError(Exception):
    """Custom exception for hardware-level camera failures."""
    pass

@dataclass
class DetectionResult:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]

class VisionSystem:
    def __init__(self, det_weights: str, cls_weights: str, camera_index: int = 0, conf_thresh: float = 0.4, required_frames: int = 3, max_attempts: int = 10):
        self.detector = YOLO(det_weights)
        self.classifier = YOLO(cls_weights)
        self.camera_index = camera_index
        self.conf_thresh = conf_thresh
        self.required_frames = required_frames
        self.max_attempts = max_attempts

    def analyze_frame(self, frame) -> Tuple[List[DetectionResult], any]:
        """Runs detection and classification on a given image frame."""
        results = []
        if frame is None or frame.size == 0:
            return results, frame

        # Run detection with class-agnostic NMS to prevent overlapping boxes on the same object
        det_results = self.detector.predict(frame, conf=self.conf_thresh, iou=0.4, agnostic_nms=True, verbose=False)[0]
        
        # Crop and classify each detected object
        for box in det_results.boxes:
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
            if top1_conf < self.conf_thresh:
                continue
                
            label = cls_result.names[top1_idx]
            
            results.append(DetectionResult(
                label=label,
                confidence=top1_conf,
                box=(x1, y1, x2, y2)
            ))
            
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
                        
                time.sleep(0.05) # small delay between frames (approx 20fps logic speed)
                
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
