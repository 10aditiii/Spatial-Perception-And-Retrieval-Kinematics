# SPARK: Spatial Perception And Retrieval Kinematics 🤖🛒

**SPARK** is an intelligent, simulated robotics system designed to autonomously navigate, map, and retrieve items in a dynamic environment (like a supermarket). It combines state-of-the-art computer vision, A* pathfinding, and a semantic "cognitive memory" system to act as a highly intelligent store assistant.

---

## 🌟 Core Features & Architecture

### 1. Two-Stage Vision Pipeline
Instead of relying on a single massive model, SPARK uses a highly efficient two-stage AI pipeline:
- **Detection (YOLOv8n):** A generic object detector scans the robot's camera feed to find the bounding boxes of any physical objects (e.g., bottles, boxes, bags) while completely ignoring background noise.
- **Classification (Custom YOLO Classifier):** The detected objects are cropped and fed into a custom-trained classifier specifically designed for grocery items. This isolates the classifier from background noise, drastically reducing hallucinations.

### 2. Confidence Calibration & NMS
Real-world hardware is messy. The vision system implements:
- **Class-Agnostic NMS (Non-Maximum Suppression):** Aggressively merges overlapping bounding boxes to prevent the AI from incorrectly identifying different parts of the same item (e.g., the cap and the label of a bottle) as separate objects.
- **Multi-Frame Verification:** In live environments, the robot requires an item to be seen with high confidence across multiple consecutive frames before committing it to memory, completely eliminating false positives from motion blur.

### 3. Cognitive Memory & Semantic Search 🧠
SPARK doesn't just record coordinates; it understands *what* things are. 
- **Persistent Mapping:** Discoveries are logged into an SQLite database (`spatial_memory.db`) with exact (X, Y) coordinates and timestamps.
- **Semantic Smart Search:** If a user asks for "Water" and the robot discovers the water has been moved, it doesn't give up. The robot accesses its semantic knowledge graph, realizes that "Water" is a *Beverage*, and dynamically recalculates its path to investigate areas where it previously saw other beverages (like Juice or Soda).

### 4. A* (A-Star) Pathfinding Navigation
The robot operates in a mathematically simulated supermarket maze. It uses a grid-based **A* pathfinding algorithm** to calculate the shortest, collision-free route to its destination, gracefully weaving around physical shelf obstacles rather than moving in basic straight lines.

### 5. Live Interactive Dashboard
The entire system is glued together by a beautiful, interactive **Streamlit Dashboard** featuring:
- A live "RViz-style" 2D simulation map showing the physical store layout, the robot's exact heading, and its planned A* navigation paths in real-time.
- Database management tools to easily seed or wipe the robot's memory.
- A "Camera Override" tool allowing developers to upload static images and simulate what the robot is seeing at any given aisle.

---

## 🛠️ Tech Stack
- **AI / Computer Vision:** Ultralytics YOLOv8, OpenCV, PyTorch
- **Navigation:** Python-based A* (A-Star) Algorithm
- **Database:** SQLite
- **Interface & Simulation:** Streamlit, Matplotlib
