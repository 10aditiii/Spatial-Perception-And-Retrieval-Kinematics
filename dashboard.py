import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os

from vision_system import VisionSystem, DetectionResult
from cognitive_memory import CognitiveMemory
from sim_world import Robot, WAYPOINTS, OBSTACLES
import matplotlib.patches as patches

def draw_map(robot, target=None, path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1E1E1E')
    
    # Grid
    ax.grid(color='#333333', linestyle='--', linewidth=0.5, zorder=1)
    
    # Draw obstacles (Shelves)
    for obs in OBSTACLES:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1], 
                                 linewidth=2, edgecolor='#555555', facecolor='#4A4A4A', 
                                 alpha=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text((obs[0]+obs[2])/2, (obs[1]+obs[3])/2, "SHELF", 
                color='#888888', fontsize=12, ha='center', va='center', weight='bold', rotation=90, zorder=3)
        
    # Draw waypoints
    for name, coords in WAYPOINTS.items():
        is_target = (name == target)
        color = '#00FF00' if is_target else '#4488FF'
        size = 200 if is_target else 80
        ax.scatter(coords[0], coords[1], c=color, s=size, edgecolors='white', linewidth=1.5, zorder=4)
        ax.text(coords[0], coords[1]+0.3, name.replace(" - ", "\n"), color='white', 
                fontsize=8, ha='center', weight='bold', zorder=5)
        
    # Draw Path
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color='#00FF00', linestyle='--', linewidth=2.5, alpha=0.7, zorder=3)
        
    # Draw Robot
    marker_angle = robot.heading_deg - 90 
    ax.scatter(robot.x, robot.y, c='#FF3366', s=400, marker=(3, 0, marker_angle), 
               edgecolors='white', linewidth=2, zorder=6)
               
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 9)
    ax.axis("off")
    
    if target:
        ax.set_title(f"Navigating to {target}...", color='white', pad=15, fontsize=14, weight='bold')
    else:
        ax.set_title(f"Current Position: {robot.current_zone}", color='white', pad=15, fontsize=14, weight='bold')
        
    return fig

st.set_page_config(layout="wide", page_title="SPARK Dashboard", page_icon="🤖")

@st.cache_resource
def load_vision_system():
    # Cache the models so they don't reload on every UI interaction
    with open("config.json", "r") as f:
        config = json.load(f)
    try:
        return VisionSystem(
            det_weights=config["models"]["detector_weights"],
            cls_weights=config["models"]["classifier_weights"],
            camera_index=config["camera"]["index"],
            conf_thresh=config["models"]["confidence_threshold"],
            required_frames=config["camera"].get("required_frames", 3),
            max_attempts=config["camera"].get("max_attempts", 10)
        )
    except Exception as e:
        st.error(f"Failed to load AI models. Ensure they exist. Error: {e}")
        return None

if "memory" not in st.session_state:
    st.session_state.memory = CognitiveMemory()
if "robot" not in st.session_state:
    st.session_state.robot = Robot()

vision = load_vision_system()

st.title("🤖 SPARK Cognitive Robotics Dashboard")
st.markdown("Live view of the robot's camera feed, spatial mapping, and cognitive memory.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Live Camera View")
    camera_placeholder = st.empty()
    camera_placeholder.info("Click 'Drive & Scan' to capture a frame from this zone.")

with col2:
    st.subheader("🗺️ Spatial Map")
    map_placeholder = st.empty()

st.divider()
st.subheader("🧠 Cognitive Memory")
memory_placeholder = st.empty()

st.sidebar.header("🕹️ Controls")
target = st.sidebar.selectbox("Select Zone to Navigate:", list(WAYPOINTS.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📷 Camera Override")
st.sidebar.caption("macOS blocks terminal camera access. Upload an image from `dataset_split` to simulate what the robot sees!")
simulated_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if st.sidebar.button("🚀 Drive & Scan", type="primary"):
    robot = st.session_state.robot
    mem = st.session_state.memory
    
    # 1. Drive
    target_x, target_y = WAYPOINTS[target]
    path = robot.calculate_path(target_x, target_y)
    
    for (px, py) in path:
        robot.x, robot.y = px, py
        
        # Live map update
        fig = draw_map(robot, target=target, path=path)
        map_placeholder.pyplot(fig)
        time.sleep(0.05)
        
    robot.current_zone = target
    robot.x, robot.y = target_x, target_y

    # 2. Scan
    if vision:
        with st.spinner("Scanning environment..."):
            if simulated_image is not None:
                import numpy as np
                file_bytes = np.asarray(bytearray(simulated_image.read()), dtype=np.uint8)
                raw_frame = cv2.imdecode(file_bytes, 1)
                results, frame = vision.analyze_frame(raw_frame)
            else:
                results, frame = vision.capture_and_analyze()
            if frame is not None:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw boxes
                for res in results:
                    x1, y1, x2, y2 = res.box
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_str = f"{res.label} ({res.confidence*100:.0f}%)"
                    cv2.putText(frame_rgb, label_str, (x1, max(20, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Remember
                    mem.remember(res.label, robot.x, robot.y, res.confidence, robot.current_zone)
                    
                camera_placeholder.image(frame_rgb, use_container_width=True)
                if not results:
                    st.sidebar.warning("Scan finished: No items detected.")
                else:
                    st.sidebar.success(f"Scan finished: Found {len(results)} items!")
            else:
                st.sidebar.error("Camera failed to return a frame.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search for Item")
search_query = st.sidebar.text_input("Item Name (e.g. 'Juice'):")
if st.sidebar.button("🔎 Find Item", type="primary") and search_query:
    mem = st.session_state.memory
    robot = st.session_state.robot
    target_zone = None
    
    mem_entry = mem.recall(search_query)
    if mem_entry:
        st.sidebar.success(f"🧠 I remember {search_query} at {mem_entry.zone}! Driving there...")
        target_zone = mem_entry.zone
    else:
        st.sidebar.warning(f"🤔 I've never seen {search_query}. Thinking...")
        fallback = mem.smart_search_order(search_query, robot.x, robot.y)
        if fallback:
            target_zone = fallback[0].zone
            st.sidebar.info(f"💡 Semantics suggest it's at {target_zone} (near {fallback[0].item}). Driving there...")
        else:
            target_zone = list(WAYPOINTS.keys())[0]
            st.sidebar.info(f"No semantic clues found. Defaulting to {target_zone}.")
            
    if target_zone:
        target_x, target_y = WAYPOINTS[target_zone]
        path = robot.calculate_path(target_x, target_y)
        
        for (px, py) in path:
            robot.x, robot.y = px, py
            fig = draw_map(robot, target=target_zone, path=path)
            map_placeholder.pyplot(fig)
            time.sleep(0.05)
            
        robot.current_zone = target_zone
        robot.x, robot.y = target_x, target_y
        st.sidebar.success(f"🏁 Arrived at {target_zone}! Use 'Drive & Scan' to look around.")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Database Management")

# Clear specific item
items = st.session_state.memory.all_items()
if items:
    item_names = [e.item for e in items]
    item_to_forget = st.sidebar.selectbox("Forget Item:", item_names)
    if st.sidebar.button("🗑️ Forget Item"):
        st.session_state.memory.forget_item(item_to_forget)
        st.rerun()

# Clear all items
if st.sidebar.button("🚨 Clear All Memory", type="primary"):
    st.session_state.memory.clear_all()
    st.rerun()

# Final redraw of map
fig = draw_map(st.session_state.robot)
map_placeholder.pyplot(fig)

# Final redraw of memory table
items = st.session_state.memory.all_items()
if items:
    df = pd.DataFrame([
        {"Item": e.item, "Zone": e.zone, "Coordinates": f"({e.x:.1f}, {e.y:.1f})", 
         "Confidence": f"{e.confidence*100:.0f}%", "Sightings": e.times_seen} 
        for e in items
    ])
    memory_placeholder.dataframe(df, use_container_width=True)
else:
    memory_placeholder.info("Memory is empty. Drive to a waypoint and scan to populate it!")
