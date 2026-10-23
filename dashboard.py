import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import math
import os

from vision_system import VisionSystem, DetectionResult, WrongModelError, EXPECTED_CLASSES
from cognitive_memory import CognitiveMemory, suggest_zone
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

def draw_detections(frame_rgb, results):
    """Draws boxes + labels with a filled background for legibility, and
    nudges label positions apart when they'd otherwise overlap (the main
    cause of the unreadable, stacked-text screenshots from last review)."""
    placed_label_rects = []  # (x1, y1, x2, y2) of text backgrounds already drawn
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.55, 2

    # Draw boxes first, biggest first, so small boxes nested in big ones
    # still get their own visible outline on top.
    for res in sorted(results, key=lambda r: (r.box[2]-r.box[0])*(r.box[3]-r.box[1]), reverse=True):
        x1, y1, x2, y2 = res.box
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for res in results:
        x1, y1, x2, y2 = res.box
        label_str = f"{res.label} ({res.confidence*100:.0f}%)"
        (tw, th), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)

        ly = max(th + 6, y1 - 6)
        lx = x1
        # Simple collision avoidance: if this label's background box would
        # overlap one already placed, push it down below the last one.
        for _ in range(8):
            rect = (lx, ly - th - 6, lx + tw + 6, ly + baseline)
            overlap = any(
                not (rect[2] < p[0] or rect[0] > p[2] or rect[3] < p[1] or rect[1] > p[3])
                for p in placed_label_rects
            )
            if not overlap:
                break
            ly += th + 10
        placed_label_rects.append((lx, ly - th - 6, lx + tw + 6, ly + baseline))

        cv2.rectangle(frame_rgb, (lx, ly - th - 6), (lx + tw + 6, ly + baseline), (0, 255, 0), -1)
        cv2.putText(frame_rgb, label_str, (lx + 3, ly), font, font_scale, (0, 0, 0), thickness)

    return frame_rgb


st.set_page_config(layout="wide", page_title="SPARK Dashboard", page_icon="🤖")

@st.cache_resource
def load_vision_system():
    # Cache the models so they don't reload on every UI interaction
    with open("config.json", "r") as f:
        config = json.load(f)
    try:
        vs = VisionSystem(
            det_weights=config["models"]["detector_weights"],
            cls_weights=config["models"]["classifier_weights"],
            camera_index=config["camera"]["index"],
            det_conf_thresh=config["models"].get("detection_confidence", 0.4),
            cls_conf_thresh=config["models"].get("classification_confidence", 0.6),
            required_frames=config["camera"].get("required_frames", 3),
            max_attempts=config["camera"].get("max_attempts", 10)
        )
        return vs, None
    except WrongModelError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Failed to load AI models. Ensure they exist. Error: {e}"

if "memory" not in st.session_state:
    st.session_state.memory = CognitiveMemory()
if "robot" not in st.session_state:
    st.session_state.robot = Robot()
if "scan_log" not in st.session_state:
    st.session_state.scan_log = []  # list of dicts: {zone, items: [(label, conf)]}
if "last_reasoning" not in st.session_state:
    st.session_state.last_reasoning = None

vision, vision_error = load_vision_system()

st.title("🤖 SPARK Cognitive Robotics Dashboard")
st.markdown("Live view of the robot's camera feed, spatial mapping, and cognitive memory.")

if vision_error:
    st.error(f"⚠️ Model check failed — not starting with a broken classifier: {vision_error}")

# Placeholders for stats / reasoning / scan log are filled at the END of the
# script, after the sidebar buttons have run, so they always show the latest
# state instead of lagging one action behind.
stats_placeholder = st.empty()

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
reasoning_placeholder = st.empty()

st.divider()
st.subheader("📜 Session Scan Log")
scan_log_placeholder = st.empty()


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
        plt.close(fig)
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

                frame_rgb = draw_detections(frame_rgb, results)
                for res in results:
                    # Remember
                    mem.remember(res.label, robot.x, robot.y, res.confidence, robot.current_zone)

                st.session_state.scan_log.append({
                    "n": len(st.session_state.scan_log) + 1,
                    "zone": robot.current_zone,
                    "items": [(r.label, r.confidence) for r in results],
                })

                camera_placeholder.image(frame_rgb, use_container_width=True)
                if not results:
                    st.sidebar.warning("Scan finished: No items detected.")
                else:
                    st.sidebar.success(f"Scan finished: Found {len(results)} items!")
                    # Memory files each item under the aisle the robot is standing in.
                    # Flag likely mix-ups (e.g. sugar scanned while the dropdown is
                    # still on Beverages) so the operator can fix them.
                    for res in results:
                        expected = suggest_zone(res.label, WAYPOINTS.keys())
                        if expected and expected != robot.current_zone:
                            st.sidebar.warning(
                                f"{res.label} is normally found in {expected}, but it was filed under "
                                f"{robot.current_zone}. If that's a mix-up, change the zone dropdown and scan again."
                            )
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
        st.session_state.last_reasoning = None  # direct recall, no semantic reasoning to show
    else:
        st.sidebar.warning(f"🤔 I've never seen {search_query}. Thinking...")
        fallback = mem.smart_search_order_detailed(search_query, robot.x, robot.y)
        if fallback and fallback[0]["semantic_bonus"] > 0:
            # The item belongs to a category we've already seen something from.
            best = fallback[0]
            target_zone = best["entry"].zone
            st.sidebar.info(f"💡 Semantics suggest it's at {target_zone} (near {best['entry'].item}). Driving there...")
            # Show the actual reasoning, not just the final answer, so it's
            # clear this is computed, not guessed.
            st.session_state.last_reasoning = {
                "mode": "semantic",
                "query": search_query,
                "candidates": fallback,
            }
        else:
            # No category knowledge to lean on. Don't just stay near where the
            # robot is standing -- explore zones it hasn't scanned yet,
            # nearest first.
            scanned_zones = {e.zone for e in mem.all_items()} | {s["zone"] for s in st.session_state.scan_log}
            unexplored = sorted(
                (
                    {"zone": name, "distance": math.hypot(x - robot.x, y - robot.y)}
                    for name, (x, y) in WAYPOINTS.items()
                    if name not in scanned_zones and name != "Checkout"
                ),
                key=lambda z: z["distance"],
            )
            if unexplored:
                target_zone = unexplored[0]["zone"]
                st.sidebar.info(f"🧭 No category clue for '{search_query}'. Exploring the nearest unscanned zone: {target_zone}.")
                st.session_state.last_reasoning = {
                    "mode": "explore",
                    "query": search_query,
                    "scanned": sorted(scanned_zones),
                    "unexplored": unexplored,
                }
            elif fallback:
                best = fallback[0]
                target_zone = best["entry"].zone
                st.sidebar.info(f"Every zone has been scanned and there's no category clue. Rechecking nearest: {target_zone}.")
                st.session_state.last_reasoning = {
                    "mode": "semantic",
                    "query": search_query,
                    "candidates": fallback,
                }
            else:
                target_zone = list(WAYPOINTS.keys())[0]
                st.sidebar.info(f"Nothing in memory yet. Defaulting to {target_zone}.")
                st.session_state.last_reasoning = None
            
    if target_zone:
        target_x, target_y = WAYPOINTS[target_zone]
        path = robot.calculate_path(target_x, target_y)
        
        for (px, py) in path:
            robot.x, robot.y = px, py
            fig = draw_map(robot, target=target_zone, path=path)
            map_placeholder.pyplot(fig)
            plt.close(fig)
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
    st.session_state.scan_log = []
    st.session_state.last_reasoning = None
    st.rerun()

# Final redraw of map
fig = draw_map(st.session_state.robot)
map_placeholder.pyplot(fig)
plt.close(fig)

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


# ---------------------------------------------------------------------------
# Final render of stats / reasoning / scan log. This runs AFTER every sidebar
# handler above, so it always reflects the scan or search that just happened.
# ---------------------------------------------------------------------------
_items_now = st.session_state.memory.all_items()
_n_items = len(_items_now)
_avg_conf = (sum(e.confidence for e in _items_now) / _n_items) if _n_items else 0.0
_categories_seen = len({e.item.lower() for e in _items_now})
with stats_placeholder.container():
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Items Remembered", _n_items)
    stat2.metric("Avg Confidence", f"{_avg_conf*100:.0f}%" if _n_items else "—")
    stat3.metric("Categories Covered", f"{_categories_seen} / {len(EXPECTED_CLASSES)}")
    stat4.metric("Scans This Session", len(st.session_state.scan_log))

_r = st.session_state.last_reasoning
if _r:
    with reasoning_placeholder.container():
        with st.expander(f"💭 Why I chose that zone for '{_r['query']}'", expanded=True):
            if _r.get("mode") == "explore":
                st.caption(
                    "No known category for this item, so there's no semantic clue to follow. "
                    "Instead of guessing near where I'm standing, I explore zones I haven't scanned yet, nearest first."
                )
                st.write("**Already scanned:** " + (", ".join(_r["scanned"]) or "none"))
                st.dataframe(
                    pd.DataFrame([
                        {"Unscanned zone": z["zone"], "Distance from robot": f"{z['distance']:.2f}"}
                        for z in _r["unexplored"]
                    ]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption(
                    "Score = distance from robot − semantic bonus (+ a small recency penalty). "
                    "Lower score wins. This is the actual computation, not a guess."
                )
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Item seen": c["entry"].item,
                            "Zone": c["entry"].zone,
                            "Category match": c["category"] if c["semantic_bonus"] > 0 else "—",
                            "Distance": f"{c['distance']:.2f}",
                            "Hours since seen": f"{c['age_hours']:.2f}",
                            "Semantic bonus": f"-{c['semantic_bonus']:.1f}" if c["semantic_bonus"] else "0",
                            "Final score": f"{c['score']:.2f}",
                        }
                        for c in _r["candidates"]
                    ]),
                    use_container_width=True, hide_index=True,
                )
else:
    reasoning_placeholder.empty()

if st.session_state.scan_log:
    _log_rows = []
    for _entry in reversed(st.session_state.scan_log):
        _items_str = ", ".join(f"{label} ({conf*100:.0f}%)" for label, conf in _entry["items"]) or "nothing found"
        _log_rows.append({"Scan #": _entry["n"], "Zone": _entry["zone"], "Found": _items_str})
    scan_log_placeholder.dataframe(pd.DataFrame(_log_rows), use_container_width=True, hide_index=True)
else:
    scan_log_placeholder.caption("No scans yet this session — use Drive & Scan to start building the log.")
