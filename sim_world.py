"""
Minimal 2D simulated environment for the robot to move through.

This is intentionally simple: a set of named waypoints (representing
shelves/aisles/zones) on an (x, y) plane. The robot has a pose and can be
commanded to move toward any waypoint. Movement is instantaneous-per-step
here (grid/waypoint navigation, not physics) -- realistic enough to test
the cognitive memory + search logic without needing a physics engine.

You can later swap this file's internals for Webots/Gazebo without
changing cognitive_memory.py, run_pipeline.py or query_item.py at all,
since they only depend on Robot.x / Robot.y / Robot.move_to().
"""
import math
import time

import json
import os

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Load store layout dynamically from config
WAYPOINTS = {k: tuple(v) for k, v in config["world"]["waypoints"].items()}
OBSTACLES = config["world"].get("obstacles", [])

import heapq

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_collision(x, y, obstacles, clearance=0.3):
    for (min_x, min_y, max_x, max_y) in obstacles:
        if min_x - clearance <= x <= max_x + clearance and min_y - clearance <= y <= max_y + clearance:
            return True
    return False

def astar_path(start, goal, obstacles, grid_res=0.2):
    start = (round(start[0]/grid_res)*grid_res, round(start[1]/grid_res)*grid_res)
    goal = (round(goal[0]/grid_res)*grid_res, round(goal[1]/grid_res)*grid_res)
    
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    directions = [(grid_res, 0), (-grid_res, 0), (0, grid_res), (0, -grid_res),
                  (grid_res, grid_res), (-grid_res, grid_res), (grid_res, -grid_res), (-grid_res, -grid_res)]
                  
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if math.hypot(current[0] - goal[0], current[1] - goal[1]) < grid_res / 2:
            goal = current # close enough
            break
            
        for dx, dy in directions:
            next_node = (current[0] + dx, current[1] + dy)
            # Add cost (diagonal is slightly more expensive)
            new_cost = cost_so_far[current] + math.hypot(dx, dy)
            
            # Boundary checks
            if not (-1.0 <= next_node[0] <= 10.0 and -1.0 <= next_node[1] <= 10.0):
                continue
                
            if is_collision(next_node[0], next_node[1], obstacles):
                continue
                
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current
                
    if goal not in came_from:
        return [start, goal] # Fallback to straight line if blocked
        
    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


class Robot:
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0):
        self.x = start_x
        self.y = start_y
        self.heading_deg = 0.0
        self.current_zone = "Entrance"

    def calculate_path(self, target_x: float, target_y: float) -> list[tuple[float, float]]:
        """Returns a list of (x,y) points forming a collision-free A* path to the target."""
        return astar_path((self.x, self.y), (target_x, target_y), OBSTACLES)

    def move_to(self, x: float, y: float, zone_name: str = "", speed_units_per_sec: float = 2.0):
        """Move from current position to (x, y) along an A* generated path."""
        path = self.calculate_path(x, y)
        for (px, py) in path:
            if px != self.x or py != self.y:
                self.heading_deg = math.degrees(math.atan2(py - self.y, px - self.x))
            self.x, self.y = px, py
            time.sleep(0.05)

        if zone_name:
            self.current_zone = zone_name
        print(f"[robot] Arrived at ({self.x:.1f}, {self.y:.1f})"
              f"{' - ' + zone_name if zone_name else ''}, heading {self.heading_deg:.0f} deg")

    def move_to_waypoint(self, name: str):
        if name not in WAYPOINTS:
            raise ValueError(f"Unknown waypoint '{name}'. Known: {list(WAYPOINTS.keys())}")
        x, y = WAYPOINTS[name]
        self.move_to(x, y, zone_name=name)

    def direction_and_distance_to(self, x: float, y: float) -> tuple[float, float]:
        """Returns (bearing_degrees, distance) from current position to a target."""
        dx, dy = x - self.x, y - self.y
        bearing = math.degrees(math.atan2(dy, dx))
        distance = math.hypot(dx, dy)
        return bearing, distance

    def nearest_waypoint_name(self) -> str:
        best_name, best_dist = None, float("inf")
        for name, (wx, wy) in WAYPOINTS.items():
            d = math.hypot(wx - self.x, wy - self.y)
            if d < best_dist:
                best_name, best_dist = name, d
        return best_name


if __name__ == "__main__":
    r = Robot()
    r.move_to_waypoint("Aisle 2 - Snacks")
    bearing, dist = r.direction_and_distance_to(*WAYPOINTS["Aisle 6 - Household"])
    print(f"Aisle 6 is {dist:.1f} units away, bearing {bearing:.0f} deg from here")
