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


# Example store layout -- edit these to match whatever zones you want to
# demo (aisle names, shelf numbers, etc.)
WAYPOINTS = {
    "Aisle 1 - Beverages": (1.0, 1.0),
    "Aisle 2 - Snacks":    (4.0, 1.0),
    "Aisle 3 - Dairy":     (7.0, 1.0),
    "Aisle 4 - Grains":    (1.0, 4.0),
    "Aisle 5 - Condiments":(4.0, 4.0),
    "Aisle 6 - Household": (7.0, 4.0),
    "Entrance":            (0.0, 0.0),
}


class Robot:
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0):
        self.x = start_x
        self.y = start_y
        self.heading_deg = 0.0
        self.current_zone = "Entrance"

    def move_to(self, x: float, y: float, zone_name: str = "", speed_units_per_sec: float = 2.0):
        """Move (in simulated steps) from current position to (x, y).
        Prints progress so it's visible during a demo/review."""
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.heading_deg = math.degrees(math.atan2(dy, dx))

        steps = max(1, int(dist / 0.5))
        for i in range(steps):
            self.x += dx / steps
            self.y += dy / steps
            time.sleep(0.05)  # small delay so movement is visible, not instantaneous

        self.x, self.y = x, y
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
