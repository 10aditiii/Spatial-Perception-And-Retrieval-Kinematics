"""
Cognitive spatial memory module.

Stores what was seen, where, and when -- and supports:
  - remember(item, location): record/update an item's last known location
  - recall(item): look up an item's stored location
  - mark_missing(item): item wasn't found where memory said it would be
  - smart_search_order(item, current_pos, all_known_locations): rank which
    other locations to check next, using distance, recency, and semantic proximity.

Persisted to a SQLite database so memory survives between runs.
"""
import re
import sqlite3
import math
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
MEMORY_FILE = config["memory"]["file_path"]

# Grouping of Freiburg Groceries Dataset classes (plus common related items a
# user might search for) into semantic categories, so a search for something
# the robot has never seen can still be steered toward the right area.
SEMANTIC_CATEGORIES = {
    "beverage": ["water", "soda", "juice", "milk", "tea", "coffee", "ethanol",
                 "cola", "coke", "pepsi", "sprite", "fanta", "lemonade", "coconut",
                 "smoothie", "drink", "beer", "wine", "latte", "espresso", "kombucha"],
    "snacks_and_sweets": ["chips", "candy", "chocolate", "cake", "nuts",
                          "cookie", "biscuit", "cracker", "popcorn", "pretzel",
                          "gummy", "brownie", "muffin", "donut", "doughnut"],
    "pantry_staples": ["pasta", "rice", "cereal", "flour", "beans", "corn",
                       "bread", "oats", "oatmeal", "noodle", "lentil", "quinoa",
                       "couscous", "macaroni", "spaghetti", "penne"],
    "condiments_and_spices": ["spices", "sugar", "oil", "vinegar", "tomato sauce", "honey",
                              "jam", "salt", "sodium chloride", "ketchup", "mustard",
                              "mayonnaise", "mayo", "soy sauce", "syrup"],
    "perishables": ["fish", "meat", "poultry", "chicken", "beef", "pork", "egg",
                    "cheese", "yogurt", "yoghurt", "butter", "salmon", "tuna", "shrimp"],
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def get_item_category(item_name: str) -> str:
    """Whole-word match (optionally plural) so 'tea' matches 'green tea' but
    not 'steak', and 'tomato_sauce' matches 'tomato sauce'."""
    text = _normalise(item_name)
    for category, items in SEMANTIC_CATEGORIES.items():
        for term in items:
            if re.search(rf"\b{re.escape(_normalise(term))}(?:s|es)?\b", text):
                return category
    return "unknown"


# Which item category each aisle is meant for (matched on the aisle's name).
# Used only to warn when a scan's result doesn't fit the aisle it was filed under.
ZONE_KEYWORD_TO_CATEGORY = {
    "beverage": "beverage",
    "snack": "snacks_and_sweets",
    "grain": "pantry_staples",
    "condiment": "condiments_and_spices",
}


def zone_category(zone_name: str):
    name = zone_name.lower()
    for keyword, category in ZONE_KEYWORD_TO_CATEGORY.items():
        if keyword in name:
            return category
    return None


def suggest_zone(item_name: str, zone_names):
    """The aisle this item normally belongs in, or None if the item's category
    is unknown or no aisle is dedicated to it."""
    category = get_item_category(item_name)
    if category == "unknown":
        return None
    for zone in zone_names:
        if zone_category(zone) == category:
            return zone
    return None


@dataclass
class MemoryEntry:
    item: str
    x: float
    y: float
    last_seen: float          # unix timestamp
    confidence: float         # classifier confidence at time of sighting
    zone: str = ""            # optional human-readable area name, e.g. "Aisle 3"
    times_seen: int = 1


class CognitiveMemory:
    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        # Using sqlite3 for better scalability and multi-threading safety in the future
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS memory (
                    item_key TEXT PRIMARY KEY,
                    original_name TEXT,
                    x REAL,
                    y REAL,
                    last_seen REAL,
                    confidence REAL,
                    zone TEXT,
                    times_seen INTEGER
                )
            ''')

    # ---------- core operations ----------
    def remember(self, item: str, x: float, y: float, confidence: float, zone: str = ""):
        """Record or update an item's location. Called every time the
        detect+classify pipeline confirms an item at a given robot position."""
        key = item.lower()
        now = time.time()
        
        with self.conn:
            cursor = self.conn.execute("SELECT times_seen, zone FROM memory WHERE item_key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                times_seen = row['times_seen'] + 1
                # Keep old zone if new zone is empty
                final_zone = zone if zone else row['zone']
                self.conn.execute('''
                    UPDATE memory 
                    SET original_name = ?, x = ?, y = ?, last_seen = ?, confidence = ?, zone = ?, times_seen = ?
                    WHERE item_key = ?
                ''', (item, x, y, now, confidence, final_zone, times_seen, key))
            else:
                self.conn.execute('''
                    INSERT INTO memory (item_key, original_name, x, y, last_seen, confidence, zone, times_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (key, item, x, y, now, confidence, zone, 1))

    def recall(self, item: str) -> Optional[MemoryEntry]:
        """Look up the last known location of an item. Returns None if
        never seen before."""
        key = item.lower()
        cursor = self.conn.execute("SELECT * FROM memory WHERE item_key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return MemoryEntry(
                item=row['original_name'],
                x=row['x'],
                y=row['y'],
                last_seen=row['last_seen'],
                confidence=row['confidence'],
                zone=row['zone'],
                times_seen=row['times_seen']
            )
        return None

    def all_items(self) -> list[MemoryEntry]:
        cursor = self.conn.execute("SELECT * FROM memory")
        results = []
        for row in cursor:
            results.append(MemoryEntry(
                item=row['original_name'],
                x=row['x'],
                y=row['y'],
                last_seen=row['last_seen'],
                confidence=row['confidence'],
                zone=row['zone'],
                times_seen=row['times_seen']
            ))
        return results

    def clear_all(self):
        """Wipes the entire cognitive memory database."""
        with self.conn:
            self.conn.execute("DELETE FROM memory")

    def forget_item(self, item: str):
        """Deletes a specific item from the cognitive memory database."""
        key = item.lower()
        with self.conn:
            self.conn.execute("DELETE FROM memory WHERE item_key = ?", (key,))

    # ---------- smart search when item is NOT where memory said ----------
    def smart_search_order_detailed(self, item: str, current_x: float, current_y: float,
                                     max_candidates: int = 5) -> list[dict]:
        """
        The item wasn't at its remembered spot. Instead of searching randomly,
        rank other known locations by a combination of:
          - distance from the robot's current position (closer = check first)
          - recency of last sighting of ANYTHING in that zone (more recently active)
          - semantic proximity (items of the same category are clustered together)

        Returns the full score breakdown per candidate (not just the ranked
        entries) so the reasoning can be shown, not just the final answer.
        """
        target_category = get_item_category(item)
        candidates = [e for e in self.all_items() if e.item.lower() != item.lower()]

        scored = []
        for e in candidates:
            dist = math.hypot(e.x - current_x, e.y - current_y)
            age_hours = (time.time() - e.last_seen) / 3600.0
            age_penalty = age_hours * 0.5

            e_category = get_item_category(e.item)
            semantic_bonus = 5.0 if (target_category != "unknown" and e_category == target_category) else 0.0

            total_score = dist + age_penalty - semantic_bonus
            scored.append({
                "entry": e,
                "distance": dist,
                "age_hours": age_hours,
                "age_penalty": age_penalty,
                "category": e_category,
                "semantic_bonus": semantic_bonus,
                "score": total_score,
            })

        scored.sort(key=lambda s: s["score"])
        return scored[:max_candidates]

    def smart_search_order(self, item: str, current_x: float, current_y: float,
                            max_candidates: int = 5) -> list[MemoryEntry]:
        """Same ranking as smart_search_order_detailed, but returns just the
        ranked MemoryEntry list (kept for backward compatibility)."""
        detailed = self.smart_search_order_detailed(item, current_x, current_y, max_candidates)
        return [d["entry"] for d in detailed]

    def mark_missing(self, item: str):
        """Optional: flag that the item wasn't found at its last known spot.
        Keeps the old location on record (still useful as a search hint)
        but lowers confidence so recall() callers know to double check."""
        key = item.lower()
        with self.conn:
            self.conn.execute('''
                UPDATE memory 
                SET confidence = confidence * 0.5
                WHERE item_key = ?
            ''', (key,))

    def close(self):
        """Close the database connection."""
        self.conn.close()


if __name__ == "__main__":
    # quick manual test
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
        
    mem = CognitiveMemory("test_memory.db")
    
    # Insert a few items
    mem.remember("Water", x=3.2, y=1.5, confidence=0.94, zone="Aisle 1 - Beverages")
    mem.remember("Juice", x=3.5, y=1.5, confidence=0.88, zone="Aisle 1 - Beverages")
    
    # We put bread very close to where we will query from (3.0, 1.0)
    mem.remember("Bread", x=3.0, y=1.0, confidence=0.9, zone="Aisle 4 - Grains") 
    
    found = mem.recall("water")
    print("Recall test:", found.item if found else None)

    # When looking for "Water" from (3.0, 1.0), Bread is closer (dist ~0)
    # But Juice (dist ~0.7) should be ranked higher because of the semantic bonus!
    order = mem.smart_search_order("Water", current_x=3.0, current_y=1.0)
    print("Search order if Water is missing:", [e.item for e in order])

    mem.close()
    os.remove("test_memory.db")
