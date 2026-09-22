"""
Cognitive spatial memory module.

Stores what was seen, where, and when -- and supports:
  - remember(item, location): record/update an item's last known location
  - recall(item): look up an item's stored location
  - mark_missing(item): item wasn't found where memory said it would be
  - smart_search_order(item, current_pos, all_known_locations): rank which
    other locations to check next, instead of searching randomly
  - forget_stale(): optional cleanup of very old entries

Persisted to a JSON file so memory survives between runs (the robot doesn't
forget everything when the program restarts).
"""
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


MEMORY_FILE = "spatial_memory.json"


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
        self.entries: dict[str, MemoryEntry] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
            self.entries = {k: MemoryEntry(**v) for k, v in raw.items()}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.entries.items()}, f, indent=2)

    # ---------- core operations ----------
    def remember(self, item: str, x: float, y: float, confidence: float, zone: str = ""):
        """Record or update an item's location. Called every time the
        detect+classify pipeline confirms an item at a given robot position."""
        key = item.lower()
        if key in self.entries:
            self.entries[key].x = x
            self.entries[key].y = y
            self.entries[key].last_seen = time.time()
            self.entries[key].confidence = confidence
            self.entries[key].zone = zone or self.entries[key].zone
            self.entries[key].times_seen += 1
        else:
            self.entries[key] = MemoryEntry(
                item=item, x=x, y=y, last_seen=time.time(),
                confidence=confidence, zone=zone,
            )
        self._save()

    def recall(self, item: str) -> Optional[MemoryEntry]:
        """Look up the last known location of an item. Returns None if
        never seen before."""
        return self.entries.get(item.lower())

    def all_items(self):
        return list(self.entries.values())

    # ---------- smart search when item is NOT where memory said ----------
    def smart_search_order(self, item: str, current_x: float, current_y: float,
                            max_candidates: int = 5) -> list[MemoryEntry]:
        """
        The item wasn't at its remembered spot. Instead of searching randomly,
        rank other known locations by a combination of:
          - distance from the robot's current position (closer = check first)
          - recency of last sighting of ANYTHING in that zone (more recently
            active zones are more likely to still be accurate)
        This does not guess where the missing item specifically is -- it
        orders the search so the robot checks the most efficient places first.
        """
        candidates = [e for e in self.entries.values() if e.item.lower() != item.lower()]

        def score(e: MemoryEntry) -> float:
            dist = math.hypot(e.x - current_x, e.y - current_y)
            age_penalty = (time.time() - e.last_seen) / 3600.0  # hours since last seen
            # lower score = check sooner: prioritize close AND recently-confirmed zones
            return dist + age_penalty * 0.5

        ranked = sorted(candidates, key=score)
        return ranked[:max_candidates]

    def mark_missing(self, item: str):
        """Optional: flag that the item wasn't found at its last known spot.
        Keeps the old location on record (still useful as a search hint)
        but lowers confidence so recall() callers know to double check."""
        key = item.lower()
        if key in self.entries:
            self.entries[key].confidence *= 0.5
            self._save()


if __name__ == "__main__":
    # quick manual test
    mem = CognitiveMemory("test_memory.json")
    mem.remember("Sodium Chloride", x=3.2, y=1.5, confidence=0.94, zone="Shelf A")
    mem.remember("Ethanol", x=5.0, y=2.0, confidence=0.88, zone="Shelf B")

    found = mem.recall("sodium chloride")
    print("Recall test:", found)

    order = mem.smart_search_order("Sodium Chloride", current_x=3.0, current_y=1.0)
    print("Search order if missing:", [e.item for e in order])

    os.remove("test_memory.json")
