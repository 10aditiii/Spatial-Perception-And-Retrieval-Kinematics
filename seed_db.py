import json
from cognitive_memory import CognitiveMemory
import os

def seed_database():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    db_path = config["memory"]["file_path"]
    waypoints = config["world"]["waypoints"]

    mem = CognitiveMemory(db_path)
    mem.clear_all()

    # Define some initial items that match the semantic categories
    items = [
        # Beverages (Aisle 1)
        ("WATER", "Aisle 1 - Beverages"),
        ("JUICE", "Aisle 1 - Beverages"),
        ("COFFEE", "Aisle 1 - Beverages"),
        ("TEA", "Aisle 1 - Beverages"),
        ("SODA", "Aisle 1 - Beverages"),
        # Snacks (Aisle 2)
        ("CHIPS", "Aisle 2 - Snacks"),
        ("CANDY", "Aisle 2 - Snacks"),
        ("NUTS", "Aisle 2 - Snacks"),
        ("COOKIES", "Aisle 2 - Snacks"),
        # Dairy (Aisle 3)
        ("CHEESE", "Aisle 3 - Dairy"),
        ("YOGURT", "Aisle 3 - Dairy"),
        ("BUTTER", "Aisle 3 - Dairy"),
        ("MILK", "Aisle 3 - Dairy"),
        # Grains (Aisle 4)
        ("BREAD", "Aisle 4 - Grains"),
        ("PASTA", "Aisle 4 - Grains"),
        ("RICE", "Aisle 4 - Grains"),
        ("CEREAL", "Aisle 4 - Grains"),
        # Condiments (Aisle 5)
        ("KETCHUP", "Aisle 5 - Condiments"),
        ("MUSTARD", "Aisle 5 - Condiments"),
        ("MAYO", "Aisle 5 - Condiments"),
        ("VINEGAR", "Aisle 5 - Condiments"),
        ("TOMATO_SAUCE", "Aisle 5 - Condiments"),
        # Household (Aisle 6)
        ("SOAP", "Aisle 6 - Household"),
        ("SPONGE", "Aisle 6 - Household"),
        ("TISSUE", "Aisle 6 - Household"),
        ("CLEANER", "Aisle 6 - Household")
    ]

    for item, zone in items:
        if zone in waypoints:
            x, y = waypoints[zone]
            mem.remember(item, x, y, 0.95, zone)
            print(f"Seeded {item} at {zone}")

    print("✅ Database successfully seeded with dummy data!")

if __name__ == "__main__":
    seed_database()
