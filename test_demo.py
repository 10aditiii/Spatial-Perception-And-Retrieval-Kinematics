from cognitive_memory import CognitiveMemory
import os

def run_test():
    db_file = "demo_test.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    mem = CognitiveMemory(db_file)
    print("--- 🤖 Simulating Scan Phase ---")
    mem.remember("Water", x=1.0, y=1.0, confidence=0.95, zone="Aisle 1 - Beverages")
    mem.remember("Juice", x=1.5, y=1.0, confidence=0.92, zone="Aisle 1 - Beverages")
    mem.remember("Bread", x=5.0, y=5.0, confidence=0.88, zone="Aisle 4 - Grains")
    mem.remember("Pasta", x=5.5, y=5.0, confidence=0.90, zone="Aisle 4 - Grains")
    mem.remember("Soap", x=8.0, y=8.0, confidence=0.98, zone="Aisle 6 - Household")
    
    print("Items in memory:")
    for entry in mem.all_items():
        print(f"  - {entry.item} at {entry.zone} ({entry.x}, {entry.y})")
        
    print("\n--- 🔍 Simulating Query Phase ---")
    query = "Water"
    found = mem.recall(query)
    print(f"User asks: 'Where is the {query}?'")
    print(f"Robot remembers: {found.zone} at ({found.x}, {found.y})")
    
    print(f"\n[Robot moves to ({found.x}, {found.y}) and scans...]")
    print(f"Uh oh, the {query} is not there!")
    
    print("\n--- 🧠 Simulating Smart Search ---")
    # Simulate current robot position at the empty Water location
    current_x, current_y = found.x, found.y
    
    print(f"Calculating search order from ({current_x}, {current_y}) for missing '{query}'...")
    order = mem.smart_search_order(query, current_x, current_y)
    
    for i, candidate in enumerate(order, 1):
        print(f"  Check #{i}: {candidate.zone} (because it has '{candidate.item}')")

    print("\nNote: It checks Juice first because they share the 'beverage' category, even though Bread might be close!")

    mem.close()
    if os.path.exists(db_file):
        os.remove(db_file)

if __name__ == "__main__":
    run_test()
