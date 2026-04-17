import json
import pandas as pd
from pathlib import Path

SCENARIOS_PATH = Path("data/episodes/scenarios/scenarios.json")
SAVE_DIR       = Path("data/dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# action label -> integer
ACTION2IDX = {
    "close": 0,
    "clean": 1, 
    "none": 2
}
IDX2ACTION = {v: k for k, v in ACTION2IDX.items()}

OBJECT_TYPES = [
    "Apple", "Book", "Bottle", "Bowl", "Bread", "ButterKnife",
    "Cabinet", "CellPhone", "Chair", "CoffeeMachine", "CounterTop",
    "CreditCard", "Cup", "DiningTable", "DishSponge", "Drawer",
    "Egg", "Faucet", "Floor", "Fork", "Fridge", "GarbageCan",
    "HousePlant", "Kettle", "Knife", "Ladle", "Lettuce", "LightSwitch",
    "Microwave", "Mug", "Pan", "PaperTowelRoll", "PepperShaker",
    "Plate", "Pot", "Potato", "SaltShaker", "Shelf", "ShelvingUnit",
    "SideTable", "Sink", "SinkBasin", "SoapBottle", "Spatula",
    "Spoon", "Statue", "Stool", "StoveBurner", "StoveKnob",
    "Toaster", "Tomato", "Vase", "Window", "WineBottle",
    "Pen", "SprayBottle", "Curtains", "Pencil", "Blinds",
    "GarbageBag", "Safe", "AluminumFoil", "Mirror",
]
OBJ2IDX = {o: i for i, o in enumerate(OBJECT_TYPES)}

def state_to_vector(state: dict) -> list:
    """
    Convert a single object state dict to a fixed-size feature vector.
    [is_open, is_dirty, visible] — all as float (0.0 or 1.0, -1.0 for None)
    """
    def b(val):
        if val is True:  return 1.0
        if val is False: return 0.0
        return -1.0

    return [
        b(state.get("is_open")),
        b(state.get("is_dirty")),
        b(state.get("visible")),
    ]

def episode_to_samples(episode: dict) -> list:
    """
    Convert one episode into training samples.
    Each sample: (feature_vector, action_label)

    For objects with action labels -> use that label
    For all other objects          -> label = 'none'
    """
    samples = []
    labeled_types = {l["object_type"]: l["action"] for l in episode["action_labels"]}

    for state in episode["state_sequence"]:
        obj_type = state["object_type"]
        if obj_type not in OBJ2IDX:
            continue

        features = state_to_vector(state)
        # one-hot object type
        obj_onehot = [0.0] * len(OBJECT_TYPES)
        obj_onehot[OBJ2IDX[obj_type]] = 1.0

        feature_vector = obj_onehot + features  # len = 64 + 3 = 67

        action = labeled_types.get(obj_type, "none")
        label  = ACTION2IDX.get(action, ACTION2IDX["none"])

        samples.append({
            "episode_id": episode["episode_id"],
            "floor_plan": episode["floor_plan"],
            "object_type": obj_type,
            "features": feature_vector,
            "label": label,
            "action": action
        })

    return samples


if __name__ == "__main__":
    with open(SCENARIOS_PATH) as f:
        episodes = json.load(f)

    all_samples = []
    for ep in episodes:
        all_samples.extend(episode_to_samples(ep))

    print(f"Total samples: {len(all_samples)}")

    # class distribution
    from collections import Counter
    label_counts = Counter(s["action"] for s in all_samples)
    print("\nClass distribution:")
    for action, count in sorted(label_counts.items()):
        pct = count / len(all_samples) * 100
        print(f"  {action:<10} {count:>5} ({pct:.1f}%)")

    # save as JSON
    save_path = SAVE_DIR / "dataset.json"
    with open(save_path, "w") as f:
        json.dump({
            "samples":    all_samples,
            "action2idx": ACTION2IDX,
            "idx2action": IDX2ACTION,
            "n_features": 67,
            "n_classes":  len(ACTION2IDX),
        }, f, indent=2)
    print(f"\n[DONE] Saved {len(all_samples)} samples to {save_path}")

    # also save as CSV for easy inspection
    df = pd.DataFrame([{
        "episode_id":  s["episode_id"],
        "floor_plan":  s["floor_plan"],
        "object_type": s["object_type"],
        "action":      s["action"],
        "label":       s["label"],
    } for s in all_samples])
    df.to_csv(SAVE_DIR / "dataset.csv", index=False)
    print(f"CSV saved to {SAVE_DIR / 'dataset.csv'}")