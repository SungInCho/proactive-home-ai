import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import ai2thor.controller

from tracking.scene_graph import SceneGraph
from inference.task_model import TaskInferenceModel

MODEL_PATH   = Path("data/models/best_model.pt")
DATASET_PATH = Path("data/dataset/dataset.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

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
OBJ2IDX    = {o: i for i, o in enumerate(OBJECT_TYPES)}
IDX2ACTION = {0: "close", 1: "clean", 2: "none"}

# FloorPlans NOT used in training
TEST_FLOOR_PLANS = [
    "FloorPlan201",  
    "FloorPlan202",
    "FloorPlan301",  
    "FloorPlan302",
    "FloorPlan401",  
]


def state_to_vector(state) -> list:
    def b(val):
        if val is True:  return 1.0
        if val is False: return 0.0
        return -1.0
    return [b(state.is_open), b(state.is_dirty), b(state.visible)]


def run_zero_shot(floor_plan: str, model, n_steps: int = 20) -> dict:
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=floor_plan,
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        width=300,
        height=300,
        fieldOfView=90,
    )

    graph = SceneGraph()
    action_counts = defaultdict(int)

    import random
    for _ in range(n_steps):
        action = random.choice([
            "MoveAhead", "MoveAhead", "RotateRight", "RotateLeft"
        ])
        event = controller.step(action=action)
        graph.update(event.metadata["objects"])

        # randomly open/dirty objects
        if random.random() < 0.3:
            openable = [o for o in event.metadata["objects"]
                       if o.get("openable") and o.get("visible") and not o.get("isOpen")]
            if openable:
                obj = random.choice(openable)
                controller.step(action="OpenObject",
                               objectId=obj["objectId"], forceAction=True)

        if random.random() < 0.3:
            dirtyable = [o for o in event.metadata["objects"]
                        if o.get("dirtyable") and o.get("visible") and not o.get("isDirty")]
            if dirtyable:
                obj = random.choice(dirtyable)
                controller.step(action="DirtyObject",
                               objectId=obj["objectId"], forceAction=True)

    # run inference on final state
    predictions = []
    for oid, state in graph.objects.items():
        obj_type = state.object_type
        if obj_type not in OBJ2IDX:
            continue

        obj_onehot = [0.0] * len(OBJECT_TYPES)
        obj_onehot[OBJ2IDX[obj_type]] = 1.0
        features = obj_onehot + state_to_vector(state)

        x = torch.tensor([features], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1).item()

        action = IDX2ACTION[pred]
        action_counts[action] += 1
        if action != "none":
            predictions.append({
                "object_type": obj_type,
                "action":      action,
                "is_open":     state.is_open,
                "is_dirty":    state.is_dirty,
            })

    controller.stop()
    return {
        "floor_plan":   floor_plan,
        "predictions":  predictions,
        "action_counts": dict(action_counts),
    }


if __name__ == "__main__":
    with open(DATASET_PATH) as f:
        data = json.load(f)

    model = TaskInferenceModel(data["n_features"], data["n_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("=== Zero-shot Test on Unseen FloorPlans ===\n")
    all_results = []

    for fp in TEST_FLOOR_PLANS:
        result = run_zero_shot(fp, model)
        all_results.append(result)

        print(f"FloorPlan: {fp}")
        print(f"  Action counts: {result['action_counts']}")
        print(f"  Actions needed ({len(result['predictions'])}):")
        for p in result["predictions"][:5]:
            print(f"    {p['object_type']:<20} → {p['action']}"
                  f" (open={p['is_open']}, dirty={p['is_dirty']})")
        print()

    # summary
    print("=== Summary ===")
    total_actions = sum(
        sum(r["action_counts"].get(a, 0) for a in ["close", "clean"])
        for r in all_results
    )
    total_objects = sum(
        sum(r["action_counts"].values()) for r in all_results
    )
    print(f"Total objects evaluated: {total_objects}")
    print(f"Total actions predicted: {total_actions}")
    print(f"Action rate: {total_actions/total_objects*100:.1f}%")