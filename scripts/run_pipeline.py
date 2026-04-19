import json
import torch
import numpy as np
import cv2
import time
from pathlib import Path
import ai2thor.controller

from tracking.scene_graph import SceneGraph
from safety.human_safety import HumanSafetyModule, SafetyState
from inference.task_model import TaskInferenceModel

# ── Config ──
MODEL_PATH   = Path("data/models/best_model.pt")
DATASET_PATH = Path("data/dataset/dataset.json")
SAVE_DIR     = Path("data/pipeline_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
IDX2ACTION = {0: "close", 1: "clean", 2: "none"}

# AI2-THOR action mapping
ACTION_MAP = {
    "close": "CloseObject",
    "clean": "CleanObject",
}


def load_model(n_features: int, n_classes: int) -> TaskInferenceModel:
    model = TaskInferenceModel(n_features, n_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def state_to_vector(state: dict) -> list:
    def b(val):
        if val is True:  return 1.0
        if val is False: return 0.0
        return -1.0
    return [b(state.get("is_open")), b(state.get("is_dirty")), b(state.get("visible"))]


def infer_actions(model, objects: dict) -> list:
    """Run inference on all objects and return list of needed actions."""
    results = []
    for oid, state in objects.items():
        obj_type = state.object_type
        if obj_type not in OBJ2IDX:
            continue

        obj_onehot = [0.0] * len(OBJECT_TYPES)
        obj_onehot[OBJ2IDX[obj_type]] = 1.0

        state_dict = {
            "is_open":     state.is_open,
            "is_dirty":    state.is_dirty,
            "visible":     state.visible,
        }
        features = obj_onehot + state_to_vector(state_dict)
        x = torch.tensor([features], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(1).item()
            conf   = probs[0][pred].item()

        action = IDX2ACTION[pred]
        if action != "none":
            results.append({
                "object_id":   oid,
                "object_type": obj_type,
                "action":      action,
                "confidence":  round(conf, 3),
            })
    return results


def execute_action(controller, action: str, object_id: str) -> bool:
    """Execute inferred action in AI2-THOR."""
    thor_action = ACTION_MAP.get(action)
    if not thor_action:
        return False
    event = controller.step(
        action=thor_action,
        objectId=object_id,
        forceAction=True,
    )
    return event.metadata["lastActionSuccess"]


def draw_overlay(frame: np.ndarray, inferences: list,
                 safety_state: SafetyState, step: int) -> np.ndarray:
    """Draw inference results and safety state on frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # safety state banner
    color_map = {
        SafetyState.PROCEED: (0, 200, 0),
        SafetyState.SLOW:    (0, 165, 255),
        SafetyState.STOP:    (0, 0, 255),
    }
    color = color_map[safety_state]
    cv2.rectangle(img, (0, 0), (w, 30), color, -1)
    cv2.putText(img, f"SAFETY: {safety_state.value}  Step: {step}",
                (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # inference results
    y_offset = 55
    for inf in inferences[:5]:
        text = f"{inf['object_type']}: {inf['action']} ({inf['confidence']:.2f})"
        cv2.putText(img, text, (5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 22

    return img


def run_pipeline(floor_plan: str = "FloorPlan1", steps: int = 15):
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=floor_plan,
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        width=640,
        height=480,
        fieldOfView=90,
    )

    # load model
    with open(DATASET_PATH) as f:
        data = json.load(f)
    model  = load_model(data["n_features"], data["n_classes"])
    graph  = SceneGraph()
    safety = HumanSafetyModule()

    actions = [
        "MoveAhead", "MoveAhead", "RotateRight",
        "MoveAhead", "MoveAhead", "RotateRight",
        "MoveAhead", "RotateLeft", "MoveAhead",
        "MoveAhead", "RotateRight", "MoveAhead",
        "MoveAhead", "MoveAhead", "RotateLeft",
    ]

    frames = []
    executed_this_episode = set()
    print(f"\n[INFO] Running pipeline on {floor_plan}\n")

    for step, action in enumerate(actions[:steps]):
        event = controller.step(action=action)
        graph.update(event.metadata["objects"])
        safety_state = safety.update(event.metadata["objects"])

        # run inference
        inferences = []
        if safety.is_safe_to_act():
            inferences = infer_actions(model, graph.objects)

            # execute top action
            if inferences:
                pending = [i for i in inferences
                        if i["object_id"] not in executed_this_episode]
                if pending:
                    top = pending[0]
                    success = execute_action(
                        controller, top["action"], top["object_id"])
                    if success:
                        executed_this_episode.add(top["object_id"])
                    print(f"Step {step:02d} | {action:<12} | "
                        f"safety={safety_state.value:<8} | "
                        f"action={top['action']} on {top['object_type']} "
                        f"(conf={top['confidence']:.2f}) | "
                        f"executed={success}")
                else:
                    print(f"Step {step:02d} | {action:<12} | "
                        f"safety={safety_state.value:<8} | all actions done")
            else:
                print(f"Step {step:02d} | {action:<12} | "
                      f"safety={safety_state.value:<8} | no action needed")
        else:
            print(f"Step {step:02d} | {action:<12} | "
                  f"safety={safety_state.value:<8} | BLOCKED")

        # save frame with overlay
        frame = event.frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_overlay = draw_overlay(frame_bgr, inferences, safety_state, step)
        frames.append(frame_overlay)
        cv2.imwrite(str(SAVE_DIR / f"step{step:03d}.png"), frame_overlay)

    # save GIF
    gif_path = str(SAVE_DIR / f"{floor_plan}_pipeline.gif")
    import imageio
    imageio.mimsave(gif_path,
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                    fps=3)
    print(f"\n[DONE] GIF saved: {gif_path}")
    controller.stop()


if __name__ == "__main__":
    run_pipeline(floor_plan="FloorPlan1", steps=15)