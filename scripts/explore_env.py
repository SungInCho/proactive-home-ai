import ai2thor.controller
import numpy as np
import cv2
import json
import os
from pathlib import Path

SAVE_DIR = Path("data/episodes/d1_exploration")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def explore_floorplan(floor_plan: str = "FloorPlan1", steps: int = 10):
    """
    Moves the agent in AI2-THOR environment
    and stores RGB, Debth, ObjectMetaData
    """
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=floor_plan,

        # Agent movement restraint
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,

        # Camera settings
        renderDepthImage=True,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=90
    )

    print(f"[INFO] Scene: {floor_plan} load complete..")
    print(f"[INFO] Total Object #: {len(controller.last_event.metadata['objects'])}")

    actions = [
        "MoveAhead", "MoveAhead", "RotateRight",
        "MoveAhead", "MoveAhead", "RotateRight",
        "MoveAhead", "RotateLeft", "MoveAhead", "MoveAhead"
    ]

    for step, action in enumerate(actions[:steps]):
        event = controller.step(action=action)

        # --- Save RGB Image ---
        rgb = event.frame # (H, W, 3) numpy array
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(SAVE_DIR / f"step{step:03d}_rgb.png"), rgb_bgr)

        #  --- Save Depth Image ---
        depth = event.depth_frame # (H, W) float32, metric: meters
        # Regularize to 0-255 for visualization
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        cv2.imwrite(str(SAVE_DIR / f"step{step:03d}_depth.png"), depth_colored)
        # Save raw depth
        np.save(str(SAVE_DIR / f"step{step:03d}_depth_raw.npy"), depth)

        # --- Save ObjectMetaData ---
        objects = event.metadata["objects"]
        # Extract core fields only
        objects_clean = []
        for obj in objects:
            objects_clean.append({
                'objectId': obj['objectId'],
                'objectType': obj['objectType'],
                'position': obj['position'],        # {x, y, z}
                'isOpen': obj.get('isOpen'),
                'isDirty': obj.get('isDirty'),
                'isPickedUp': obj.get('isPickedUp'),
                'distance': obj.get('distance'),
                'visible': obj['visible']
            })

        with open(SAVE_DIR / f"step{step:03d}_metadata.json", "w") as f:
            json.dump({
                'step': step,
                'action': action,
                'action_success': event.metadata['lastActionSuccess'],
                'agent_position': event.metadata['agent']['position'],
                'agent_rotation': event.metadata['agent']['rotation'],
                'objects': objects_clean,
            }, f, indent=2)

        print(f"   Step {step:02d} | action: {action:12s} | "
              f"success={event.metadata['lastActionSuccess']} | "
              f"visible_objects={sum(1 for o in objects if o['visible'])}")

    controller.stop()
    print(f"\n[DONE] Saved RGB, Depth, ObjectMetaData: {SAVE_DIR}")
    return SAVE_DIR


def print_object_summary(save_dir: Path, step: int = 0):
    """
    Print object summary from the saved metadata
    """
    meta_path = SAVE_DIR / f"step{step:03d}_metadata.json"
    with open(meta_path) as f:
        data = json.load(f)

    print(f"\n=== Step {step} Object Summary ===")
    print(f"{'Type':<25} {'Position':^35} {'Open':^6} {'Dirty':^6} {'Visible':^8}")
    print("-" * 85)
    for obj in sorted(data["objects"], key=lambda x: x["objectType"]):
        pos = obj["position"]
        pos_str = f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})"
        print(f"{obj['objectType']:<25} {pos_str:^35} "
              f"{str(obj['isOpen']):^6} {str(obj['isDirty']):^6} {str(obj['visible']):^8}")


if __name__ == "__main__":
    # FloorPlan1 = Kitchen, FloorPlan201 = Living room, FloorPlan301 = Bedroom
    save_dir = explore_floorplan(floor_plan="FloorPlan1", steps=10)
    print_object_summary(save_dir, step=0)