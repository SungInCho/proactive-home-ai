import json
import random
import time
import ai2thor.controller
from pathlib import Path
from tracking.scene_graph import SceneGraph

SAVE_DIR = Path("data/episodes/scenarios")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FLOOR_PLANS = [f"FloorPlan{i}" for i in range(1, 31)]

OPENABLE_TYPES = [
    "Fridge", "Microwave", "Cabinet",
    "Drawer", "Book", "Box"
]

def run_scenario(floor_plan: str, episode_id: int) -> dict:
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
    event = controller.step("Pass")
    graph.update(event.metadata["objects"])

    # random walk + random interactions
    actions_taken = []
    for _ in range(20):
        action = random.choice([
            "MoveAhead", "MoveAhead", "MoveAhead",
            "RotateRight", "RotateLeft",
        ])
        event = controller.step(action=action)
        graph.update(event.metadata["objects"])
        actions_taken.append(action)

        # randomly open a visible openable object
        if random.random() < 0.3:
            openable = [
                o for o in event.metadata["objects"]
                if o.get("openable")
                and o.get("visible")
                and not o.get("isOpen")
                and o["objectType"] in OPENABLE_TYPES
            ]
            if openable:
                obj = random.choice(openable)
                event = controller.step(
                    action="OpenObject",
                    objectId=obj["objectId"],
                    forceAction=True,
                )
                graph.update(event.metadata["objects"])
                actions_taken.append(f"OpenObject:{obj['objectType']}")
        
        # randomly dirty a visible object
        if random.random() < 0.5:
            dirtyable = [
                o for o in event.metadata["objects"]
                if o.get("dirtyable")
                and o.get("visible")
                and not o.get("isDirty")
            ]
            if dirtyable:
                obj = random.choice(dirtyable)
                event = controller.step(
                    action="DirtyObject",
                    objectId=obj["objectId"],
                    forceAction=True,
                )
                graph.update(event.metadata["objects"])
                actions_taken.append(f"DirtyObject:{obj['objectType']}")


    # collect state sequences and labels
    anomalies = graph.get_anomalies(time_threshold=0.0)
    state_sequence = []
    for oid, state in graph.objects.items():
        state_sequence.append({
            "object_type": state.object_type,
            "is_open":     state.is_open,
            "is_dirty":    state.is_dirty,
            "is_picked_up":state.is_picked_up,
            "visible":     state.visible,
        })

    # label: what action is needed
    action_labels = []
    for a in anomalies:
        if a["reason"].startswith("open"):
            action_labels.append({
                "object_type": a["object_type"],
                "action":      "close",
                "reason":      a["reason"],
            })
        elif a["reason"].startswith("dirty"):
            action_labels.append({
                "object_type": a["object_type"],
                "action":      "clean",
                "reason":      a["reason"],
            })

    episode = {
        "episode_id":     episode_id,
        "floor_plan":     floor_plan,
        "actions_taken":  actions_taken,
        "state_sequence": state_sequence,
        "action_labels":  action_labels,
    }

    controller.stop()
    return episode


if __name__ == "__main__":
    all_episodes = []
    episode_id = 0

    for floor_plan in FLOOR_PLANS:
        print(f"[INFO] Running {floor_plan}...")
        for i in range(10):  # 5 episodes per floor plan
            episode = run_scenario(floor_plan, episode_id)
            all_episodes.append(episode)
            n_labels = len(episode["action_labels"])
            print(f"  Episode {episode_id}: {n_labels} action labels")
            episode_id += 1

    # save
    save_path = SAVE_DIR / "scenarios.json"
    with open(save_path, "w") as f:
        json.dump(all_episodes, f, indent=2)

    print(f"\n[DONE] {len(all_episodes)} episodes saved to {save_path}")
    labeled = [e for e in all_episodes if e["action_labels"]]
    print(f"Episodes with labels: {len(labeled)} / {len(all_episodes)}")