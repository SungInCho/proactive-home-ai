import time
import ai2thor.controller
from tracking.scene_graph import SceneGraph

def run_scene_tracker():
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=True,
        width=640,
        height=480,
        fieldOfView=90,
    )

    graph = SceneGraph()

    actions = [
        "MoveAhead", "MoveAhead", "RotateRight",
        "MoveAhead", "RotateRight", "MoveAhead",
        # simulate object interaction
        "OpenObject",
        "MoveAhead", "MoveAhead",
        "RotateLeft", "MoveAhead",
    ]

    print("[INFO] Starting scene tracking...\n")

    for step, action in enumerate(actions):
        if action == "OpenObject":
            # open nearest visible openable object
            event = controller.step(
                action=action,
                objectId = _get_nearest_openable(controller),
                forceAction=True
            )
        else:
            event = controller.step(action=action)

        # update scene graph every step
        graph.update(event.metadata["objects"])

        print(f"Step {step:02d} | action={action:12s} | "
            f"success={event.metadata['lastActionSuccess']}")
        
        # check anomalies every 3 steps (using low threshold for demo)
        if step % 3 == 0:
            anomalies = graph.get_anomalies(time_threshold=0.0)
            if anomalies:
                print(f"  [ANOMALY] {len(anomalies)} objects in abnormal state:")
                for a in anomalies:
                    print(f"    - {a['object_type']}: {a['reason']}")

        time.sleep(0.1)

    print("\n")
    graph.print_summary()

    # show history for drawers
    graph.print_history("Book")

    controller.stop()

def _get_nearest_openable(controller) -> str:
    """Get objectId of nearest visible openable object"""
    objects = controller.last_event.metadata["objects"]
    openable = [
        o for o in objects
        if o.get("openable") 
        and o.get("visible") 
        and not o.get("isOpen") 
    ]
    if not openable:
        return ""
    # return closest one
    return min(openable, key=lambda o: o.get("distance", 999))["objectId"]

# Run this code only when the file is executed directly, not when imported.
if __name__ == "__main__":
    run_scene_tracker()