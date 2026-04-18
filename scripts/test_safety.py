import ai2thor.controller
from tracking.scene_graph import SceneGraph
from safety.human_safety import HumanSafetyModule, SafetyState


def run_safety_test():
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        width=300,
        height=300,
        fieldOfView=90,
    )

    graph  = SceneGraph()
    safety = HumanSafetyModule()

    # simulate human approaching and leaving
    scenarios = [
        # (action, simulated_human_distance)
        ("MoveAhead", 5.0),   # human far away → PROCEED
        ("MoveAhead", 4.0),
        ("MoveAhead", 2.5),   # human getting closer → SLOW
        ("MoveAhead", 2.0),
        ("MoveAhead", 1.2),   # human very close → STOP
        ("MoveAhead", 1.0),
        ("MoveAhead", 2.0),   # human moving away → SLOW
        ("MoveAhead", 3.5),   # human far again → PROCEED
        ("MoveAhead", 5.0),
    ]

    print("[INFO] Starting Human Safety Test\n")
    print(f"{'Step':>4} {'Action':<12} {'Human Dist':>10} {'Safety State':<10} {'Can Act?':>8} {'Speed':>6}")
    print("-" * 60)

    for step, (action, human_dist) in enumerate(scenarios):
        event = controller.step(action=action)
        graph.update(event.metadata["objects"])

        # simulate human presence by injecting distance
        safety.human_dist = human_dist
        if human_dist <= 1.5:
            safety.state = SafetyState.STOP
        elif human_dist <= 3.0:
            safety.state = SafetyState.SLOW
        else:
            safety.state = SafetyState.PROCEED

        can_act      = safety.is_safe_to_act()
        speed_factor = safety.get_speed_factor()

        print(f"{step:>4} {action:<12} {human_dist:>9.1f}m "
              f"{safety.state.value:<10} {str(can_act):>8} {speed_factor:>6.1f}")

    print("\n[INFO] Safety state transitions:")
    print("  5.0m -> PROCEED (speed=1.0) - normal operation")
    print("  2.5m -> SLOW    (speed=0.5) - cautious mode")
    print("  1.2m -> STOP    (speed=0.0) - action blocked")
    print("  3.5m -> PROCEED (speed=1.0) — resumed after human leaves")

    controller.stop()


if __name__ == "__main__":
    run_safety_test()