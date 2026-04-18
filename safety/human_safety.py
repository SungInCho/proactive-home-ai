from enum import Enum


class SafetyState(Enum):
    PROCEED = "PROCEED"
    SLOW    = "SLOW"
    STOP    = "STOP"


THRESHOLDS = {
    SafetyState.STOP: 1.5,   # meters
    SafetyState.SLOW: 3.0,
}


class HumanSafetyModule:
    def __init__(self):
        self.state = SafetyState.PROCEED
        self.prev_state = SafetyState.PROCEED
        self.human_dist = None

    def update(self, metadata_objects: list[dict]) -> SafetyState:
        """
        Check human distance from metadata and update safety state.
        Returns current SafetyState.
        """
        humans = [
            o for o in metadata_objects
            if o["objectType"] == "Agent" or "person" in o["objectType"].lower()
        ]

        if not humans:
            self.human_dist = None
            self.state = SafetyState.PROCEED
            return self.state

        # closest human
        dist = min(h.get("distance", 999) for h in humans)
        self.human_dist = dist

        if dist <= THRESHOLDS[SafetyState.STOP]:
            self.state = SafetyState.STOP
        elif dist <= THRESHOLDS[SafetyState.SLOW]:
            self.state = SafetyState.SLOW
        else:
            self.state = SafetyState.PROCEED

        return self.state

    def is_safe_to_act(self) -> bool:
        return self.state != SafetyState.STOP

    def get_speed_factor(self) -> float:
        """Return speed multiplier based on safety state."""
        if self.state == SafetyState.STOP: return 0.0
        if self.state == SafetyState.SLOW: return 0.5
        return 1.0

    def log(self):
        dist_str = f"{self.human_dist:.2f}m" if self.human_dist else "N/A"
        print(f"[SAFETY] state={self.state.value:<8} "
              f"human_dist={dist_str} "
              f"speed_factor={self.get_speed_factor()}")