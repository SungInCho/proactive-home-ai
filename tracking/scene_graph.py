import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

@dataclass
class ObjectState:
    object_id: str
    object_type: str
    position: dict   # {x, y, z}
    is_open: Optional[bool] = None
    is_dirty: Optional[bool] = None
    is_picked_up: Optional[bool] = None
    visible: bool = False
    timestamp: float = field(default_factory=time.time) # record time

class SceneGraph:
    def __init__(self):
        # current state of the object
        self.objects: dict[str, ObjectState] = {}
        # history of states of the object
        self.history: dict[str, list[ObjectState]] = defaultdict(list) # creates list when key doesn't exist

    def update(self, metadata_objects: list[dict]):
        """
        Update scene graph from AI2-THOR ObjectMetadata list.
        Call this every step.
        """
        now = time.time()
        for obj in metadata_objects:
            oid = obj["objectId"]
            new_state = ObjectState(
                object_id = oid,
                object_type = obj["objectType"],
                position = obj["position"],
                is_open = obj.get("isOpen"),
                is_dirty = obj.get("isDirty"),
                is_picked_up = obj.get("isPickedUp"),
                visible = obj.get("visible", False),
                timestamp = now
            )

            if oid in self.objects:
                prev = self.objects[oid]
                if self._state_changed(prev, new_state):
                    self.history[oid].append(prev)
                
            self.objects[oid] = new_state

    def _state_changed(self, prev: ObjectState, curr: ObjectState) -> bool:
        """Detect meaningful state changes"""
        return (
            prev.is_open != curr.is_open or
            prev.is_dirty != curr.is_dirty or
            prev.is_picked_up != curr.is_picked_up or
            prev.visible != curr.visible
        )
    
    def get_object(self, object_type: str) -> list[ObjectState]:
        """Get all objects of a given type."""
        return [o for o in self.objects.values()
                if o.object_type.lower() == object_type.lower()]
    
    def get_anomalies(self, time_threshold: float = 300.0) -> list[dict]:
        """
        Find objects in abnormal states for longer than threshold (seconds).
        e.g. drawer left open, cup left out of place.
        """
        now = time.time()
        anomalies = []

        for oid, state in self.objects.items():
            elapsed = now - state.timestamp
            reason = None

            if state.is_open is True and elapsed > time_threshold:
                reason = f"open for {elapsed: .0f}s"
            elif state.is_dirty is True and elapsed > time_threshold:
                reason = f"dirty for {elapsed: .0f}s"
            elif state.is_picked_up is True and elapsed > time_threshold:
                reason = f"picked up for {elapsed:.0f}s"
            
            if reason:
                anomalies.append({
                    "object_id": oid,
                    "object_type": state.object_type,
                    "reason": reason,
                    "position": state.position
                })

        return anomalies
    
    def print_summary(self):
        """Print current state of all visible objects."""
        visible = [o for o in self.objects.values() if o.visible]
        print(f"\n=== Scene Graph — {len(self.objects)} objects "
              f"({len(visible)} visible) ===")
        print(f"{'Type':<25} {'Open':^6} {'Dirty':^6} {'Picked':^6} {'Visible':^8}")
        print("-" * 58)
        for o in sorted(self.objects.values(), key=lambda x: x.object_type):
            print(f"{o.object_type:<25} {str(o.is_open):^6} "
                  f"{str(o.is_dirty):^6} {str(o.is_picked_up):^6} "
                  f"{str(o.visible):^8}")
    
    def print_history(self, object_type: str):
        """Print state change history for a given object type."""
        matches = [oid for oid, o in self.objects.items()
                   if o.object_type.lower() == object_type.lower()]
        if not matches:
            print(f"No objects of type '{object_type}' found.")
            return

        for oid in matches:
            history = self.history.get(oid, [])
            current = self.objects[oid]
            print(f"\n=== History: {oid} ===")
            for h in history:
                t = time.strftime("%H:%M:%S", time.localtime(h.timestamp))
                print(f"  [{t}] open={h.is_open} dirty={h.is_dirty} "
                      f"visible={h.visible}")
            t = time.strftime("%H:%M:%S", time.localtime(current.timestamp))
            print(f"  [{t}] open={current.is_open} dirty={current.is_dirty} "
                  f"visible={current.visible} ← current")