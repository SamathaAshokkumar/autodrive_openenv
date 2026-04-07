"""Driving action handler.

Equivalent to kube's command handler: parses and applies one action at a time.
"""

from __future__ import annotations

from typing import Tuple

from .constants import DRIVING_ACTIONS


class DrivingActionHandler:
    def normalize(self, action: str, value: float = 0.0) -> Tuple[str, float]:
        name = str(action or "wait").strip().lower()
        if name not in DRIVING_ACTIONS:
            return "wait", 0.0
        return name, float(value or 0.0)

    def apply(self, simulator, action: str, value: float) -> str:
        action, value = self.normalize(action, value)
        simulator.last_action = {"action": action, "value": value}
        simulator.decision_log = f"Applied {action}:{value:.2f}"

        if action == "accelerate":
            simulator.ego["speed"] = min(simulator.ego["max_speed"], simulator.ego["speed"] + max(value, 0.2) * 4.0)
            return simulator.decision_log
        if action == "brake":
            simulator.ego["speed"] = max(0.0, simulator.ego["speed"] - max(value, 0.2) * 6.0)
            return simulator.decision_log
        if action == "steer_left":
            simulator.ego["steering"] = max(-25.0, simulator.ego["steering"] - max(value, 5.0))
            simulator.ego["lane_position"] -= 0.35
            return simulator.decision_log
        if action == "steer_right":
            simulator.ego["steering"] = min(25.0, simulator.ego["steering"] + max(value, 5.0))
            simulator.ego["lane_position"] += 0.35
            return simulator.decision_log
        if action == "change_lane_left":
            simulator.ego["lane"] = "left"
            simulator.ego["lane_position"] -= 1.0
            simulator.decision_log = "Changed lane left"
            return simulator.decision_log
        if action == "change_lane_right":
            simulator.ego["lane"] = "right"
            simulator.ego["lane_position"] += 1.0
            simulator.decision_log = "Changed lane right"
            return simulator.decision_log
        if action == "horn":
            simulator.ego["horn"] = 1
            simulator.decision_log = "Horn used"
            return simulator.decision_log

        simulator.ego["horn"] = 0
        simulator.ego["steering"] *= 0.5
        simulator.decision_log = "Maintained cautious wait"
        return simulator.decision_log
