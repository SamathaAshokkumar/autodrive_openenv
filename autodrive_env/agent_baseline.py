"""Simple baseline agents for AutoDrive Gym."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def choose_action(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    sensor = raw_obs.get("sensor_data", {})
    ego = raw_obs.get("ego_state", {})
    environment = raw_obs.get("environment", {})
    objects = sensor.get("objects", []) or []
    min_distance = min((float(obj.get("distance", 999.0)) for obj in objects), default=999.0)
    signal = str(sensor.get("traffic_signal", environment.get("traffic_signal", "none"))).lower()
    lane_info = str(sensor.get("lane_info", environment.get("lane_status", "clear"))).lower()
    speed = float(ego.get("speed", 0.0))
    crossing_hazard = any(bool(obj.get("crossing")) and float(obj.get("distance", 999.0)) < 10.0 for obj in objects)
    moving_into_path = any(
        str(obj.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(obj.get("distance", 999.0)) < 8.0
        for obj in objects
    )

    if signal == "red" and speed > 1.0:
        return {"action": "brake", "value": 0.8}
    if min_distance < 4.0:
        return {"action": "brake", "value": 0.9}
    if crossing_hazard or moving_into_path:
        return {"action": "wait", "value": 0.0}
    if min_distance < 7.0:
        return {"action": "brake", "value": 0.5}
    if "missing" in lane_info or "faded" in lane_info:
        return {"action": "accelerate", "value": 0.2 if speed < 8.0 else 0.1}
    if speed < 3.0 and min_distance > 9.0:
        return {"action": "accelerate", "value": 0.4}
    return {"action": "accelerate", "value": 0.4}


@dataclass
class ModularBaselineAgent:
    name: str = "modular_baseline"

    def act(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        return choose_action(raw_obs)


@dataclass
class ConservativeAgent(ModularBaselineAgent):
    name: str = "conservative_agent"

    def act(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        action = choose_action(raw_obs)
        if action["action"] == "accelerate":
            action["action"] = "wait"
            action["value"] = 0.0
        return action


def available_agents() -> List[ModularBaselineAgent]:
    return [ModularBaselineAgent(), ConservativeAgent()]
