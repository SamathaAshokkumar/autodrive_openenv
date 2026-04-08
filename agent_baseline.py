"""Baseline agents for AutoDrive Gym.

ModularBaselineAgent — stage-aware heuristic agent that:
  1. Brakes / waits while hazard is approaching and distance is low.
  2. Accelerates once the stage is 'clearing' or 'cleared'.
  3. Handles sudden alerts (ambulance, animal, police) with context-specific actions.
  4. Never oscillates more than 3 identical steps in a row — forces a different action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def choose_action(raw_obs: Dict[str, Any], history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    sensor      = raw_obs.get("sensor_data", {}) or {}
    ego         = raw_obs.get("ego_state", {}) or {}
    environment = raw_obs.get("environment", {}) or {}
    alerts      = raw_obs.get("active_alerts", []) or []
    hint        = str(raw_obs.get("hint", "") or "").lower()
    stage       = str(raw_obs.get("scenario_stage", "approaching") or "approaching").lower()
    hazard_dist = float(raw_obs.get("hazard_distance", 999.0) or 999.0)
    hazard_type = str(raw_obs.get("hazard_type", "") or "")
    objects     = sensor.get("objects", []) or []
    signal      = str(sensor.get("traffic_signal", environment.get("traffic_signal", "none"))).lower()
    speed       = float(ego.get("speed", 0.0))
    history     = history or []

    min_dist = min((float(o.get("distance", 999.0)) for o in objects), default=999.0)
    on_road_hazard = any(
        o.get("on_road", True) and float(o.get("distance", 999.0)) < 9.0
        for o in objects
    )
    crossing = any(
        bool(o.get("crossing")) and float(o.get("distance", 999.0)) < 10.0
        for o in objects
    )
    aggressive = any(
        str(o.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(o.get("distance", 999.0)) < 9.0
        for o in objects
    )

    # Detect last 3 actions to avoid oscillation
    recent = [h.get("action") for h in (history[-3:] if len(history) >= 3 else history)]
    stuck_in_same = len(set(recent)) == 1 and len(recent) == 3

    # ── 1. Sudden alert handling ─────────────────────────────────────────────
    for alert in alerts:
        alert_l = alert.lower()
        if "ambulance" in alert_l:
            return {"action": "steer_left", "value": 0.6}   # create corridor
        if "animal" in alert_l or "cow" in alert_l or "dog" in alert_l:
            return {"action": "brake", "value": 0.9}
        if "police" in alert_l or "flagman" in alert_l:
            return {"action": "wait", "value": 0.0}
        if "traffic jam" in alert_l:
            return {"action": "brake", "value": 0.7}
        if "pothole" in alert_l or "speed breaker" in alert_l:
            return {"action": "brake", "value": 0.6}
        if "child" in alert_l or "pedestrian" in alert_l:
            return {"action": "brake", "value": 1.0}
        if "bike" in alert_l and "zig" in alert_l:
            return {"action": "brake", "value": 0.5}

    # ── 2. Hint override from environment (highest priority) ─────────────────
    if "accelerate" in hint:
        return {"action": "accelerate", "value": 0.5}

    # ── 3. Stage logic ────────────────────────────────────────────────────────
    if stage in ("clearing", "cleared"):
        # Hazard has passed — must accelerate
        if stuck_in_same and recent[0] in ("brake", "wait"):
            return {"action": "accelerate", "value": 0.5}
        return {"action": "accelerate", "value": 0.4}

    # ── 4. Red light ──────────────────────────────────────────────────────────
    if signal == "red" and speed > 1.0:
        return {"action": "brake", "value": 0.8}

    # ── 5. Critical distance ──────────────────────────────────────────────────
    if min_dist < 3.5:
        return {"action": "brake", "value": 1.0}
    if min_dist < 6.0 or crossing or aggressive:
        if stuck_in_same and recent[0] == "brake":
            return {"action": "wait", "value": 0.0}
        return {"action": "brake", "value": 0.8}
    if min_dist < 10.0 or on_road_hazard:
        if stuck_in_same and recent[0] in ("brake", "wait"):
            # Try slight steer to get unstuck
            return {"action": "steer_left", "value": 0.3}
        return {"action": "brake", "value": 0.5}

    # ── 6. Ambulance-type in hazard_type even without an active alert ─────────
    if "ambulance" in hazard_type:
        return {"action": "steer_left", "value": 0.5}

    # ── 7. Lane discipline ────────────────────────────────────────────────────
    lane_info = str(sensor.get("lane_info", environment.get("lane_status", "clear"))).lower()
    if "missing" in lane_info or "faded" in lane_info:
        return {"action": "accelerate", "value": 0.2 if speed < 8.0 else 0.1}

    # ── 8. Normal forward motion ──────────────────────────────────────────────
    if speed < 5.0 and hazard_dist > 14.0:
        return {"action": "accelerate", "value": 0.5}
    if speed >= 5.0 and hazard_dist > 14.0:
        return {"action": "accelerate", "value": 0.3}

    return {"action": "wait", "value": 0.0}


@dataclass
class ModularBaselineAgent:
    name: str = "modular_baseline"
    _history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self._history = []

    def act(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        action = choose_action(raw_obs, self._history)
        self._history.append(action)
        if len(self._history) > 10:
            self._history = self._history[-10:]
        return action


@dataclass
class ConservativeAgent(ModularBaselineAgent):
    name: str = "conservative_agent"

    def act(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        action = choose_action(raw_obs, self._history)
        # Conservative: never choose accelerate unless the hint says so
        hint = str(raw_obs.get("hint", "") or "").lower()
        if action["action"] == "accelerate" and "accelerate" not in hint:
            action = {"action": "wait", "value": 0.0}
        self._history.append(action)
        if len(self._history) > 10:
            self._history = self._history[-10:]
        return action


def available_agents() -> List[ModularBaselineAgent]:
    return [ModularBaselineAgent(), ConservativeAgent()]

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
