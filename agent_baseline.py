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
    road_geo    = raw_obs.get("road_geometry", {}) or {}
    environment = raw_obs.get("environment", {}) or {}
    alerts      = raw_obs.get("active_alerts", []) or []
    hint        = str(raw_obs.get("hint", "") or "").lower()
    stage       = str(raw_obs.get("scenario_stage", "approaching") or "approaching").lower()
    hazard_dist = float(raw_obs.get("hazard_distance", 999.0) or 999.0)
    hazard_type = str(raw_obs.get("hazard_type", "") or "").lower()
    objects     = sensor.get("objects", []) or []
    signal      = str(sensor.get("traffic_signal", environment.get("traffic_signal", "none"))).lower()
    speed       = float(ego.get("speed", 0.0))
    lane        = str(ego.get("lane", "center")).lower()
    lane_offset = float(ego.get("lane_offset_m", ego.get("lane_position", 0.0)))
    road_cond   = str(environment.get("road_condition", "normal")).lower()
    history     = history or []

    # Road geometry — how much space is available left/right
    can_go_left  = bool(road_geo.get("can_steer_left",        lane_offset < 2.0))
    can_go_right = bool(road_geo.get("can_steer_right",       lane_offset > -2.0))
    can_cl_left  = bool(road_geo.get("can_change_lane_left",  False))
    can_cl_right = bool(road_geo.get("can_change_lane_right", False))
    space_left   = float(road_geo.get("space_left_m",  2.5 + lane_offset))
    space_right  = float(road_geo.get("space_right_m", 2.5 - lane_offset))

    min_dist = min((float(o.get("distance", 999.0)) for o in objects), default=999.0)
    min_ttc  = min((float(o.get("ttc_s", 99.9)) for o in objects if float(o.get("distance", 999)) < 20), default=99.9)

    # ── Lateral-aware dodge direction ─────────────────────────────────────────
    # rel_lat_m: + = object is to the RIGHT of ego, - = object is to the LEFT
    # Rule: if object is RIGHT → steer LEFT (and vice versa)
    def _dodge_dir(obj_list: List[Dict[str, Any]]) -> str | None:
        for o in sorted(obj_list, key=lambda x: float(x.get("distance", 999))):
            lat = float(o.get("rel_lat_m", o.get("lateral_offset_m", 0.0)))
            if lat >= 0 and can_go_left:   return "steer_left"
            if lat < 0  and can_go_right:  return "steer_right"
            if space_left >= space_right and can_go_left:  return "steer_left"
            if can_go_right:                               return "steer_right"
        return None

    # ── Sensor object analysis ─────────────────────────────────────────────
    on_road_hazard = any(
        o.get("on_road", True) and float(o.get("distance", 999.0)) < 9.0
        for o in objects
    )
    crossing = any(
        bool(o.get("crossing")) and float(o.get("distance", 999.0)) < 10.0
        for o in objects
    )
    # Right-side threats — object is to the RIGHT (rel_lat_m >= 0)
    right_threat = any(
        str(o.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(o.get("rel_lat_m", o.get("lateral_offset_m", 0.0))) >= 0
        and float(o.get("distance", 999.0)) < 9.0
        for o in objects
    )
    # Left-side threats — object is to the LEFT (rel_lat_m < 0)
    left_threat = any(
        str(o.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge"}
        and float(o.get("rel_lat_m", o.get("lateral_offset_m", 0.0))) < 0
        and float(o.get("distance", 999.0)) < 9.0
        for o in objects
    )
    # Lane blocked ahead
    lane_blocked_right = any(
        str(o.get("blocking_lane", "")).lower() == "right" and float(o.get("distance", 999.0)) < 15.0
        for o in objects
    )
    lane_blocked_left = any(
        str(o.get("blocking_lane", "")).lower() == "left" and float(o.get("distance", 999.0)) < 15.0
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

    # ── 0. Emergency safety — absolute override ──────────────────────────────
    if min_dist < 2.5:
        return {"action": "brake", "value": 1.0}

    # ── 1. Stage clearing/cleared — accelerate out before alerts/hints ────────
    # Prevents pedestrian-alert or steer-hint from trapping the agent after hazard clears.
    # Ambulance exception only if it is STILL close; a distant ambulance has already passed.
    _has_ambulance = any("ambulance" in a.lower() for a in alerts) and min_dist < 12.0
    if stage in ("clearing", "cleared") and min_dist > 4.5 and not _has_ambulance:
        return {"action": "accelerate", "value": 0.5 if stuck_in_same else 0.4}

    # ── 2. Sudden alert handling (only fires while hazard is still active) ────
    for alert in alerts:
        alert_l = alert.lower()
        if "ambulance" in alert_l:
            # If stage is clearing and ambulance already passed (> 12m), ignore this stale alert
            if stage in ("clearing", "cleared") and min_dist > 12.0:
                continue
            return {"action": "steer_left", "value": 0.7}
        if "animal" in alert_l or "cow" in alert_l or "dog" in alert_l:
            if hazard_dist > 10.0:
                return {"action": "horn", "value": 0.8}
            return {"action": "brake", "value": 0.9}
        if "police" in alert_l or "flagman" in alert_l:
            return {"action": "wait", "value": 0.0}
        if "traffic jam" in alert_l:
            return {"action": "brake", "value": 0.6}
        if "pothole" in alert_l or "speed breaker" in alert_l:
            if hazard_dist < 3.5:  # too close to dodge — brake hard
                return {"action": "brake", "value": 1.0}
            if hazard_dist < 8.0:
                dodge = _dodge_dir(objects) or ("steer_left" if can_go_left else "steer_right")
                return {"action": dodge, "value": 0.5}
            return {"action": "brake", "value": 0.5}
        if "child" in alert_l or "pedestrian" in alert_l:
            if hazard_dist > 10.0:
                return {"action": "horn", "value": 0.6}
            return {"action": "brake", "value": 1.0}
        if "wedding" in alert_l or "procession" in alert_l:
            if hazard_dist > 12.0:
                return {"action": "horn", "value": 0.4}
            return {"action": "brake", "value": 0.7}
        if "fog" in alert_l or "rain" in alert_l or "waterlog" in alert_l:
            if not stuck_in_same:
                return {"action": "brake", "value": 0.4}
            return {"action": "steer_left", "value": 0.2}
        if "construction" in alert_l:
            return {"action": "wait", "value": 0.0}
        if "bike" in alert_l and "zig" in alert_l:
            dodge = _dodge_dir(objects) or "steer_right"
            return {"action": dodge, "value": 0.4}

    # ── 3. Hint override (ONLY while approaching — not after stage clears) ────
    if stage not in ("clearing", "cleared"):
        for kw, act, val in [
            ("steer_left",          "steer_left",         0.5),
            ("steer_right",         "steer_right",        0.5),
            ("change_lane_left",    "change_lane_left",   0.7),
            ("change_lane_right",   "change_lane_right",  0.7),
            ("horn",                "horn",               0.6),
            ("accelerate",          "accelerate",         0.5),
        ]:
            if kw in hint:
                return {"action": act, "value": val}

    # ── 3b. Adversarial probe — if stuck waiting with distant hazard, try forward
    # Handles scenarios with no clearing event (e.g. adversarial vehicle that never
    # moves away). After 3 consecutive wait/brake steps, nudge forward to test gap.
    if stuck_in_same and recent[0] in ("wait", "brake") and hazard_dist > 9.0 and min_dist > 9.0:
        return {"action": "accelerate", "value": 0.2}

    # ── 4. Red / policeman signal ────────────────────────────────────────────
    if signal in ("red", "stop"):
        return {"action": "brake", "value": 0.8}
    if signal == "policeman_override":
        return {"action": "wait", "value": 0.0}

    # ── 5. Hazard-type specific actions ───────────────────────────────────────
    if "ambulance" in hazard_type:
        return {"action": "steer_left", "value": 0.6}
    if "pothole" in hazard_type or "speed_breaker" in hazard_type or "ridge" in hazard_type:
        if hazard_dist < 10.0:
            pothole_objs = [o for o in objects if o.get("type") in {"pothole", "ridge", "speed_breaker"}]
            dodge = _dodge_dir(pothole_objs or objects) or ("steer_left" if can_go_left else "steer_right")
            return {"action": dodge, "value": 0.5}
        return {"action": "brake", "value": 0.5}
    if "animal" in hazard_type or "cow" in hazard_type:
        if hazard_dist > 10.0:
            return {"action": "horn", "value": 0.8}
        return {"action": "brake", "value": 0.9}
    if "pedestrian" in hazard_type or "child" in hazard_type:
        if hazard_dist > 12.0:
            return {"action": "horn", "value": 0.5}
        return {"action": "brake", "value": 1.0}
    if "wedding" in hazard_type or "procession" in hazard_type:
        if hazard_dist > 15.0:
            return {"action": "horn", "value": 0.4}
        return {"action": "wait", "value": 0.0}
    if "traffic_jam" in hazard_type or "jam" in hazard_type:
        return {"action": "brake", "value": 0.6}
    if "adversarial" in hazard_type or "cut_in" in hazard_type or "auto_cut_in" in hazard_type:
        # Adversarial vehicle: dodge if it's to one side, otherwise brake
        if hazard_dist < 7.0:
            dodge = _dodge_dir(objects) or ("steer_right" if can_go_right else "steer_left")
            return {"action": dodge, "value": 0.5}
        return {"action": "brake", "value": 0.5}

    # ── 6. Slippery / low-visibility road ────────────────────────────────────
    if "wet" in road_cond or "slippery" in road_cond or "fog" in road_cond or "rain" in road_cond:
        if hazard_dist < 14.0:
            return {"action": "brake", "value": 0.4}
        return {"action": "accelerate", "value": 0.2}   # slow cautious pace

    # ── 7. Lane-change to dodge blocked lane ──────────────────────────────────
    if lane_blocked_right and can_cl_left:
        return {"action": "change_lane_left", "value": 0.8}
    if lane_blocked_left and can_cl_right:
        return {"action": "change_lane_right", "value": 0.8}

    # ── 8. Imminent TTC threat — dodge using lateral position ─────────────────
    if min_ttc < 2.5 and min_dist < 6.0:
        dodge = _dodge_dir(objects)
        if dodge:
            return {"action": dodge, "value": 0.7}

    # ── 9. Critical distance / aggressive object ──────────────────────────────
    if min_dist < 3.5:
        return {"action": "brake", "value": 1.0}
    if aggressive or right_threat:
        if hazard_dist < 7.0:
            dodge = _dodge_dir(objects) or "steer_left"
            return {"action": dodge, "value": 0.5}
        return {"action": "brake", "value": 0.7}
    if left_threat and hazard_dist < 7.0:
        dodge = _dodge_dir(objects) or "steer_right"
        return {"action": dodge, "value": 0.5}
    if min_dist < 6.0 or crossing:
        if stuck_in_same and recent[0] == "brake":
            return {"action": "wait", "value": 0.0}
        return {"action": "brake", "value": 0.8}
    if min_dist < 10.0 or on_road_hazard:
        if stuck_in_same and recent[0] in ("brake", "wait"):
            dodge = _dodge_dir(objects)
            if dodge:
                return {"action": dodge, "value": 0.3}
        return {"action": "brake", "value": 0.5}

    # ── 10. Lane discipline ────────────────────────────────────────────────────
    lane_info = str(sensor.get("lane_info", environment.get("lane_status", "clear"))).lower()
    if "missing" in lane_info or "faded" in lane_info:
        return {"action": "accelerate", "value": 0.2 if speed < 8.0 else 0.1}

    # ── 11. Normal forward motion ─────────────────────────────────────────────
    # Object >14m: safe to accelerate
    if hazard_dist > 14.0 and min_dist > 14.0:
        return {"action": "accelerate", "value": 0.5 if speed < 5.0 else 0.3}
    # Object in 10-14m range: unrecognised type or clearing — gentle brake rather than stall
    return {"action": "brake", "value": 0.3}


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