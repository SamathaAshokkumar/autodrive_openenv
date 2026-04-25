"""Driving backend for AutoDrive Gym.

Equivalent to kube's backend: reset base state, inject scenario, execute one
action, advance the world, and compute programmatic safety checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import atan2, hypot
from typing import Any, Dict, List

from .constants import DEFAULT_SCENE_ENV, DEFAULT_VEHICLE_PROFILE
from .driving_actions import DrivingActionHandler
from .scenario_injectors import ScenarioInjector


@dataclass
class LightweightDrivingSimulator:
    ego: Dict[str, Any] = field(default_factory=dict)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    vehicle_profile: Dict[str, Any] = field(default_factory=dict)
    scenario: Dict[str, Any] = field(default_factory=dict)
    event_log: str = ""
    decision_log: str = ""
    last_action: Dict[str, Any] = field(default_factory=lambda: {"action": "wait", "value": 0.0})
    steps_without_progress: int = 0
    goal_x: float = 120.0
    tick: int = 0
    triggered_events: List[str] = field(default_factory=list)
    current_stage: str = "approaching"
    primary_hazard_type: str = "unknown"
    active_secondary_hazard: str = ""
    active_secondary_stage: str = ""
    # ── Progress / stuck tracking ─────────────────────────────────────────
    consecutive_same_action: int = 0
    _prev_near_hazard: bool = False
    hazard_cleared_this_step: bool = False
    # ── Distance trend (for moving-away detection) ────────────────────────
    _prev_hazard_dist: float = 999.0
    hazard_moving_away: bool = False

    def reset(self):
        self.ego = {
            "x": 0.0,
            "y": 0.0,
            "speed": 0.0,
            "heading": "straight",
            "steering": 0.0,
            "lane": "center",
            "lane_position": 0.0,
            "horn": 0,
            "max_speed": DEFAULT_VEHICLE_PROFILE["max_speed"],
        }
        self.objects = []
        self.environment = dict(DEFAULT_SCENE_ENV)
        self.vehicle_profile = dict(DEFAULT_VEHICLE_PROFILE)
        self.event_log = ""
        self.decision_log = ""
        self.last_action = {"action": "wait", "value": 0.0}
        self.steps_without_progress = 0
        self.tick = 0
        self.triggered_events = []
        self.current_stage = "approaching"
        self.primary_hazard_type = "unknown"
        self.active_secondary_hazard = ""
        self.active_secondary_stage = ""
        self.consecutive_same_action = 0
        self._prev_near_hazard = False
        self.hazard_cleared_this_step = False
        self._prev_hazard_dist = 999.0
        self.hazard_moving_away = False

    def update(self):
        # Alerts and cleared-flag are per-tick only — clear at start of each step
        self.event_log = ""
        self.hazard_cleared_this_step = False
        self.tick += 1
        previous_x = self.ego["x"]
        self.ego["x"] += self.ego["speed"] * 0.2
        if abs(self.ego["x"] - previous_x) < 0.2:
            self.steps_without_progress += 1
        else:
            self.steps_without_progress = 0

        for obj in self.objects:
            obj["x"] += obj.get("vx", 0.0)
            obj["y"] += obj.get("vy", 0.0)
            behavior = obj.get("behavior", "")
            if behavior == "sudden_cross":
                obj["y"] -= 0.6
            elif behavior == "cut_in":
                obj["y"] += 0.5
                obj["x"] -= 0.2
            elif behavior == "blind_spot_merge":
                obj["y"] -= 0.4
                obj["x"] += 0.1
            elif behavior == "emergency_pass":
                obj["x"] += 1.6
                obj["y"] += -0.25 if obj["y"] > 0 else 0.25
            elif behavior == "zig_zag":
                obj["y"] += 0.5 if int(obj["x"]) % 2 == 0 else -0.5

        self._apply_dynamic_events()
        self.objects = [obj for obj in self.objects if obj["x"] > self.ego["x"] - 12 and abs(obj["y"]) < 6]

    def _apply_dynamic_events(self):
        for event in self.scenario.get("dynamic_events", []):
            event_id = f"{event.get('trigger_step')}:{event.get('kind')}:{event.get('message', '')}"
            if event_id in self.triggered_events:
                continue
            if int(event.get("trigger_step", 9999)) != self.tick:
                continue
            kind = event.get("kind", "")
            message = event.get("message", "")
            if kind == "spawn_vehicle" and event.get("actor"):
                actor = dict(event["actor"])
                self.objects.append(
                    {
                        "type": actor.get("type", "vehicle"),
                        "x": float(actor.get("x", 10.0)),
                        "y": float(actor.get("y", 0.0)),
                        "vx": float(actor.get("vx", 0.0)),
                        "vy": float(actor.get("vy", 0.0)),
                        "behavior": actor.get("behavior", "static"),
                        "lane": actor.get("lane", "center"),
                    }
                )
                if event.get("hazard_type"):
                    self.active_secondary_hazard = str(event.get("hazard_type", ""))
                    self.active_secondary_stage = "active"
            elif kind == "clear_crossing":
                for obj in self.objects:
                    if obj.get("type") == "pedestrian":
                        obj["y"] = -4.0
                        obj["x"] = self.ego["x"] + 10.0
            elif kind == "move_actor_ahead":
                for obj in self.objects:
                    if obj.get("type") in {"auto", "bike"}:
                        obj["x"] = max(obj["x"], self.ego["x"] + 10.0)
                        obj["y"] = 0.0
            elif kind == "clear_static_obstacle":
                self.objects = [obj for obj in self.objects if obj.get("behavior") not in {"ridge", "static"}]
            elif kind == "change_signal":
                self.environment["traffic_signal"] = event.get("traffic_signal", self.environment.get("traffic_signal", "none"))

            if message:
                self.event_log = message
                lowered = message.lower()
                if "clear" in lowered or "crossed past" in lowered or "pulling away" in lowered or "waves your lane forward" in lowered:
                    self.current_stage = "clearing"
                    self.hazard_cleared_this_step = True
                    if self.active_secondary_hazard:
                        self.active_secondary_stage = "clearing"
            self.triggered_events.append(event_id)

    def _secondary_hazard_status(self, hazard_type: str, objects: List[Dict[str, Any]]) -> str:
        if not hazard_type:
            return ""
        hazard_objects = []
        if hazard_type == "ambulance_approach":
            hazard_objects = [obj for obj in objects if obj.get("type") == "ambulance"]
        elif hazard_type == "animal_crossing":
            hazard_objects = [obj for obj in objects if obj.get("type") in {"cow", "dog", "animal"}]
        elif hazard_type == "police_override":
            hazard_objects = [obj for obj in objects if obj.get("type") == "traffic_police"]
        elif hazard_type == "traffic_jam":
            hazard_objects = [obj for obj in objects if obj.get("type") in {"car", "truck", "bus", "auto"}]
        elif hazard_type == "speed_breaker":
            hazard_objects = [obj for obj in objects if obj.get("type") == "speed_breaker"]
        elif hazard_type == "pothole_ahead":
            hazard_objects = [obj for obj in objects if obj.get("type") == "pothole"]
        elif hazard_type == "crowded_market":
            hazard_objects = [obj for obj in objects if obj.get("type") in {"pedestrian", "auto", "traffic_police"}]
        else:
            hazard_objects = objects

        if not hazard_objects:
            return "cleared"

        nearest = min(float(obj.get("distance", 999.0)) for obj in hazard_objects)
        if hazard_type in {"ambulance_approach", "police_override"}:
            return "active" if nearest < 12.0 else "clearing"
        if hazard_type in {"traffic_jam", "crowded_market"}:
            return "active" if nearest < 9.0 else "clearing"
        if hazard_type in {"animal_crossing", "speed_breaker", "pothole_ahead"}:
            return "active" if nearest < 8.0 else "clearing"
        return "active" if nearest < 8.0 else "clearing"

    def hazard_summary(self) -> Dict[str, Any]:
        snapshot = self.sensor_snapshot()
        objects = snapshot["objects"]
        hazard_distance = min((obj["distance"] for obj in objects), default=999.0)

        # ── Distance trend: detect when hazard is moving AWAY ─────────────────
        # This is critical for real-world behaviour: if the hazard is receding,
        # the agent should ease off the brakes — not keep braking forever.
        self.hazard_moving_away = (
            hazard_distance > self._prev_hazard_dist + 0.3
            and self._prev_hazard_dist < 999.0
        )
        self._prev_hazard_dist = hazard_distance

        # ── Auto stage-transition based on physical distance ──────────────────
        # Only applies to the PRIMARY hazard (secondary has its own logic).
        # Prevents the stage being stuck at "approaching" when hazard has moved
        # far away naturally (e.g. bike merges past, auto drives ahead).
        if not self.active_secondary_hazard:
            # On-road objects still physically in the agent's path
            on_road_ahead = [
                o for o in objects
                if o.get("on_road") and float(o.get("rel_fwd_m", o["distance"])) > 0
            ]
            nearest_on_road = min((o["distance"] for o in on_road_ahead), default=999.0)

            if self.current_stage == "approaching" and nearest_on_road > 17.0:
                # Hazard has moved to a safe distance — transition to clearing
                self.current_stage = "clearing"
                self.hazard_cleared_this_step = True
            elif self.current_stage == "clearing" and nearest_on_road > 26.0:
                # Hazard is now far enough that the road is genuinely open
                self.current_stage = "cleared"
                self.hazard_cleared_this_step = True

        if self.active_secondary_hazard:
            secondary_status = self._secondary_hazard_status(self.active_secondary_hazard, objects)
            if secondary_status == "cleared":
                self.active_secondary_hazard = ""
                self.active_secondary_stage = ""
            else:
                hazard_type = self.active_secondary_hazard
                self.active_secondary_stage = secondary_status
                hazard_distance = min(
                    (
                        float(obj["distance"])
                        for obj in objects
                        if self._object_matches_hazard(obj, self.active_secondary_hazard)
                    ),
                    default=hazard_distance,
                )
                status = secondary_status
                self.current_stage = status
                return {
                    "hazard_type": hazard_type,
                    "hazard_distance": round(float(hazard_distance), 2),
                    "hazard_status": status,
                    "scenario_stage": self.current_stage,
                }
        # Surface the primary hazard objects so the agent always sees real distance
        return {
            "hazard_type": self.primary_hazard_type,
            "hazard_distance": round(float(hazard_distance), 2),
            "hazard_status": self.current_stage,
            "scenario_stage": self.current_stage,
        }

    def _object_matches_hazard(self, obj: Dict[str, Any], hazard_type: str) -> bool:
        object_type = str(obj.get("type", ""))
        if hazard_type == "ambulance_approach":
            return object_type == "ambulance"
        if hazard_type == "animal_crossing":
            return object_type in {"cow", "dog", "animal"}
        if hazard_type == "police_override":
            return object_type == "traffic_police"
        if hazard_type == "traffic_jam":
            return object_type in {"car", "truck", "bus", "auto"}
        if hazard_type == "speed_breaker":
            return object_type == "speed_breaker"
        if hazard_type == "pothole_ahead":
            return object_type == "pothole"
        if hazard_type == "crowded_market":
            return object_type in {"pedestrian", "auto", "traffic_police"}
        return True

    def sensor_snapshot(self) -> Dict[str, Any]:
        """Return rich per-object sensor state: position, velocity, TTC, lateral placement,
        closing speed, trajectory, dominance, intent confidence, and noisy perception."""
        ego_x    = self.ego["x"]
        ego_lat  = self.ego["lane_position"]
        ego_spd  = self.ego["speed"]

        # ── Social priority / dominance table ─────────────────────────────
        _DOMINANCE: Dict[str, str] = {
            "truck": "high", "bus": "high", "ambulance": "high",
            "traffic_police": "high",
            "auto": "medium", "car": "medium",
            "bike": "low", "pedestrian": "low",
            "cow": "low", "dog": "low",
            "speed_breaker": "none", "pothole": "none",
        }
        # Negotiation is required when a moving actor with medium/high dominance
        # is in or adjacent to the ego lane at medium range
        def _needs_negotiation(obj_type: str, in_lane: bool, dist: float) -> bool:
            dom = _DOMINANCE.get(obj_type, "low")
            return dom in ("high", "medium") and dist < 18.0 and in_lane

        detected_objects = []
        for obj in self.objects:
            # ── World-space deltas ─────────────────────────────────────────
            fwd_dx  = obj["x"] - ego_x
            lat_dy  = obj["y"] - ego_lat

            true_dist = hypot(fwd_dx, lat_dy)

            # ── Noisy perception: add gaussian jitter on distance for far objects ──
            # Real sensors have measurement noise that grows with distance.
            # This forces the agent to reason probabilistically, not just react to exact values.
            perception_noise = 0.0
            if true_dist > 12.0:
                noise_sigma = 0.04 * true_dist  # 4% of distance — realistic lidar noise
                perception_noise = random.gauss(0.0, noise_sigma)
            dist = round(max(0.5, true_dist + perception_noise), 2)

            angle_d = round(atan2(lat_dy, fwd_dx) * 57.2958, 1)

            # ── Object velocity ────────────────────────────────────────────
            vx = obj.get("vx", 0.0)
            vy = obj.get("vy", 0.0)
            obj_speed = round(hypot(vx, vy), 2)

            # ── Closing speed (positive = approaching ego, negative = receding) ──
            # This is the key signal: a bike 5m away but closing_speed<0 is safe.
            # A car 15m away with high closing_speed is dangerous.
            closing_speed = round(ego_spd - vx, 2)

            # ── Trajectory classification ──────────────────────────────────
            # Tells the agent what the object is likely to do next.
            if obj.get("behavior") in {"sudden_cross", "fake_cross"}:
                trajectory = "will_cross"
            elif obj.get("behavior") == "cut_in":
                trajectory = "merging_into_lane"
            elif obj.get("behavior") == "emergency_pass":
                trajectory = "overtaking_fast"
            elif obj.get("behavior") == "blind_spot_merge":
                trajectory = "merging_from_blind_spot"
            elif obj.get("behavior") == "zig_zag":
                trajectory = "erratic_unpredictable"
            elif obj.get("behavior") == "static":
                trajectory = "stationary"
            elif vx < -0.1:
                trajectory = "moving_toward_ego"
            elif vx > 0.3 and closing_speed < -0.5:
                trajectory = "moving_away"
            else:
                trajectory = "unknown"

            # ── Intent uncertainty: confidence based on how predictable the behavior is ──
            # Static/known behaviors → high confidence.
            # Erratic/unknown behaviors → low confidence (agent should be extra cautious).
            _intent_conf_map = {
                "static": 0.95, "ridge": 0.95,
                "emergency_pass": 0.90,
                "sudden_cross": 0.45,      # could stop, could continue
                "cut_in": 0.55,            # partial commitment
                "blind_spot_merge": 0.50,  # hard to predict
                "zig_zag": 0.25,           # very uncertain
                "fake_cross": 0.20,        # deliberately deceptive
            }
            intent_confidence = round(
                _intent_conf_map.get(obj.get("behavior", ""), 0.60)
                + random.gauss(0.0, 0.05),  # add small noise
                2
            )
            intent_confidence = max(0.1, min(0.99, intent_confidence))

            # ── TTC ────────────────────────────────────────────────────────
            if fwd_dx > 0 and closing_speed > 0.3:
                ttc = round(min(fwd_dx / closing_speed, 99.9), 1)
            else:
                ttc = 99.9

            heading_deg = round(atan2(vy, vx) * 57.2958, 1) if obj_speed > 0.05 else 0.0

            if lat_dy > 0.4:
                side = "right"
            elif lat_dy < -0.4:
                side = "left"
            else:
                side = "center"

            in_ego_lane = abs(lat_dy) < 1.8
            obj_type = obj["type"]
            dominance = _DOMINANCE.get(obj_type, "low")
            negotiation_required = _needs_negotiation(obj_type, in_ego_lane, dist)

            detected_objects.append({
                # ── Identification ─────────────────────────
                "type":                 obj_type,
                "behavior":             obj.get("behavior", "static"),
                # ── Range & bearing ────────────────────────
                "distance":             dist,
                "angle_deg":            angle_d,
                # ── Absolute world position ─────────────────
                "abs_pos":              [round(obj["x"], 2), round(obj["y"], 2)],
                # ── Relative position ───────────────────────
                "rel_fwd_m":            round(fwd_dx, 2),
                "rel_lat_m":            round(lat_dy, 2),
                # ── Motion ──────────────────────────────────
                "velocity":             [round(vx, 2), round(vy, 2)],
                "speed_mps":            obj_speed,
                "heading_deg":          heading_deg,
                "is_stationary":        obj_speed < 0.05,
                # ── Relative motion to ego ───────────────────
                "closing_speed":        closing_speed,   # + = approaching, - = receding
                "trajectory":           trajectory,      # what is the object doing?
                # ── Intent uncertainty ───────────────────────
                "intent_confidence":    intent_confidence,   # 0=very uncertain, 1=certain
                # ── Collision risk ───────────────────────────
                "ttc_s":                ttc,
                # ── Social priority ──────────────────────────
                "dominance":            dominance,           # high/medium/low/none
                "negotiation_required": negotiation_required,
                # ── Road placement ───────────────────────────
                "side":                 side,
                "in_ego_lane":          in_ego_lane,
                "on_road":              abs(obj["y"]) < 2.2,
                "lane":                 obj.get("lane", "unknown"),
                "crossing":             obj.get("behavior") in {"sudden_cross", "fake_cross"},
            })

        return {
            "objects":        sorted(detected_objects, key=lambda o: o["distance"])[:8],
            "lane_info":      self.environment.get("lane_status", "clear"),
            "traffic_signal": self.environment.get("traffic_signal", "none"),
            "camera_view":    "front_rgb_summary",
        }

    def check_collision(self) -> bool:
        return any(hypot(obj["x"] - self.ego["x"], obj["y"] - self.ego["y"]) < 1.2 for obj in self.objects)

    def check_near_miss(self) -> bool:
        return any(hypot(obj["x"] - self.ego["x"], obj["y"] - self.ego["y"]) < 2.5 for obj in self.objects)

    def check_goal(self) -> bool:
        return self.ego["x"] >= self.goal_x

    def check_offroad(self) -> bool:
        return abs(self.ego["lane_position"]) > 2.5

    def check_overspeed(self) -> bool:
        return self.ego["speed"] > self.ego["max_speed"]

    def check_stuck(self) -> bool:
        # Braking near a hazard is intentional — do not flag it as stuck
        nearby_hazard = any(
            hypot(obj["x"] - self.ego["x"], obj["y"] - self.ego["y"]) < 15.0
            for obj in self.objects
        )
        # When hazard just cleared, give the agent a fresh grace window
        if self._prev_near_hazard and not nearby_hazard:
            self.steps_without_progress = 0
            self.consecutive_same_action = 0
        self._prev_near_hazard = nearby_hazard
        if nearby_hazard:
            return False
        # 5 consecutive identical actions with no forward movement = stuck
        if self.consecutive_same_action >= 5 and self.steps_without_progress >= 3:
            return True
        # General no-progress threshold (user-requested: 2 extra steps, was 6)
        return self.steps_without_progress >= 8


class DrivingBackend:
    def __init__(self):
        self.simulator = LightweightDrivingSimulator()
        self.actions = DrivingActionHandler()
        self.injector = ScenarioInjector()
        self.simulator.reset()

    def reset(self):
        self.simulator.reset()

    def inject_scenario(self, scenario: Dict[str, Any]) -> str:
        result = self.injector.inject(self.simulator, scenario)
        self.simulator.primary_hazard_type = str(scenario.get("type", "unknown"))
        self.simulator.active_secondary_hazard = ""
        self.simulator.active_secondary_stage = ""
        return result

    def execute(self, action: str, value: float) -> str:
        # Track consecutive same actions for stuck detection
        if action == self.simulator.last_action.get("action"):
            self.simulator.consecutive_same_action += 1
        else:
            self.simulator.consecutive_same_action = 1
        return self.actions.apply(self.simulator, action, value)

    def update(self):
        self.simulator.update()

    def build_observation(self, steps_taken: int, max_steps: int, hint: str = "", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        sensor_data  = self.simulator.sensor_snapshot()
        hazard       = self.simulator.hazard_summary()
        ego_lat      = self.simulator.ego["lane_position"]   # + = right, - = left

        # ── Road geometry ────────────────────────────────────────────────────
        road_w   = float(self.simulator.environment.get("road_width_m",    7.2))
        lane_w   = float(self.simulator.environment.get("lane_width_m",    3.6))
        num_l    = int(self.simulator.environment.get("num_lanes",         2))
        shoulder = float(self.simulator.environment.get("shoulder_width_m", 1.1))
        # Drivable surface (without shoulder): ±2.5m from centre
        half_drivable = 2.5
        space_left_m  = round(half_drivable + ego_lat, 2)    # how much space is to the LEFT
        space_right_m = round(half_drivable - ego_lat, 2)    # how much space is to the RIGHT

        road_geometry = {
            "road_width_m":         road_w,
            "lane_width_m":         lane_w,
            "num_lanes":            num_l,
            "shoulder_width_m":     shoulder,
            "surface_type":         self.simulator.environment.get("surface_type", "asphalt"),
            # Ego position on the road
            "ego_lat_offset_m":     round(ego_lat, 2),   # + = right of centre, - = left
            "space_left_m":         max(0.0, space_left_m),
            "space_right_m":        max(0.0, space_right_m),
            # Action feasibility flags
            "can_steer_left":       space_left_m  > 0.6,
            "can_steer_right":      space_right_m > 0.6,
            "can_change_lane_left": space_left_m  > lane_w * 0.7,
            "can_change_lane_right":space_right_m > lane_w * 0.7,
        }

        # ── Ego state (richer than before) ───────────────────────────────────
        ego_state = {
            "speed":         round(self.simulator.ego["speed"], 2),
            "steering_deg":  round(self.simulator.ego["steering"], 2),
            "position":      [round(self.simulator.ego["x"], 2), round(self.simulator.ego["y"], 2)],
            "lane":          self.simulator.ego["lane"],
            "lane_offset_m": round(ego_lat, 2),   # + = right of lane centre
            "heading":       self.simulator.ego["heading"],
            "horn_active":   bool(self.simulator.ego.get("horn", 0)),
        }

        environment = {
            "road_condition":  self.simulator.environment.get("road_condition", "normal"),
            "visibility":      self.simulator.environment.get("visibility", "clear"),
            "lane_status":     self.simulator.environment.get("lane_status", "clear"),
            "traffic_signal":  self.simulator.environment.get("traffic_signal", "none"),
        }

        # ── Signal trust score (India-specific: signals aren't absolute truth) ──
        # Green light at a blocked intersection? Red light + police waving? Trust falls.
        _sig = environment["traffic_signal"]
        _road = environment["road_condition"]
        if _sig == "policeman_override":
            signal_trust = 0.30   # low — human is overriding, trust human more than signal
        elif _sig in ("red", "green") and _road == "construction":
            signal_trust = 0.55   # moderate — signal may not reflect actual flow
        elif _sig == "none":
            signal_trust = 1.0    # no signal — trust own judgment fully
        else:
            signal_trust = 0.85   # normal signal, mostly reliable
        # Add slight noise — real signals can flicker or be obscured
        signal_trust = round(max(0.1, min(1.0, signal_trust + random.gauss(0.0, 0.03))), 2)

        # ── Time context (behaviour modifier based on time of day) ────────────
        # scenario zone_cues may carry time_of_day; derive a context bucket from it
        _time_str = ""
        if metadata and "scenario" in metadata:
            _time_str = (metadata["scenario"].get("zone_cues") or {}).get("time_of_day", "")
        if _time_str:
            _hour = int(_time_str.split(":")[0]) if ":" in _time_str else 12
        else:
            # default: use tick as a proxy (early ticks = morning context)
            _hour = (self.simulator.tick % 24)
        if 7 <= _hour <= 9 or 17 <= _hour <= 20:
            time_context = "rush_hour"       # aggressive negotiation, dense traffic
        elif 10 <= _hour <= 16:
            time_context = "midday"          # moderate, school release possible
        elif 20 <= _hour <= 23:
            time_context = "evening"         # faster driving, less pedestrians
        elif _hour < 6 or _hour >= 23:
            time_context = "night"           # fewer hazards but higher speeds
        else:
            time_context = "morning"         # school run hours

        # ── Road quality and surface irregularities ───────────────────────────
        _rc = environment["road_condition"]
        if _rc in ("waterlogged", "potholes"):
            road_quality = "poor"
        elif _rc in ("construction", "ridge"):
            road_quality = "fair"
        else:
            road_quality = "good"
        surface_irregularities = []
        if _rc == "potholes" or any(o.get("type") == "pothole" for o in sensor_data.get("objects", [])):
            surface_irregularities.append("pothole")
        if _rc == "ridge" or any(o.get("type") == "speed_breaker" for o in sensor_data.get("objects", [])):
            surface_irregularities.append("speed_breaker")
        if _rc == "waterlogged":
            surface_irregularities.append("waterlogged")

        # ── Occlusion: is there a parked vehicle or blind curve blocking view? ──
        # Simple heuristic: stationary object > 2m wide lateral offset is blocking sightline
        occlusion = any(
            o.get("is_stationary") and abs(o.get("rel_lat_m", 0)) > 1.5 and o["distance"] < 20.0
            for o in sensor_data.get("objects", [])
        )
        visibility_level = environment.get("visibility", "clear")
        if visibility_level in ("foggy", "night", "rainy", "dusty"):
            occlusion = True   # degraded visibility counts as occlusion

        # ── Nearest object for quick scene summary ───────────────────────────
        objects = sensor_data["objects"]
        nearest = objects[0] if objects else None
        near_desc = (
            f"{nearest['type']}@{nearest['distance']:.1f}m({nearest['side']},{'in-lane' if nearest['in_ego_lane'] else 'off-lane'},ttc={nearest['ttc_s']}s)"
            if nearest else "none"
        )
        scene_summary = (
            f"nearest={near_desc} | n_objects={len(objects)} "
            f"| lane={environment['lane_status']} | signal={environment['traffic_signal']} "
            f"| road={environment['road_condition']} | speed={ego_state['speed']:.1f}km/h "
            f"| space_L={road_geometry['space_left_m']:.1f}m space_R={road_geometry['space_right_m']:.1f}m"
        )

        if self.simulator.event_log:
            _command_output = f"{self.simulator.event_log} | {self.simulator.decision_log}"
        else:
            _command_output = self.simulator.decision_log

        return {
            "command_output":  _command_output,
            "scene_summary":   scene_summary,
            "active_alerts":   [self.simulator.event_log] if self.simulator.event_log else [],
            "sensor_data":     sensor_data,
            "ego_state":       ego_state,
            "road_geometry":   road_geometry,
            "environment":     environment,
            "vehicle_profile": self.simulator.vehicle_profile,
            "event_log":       self.simulator.event_log,
            "hint":            hint,
            "steps_taken":     steps_taken,
            "max_steps":       max_steps,
            "hazard_type":     hazard["hazard_type"],
            "hazard_distance": hazard["hazard_distance"],
            "hazard_status":   hazard["hazard_status"],
            "scenario_stage":  hazard["scenario_stage"],
            # Distance trend — critical for correct action when hazard is receding
            "hazard_moving_away": self.simulator.hazard_moving_away,
            "hazard_trend":    "receding" if self.simulator.hazard_moving_away else "approaching_or_static",
            # ── India-specific context fields ──────────────────────────────
            "signal_trust_score":      signal_trust,     # 0=don't trust signal, 1=fully trust
            "time_context":            time_context,     # rush_hour/midday/evening/night/morning
            "road_quality":            road_quality,     # good/fair/poor
            "surface_irregularities":  surface_irregularities,
            "occlusion":               occlusion,        # True = view blocked, act cautiously
            "metadata":        metadata or {},
        }

    def programmatic_checks(self) -> Dict[str, Any]:
        snapshot = self.simulator.sensor_snapshot()
        min_distance = min((obj["distance"] for obj in snapshot["objects"]), default=999.0)
        ahead_hazards = [
            obj for obj in snapshot["objects"]
            if obj["distance"] < 7.5
            and obj.get("on_road", True)
            and obj.get("rel_fwd_m", obj["distance"]) > 0  # object must be physically ahead
        ]
        # Stage must have transitioned to clearing/cleared (dynamic event fired) AND
        # ego must have moved at least 2m from start — prevents false-positive at episode open.
        incident_cleared = (
            len(ahead_hazards) == 0
            and self.simulator.current_stage in ("clearing", "cleared")
            and self.simulator.ego["x"] > 2.0
        )
        # Require meaningful speed so a stalled ego doesn't count as "progress restored"
        progress_restored = incident_cleared and self.simulator.ego["speed"] >= 0.5
        return {
            "collision": self.simulator.check_collision(),
            "near_miss": self.simulator.check_near_miss(),
            "offroad": self.simulator.check_offroad(),
            "overspeed": self.simulator.check_overspeed(),
            "reached_goal": self.simulator.check_goal(),
            "stuck": self.simulator.check_stuck(),
            "incident_cleared": incident_cleared,
            "progress_restored": progress_restored,
            "safe_distance": min_distance >= 3.0,
            "minimum_distance": round(min_distance, 2),
            "signal_respected": not (
                self.simulator.environment.get("traffic_signal") == "red" and self.simulator.ego["speed"] > 1.5
            ),
        }
