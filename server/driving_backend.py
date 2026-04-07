"""Driving backend for AutoDrive Gym.

Equivalent to kube's backend: reset base state, inject scenario, execute one
action, advance the world, and compute programmatic safety checks.
"""

from __future__ import annotations

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

    def update(self):
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
        return {
            "hazard_type": "",
            "hazard_distance": 999.0,
            "hazard_status": "",
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
        detected_objects = []
        for obj in self.objects:
            dx = obj["x"] - self.ego["x"]
            dy = obj["y"] - self.ego["y"]
            detected_objects.append(
                {
                    "type": obj["type"],
                    "distance": round(hypot(dx, dy), 2),
                    "angle": round(atan2(dy, dx), 3),
                    "lane": obj.get("lane", "unknown"),
                    "crossing": obj.get("behavior") in {"sudden_cross", "fake_cross"},
                    "on_road": abs(dy) < 2.2,
                    "behavior": obj.get("behavior", "static"),
                }
            )
        return {
            "objects": sorted(detected_objects, key=lambda item: item["distance"])[:8],
            "lane_info": self.environment.get("lane_status", "clear"),
            "traffic_signal": self.environment.get("traffic_signal", "none"),
            "camera_view": self.environment.get("camera_view", "front_rgb_summary"),
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
        return self.steps_without_progress >= 6


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
        return self.actions.apply(self.simulator, action, value)

    def update(self):
        self.simulator.update()

    def build_observation(self, steps_taken: int, max_steps: int, hint: str = "", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        sensor_data = self.simulator.sensor_snapshot()
        hazard = self.simulator.hazard_summary()
        ego_state = {
            "speed": round(self.simulator.ego["speed"], 2),
            "steering": round(self.simulator.ego["steering"], 2),
            "position": [round(self.simulator.ego["x"], 2), round(self.simulator.ego["y"], 2)],
            "lane": self.simulator.ego["lane"],
            "heading": self.simulator.ego["heading"],
        }
        environment = {
            "road_condition": self.simulator.environment.get("road_condition", "normal"),
            "visibility": self.simulator.environment.get("visibility", "clear"),
            "lane_status": self.simulator.environment.get("lane_status", "clear"),
            "traffic_signal": self.simulator.environment.get("traffic_signal", "none"),
        }
        scene_summary = (
            f"objects={len(sensor_data['objects'])} | lane={environment['lane_status']} | "
            f"signal={environment['traffic_signal']} | road={environment['road_condition']} | "
            f"speed={ego_state['speed']}"
        )
        return {
            "command_output": self.simulator.decision_log or self.simulator.event_log,
            "scene_summary": scene_summary,
            "active_alerts": [self.simulator.event_log] if self.simulator.event_log else [],
            "sensor_data": sensor_data,
            "ego_state": ego_state,
            "environment": environment,
            "vehicle_profile": self.simulator.vehicle_profile,
            "event_log": self.simulator.event_log,
            "hint": hint,
            "steps_taken": steps_taken,
            "max_steps": max_steps,
            "hazard_type": hazard["hazard_type"],
            "hazard_distance": hazard["hazard_distance"],
            "hazard_status": hazard["hazard_status"],
            "scenario_stage": hazard["scenario_stage"],
            "metadata": metadata or {},
        }

    def programmatic_checks(self) -> Dict[str, Any]:
        snapshot = self.simulator.sensor_snapshot()
        min_distance = min((obj["distance"] for obj in snapshot["objects"]), default=999.0)
        ahead_hazards = [
            obj for obj in snapshot["objects"]
            if obj["distance"] < 7.5 and obj.get("on_road", True) and abs(float(obj.get("angle", 0.0))) < 1.2
        ]
        incident_cleared = len(ahead_hazards) == 0 and self.simulator.ego["x"] > 6.0
        progress_restored = incident_cleared and self.simulator.ego["speed"] >= 1.0
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
