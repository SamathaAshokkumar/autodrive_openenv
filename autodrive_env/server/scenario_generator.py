"""Driving scenario generator with kube-like structure."""

from __future__ import annotations

from copy import deepcopy
import random

from .constants import DEFAULT_SCENE_ENV, DEFAULT_VEHICLE_PROFILE
from ..models import ScenarioSpec


SCENARIO_POOL = [
    ScenarioSpec(
        name="pedestrian_crossing_school_zone",
        type="pedestrian_crossing",
        difficulty=0.2,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "faded", "traffic_signal": "none"},
        actors=[{"type": "pedestrian", "x": 14, "y": 1.2, "vx": -0.6, "behavior": "sudden_cross", "lane": "left"}],
        root_cause="A pedestrian suddenly enters the lane in a dense school-zone style street.",
        alert_message="ALERT: vulnerable road user crossing suddenly ahead",
        correct_fix_description="Slow down or brake and yield while maintaining lane discipline.",
        expected_behavior=["brake", "wait"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "clear_crossing", "message": "Pedestrian has crossed past the ego lane."},
        ],
    ),
    ScenarioSpec(
        name="auto_cut_in_market_road",
        type="auto_cut_in",
        difficulty=0.25,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "lane_status": "missing"},
        actors=[{"type": "auto", "x": 10, "y": -1.2, "vx": 0.2, "behavior": "cut_in", "lane": "left"}],
        root_cause="An auto-rickshaw cuts in unpredictably from the side.",
        alert_message="ALERT: auto-rickshaw cutting into your path",
        correct_fix_description="Reduce speed, avoid aggressive steering, keep safe clearance.",
        expected_behavior=["brake", "wait"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "move_actor_ahead", "message": "Auto-rickshaw completes the merge and starts pulling away."},
            {"trigger_step": 5, "kind": "spawn_vehicle", "message": "A distant ambulance siren is heard from behind.", "actor": {"type": "ambulance", "x": -10, "y": 1.5, "vx": 2.2, "behavior": "emergency_pass", "lane": "right"}},
        ],
    ),
    ScenarioSpec(
        name="bike_blind_spot_merge",
        type="bike_blind_spot",
        difficulty=0.3,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV},
        actors=[{"type": "bike", "x": 8, "y": 1.8, "vx": 0.4, "behavior": "blind_spot_merge", "lane": "right"}],
        root_cause="A bike appears from the blind spot and merges aggressively.",
        alert_message="ALERT: bike emerging from blind spot",
        correct_fix_description="Hold lane or gently brake instead of oscillating.",
        expected_behavior=["wait", "brake"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "move_actor_ahead", "message": "The bike stabilizes ahead after the merge."},
        ],
    ),
    ScenarioSpec(
        name="pothole_ahead_after_rain",
        type="pothole_ahead",
        difficulty=0.4,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "potholes", "visibility": "clear"},
        actors=[{"type": "pothole", "x": 16, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"}],
        root_cause="A deep pothole appears in-lane after rain.",
        alert_message="ALERT: major road defect detected ahead",
        correct_fix_description="Slow down and make a smooth avoidance or controlled pass.",
        expected_behavior=["brake", "steer_left", "steer_right"],
    ),
    ScenarioSpec(
        name="speed_breaker_crowded_lane",
        type="speed_breaker",
        difficulty=0.32,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "ridge", "lane_status": "faded"},
        actors=[{"type": "speed_breaker", "x": 13, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center"}],
        root_cause="An unmarked speed breaker appears in a crowded mixed-traffic lane.",
        alert_message="ALERT: ridge / speed breaker detected ahead",
        correct_fix_description="Slow smoothly, stay stable, then recover speed after crossing.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "clear_static_obstacle", "message": "The speed breaker is now behind the vehicle."},
        ],
    ),
    ScenarioSpec(
        name="traffic_light_ambiguity_police_override",
        type="traffic_light_ambiguity",
        difficulty=0.45,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "traffic_signal": "policeman_override"},
        actors=[{"type": "traffic_police", "x": 12, "y": 0.5, "vx": 0.0, "behavior": "signal_override", "lane": "center"}],
        root_cause="Signal cues are ambiguous and a human override is present.",
        alert_message="ALERT: conflicting traffic control signals ahead",
        correct_fix_description="Proceed cautiously or wait while prioritizing safety over speed.",
        expected_behavior=["wait", "brake"],
    ),
    ScenarioSpec(
        name="crowded_market_spillover",
        type="crowded_market",
        difficulty=0.38,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "missing", "road_condition": "normal"},
        actors=[
            {"type": "pedestrian", "x": 12, "y": 1.4, "vx": -0.4, "behavior": "sudden_cross", "lane": "left"},
            {"type": "auto", "x": 16, "y": -1.0, "vx": 0.1, "behavior": "cut_in", "lane": "left"},
        ],
        root_cause="A crowded market area causes unpredictable spillover into the lane.",
        alert_message="ALERT: crowded market activity intruding into the roadway",
        correct_fix_description="Brake early, remain patient, then recover once the lane opens.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "spawn_vehicle", "message": "A traffic police officer appears ahead managing flow.", "actor": {"type": "traffic_police", "x": 20, "y": 0.5, "vx": 0.0, "behavior": "signal_override", "lane": "center"}},
            {"trigger_step": 5, "kind": "clear_crossing", "message": "The market spillover begins to clear."},
        ],
    ),
    ScenarioSpec(
        name="ambulance_from_rear_corridor",
        type="ambulance_approach",
        difficulty=0.42,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "missing"},
        actors=[{"type": "ambulance", "x": -9, "y": 1.2, "vx": 2.5, "behavior": "emergency_pass", "lane": "right"}],
        root_cause="An ambulance approaches quickly from behind in mixed urban traffic.",
        alert_message="ALERT: ambulance approaching from behind",
        correct_fix_description="Avoid abrupt movement, create space, and yield predictably.",
        expected_behavior=["wait", "steer_left", "steer_right", "brake"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "change_signal", "message": "Traffic slows and opens a narrow corridor for the ambulance.", "traffic_signal": "none"},
        ],
    ),
    ScenarioSpec(
        name="traffic_jam_bottleneck",
        type="traffic_jam",
        difficulty=0.4,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "missing"},
        actors=[
            {"type": "car", "x": 11, "y": 0.2, "vx": 0.0, "behavior": "static", "lane": "center"},
            {"type": "auto", "x": 15, "y": -0.8, "vx": 0.0, "behavior": "static", "lane": "left"},
        ],
        root_cause="A sudden mixed-traffic bottleneck creates a traffic jam ahead.",
        alert_message="ALERT: sudden traffic jam forming ahead",
        correct_fix_description="Slow smoothly, avoid tailgating, and recover only when the jam opens.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "move_actor_ahead", "message": "The bottleneck begins to open and traffic starts moving."},
        ],
    ),
    ScenarioSpec(
        name="cow_crossing_narrow_lane",
        type="animal_crossing",
        difficulty=0.36,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "missing"},
        actors=[{"type": "cow", "x": 13, "y": 1.0, "vx": -0.3, "behavior": "sudden_cross", "lane": "left"}],
        root_cause="A cow wanders into a narrow road corridor unexpectedly.",
        alert_message="ALERT: animal entering the roadway",
        correct_fix_description="Brake early, yield, avoid aggressive steering, then continue once clear.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 5, "kind": "clear_crossing", "message": "The animal has crossed and the lane begins to clear."},
        ],
    ),
    ScenarioSpec(
        name="rain_slippery_pothole_combo",
        type="rain_slippery_road",
        difficulty=0.55,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "waterlogged", "visibility": "rainy", "lane_status": "faded"},
        actors=[
            {"type": "pothole", "x": 14, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"},
            {"type": "bike", "x": 18, "y": 1.2, "vx": 0.2, "behavior": "zig_zag", "lane": "right"},
        ],
        root_cause="Rain reduces visibility and traction while a pothole and unstable bike behavior appear together.",
        alert_message="ALERT: slippery road with pothole and unstable traffic",
        correct_fix_description="Reduce speed, avoid harsh steering, pass the pothole smoothly, then recover carefully.",
        expected_behavior=["brake", "steer_left", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 5, "kind": "clear_static_obstacle", "message": "The deepest waterlogged pothole has been passed."},
        ],
    ),
    ScenarioSpec(
        name="police_manual_override_junction",
        type="police_override",
        difficulty=0.46,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "traffic_signal": "red"},
        actors=[{"type": "traffic_police", "x": 10, "y": 0.2, "vx": 0.0, "behavior": "signal_override", "lane": "center"}],
        root_cause="A police officer overrides the signal flow at a busy junction.",
        alert_message="ALERT: police hand-signal override at junction",
        correct_fix_description="Treat the override cautiously and proceed only when the path is safe.",
        expected_behavior=["wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "change_signal", "message": "Police waves your lane forward.", "traffic_signal": "policeman_override"},
        ],
    ),
]

SUDDEN_EVENT_POOL = [
    {
        "min_difficulty": 0.25,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: ambulance approaching quickly from behind.",
            "hazard_type": "ambulance_approach",
            "actor": {"type": "ambulance", "x": -9, "y": 1.4, "vx": 2.5, "behavior": "emergency_pass", "lane": "right"},
        },
    },
    {
        "min_difficulty": 0.3,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: animal entering the road ahead.",
            "hazard_type": "animal_crossing",
            "actor": {"type": "dog", "x": 11, "y": 1.1, "vx": -0.35, "behavior": "sudden_cross", "lane": "left"},
        },
    },
    {
        "min_difficulty": 0.35,
        "trigger_step": 4,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: traffic police manually overriding junction flow ahead.",
            "hazard_type": "police_override",
            "actor": {"type": "traffic_police", "x": 18, "y": 0.4, "vx": 0.0, "behavior": "signal_override", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.35,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: dense traffic jam forming ahead.",
            "hazard_type": "traffic_jam",
            "actor": {"type": "car", "x": 15, "y": 0.1, "vx": 0.0, "behavior": "static", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.25,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: uneven speed breaker appears ahead.",
            "hazard_type": "speed_breaker",
            "actor": {"type": "speed_breaker", "x": 14, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center"},
        },
    },
]


class ScenarioGenerator:
    def __init__(self, llm=None, mode: str = "simple"):
        self.llm = llm
        self.mode = mode

    def generate(self, skill_profile: dict, difficulty: float, fault_type_hint: str | None = None) -> ScenarioSpec:
        candidates = [scenario for scenario in SCENARIO_POOL if scenario.difficulty <= difficulty + 0.25]
        if fault_type_hint:
            hinted = [scenario for scenario in candidates if scenario.type == fault_type_hint]
            if hinted:
                return self._with_secondary_event(random.choice(hinted), difficulty)
        if not candidates:
            return self._with_secondary_event(SCENARIO_POOL[0], difficulty)
        weak_types = {k for k, v in (skill_profile or {}).items() if v < 0.6}
        weak_candidates = [scenario for scenario in candidates if scenario.type in weak_types]
        return self._with_secondary_event(random.choice(weak_candidates or candidates), difficulty)

    def _with_secondary_event(self, scenario: ScenarioSpec, difficulty: float) -> ScenarioSpec:
        chosen = deepcopy(scenario)
        if chosen.type == "adversarial":
            return chosen
        eligible = [item for item in SUDDEN_EVENT_POOL if difficulty >= item["min_difficulty"] and item["event"]["hazard_type"] != chosen.type]
        if not eligible:
            return chosen
        selected = deepcopy(random.choice(eligible))
        event = selected["event"]
        event["trigger_step"] = selected["trigger_step"]
        chosen.dynamic_events.append(event)
        return chosen
