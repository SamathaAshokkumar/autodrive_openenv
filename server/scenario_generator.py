"""Driving scenario generator with kube-like structure.

Scenarios are ordered by difficulty so the curriculum naturally serves easier
tasks first, progressively unlocking harder ones.

Difficulty tiers:
  0.20 – 0.35  : Easy   — single hazard, clear resolution signal
  0.36 – 0.50  : Medium — multi-step adaptation required
  0.50 – 0.70  : Hard   — combined hazards, degraded sensing, time pressure
  0.70+        : Expert — adversarial, multi-agent, compounding failures
"""

from __future__ import annotations

from copy import deepcopy
import random

from .constants import DEFAULT_SCENE_ENV, DEFAULT_VEHICLE_PROFILE
from .zone_api import build_zone_cues
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

    # ── MEDIUM-HARD (0.50 – 0.65) ──────────────────────────────────────────────

    ScenarioSpec(
        name="school_bus_stop_rush",
        type="pedestrian_crossing",
        difficulty=0.52,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "lane_status": "faded", "road_condition": "normal"},
        actors=[
            {"type": "pedestrian", "x": 10, "y": 1.5, "vx": -0.7, "behavior": "sudden_cross", "lane": "left"},
            {"type": "pedestrian", "x": 12, "y": -1.3, "vx": -0.5, "behavior": "sudden_cross", "lane": "right"},
            {"type": "bike", "x": 16, "y": 1.0, "vx": 0.3, "behavior": "blind_spot_merge", "lane": "right"},
        ],
        root_cause="Children rushing off a stopped school bus cause multiple simultaneous crossings.",
        alert_message="ALERT: multiple pedestrians crossing near a school bus stop",
        correct_fix_description="Brake fully, wait for all pedestrians to clear, then proceed slowly checking both sides.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 5, "kind": "clear_crossing", "message": "All pedestrians have crossed. Lane is clear."},
        ],
    ),

    ScenarioSpec(
        name="construction_zone_one_lane",
        type="traffic_light_ambiguity",
        difficulty=0.54,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "construction", "lane_status": "missing", "visibility": "dusty"},
        actors=[
            {"type": "car",   "x": 11, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"},
            {"type": "truck", "x": 19, "y": 0.0, "vx": 0.8, "behavior": "static", "lane": "center"},
            {"type": "traffic_police", "x": 14, "y": 0.5, "vx": 0.0, "behavior": "signal_override", "lane": "center"},
        ],
        root_cause="A construction zone narrows traffic to one lane; a flagman controls alternating flow.",
        alert_message="ALERT: construction zone — single lane ahead, flagman controlling flow",
        correct_fix_description="Wait for the flagman's cue, maintain large buffer from construction vehicles, then proceed.",
        expected_behavior=["wait", "brake", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "change_signal", "message": "Flagman signals your lane to move. Construction truck pulling away.", "traffic_signal": "policeman_override"},
            {"trigger_step": 6, "kind": "move_actor_ahead", "message": "Construction vehicle clears the single lane."},
        ],
    ),

    ScenarioSpec(
        name="night_fog_oncoming_highbeam",
        type="rain_slippery_road",
        difficulty=0.58,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "foggy", "traffic_signal": "none", "lane_status": "faded"},
        actors=[
            {"type": "car", "x": 14, "y": 0.2, "vx": -1.8, "behavior": "zig_zag", "lane": "center"},
            {"type": "bike", "x": 11, "y": 1.0, "vx": 0.2, "behavior": "blind_spot_merge", "lane": "right"},
        ],
        root_cause="Fog and an oncoming car's high-beam blind the ego temporarily; a bike is also merging.",
        alert_message="ALERT: low visibility — oncoming high-beam vehicle and merging bike ahead",
        correct_fix_description="Slow down significantly, do not swerve into oncoming traffic, wait for the high-beam to pass.",
        expected_behavior=["brake", "wait", "steer_left", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "move_actor_ahead", "message": "Oncoming car has passed and fog is thinning."},
            {"trigger_step": 2, "kind": "spawn_vehicle", "message": "Sudden alert: speed breaker hidden in fog!", "actor": {"type": "speed_breaker", "x": 13, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center"}},
        ],
    ),

    ScenarioSpec(
        name="waterlogged_underpass",
        type="rain_slippery_road",
        difficulty=0.60,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "waterlogged", "visibility": "rainy", "lane_status": "faded"},
        actors=[
            {"type": "pothole", "x": 10, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"},
            {"type": "pothole", "x": 16, "y": 0.6, "vx": 0.0, "behavior": "static", "lane": "center"},
            {"type": "car",     "x": 20, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"},
        ],
        root_cause="A flooded underpass hides multiple potholes; a stalled car blocks the exit.",
        alert_message="ALERT: waterlogged underpass — hidden potholes, stalled vehicle ahead",
        correct_fix_description="Reduce speed to minimum, steer around potholes carefully, and stop clear of the stalled car.",
        expected_behavior=["brake", "steer_left", "steer_right", "wait"],
        dynamic_events=[
            {"trigger_step": 5, "kind": "move_actor_ahead", "message": "Stalled car is being pushed to the shoulder. Path opening."},
            {"trigger_step": 3, "kind": "spawn_vehicle", "message": "Sudden alert: ambulance approaching from behind.", "hazard_type": "ambulance_approach", "actor": {"type": "ambulance", "x": -8, "y": 1.0, "vx": 2.5, "behavior": "emergency_pass", "lane": "right"}},
        ],
    ),

    ScenarioSpec(
        name="wedding_procession_blockage",
        type="crowded_market",
        difficulty=0.62,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "lane_status": "missing"},
        actors=[
            {"type": "pedestrian", "x": 10, "y": 1.3, "vx": -0.2, "behavior": "sudden_cross", "lane": "left"},
            {"type": "pedestrian", "x": 12, "y": -1.0, "vx": -0.3, "behavior": "sudden_cross", "lane": "right"},
            {"type": "auto",       "x": 15, "y": 0.0, "vx": 0.0,  "behavior": "static",       "lane": "center"},
            {"type": "car",        "x": 18, "y": 0.5, "vx": 0.1,  "behavior": "cut_in",        "lane": "left"},
        ],
        root_cause="A wedding procession spills onto the road; horn-blaring vehicles, dancers, and autos block all lanes.",
        alert_message="ALERT: wedding procession blocking multiple lanes — pedestrians and vehicles mixed",
        correct_fix_description="Brake early, use horn sparingly, creep forward only when a gap opens, and respect all pedestrians.",
        expected_behavior=["brake", "horn", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "spawn_vehicle", "message": "Sudden alert: traffic police arriving to manage procession.", "actor": {"type": "traffic_police", "x": 22, "y": 0.2, "vx": 0.0, "behavior": "signal_override", "lane": "center"}},
            {"trigger_step": 7, "kind": "clear_crossing", "message": "Procession moving to the side. A narrow corridor is opening."},
        ],
    ),

    # ── HARD (0.65 – 0.80) ─────────────────────────────────────────────────────

    ScenarioSpec(
        name="highway_merge_truck_blind",
        type="bike_blind_spot",
        difficulty=0.68,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear", "lane_status": "faded"},
        actors=[
            {"type": "truck", "x": 8,  "y": 2.0,  "vx": 1.2, "behavior": "cut_in",            "lane": "right"},
            {"type": "bike",  "x": 10, "y": -1.5, "vx": 0.5, "behavior": "blind_spot_merge",  "lane": "left"},
            {"type": "car",   "x": 15, "y": 0.0,  "vx": 0.8, "behavior": "zig_zag",           "lane": "center"},
        ],
        root_cause="A large truck cuts in from the right while a bike merges from the left blind spot simultaneously.",
        alert_message="ALERT: simultaneous threats — truck cut-in from right, bike from blind spot left",
        correct_fix_description="Brake moderately, hold center lane, do not steer aggressively in either direction, wait for gap.",
        expected_behavior=["brake", "wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "move_actor_ahead", "message": "Truck has completed merge and the bike stabilises ahead."},
        ],
    ),

    ScenarioSpec(
        name="multi_agent_rush_hour",
        type="multi_agent_chaos",
        difficulty=0.72,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "lane_status": "missing", "traffic_signal": "none"},
        actors=[
            {"type": "auto",       "x": 9,  "y": -1.0, "vx": 0.3,  "behavior": "cut_in",           "lane": "left"},
            {"type": "bike",       "x": 11, "y": 1.5,  "vx": 0.4,  "behavior": "blind_spot_merge",  "lane": "right"},
            {"type": "pedestrian", "x": 13, "y": 1.2,  "vx": -0.6, "behavior": "sudden_cross",      "lane": "left"},
            {"type": "car",        "x": 17, "y": 0.0,  "vx": 0.0,  "behavior": "static",            "lane": "center"},
        ],
        root_cause="Peak-hour chaos: auto, bike, pedestrian, and stalled car all appear within a 20-second window.",
        alert_message="ALERT: rush-hour multi-agent chaos — simultaneous threats from all directions",
        correct_fix_description="Brake decisively, scan all directions, wait for the lowest-risk gap, then accelerate smoothly.",
        expected_behavior=["brake", "wait", "steer_left", "steer_right", "accelerate"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "spawn_vehicle", "message": "Sudden alert: ambulance approaching — clear a corridor immediately.", "hazard_type": "ambulance_approach", "actor": {"type": "ambulance", "x": -7, "y": 1.0, "vx": 3.0, "behavior": "emergency_pass", "lane": "right"}},
            {"trigger_step": 6, "kind": "clear_crossing", "message": "Pedestrian clears. Auto-rickshaw has merged ahead."},
            {"trigger_step": 8, "kind": "move_actor_ahead", "message": "Stalled car being towed. Lane opening."},
        ],
    ),

    # ── ZONE-INFERENCE SCENARIOS (Theme 3.1 — World Modeling) ─────────────────
    # Key design: zone_cues give INDIRECT SIGNALS only.
    # No "zone_type" label is ever given to the agent — it must INFER behavior.

    ScenarioSpec(
        name="hospital_zone_slow_and_quiet",
        type="hospital_zone_inference",
        difficulty=0.35,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear"},
        actors=[
            {"type": "pedestrian", "x": 14, "y": 1.0, "vx": -0.3, "behavior": "sudden_cross", "lane": "left"},
        ],
        root_cause=(
            "The vehicle enters a zone near a hospital. No zone label is given. "
            "The agent must infer appropriate behavior from nearby POIs and ambient cues."
        ),
        alert_message="ALERT: vulnerable pedestrian crossing ahead in slow-moving area",
        correct_fix_description=(
            "Slow down, avoid horn use, yield to pedestrian — the agent must infer "
            "from nearby_places=['hospital'] and visible_signs that this is a sensitive zone."
        ),
        expected_behavior=["brake", "wait"],
        zone_cues=build_zone_cues("hospital_zone"),
        dynamic_events=[
            {"trigger_step": 4, "kind": "clear_crossing", "message": "Pedestrian has crossed. Narrow corridor ahead."},
            {"trigger_step": 6, "kind": "spawn_vehicle", "message": "Sudden alert: ambulance exiting hospital — yield immediately.",
             "actor": {"type": "ambulance", "x": 5, "y": -1.0, "vx": 2.0, "behavior": "emergency_pass", "lane": "right"}},
        ],
    ),
    ScenarioSpec(
        name="school_zone_children_crossing",
        type="school_zone_inference",
        difficulty=0.38,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear"},
        actors=[
            {"type": "pedestrian", "x": 11, "y": 1.2, "vx": -0.7, "behavior": "sudden_cross", "lane": "left"},
            {"type": "pedestrian", "x": 13, "y": -1.0, "vx": -0.5, "behavior": "sudden_cross", "lane": "right"},
        ],
        root_cause=(
            "Two children dart across the road. The agent sees nearby_places=['school', 'playground'] "
            "in zone_cues and must infer it must slow to school-zone speed without being told."
        ),
        alert_message="ALERT: multiple pedestrians crossing — dense zone ahead",
        correct_fix_description=(
            "Brake immediately and wait. Must NOT use horn (school zone etiquette). "
            "Agent infers school zone from POI signals, not direct label."
        ),
        expected_behavior=["brake", "wait"],
        zone_cues=build_zone_cues("school_zone"),
        dynamic_events=[
            {"trigger_step": 5, "kind": "clear_crossing", "message": "Children have crossed. Road opening ahead."},
        ],
    ),
    ScenarioSpec(
        name="temple_zone_procession",
        type="temple_zone_inference",
        difficulty=0.45,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "lane_status": "missing", "visibility": "clear"},
        actors=[
            {"type": "pedestrian", "x": 10, "y": 0.8, "vx": -0.3, "behavior": "sudden_cross", "lane": "center"},
            {"type": "pedestrian", "x": 12, "y": -0.5, "vx": -0.2, "behavior": "sudden_cross", "lane": "center"},
            {"type": "auto",       "x": 16, "y": -1.2, "vx": 0.0,  "behavior": "static",       "lane": "left"},
        ],
        root_cause=(
            "A religious procession partially blocks the road near a temple. "
            "Agent sees nearby_places=['temple'] and visible_signs=['Silence Zone', 'No Horn']. "
            "It must not honk even when path is blocked — infer from signals."
        ),
        alert_message="ALERT: procession and parked auto blocking road — silence zone",
        correct_fix_description=(
            "Slow down and wait. Absolutely no horn use — visible_signs indicate silence zone. "
            "Agent must infer from POI cues that honking is socially unacceptable here."
        ),
        expected_behavior=["brake", "wait", "accelerate"],
        zone_cues=build_zone_cues("temple_zone"),
        dynamic_events=[
            {"trigger_step": 5, "kind": "clear_crossing", "message": "Procession moving to the side — narrow corridor opens."},
            {"trigger_step": 7, "kind": "move_actor_ahead", "message": "Auto has cleared. Lane slowly opening."},
        ],
    ),
    ScenarioSpec(
        name="market_zone_dense_hawkers",
        type="market_zone_inference",
        difficulty=0.42,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "lane_status": "missing"},
        actors=[
            {"type": "pedestrian", "x": 9,  "y": 1.3, "vx": -0.5, "behavior": "sudden_cross", "lane": "left"},
            {"type": "auto",       "x": 13, "y": -0.8, "vx": 0.1, "behavior": "cut_in",       "lane": "left"},
            {"type": "pedestrian", "x": 15, "y": 0.5, "vx": -0.4, "behavior": "sudden_cross", "lane": "center"},
        ],
        root_cause=(
            "A busy market area with hawkers and jaywalking shoppers. "
            "Agent sees nearby_places=['market'] and ambient_cues about hawker carts. "
            "Must slow and be patient — inferred, not told."
        ),
        alert_message="ALERT: dense market spillover — multiple pedestrians and auto cutting in",
        correct_fix_description=(
            "Brake early, be patient with auto cut-in. Horn use must be minimal despite density. "
            "Agent infers from market zone cues that assertive driving is socially inappropriate."
        ),
        expected_behavior=["brake", "wait", "accelerate"],
        zone_cues=build_zone_cues("market_zone"),
        dynamic_events=[
            {"trigger_step": 4, "kind": "clear_crossing", "message": "Pedestrians clear. Auto completes merge ahead."},
        ],
    ),
    ScenarioSpec(
        name="zone_ambiguity_hospital_at_night",
        type="zone_ambiguity",
        difficulty=0.62,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "night_clear"},
        actors=[
            {"type": "car", "x": 18, "y": 0.0, "vx": 0.5, "behavior": "static", "lane": "center"},
        ],
        root_cause=(
            "Ambiguity test: hospital is nearby but it's 3am and the road is empty. "
            "Should the agent still slow down? Answer: yes — ambulances could emerge any time. "
            "Agent must reason about time_sensitivity, not just apply a blanket rule."
        ),
        alert_message="ALERT: low-visibility night road — distant stalled vehicle ahead",
        correct_fix_description=(
            "Maintain slow speed despite the empty road — hospital proximity means ambulances "
            "may emerge at any time. Agent must weigh pedestrian_density vs time_of_day and "
            "decide appropriately without explicit instruction."
        ),
        expected_behavior=["brake", "wait"],
        zone_cues={
            **build_zone_cues("hospital_zone"),
            "time_of_day": "03:00",
            "pedestrian_density": "very_low",
            "ambient_cues": ["ambulance parked outside main entrance", "hospital lights on"],
        },
        dynamic_events=[
            {"trigger_step": 3, "kind": "spawn_vehicle", "message": "Sudden alert: ambulance exiting hospital at speed — yield!",
             "actor": {"type": "ambulance", "x": 6, "y": -1.5, "vx": 2.8, "behavior": "emergency_pass", "lane": "right"}},
        ],
    ),

    # ── AMBIGUOUS ZONE INFERENCE — correct answer is NOT always brake ──────────
    # These scenarios test that the agent reasons from CONDITIONS, not just zone type.
    # A good agent learns: zone type + context → appropriate speed.
    # A bad agent always brakes near sensitive zones regardless of actual conditions.

    ScenarioSpec(
        name="hospital_zone_low_crowd_clear_road",
        type="hospital_zone_inference",
        difficulty=0.40,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear",
                     "traffic_signal": "green"},
        actors=[],
        root_cause=(
            "Hospital zone visible (signs: Slow-Hospital). "
            "But: pedestrian_density=very_low, road is empty, no hazard within 30m, signal=green. "
            "CORRECT ACTION: maintain current speed (accelerate from rest if moving). "
            "WRONG: unnecessary braking with no hazard present wastes time and frustrates flow."
        ),
        alert_message="Notice: hospital zone ahead — road currently clear, signal green",
        correct_fix_description=(
            "Maintain speed or resume normal driving. Zone sign is present but no hazard exists. "
            "Only slow if a hazard appears. Over-braking on empty hospital roads is incorrect."
        ),
        expected_behavior=["accelerate", "wait"],
        zone_cues={
            **build_zone_cues("hospital_zone"),
            "pedestrian_density": "very_low",
            "time_of_day": "14:00",
            "ambient_cues": ["hospital entrance 40m to the right", "no patients visible on road"],
        },
        dynamic_events=[
            {"trigger_step": 4, "kind": "spawn_vehicle",
             "message": "Sudden: ambulance exiting driveway — now the agent SHOULD brake and yield.",
             "actor": {"type": "ambulance", "x": 8, "y": -1.5, "vx": 1.5, "behavior": "emergency_pass", "lane": "right"}},
        ],
    ),

    ScenarioSpec(
        name="school_zone_after_hours_empty",
        type="school_zone_inference",
        difficulty=0.42,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear",
                     "traffic_signal": "none"},
        actors=[],
        root_cause=(
            "School zone signs are visible but time_of_day=22:00 (night), "
            "pedestrian_density=none, no children, school is closed. "
            "CORRECT ACTION: resume normal driving speed. "
            "WRONG: mandatory brake in school zone at 10pm shows lack of contextual reasoning."
        ),
        alert_message="Notice: school zone — no activity detected, late night",
        correct_fix_description=(
            "Proceed at normal speed. School zone is inactive at this hour. "
            "Good agent reads pedestrian_density and time_of_day to contextualise zone cues."
        ),
        expected_behavior=["accelerate"],
        zone_cues={
            **build_zone_cues("school_zone"),
            "pedestrian_density": "none",
            "time_of_day": "22:00",
            "ambient_cues": ["school gate closed and locked", "no buses or children visible", "streetlights on"],
        },
        dynamic_events=[],
    ),

    ScenarioSpec(
        name="market_zone_early_morning_quiet",
        type="market_zone_inference",
        difficulty=0.38,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear"},
        actors=[],
        root_cause=(
            "Market zone signs visible but time_of_day=06:00, stalls not yet open, "
            "pedestrian_density=very_low. "
            "CORRECT: maintain or gently increase speed — market is inactive. "
            "WRONG: braking to crawl speed in an empty market street is unnecessary."
        ),
        alert_message="Notice: market area — early morning, few pedestrians",
        correct_fix_description=(
            "Proceed at moderate speed. No hawkers or jaywalkers in road. "
            "Contextually-aware agent recognises this is a pre-opening hour."
        ),
        expected_behavior=["accelerate", "wait"],
        zone_cues={
            **build_zone_cues("market_zone"),
            "pedestrian_density": "very_low",
            "time_of_day": "06:00",
            "ambient_cues": ["shop shutters closed", "street sweeper at far end", "no pedestrians on road"],
        },
        dynamic_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "Sudden: early delivery truck reversing out of alley — slow now.",
             "actor": {"type": "truck", "x": 9, "y": 0.0, "vx": -0.5, "behavior": "zig_zag", "lane": "center"}},
        ],
    ),

    ScenarioSpec(
        name="temple_zone_no_procession_normal_flow",
        type="temple_zone_inference",
        difficulty=0.44,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear"},
        actors=[],
        root_cause=(
            "Temple zone with No-Horn sign visible. But it's a regular Tuesday afternoon, "
            "no procession, no devotees on road, pedestrian density=low. "
            "CORRECT: proceed at normal speed, avoid horn, but no need to brake. "
            "Signs say 'No Horn' — that is the constraint, not speed reduction."
        ),
        alert_message="Notice: temple zone — silence required, road currently clear",
        correct_fix_description=(
            "Maintain speed. Absolute constraint: do NOT use horn. "
            "No speed reduction needed — no hazard in road at this moment."
        ),
        expected_behavior=["accelerate", "wait"],
        zone_cues={
            **build_zone_cues("temple_zone"),
            "pedestrian_density": "low",
            "time_of_day": "15:00",
            "ambient_cues": ["temple gate visible on right", "no procession active", "occasional pedestrian on footpath only"],
        },
        dynamic_events=[
            {"trigger_step": 4, "kind": "spawn_vehicle",
             "message": "A devotee suddenly steps off footpath into road — brake carefully, still no horn.",
             "actor": {"type": "pedestrian", "x": 10, "y": 0.8, "vx": -0.3, "behavior": "sudden_cross", "lane": "left"}},
        ],
    ),

    ScenarioSpec(
        name="highway_after_sensitive_zone",
        type="zone_ambiguity",
        difficulty=0.50,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal", "visibility": "clear",
                     "lane_status": "clear", "traffic_signal": "none"},
        actors=[],
        root_cause=(
            "The ego vehicle has just passed through a hospital zone and is now on an "
            "open highway stretch. zone_cues now show nearby_places=['highway_entry'], "
            "pedestrian_density=low, no sensitive POI. "
            "CORRECT: ACCELERATE back to highway speed — agent must shed the earlier slow-zone "
            "constraint and adapt to the new context. "
            "WRONG: continuing hospital-zone caution on an open highway is over-conservative."
        ),
        alert_message="Hospital zone cleared — entering open highway segment",
        correct_fix_description=(
            "Accelerate to highway speed. Zone context has changed. "
            "Continuing to crawl after a sensitive zone has ended is a reasoning failure."
        ),
        expected_behavior=["accelerate"],
        zone_cues={
            **build_zone_cues("highway_entry"),
            "pedestrian_density": "low",
            "time_of_day": "11:00",
            "ambient_cues": ["open road visible for 500m", "no pedestrians", "highway speed limit sign: 80kph"],
        },
        dynamic_events=[],
    ),

    ScenarioSpec(
        name="construction_zone_flagman_waves_forward",
        type="traffic_light_ambiguity",
        difficulty=0.55,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "construction", "lane_status": "missing",
                     "traffic_signal": "policeman_override"},
        actors=[
            {"type": "traffic_police", "x": 12, "y": 0.3, "vx": 0.0, "behavior": "signal_override", "lane": "center"},
        ],
        root_cause=(
            "Construction zone — single lane, flagman present. "
            "At step 1: flagman is holding stop sign. At step 3: flagman waves forward. "
            "CORRECT: wait when flagman says stop, then ACCELERATE when waved forward. "
            "WRONG: staying stopped after the forward wave — agent must respond to dynamic signal."
        ),
        alert_message="ALERT: construction zone flagman — obey hand signals",
        correct_fix_description=(
            "Wait when flagman shows stop. Immediately accelerate when flagman waves forward. "
            "Tests dynamic signal reading — not a static brake-and-wait scenario."
        ),
        expected_behavior=["wait", "accelerate"],
        dynamic_events=[
            {"trigger_step": 3, "kind": "change_signal",
             "message": "Flagman waves your lane forward — proceed through construction zone.",
             "traffic_signal": "policeman_override"},
        ],
    ),

    ScenarioSpec(
        name="police_zone_stop_then_proceed",
        type="police_override",
        difficulty=0.48,
        vehicle_profile=dict(DEFAULT_VEHICLE_PROFILE),
        environment={**DEFAULT_SCENE_ENV, "road_condition": "normal",
                     "traffic_signal": "red"},
        actors=[
            {"type": "traffic_police", "x": 11, "y": 0.3, "vx": 0.0, "behavior": "signal_override", "lane": "center"},
        ],
        root_cause=(
            "Police officer at junction with red light. Officer initially holds stop hand. "
            "At step 4 they wave the agent's lane forward (overriding the red light). "
            "CORRECT: stop when signal=red+stop hand, then accelerate when officer waves. "
            "WRONG: remaining stopped even after police wave-through — over-deferring."
        ),
        alert_message="ALERT: police override junction — read officer hand signal",
        correct_fix_description=(
            "Stop on red. When officer waves forward, ignore red light and accelerate. "
            "Police override trumps traffic signal — agent must understand authority hierarchy."
        ),
        expected_behavior=["wait", "brake", "accelerate"],
        dynamic_events=[
            {"trigger_step": 4, "kind": "change_signal",
             "message": "Police officer waves your lane forward — red light overridden, proceed.",
             "traffic_signal": "policeman_override"},
        ],
    ),
]

SUDDEN_EVENT_POOL = [
    {
        "min_difficulty": 0.20,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: ambulance approaching quickly from behind — pull left and create a corridor.",
            "hazard_type": "ambulance_approach",
            "actor": {"type": "ambulance", "x": -9, "y": 1.4, "vx": 2.5, "behavior": "emergency_pass", "lane": "right"},
        },
    },
    {
        "min_difficulty": 0.25,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: animal entering the road ahead — brake and wait.",
            "hazard_type": "animal_crossing",
            "actor": {"type": "dog", "x": 11, "y": 1.1, "vx": -0.35, "behavior": "sudden_cross", "lane": "left"},
        },
    },
    {
        "min_difficulty": 0.25,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: uneven speed breaker appears ahead — slow down before hitting it.",
            "hazard_type": "speed_breaker",
            "actor": {"type": "speed_breaker", "x": 14, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.30,
        "trigger_step": 4,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: a child has dashed into the road — emergency brake!",
            "hazard_type": "pedestrian_crossing",
            "actor": {"type": "pedestrian", "x": 9, "y": 0.8, "vx": -0.9, "behavior": "sudden_cross", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.35,
        "trigger_step": 4,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: traffic police manually overriding junction flow — slow and obey hand signal.",
            "hazard_type": "police_override",
            "actor": {"type": "traffic_police", "x": 18, "y": 0.4, "vx": 0.0, "behavior": "signal_override", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.35,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: dense traffic jam forming ahead — maintain large buffer and slow immediately.",
            "hazard_type": "traffic_jam",
            "actor": {"type": "car", "x": 15, "y": 0.1, "vx": 0.0, "behavior": "static", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.40,
        "trigger_step": 3,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: pothole! — steer around it and reduce speed.",
            "hazard_type": "pothole_ahead",
            "actor": {"type": "pothole", "x": 13, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center"},
        },
    },
    {
        "min_difficulty": 0.45,
        "trigger_step": 4,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: cow wandering into the road — brake and yield.",
            "hazard_type": "animal_crossing",
            "actor": {"type": "cow", "x": 12, "y": 1.0, "vx": -0.2, "behavior": "sudden_cross", "lane": "left"},
        },
    },
    {
        "min_difficulty": 0.50,
        "trigger_step": 2,
        "event": {
            "kind": "spawn_vehicle",
            "message": "Sudden alert: zig-zagging bike ahead possibly intoxicated — keep maximum distance.",
            "hazard_type": "bike_blind_spot",
            "actor": {"type": "bike", "x": 12, "y": 0.5, "vx": 0.4, "behavior": "zig_zag", "lane": "center"},
        },
    },
]


class ScenarioGenerator:
    def __init__(self, llm=None, mode: str = "simple"):
        self.llm = llm
        self.mode = mode

    def generate(self, skill_profile: dict, difficulty: float, fault_type_hint: str | None = None) -> ScenarioSpec:
        # Always include a buffer so the agent sees scenarios slightly above its comfort zone
        candidates = [s for s in SCENARIO_POOL if s.difficulty <= difficulty + 0.30]
        if not candidates:
            candidates = [SCENARIO_POOL[0]]

        if fault_type_hint:
            hinted = [s for s in candidates if s.type == fault_type_hint]
            if hinted:
                return self._with_secondary_event(random.choice(hinted), difficulty)

        weak_types = {k for k, v in (skill_profile or {}).items() if v < 0.6}
        weak_candidates = [s for s in candidates if s.type in weak_types]

        # Prefer unmastered weak areas; fall back to full candidate pool
        pool = weak_candidates or candidates

        # Among equals, slightly prefer scenarios right at the current difficulty frontier
        # so the agent is always being stretched
        frontier = [s for s in pool if abs(s.difficulty - difficulty) <= 0.20]
        chosen = random.choice(frontier or pool)

        # ── Multi-scenario combination (difficulty >= 0.50) ───────────────────
        # At intermediate+ difficulty, combine up to 3 scenarios into one episode.
        # This is the core of real Indian driving: nothing happens in isolation.
        # Combinations are created by merging actors and cascading dynamic events.
        if difficulty >= 0.50:
            return self._with_multi_scenario_combination(chosen, candidates, difficulty)

        return self._with_secondary_event(chosen, difficulty)

    def _with_multi_scenario_combination(
        self,
        primary: ScenarioSpec,
        candidates: list,
        difficulty: float,
    ) -> ScenarioSpec:
        """Combine 2-3 scenarios into one episode by merging actors + events.

        Design rules:
        - Max 3 combined scenarios per episode (prevents chaos overload)
        - Secondary scenarios' actors are offset in time (trigger_step) so the
          agent faces them sequentially, not all at once
        - Zone cues are merged from all contributing scenarios
        - Difficulty gates the number of combinations:
            0.50-0.65 → 2 scenarios combined
            0.65+     → 2-3 scenarios combined
        - Always add a SUDDEN_EVENT_POOL event if room permits
        """
        chosen = deepcopy(primary)
        if chosen.type == "adversarial":
            return chosen

        # How many scenarios to combine (2 or 3 based on difficulty)
        max_extra = 1 if difficulty < 0.65 else 2
        n_extra = random.randint(1, max_extra)

        # Pick secondary scenarios that are NOT the same type as primary
        secondaries = [
            s for s in candidates
            if s.type != chosen.type
            and s.difficulty <= difficulty + 0.10
            and s.actors  # must have actual actors to inject
        ]
        random.shuffle(secondaries)
        secondaries = secondaries[:n_extra]

        step_offset = 4  # secondary hazards appear after this many steps
        for sec in secondaries:
            step_offset += 3
            # Inject secondary actors as dynamic spawn events
            for actor in sec.actors[:2]:  # at most 2 actors per secondary
                chosen.dynamic_events.append({
                    "trigger_step": step_offset,
                    "kind": "spawn_vehicle",
                    "message": f"Sudden alert ({sec.type}): {sec.alert_message}",
                    "hazard_type": sec.type,
                    "actor": dict(actor),
                })
            step_offset += 2
            # Merge zone cues (add any new ambient cues from the secondary)
            sec_cues = getattr(sec, "zone_cues", {}) or {}
            if sec_cues and not chosen.zone_cues:
                chosen.zone_cues = sec_cues
            elif sec_cues:
                # Merge ambient cues without duplicating
                existing_cues = chosen.zone_cues.get("ambient_cues", []) or []
                new_cues = sec_cues.get("ambient_cues", []) or []
                merged = list(dict.fromkeys(existing_cues + new_cues))
                chosen.zone_cues = {**chosen.zone_cues, "ambient_cues": merged[:4]}

        # Add one SUDDEN_EVENT_POOL event at the end if difficulty warrants it
        eligible = [
            item for item in SUDDEN_EVENT_POOL
            if difficulty >= item["min_difficulty"]
            and item["event"].get("hazard_type") != chosen.type
        ]
        if eligible:
            selected = deepcopy(random.choice(eligible))
            event = selected["event"]
            event["trigger_step"] = step_offset + 1
            chosen.dynamic_events.append(event)

        # Update name and root_cause to reflect the combination
        if secondaries:
            combo_types = "+".join(s.type for s in secondaries)
            chosen.name = f"{chosen.name}__with_{combo_types}"
            chosen.root_cause = (
                f"[COMBINED EPISODE: {chosen.type} + {combo_types}] "
                + chosen.root_cause
            )

        return chosen

    def _with_secondary_event(self, scenario: ScenarioSpec, difficulty: float) -> ScenarioSpec:
        chosen = deepcopy(scenario)
        if chosen.type == "adversarial":
            return chosen
        eligible = [
            item for item in SUDDEN_EVENT_POOL
            if difficulty >= item["min_difficulty"]
            and item["event"].get("hazard_type") != chosen.type
        ]
        if not eligible:
            return chosen
        selected = deepcopy(random.choice(eligible))
        event = selected["event"]
        event["trigger_step"] = selected["trigger_step"]
        chosen.dynamic_events.append(event)
        return chosen