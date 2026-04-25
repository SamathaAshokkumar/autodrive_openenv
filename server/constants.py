"""Server-side constants for AutoDrive Gym."""

DRIVING_ACTIONS = {
    "accelerate",
    "brake",
    "steer_left",
    "steer_right",
    "horn",
    "wait",
    "change_lane_left",
    "change_lane_right",
}

SCENARIO_TYPES = {
    "pedestrian_crossing":     {"tier": 1, "min_difficulty": 0.00},
    "auto_cut_in":             {"tier": 1, "min_difficulty": 0.00},
    "bike_blind_spot":         {"tier": 1, "min_difficulty": 0.00},
    "pothole_ahead":           {"tier": 2, "min_difficulty": 0.25},
    "speed_breaker":           {"tier": 2, "min_difficulty": 0.25},
    "crowded_market":          {"tier": 2, "min_difficulty": 0.30},
    "ambulance_approach":      {"tier": 2, "min_difficulty": 0.35},
    "police_override":         {"tier": 2, "min_difficulty": 0.35},
    "traffic_jam":             {"tier": 2, "min_difficulty": 0.35},
    "animal_crossing":         {"tier": 2, "min_difficulty": 0.32},
    "rain_slippery_road":      {"tier": 3, "min_difficulty": 0.50},
    "traffic_light_ambiguity": {"tier": 2, "min_difficulty": 0.30},
    "school_bus_stop":         {"tier": 3, "min_difficulty": 0.50},
    "construction_zone":       {"tier": 3, "min_difficulty": 0.52},
    "night_fog":               {"tier": 3, "min_difficulty": 0.56},
    "waterlogged_underpass":   {"tier": 3, "min_difficulty": 0.58},
    "wedding_procession":      {"tier": 3, "min_difficulty": 0.60},
    "highway_merge_truck":     {"tier": 4, "min_difficulty": 0.65},
    "multi_agent_chaos":       {"tier": 4, "min_difficulty": 0.70},
    "adversarial":             {"tier": 5, "min_difficulty": 0.75},
    # Zone-inference scenarios (Theme 3.1 — world modeling)
    "hospital_zone_inference": {"tier": 2, "min_difficulty": 0.35},
    "school_zone_inference":   {"tier": 2, "min_difficulty": 0.35},
    "temple_zone_inference":   {"tier": 3, "min_difficulty": 0.45},
    "market_zone_inference":   {"tier": 2, "min_difficulty": 0.38},
    "zone_ambiguity":          {"tier": 4, "min_difficulty": 0.60},
}

# Actor intent vocabulary (used by IntentInferenceAgent)
ACTOR_INTENTS = {
    "rush":       "Driver in a hurry — weaves, cuts gaps, may run signals",
    "cautious":   "Slow, careful driver — predictable, respects rules",
    "distracted": "Mobile-phone / inattentive driver — erratic, drifts",
    "aggressive": "Hostile driver — tailgates, cuts in, ignores signals",
    "yielding":   "Cooperative driver — signals intent, gives right-of-way",
}

# Negotiation outcome vocabulary
NEGOTIATION_OUTCOMES = {
    "smooth_pass":    "Ego proceeds without delay — good negotiation",
    "delayed_pass":   "Ego waits briefly then proceeds — acceptable",
    "deadlock":       "Both agents pause — minor inefficiency",
    "collision_risk": "Confrontation — likely near-miss or worse",
}

# Zone type vocabulary (for internal labeling only — NEVER exposed to agent)
ZONE_TYPES = {
    "hospital_zone":    "Slow, no horn, watch for ambulances",
    "school_zone":      "Very slow, watch for children, especially at school hours",
    "temple_zone":      "Slow, no horn, expect processions",
    "market_zone":      "Slow, high pedestrian density, expect jaywalking",
    "highway_merge":    "Assertive merge, check blind spots, maintain speed",
    "police_checkpoint": "Full compliance, no violations",
    "construction_zone": "Slow, watch workers, expect lane shifts",
}

DEFAULT_VEHICLE_PROFILE = {
    "length": 4.2,
    "width": 1.8,
    "height": 1.55,
    "wheel_base": 2.65,
    "turning_radius": 5.5,
    "max_speed": 60.0,
    "camera_fov": 120,
    "sensor_range": 30.0,
    "camera_mounts": {
        "front": {"x": 1.4, "y": 0.0, "z": 1.45, "pitch": 0.0, "yaw": 0.0},
        "rear": {"x": -1.2, "y": 0.0, "z": 1.35, "pitch": 0.0, "yaw": 180.0},
    },
    "sensor_mounts": {
        "lidar": {"x": 0.0, "y": 0.0, "z": 1.8},
        "front_radar": {"x": 1.6, "y": 0.0, "z": 0.6},
        "imu": {"x": 0.0, "y": 0.0, "z": 0.8},
        "gnss": {"x": 0.0, "y": 0.0, "z": 1.9},
    },
}

DEFAULT_SCENE_ENV = {
    "road_condition": "normal",
    "visibility": "clear",
    "lane_status": "clear",
    "traffic_signal": "none",
    "region": "india",
    # Road geometry — standard 2-lane undivided Indian road
    "road_width_m": 7.2,        # total road width including shoulders
    "lane_width_m": 3.6,        # width of one lane
    "num_lanes": 2,             # total lanes (both directions)
    "shoulder_width_m": 1.1,    # paved shoulder each side
    "surface_type": "asphalt",  # asphalt | concrete | gravel | waterlogged
}

MAX_STEPS = 20