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
}

MAX_STEPS = 20
