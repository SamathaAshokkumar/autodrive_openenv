"""Adversarial designer for advanced driving scenarios."""

from __future__ import annotations

import random


class AdversarialDesigner:
    def __init__(self, llm=None, backend=None, max_steps: int = 20):
        self.llm = llm
        self.backend = backend
        self.max_steps = max_steps

    def design(self, skill_profile: dict, difficulty: float):
        return {
            "name": "adversarial_chaos_intersection",
            "type": "adversarial",
            "difficulty": max(0.75, difficulty),
            "vehicle_profile": {
                "length": 4.2,
                "width": 1.8,
                "turning_radius": 5.5,
                "max_speed": 60.0,
                "camera_fov": 120,
                "sensor_range": 30.0,
            },
            "environment": {
                "road_condition": random.choice(["normal", "waterlogged", "potholes"]),
                "visibility": random.choice(["clear", "rain", "low_visibility"]),
                "lane_status": random.choice(["clear", "faded", "missing"]),
                "traffic_signal": random.choice(["red", "green", "policeman_override"]),
            },
            "actors": [
                {"type": "pedestrian", "x": 14, "y": 1.0, "vx": -0.4, "behavior": "fake_cross", "lane": "left"},
                {"type": "vehicle", "x": 11, "y": -1.5, "vx": 0.4, "behavior": "zig_zag", "lane": "right"},
            ],
            "root_cause": "Multiple unpredictable Indian-road actors create an adversarial scene.",
            "alert_message": "ALERT: multiple unpredictable agents ahead",
            "correct_fix_description": "Use smooth, conservative control to avoid all hazards and stay drivable.",
            "expected_behavior": ["brake", "wait", "steer_left", "steer_right"],
            "red_herrings": ["a pedestrian looks like they may cross but may stop", "vehicle lateral movement may reverse"],
        }
