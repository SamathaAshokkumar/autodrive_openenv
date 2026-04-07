"""Scenario injection for AutoDrive Gym."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict


class ScenarioInjector:
    def inject(self, simulator, scenario: Dict) -> str:
        simulator.objects = []
        simulator.event_log = scenario.get("alert_message", "Driving challenge ahead")
        simulator.environment = deepcopy(scenario.get("environment", {}))
        simulator.vehicle_profile = deepcopy(scenario.get("vehicle_profile", {}))
        simulator.scenario = deepcopy(scenario)
        for actor in scenario.get("actors", []):
            simulator.objects.append(
                {
                    "type": actor.get("type", "vehicle"),
                    "x": float(actor.get("x", actor.get("position", [10, 0])[0])),
                    "y": float(actor.get("y", actor.get("position", [10, 0])[1])),
                    "vx": float(actor.get("vx", -0.5)),
                    "vy": float(actor.get("vy", 0.0)),
                    "behavior": actor.get("behavior", "static"),
                    "lane": actor.get("lane", "center"),
                }
            )
        return f"Injected scenario {scenario.get('name', 'unknown')}"
