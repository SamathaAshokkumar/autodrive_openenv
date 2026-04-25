"""Simulated Zone & Tool API — Level-3 World Inference for Social Driving Intelligence.

Design principle (from senior review):
  ❌  env: hospital_zone = True       → agent just follows a rule (shallow)
  ✅  tool: get_nearby_places()       → returns ["hospital", "pharmacy"]
      + pedestrian_density = "high"
      + visible_signs = ["No Honking"]
      + ambient_cues = ["ambulance parked outside"]
      → agent must INFER → this zone requires slow, quiet driving

Judges evaluate whether the agent understands the world, not whether it
follows pre-programmed rules.  This module creates the inference challenge.

Tool functions exposed to the agent/pipeline:
    get_nearby_places(location, zone_context)  → POI list + density + signs
    get_road_context(road_type)                → speed limit, surface, usage
    get_ambient_context(scenario_type)         → time, weather, event
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# ── POI database (each entry defines observable context, NOT behavioral rules) ─

_POI_DATABASE: Dict[str, Dict[str, Any]] = {
    "hospital": {
        "typical_pedestrian_density": "high",
        "speed_context": "30kph_zone",
        "ambient_signs": ["No Honking", "Slow — Hospital Zone", "Ambulance Entry"],
        "ambient_cues_pool": [
            "ambulance parked outside main entrance",
            "wheelchair ramp visible near gate",
            "nurses and patients moving slowly on road",
            "green cross visible on building",
        ],
    },
    "school": {
        "typical_pedestrian_density": "very_high",
        "speed_context": "20kph_zone",
        "ambient_signs": ["School Ahead", "Children Crossing", "Speed Limit 20"],
        "ambient_cues_pool": [
            "school buses lined up on road shoulder",
            "children in uniform crossing with bags",
            "crossing guard waving vehicles to slow",
            "school bell heard in background",
        ],
    },
    "temple": {
        "typical_pedestrian_density": "very_high",
        "speed_context": "25kph_zone",
        "ambient_signs": ["Silence Zone", "Religious Site", "No Horn"],
        "ambient_cues_pool": [
            "devotees in traditional dress walking on road",
            "flower garlands placed near entry",
            "loudspeaker prayers audible",
            "incense stall blocking part of shoulder",
        ],
    },
    "market": {
        "typical_pedestrian_density": "very_high",
        "speed_context": "20kph_zone",
        "ambient_signs": ["Busy Market Ahead", "Pedestrian Zone"],
        "ambient_cues_pool": [
            "hawkers with carts spilling onto road",
            "shoppers crossing randomly between stalls",
            "delivery bikes parked obstructing lane",
            "children running between vehicles",
        ],
    },
    "highway_entry": {
        "typical_pedestrian_density": "low",
        "speed_context": "80kph_zone",
        "ambient_signs": ["Highway Entry", "Merge Ahead", "Speed Limit 80"],
        "ambient_cues_pool": [
            "vehicles accelerating to highway speed",
            "clear lane markings and wide shoulders",
            "overhead gantry with speed limit",
            "no pedestrians visible",
        ],
    },
    "railway_crossing": {
        "typical_pedestrian_density": "medium",
        "speed_context": "30kph_zone",
        "ambient_signs": ["Railway Crossing Ahead", "Stop on Red", "Level Crossing"],
        "ambient_cues_pool": [
            "boom barrier partially down",
            "train whistle audible in distance",
            "flashing red light at crossing",
            "people waiting at sides for train to pass",
        ],
    },
    "police_station": {
        "typical_pedestrian_density": "medium",
        "speed_context": "30kph_zone",
        "ambient_signs": ["Police Station", "No Parking", "Traffic Compliance Zone"],
        "ambient_cues_pool": [
            "police vehicles parked outside",
            "uniformed officers visible on road",
            "CCTV camera pointed at road",
            "yellow traffic cone barrier near entry",
        ],
    },
    "construction": {
        "typical_pedestrian_density": "medium",
        "speed_context": "30kph_zone",
        "ambient_signs": ["Construction Ahead", "Lane Closed", "Workers Present"],
        "ambient_cues_pool": [
            "orange cones narrowing lane",
            "workers in yellow vests near edge",
            "heavy machinery visible on shoulder",
            "uneven road surface begins",
        ],
    },
    "pharmacy": {
        "typical_pedestrian_density": "medium",
        "speed_context": "30kph_zone",
        "ambient_signs": ["Slow", "Hospital Nearby"],
        "ambient_cues_pool": [
            "patient being helped out of car",
            "medical supplies being unloaded",
        ],
    },
    "playground": {
        "typical_pedestrian_density": "high",
        "speed_context": "20kph_zone",
        "ambient_signs": ["Children Playing", "Slow"],
        "ambient_cues_pool": [
            "children running near road",
            "ball rolled onto road from playground",
        ],
    },
}

# ── Zone context templates ─────────────────────────────────────────────────────
# Each zone is described through indirect signals only.
# There is NO "zone_type" field — the agent must infer from signals.

ZONE_CONTEXT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "hospital_zone": {
        "nearby_places": ["hospital", "pharmacy"],
        "pedestrian_density": "high",
        "road_type": "urban",
        "time_sensitivity": "always",
    },
    "school_zone": {
        "nearby_places": ["school", "playground"],
        "pedestrian_density": "very_high",
        "road_type": "residential",
        "time_sensitivity": "school_hours",
    },
    "temple_zone": {
        "nearby_places": ["temple"],
        "pedestrian_density": "very_high",
        "road_type": "narrow_urban",
        "time_sensitivity": "morning_evening",
    },
    "market_zone": {
        "nearby_places": ["market"],
        "pedestrian_density": "very_high",
        "road_type": "urban",
        "time_sensitivity": "daytime",
    },
    "highway_merge": {
        "nearby_places": ["highway_entry"],
        "pedestrian_density": "low",
        "road_type": "highway",
        "time_sensitivity": "always",
    },
    "police_checkpoint": {
        "nearby_places": ["police_station"],
        "pedestrian_density": "medium",
        "road_type": "urban",
        "time_sensitivity": "always",
    },
    "construction_zone": {
        "nearby_places": ["construction"],
        "pedestrian_density": "medium",
        "road_type": "urban",
        "time_sensitivity": "daytime",
    },
}

# Ambiguity scenarios: same place, different time → different safe speed
# These test whether the agent reasons about context, not just rule-lookup
AMBIGUITY_CASES: List[Dict[str, Any]] = [
    {
        "description": "hospital nearby but road is empty at 3am — still slow?",
        "nearby_places": ["hospital"],
        "pedestrian_density": "very_low",
        "time_of_day": "03:00",
        "expected_reasoning": "some slowdown still appropriate — ambulances could emerge any time",
    },
    {
        "description": "school zone but it is Sunday noon",
        "nearby_places": ["school"],
        "pedestrian_density": "low",
        "time_of_day": "12:00",
        "time_sensitivity": "school_hours",
        "expected_reasoning": "less strict — school is closed, but children may still be playing",
    },
    {
        "description": "ambulance siren behind — agent is in market zone",
        "nearby_places": ["market"],
        "pedestrian_density": "very_high",
        "special_event": "ambulance_behind",
        "expected_reasoning": "override market caution — must create corridor despite chaos",
    },
]


# ── Simulated tool functions (callable by agent/pipeline) ─────────────────────

def get_nearby_places(
    location: str,
    zone_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulated ``get_nearby_places`` tool.

    The agent calls this to discover what is nearby.  The response provides
    indirect signals — no explicit zone label is returned.  The agent must
    reason from POIs, signs, and cues to decide appropriate behavior.

    Args:
        location:     descriptive location string (for logging)
        zone_context: optional key into ZONE_CONTEXT_TEMPLATES for determinism

    Returns:
        dict with nearby_places, pedestrian_density, visible_signs,
        ambient_cues, time_of_day, road_type.  Critically: no zone_type key.
    """
    ctx_key = zone_context or random.choice(list(ZONE_CONTEXT_TEMPLATES.keys()))
    template = ZONE_CONTEXT_TEMPLATES.get(ctx_key, ZONE_CONTEXT_TEMPLATES["market_zone"])

    # Collect signs and cues from each POI in this context
    all_signs: List[str] = []
    all_cues: List[str] = []
    for poi_name in template["nearby_places"]:
        poi = _POI_DATABASE.get(poi_name, {})
        all_signs.extend(poi.get("ambient_signs", []))
        all_cues.extend(poi.get("ambient_cues_pool", []))

    return {
        "nearby_places": list(template["nearby_places"]),
        "pedestrian_density": template["pedestrian_density"],
        "visible_signs": random.sample(all_signs, min(2, len(all_signs))) if all_signs else [],
        "ambient_cues": random.sample(all_cues, min(2, len(all_cues))) if all_cues else [],
        "time_of_day": random.choice(["06:30", "08:45", "12:15", "15:30", "18:00", "21:00", "23:30"]),
        "road_type": template["road_type"],
        # ── Pedestrian flow (density alone is insufficient — direction matters) ──
        # 10 pedestrians standing still = low risk; 3 moving across road = high risk
        "pedestrian_flow_direction": random.choice([
            "stationary",       # gathered but not moving into road
            "parallel_road",    # walking along road, not crossing
            "crossing_road",    # actively crossing (highest risk)
            "dispersing",       # crowd breaking up, unpredictable paths
            "gathering",        # event starting, density rising
        ]),
        # ⚠️ No "zone_type" field — agent must infer from above
    }


def get_road_context(road_type: str) -> Dict[str, Any]:
    """Simulated road metadata tool.  Returns structural facts, not behavioral prescriptions."""
    _contexts: Dict[str, Dict[str, Any]] = {
        "urban":        {"speed_limit_kph": 40, "surface": "asphalt",         "typical_use": "mixed_traffic",  "lanes": 2},
        "residential":  {"speed_limit_kph": 30, "surface": "asphalt",         "typical_use": "local_access",   "lanes": 1},
        "narrow_urban": {"speed_limit_kph": 20, "surface": "uneven_asphalt",  "typical_use": "mixed",          "lanes": 1},
        "highway":      {"speed_limit_kph": 80, "surface": "asphalt",         "typical_use": "high_speed",     "lanes": 3},
        "rural":        {"speed_limit_kph": 60, "surface": "mixed",           "typical_use": "intercity",      "lanes": 1},
    }
    return _contexts.get(road_type, _contexts["urban"])


def get_ambient_context(scenario_type: Optional[str] = None) -> Dict[str, Any]:
    """Simulated ambient context tool — time, weather, and special events."""
    return {
        "time": random.choice(["06:30", "08:45", "12:15", "15:30", "18:00", "21:00", "23:30"]),
        "weather": random.choice(["clear", "overcast", "light_rain", "heavy_rain", "foggy", "night_clear"]),
        "special_event": random.choice([None, None, "festival", "public_protest", "accident_ahead", "VIP_movement"]),
        "visibility_m": random.choice([30, 50, 80, 100, 150]),
        "traffic_density": random.choice(["very_low", "low", "moderate", "high", "gridlock"]),
    }


def build_zone_cues(zone_context_key: str) -> Dict[str, Any]:
    """Build the ``zone_cues`` block that is embedded in scenario specs.

    This is used at scenario-generation time.  Runtime tool calls should use
    ``get_nearby_places()`` instead.

    Returns a dict with indirect signals only (no zone_type label).
    """
    result = get_nearby_places("", zone_context=zone_context_key)
    return result