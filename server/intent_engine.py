"""Intent Engine — hidden actor-intent modeling for Social Driving Intelligence.

Each road actor carries a hidden_intent that is NEVER directly exposed in
observations.  The ego agent must infer intent by observing *behavioral signals*.

Intent types:
  rush        — speeding, tight gap acceptance, weaving, signal violations
  cautious    — slow, large margins, hesitant at junctions, rule-follower
  distracted  — variable speed, lane drift, delayed reactions
  aggressive  — tailgates, cuts in, honks constantly, ignores right-of-way
  yielding    — cooperative, signals before lane change, gives right-of-way

Observable signals (agent sees these, NOT the intent string):
  speed_variance    high → distracted / aggressive  |  low → cautious
  gap_acceptance    tight → aggressive / rush        |  large → cautious / yielding
  lane_adherence    poor → distracted               |  good → cautious
  reaction_delay_s  long → distracted               |  short → rush / aggressive
  signal_compliance False → aggressive / rush        |  True → cautious / yielding
  honk_rate         high → aggressive               |  near-zero → yielding
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


# ── Intent profiles ────────────────────────────────────────────────────────────

_INTENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "rush": {
        "speed_variance": 0.75,
        "gap_acceptance": 0.25,      # tight gaps
        "lane_adherence": 0.55,
        "reaction_delay_s": 0.4,
        "signal_compliance": False,
        "honk_rate": 0.60,
        "description": "Driver in a hurry — weaves, cuts gaps, may run signals",
    },
    "cautious": {
        "speed_variance": 0.15,
        "gap_acceptance": 0.90,      # keeps big gaps
        "lane_adherence": 0.90,
        "reaction_delay_s": 1.20,
        "signal_compliance": True,
        "honk_rate": 0.05,
        "description": "Slow, careful driver — predictable, respects rules",
    },
    "distracted": {
        "speed_variance": 0.85,      # erratic speed
        "gap_acceptance": 0.50,
        "lane_adherence": 0.30,      # drifts across lanes
        "reaction_delay_s": 2.00,
        "signal_compliance": False,
        "honk_rate": 0.10,
        "description": "Mobile-phone / inattentive driver — erratic, drifts",
    },
    "aggressive": {
        "speed_variance": 0.60,
        "gap_acceptance": 0.15,      # very tight gaps
        "lane_adherence": 0.40,
        "reaction_delay_s": 0.20,
        "signal_compliance": False,
        "honk_rate": 0.90,
        "description": "Hostile driver — tailgates, cuts in, ignores signals",
    },
    "yielding": {
        "speed_variance": 0.20,
        "gap_acceptance": 0.85,
        "lane_adherence": 0.85,
        "reaction_delay_s": 0.80,
        "signal_compliance": True,
        "honk_rate": 0.02,
        "description": "Cooperative driver — signals intent, gives right-of-way",
    },
}

# Per actor-type: probability weights over intents
_TYPE_INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "auto":       {"rush": 0.30, "aggressive": 0.35, "distracted": 0.15, "cautious": 0.10, "yielding": 0.10},
    "bike":       {"rush": 0.25, "distracted": 0.30, "aggressive": 0.20, "cautious": 0.15, "yielding": 0.10},
    "car":        {"rush": 0.20, "cautious": 0.25,   "distracted": 0.25, "aggressive": 0.20, "yielding": 0.10},
    "truck":      {"cautious": 0.30, "rush": 0.20,   "distracted": 0.20, "aggressive": 0.20, "yielding": 0.10},
    "pedestrian": {"cautious": 0.30, "distracted": 0.35, "rush": 0.20, "aggressive": 0.05, "yielding": 0.10},
    "default":    {"rush": 0.20, "cautious": 0.20, "distracted": 0.20, "aggressive": 0.20, "yielding": 0.20},
}

# Negotiation outcomes given (ego_action, actor_intent) pairs
# Key: (ego_action_category, actor_intent) → (outcome, social_score_delta)
NEGOTIATION_OUTCOMES: Dict[tuple, tuple] = {
    ("yield", "aggressive"): ("success", +0.15),   # ego yields to aggressor — social win
    ("yield", "cautious"):   ("success", +0.10),   # mutual courtesy
    ("yield", "rush"):       ("success", +0.08),   # sensible yield to rushing actor
    ("yield", "yielding"):   ("deadlock", -0.05),  # both yield → slight delay
    ("yield", "distracted"): ("success", +0.12),   # safe choice against unpredictable actor
    ("assert", "aggressive"): ("risky", -0.20),    # confrontation with aggressor
    ("assert", "cautious"):  ("success", +0.05),   # asserting against cautious is fine
    ("assert", "rush"):      ("risky", -0.15),     # racing a rusher
    ("assert", "yielding"):  ("success", +0.08),   # smooth assert to yielding actor
    ("assert", "distracted"): ("risky", -0.10),    # asserting against unpredictable
    ("wait", "aggressive"):  ("success", +0.10),   # strategic wait vs aggressor
    ("wait", "cautious"):    ("deadlock", -0.08),  # both cautious → minor deadlock
    ("wait", "rush"):        ("success", +0.12),   # let rusher pass — wise
    ("wait", "yielding"):    ("success", +0.05),   # cooperative sequence
    ("wait", "distracted"):  ("success", +0.15),   # wait for distracted to clarify
}


# ── Public API ─────────────────────────────────────────────────────────────────

def assign_intents(actors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign hidden intents to each actor (stored in ``actor["hidden_intent"]``).

    The intent key is set on the actor dict in-place.  It must NEVER be copied
    into the agent's observation — only behavioral signals are exposed.
    """
    for actor in actors:
        weights = _TYPE_INTENT_WEIGHTS.get(
            actor.get("type", ""), _TYPE_INTENT_WEIGHTS["default"]
        )
        intent = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]
        actor["hidden_intent"] = intent
    return actors


def get_observable_signals(actor: Dict[str, Any]) -> Dict[str, Any]:
    """Return observable behavioral signals derived from the actor's hidden intent.

    Gaussian noise is added to every continuous signal so the agent must reason
    probabilistically rather than deterministically reversing the intent.
    """
    intent = actor.get("hidden_intent", "cautious")
    profile = _INTENT_PROFILES.get(intent, _INTENT_PROFILES["cautious"])

    def _noise(v: float, sigma: float = 0.08) -> float:
        return max(0.0, min(1.0, v + random.gauss(0, sigma)))

    return {
        "speed_variance":    round(_noise(profile["speed_variance"]), 2),
        "gap_acceptance":    round(_noise(profile["gap_acceptance"]), 2),
        "lane_adherence":    round(_noise(profile["lane_adherence"]), 2),
        "reaction_delay_s":  round(max(0.1, profile["reaction_delay_s"] + random.gauss(0, 0.15)), 2),
        "signal_compliance": profile["signal_compliance"],
        "honk_rate":         round(_noise(profile["honk_rate"], 0.05), 2),
    }


def enrich_sensor_objects(sensor_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach behavioral_signals to each sensor object; strip hidden_intent.

    Call this just before building an observation so the agent never sees the
    raw intent string but always sees the noisy observable signals.
    """
    result = []
    for obj in sensor_objects:
        enriched = dict(obj)
        enriched["behavior_signals"] = get_observable_signals(obj)
        enriched.pop("hidden_intent", None)   # never expose intent to agent
        result.append(enriched)
    return result


def score_negotiation(
    ego_action: str,
    actor: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a negotiation interaction between ego and one actor.

    Maps the ego action to a category (yield / assert / wait), looks up the
    outcome table, and returns an outcome dict.

    Args:
        ego_action: raw driving action string (brake, wait, accelerate, etc.)
        actor: actor dict including hidden_intent

    Returns:
        {"outcome": str, "social_delta": float, "inferred_likely_intent": str}
    """
    action_category_map = {
        "brake": "yield",
        "wait": "wait",
        "steer_left": "yield",
        "steer_right": "yield",
        "change_lane_left": "yield",
        "change_lane_right": "yield",
        "horn": "assert",
        "accelerate": "assert",
    }
    category = action_category_map.get(ego_action, "wait")
    intent = actor.get("hidden_intent", "cautious")
    outcome, delta = NEGOTIATION_OUTCOMES.get((category, intent), ("neutral", 0.0))

    # Derive the most likely inferred intent from observable signals (for pipeline)
    signals = get_observable_signals(actor)
    inferred = _infer_intent_from_signals(signals)

    return {
        "outcome": outcome,
        "social_delta": round(delta, 3),
        "ego_action_category": category,
        "inferred_likely_intent": inferred,
        "true_intent": intent,   # kept for reward computation only, never shown to agent
    }


def _infer_intent_from_signals(signals: Dict[str, Any]) -> str:
    """Heuristic intent inference from observable signals (what agent would guess)."""
    honk = signals.get("honk_rate", 0.2)
    gap = signals.get("gap_acceptance", 0.5)
    lane = signals.get("lane_adherence", 0.7)
    speed_var = signals.get("speed_variance", 0.3)
    delay = signals.get("reaction_delay_s", 0.8)
    compliance = signals.get("signal_compliance", True)

    if honk > 0.6 and gap < 0.3:
        return "aggressive"
    if speed_var > 0.7 and lane < 0.4:
        return "distracted"
    if gap > 0.75 and compliance:
        return "cautious" if delay > 1.0 else "yielding"
    if speed_var > 0.5 and not compliance:
        return "rush"
    return "cautious"
