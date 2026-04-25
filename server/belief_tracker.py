"""Belief Tracker — Theory of Mind for AutoDrive Gym.

The agent maintains a probabilistic belief state about each actor's intent.
This is updated Bayesian-style every step based on observed behavioral signals.

Why this matters (senior reviewer's key insight):
  - "Theory of Mind" means the agent models other agents' mental states.
  - A good driver doesn't just react; they PREDICT what others will do.
  - Reward for correct predictions; penalty for surprises.

Design:
  - Each actor seen in an episode gets a belief distribution over 5 intents.
  - Beliefs updated using likelihood P(signal | intent) × prior.
  - Agent accumulates confidence over multiple timesteps.
  - BeliefTracker surfaces the dominant intent + confidence for pipeline context.

Exposed to the pipeline as:
  observation["belief_state"] = {
      "auto_0": {"likely_intent": "aggressive", "confidence": 0.84, "history": [...]},
      "pedestrian_1": {"likely_intent": "distracted", "confidence": 0.71, ...},
  }
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── P(signal | intent) likelihood tables ─────────────────────────────────────
# For each intent × signal, what's the probability of observing "high" signal?
# These are hand-tuned based on the intent profiles in intent_engine.py

_LIKELIHOOD: Dict[str, Dict[str, float]] = {
    # P(high signal value | intent) — e.g. P(high honk_rate | aggressive) = 0.90
    "rush": {
        "honk_rate_high":        0.60, "honk_rate_low":        0.40,
        "gap_tight":             0.75, "gap_large":             0.25,
        "lane_poor":             0.45, "lane_good":             0.55,
        "reaction_slow":         0.15, "reaction_fast":         0.85,
        "signal_violated":       0.65, "signal_obeyed":         0.35,
        "speed_erratic":         0.70, "speed_stable":          0.30,
    },
    "cautious": {
        "honk_rate_high":        0.05, "honk_rate_low":        0.95,
        "gap_tight":             0.10, "gap_large":             0.90,
        "lane_poor":             0.10, "lane_good":             0.90,
        "reaction_slow":         0.80, "reaction_fast":         0.20,
        "signal_violated":       0.05, "signal_obeyed":         0.95,
        "speed_erratic":         0.15, "speed_stable":          0.85,
    },
    "distracted": {
        "honk_rate_high":        0.10, "honk_rate_low":        0.90,
        "gap_tight":             0.50, "gap_large":             0.50,
        "lane_poor":             0.85, "lane_good":             0.15,
        "reaction_slow":         0.90, "reaction_fast":         0.10,
        "signal_violated":       0.55, "signal_obeyed":         0.45,
        "speed_erratic":         0.85, "speed_stable":          0.15,
    },
    "aggressive": {
        "honk_rate_high":        0.90, "honk_rate_low":        0.10,
        "gap_tight":             0.90, "gap_large":             0.10,
        "lane_poor":             0.60, "lane_good":             0.40,
        "reaction_slow":         0.10, "reaction_fast":         0.90,
        "signal_violated":       0.80, "signal_obeyed":         0.20,
        "speed_erratic":         0.60, "speed_stable":          0.40,
    },
    "yielding": {
        "honk_rate_high":        0.02, "honk_rate_low":        0.98,
        "gap_tight":             0.15, "gap_large":             0.85,
        "lane_poor":             0.15, "lane_good":             0.85,
        "reaction_slow":         0.40, "reaction_fast":         0.60,
        "signal_violated":       0.05, "signal_obeyed":         0.95,
        "speed_erratic":         0.20, "speed_stable":          0.80,
    },
}

_INTENTS = list(_LIKELIHOOD.keys())


def _uniform_prior() -> Dict[str, float]:
    return {intent: 1.0 / len(_INTENTS) for intent in _INTENTS}


def _signals_to_observations(signals: Dict[str, Any]) -> List[str]:
    """Convert a behavioral signals dict into a list of observation keys."""
    obs = []
    obs.append("honk_rate_high"   if signals.get("honk_rate", 0.2) > 0.4    else "honk_rate_low")
    obs.append("gap_tight"        if signals.get("gap_acceptance", 0.5) < 0.35 else "gap_large")
    obs.append("lane_poor"        if signals.get("lane_adherence", 0.7) < 0.45 else "lane_good")
    obs.append("reaction_slow"    if signals.get("reaction_delay_s", 0.8) > 1.0  else "reaction_fast")
    obs.append("signal_violated"  if not signals.get("signal_compliance", True)  else "signal_obeyed")
    obs.append("speed_erratic"    if signals.get("speed_variance", 0.3) > 0.55   else "speed_stable")
    return obs


def _bayesian_update(
    prior: Dict[str, float],
    observations: List[str],
) -> Dict[str, float]:
    """Multiply each intent's prior by the product of likelihoods, then normalise."""
    posterior = {}
    for intent in _INTENTS:
        log_prob = math.log(prior[intent] + 1e-12)
        for obs_key in observations:
            lik = _LIKELIHOOD[intent].get(obs_key, 0.5)
            log_prob += math.log(lik + 1e-9)
        posterior[intent] = math.exp(log_prob)

    total = sum(posterior.values()) + 1e-12
    return {intent: round(v / total, 4) for intent, v in posterior.items()}


# ── Per-actor belief ──────────────────────────────────────────────────────────

@dataclass
class ActorBelief:
    """Running belief about one actor's intent probability distribution."""
    actor_id:  str
    actor_type: str
    beliefs:   Dict[str, float] = field(default_factory=_uniform_prior)
    history:   List[Dict[str, Any]] = field(default_factory=list)
    steps_observed: int = 0

    @property
    def dominant_intent(self) -> str:
        return max(self.beliefs, key=lambda k: self.beliefs[k])

    @property
    def confidence(self) -> float:
        """P(best intent) — how certain is the tracker?"""
        vals = list(self.beliefs.values())
        return round(max(vals), 4) if vals else 0.2

    @property
    def entropy(self) -> float:
        """Shannon entropy of belief distribution — low = very certain."""
        return round(-sum(p * math.log(p + 1e-9) for p in self.beliefs.values()), 3)

    def update(self, signals: Dict[str, Any]) -> None:
        obs_keys = _signals_to_observations(signals)
        self.beliefs = _bayesian_update(self.beliefs, obs_keys)
        self.steps_observed += 1
        self.history.append({
            "step":         self.steps_observed,
            "signals":      {k: round(float(v), 2) if isinstance(v, float) else v
                             for k, v in signals.items()},
            "dominant":     self.dominant_intent,
            "confidence":   self.confidence,
            "beliefs":      dict(self.beliefs),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id":         self.actor_id,
            "actor_type":       self.actor_type,
            "likely_intent":    self.dominant_intent,
            "confidence":       self.confidence,
            "entropy":          self.entropy,   # low = agent is sure
            "beliefs":          dict(self.beliefs),
            "steps_observed":   self.steps_observed,
            "history":          self.history[-3:],  # last 3 updates for trace
        }


# ── Belief Tracker ─────────────────────────────────────────────────────────────

class BeliefTracker:
    """Session-level Theory-of-Mind tracker with behavioral memory.

    Maintains one ActorBelief per actor seen in the current episode.
    Beliefs accumulate over multiple steps — confidence grows as more
    evidence arrives, which is exactly what a good driver does.

    Behavioral memory: tracks patterns across steps for each actor
    so the agent can learn "this auto has been aggressive twice already" —
    mirroring how humans mentally flag unpredictable road users.
    """

    def __init__(self) -> None:
        self._actors: Dict[str, ActorBelief] = {}
        self._step: int = 0
        # Behavioral memory: actor_id → list of (step, dominant_intent, trajectory)
        self._behavioral_memory: Dict[str, List[Dict[str, Any]]] = {}

    def reset(self) -> None:
        """Clear all beliefs at episode start."""
        self._actors = {}
        self._step   = 0
        self._behavioral_memory = {}

    def update_from_sensor_objects(
        self,
        sensor_objects: List[Dict[str, Any]],
    ) -> None:
        """Update beliefs for all actors visible in this step's sensor data."""
        self._step += 1
        for i, obj in enumerate(sensor_objects[:6]):
            actor_type = str(obj.get("type", "unknown"))
            actor_id   = f"{actor_type}_{i}"

            if actor_id not in self._actors:
                self._actors[actor_id] = ActorBelief(
                    actor_id=actor_id,
                    actor_type=actor_type,
                )
            if actor_id not in self._behavioral_memory:
                self._behavioral_memory[actor_id] = []

            signals = obj.get("behavior_signals", {})
            if signals:
                self._actors[actor_id].update(signals)

            # Record behavioral memory snapshot (trajectory + intent + confidence)
            self._behavioral_memory[actor_id].append({
                "step":       self._step,
                "trajectory": obj.get("trajectory", "unknown"),
                "intent":     self._actors[actor_id].dominant_intent,
                "confidence": self._actors[actor_id].confidence,
            })
            # Keep only last 8 steps per actor (avoid unbounded growth)
            if len(self._behavioral_memory[actor_id]) > 8:
                self._behavioral_memory[actor_id].pop(0)

    def get_behavioral_pattern(self, actor_id: str) -> str:
        """Summarise observed behavioral pattern for an actor as a short string.

        E.g. "aggressive×3_consecutive" or "distracted→rush transition" or "erratic_unpredictable×2"
        This surfaces to the pipeline as context so the LLM can say "this auto has been
        aggressive twice already — treat it as a high-risk actor."
        """
        mem = self._behavioral_memory.get(actor_id, [])
        if not mem:
            return "no_history"

        intents = [m["intent"] for m in mem]
        trajectories = [m["trajectory"] for m in mem]

        # Count consecutive repeats of the last intent
        last = intents[-1]
        consecutive = sum(1 for x in reversed(intents) if x == last)

        # Detect transitions (e.g. cautious → aggressive)
        if len(intents) >= 3 and intents[-3] != intents[-1]:
            pattern = f"{intents[-3]}→{intents[-1]}"
        elif consecutive >= 3:
            pattern = f"{last}×{consecutive}_consistent"
        else:
            pattern = last

        # Flag erratic trajectory
        erratic_count = sum(1 for t in trajectories if t == "erratic_unpredictable")
        if erratic_count >= 2:
            pattern += "_erratic"

        return pattern

    def get_belief_state(self) -> Dict[str, Any]:
        """Return full belief state dict for embedding in observation."""
        result = {}
        for actor_id, belief in self._actors.items():
            entry = belief.to_dict()
            entry["behavioral_pattern"] = self.get_behavioral_pattern(actor_id)
            result[actor_id] = entry
        return result

    def get_dominant_threat_intent(self) -> str:
        """Return the most dangerous inferred intent across all actors."""
        if not self._actors:
            return "unknown"
        _priority = ["aggressive", "distracted", "rush", "cautious", "yielding"]
        intents = {b.dominant_intent for b in self._actors.values()}
        for p in _priority:
            if p in intents:
                return p
        return "unknown"

    def get_high_confidence_beliefs(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Return actors where the tracker is sufficiently confident (for debug/demo)."""
        return [
            b.to_dict()
            for b in self._actors.values()
            if b.confidence >= threshold
        ]

    def belief_summary_line(self) -> str:
        """One-line summary for training console output."""
        if not self._actors:
            return "beliefs: none"
        parts = []
        for actor_id, belief in list(self._actors.items())[:3]:
            pattern = self.get_behavioral_pattern(actor_id)
            parts.append(f"{actor_id}→{belief.dominant_intent}({belief.confidence:.0%},{pattern})")
        return "beliefs: " + " | ".join(parts)

    @property
    def step(self) -> int:
        return self._step
