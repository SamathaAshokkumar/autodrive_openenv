"""Multi-agent pipeline for AutoDrive Gym — Theme 1: Multi-Agent Interactions.

Six specialised LLM sub-agents collaborate on every driving step:

  PerceptionAgent       — reads raw sensor_data → structured threat manifest
  ContextAgent          — reasons about threats + ego/env state → situation analysis
  IntentInferenceAgent  — infers hidden actor intents from observable behavioral signals
                          (crucial: never given the intent directly — must reason)
  NegotiationAgent      — decides yield / assert / wait strategy per actor interaction
  DecisionAgent         — selects optimal action given context + intent + negotiation
  OversightAgent        — validates & may veto the decision (Fleet AI / Scalable Oversight)

Covering multiple hackathon themes:
  • Theme 1 core   : multi-agent cooperation with intent modeling and negotiation
  • Theme 3.1      : world modeling — agent infers zone type from indirect signals
  • Fleet AI bonus : OversightAgent monitors DecisionAgent and can override it
  • Theme 4        : each agent learns from feedback inside the pipeline trace
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .intent_engine import enrich_sensor_objects, _infer_intent_from_signals
from .zone_api import get_nearby_places, get_road_context, get_ambient_context

logger = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

_PERCEPTION_SYSTEM = """You are the PERCEPTION agent for an autonomous vehicle driving in India.
Analyse the raw sensor readings and return ONLY valid JSON with no markdown:
{
  "threats": [
    {
      "type": "<object type>",
      "distance_m": <float>,
      "severity": "<low|medium|high|critical>",
      "predicted_trajectory": "<will_cross|moving_away|stationary|approaching>"
    }
  ],
  "scene_risk": <float 0.0-1.0>,
  "priority_hazard": "<type of the most urgent threat, or 'none'>",
  "visibility_factor": <float 0.0-1.0>,
  "threat_count": <int>
}
Rules: include only objects within 20 m that are on-road. Do not invent objects."""

_CONTEXT_SYSTEM = """You are the CONTEXT agent for an autonomous vehicle driving in India.
Given a structured threat manifest and the vehicle's current state, reason about the
situation and return ONLY valid JSON with no markdown:
{
  "situation_summary": "<2 sentences describing what is happening right now>",
  "time_to_critical_event_s": <float, estimated seconds until action is needed>,
  "recommended_strategy": "<brake_and_wait|accelerate_past|steer_around|horn_and_proceed|yield>",
  "actor_intents": {"<actor_type>": "<what the actor is likely trying to do>"},
  "risk_level": "<safe|caution|danger|critical>",
  "should_override_for_alert": <true|false>
}"""

_DECISION_SYSTEM = """You are the DECISION agent for an autonomous vehicle driving in India.
Given the situation context, select the single best driving action.
Return ONLY valid JSON with no markdown:
{
  "action": "<one of: accelerate|brake|steer_left|steer_right|horn|wait|change_lane_left|change_lane_right>",
  "value": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining why>"
}
value is intensity: 0.0=none, 1.0=maximum. Use value=0.0 for horn/wait/lane_change.
Do NOT repeat an action chosen more than twice in a row — check last_3_actions."""

_INTENT_INFERENCE_SYSTEM = """You are the INTENT INFERENCE agent for an autonomous vehicle driving in India.
You receive observable behavioral signals for each nearby road actor.
You must infer the MOST LIKELY hidden intent of each actor from these signals ONLY.
You are NEVER told the actual intent — you must reason from evidence.

Observable signals you receive per actor:
  speed_variance    (0-1: high=erratic/aggressive, low=cautious)
  gap_acceptance    (0-1: tight=aggressive/rush, large=yielding/cautious)
  lane_adherence    (0-1: low=distracted, high=disciplined)
  reaction_delay_s  (seconds: long=distracted, short=rush/aggressive)
  signal_compliance (bool: true=rule-follower, false=rule-breaker)
  honk_rate         (0-1: high=aggressive, low=calm)

Intent vocabulary: rush | cautious | distracted | aggressive | yielding

Return ONLY valid JSON with no markdown:
{
  "actor_intent_map": {
    "<actor_type_index>": {
      "inferred_intent": "<intent>",
      "confidence": <float 0.0-1.0>,
      "key_signal": "<which signal most influenced this inference>",
      "reasoning": "<one sentence>"
    }
  },
  "dominant_scene_intent": "<the most dangerous inferred intent in the scene>",
  "negotiation_complexity": "<simple|moderate|complex>"
}"""

_NEGOTIATION_SYSTEM = """You are the NEGOTIATION agent for an autonomous vehicle driving in India.
You receive the situation context AND inferred actor intents.
Your job: decide the optimal negotiation strategy for each key interaction.

Indian roads require implicit negotiation — who yields, who asserts, who waits.
Decision must account for actor intent (aggressive actors need ego to yield more often).

Negotiation vocabulary:
  yield  — ego backs off, gives right-of-way (use for aggressive/unpredictable actors)
  assert — ego proceeds confidently (use with cautious/yielding actors)
  wait   — ego holds position briefly to observe (use for distracted/unclear intent)
  horn   — ego signals presence (use when actor seems unaware)

Return ONLY valid JSON with no markdown:
{
  "negotiation_plan": [
    {
      "actor_type": "<type>",
      "inferred_intent": "<intent>",
      "ego_strategy": "<yield|assert|wait|horn>",
      "rationale": "<one sentence>"
    }
  ],
  "overall_approach": "<defensive|balanced|assertive>",
  "social_compliance_risk": "<low|medium|high>",
  "expected_outcome": "<smooth_pass|delayed_pass|collision_risk|deadlock>"
}"""

_OVERSIGHT_SYSTEM = """You are the OVERSIGHT safety agent for an autonomous vehicle fleet in India.
Your job: review any proposed driving action and either approve it or override it.
Return ONLY valid JSON with no markdown:
{
  "approved": <true|false>,
  "safety_note": "<one sentence assessment>",
  "override_action": null or {"action": "<action>", "value": <float>},
  "override_reason": null or "<why the original action was unsafe>",
  "oversight_confidence": <float 0.0-1.0>
}
Override ONLY when the action would directly cause collision, leave road, or violate
an active emergency protocol (ambulance siren, police override).
If the hazard has cleared and the vehicle is stopped, override 'wait' to 'accelerate'."""


# ── Individual agents ─────────────────────────────────────────────────────────

class PerceptionAgent:
    """Stage 1: interprets raw sensor readings into a structured threat manifest."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, sensor_data: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
        prompt = json.dumps({
            "sensor_objects": sensor_data.get("objects", [])[:8],
            "traffic_signal": environment.get("traffic_signal", "none"),
            "road_condition": environment.get("road_condition", "normal"),
            "visibility": environment.get("visibility", "clear"),
        }, indent=2)
        try:
            result = self.llm.chat_json(_PERCEPTION_SYSTEM, prompt, temperature=0.05, max_tokens=350)
            if "threats" not in result:
                raise ValueError("missing threats key")
            return result
        except Exception as exc:
            logger.debug("PerceptionAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_perception(sensor_data, environment)


class ContextAgent:
    """Stage 2: reasons about threats + ego/env state to produce a situation analysis."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(
        self,
        threat_summary: Dict[str, Any],
        ego_state: Dict[str, Any],
        active_alerts: List[str],
        scenario_stage: str,
    ) -> Dict[str, Any]:
        prompt = json.dumps({
            "threat_summary": threat_summary,
            "ego_speed_kmh": round(float(ego_state.get("speed", 0.0)), 1),
            "ego_lane": ego_state.get("lane", "center"),
            "active_alerts": active_alerts,
            "scenario_stage": scenario_stage,
        }, indent=2)
        try:
            result = self.llm.chat_json(_CONTEXT_SYSTEM, prompt, temperature=0.10, max_tokens=350)
            if "recommended_strategy" not in result:
                raise ValueError("missing recommended_strategy")
            return result
        except Exception as exc:
            logger.debug("ContextAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_context(threat_summary, ego_state, scenario_stage, active_alerts)


class DecisionAgent:
    """Stage 3: selects the optimal driving action given the situation context."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, context: Dict[str, Any], history_actions: List[str]) -> Dict[str, Any]:
        prompt = json.dumps({
            "context": context,
            "last_3_actions": history_actions[-3:] if history_actions else [],
        }, indent=2)
        try:
            result = self.llm.chat_json(_DECISION_SYSTEM, prompt, temperature=0.15, max_tokens=180)
            if "action" not in result:
                raise ValueError("missing action key")
            # Clamp value
            result["value"] = max(0.0, min(1.0, float(result.get("value", 0.0))))
            return result
        except Exception as exc:
            logger.debug("DecisionAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_decision(context)


class OversightAgent:
    """Stage 4: validates the proposed action, may veto or modify it.

    This implements the *Fleet AI / Scalable Oversight* sub-theme: an oversight
    agent watches the DecisionAgent's output and can override it when unsafe.
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(
        self,
        proposed: Dict[str, Any],
        threat_summary: Dict[str, Any],
        validation_state: Dict[str, Any],
        scenario_stage: str,
    ) -> Dict[str, Any]:
        prompt = json.dumps({
            "proposed_action": proposed,
            "scene_risk": threat_summary.get("scene_risk", 0.5),
            "priority_hazard": threat_summary.get("priority_hazard", "none"),
            "scenario_stage": scenario_stage,
            "validation_flags": {
                "collision": validation_state.get("collision", False),
                "near_miss": validation_state.get("near_miss", False),
                "offroad": validation_state.get("offroad", False),
                "stuck": validation_state.get("stuck", False),
            },
        }, indent=2)
        try:
            result = self.llm.chat_json(_OVERSIGHT_SYSTEM, prompt, temperature=0.05, max_tokens=220)
            if "approved" not in result:
                raise ValueError("missing approved")
            return result
        except Exception as exc:
            logger.debug("OversightAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_oversight(proposed, threat_summary, validation_state, scenario_stage)


class IntentInferenceAgent:
    """Stage 2.5: infers hidden actor intents from observable behavioral signals.

    The agent is NEVER given the intent directly.  It receives noisy behavioral
    signals (speed_variance, gap_acceptance, lane_adherence, etc.) and must reason
    about what each actor is likely trying to do.  This implements the
    'Theory of Mind' capability the senior reviewer identified as a game-changer.
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        # Enrich objects with observable signals (hides raw hidden_intent)
        enriched_objects = enrich_sensor_objects(sensor_data.get("objects", [])[:5])
        actor_signals = [
            {
                "index": f"{obj.get('type', 'actor')}_{i}",
                "type": obj.get("type", "unknown"),
                "distance_m": obj.get("distance", 999.0),
                "behavior_signals": obj.get("behavior_signals", {}),
            }
            for i, obj in enumerate(enriched_objects)
        ]
        prompt = json.dumps({"actors": actor_signals}, indent=2)
        try:
            result = self.llm.chat_json(_INTENT_INFERENCE_SYSTEM, prompt, temperature=0.15, max_tokens=400)
            if "actor_intent_map" not in result:
                raise ValueError("missing actor_intent_map")
            return result
        except Exception as exc:
            logger.debug("IntentInferenceAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_intent_inference(enriched_objects)


class NegotiationAgent:
    """Stage 3.5: decides yield / assert / wait strategy per actor interaction.

    Indian roads require implicit social negotiation at every merge, crossing,
    and lane transition. This agent takes inferred actor intents and produces a
    per-actor negotiation plan, enabling the DecisionAgent to pick a socially
    intelligent action rather than just a mechanically safe one.
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def run(
        self,
        context: Dict[str, Any],
        intent_inference: Dict[str, Any],
        ego_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = json.dumps({
            "situation_summary": context.get("situation_summary", ""),
            "risk_level": context.get("risk_level", "caution"),
            "recommended_strategy": context.get("recommended_strategy", "brake_and_wait"),
            "actor_intent_map": intent_inference.get("actor_intent_map", {}),
            "dominant_scene_intent": intent_inference.get("dominant_scene_intent", "unknown"),
            "ego_speed_kmh": round(float(ego_state.get("speed", 0.0)), 1),
        }, indent=2)
        try:
            result = self.llm.chat_json(_NEGOTIATION_SYSTEM, prompt, temperature=0.15, max_tokens=400)
            if "negotiation_plan" not in result:
                raise ValueError("missing negotiation_plan")
            return result
        except Exception as exc:
            logger.debug("NegotiationAgent LLM failed (%s); using heuristic", exc)
            return _heuristic_negotiation(context, intent_inference)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class MultiAgentPipeline:
    """Orchestrates all six agents to produce a final driving action + trace.

    Usage::

        pipeline = MultiAgentPipeline(llm_client)
        action, trace = pipeline.run(observation_dict, history_list)
        # action  → {"action": "brake", "value": 0.8}
        # trace   → full reasoning chain for each of the 6 agents
    """

    def __init__(self, llm) -> None:
        self.perception  = PerceptionAgent(llm)
        self.context     = ContextAgent(llm)
        self.intent      = IntentInferenceAgent(llm)
        self.negotiation = NegotiationAgent(llm)
        self.decision    = DecisionAgent(llm)
        self.oversight   = OversightAgent(llm)

    def run(
        self,
        observation: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the full 6-stage pipeline.

        Returns
        -------
        (final_action, pipeline_trace)
            final_action : ``{"action": str, "value": float}``
            pipeline_trace : full per-stage reasoning for logging / demo
        """
        history = history or []
        sensor_data     = observation.get("sensor_data", {}) or {}
        ego_state       = observation.get("ego_state", {}) or {}
        environment     = observation.get("environment", {}) or {}
        active_alerts   = observation.get("active_alerts", []) or []
        scenario_stage  = str(observation.get("scenario_stage", "approaching") or "approaching")
        validation_state = observation.get("validation", {}) or {}
        history_actions = [h.get("action", "wait") for h in history[-3:]]
        zone_cues       = observation.get("zone_cues", {}) or {}

        # ── Stage 1: Perception ───────────────────────────────────────────────
        threats = self.perception.run(sensor_data, environment)

        # ── Stage 2: Context ──────────────────────────────────────────────────
        context = self.context.run(threats, ego_state, active_alerts, scenario_stage)

        # ── Stage 2.5: Intent Inference ───────────────────────────────────────
        # Agent infers actor intents from behavioral signals (never from labels)
        intent_inference = self.intent.run(sensor_data)

        # ── Stage 3: Negotiation ──────────────────────────────────────────────
        # Agent decides yield/assert/wait strategy based on inferred intents
        negotiation = self.negotiation.run(context, intent_inference, ego_state)

        # ── Enrich context with world-model inference from zone cues ──────────
        # If the scenario has zone_cues, we surface them to the DecisionAgent
        # so it can reason about appropriate zone behavior without being told
        # the zone type directly.
        if zone_cues:
            context["zone_inference_input"] = {
                "nearby_places": zone_cues.get("nearby_places", []),
                "visible_signs": zone_cues.get("visible_signs", []),
                "ambient_cues":  zone_cues.get("ambient_cues", []),
                "pedestrian_density": zone_cues.get("pedestrian_density", "unknown"),
            }

        # ── Stage 3.5: Decision (now informed by intent + negotiation) ────────
        negotiation_hint = negotiation.get("overall_approach", "balanced")
        context_enriched = {
            **context,
            "negotiation_approach": negotiation_hint,
            "negotiation_plan_summary": [
                f"{p['actor_type']}: {p['ego_strategy']} ({p['inferred_intent']})"
                for p in negotiation.get("negotiation_plan", [])
            ],
        }
        proposed = self.decision.run(context_enriched, history_actions)

        # ── Stage 4: Oversight ────────────────────────────────────────────────
        oversight = self.oversight.run(proposed, threats, validation_state, scenario_stage)

        # Resolve final action
        if not oversight.get("approved", True) and oversight.get("override_action"):
            final_action = dict(oversight["override_action"])
            was_overridden = True
        else:
            final_action = {
                "action": str(proposed.get("action", "wait")),
                "value": float(proposed.get("value", 0.0)),
            }
            was_overridden = False

        # Clamp value
        final_action["value"] = max(0.0, min(1.0, final_action["value"]))

        pipeline_trace = {
            "perception":         threats,
            "context":            context,
            "intent_inference":   intent_inference,
            "negotiation":        negotiation,
            "proposed_decision":  proposed,
            "oversight":          oversight,
            "was_overridden":     was_overridden,
            "final_action":       final_action,
        }
        return final_action, pipeline_trace


# ── Heuristic fallbacks (no LLM required) ────────────────────────────────────

def _heuristic_intent_inference(enriched_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic intent inference when LLM is unavailable."""
    actor_intent_map: Dict[str, Any] = {}
    for i, obj in enumerate(enriched_objects[:4]):
        signals = obj.get("behavior_signals", {})
        inferred = _infer_intent_from_signals(signals)
        actor_intent_map[f"{obj.get('type', 'actor')}_{i}"] = {
            "inferred_intent": inferred,
            "confidence": 0.55,
            "key_signal": "honk_rate" if signals.get("honk_rate", 0) > 0.5 else "gap_acceptance",
            "reasoning": f"Heuristic: signals suggest {inferred} behavior",
        }
    dominant = "aggressive"
    if actor_intent_map:
        intents = [v["inferred_intent"] for v in actor_intent_map.values()]
        for prio in ("aggressive", "distracted", "rush", "cautious", "yielding"):
            if prio in intents:
                dominant = prio
                break
    return {
        "actor_intent_map": actor_intent_map,
        "dominant_scene_intent": dominant,
        "negotiation_complexity": "moderate",
    }


def _heuristic_negotiation(
    context: Dict[str, Any],
    intent_inference: Dict[str, Any],
) -> Dict[str, Any]:
    """Heuristic negotiation plan when LLM is unavailable."""
    dominant = intent_inference.get("dominant_scene_intent", "cautious")
    risk = context.get("risk_level", "caution")

    if dominant in ("aggressive", "rush") or risk == "critical":
        approach = "defensive"
    elif dominant == "yielding" and risk == "safe":
        approach = "assertive"
    else:
        approach = "balanced"

    # Build one plan entry per inferred actor
    plan = []
    for key, info in intent_inference.get("actor_intent_map", {}).items():
        intent = info.get("inferred_intent", "cautious")
        strategy = "yield" if intent in ("aggressive", "rush", "distracted") else "assert" if intent == "yielding" else "wait"
        plan.append({
            "actor_type": key,
            "inferred_intent": intent,
            "ego_strategy": strategy,
            "rationale": f"Heuristic: {intent} actor → {strategy}",
        })

    return {
        "negotiation_plan": plan,
        "overall_approach": approach,
        "social_compliance_risk": "high" if dominant == "aggressive" else "medium",
        "expected_outcome": "smooth_pass" if approach == "assertive" else "delayed_pass",
    }


def _heuristic_perception(sensor_data: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
    objects = sensor_data.get("objects", [])
    threats = []
    for obj in objects[:6]:
        d = float(obj.get("distance", 999.0))
        if not obj.get("on_road", True) or d > 25.0:
            continue
        sev = "critical" if d < 5.0 else "high" if d < 9.0 else "medium" if d < 16.0 else "low"
        beh = str(obj.get("behavior", "static"))
        traj = (
            "will_cross" if beh in ("sudden_cross",)
            else "approaching" if beh in ("cut_in", "blind_spot_merge", "zig_zag")
            else "stationary"
        )
        threats.append({
            "type": obj.get("type", "unknown"),
            "distance_m": round(d, 1),
            "severity": sev,
            "predicted_trajectory": traj,
        })
    scene_risk = min(1.0, sum(0.4 if t["severity"] == "critical" else 0.25 if t["severity"] == "high" else 0.1 for t in threats))
    priority = threats[0]["type"] if threats else "none"
    vis = environment.get("visibility", "clear")
    return {
        "threats": threats,
        "scene_risk": round(scene_risk, 3),
        "priority_hazard": priority,
        "visibility_factor": 1.0 if vis == "clear" else 0.5 if vis == "low_visibility" else 0.75,
        "threat_count": len(threats),
    }


def _heuristic_context(
    threats: Dict[str, Any],
    ego: Dict[str, Any],
    stage: str,
    alerts: List[str],
) -> Dict[str, Any]:
    risk = threats.get("scene_risk", 0.0)
    priority = threats.get("priority_hazard", "none")
    has_ambulance = any("ambulance" in a.lower() for a in alerts)
    has_police = any("police" in a.lower() for a in alerts)

    if stage in ("clearing", "cleared"):
        return {
            "situation_summary": "Hazard has cleared. The road ahead is opening up — resume forward motion.",
            "time_to_critical_event_s": 99.0,
            "recommended_strategy": "accelerate_past",
            "actor_intents": {},
            "risk_level": "safe",
            "should_override_for_alert": False,
        }
    if has_ambulance:
        return {
            "situation_summary": "Ambulance approaching from behind — must clear the lane immediately.",
            "time_to_critical_event_s": 2.0,
            "recommended_strategy": "steer_around",
            "actor_intents": {"ambulance": "requesting corridor"},
            "risk_level": "critical",
            "should_override_for_alert": True,
        }
    if has_police:
        return {
            "situation_summary": "Traffic police is directing flow — yield and wait for signal.",
            "time_to_critical_event_s": 3.0,
            "recommended_strategy": "yield",
            "actor_intents": {"police": "directing traffic"},
            "risk_level": "caution",
            "should_override_for_alert": True,
        }
    if risk >= 0.65:
        return {
            "situation_summary": f"High-risk scene — {priority} on path. Immediate braking required.",
            "time_to_critical_event_s": 1.5,
            "recommended_strategy": "brake_and_wait",
            "actor_intents": {priority: "blocking or crossing path"},
            "risk_level": "critical",
            "should_override_for_alert": False,
        }
    if risk >= 0.3:
        return {
            "situation_summary": f"Moderate risk — {priority} nearby. Slow down and observe.",
            "time_to_critical_event_s": 3.5,
            "recommended_strategy": "brake_and_wait",
            "actor_intents": {priority: "uncertain intent"},
            "risk_level": "caution",
            "should_override_for_alert": False,
        }
    return {
        "situation_summary": "Scene appears clear. Proceed with normal caution.",
        "time_to_critical_event_s": 10.0,
        "recommended_strategy": "accelerate_past",
        "actor_intents": {},
        "risk_level": "safe",
        "should_override_for_alert": False,
    }


def _heuristic_decision(context: Dict[str, Any]) -> Dict[str, Any]:
    strategy = context.get("recommended_strategy", "brake_and_wait")
    _MAP = {
        "brake_and_wait": ("brake", 0.8),
        "accelerate_past": ("accelerate", 0.6),
        "steer_around": ("steer_left", 0.5),
        "horn_and_proceed": ("horn", 0.0),
        "yield": ("wait", 0.0),
    }
    action, value = _MAP.get(strategy, ("wait", 0.0))
    return {"action": action, "value": value, "reasoning": f"Heuristic fallback: {strategy}"}


def _heuristic_oversight(
    proposed: Dict[str, Any],
    threats: Dict[str, Any],
    validation: Dict[str, Any],
    stage: str,
) -> Dict[str, Any]:
    risk = threats.get("scene_risk", 0.0)
    action = proposed.get("action", "wait")

    # Override if about to accelerate into a critical scene
    if action == "accelerate" and risk > 0.8:
        return {
            "approved": False,
            "safety_note": "Overriding: accelerating into a high-risk scene is unsafe.",
            "override_action": {"action": "brake", "value": 0.85},
            "override_reason": "scene_risk > 0.8 while accelerating",
            "oversight_confidence": 0.95,
        }
    # Override wait when hazard has cleared
    if action == "wait" and stage in ("clearing", "cleared"):
        return {
            "approved": False,
            "safety_note": "Hazard cleared — overriding wait to resume forward motion.",
            "override_action": {"action": "accelerate", "value": 0.5},
            "override_reason": "stuck wait after hazard cleared",
            "oversight_confidence": 0.90,
        }
    return {
        "approved": True,
        "safety_note": "Action is safe — approved.",
        "override_action": None,
        "override_reason": None,
        "oversight_confidence": 0.85,
    }