"""Standalone grader for the OpenEnv validator.

This file has ZERO external dependencies — only Python builtins.
The validator imports this as: autodrive_env.server.grader:HeuristicGrader

HeuristicGrader() can be instantiated with no arguments.
Its score is ALWAYS strictly between 0.02 and 0.98 (never 0.0 or 1.0).

Reward dimensions (senior reviewer recommendation):
  safety            (40%) — collision avoidance, safe distance, signal compliance
  efficiency        (20%) — forward progress, no unnecessary stops
  social_compliance (20%) — zone-appropriate behavior inferred from context signals
  smoothness        (10%) — no harsh braking, no oscillation / jerk
  negotiation       (10%) — correct yield/assert/wait per actor intent
"""

from __future__ import annotations

from typing import Any, Dict, List

_LO = 0.02
_HI = 0.98


def _clamp(v: float) -> float:
    return max(_LO, min(_HI, float(v)))


class HeuristicGrader:
    """Callable grader for the OpenEnv / HF validator.

    Accepted call signatures (validator may use either):
        grader(observation, action, result_state, scenario, history)
        grader(episode_dict)   # single dict with all fields
        grader()               # returns a neutral score

    Returns:
        {"score": float, "safety": float, "efficiency": float,
         "compliance": float, "feedback": str}
        where every float is strictly in (0.02, 0.98).
    """

    def __init__(self, persona: str = "principal"):
        self.persona = persona  # kept for API compatibility; always strict

    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        # Unpack args flexibly so the grader works however the validator calls it
        obs: Dict[str, Any] = {}
        act: Dict[str, Any] = {}
        rs: Dict[str, Any] = {}
        sc: Dict[str, Any] = {}
        hist: List[Any] = []

        if len(args) == 0 and not kwargs:
            # Called with no arguments → return neutral passing score
            pass
        elif len(args) == 1 and isinstance(args[0], dict):
            # Called with a single episode / trajectory dict
            ep = args[0]
            obs = ep.get("observation", {}) or {}
            act = ep.get("action", {}) or {}
            rs  = ep.get("result_state", ep.get("result", {})) or {}
            sc  = ep.get("scenario", {}) or {}
            hist = ep.get("history", []) or []
            if not isinstance(obs, dict):
                obs = {}
            if not isinstance(act, dict):
                act = {}
        elif len(args) >= 3:
            # Called as grader(obs, action, result_state[, scenario, history])
            obs  = args[0] if isinstance(args[0], dict) else {}
            act  = args[1] if isinstance(args[1], dict) else {}
            rs   = args[2] if isinstance(args[2], dict) else {}
            sc   = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
            hist = args[4] if len(args) > 4 and isinstance(args[4], list) else []
        elif kwargs:
            obs  = kwargs.get("observation", {}) or {}
            act  = kwargs.get("action", {}) or {}
            rs   = kwargs.get("result_state", {}) or {}
            sc   = kwargs.get("scenario", {}) or {}
            hist = kwargs.get("history", []) or []

        return self._grade(obs, act, rs, sc, hist)

    # ------------------------------------------------------------------
    def _grade(
        self,
        obs: Dict[str, Any],
        act: Dict[str, Any],
        rs: Dict[str, Any],
        sc: Dict[str, Any],
        hist: List[Any],
    ) -> Dict[str, Any]:
        action    = act.get("action", "wait") if isinstance(act, dict) else "wait"
        hd        = float(obs.get("hazard_distance", 999.0) or 999.0)
        stage     = str(obs.get("scenario_stage", "approaching"))
        expected  = sc.get("expected_behavior", []) if isinstance(sc, dict) else []
        hist_acts = [h.get("action") for h in hist if isinstance(h, dict)] if hist else []

        # Pull pipeline trace for intent/negotiation scoring
        pipeline_trace     = obs.get("pipeline_trace", {}) or {}
        negotiation_result = pipeline_trace.get("negotiation", {}) or {}
        intent_inference   = pipeline_trace.get("intent_inference", {}) or {}
        zone_cues          = obs.get("zone_cues", {}) or {}

        # ── 1. Safety (raw 0–1, weight 40%) ──────────────────────────────────
        if rs.get("collision"):
            safety = 0.04
        elif rs.get("near_miss"):
            safety = 0.22
        elif rs.get("offroad"):
            safety = 0.18
        elif rs.get("safe_distance"):
            safety = 0.86
        else:
            safety = 0.54

        if action == "brake" and hd < 12.0:
            safety = min(0.95, safety + 0.10)
        if not rs.get("signal_respected", True):
            safety = max(0.05, safety - 0.28)

        # ── 2. Efficiency (raw 0–1, weight 20%) ───────────────────────────────
        if stage in ("clearing", "cleared") and action == "accelerate":
            efficiency = 0.88
        elif stage == "approaching" and action in ("brake", "wait"):
            efficiency = 0.76
        elif stage in ("clearing", "cleared") and action in ("brake", "wait"):
            efficiency = 0.24
        elif rs.get("progress_restored"):
            efficiency = 0.80
        elif rs.get("stuck"):
            efficiency = 0.10
        else:
            efficiency = 0.54

        # Oscillation drag (3+ identical actions in a row)
        if len(hist_acts) >= 3 and len(set(hist_acts[-3:])) == 1:
            efficiency = max(0.05, efficiency - 0.22)

        if action == "wait" and hd > 14.0 and stage not in ("approaching",):
            efficiency = max(0.05, efficiency - 0.18)

        # ── 3. Social Compliance (raw 0–1, weight 20%) ───────────────────────
        # Agent must infer correct zone behavior from indirect signals (zone_cues).
        # We reward behavior appropriate to inferred zone context — the agent is
        # NOT told the zone type directly.
        compliance = 0.86 if rs.get("signal_respected", True) else 0.12
        if action in expected:
            compliance = min(0.95, compliance + 0.08)

        # Zone-appropriate behavior scoring
        nearby_places = zone_cues.get("nearby_places", [])
        pedestrian_density = zone_cues.get("pedestrian_density", "low")
        _sensitive_pois = {"hospital", "school", "temple", "pharmacy", "playground"}
        in_sensitive_zone = bool(_sensitive_pois.intersection(set(nearby_places)))
        high_density = pedestrian_density in ("high", "very_high")

        if in_sensitive_zone or high_density:
            # Reward cautious behavior in sensitive zones
            if action in ("brake", "wait"):
                compliance = min(0.97, compliance + 0.10)
            elif action == "horn" and in_sensitive_zone:
                # Honking in a hospital/temple/school zone is anti-social
                compliance = max(0.04, compliance - 0.30)
            elif action == "accelerate" and high_density:
                # Accelerating through a dense zone is risky and anti-social
                compliance = max(0.05, compliance - 0.20)
        elif "highway_entry" in nearby_places:
            # Reward decisive acceleration on highway entry
            if action == "accelerate":
                compliance = min(0.96, compliance + 0.12)
            elif action in ("brake", "wait") and stage not in ("approaching",):
                compliance = max(0.06, compliance - 0.15)

        # ── 4. Smoothness (raw 0–1, weight 10%) ──────────────────────────────
        # Penalizes: harsh full-value brakes, oscillation, abrupt direction changes
        smoothness = 0.78  # default: neutral
        action_value = float(act.get("value", 0.0) if isinstance(act, dict) else 0.0)
        if action == "brake" and action_value > 0.85 and hd > 8.0:
            # Harsh brake when hazard not critical → jerk penalty
            smoothness = max(0.10, smoothness - 0.30)
        if len(hist_acts) >= 2:
            last = hist_acts[-1] if hist_acts else None
            second_last = hist_acts[-2] if len(hist_acts) >= 2 else None
            # Oscillating between opposing actions (steer_left↔steer_right, brake↔accelerate)
            opposing = {("brake", "accelerate"), ("accelerate", "brake"),
                        ("steer_left", "steer_right"), ("steer_right", "steer_left")}
            if (last, action) in opposing or (second_last, action) in opposing:
                smoothness = max(0.05, smoothness - 0.25)
        if action in ("brake", "wait") and stage in ("clearing", "cleared"):
            # Still stopping when road is clear → jerk pattern
            smoothness = max(0.10, smoothness - 0.20)

        # ── 5. Negotiation Success (raw 0–1, weight 10%) ──────────────────────
        # Scores whether the ego's action matched the recommended negotiation strategy
        negotiation = 0.65  # default: neutral
        neg_plan = negotiation_result.get("negotiation_plan", [])
        expected_outcome = negotiation_result.get("expected_outcome", "")
        overall_approach = negotiation_result.get("overall_approach", "balanced")

        if expected_outcome == "smooth_pass":
            if action in ("accelerate", "change_lane_left", "change_lane_right"):
                negotiation = 0.88
            elif action in ("brake", "wait"):
                negotiation = max(0.25, negotiation - 0.15)
        elif expected_outcome == "delayed_pass":
            if action in ("wait", "brake"):
                negotiation = 0.82
        elif expected_outcome == "collision_risk":
            if action in ("brake", "wait", "steer_left", "steer_right"):
                negotiation = 0.90
            elif action == "accelerate":
                negotiation = 0.08

        # Alignment with negotiation plan
        for plan_entry in neg_plan:
            suggested = plan_entry.get("ego_strategy", "wait")
            actor_intent = plan_entry.get("inferred_intent", "cautious")
            if suggested == "yield" and action in ("brake", "wait", "steer_left", "steer_right"):
                negotiation = min(0.95, negotiation + 0.08)
            elif suggested == "assert" and action in ("accelerate", "horn"):
                negotiation = min(0.95, negotiation + 0.08)
            elif suggested == "yield" and action == "accelerate" and actor_intent in ("aggressive", "rush"):
                # Asserting into an aggressive actor = bad negotiation
                negotiation = max(0.06, negotiation - 0.25)

        # ── Weighted composite → clamped strictly to [0.02, 0.98] ─────────────
        composite = (
            0.40 * safety
            + 0.20 * efficiency
            + 0.20 * compliance
            + 0.10 * smoothness
            + 0.10 * negotiation
        )
        score = _clamp(composite)

        # ── Feedback ──────────────────────────────────────────────────────────
        parts: List[str] = []
        if rs.get("collision"):
            parts.append("collision detected")
        elif rs.get("near_miss"):
            parts.append("near miss")
        if stage in ("clearing", "cleared") and action == "accelerate":
            parts.append("correct resume after clearing")
        elif stage == "approaching" and action == "brake":
            parts.append("correct defensive brake")
        if not rs.get("signal_respected", True):
            parts.append("signal violated")
        if in_sensitive_zone and action == "horn":
            parts.append("inappropriate horn in sensitive zone")
        if negotiation_result.get("expected_outcome") == "smooth_pass" and action == "accelerate":
            parts.append("good negotiation: assertive in clear path")
        if not parts:
            parts.append("nominal step")

        return {
            "score":               round(score, 4),
            "safety":              round(_clamp(safety), 4),
            "efficiency":          round(_clamp(efficiency), 4),
            "compliance":          round(_clamp(compliance), 4),
            "smoothness":          round(_clamp(smoothness), 4),
            "negotiation":         round(_clamp(negotiation), 4),
            "feedback":            "; ".join(parts),
        }
