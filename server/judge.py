"""Judge module for AutoDrive Gym.

Three-layer evaluation:
1. LLMJudge       -- calls the configured LLM with a detailed rubric prompt.
2. HeuristicJudge -- deterministic fallback, never calls an LLM.
3. HeuristicGrader -- callable grader for the OpenEnv validator.
                      Score is ALWAYS strictly between 0 and 1 (exclusive).
                      Import path: autodrive_env.server.judge:HeuristicGrader
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .llm_client import LLMClient

PERSONAS = {
    "junior": (
        "You are a lenient Indian driving safety mentor evaluating a learner agent in Indian road conditions. "
        "Reward cautious, hesitant-but-safe behavior generously. Gently penalise only truly dangerous mistakes "
        "(collision, offroad). Give partial credit for any attempt to slow down or yield. "
        "Scoring range: -1.0 (collision) to +1.0 (perfect safe response). Be encouraging."
    ),
    "senior": (
        "You are a senior autonomous driving evaluator for Indian road conditions. "
        "Reward decisions that are BOTH safe AND efficient — smooth progress matters as much as safety. "
        "Penalise over-braking that stalls traffic unnecessarily, and under-braking near hazards. "
        "Penalise repeated same-action oscillation. "
        "Scoring range: -1.0 (collision/severe error) to +1.0 (precise, smooth, efficient response)."
    ),
    "principal": (
        "You are a strict principal safety auditor for a Tier-1 AV company operating in India. "
        "Evaluate rigorously against professional driving norms. "
        "Penalise oscillating (brake-wait-brake-wait) sequences heavily — this is unacceptable. "
        "Penalise failure to resume speed promptly after hazard fully clears. "
        "Any collision, near-miss, or signal violation must score at or near -1.0. "
        "Indecisive waiting when the path is clear is also penalised. "
        "Scoring range: -1.0 to +1.0. Be precise and unforgiving of professional-level errors."
    ),
}

_JUDGE_SYSTEM = """You are an autonomous driving safety judge for Indian road conditions.
Return ONLY valid JSON -- no markdown, no explanation:
{
  "score": <float between -1.0 and 1.0>,
  "safety": <float 0.0-1.0>,
  "efficiency": <float 0.0-1.0>,
  "compliance": <float 0.0-1.0>,
  "feedback": "<one sentence assessment>"
}
Scoring guide: score = 0.5*safety + 0.3*efficiency + 0.2*compliance
"""

_VERIFY_SYSTEM = """You are verifying whether an autonomous driving episode was resolved safely.
Return ONLY valid JSON:
{
  "resolved": true | false,
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>"
}
"""


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        result_state: Dict[str, Any],
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
        persona: str = "junior",
    ) -> Tuple[float, str]:
        recent = history[-4:] if len(history) >= 4 else history
        oscillating = len(set(h.get("action") for h in recent)) == 1 and len(recent) >= 3
        prompt = json.dumps({
            "scenario_type": scenario.get("type", "unknown"),
            "root_cause": scenario.get("root_cause", ""),
            "expected_behavior": scenario.get("expected_behavior", []),
            "action_taken": action,
            "result": {
                "collision": result_state.get("collision", False),
                "near_miss": result_state.get("near_miss", False),
                "offroad": result_state.get("offroad", False),
                "stuck": result_state.get("stuck", False),
                "safe_distance": result_state.get("safe_distance", True),
                "signal_respected": result_state.get("signal_respected", True),
                "incident_cleared": result_state.get("incident_cleared", False),
                "progress_restored": result_state.get("progress_restored", False),
            },
            "hazard_distance": observation.get("hazard_distance", 999.0) if isinstance(observation, dict) else 999.0,
            "is_oscillating": oscillating,
            "steps_into_episode": len(history),
            "persona": persona,
            "persona_instruction": PERSONAS.get(persona, PERSONAS["junior"]),
        }, indent=2)
        try:
            res = self.llm.chat_json(_JUDGE_SYSTEM, prompt, temperature=0.1, max_tokens=220)
            if "score" not in res:
                raise ValueError("missing score")
            return max(-1.0, min(1.0, float(res["score"]))), str(res.get("feedback", ""))
        except Exception:
            return HeuristicJudge.evaluate(observation, action, result_state, scenario, history, persona)

    def verify_resolution(
        self,
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
        observation: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Tuple[bool, str]:
        prompt = json.dumps({
            "scenario_type": scenario.get("type", "unknown"),
            "last_3_actions": [{"action": h.get("action"), "reward": h.get("reward")} for h in history[-3:]],
            "validation": validation,
        }, indent=2)
        try:
            res = self.llm.chat_json(_VERIFY_SYSTEM, prompt, temperature=0.05, max_tokens=150)
            if "resolved" not in res:
                raise ValueError("missing resolved flag")
            return bool(res["resolved"]), str(res.get("reason", ""))
        except Exception:
            return HeuristicJudge.verify_resolution(scenario, history, observation, validation)


class HeuristicJudge:
    """Deterministic step-level evaluation -- used as LLMJudge fallback."""

    @staticmethod
    def evaluate(
        observation: Dict[str, Any],
        action: Dict[str, Any],
        result_state: Dict[str, Any],
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
        persona: str = "junior",
    ) -> Tuple[float, str]:
        score = 0.0
        feedback: List[str] = []
        act = action.get("action", "wait") if isinstance(action, dict) else "wait"
        rs = result_state if isinstance(result_state, dict) else {}
        obs = observation if isinstance(observation, dict) else {}

        if rs.get("collision"):
            return -1.0, "collision -- maximum penalty"
        if rs.get("offroad"):
            score -= 0.55
            feedback.append("left drivable area")
        if not rs.get("signal_respected", True):
            score -= 0.55
            feedback.append("signal violation")
        if rs.get("near_miss"):
            score -= 0.25 if persona == "junior" else 0.45
            feedback.append("near miss")

        hazard_dist = float(obs.get("hazard_distance", 999.0) or 999.0)
        stage = str(obs.get("scenario_stage", "approaching"))
        objects = obs.get("sensor_data", {}).get("objects", []) if isinstance(obs.get("sensor_data"), dict) else []

        if act == "brake" and hazard_dist < 12.0:
            score += 0.40
            feedback.append("braking for nearby hazard")
        elif act == "brake" and hazard_dist >= 12.0 and persona == "principal":
            score -= 0.15
            feedback.append("unnecessary braking")
        if act == "wait" and hazard_dist < 10.0:
            score += 0.20
            feedback.append("cautious wait near hazard")
        if act == "horn" and any(str(o.get("type", "")).lower() in {"pedestrian", "bike", "dog", "cow"} for o in objects):
            score += 0.10
            feedback.append("horn used as social signal")
        if stage in ("clearing", "cleared") and act == "accelerate":
            score += 0.40
            feedback.append("accelerating after hazard cleared")
        elif stage in ("clearing", "cleared") and act in ("brake", "wait") and hazard_dist > 15.0:
            score -= 0.10 if persona == "junior" else 0.25
            feedback.append("still braking after hazard cleared")
        if len(history) >= 3 and len(set(h.get("action") for h in history[-3:])) == 1:
            score -= 0.10 if persona == "junior" else 0.25
            feedback.append("oscillating")
        if rs.get("safe_distance"):
            score += 0.15
            feedback.append("safe distance maintained")
        if rs.get("incident_cleared") and rs.get("progress_restored"):
            score += 0.35
            feedback.append("incident fully resolved")
        if persona == "principal" and act == "wait" and hazard_dist > 14.0 and stage != "approaching":
            score -= 0.20
            feedback.append("unjustified wait")

        return max(-1.0, min(1.0, round(score, 3))), "; ".join(feedback) if feedback else "neutral step"

    @staticmethod
    def verify_resolution(
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
        observation: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if validation.get("collision"):
            return False, "collision occurred"
        if validation.get("offroad"):
            return False, "vehicle left drivable area"
        if not validation.get("signal_respected", True):
            return False, "signal violation at end"
        if validation.get("stuck"):
            return False, "vehicle stuck at end of episode"
        if float(validation.get("minimum_distance", 999.0)) < 2.5:
            return False, "hazard was too close at end"
        if validation.get("reached_goal"):
            return True, "goal reached safely"
        if (validation.get("incident_cleared") and validation.get("progress_restored")
                and validation.get("safe_distance") and not validation.get("near_miss")):
            return True, "incident cleared and progress restored"
        if validation.get("incident_cleared") and validation.get("safe_distance"):
            return True, "incident cleared with safe distance"
        if validation.get("safe_distance") and not validation.get("near_miss") and len(history) >= 4:
            if "accelerate" in [h.get("action") for h in history[-4:]]:
                return True, "safe navigation and agent accelerated"
        return False, "hazard not fully resolved"


class HeuristicGrader:
    """Callable grader for the OpenEnv validator.

    Score is ALWAYS strictly between 0.02 and 0.98 -- never 0.0, never 1.0.
    Can be instantiated with no arguments: HeuristicGrader()
    persona controls strictness: 'junior' (lenient), 'senior' (balanced), 'principal' (strict).
    """

    _LO = 0.02
    _HI = 0.98

    def __init__(self, persona: str = "principal"):
        self.persona = persona

    @classmethod
    def _clamp(cls, v: float) -> float:
        """Clamp to (_LO, _HI) which is strictly inside (0, 1)."""
        return max(cls._LO, min(cls._HI, float(v)))

    def __call__(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        result_state: Dict[str, Any],
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        act = action.get("action", "wait") if isinstance(action, dict) else "wait"
        obs = observation if isinstance(observation, dict) else {}
        rs = result_state if isinstance(result_state, dict) else {}
        sc = scenario if isinstance(scenario, dict) else {}
        hist = history if isinstance(history, list) else []

        hazard_dist = float(obs.get("hazard_distance", 999.0) or 999.0)
        stage = str(obs.get("scenario_stage", "approaching"))
        expected = sc.get("expected_behavior", [])

        # Safety dimension
        if rs.get("collision"):
            safety = 0.04
        elif rs.get("near_miss"):
            safety = 0.22
        elif rs.get("offroad"):
            safety = 0.18
        elif rs.get("safe_distance"):
            safety = 0.88
        else:
            safety = 0.55
        if act == "brake" and hazard_dist < 12.0:
            safety = min(0.96, safety + 0.10)
        if not rs.get("signal_respected", True):
            safety = max(0.04, safety - 0.30)

        # Efficiency dimension
        if stage in ("clearing", "cleared") and act == "accelerate":
            efficiency = 0.90
        elif stage == "approaching" and act in ("brake", "wait"):
            efficiency = 0.78
        elif stage in ("clearing", "cleared") and act in ("brake", "wait"):
            efficiency = 0.22
        elif rs.get("progress_restored"):
            efficiency = 0.82
        elif rs.get("stuck"):
            efficiency = 0.08
        else:
            efficiency = 0.55
        if len(hist) >= 3 and len(set(h.get("action") for h in hist[-3:])) == 1:
            efficiency = max(0.04, efficiency - 0.25)

        # Compliance dimension
        compliance = 0.88 if rs.get("signal_respected", True) else 0.10
        if act in expected:
            compliance = min(0.96, compliance + 0.08)

        # Persona-based strictness modifier
        persona = self.persona
        if persona == "principal":
            # Strict: penalise unnecessary waiting when hazard is far
            if act == "wait" and hazard_dist > 14.0 and stage not in ("approaching",):
                efficiency = max(0.04, efficiency - 0.20)
            # Extra oscillation penalty
            if len(hist) >= 4 and len(set(h.get("action") for h in hist[-4:])) == 1:
                efficiency = max(0.02, efficiency - 0.15)
        elif persona == "senior":
            # Penalise stalling after clearing moderately
            if stage in ("clearing", "cleared") and act in ("brake", "wait") and hazard_dist > 12.0:
                efficiency = max(0.04, efficiency - 0.10)

        # Weighted composite, clamped strictly to [0.02, 0.98]
        composite = 0.50 * safety + 0.30 * efficiency + 0.20 * compliance
        score = self._clamp(composite)

        parts: List[str] = []
        if rs.get("collision"):
            parts.append("collision detected")
        elif rs.get("near_miss"):
            parts.append("near miss")
        if stage in ("clearing", "cleared") and act == "accelerate":
            parts.append("correct resume after clearing")
        elif stage == "approaching" and act == "brake":
            parts.append("correct defensive brake")
        if not rs.get("signal_respected", True):
            parts.append("signal violated")
        if not parts:
            parts.append("nominal step")

        return {
            "score": round(score, 4),
            "safety": round(self._clamp(safety), 4),
            "efficiency": round(self._clamp(efficiency), 4),
            "compliance": round(self._clamp(compliance), 4),
            "feedback": "; ".join(parts),
        }
