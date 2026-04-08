"""Judge module for AutoDrive Gym.

Three-layer evaluation:
1. LLMJudge    — calls the configured LLM with a detailed rubric prompt.
2. HeuristicJudge — deterministic fallback that never calls an LLM.
3. HeuristicGrader — callable wrapper for OpenEnv /grader endpoint.
                     Returns a multi-dimensional score dict in (0, 1).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .llm_client import LLMClient

# ── Judge persona system prompts ─────────────────────────────────────────────

PERSONAS = {
    "junior": (
        "You are a lenient Indian driving safety mentor evaluating a learner agent. "
        "Reward cautious, hesitant-but-safe behavior. Gently penalise only dangerous mistakes. "
        "Scoring range: -1.0 (collision) to +1.0 (perfect safe response). "
        "Be encouraging; partial credit for any attempt to slow down or yield."
    ),
    "senior": (
        "You are a senior autonomous driving evaluator for Indian road conditions. "
        "Reward decisions that are BOTH safe AND efficient. "
        "Penalise over-braking that causes unnecessary stalls, as well as under-braking near hazards. "
        "Scoring range: -1.0 (collision/severe error) to +1.0 (precise, smooth, efficient response)."
    ),
    "principal": (
        "You are a strict principal safety auditor for a Tier-1 AV company. "
        "Evaluate against professional Indian driving norms. "
        "Penalise oscillating (brake-wait-brake-wait) heavily. "
        "Penalise failure to resume after hazard clears. "
        "Any collision, near-miss, or signal violation scores -1.0. "
        "Scoring range: -1.0 to +1.0. Be precise."
    ),
}

_JUDGE_SYSTEM = """You are an autonomous driving safety judge for Indian road conditions.

Return ONLY valid JSON — no markdown, no explanation:
{
  "score": <float between -1.0 and 1.0>,
  "safety": <float 0.0-1.0>,
  "efficiency": <float 0.0-1.0>,
  "compliance": <float 0.0-1.0>,
  "feedback": "<one sentence assessment>"
}

Scoring guide:
  score = weighted average: 0.5*safety + 0.3*efficiency + 0.2*compliance
  safety    : 1.0 = no hazard proximity risk; 0.0 = imminent collision risk
  efficiency: 1.0 = smooth, timely progress; 0.0 = stalled or oscillating
  compliance: 1.0 = signal/law respected; 0.0 = red-light run or illegal manoeuvre
"""

_VERIFY_SYSTEM = """You are verifying whether an autonomous driving episode was resolved safely.

Return ONLY valid JSON:
{
  "resolved": true | false,
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>"
}

resolved = true ONLY when:
- No collision or near-miss occurred in the last few steps
- The vehicle made forward progress after the hazard cleared
- No signal violations or stuck state at episode end
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
        repeated_actions = [h.get("action") for h in recent]
        oscillating = len(set(repeated_actions)) == 1 and len(repeated_actions) >= 3

        prompt = json.dumps({
            "scenario_type": scenario.get("type", "unknown"),
            "root_cause": scenario.get("root_cause", ""),
            "correct_fix": scenario.get("correct_fix_description", ""),
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
            "ego_speed": observation.get("ego_state", {}).get("speed", 0.0) if isinstance(observation, dict) else 0.0,
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
            score = max(-1.0, min(1.0, float(res["score"])))
            feedback = str(res.get("feedback", ""))
            return score, feedback
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
            "expected_behavior": scenario.get("expected_behavior", []),
            "last_3_actions": [{"action": h.get("action"), "reward": h.get("reward")} for h in history[-3:]],
            "validation": validation,
            "ego_speed": observation.get("ego_state", {}).get("speed", 0.0) if isinstance(observation, dict) else 0.0,
        }, indent=2)

        try:
            res = self.llm.chat_json(_VERIFY_SYSTEM, prompt, temperature=0.05, max_tokens=150)
            if "resolved" not in res:
                raise ValueError("missing resolved flag")
            return bool(res["resolved"]), str(res.get("reason", ""))
        except Exception:
            return HeuristicJudge.verify_resolution(scenario, history, observation, validation)


class HeuristicJudge:
    """Deterministic step-level evaluation — used as LLMJudge fallback."""

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
        act = action.get("action", "wait")

        # ── Hard failures ────────────────────────────────────────────────────
        if result_state.get("collision"):
            return -1.0, "collision — maximum penalty"
        if result_state.get("offroad"):
            score -= 0.55
            feedback.append("left drivable area")
        if not result_state.get("signal_respected", True):
            score -= 0.55
            feedback.append("signal violation")

        # ── Near-miss (severity depends on persona) ──────────────────────────
        if result_state.get("near_miss"):
            penalty = 0.25 if persona == "junior" else 0.45
            score -= penalty
            feedback.append("near miss")

        # ── Safety-positive actions ──────────────────────────────────────────
        obs_dict = observation if isinstance(observation, dict) else {}
        hazard_dist = float(obs_dict.get("hazard_distance", 999.0) or 999.0)
        objects = obs_dict.get("sensor_data", {}).get("objects", []) if isinstance(obs_dict.get("sensor_data"), dict) else []

        if act == "brake" and hazard_dist < 12.0:
            score += 0.40
            feedback.append("braking for nearby hazard")
        elif act == "brake" and hazard_dist >= 12.0 and persona == "principal":
            score -= 0.15
            feedback.append("unnecessary braking when path is clear")

        if act == "wait" and hazard_dist < 10.0:
            score += 0.20
            feedback.append("cautious wait near hazard")

        if act == "horn" and any(str(obj.get("type", "")).lower() in {"pedestrian", "bike", "dog", "cow"} for obj in objects):
            score += 0.10
            feedback.append("horn used as social signal")

        # ── Efficiency (resume after clearing) ──────────────────────────────
        stage = str(obs_dict.get("scenario_stage", "approaching"))
        if stage in ("clearing", "cleared") and act == "accelerate":
            score += 0.40
            feedback.append("accelerating after hazard cleared — correct")
        elif stage in ("clearing", "cleared") and act in ("brake", "wait") and hazard_dist > 15.0:
            penalty = 0.10 if persona == "junior" else 0.25
            score -= penalty
            feedback.append("still braking after hazard cleared")

        # ── Oscillation penalty ──────────────────────────────────────────────
        if len(history) >= 3:
            recent = [h.get("action") for h in history[-3:]]
            if len(set(recent)) == 1:
                penalty = 0.10 if persona == "junior" else 0.25
                score -= penalty
                feedback.append("oscillating / stuck in same action")

        # ── Safe distance bonus ──────────────────────────────────────────────
        if result_state.get("safe_distance"):
            score += 0.15
            feedback.append("safe distance maintained")

        if result_state.get("incident_cleared") and result_state.get("progress_restored"):
            score += 0.35
            feedback.append("incident fully resolved with progress")

        # ── Persona strictness modifier ──────────────────────────────────────
        if persona == "principal":
            # Strict mode: penalise anything that isn't decisive
            if act == "wait" and hazard_dist > 14.0 and stage not in ("approaching",):
                score -= 0.20
                feedback.append("unjustified wait at safe distance")

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
            return True, "incident cleared and safe progress restored"
        # Partial success: cleared with safe distance but no explicit progress signal
        if validation.get("incident_cleared") and validation.get("safe_distance"):
            return True, "incident cleared with safe distance maintained"
        if validation.get("safe_distance") and not validation.get("near_miss") and len(history) >= 4:
            recent = [h.get("action") for h in history[-4:]]
            if "accelerate" in recent:
                return True, "safe navigation maintained and agent accelerated"
        return False, "hazard not fully resolved or no forward progress"


class HeuristicGrader:
    """Callable multi-dimensional grader for the OpenEnv /grader endpoint.

    Returns score strictly in (0.001, 0.999) with sub-dimension breakdown.
    Importable as: autodrive_env.server.judge:HeuristicGrader
    """

    _EPS = 1e-3

    def __init__(self, persona: str = "junior"):
        self.persona = persona

    @staticmethod
    def _clamp(v: float) -> float:
        return max(HeuristicGrader._EPS, min(1.0 - HeuristicGrader._EPS, v))

    def __call__(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        result_state: Dict[str, Any],
        scenario: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        act = action.get("action", "wait")
        obs_dict = observation if isinstance(observation, dict) else {}
        hazard_dist = float(obs_dict.get("hazard_distance", 999.0) or 999.0)
        stage = str(obs_dict.get("scenario_stage", "approaching"))
        expected = scenario.get("expected_behavior", [])

        # ── Safety dimension (0–1) ───────────────────────────────────────────
        if result_state.get("collision"):
            safety = 0.01
        elif result_state.get("near_miss"):
            safety = 0.25
        elif result_state.get("offroad"):
            safety = 0.20
        elif result_state.get("safe_distance"):
            safety = 0.90
        else:
            safety = 0.55

        if act == "brake" and hazard_dist < 12.0:
            safety = min(0.999, safety + 0.10)
        if not result_state.get("signal_respected", True):
            safety = max(0.01, safety - 0.30)

        # ── Efficiency dimension (0–1) ────────────────────────────────────────
        if stage in ("clearing", "cleared") and act == "accelerate":
            efficiency = 0.90
        elif stage == "approaching" and act in ("brake", "wait"):
            efficiency = 0.80
        elif stage in ("clearing", "cleared") and act in ("brake", "wait"):
            efficiency = 0.25  # stalling after clearing
        elif result_state.get("progress_restored"):
            efficiency = 0.85
        elif result_state.get("stuck"):
            efficiency = 0.10
        else:
            efficiency = 0.55

        # ── Compliance dimension (0–1) ────────────────────────────────────────
        compliance = 0.90 if result_state.get("signal_respected", True) else 0.10
        if act in expected:
            compliance = min(0.999, compliance + 0.08)

        # ── Oscillation penalty across all dims ──────────────────────────────
        if len(history) >= 3 and len(set(h.get("action") for h in history[-3:])) == 1:
            efficiency = max(0.01, efficiency - 0.25)

        # ── Weighted composite ────────────────────────────────────────────────
        composite = 0.50 * safety + 0.30 * efficiency + 0.20 * compliance
        score = self._clamp(composite)

        feedback_parts = []
        if result_state.get("collision"):
            feedback_parts.append("collision")
        elif result_state.get("near_miss"):
            feedback_parts.append("near miss")
        if stage in ("clearing", "cleared") and act == "accelerate":
            feedback_parts.append("correct resume after clearing")
        elif stage == "approaching" and act == "brake":
            feedback_parts.append("correct defensive brake")
        if not result_state.get("signal_respected", True):
            feedback_parts.append("signal violated")
        if not feedback_parts:
            feedback_parts.append("nominal step")

        return {
            "score": round(score, 4),
            "safety": round(self._clamp(safety), 4),
            "efficiency": round(self._clamp(efficiency), 4),
            "compliance": round(self._clamp(compliance), 4),
            "feedback": "; ".join(feedback_parts),
        }

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, observation: Dict[str, Any], action: Dict[str, Any], result_state: Dict[str, Any],
                 scenario: Dict[str, Any], history: List[Dict[str, Any]], persona: str = "junior") -> Tuple[float, str]:
        prompt = (
            f"SCENARIO: {scenario}\n"
            f"OBSERVATION: {observation}\n"
            f"ACTION: {action}\n"
            f"RESULT_STATE: {result_state}\n"
            f"HISTORY: {history[-5:]}\n"
            'Return JSON {"score": <float -1 to 1>, "feedback": "<short evaluation>"}'
        )
        try:
            res = self.llm.chat_json(PERSONAS[persona], prompt, temperature=0.2, max_tokens=200)
            if "score" not in res:
                raise ValueError("judge response missing score")
            return max(-1.0, min(1.0, float(res.get("score", 0.0)))), str(res.get("feedback", ""))
        except Exception:
            return HeuristicJudge.evaluate(observation, action, result_state, scenario, history, persona)

    def verify_resolution(self, scenario: Dict[str, Any], history: List[Dict[str, Any]],
                          observation: Dict[str, Any], validation: Dict[str, Any]) -> Tuple[bool, str]:
        prompt = (
            f"SCENARIO: {scenario}\nHISTORY: {history[-5:]}\nOBSERVATION: {observation}\nVALIDATION: {validation}\n"
            'Return JSON {"resolved": true/false, "reason": "<short explanation>"}'
        )
        try:
            res = self.llm.chat_json("Verify whether the driving response was truly safe and correct.", prompt, temperature=0.1, max_tokens=200)
            if "resolved" not in res:
                raise ValueError("verification response missing resolved flag")
            return bool(res.get("resolved", False)), str(res.get("reason", ""))
        except Exception:
            return HeuristicJudge.verify_resolution(scenario, history, observation, validation)


class HeuristicJudge:
    @staticmethod
    def evaluate(observation: Dict[str, Any], action: Dict[str, Any], result_state: Dict[str, Any],
                 scenario: Dict[str, Any], history: List[Dict[str, Any]], persona: str = "junior") -> Tuple[float, str]:
        score = 0.0
        feedback = []
        if result_state.get("collision"):
            return -1.0, "collision occurred"
        if result_state.get("near_miss"):
            score -= 0.4
            feedback.append("near miss detected")
        if result_state.get("offroad"):
            score -= 0.5
            feedback.append("left drivable region")
        if not result_state.get("signal_respected", True):
            score -= 0.6
            feedback.append("signal was not respected")
        if action.get("action") == "brake" and observation.get("sensor_data", {}).get("objects"):
            score += 0.3
            feedback.append("braking for visible hazard")
        if action.get("action") == "wait":
            score += 0.15
            feedback.append("cautious wait")
        if len(history) >= 1 and history[-1].get("action") == action.get("action"):
            score -= 0.2
            feedback.append("oscillation/repetition")
        return max(-1.0, min(1.0, score)), "; ".join(feedback) if feedback else "neutral"

    @staticmethod
    def verify_resolution(scenario: Dict[str, Any], history: List[Dict[str, Any]], observation: Dict[str, Any],
                          validation: Dict[str, Any]) -> Tuple[bool, str]:
        if validation.get("collision"):
            return False, "collision occurred"
        if validation.get("offroad"):
            return False, "vehicle left drivable area"
        if not validation.get("signal_respected", True):
            return False, "signal violation"
        if validation.get("stuck"):
            return False, "vehicle got stuck"
        if validation.get("minimum_distance", 999.0) < 2.5:
            return False, "hazard remained too close"
        if validation.get("reached_goal"):
            return True, "goal reached safely"
        if validation.get("incident_cleared") and validation.get("progress_restored") and validation.get("safe_distance") and not validation.get("near_miss"):
            return True, "incident cleared and safe progress restored"
        if validation.get("safe_distance") and not validation.get("near_miss"):
            return True, "safe navigation maintained through the incident"
        return False, "safety outcome still uncertain"


class HeuristicGrader:
    """Callable grader wrapper returning normalized 0.0-1.0 score and feedback.

    The OpenEnv/HF validator expects an importable callable grader. This class
    delegates to HeuristicJudge and converts its -1..1 score to 0..1.
    """

    def __init__(self, persona: str = "junior"):
        self.persona = persona

    def __call__(self, observation: Dict[str, Any], action: Dict[str, Any], result_state: Dict[str, Any],
                 scenario: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        score, feedback = HeuristicJudge.evaluate(observation, action, result_state, scenario, history, self.persona)
        # Normalize from [-1, 1] to (0, 1) with a tiny epsilon to avoid exact 0.0 or 1.0
        eps = 1e-3
        raw = (float(score) + 1.0) / 2.0
        norm = max(eps, min(1.0 - eps, raw))
        return {"score": norm, "feedback": feedback}
