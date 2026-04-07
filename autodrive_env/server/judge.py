"""Judge module for AutoDrive Gym."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .llm_client import LLMClient

PERSONAS = {
    "junior": "You are a lenient Indian driving safety mentor. Reward cautious, understandable behavior.",
    "senior": "You are a standard autonomous driving evaluator for Indian road conditions. Reward safe and efficient decisions.",
    "principal": "You are a strict expert evaluator. Penalize unsafe, oscillating, or unrealistic driving heavily.",
}


class LLMJudge:
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
