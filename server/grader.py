"""Standalone grader for the OpenEnv validator.

This file has ZERO external dependencies — only Python builtins.
The validator imports this as: autodrive_env.server.grader:HeuristicGrader

HeuristicGrader() can be instantiated with no arguments.
Its score is ALWAYS strictly between 0.02 and 0.98 (never 0.0 or 1.0).
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

        # ── Safety (raw 0–1) ──────────────────────────────────────────────
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

        # ── Efficiency (raw 0–1) ──────────────────────────────────────────
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

        # Principal: extra penalty for unjustified wait after clearing
        if action == "wait" and hd > 14.0 and stage not in ("approaching",):
            efficiency = max(0.05, efficiency - 0.18)

        # ── Compliance (raw 0–1) ──────────────────────────────────────────
        compliance = 0.86 if rs.get("signal_respected", True) else 0.12
        if action in expected:
            compliance = min(0.95, compliance + 0.08)

        # ── Weighted composite → clamped strictly to [0.02, 0.98] ─────────
        composite = 0.50 * safety + 0.30 * efficiency + 0.20 * compliance
        score = _clamp(composite)

        # ── Feedback ──────────────────────────────────────────────────────
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
        if not parts:
            parts.append("nominal step")

        return {
            "score":      round(score, 4),
            "safety":     round(_clamp(safety), 4),
            "efficiency": round(_clamp(efficiency), 4),
            "compliance": round(_clamp(compliance), 4),
            "feedback":   "; ".join(parts),
        }
