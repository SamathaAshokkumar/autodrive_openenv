"""
Task graders for AutoDrive Gym.
Zero external dependencies — only Python builtins.

All return values are strictly in (0.01, 0.99) — never 0.0, never 1.0.
"""

from __future__ import annotations
from typing import Any, Dict

_LO = 0.02
_HI = 0.98


def _clamp(v: float) -> float:
    return max(_LO, min(_HI, float(v)))


def grade_action(task_id: str, action: str, signals: Dict[str, Any]) -> float:
    """Score a single driving action for the given task and sensor signals.

    Args:
        task_id:  One of 'pedestrian_crossing', 'auto_cut_in', 'bike_blind_spot'.
        action:   The driving action string (brake, wait, accelerate, etc.).
        signals:  Dict with keys: hazard_distance, scenario_stage, collision,
                  near_miss, safe_distance, signal_respected, stuck.

    Returns:
        float strictly in (0.02, 0.98).
    """
    action = str(action).strip().lower()

    # Normalise action variants
    if action not in ("brake", "wait", "accelerate", "horn",
                      "steer_left", "steer_right",
                      "change_lane_left", "change_lane_right"):
        action = "wait"  # unknown → treat as passive

    if task_id == "pedestrian_crossing":
        raw = _grade_pedestrian(action, signals)
    elif task_id == "auto_cut_in":
        raw = _grade_auto_cut_in(action, signals)
    elif task_id == "bike_blind_spot":
        raw = _grade_bike_blind_spot(action, signals)
    else:
        raw = _grade_generic(action, signals)

    return round(_clamp(raw), 4)


# ── Task-specific graders ────────────────────────────────────────────────────

def _grade_pedestrian(action: str, signals: Dict[str, Any]) -> float:
    """Pedestrian crossing: must brake early and wait until clear."""
    stage = str(signals.get("scenario_stage", "approaching"))
    hd = float(signals.get("hazard_distance", 999.0) or 999.0)
    collision = bool(signals.get("collision", False))
    near_miss = bool(signals.get("near_miss", False))
    safe_dist = bool(signals.get("safe_distance", True))

    if collision:
        return 0.04   # catastrophic
    if near_miss:
        return 0.18

    if stage == "approaching":
        if action == "brake":
            # Better score the closer the hazard (more urgent braking)
            urgency = max(0.0, 1.0 - hd / 20.0)
            return round(0.65 + 0.25 * urgency, 4)
        if action == "wait":
            return 0.62
        if action == "horn":
            return 0.55  # valid social signal near pedestrian
        if action == "accelerate":
            return 0.12  # dangerous when approaching

    elif stage in ("clearing", "cleared"):
        if action == "accelerate":
            return 0.88  # correct: resume after clear
        if action in ("brake", "wait"):
            if hd > 15.0:
                return 0.30  # still braking when clear — overcautious
            return 0.55

    # Bonus for maintaining safe distance
    base = 0.55
    if safe_dist:
        base += 0.08
    return base


def _grade_auto_cut_in(action: str, signals: Dict[str, Any]) -> float:
    """Auto-rickshaw cut-in: must yield, not accelerate into conflict."""
    stage = str(signals.get("scenario_stage", "approaching"))
    hd = float(signals.get("hazard_distance", 999.0) or 999.0)
    collision = bool(signals.get("collision", False))
    near_miss = bool(signals.get("near_miss", False))
    safe_dist = bool(signals.get("safe_distance", True))

    if collision:
        return 0.04
    if near_miss:
        return 0.20

    if stage == "approaching":
        if action == "brake":
            urgency = max(0.0, 1.0 - hd / 15.0)
            return round(0.68 + 0.22 * urgency, 4)
        if action == "wait":
            return 0.65
        if action in ("change_lane_left", "change_lane_right"):
            return 0.58  # lane change is a valid response to cut-in
        if action == "accelerate":
            return 0.10  # most dangerous — closing gap

    elif stage in ("clearing", "cleared"):
        if action == "accelerate":
            return 0.86
        if action in ("brake", "wait") and hd > 12.0:
            return 0.32

    base = 0.55
    if safe_dist:
        base += 0.08
    return base


def _grade_bike_blind_spot(action: str, signals: Dict[str, Any]) -> float:
    """Bike from blind spot: must be cautious, then resume."""
    stage = str(signals.get("scenario_stage", "approaching"))
    hd = float(signals.get("hazard_distance", 999.0) or 999.0)
    collision = bool(signals.get("collision", False))
    near_miss = bool(signals.get("near_miss", False))
    safe_dist = bool(signals.get("safe_distance", True))

    if collision:
        return 0.04
    if near_miss:
        return 0.22

    if stage == "approaching":
        if action == "wait":
            return 0.72   # best: hold position, scan blind spot
        if action == "brake":
            return 0.68
        if action == "horn":
            return 0.55   # horn helps alert the cyclist
        if action == "accelerate":
            return 0.12

    elif stage in ("clearing", "cleared"):
        if action == "accelerate":
            return 0.88
        if action in ("brake", "wait") and hd > 12.0:
            return 0.28

    base = 0.55
    if safe_dist:
        base += 0.08
    return base


def _grade_generic(action: str, signals: Dict[str, Any]) -> float:
    """Fallback grader for unknown task types."""
    if signals.get("collision"):
        return 0.04
    if action in ("brake", "wait"):
        return 0.62
    if action == "accelerate":
        return 0.45
    return 0.50
