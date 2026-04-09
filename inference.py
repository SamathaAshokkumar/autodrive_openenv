#!/usr/bin/env python3
"""AutoDrive Gym inference runner — outputs [START]/[STEP]/[END] per task."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from autodrive_env import AutoDriveAction, AutoDriveClient
from autodrive_env.agent_baseline import ModularBaselineAgent

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or "missing-token"
BENCHMARK    = os.getenv("AUTODRIVE_BENCHMARK", "autodrive_env")
MAX_STEPS    = int(os.getenv("AUTODRIVE_MAX_STEPS", "15"))

# Tasks that must each get their own [START]/[END] log block
TASK_IDS = [
    "pedestrian_crossing",
    "auto_cut_in",
    "bike_blind_spot",
    "pothole_ahead",
    "speed_breaker",
    "traffic_light_ambiguity",
    "animal_crossing",
    "ambulance_approach",
    "traffic_jam",
    "adversarial",
]

# Score bounds: strictly in (0, 1)
_SCORE_LO = 0.02
_SCORE_HI = 0.98

SYSTEM_PROMPT = """You are an autonomous driving agent navigating dense Indian road conditions.
Choose the BEST SINGLE action for the current situation — not just brake or accelerate.

ACTION MENU (use all of them as appropriate):
  brake            — slow or stop for hazards, red lights, pedestrians, animals
  accelerate       — build speed when the path is clear or hazard has passed
  wait             — hold position when yielding to a signal/police/procession
  steer_left       — veer left to dodge pothole, create ambulance corridor, or avoid right-side cut-in
  steer_right      — veer right to dodge left-side obstacle or create lane space
  horn             — warn pedestrians/animals; sparingly in social situations (processions)
  change_lane_left — move fully to left lane when right lane is blocked
  change_lane_right— move fully to right lane when left lane is blocked

SCENARIO → ACTION MAPPING:
  pedestrian/child crossing  : brake (high value) → wait → accelerate once cleared
  auto/bike cut-in           : brake → wait → accelerate
  bike blind spot merge      : wait or brake, steer_right if safe gap on right
  pothole / speed breaker    : brake, then steer_left or steer_right to go around
  ambulance from behind      : steer_left (create corridor) — do NOT brake hard
  animal on road             : brake (high) → horn → wait → accelerate
  police/flagman override    : wait → accelerate on signal
  traffic jam                : brake → wait (repeating) → accelerate when gap opens
  foggy/rainy visibility     : brake (moderate) → steer_left cautiously, no sudden moves
  wedding procession         : brake → horn (once) → wait until gap → accelerate
  construction single lane   : wait → brake → accelerate when flagman signals

STAGE RULES:
  approaching : hazard still ahead — use brake/wait/steer/horn as appropriate
  clearing    : hazard moving away — ease off, use accelerate gently or wait
  cleared     : hazard gone — accelerate to resume normal speed

DISTANCE RULES:
  < 4m  : brake(1.0) or hard emergency only
  4-8m  : brake(0.7-0.9) or steer to avoid
  8-14m : brake(0.4-0.6) or wait or horn
  >14m  : accelerate, change_lane, or maintain with wait

ANTI-LOOP: Never repeat the same action more than 2 times in a row — switch tactic.
If hint says 'accelerate' or 'steer', follow it immediately.

Return ONLY valid JSON — no explanation, no markdown:
{"action": "<one of the 8 actions above>", "value": <float 0.0-1.0>}
"""


# ── Log helpers matching the validator's expected format ─────────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, value: float, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action}:{value:.2f} reward={reward:.4f} "
          f"done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.4f} rewards={rewards_str}", flush=True)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        start = text.index("{")
        end   = text.rindex("}")
        data  = json.loads(text[start:end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def normalize_action(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action = str(payload.get("action", "")).strip().lower()
    if not action:
        return None
    try:
        value = float(payload.get("value", 0.0))
    except Exception:
        value = 0.0
    return {"action": action, "value": max(0.0, min(1.0, value))}


# ── LLM call ─────────────────────────────────────────────────────────────────

def _scene_guidance(task_id: str, hazard_type: str, hazard_dist: float,
                    stage: str, alerts: List[str], hint: str,
                    sensor: Dict[str, Any], environment: Dict[str, Any]) -> str:
    """Return a terse, action-directive sentence that names the preferred action."""
    hint_l = hint.lower()

    # Explicit hint from environment wins
    for kw, directive in [
        ("steer_left",  "STEER_LEFT to create space."),
        ("steer_right", "STEER_RIGHT to avoid obstacle."),
        ("change_lane", "CHANGE_LANE as instructed."),
        ("horn",        "HORN to warn — then wait."),
        ("accelerate",  "Path is clear — ACCELERATE now."),
    ]:
        if kw in hint_l:
            return directive

    # Stage-based override
    if stage in ("clearing", "cleared"):
        return "Hazard has passed — ACCELERATE to resume."

    # Alert-driven
    for alert in alerts:
        al = alert.lower()
        if "ambulance" in al:
            return "AMBULANCE behind — STEER_LEFT immediately to open corridor. Do NOT brake hard."
        if "animal" in al or "cow" in al or "dog" in al:
            return f"ANIMAL on road — BRAKE hard, then HORN once, then WAIT until clear."
        if "police" in al or "flagman" in al:
            return "POLICE/FLAGMAN present — WAIT for hand signal, then ACCELERATE when waved."
        if "wedding" in al or "procession" in al:
            return "PROCESSION blocking road — BRAKE, use HORN once, then WAIT for gap."
        if "traffic jam" in al:
            return "TRAFFIC JAM ahead — BRAKE slowly, WAIT, then ACCELERATE when gap opens."
        if "child" in al:
            return "CHILD on road — BRAKE(1.0) and wait fully."
        if "pothole" in al or "speed breaker" in al:
            return "POTHOLE/RIDGE ahead — BRAKE to slow, then STEER_LEFT or STEER_RIGHT to avoid."

    # Hazard-type directives
    ht = (hazard_type or task_id).lower()
    if "ambulance" in ht:
        return "AMBULANCE approaching — STEER_LEFT to clear a corridor."
    if "pothole" in ht or "speed_breaker" in ht or "ridge" in ht:
        if hazard_dist < 10.0:
            return f"Obstacle {hazard_dist:.0f}m — STEER_LEFT or STEER_RIGHT to dodge it."
        return f"Obstacle {hazard_dist:.0f}m — BRAKE smoothly to reduce speed first."
    if "animal" in ht or "cow" in ht:
        return "Animal on road — BRAKE and HORN; wait for it to move."
    if "police" in ht or "flagman" in ht:
        return "Police directing — WAIT for signal, then ACCELERATE."
    if "traffic_jam" in ht:
        return "Jam ahead — BRAKE to stop, then WAIT in queue."

    # Check environment signal
    sig = str(environment.get("traffic_signal", "none")).lower()
    if sig == "red":
        return "RED LIGHT — BRAKE to stop and WAIT."
    if sig == "policeman_override":
        return "Officer directing traffic — WAIT for hand signal."

    # Distance-based fallback
    if hazard_dist < 5.0:
        return f"DANGER — {hazard_type} only {hazard_dist:.1f}m! BRAKE(1.0) immediately."
    if hazard_dist < 9.0:
        return f"CAUTION — {hazard_type} at {hazard_dist:.1f}m. BRAKE(0.7) or WAIT."
    if hazard_dist < 14.0:
        return f"{hazard_type} at {hazard_dist:.1f}m — BRAKE(0.4) or HORN to warn, then WAIT."
    return "Path looks clear — ACCELERATE or CHANGE_LANE if another vehicle is blocking."


def build_prompt(observation: Any, step_index: int, task_id: str) -> str:
    stage       = getattr(observation, "scenario_stage",  "") or "approaching"
    hazard_dist = float(getattr(observation, "hazard_distance", 999.0) or 999.0)
    hazard_type = getattr(observation, "hazard_type",     "") or task_id
    alerts      = getattr(observation, "active_alerts",   []) or []
    hint        = getattr(observation, "hint",             "") or ""
    ego         = getattr(observation, "ego_state",        {}) or {}
    sensor      = getattr(observation, "sensor_data",      {}) or {}
    environment = getattr(observation, "environment",      {}) or {}
    cmd_out     = getattr(observation, "command_output",   "") or ""

    guidance = _scene_guidance(task_id, hazard_type, hazard_dist, stage,
                               alerts, hint, sensor, environment)

    obs_dict = {
        "task":            task_id,
        "guidance":        guidance,
        "scenario_stage":  stage,
        "hazard_type":     hazard_type,
        "hazard_distance": round(hazard_dist, 1),
        "active_alerts":   alerts,
        "traffic_signal":  environment.get("traffic_signal", "none"),
        "road_condition":  environment.get("road_condition", "normal"),
        "visibility":      environment.get("visibility", "clear"),
        "ego_speed":       ego.get("speed", 0.0),
        "ego_lane":        ego.get("lane", "center"),
        "nearby_objects":  [
            {"type": o.get("type"), "dist": round(float(o.get("distance", 99)), 1),
             "behavior": o.get("behavior", "")}
            for o in (sensor.get("objects") or [])[:4]
        ],
        "hint":            hint,
        "command_output":  cmd_out,
        "step":            step_index,
        "max_steps":       MAX_STEPS,
    }
    return (
        f"STEP={step_index} | TASK={task_id} | STAGE={stage} | HAZARD_DIST={hazard_dist:.1f}m\n"
        f">> {guidance}\n"
        f"OBSERVATION:\n{json.dumps(obs_dict, indent=2)}\n\n"
        "Return JSON action only."
    )


def call_llm(client: OpenAI, observation: Any, step_index: int, task_id: str) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(observation, step_index, task_id)},
        ],
        temperature=0.2,
        max_tokens=128,
    )
    content = completion.choices[0].message.content or ""
    parsed  = parse_json_response(content)
    if parsed is None:
        raise RuntimeError(f"Invalid JSON: {content[:120]}")
    normalized = normalize_action(parsed)
    if normalized is None:
        raise RuntimeError(f"Missing action key: {content[:120]}")
    return normalized


def fallback_action(agent: ModularBaselineAgent, observation: Any) -> Dict[str, Any]:
    raw = {
        "sensor_data":    getattr(observation, "sensor_data",    {}) or {},
        "ego_state":      getattr(observation, "ego_state",      {}) or {},
        "environment":    getattr(observation, "environment",    {}) or {},
        "active_alerts":  getattr(observation, "active_alerts",  []) or [],
        "hint":           getattr(observation, "hint",           "") or "",
        "scenario_stage": getattr(observation, "scenario_stage", "") or "",
        "hazard_distance":getattr(observation, "hazard_distance", 999.0),
        "hazard_type":    getattr(observation, "hazard_type",    "") or "",
    }
    payload = agent.act(raw)
    return {"action": str(payload.get("action", "wait")), "value": float(payload.get("value", 0.0))}


# ── Score computation ─────────────────────────────────────────────────────────

def compute_episode_score(rewards: List[float], resolved: bool,
                          observation: Any) -> float:
    """Return episode score strictly in (_SCORE_LO, _SCORE_HI)."""
    if not rewards:
        return _SCORE_LO

    score = sum(rewards) / len(rewards)   # mean of per-step rewards

    # Resolution bonus — never let it hit 1.0
    if resolved:
        score = max(score, 0.75)

    validation = getattr(observation, "validation", {}) or {} if observation else {}
    if validation.get("collision"):
        score *= 0.30
    if validation.get("stuck"):
        score *= 0.65

    return max(_SCORE_LO, min(_SCORE_HI, round(score, 4)))


# ── Per-task runner ──────────────────────────────────────────────────────────

def run_task(llm_client: OpenAI, fallback: ModularBaselineAgent,
             env_ctx: Any, task_id: str, max_turns: int) -> None:
    """Run one full episode for task_id, emit [START]/[STEP]/[END]."""
    log_start(task_id)

    rewards: List[float] = []
    resolved   = False
    steps_done = 0
    observation = None

    try:
        env = env_ctx.__enter__() if hasattr(env_ctx, "__enter__") else env_ctx

        try:
            reset_result = env.reset(task_id=task_id)
        except Exception:
            reset_result = env.reset()

        observation = reset_result.observation

        for step_index in range(1, max_turns + 1):
            llm_error = None
            try:
                action_dict = call_llm(llm_client, observation, step_index, task_id)
            except Exception as exc:
                llm_error = str(exc).replace("\n", " ")[:120]
                action_dict = fallback_action(fallback, observation)

            action = AutoDriveAction(
                action=action_dict["action"],
                value=float(action_dict["value"]),
            )

            try:
                step_result = env.step(action)
            except Exception as exc:
                log_step(step_index, action_dict["action"], action_dict["value"],
                         _SCORE_LO, True, str(exc)[:120])
                steps_done = step_index
                break

            observation = step_result.observation
            reward      = float(step_result.reward or _SCORE_LO)
            # Clamp per-step reward to be strictly in (0, 1)
            reward      = max(_SCORE_LO, min(_SCORE_HI, reward))
            done        = bool(step_result.done)

            rewards.append(reward)
            steps_done = step_index

            log_step(step_index, action_dict["action"], action_dict["value"],
                     reward, done, llm_error)

            if done:
                resolution = getattr(observation, "resolution", {}) or {}
                resolved   = bool(resolution.get("verified", False))
                break

    except Exception as exc:
        log_step(1, "wait", 0.0, _SCORE_LO, True, str(exc)[:120])
        steps_done = 1

    score = compute_episode_score(rewards, resolved, observation)
    log_end(resolved, steps_done, score, rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDrive Gym inference runner")
    parser.add_argument("--base-url",  default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--max-turns", type=int, default=MAX_STEPS)
    return parser.parse_args()


def main() -> int:
    args      = parse_args()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fallback   = ModularBaselineAgent()

    env_ctx = AutoDriveClient(base_url=args.base_url).sync()

    for task_id in TASK_IDS:
        try:
            run_task(llm_client, fallback, env_ctx, task_id, args.max_turns)
        except Exception as exc:
            log_start(task_id)
            log_step(1, "wait", 0.0, _SCORE_LO, True, str(exc)[:120])
            log_end(False, 0, _SCORE_LO, [])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
