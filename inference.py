#!/usr/bin/env python3
"""AutoDrive Gym inference runner with strict START/STEP/END logs."""

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
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "missing-token"
BENCHMARK = os.getenv("AUTODRIVE_BENCHMARK", "autodrive_env")
MAX_STEPS = int(os.getenv("AUTODRIVE_MAX_STEPS", "15"))

# Exactly three official graded tasks
TASK_IDS = [
    "pedestrian_crossing",
    "bike_blind_spot",
    "adversarial",
]

_SCORE_LO = 0.02
_SCORE_HI = 0.98

SYSTEM_PROMPT = """You are an autonomous driving agent operating in dense Indian road conditions.

STAGE LOGIC:
- scenario_stage="approaching": hazard ahead, prefer brake or wait
- scenario_stage="clearing": hazard passing, gently accelerate
- scenario_stage="cleared": road open, resume progress

DECISION GUIDE:
- hazard_distance < 5  : brake strongly
- hazard_distance 5-12 : brake softly or wait
- hazard_distance > 12 or stage in clearing/cleared : accelerate

Return ONLY valid JSON:
{"action": "accelerate|brake|steer_left|steer_right|horn|wait|change_lane_left|change_lane_right", "value": <float 0.0-1.0>}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDrive Gym inference runner")
    parser.add_argument("--base-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("AUTODRIVE_EPISODES", "3")))
    parser.add_argument("--max-turns", type=int, default=MAX_STEPS)
    return parser.parse_args()


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.4f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_text}",
        flush=True,
    )


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
        end = text.rindex("}")
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
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


def build_prompt(observation: Any, step_index: int, task_id: str) -> str:
    stage = getattr(observation, "scenario_stage", "") or "approaching"
    hazard_dist = float(getattr(observation, "hazard_distance", 999.0) or 999.0)
    hazard_type = getattr(observation, "hazard_type", "") or task_id
    alerts = getattr(observation, "active_alerts", []) or []
    hint = getattr(observation, "hint", "") or ""

    if hint and "accelerate" in hint.lower():
        guidance = "Road is clear. Accelerate."
    elif stage in ("clearing", "cleared"):
        guidance = "Hazard is clearing. Resume safe forward progress."
    elif hazard_dist < 5.0:
        guidance = f"DANGER: {hazard_type} at {hazard_dist:.1f}m. Brake immediately."
    elif hazard_dist <= 12.0:
        guidance = f"CAUTION: {hazard_type} at {hazard_dist:.1f}m. Brake or wait."
    else:
        guidance = "Path mostly clear. Accelerate smoothly."

    payload = {
        "task_id": task_id,
        "step_index": step_index,
        "scenario_stage": stage,
        "hazard_distance": round(hazard_dist, 2),
        "hazard_type": hazard_type,
        "active_alerts": alerts,
        "hint": hint,
        "command_output": getattr(observation, "command_output", "") or "",
        "scene_summary": getattr(observation, "scene_summary", "") or "",
        "ego_state": getattr(observation, "ego_state", {}) or {},
        "sensor_data": getattr(observation, "sensor_data", {}) or {},
        "environment": getattr(observation, "environment", {}) or {},
        "guidance": guidance,
    }
    return json.dumps(payload, indent=2)


def call_llm(client: OpenAI, observation: Any, step_index: int, task_id: str) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation, step_index, task_id)},
        ],
        temperature=0.2,
        max_tokens=120,
    )
    content = completion.choices[0].message.content or ""
    parsed = parse_json_response(content)
    if parsed is None:
        raise RuntimeError(f"Model returned invalid JSON: {content[:200]}")
    normalized = normalize_action(parsed)
    if normalized is None:
        raise RuntimeError(f"Model response missing action: {content[:200]}")
    return normalized


def fallback_action(agent: ModularBaselineAgent, observation: Any) -> Dict[str, Any]:
    payload = agent.act(
        {
            "sensor_data": getattr(observation, "sensor_data", {}) or {},
            "ego_state": getattr(observation, "ego_state", {}) or {},
            "environment": getattr(observation, "environment", {}) or {},
            "active_alerts": getattr(observation, "active_alerts", []) or [],
            "hint": getattr(observation, "hint", "") or "",
            "scenario_stage": getattr(observation, "scenario_stage", "") or "",
            "hazard_distance": getattr(observation, "hazard_distance", 999.0),
            "hazard_type": getattr(observation, "hazard_type", "") or "",
        }
    )
    return {
        "action": str(payload.get("action", "wait")),
        "value": float(payload.get("value", 0.0)),
    }


def format_action(action_name: str, value: float) -> str:
    if action_name == "wait":
        return "wait"
    return f"{action_name}({value:.2f})"


def normalized_score(observation: Any, resolved: bool) -> float:
    if observation is None:
        return _SCORE_LO

    stage_scores = getattr(observation, "stage_scores", {}) or {}
    validation = getattr(observation, "validation", {}) or {}

    decision = float(stage_scores.get("decision_score", 0.0))
    safety = float(stage_scores.get("safety_score", 0.0))
    efficiency = float(stage_scores.get("efficiency_score", 0.0))

    decision_norm = max(0.0, min((decision + 1.0) / 2.0, 1.0))
    safety_norm = max(0.0, min((safety + 1.0) / 2.0, 1.0))
    efficiency_norm = max(0.0, min((efficiency + 1.0) / 2.0, 1.0))

    score = 0.40 * decision_norm + 0.40 * safety_norm + 0.20 * efficiency_norm

    if validation.get("stuck"):
        score -= 0.30
    if validation.get("collision"):
        score -= 0.50
    if validation.get("near_miss"):
        score -= 0.10
    if validation.get("offroad"):
        score -= 0.15
    if not validation.get("signal_respected", True):
        score -= 0.15
    if validation.get("incident_cleared"):
        score += 0.05
    if validation.get("progress_restored"):
        score += 0.05

    if resolved:
        score = max(score, 0.95)

    return max(_SCORE_LO, min(_SCORE_HI, score))


def main() -> int:
    args = parse_args()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fallback = ModularBaselineAgent()

    env_ctx = None
    try:
        env_ctx = AutoDriveClient(base_url=args.base_url).sync()
        env = env_ctx.__enter__()

        for episode_index in range(args.episodes):
            task_id = TASK_IDS[episode_index % len(TASK_IDS)]
            rewards: List[float] = []
            success = False
            steps_taken = 0
            observation = None

            log_start(task_id)

            try:
                reset_result = env.reset(task_id=task_id)
                observation = reset_result.observation

                for step_index in range(1, args.max_turns + 1):
                    llm_error = None
                    try:
                        action_dict = call_llm(llm_client, observation, step_index, task_id)
                    except Exception as exc:
                        llm_error = str(exc).replace("\n", " ")[:200]
                        action_dict = fallback_action(fallback, observation)

                    action = AutoDriveAction(
                        action=action_dict["action"],
                        value=float(action_dict["value"]),
                    )
                    step_result = env.step(action)
                    observation = step_result.observation
                    reward = float(step_result.reward or 0.0)
                    done = bool(step_result.done)
                    rewards.append(reward)
                    steps_taken = step_index

                    log_step(
                        step=step_index,
                        action=format_action(action.action, action.value),
                        reward=reward,
                        done=done,
                        error=llm_error,
                    )

                    if done:
                        resolution = getattr(observation, "resolution", {}) or {}
                        success = bool(resolution.get("verified"))
                        break

                score = normalized_score(observation, success)
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            except Exception:
                log_end(success=False, steps=steps_taken, score=_SCORE_LO, rewards=rewards)
    finally:
        if env_ctx is not None:
            try:
                env_ctx.__exit__(None, None, None)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
