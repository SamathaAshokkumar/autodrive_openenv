#!/usr/bin/env python3
"""Hackathon-compliant inference entrypoint for AutoDrive Gym."""

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
TASK_NAME = os.getenv("AUTODRIVE_TASK", "autodrive")
MAX_STEPS = int(os.getenv("AUTODRIVE_MAX_STEPS", "20"))

SYSTEM_PROMPT = """You are an autonomous driving agent for Indian road conditions.

Priorities:
1. Safety
2. Smooth recovery
3. Forward progress

Return ONLY JSON:
{
  "action": "accelerate|brake|steer_left|steer_right|horn|wait|change_lane_left|change_lane_right",
  "value": <float 0 to 1>
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hackathon inference runner for AutoDrive Gym")
    parser.add_argument("--base-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--max-turns", type=int, default=MAX_STEPS)
    return parser.parse_args()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
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
    value = max(0.0, min(1.0, value))
    return {"action": action, "value": value}


def build_prompt(observation: Any, step_index: int) -> str:
    raw_obs = {
        "command_output": getattr(observation, "command_output", ""),
        "scene_summary": getattr(observation, "scene_summary", ""),
        "sensor_data": getattr(observation, "sensor_data", {}) or {},
        "ego_state": getattr(observation, "ego_state", {}) or {},
        "environment": getattr(observation, "environment", {}) or {},
        "event_log": getattr(observation, "event_log", ""),
        "steps_taken": getattr(observation, "steps_taken", 0),
        "max_steps": getattr(observation, "max_steps", MAX_STEPS),
        "scenario_type": getattr(observation, "scenario_type", ""),
    }
    return (
        f"STEP_INDEX: {step_index}\n"
        f"SCENARIO_TYPE: {raw_obs['scenario_type']}\n"
        f"OBSERVATION:\n{json.dumps(raw_obs, indent=2)}\n\n"
        "Choose the safest next action. Return JSON only."
    )


def call_llm(client: OpenAI, observation: Any, step_index: int) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation, step_index)},
        ],
        temperature=0.2,
        max_tokens=128,
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
        }
    )
    return {"action": str(payload.get("action", "wait")), "value": float(payload.get("value", 0.0))}


def score_from_result(rewards: List[float], observation: Any) -> float:
    resolution = getattr(observation, "resolution", {}) or {}
    if resolution.get("verified"):
        return 1.0
    if not rewards:
        return 0.0
    reward_sum = sum(rewards)
    reward_cap = max(len(rewards) * 10.0, 1.0)
    return max(0.0, min(reward_sum / reward_cap, 1.0))


def main() -> int:
    args = parse_args()

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fallback = ModularBaselineAgent()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env_ctx = None
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env_ctx = AutoDriveClient(base_url=args.base_url).sync()
        env = env_ctx.__enter__()

        reset_result = env.reset()
        observation = reset_result.observation

        for step in range(1, args.max_turns + 1):
            if getattr(reset_result, "done", False):
                break

            error_msg = None
            try:
                action_dict = call_llm(llm_client, observation, step)
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")[:200]
                action_dict = fallback_action(fallback, observation)

            action = AutoDriveAction(action=action_dict["action"], value=float(action_dict["value"]))

            try:
                step_result = env.step(action)
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")[:200]
                log_step(step=step, action=action.action, reward=0.0, done=True, error=error_msg)
                steps_taken = step
                break

            observation = step_result.observation
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)
            rewards.append(reward)
            steps_taken = step

            action_str = action.action
            if action.value:
                action_str = f"{action.action}:{action.value:.2f}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        if env is not None:
            try:
                state = env.state()
                last_observation = observation if "observation" in locals() else None
                score = score_from_result(rewards, last_observation) if last_observation is not None else 0.0
                resolution = getattr(last_observation, "resolution", {}) if last_observation is not None else {}
                success = bool(resolution.get("verified")) or score >= 0.5
                if getattr(state, "is_resolved", False):
                    success = True
                    score = max(score, 1.0)
            except Exception:
                score = max(0.0, min(sum(rewards) / max(len(rewards) * 10.0, 1.0), 1.0)) if rewards else 0.0
                success = score >= 0.5
    except Exception:
        success = False
        score = 0.0
    finally:
        if env_ctx is not None:
            try:
                env_ctx.__exit__(None, None, None)
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
