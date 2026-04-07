#!/usr/bin/env python3
"""Hackathon-compliant inference entrypoint for AutoDrive Gym.

This script is intentionally separate from eval.py.
It follows the requested contract:
- root-level file named `inference.py`
- uses OpenAI client for all LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- emits structured stdout with [START], [STEP], and [END]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from autodrive_env import AutoDriveAction, AutoDriveClient
from autodrive_env.agent_baseline import ModularBaselineAgent


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
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=20)
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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
    return {"action": action, "value": max(0.0, min(1.0, value))}


def build_prompt(observation: Any, step_index: int) -> str:
    raw_obs = {
        "command_output": getattr(observation, "command_output", ""),
        "scene_summary": getattr(observation, "scene_summary", ""),
        "sensor_data": getattr(observation, "sensor_data", {}) or {},
        "ego_state": getattr(observation, "ego_state", {}) or {},
        "environment": getattr(observation, "environment", {}) or {},
        "event_log": getattr(observation, "event_log", ""),
        "steps_taken": getattr(observation, "steps_taken", 0),
        "max_steps": getattr(observation, "max_steps", 20),
        "scenario_type": getattr(observation, "scenario_type", ""),
    }
    return (
        f"STEP_INDEX: {step_index}\n"
        f"SCENARIO_TYPE: {raw_obs['scenario_type']}\n"
        f"OBSERVATION:\n{json.dumps(raw_obs, indent=2)}\n\n"
        "Choose the safest next action. Return JSON only."
    )


def call_llm(client: OpenAI, model_name: str, observation: Any, step_index: int) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=model_name,
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


def format_value(value: float) -> str:
    return f"{value:.2f}"


def main() -> int:
    args = parse_args()

    api_base_url = require_env("API_BASE_URL")
    model_name = require_env("MODEL_NAME")
    api_key = require_env("HF_TOKEN")

    llm = OpenAI(base_url=api_base_url, api_key=api_key)
    fallback = ModularBaselineAgent()

    with AutoDriveClient(base_url=args.base_url).sync() as env:
        for episode_index in range(1, args.episodes + 1):
            reset_result = env.reset()
            observation = reset_result.observation
            scenario_type = getattr(observation, "scenario_type", "unknown")
            task = getattr(observation, "command_output", "") or getattr(observation, "event_log", "")
            print(f"[START] episode={episode_index} scenario_type={scenario_type} steps_taken=0 max_steps={getattr(observation, 'max_steps', args.max_turns)} task={json.dumps(task)}", flush=True)

            total_reward = 0.0
            final_done = False
            final_reason = ""
            final_resolution = "UNRESOLVED"

            for step_index in range(1, args.max_turns + 1):
                try:
                    action_dict = call_llm(llm, model_name, observation, step_index)
                except Exception:
                    action_dict = fallback.act(
                        {
                            "sensor_data": getattr(observation, "sensor_data", {}) or {},
                            "ego_state": getattr(observation, "ego_state", {}) or {},
                            "environment": getattr(observation, "environment", {}) or {},
                        }
                    )

                action = AutoDriveAction(action=action_dict["action"], value=float(action_dict["value"]))
                step_result = env.step(action)
                observation = step_result.observation
                reward = float(step_result.reward or 0.0)
                total_reward += reward
                done = bool(step_result.done)
                sudden_alerts = getattr(observation, "active_alerts", []) or []
                sudden_alert = next((str(item).strip() for item in sudden_alerts if str(item).strip()), "")

                print(
                    f"[STEP] episode={episode_index} step={step_index} action={action.action} value={format_value(action.value)} "
                    f"reward={reward:.3f} done={str(done).lower()} sudden_alert={json.dumps(sudden_alert)}",
                    flush=True,
                )

                if done:
                    resolution = getattr(observation, "resolution", {}) or {}
                    final_done = True
                    final_reason = str(resolution.get("reason", "") or "")
                    final_resolution = "RESOLVED" if resolution.get("verified") else "UNRESOLVED"
                    break

            state = env.state()
            print(
                f"[END] episode={episode_index} resolution={final_resolution} total_reward={total_reward:.3f} "
                f"step_count={getattr(state, 'step_count', 0)} incident_type={getattr(state, 'incident_type', '')} "
                f"done={str(final_done).lower()} reason={json.dumps(final_reason)}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
