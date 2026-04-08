#!/usr/bin/env python3
"""Hackathon inference runner with eval-style readable logs."""

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
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("AUTODRIVE_EPISODES", "1")))
    parser.add_argument("--max-turns", type=int, default=MAX_STEPS)
    return parser.parse_args()


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


def format_action(action_name: str, value: float) -> str:
    if action_name == "wait":
        return "wait"
    return f"{action_name}: {value:.2f}"


def normalized_score(rewards: List[float], observation: Any, resolved: bool) -> float:
    if resolved:
        return 1.0
    if observation is None:
        return 0.0

    stage_scores = getattr(observation, "stage_scores", {}) or {}
    validation = getattr(observation, "validation", {}) or {}

    decision = float(stage_scores.get("decision_score", 0.0))
    safety = float(stage_scores.get("safety_score", 0.0))
    efficiency = float(stage_scores.get("efficiency_score", 0.0))

    decision_norm = max(0.0, min((decision + 1.0) / 2.0, 1.0))
    safety_norm = max(0.0, min((safety + 1.0) / 2.0, 1.0))
    efficiency_norm = max(0.0, min((efficiency + 1.0) / 2.0, 1.0))

    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        reward_norm = max(0.0, min((avg_reward + 2.0) / 4.0, 1.0))
    else:
        reward_norm = 0.0

    score = (
        0.35 * decision_norm
        + 0.35 * safety_norm
        + 0.20 * efficiency_norm
        + 0.10 * reward_norm
    )

    if validation.get("stuck"):
        score -= 0.35
    if validation.get("collision"):
        score -= 0.60
    if validation.get("near_miss"):
        score -= 0.15
    if validation.get("offroad"):
        score -= 0.20
    if not validation.get("signal_respected", True):
        score -= 0.20

    if validation.get("incident_cleared"):
        score += 0.05
    if validation.get("progress_restored"):
        score += 0.05

    return max(0.0, min(score, 1.0))


def main() -> int:
    args = parse_args()

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fallback = ModularBaselineAgent()

    env_ctx = None
    try:
        env_ctx = AutoDriveClient(base_url=args.base_url).sync()
        env = env_ctx.__enter__()

        for episode_index in range(1, args.episodes + 1):
            rewards: List[float] = []
            resolved = False
            final_reason = ""
            scenario_type = "unknown"
            judge_persona = "unknown"
            task_brief = ""

            try:
                reset_result = env.reset()
                observation = reset_result.observation
                scenario_type = getattr(observation, "scenario_type", "unknown")
                judge_persona = getattr(observation, "judge_persona", "") or "unknown"
                task_brief = getattr(observation, "command_output", "") or getattr(observation, "event_log", "")

                print("[START]")
                print(f"Episode {episode_index}: {scenario_type} ({judge_persona} judge)")
                print("-" * 72)
                print(f"Task: {task_brief}")

                for step_index in range(1, args.max_turns + 1):
                    llm_error = None
                    try:
                        action_dict = call_llm(llm_client, observation, step_index)
                    except Exception as exc:
                        llm_error = str(exc).replace("\n", " ")[:200]
                        action_dict = fallback_action(fallback, observation)

                    action = AutoDriveAction(action=action_dict["action"], value=float(action_dict["value"]))

                    try:
                        step_result = env.step(action)
                    except Exception as exc:
                        llm_error = str(exc).replace("\n", " ")[:200]
                        print(f"[STEP] Step {step_index}: {format_action(action.action, action.value):<24} reward={0.0:+.2f}  <- {llm_error}")
                        break

                    observation = step_result.observation
                    reward = float(step_result.reward or 0.0)
                    rewards.append(reward)
                    notes: List[str] = []
                    validation = getattr(observation, "validation", {}) or {}
                    if validation.get("stuck"):
                        notes.append("resolution_gap:vehicle got stuck")
                    if validation.get("near_miss"):
                        notes.append("near_miss")
                    if validation.get("collision"):
                        notes.append("collision")
                    if llm_error:
                        notes.append(llm_error)
                    notes_text = f"  <- {', '.join(notes)}" if notes else ""

                    print(
                        f"[STEP] Step {step_index}: {format_action(action.action, action.value):<24} "
                        f"reward={reward:+.2f}{notes_text}"
                    )

                    stage = getattr(observation, "scenario_stage", "") or "?"
                    nearest = getattr(observation, "hazard_distance", None)
                    if nearest in (None, ""):
                        nearest = "?"
                    source = "remote_llm" if llm_error is None else "fallback_agent"
                    trace_extra = ""
                    if llm_error:
                        trace_extra = " | fallback after llm error"
                    print(f"         trace: phase={stage} nearest={nearest} source={source}{trace_extra}")

                    sudden_alerts = getattr(observation, "active_alerts", []) or []
                    sudden_alert = next((str(item).strip() for item in sudden_alerts if str(item).strip()), "")
                    if sudden_alert:
                        print(f"         sudden alert: {sudden_alert.removeprefix('Sudden alert: ').strip()}")

                    if bool(step_result.done):
                        resolution = getattr(observation, "resolution", {}) or {}
                        resolved = bool(resolution.get("verified"))
                        final_reason = str(resolution.get("reason", "") or "")
                        break

                score = normalized_score(rewards, observation, resolved)
                print("[END]")
                verdict = "RESOLVED" if resolved else "UNRESOLVED"
                suffix = f" - {final_reason}" if final_reason else ""
                print(f"-> Judge verification: {verdict}{suffix} | score={score:.2f}")
            except Exception as exc:
                print("[START]")
                print(f"Episode {episode_index}: {scenario_type} ({judge_persona} judge)")
                print("-" * 72)
                print(f"Task: {task_brief or 'failed to initialize task'}")
                print("[END]")
                print(f"-> Judge verification: UNRESOLVED - {str(exc).replace(chr(10), ' ')[:200]} | score=0.00")
    finally:
        if env_ctx is not None:
            try:
                env_ctx.__exit__(None, None, None)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
