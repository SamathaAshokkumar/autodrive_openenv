#!/usr/bin/env python3
"""Stable judge-facing evaluation script for AutoDrive Gym."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autodrive_env import AutoDriveAction, AutoDriveClient
from autodrive_env.agent_baseline import ModularBaselineAgent
from autodrive_env.server.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous driving agent operating in Indian road conditions.

Operate in 3 phases:

PHASE 1: EMERGENCY
- Immediate danger exists.
- Strong brake or one minimal evasive action is allowed.

PHASE 2: STABILIZE
- Hazard is still present but controlled.
- Slow down, wait briefly, or make a small steering correction.

PHASE 3: RECOVER
- Danger has reduced or cleared.
- Resume safe forward progress.

Rules:
- Avoid collisions at all costs.
- Do not stay in PHASE 1 forever.
- After 2-3 brake actions, reassess whether the scene has moved to STABILIZE or RECOVER.
- Repeating the same action without progress is a failure.
- Prefer smooth, safe recovery over panic steering.

Return ONLY valid JSON:
{
  "action": "accelerate|brake|steer_left|steer_right|horn|wait|change_lane_left|change_lane_right",
  "value": float
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stable AutoDrive inference against OpenEnv.")
    parser.add_argument("--base-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output", default="inference_report.json")
    parser.add_argument("--force-baseline", action="store_true")
    return parser.parse_args()


def parse_submission(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action", "")).strip().lower()
    if not action:
        return None
    try:
        value = float(payload.get("value", 0.0))
    except Exception:
        value = 0.0
    reasoning = str(payload.get("reasoning", "")).strip()
    return {"action": action, "value": value, "reasoning": reasoning}


def summarize_feedback(metadata: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    validation = metadata.get("validation", {})
    if validation.get("collision"):
        notes.append("collision")
    if validation.get("near_miss"):
        notes.append("near_miss")
    if validation.get("offroad"):
        notes.append("offroad")
    if not validation.get("signal_respected", True):
        notes.append("signal_violation")
    resolution = metadata.get("resolution", {})
    if resolution.get("verified") is False and resolution.get("reason") and resolution.get("reason") != "episode in progress":
        notes.append(f"resolution_gap:{resolution['reason']}")
    return notes


def summarize_inference(trace: Dict[str, Any], scenario_type: str) -> List[str]:
    notes: List[str] = []
    risk = trace.get("risk_summary", {}) or {}
    hazard_type = risk.get("hazard_type") or ""
    hazard_status = risk.get("hazard_status") or ""
    phase = trace.get("phase", "unknown")
    if hazard_type:
        notes.append(f"hazard={hazard_type}")
    if hazard_status:
        notes.append(f"status={hazard_status}")
    notes.append(f"phase={phase}")
    guardrail = (trace.get("guardrail_note", "") or "").strip()
    if guardrail:
        notes.append(guardrail)
    reasoning = (trace.get("reasoning", "") or "").strip()
    if reasoning:
        notes.append(f"reason={reasoning[:120]}")
    return notes


def observation_bundle(observation) -> Dict[str, Any]:
    metadata = getattr(observation, "metadata", {}) or {}
    return {
        "scenario_type": getattr(observation, "scenario_type", "") or metadata.get("scenario", {}).get("type", "unknown"),
        "judge_persona": getattr(observation, "judge_persona", "") or metadata.get("judge_persona", ""),
        "stage_scores": getattr(observation, "stage_scores", {}) or metadata.get("stage_scores", {}) or {},
        "validation": getattr(observation, "validation", {}) or metadata.get("validation", {}) or {},
        "resolution": getattr(observation, "resolution", {}) or metadata.get("resolution", {}) or {},
        "metadata": metadata,
    }


def format_action(submission: Dict[str, Any]) -> str:
    action = submission.get("action", "wait")
    value = float(submission.get("value", 0.0))
    if action in {"wait", "horn"}:
        return action if action == "horn" else "wait"
    return f"{action}: {value:.2f}"


def compact_history(step_trace: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    recent = step_trace[-limit:]
    history: List[Dict[str, Any]] = []
    for step in recent:
        history.append(
            {
                "step": step["step"],
                "action": step["submitted_action"].get("action", "wait"),
                "value": round(float(step["submitted_action"].get("value", 0.0)), 2),
                "reward": round(float(step["reward"]), 2),
                "notes": step.get("notes", []),
            }
        )
    return history


def repeated_action_count(step_trace: List[Dict[str, Any]], submission: Dict[str, Any]) -> int:
    count = 0
    action = submission.get("action", "wait")
    value = round(float(submission.get("value", 0.0)), 2)
    for step in reversed(step_trace):
        prev = step.get("submitted_action", {})
        prev_action = prev.get("action", "wait")
        prev_value = round(float(prev.get("value", 0.0)), 2)
        if prev_action == action and prev_value == value:
            count += 1
        else:
            break
    return count


def override_stalled_action(raw_obs: Dict[str, Any], submission: Dict[str, Any], step_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    repeated = repeated_action_count(step_trace, submission)
    if repeated < 2:
        return submission

    objects = raw_obs.get("sensor_data", {}).get("objects", []) or []
    ego_state = raw_obs.get("ego_state", {}) or {}
    speed = float(ego_state.get("speed", 0.0))
    nearest = min((float(obj.get("distance", 999.0)) for obj in objects), default=999.0)
    crossing_now = any(bool(obj.get("crossing")) and float(obj.get("distance", 999.0)) < 7.0 for obj in objects)
    intrusive_actor = any(
        str(obj.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(obj.get("distance", 999.0)) < 6.0
        for obj in objects
    )

    # If the model is freezing on the same action after the hazard has opened up,
    # nudge it toward controlled forward progress.
    if submission.get("action") in {"wait", "brake"} and nearest > 8.0 and not crossing_now and not intrusive_actor:
        return {"action": "accelerate", "value": 0.25 if speed < 4.0 else 0.15}
    return submission


def recent_same_action_count(step_trace: List[Dict[str, Any]], action_name: str) -> int:
    count = 0
    for step in reversed(step_trace):
        if step.get("submitted_action", {}).get("action") == action_name:
            count += 1
        else:
            break
    return count


def classify_phase(raw_obs: Dict[str, Any], step_trace: List[Dict[str, Any]]) -> str:
    hazard_status = str(raw_obs.get("hazard_status", "")).lower()
    if hazard_status == "cleared":
        return "recover"
    if hazard_status == "clearing":
        return "stabilize"
    sensor = raw_obs.get("sensor_data", {}) or {}
    objects = sensor.get("objects", []) or []
    nearest = min((float(obj.get("distance", 999.0)) for obj in objects), default=999.0)
    crossing_close = any(bool(obj.get("crossing")) and float(obj.get("distance", 999.0)) < 7.0 for obj in objects)
    intrusive_close = any(
        str(obj.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(obj.get("distance", 999.0)) < 6.5
        for obj in objects
    )
    repeated_brakes = recent_same_action_count(step_trace, "brake")

    if nearest < 4.5 or crossing_close or intrusive_close:
        return "emergency"
    if nearest < 8.5 or repeated_brakes > 0:
        return "stabilize"
    return "recover"


def build_risk_summary(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    sensor = raw_obs.get("sensor_data", {}) or {}
    objects = sensor.get("objects", []) or []
    nearest = min((float(obj.get("distance", 999.0)) for obj in objects), default=999.0)
    return {
        "hazard_type": raw_obs.get("hazard_type", ""),
        "hazard_status": raw_obs.get("hazard_status", ""),
        "hazard_distance": raw_obs.get("hazard_distance", nearest),
        "nearest_object_distance": round(nearest, 2),
        "crossing_hazard": any(bool(obj.get("crossing")) and float(obj.get("distance", 999.0)) < 10.0 for obj in objects),
        "intrusive_vehicle": any(
            str(obj.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
            for obj in objects
        ),
        "lane_info": sensor.get("lane_info", "clear"),
        "traffic_signal": sensor.get("traffic_signal", "none"),
    }


@dataclass
class EpisodeMemory:
    scenario_type: str = ""
    task_brief: str = ""
    phase: str = "emergency"
    last_action: str = ""
    repeated_action_count: int = 0


def guardrail_reason(
    original: Dict[str, Any],
    final: Dict[str, Any],
) -> str:
    if original.get("action") == final.get("action") and round(float(original.get("value", 0.0)), 2) == round(float(final.get("value", 0.0)), 2):
        return ""
    return f"guardrail adjusted {original.get('action', 'wait')} -> {final.get('action', 'wait')}"


def apply_rule_guardrails(
    raw_obs: Dict[str, Any],
    submission: Dict[str, Any],
    step_trace: List[Dict[str, Any]],
    memory: EpisodeMemory,
) -> Dict[str, Any]:
    sensor = raw_obs.get("sensor_data", {}) or {}
    ego = raw_obs.get("ego_state", {}) or {}
    objects = sensor.get("objects", []) or []
    speed = float(ego.get("speed", 0.0))
    nearest = min((float(obj.get("distance", 999.0)) for obj in objects), default=999.0)
    crossing_close = any(bool(obj.get("crossing")) and float(obj.get("distance", 999.0)) < 6.5 for obj in objects)
    intrusive_close = any(
        str(obj.get("behavior", "")).lower() in {"cut_in", "blind_spot_merge", "zig_zag"}
        and float(obj.get("distance", 999.0)) < 6.0
        for obj in objects
    )
    red_signal = str(sensor.get("traffic_signal", "none")).lower() == "red"
    action = submission.get("action", "wait")
    value = float(submission.get("value", 0.0))

    if red_signal and speed > 1.0:
        return {"action": "brake", "value": max(0.7, value)}

    if memory.phase == "emergency":
        if action == "accelerate":
            return {"action": "brake", "value": 0.8 if nearest < 5.0 else 0.6}
        if nearest < 4.0 and action not in {"brake", "wait"}:
            return {"action": "brake", "value": 0.9}

    if action in {"steer_left", "steer_right", "change_lane_left", "change_lane_right"} and nearest < 4.5:
        return {"action": "brake", "value": 0.8}

    if action in {"wait", "brake"} and memory.repeated_action_count >= 2 and not crossing_close and not intrusive_close and nearest > 8.0:
        return {"action": "accelerate", "value": 0.25 if speed < 4.0 else 0.15}

    if memory.phase == "recover" and action in {"wait", "brake"} and nearest > 9.0:
        return {"action": "accelerate", "value": 0.3 if speed < 5.0 else 0.2}

    if len(step_trace) >= 2:
        recent_actions = [step["submitted_action"].get("action") for step in step_trace[-2:]] + [action]
        if recent_actions == ["steer_left", "steer_right", "steer_left"] or recent_actions == ["steer_right", "steer_left", "steer_right"]:
            return {"action": "brake", "value": 0.6}

    return submission


def improvement_delta(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(values[-1] - values[0], 3)


def build_report_notes(episodes: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    total = max(1, len(episodes))
    success_count = sum(1 for ep in episodes if (ep.get("resolution", {}) or {}).get("verified"))
    success_rate = success_count / total
    if success_rate < 0.5:
        notes.append("Low success rate indicates weak hazard completion or recovery behavior.")

    stuck_count = 0
    guardrail_count = 0
    accelerate_guardrail_count = 0
    for episode in episodes:
        for step in episode.get("step_trace", []):
            for note in step.get("notes", []):
                if "vehicle got stuck" in note:
                    stuck_count += 1
            trace = step.get("decision_trace", {}) or {}
            guardrail_note = (trace.get("guardrail_note", "") or "").strip()
            if guardrail_note:
                guardrail_count += 1
                if "-> accelerate" in guardrail_note:
                    accelerate_guardrail_count += 1

    if stuck_count:
        notes.append("Agent still gets stuck in some scenarios after slowing or steering; recovery logic needs improvement.")
    if guardrail_count >= total:
        notes.append("Frequent guardrail interventions suggest the raw policy still relies heavily on rule corrections.")
    if accelerate_guardrail_count:
        notes.append("The policy often hesitates after hazards clear, so guardrails are nudging progress restoration.")
    if not notes:
        notes.append("Agent handled the sampled scenarios without major report-level concerns in this run.")
    return notes


@dataclass
class WeaknessTracker:
    scenario_stage_history: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    weakness_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    weakness_to_types: Dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def record(self, scenario_type: str, stage_scores: Dict[str, float], metadata: Dict[str, Any]):
        for stage_name, score in stage_scores.items():
            self.scenario_stage_history[scenario_type][stage_name].append(float(score))
        for note in summarize_feedback(metadata):
            self.weakness_counts[note] += 1
            self.weakness_to_types[note].add(scenario_type)

    def improvement_report(self) -> Dict[str, Any]:
        by_type: Dict[str, Any] = {}
        for scenario_type, stage_map in self.scenario_stage_history.items():
            by_type[scenario_type] = {}
            for stage_name, values in stage_map.items():
                by_type[scenario_type][stage_name] = {
                    "attempts": len(values),
                    "first": round(values[0], 3),
                    "latest": round(values[-1], 3),
                    "best": round(max(values), 3),
                    "delta": improvement_delta(values),
                }
        ranked_weaknesses = [
            {"weakness": weakness, "count": count, "scenario_types": sorted(self.weakness_to_types.get(weakness, []))}
            for weakness, count in sorted(self.weakness_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        return {"by_scenario_type": by_type, "ranked_weaknesses": ranked_weaknesses}


class HybridDrivingAgent:
    def __init__(self):
        self.client = LLMClient()
        provider = getattr(self.client, "provider", "mock")
        self.provider = provider
        if provider == "openai":
            self.model_name = getattr(self.client, "openai_model", "openai")
        elif provider == "groq":
            self.model_name = getattr(self.client, "groq_model", "groq")
        elif provider == "hf":
            self.model_name = getattr(self.client, "hf_model", "huggingface")
        else:
            self.model_name = "rule-based-baseline"
        self.has_remote_model = provider != "mock"
        self.fallback = ModularBaselineAgent()
        self.memory = EpisodeMemory()
        self.last_decision_trace: Dict[str, Any] = {}

    def begin_episode(self, scenario_type: str, task_brief: str) -> None:
        self.memory = EpisodeMemory(scenario_type=scenario_type, task_brief=task_brief, phase="emergency")
        self.last_decision_trace = {}

    def act(self, raw_obs: Dict[str, Any], reminders: List[str], step_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.memory.phase = classify_phase(raw_obs, step_trace)
        if step_trace:
            last_action = step_trace[-1]["submitted_action"].get("action", "")
            if last_action == self.memory.last_action:
                self.memory.repeated_action_count += 1
            else:
                self.memory.repeated_action_count = 0
            self.memory.last_action = last_action

        if not self.has_remote_model:
            baseline_action = self.fallback.act(raw_obs)
            final_action = apply_rule_guardrails(raw_obs, baseline_action, step_trace, self.memory)
            self.last_decision_trace = {
                "phase": self.memory.phase,
                "risk_summary": build_risk_summary(raw_obs),
                "action_source": "baseline",
                "model_action": baseline_action,
                "final_action": final_action,
                "guardrail_note": guardrail_reason(baseline_action, final_action),
                "reasoning": "baseline heuristic action",
            }
            return final_action
        reminder_text = "\n".join(f"- {item}" for item in reminders) if reminders else "- none yet"
        recent_history = compact_history(step_trace)
        repeated_warning = ""
        if len(recent_history) >= 3:
            last_actions = [item["action"] for item in recent_history[-3:]]
            if len(set(last_actions)) == 1:
                repeated_warning = "WARNING: You are repeating actions. Change strategy if the hazard is reducing.\n\n"
        prompt = (
            "OBJECTIVE:\n"
            "- Avoid collisions, near misses, offroad behavior, and signal violations.\n"
            "- Use phase-based reasoning: EMERGENCY -> STABILIZE -> RECOVER.\n"
            "- Do not freeze with repeated brake/wait actions once the immediate hazard has cleared.\n"
            "- Make safe forward progress when possible.\n\n"
            f"EPISODE_CONTEXT:\n- scenario_type: {self.memory.scenario_type}\n- task_brief: {self.memory.task_brief}\n- current_phase: {self.memory.phase}\n\n"
            f"RISK_SUMMARY:\n{json.dumps(build_risk_summary(raw_obs), indent=2)}\n\n"
            f"WEAKNESS_REMINDERS:\n{reminder_text}\n\n"
            f"RECENT_ACTION_HISTORY:\n{json.dumps(recent_history, indent=2)}\n\n"
            f"{repeated_warning}"
            f"RAW_OBSERVATION:\n{json.dumps(raw_obs, indent=2)}\n\n"
            "Pick exactly one next action. Return JSON only."
        )
        response = self.client.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=128)
        submission = parse_submission(response)
        if submission is not None:
            final_action = apply_rule_guardrails(raw_obs, submission, step_trace, self.memory)
            self.last_decision_trace = {
                "phase": self.memory.phase,
                "risk_summary": build_risk_summary(raw_obs),
                "action_source": "remote_llm",
                "model_action": submission,
                "final_action": final_action,
                "guardrail_note": guardrail_reason(submission, final_action),
                "reasoning": submission.get("reasoning", ""),
            }
            return final_action
        logger.warning("Remote response invalid; falling back to baseline for this step.")
        baseline_action = self.fallback.act(raw_obs)
        final_action = apply_rule_guardrails(raw_obs, baseline_action, step_trace, self.memory)
        self.last_decision_trace = {
            "phase": self.memory.phase,
            "risk_summary": build_risk_summary(raw_obs),
            "action_source": "baseline_after_invalid_remote",
            "model_action": baseline_action,
            "final_action": final_action,
            "guardrail_note": guardrail_reason(baseline_action, final_action),
            "reasoning": "remote response invalid, baseline heuristic action",
        }
        return final_action


def build_action(submission: Dict[str, Any]) -> AutoDriveAction:
    return AutoDriveAction(action=submission.get("action", "wait"), value=float(submission.get("value", 0.0)))


def run_episode(env, agent, max_turns: int, tracker: WeaknessTracker) -> Dict[str, Any]:
    result = env.reset()
    observation = result.observation
    episode_reward = 0.0
    steps: List[Dict[str, Any]] = []
    reminders: List[str] = []
    scenario_type = "unknown"
    initial_bundle = observation_bundle(observation)
    scenario_type = initial_bundle["scenario_type"] or "unknown"
    initial_brief = getattr(observation, "command_output", "") or getattr(observation, "event_log", "")
    if hasattr(agent, "begin_episode"):
        agent.begin_episode(scenario_type, initial_brief)

    for step_index in range(max_turns):
        bundle = observation_bundle(observation)
        scenario_type = bundle["scenario_type"] or scenario_type
        raw_obs = {
            "command_output": getattr(observation, "command_output", ""),
            "scene_summary": getattr(observation, "scene_summary", ""),
            "sensor_data": getattr(observation, "sensor_data", {}) or {},
            "ego_state": getattr(observation, "ego_state", {}) or {},
            "environment": getattr(observation, "environment", {}) or {},
            "hazard_type": getattr(observation, "hazard_type", ""),
            "hazard_distance": getattr(observation, "hazard_distance", 999.0),
            "hazard_status": getattr(observation, "hazard_status", ""),
            "scenario_stage": getattr(observation, "scenario_stage", ""),
            "event_log": getattr(observation, "event_log", ""),
            "steps_taken": getattr(observation, "steps_taken", 0),
            "max_steps": getattr(observation, "max_steps", max_turns),
            "hint": getattr(observation, "hint", ""),
            "vehicle_profile": getattr(observation, "vehicle_profile", {}) or {},
        }
        submission = agent.act(raw_obs, reminders, steps)
        submission = override_stalled_action(raw_obs, submission, steps)
        result = env.step(build_action(submission))
        observation = result.observation
        bundle = observation_bundle(observation)
        stage_scores = bundle["stage_scores"]
        notes = summarize_feedback({
            "validation": bundle["validation"],
            "resolution": bundle["resolution"],
        })
        reminders = notes[:3]
        steps.append(
            {
                "step": step_index + 1,
                "submitted_action": submission,
                "decision_trace": getattr(agent, "last_decision_trace", {}),
                "sudden_alert": next(
                    (
                        str(alert).strip()
                        for alert in (getattr(observation, "active_alerts", []) or [])
                        if str(alert).strip().lower().startswith("sudden alert:")
                    ),
                    "",
                ),
                "judge_persona": bundle["judge_persona"],
                "stage_scores": {
                    "decision": round(float(stage_scores.get("decision_score", 0.0)), 3),
                    "safety": round(float(stage_scores.get("safety_score", 0.0)), 3),
                    "efficiency": round(float(stage_scores.get("efficiency_score", 0.0)), 3),
                },
                "reward": round(float(result.reward or 0.0), 3),
                "notes": notes,
                "inference_notes": summarize_inference(getattr(agent, "last_decision_trace", {}) or {}, scenario_type),
            }
        )
        episode_reward += float(result.reward or 0.0)
        if result.done:
            break

    final_bundle = observation_bundle(observation)
    final_stage_scores = final_bundle["stage_scores"]
    scenario_type = final_bundle["scenario_type"] or scenario_type
    tracker.record(
        scenario_type,
        final_stage_scores,
        {"validation": final_bundle["validation"], "resolution": final_bundle["resolution"]},
    )
    return {
        "task_brief": initial_brief,
        "scenario_type": scenario_type,
        "steps_taken": len(steps),
        "total_reward": round(episode_reward, 3),
        "judge_persona": final_bundle["judge_persona"],
        "final_stage_grades": {
            "decision": round(float(final_stage_scores.get("decision_score", 0.0)), 3),
            "safety": round(float(final_stage_scores.get("safety_score", 0.0)), 3),
            "efficiency": round(float(final_stage_scores.get("efficiency_score", 0.0)), 3),
        },
        "resolution": final_bundle["resolution"] or {},
        "step_trace": steps,
    }


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    tracker = WeaknessTracker()
    agent = HybridDrivingAgent()
    logger.info("Connecting to %s", args.base_url)
    episodes: List[Dict[str, Any]] = []
    with AutoDriveClient(base_url=args.base_url).sync() as env:
        for episode_index in range(args.episodes):
            logger.info("Episode %s/%s", episode_index + 1, args.episodes)
            episode = run_episode(env, agent, args.max_turns, tracker)
            episodes.append(episode)
            print()
            print(f"Episode {episode_index + 1}: {episode['scenario_type']} ({episode['judge_persona'] or 'unknown'} judge)")
            print("-" * 72)
            print(f"Task: {episode['task_brief']}")
            for step in episode["step_trace"]:
                action_text = format_action(step["submitted_action"])
                reward_text = f"{step['reward']:+.2f}"
                notes = f"  <- {', '.join(step['notes'])}" if step["notes"] else ""
                print(f"Step {step['step']}: {action_text:<24} reward={reward_text}{notes}")
                trace = step.get("decision_trace", {}) or {}
                if trace:
                    phase = trace.get("phase", "?")
                    risk = trace.get("risk_summary", {}) or {}
                    nearest = risk.get("nearest_object_distance", "?")
                    guardrail = trace.get("guardrail_note", "")
                    source = trace.get("action_source", "unknown")
                    extra = f" | {guardrail}" if guardrail else ""
                    print(f"         trace: phase={phase} nearest={nearest} source={source}{extra}")
                    reasoning = (trace.get("reasoning", "") or "").strip()
                    if reasoning:
                        print(f"         reason: {reasoning[:160]}")
                sudden_alert = (step.get("sudden_alert", "") or "").strip()
                if sudden_alert:
                    print(f"         sudden alert: {sudden_alert.removeprefix('Sudden alert: ').strip()}")
            resolution = episode.get("resolution", {}) or {}
            verdict = "RESOLVED" if resolution.get("verified") else "UNRESOLVED"
            reason = resolution.get("reason", "")
            print(f"-> Judge verification: {verdict}" + (f" - {reason}" if reason else ""))
    totals = [ep["total_reward"] for ep in episodes]
    decision_scores = [ep["final_stage_grades"]["decision"] for ep in episodes]
    safety_scores = [ep["final_stage_grades"]["safety"] for ep in episodes]
    efficiency_scores = [ep["final_stage_grades"]["efficiency"] for ep in episodes]
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment_url": args.base_url,
        "model": {
            "name": agent.model_name,
            "source": agent.provider if agent.has_remote_model and not args.force_baseline else "modular_baseline",
            "token_configured": bool(
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GROQ_API_KEY")
            ),
        },
        "tasks_executed": episodes,
        "summary": {
            "episodes": len(episodes),
            "average_reward": round(sum(totals) / max(1, len(totals)), 3),
            "average_stage_grades": {
                "decision": round(sum(decision_scores) / max(1, len(decision_scores)), 3),
                "safety": round(sum(safety_scores) / max(1, len(safety_scores)), 3),
                "efficiency": round(sum(efficiency_scores) / max(1, len(efficiency_scores)), 3),
            },
        },
        "improvement_tracking": tracker.improvement_report(),
        "notes": build_report_notes(episodes),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    weaknesses = report["improvement_tracking"]["ranked_weaknesses"][:3]
    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)
    success_count = sum(1 for ep in episodes if (ep.get("resolution", {}) or {}).get("verified"))
    success_rate = 100.0 * success_count / max(1, len(episodes))
    print(f"Success rate: {success_count}/{len(episodes)} ({success_rate:.1f}%)")
    print(f"Average reward: {report['summary']['average_reward']:.3f}")
    print(
        "Average stage grades: "
        + ", ".join(
            f"{name}={value:.2f}" for name, value in report["summary"]["average_stage_grades"].items()
        )
    )
    if weaknesses:
        weakness_text = "; ".join(
            f"{item['weakness']} ({item['count']})" for item in weaknesses
        )
        print(f"Agent struggles: {weakness_text}")
    else:
        print("Agent struggles: none surfaced in this run")
    print(f"Detailed report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
