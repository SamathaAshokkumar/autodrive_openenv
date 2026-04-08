"""AutoDrive Gym environment implementation with kube-style structure."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from ..models import AutoDriveAction, AutoDriveObservation, AutoDriveState
from .adversarial_designer import AdversarialDesigner
from .constants import MAX_STEPS
from .curriculum import CurriculumController
from .driving_backend import DrivingBackend
from .judge import LLMJudge
from .llm_client import LLMClient
from .scenario_generator import ScenarioGenerator

logger = logging.getLogger(__name__)

_NARRATOR_SYSTEM = """You are the onboard perception narrator for an autonomous vehicle in India.
Convert the raw sensor/ego/environment snapshot into ONE concise, vivid, action-oriented sentence (max 25 words).
Focus on the most dangerous or important element the driver should act on RIGHT NOW.
Return ONLY the sentence — no JSON, no markdown."""


class AutoDriveGymEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self.backend = DrivingBackend()
        self.curriculum = CurriculumController()
        self.llm = LLMClient()
        self.judge = LLMJudge(self.llm)
        self.generator = ScenarioGenerator(self.llm)
        self.designer = AdversarialDesigner(self.llm, self.backend, max_steps=MAX_STEPS)
        self.scenario = None
        self.history = []
        self._step_count = 0
        self.max_steps = MAX_STEPS
        self._state = AutoDriveState(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> AutoDriveObservation:
        self.backend.reset()
        difficulty = self.curriculum.get_difficulty()
        incident_type = self.curriculum.pick_fault_type()
        if incident_type == "adversarial":
            self.scenario = self.designer.design(self.curriculum.get_skill_profile(), difficulty)
        else:
            self.scenario = self.generator.generate(self.curriculum.get_skill_profile(), difficulty, fault_type_hint=incident_type).__dict__
        self.backend.inject_scenario(self.scenario)
        self.history = []
        self._step_count = 0
        self.max_steps = int(MAX_STEPS + 6 * difficulty)
        self._state = AutoDriveState(
            episode_id=str(uuid4()),
            step_count=0,
            incident_id=self.scenario.get("name", ""),
            difficulty=difficulty,
            incident_type=self.scenario.get("type", ""),
            root_cause=self.scenario.get("root_cause", ""),
            correct_fix=self.scenario.get("correct_fix_description", ""),
            judge_persona=self.curriculum.get_judge_persona(),
            curriculum_stats=self.curriculum.get_stats(),
        )
        obs = self.backend.build_observation(
            steps_taken=0,
            max_steps=self.max_steps,
            hint=self._hint(),
            metadata={"scenario": self.scenario, "judge_persona": self.curriculum.get_judge_persona()},
        )
        obs["scenario_type"] = self.scenario.get("type", "")
        obs["judge_persona"] = self.curriculum.get_judge_persona()
        obs["stage_scores"] = {}
        obs["validation"] = {}
        obs["resolution"] = {}
        return self._to_observation(obs, reward=0.0, done=False)

    def step(self, action: AutoDriveAction) -> AutoDriveObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        before = self.backend.build_observation(self._step_count - 1, self.max_steps, hint=self._hint())
        command_output = self.backend.execute(action.action, action.value)
        self.backend.update()
        validation = self.backend.programmatic_checks()
        result_state = {**validation, "command_output": command_output}
        persona = self.curriculum.get_judge_persona()
        judge_score, judge_feedback = self.judge.evaluate(before, {"action": action.action, "value": action.value}, result_state, self.scenario, self.history, persona)

        repeat_count = sum(1 for item in self.history if item.get("action") == action.action and abs(item.get("value", 0.0) - action.value) < 0.1)
        # keep repeat penalty small and negative (used as a mild efficiency factor)
        repeat_penalty = -0.2 * repeat_count if repeat_count else 0.0

        done = bool(
            validation["collision"]
            or validation["reached_goal"]
            or validation["progress_restored"]
            or validation["stuck"]
            or self._step_count >= self.max_steps
        )
        success = False
        resolution_bonus = 0.0
        resolution_reason = "episode in progress"
        if done:
            success, resolution_reason = self.judge.verify_resolution(self.scenario, self.history, before, validation)
            if success:
                resolution_bonus = 5.0 + 2.0 * (1.0 - self._step_count / max(self.max_steps, 1))

        reward = self._compute_reward(validation, judge_score, repeat_penalty, resolution_bonus, done, success)

        self.history.append({
            "step": self._step_count,
            "action": action.action,
            "value": action.value,
            "reward": reward,
            "judge_feedback": judge_feedback,
        })

        self._state.cumulative_reward += reward
        if done:
            self.curriculum.record(self.scenario.get("type", "unknown"), success, self._step_count, reward)
            self._state.is_resolved = success
        self._state.curriculum_stats = self.curriculum.get_stats()

        obs = self.backend.build_observation(
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            hint=judge_feedback if persona != "principal" else "",
            metadata={
                "scenario": self.scenario,
                "judge_score": judge_score,
                "judge_feedback": judge_feedback,
                "judge_persona": persona,
                "validation": validation,
                "resolution": {"verified": success, "reason": resolution_reason, "bonus": round(resolution_bonus, 3)},
                "stage_scores": {
                    "decision_score": round(judge_score, 3),
                    "safety_score": 1.0 if not validation["collision"] and validation.get("safe_distance") else 0.0,
                    "efficiency_score": round(max(0.0, min(1.0, 1.0 - 0.05 * self._step_count + repeat_penalty)), 3),
                },
            },
        )
        backend_event_log = str(obs.get("event_log", "") or "").strip()
        current_stage = self.backend.simulator.current_stage

        # LLM narrator: generate a vivid human-readable description of the scene
        narrative = self._narrate(obs, backend_event_log, current_stage)

        # Combine environment events with action log so the agent sees both
        if backend_event_log:
            obs["command_output"] = f"{backend_event_log} | {command_output}"
        else:
            obs["command_output"] = command_output
        obs["scene_summary"] = narrative  # override with LLM-generated description
        obs["event_log"] = backend_event_log
        # Surface both sudden alerts AND clearing events so the agent adapts
        if backend_event_log and backend_event_log.lower().startswith("sudden alert:"):
            obs["active_alerts"] = [backend_event_log]
        elif backend_event_log and current_stage in ("clearing", "cleared"):
            obs["active_alerts"] = [backend_event_log]
        else:
            obs["active_alerts"] = []
        # Override the hint when the hazard has cleared to prompt acceleration
        if current_stage in ("clearing", "cleared"):
            obs["hint"] = "Hazard has cleared. Accelerate NOW to restore forward progress."
        obs["scenario_type"] = self.scenario.get("type", "")
        obs["judge_persona"] = persona
        # Normalize stage scores into non-negative ranges to avoid exact -1/0/1 extremes
        def _norm_decision(x: float) -> float:
            eps = 1e-3
            v = max(-1.0, min(1.0, x))
            return round(max(eps, min(1.0 - eps, (v + 1.0) / 2.0)), 3)

        obs["stage_scores"] = {
            "decision_score": _norm_decision(judge_score),
            "safety_score": round(max(1e-3, min(1.0 - 1e-3, 1.0 if not validation["collision"] and validation.get("safe_distance") else 0.0)), 3),
            "efficiency_score": round(max(1e-3, min(1.0 - 1e-3, max(0.0, min(1.0, 1.0 - 0.05 * self._step_count + repeat_penalty)))), 3),
        }
        obs["validation"] = validation
        obs["resolution"] = {"verified": success, "reason": resolution_reason, "bonus": round(resolution_bonus, 3)}
        return self._to_observation(obs, reward=reward, done=done)

    def _compute_reward(self, validation: dict, judge_score: float, repeat_penalty: float, resolution_bonus: float, done: bool, success: bool) -> float:
        # Produce a small, positive reward in (0,1) based primarily on normalized judge score.
        eps = 1e-3
        # map judge_score from [-1,1] -> [0,1]
        base = max(-1.0, min(1.0, judge_score))
        reward = (base + 1.0) / 2.0

        # Mild multiplicative penalties for severe failures (scale down but keep >= eps)
        if validation.get("collision"):
            reward *= 0.05
        if validation.get("near_miss"):
            reward *= 0.5
        if validation.get("offroad"):
            reward *= 0.2

        # Add small positive bonuses for safe behaviors
        if validation.get("safe_distance"):
            reward += 0.1
        if validation.get("signal_respected"):
            reward += 0.05
        if validation.get("incident_cleared") and validation.get("progress_restored"):
            reward += 0.1
        elif validation.get("incident_cleared") and not validation.get("progress_restored"):
            reward += 0.02

        # Apply repeat penalty as a multiplicative factor (repeat_penalty is <= 0)
        repeat_factor = max(0.0, 1.0 + repeat_penalty)
        reward *= repeat_factor

        # If episode ended unsuccessfully, scale down
        if done and not success:
            reward *= 0.5

        # Incorporate a tiny fraction of resolution_bonus (original bonus may be large)
        reward += max(0.0, min(1.0, resolution_bonus * 0.01))

        # Clamp into (0,1) to satisfy validator (not 0.0 or 1.0)
        reward = max(eps, min(1.0 - eps, reward))
        return round(reward, 3)

    def _narrate(self, obs: dict, event_log: str, stage: str) -> str:
        """Ask the LLM to produce a vivid one-line scene description.
        Falls back to a rule-based description if the LLM fails or is mocked."""
        sensor = obs.get("sensor_data", {}) or {}
        objects = sensor.get("objects", []) or []
        ego = obs.get("ego_state", {}) or {}
        env = obs.get("environment", {}) or {}
        hazard_dist = float(obs.get("hazard_distance", 999.0) or 999.0)
        hazard_type = obs.get("hazard_type", "") or ""

        # quick rule-based fallback (used when LLM unavailable)
        def _rule_based() -> str:
            closest = objects[0] if objects else {}
            t = closest.get("type", "obstacle") if closest else "obstacle"
            d = round(float(closest.get("distance", hazard_dist)), 1) if closest else hazard_dist
            speed = round(float(ego.get("speed", 0.0)), 1)
            road = env.get("road_condition", "normal")
            sig = env.get("traffic_signal", "none")
            if event_log and event_log.lower().startswith("sudden alert:"):
                return event_log
            if stage in ("clearing", "cleared"):
                return f"Hazard CLEARED — road opening ahead. Speed: {speed} km/h. Accelerate."
            if d < 5.0:
                return f"CRITICAL: {t} only {d}m ahead! Speed {speed} km/h — brake hard."
            if d < 12.0:
                return f"Caution: {t} at {d}m. Road: {road}. Signal: {sig}. Speed: {speed} km/h."
            return f"Path clear. Nearest object: {d}m. Speed: {speed} km/h. Road: {road}."

        try:
            snapshot = {
                "nearest_objects": [{"type": o.get("type"), "distance": o.get("distance"), "on_road": o.get("on_road")} for o in objects[:3]],
                "ego_speed_kmh": round(float(ego.get("speed", 0.0)), 1),
                "scenario_stage": stage,
                "hazard_type": hazard_type,
                "hazard_distance_m": round(hazard_dist, 1),
                "event": event_log or "none",
                "road_condition": env.get("road_condition", "normal"),
                "traffic_signal": env.get("traffic_signal", "none"),
            }
            result = self.llm.chat_json(
                _NARRATOR_SYSTEM,
                json.dumps(snapshot),
                temperature=0.4,
                max_tokens=60,
            )
            text = result.get("text", "") or ""
            if text and len(text) > 8:
                return text.strip()
            return _rule_based()
        except Exception:
            return _rule_based()

    def _hint(self) -> str:
        # After a hazard clears, tell the agent to accelerate explicitly
        if self.backend.simulator.current_stage in ("clearing", "cleared"):
            return "Hazard has cleared. Accelerate NOW to restore forward progress."
        persona = self.curriculum.get_judge_persona()
        if persona == "junior":
            return "Dense traffic, expect unpredictable agents. Brake early near hazards."
        if persona == "senior":
            return "Balance safety, smoothness, and realistic Indian-road behavior."
        return ""

    def _to_observation(self, payload: dict, reward: float, done: bool) -> AutoDriveObservation:
        return AutoDriveObservation(
            command_output=payload.get("command_output", ""),
            scene_summary=payload.get("scene_summary", ""),
            active_alerts=payload.get("active_alerts", []),
            sensor_data=payload.get("sensor_data", {}),
            ego_state=payload.get("ego_state", {}),
            environment=payload.get("environment", {}),
            vehicle_profile=payload.get("vehicle_profile", {}),
            event_log=payload.get("event_log", ""),
            hint=payload.get("hint", ""),
            steps_taken=payload.get("steps_taken", 0),
            max_steps=payload.get("max_steps", self.max_steps),
            hazard_type=payload.get("hazard_type", ""),
            hazard_distance=payload.get("hazard_distance", 999.0),
            hazard_status=payload.get("hazard_status", ""),
            scenario_stage=payload.get("scenario_stage", ""),
            scenario_type=payload.get("scenario_type", ""),
            judge_persona=payload.get("judge_persona", ""),
            stage_scores=payload.get("stage_scores", {}),
            validation=payload.get("validation", {}),
            resolution=payload.get("resolution", {}),
            reward=reward,
            done=done,
            metadata=payload.get("metadata", {}),
        )

    @property
    def state(self) -> AutoDriveState:
        return self._state
