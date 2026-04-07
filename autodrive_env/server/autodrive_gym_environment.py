"""AutoDrive Gym environment implementation with kube-style structure."""

from __future__ import annotations

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
                    "safety_score": 1.0 if not validation["collision"] and validation["safe_distance"] else -1.0,
                    "efficiency_score": round(max(-1.0, 1.0 - 0.1 * self._step_count + repeat_penalty), 3),
                },
            },
        )
        backend_event_log = str(obs.get("event_log", "") or "").strip()
        obs["command_output"] = command_output
        obs["event_log"] = backend_event_log
        obs["active_alerts"] = [backend_event_log] if backend_event_log.lower().startswith("sudden alert:") else []
        obs["scenario_type"] = self.scenario.get("type", "")
        obs["judge_persona"] = persona
        obs["stage_scores"] = {
            "decision_score": round(judge_score, 3),
            "safety_score": 1.0 if not validation["collision"] and validation["safe_distance"] else -1.0,
            "efficiency_score": round(max(-1.0, 1.0 - 0.1 * self._step_count + repeat_penalty), 3),
        }
        obs["validation"] = validation
        obs["resolution"] = {"verified": success, "reason": resolution_reason, "bonus": round(resolution_bonus, 3)}
        return self._to_observation(obs, reward=reward, done=done)

    def _compute_reward(self, validation: dict, judge_score: float, repeat_penalty: float, resolution_bonus: float, done: bool, success: bool) -> float:
        reward = judge_score + repeat_penalty
        if validation["collision"]:
            reward -= 10.0
        if validation["near_miss"]:
            reward -= 3.0
        if validation["overspeed"]:
            reward -= 1.0
        if validation["offroad"]:
            reward -= 2.0
        if validation["safe_distance"]:
            reward += 1.0
        if validation.get("incident_cleared") and validation.get("progress_restored"):
            reward += 2.0
        elif validation.get("incident_cleared") and not validation.get("progress_restored"):
            reward += 0.5
        if validation["signal_respected"]:
            reward += 0.5
        if validation["stuck"]:
            reward -= 4.0
        repeated_brakes = sum(1 for item in self.history[-3:] if item.get("action") == "brake")
        if repeated_brakes >= 3:
            reward -= 2.0
        if done and not success:
            reward -= 2.0
        reward += resolution_bonus
        return round(reward, 3)

    def _hint(self) -> str:
        persona = self.curriculum.get_judge_persona()
        if persona == "junior":
            return "Dense traffic, expect unpredictable agents."
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
