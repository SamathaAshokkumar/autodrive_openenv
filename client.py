"""AutoDrive Gym client."""

from typing import Dict

from .models import AutoDriveAction, AutoDriveObservation, AutoDriveState
from .openenv_compat import EnvClient, StepResult


class AutoDriveClient(EnvClient[AutoDriveAction, AutoDriveObservation, AutoDriveState]):
    """Client for the AutoDrive Gym environment."""

    def __init__(self, base_url: str, **kwargs):
        kwargs.setdefault("message_timeout_s", 180.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: AutoDriveAction) -> Dict:
        return {"action": action.action, "value": action.value}

    def _parse_result(self, payload: Dict) -> StepResult[AutoDriveObservation]:
        obs_data = payload.get("observation", {})
        observation = AutoDriveObservation(
            command_output=obs_data.get("command_output", ""),
            scene_summary=obs_data.get("scene_summary", ""),
            active_alerts=obs_data.get("active_alerts", []),
            sensor_data=obs_data.get("sensor_data", {}),
            ego_state=obs_data.get("ego_state", {}),
            environment=obs_data.get("environment", {}),
            vehicle_profile=obs_data.get("vehicle_profile", {}),
            event_log=obs_data.get("event_log", ""),
            hint=obs_data.get("hint", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 20),
            hazard_type=obs_data.get("hazard_type", ""),
            hazard_distance=obs_data.get("hazard_distance", 999.0),
            hazard_status=obs_data.get("hazard_status", ""),
            scenario_stage=obs_data.get("scenario_stage", ""),
            scenario_type=obs_data.get("scenario_type", ""),
            judge_persona=obs_data.get("judge_persona", ""),
            stage_scores=obs_data.get("stage_scores", {}),
            validation=obs_data.get("validation", {}),
            resolution=obs_data.get("resolution", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> AutoDriveState:
        return AutoDriveState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            incident_id=payload.get("incident_id", ""),
            difficulty=payload.get("difficulty", 0.2),
            incident_type=payload.get("incident_type", ""),
            root_cause=payload.get("root_cause", ""),
            correct_fix=payload.get("correct_fix", ""),
            is_resolved=payload.get("is_resolved", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            judge_persona=payload.get("judge_persona", "junior"),
            curriculum_stats=payload.get("curriculum_stats", {}),
        )
