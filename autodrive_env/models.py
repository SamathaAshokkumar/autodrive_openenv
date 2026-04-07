"""Typed OpenEnv models for AutoDrive Gym."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .openenv_compat import Action, Field, Observation, State


class AutoDriveAction(Action):
    """Single driving action per step, mirroring kube's one-command loop."""

    action: str = Field(default="wait", description="Driving action such as brake, accelerate, steer_left, horn, wait")
    value: float = Field(default=0.0, description="Action intensity or steering angle proxy")


class AutoDriveObservation(Observation):
    """Raw observation passed to the agent."""

    command_output: str = ""
    scene_summary: str = ""
    active_alerts: List[str] = Field(default_factory=list)
    sensor_data: Dict[str, Any] = Field(default_factory=dict)
    ego_state: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    vehicle_profile: Dict[str, Any] = Field(default_factory=dict)
    event_log: str = ""
    hint: str = ""
    steps_taken: int = 0
    max_steps: int = 20
    hazard_type: str = ""
    hazard_distance: float = 999.0
    hazard_status: str = ""
    scenario_stage: str = ""
    scenario_type: str = ""
    judge_persona: str = ""
    stage_scores: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    resolution: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    reward: float | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AutoDriveState(State):
    """Episode metadata."""

    incident_id: str = ""
    difficulty: float = 0.2
    incident_type: str = ""
    root_cause: str = ""
    correct_fix: str = ""
    is_resolved: bool = False
    cumulative_reward: float = 0.0
    judge_persona: str = "junior"
    curriculum_stats: Dict[str, Any] = {}


@dataclass
class ScenarioSpec:
    """Driving scenario definition."""

    name: str
    type: str
    difficulty: float
    vehicle_profile: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    actors: List[Dict[str, Any]] = field(default_factory=list)
    root_cause: str = ""
    alert_message: str = ""
    correct_fix_description: str = ""
    expected_behavior: List[str] = field(default_factory=list)
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AdversarialScenarioSpec:
    """Adversarial multi-actor scenario."""

    name: str
    type: str
    difficulty: float
    vehicle_profile: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    actors: List[Dict[str, Any]] = field(default_factory=list)
    root_cause: str = ""
    alert_message: str = ""
    correct_fix_description: str = ""
    expected_behavior: List[str] = field(default_factory=list)
    red_herrings: List[str] = field(default_factory=list)
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)
