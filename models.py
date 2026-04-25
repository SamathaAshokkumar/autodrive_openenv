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
    road_geometry: Dict[str, Any] = Field(default_factory=dict)
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
    # Theme 1: multi-agent pipeline trace (Perception→Context→IntentInference→Negotiation→Decision→Oversight)
    pipeline_trace: Dict[str, Any] = Field(default_factory=dict)
    # Theme 1: fleet context (other vehicle positions, shared alerts, oversight notes)
    fleet_context: Dict[str, Any] = Field(default_factory=dict)
    # Theme 2: long-horizon route state (checkpoint progress, remaining checkpoints)
    route_state: Dict[str, Any] = Field(default_factory=dict)
    # Theme 4: self-improvement context (weak spots targeted, trigger count)
    self_improve_context: Dict[str, Any] = Field(default_factory=dict)
    # Theme 1 + 3.1: inferred actor intent map (from IntentInferenceAgent)
    intent_context: Dict[str, Any] = Field(default_factory=dict)
    # Theme 1: negotiation plan and outcome (from NegotiationAgent)
    negotiation_context: Dict[str, Any] = Field(default_factory=dict)
    # Theme 3.1: indirect zone signals (nearby_places, signs, cues — NOT explicit zone label)
    zone_cues: Dict[str, Any] = Field(default_factory=dict)
    # Theme 3.2: Bayesian Theory-of-Mind belief state (per-actor intent distributions)
    belief_state: Dict[str, Any] = Field(default_factory=dict)


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
    # Theme 3.1: indirect zone signals — nearby POIs, signs, ambient cues
    # These are NEVER a direct zone label; agent must infer appropriate behavior
    zone_cues: Dict[str, Any] = field(default_factory=dict)
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