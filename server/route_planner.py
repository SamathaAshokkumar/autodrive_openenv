"""Multi-checkpoint city route planner — Theme 2: Long-Horizon Planning.

10-checkpoint continuous Indian city journey, each with:
  • A primary hazard (scenario_type — authentic Indian road condition)
  • 2–3 inline secondary_events fired mid-segment at specific step offsets
  • SUDDEN ALERTS: unpredictable events (ambulance from rear, pothole
    discovered mid-lane, road bridge/speed-bump, flash crowd, cattle on road)
    pushed to ALL agents even during task execution via SuddenAlertEmitter

The environment is fully interactive: every action causes a state change and
the new state is returned to the agent.  SuddenAlerts follow the same
mechanism — they modify simulator state and the new state is sent as the
very next observation, so agents must react immediately.

Journey:
  Narrows Lane  ──►  Market Bazaar  ──►  Railway Gate  ──►  School Zone
  ──►  Flyover Entry  ──►  Hospital Gate  ──►  Petrol-Pump Junction
  ──►  Construction Zone  ──►  Night Highway  ──►  Toll Plaza  ──►  DESTINATION

Reward structure:
  • Per-checkpoint clear        → +2.0 – 6.0  (scales with difficulty)
  • Time bonus                  → +0.5 × (budget_remaining / budget)
  • Sudden-alert handled well   → +0.3 per alert (tracked externally)
  • Full route completion bonus → +20.0
  • Checkpoint failure          → −2.0 (penalty-and-continue) or retry/abort
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Sudden‑alert probability table ────────────────────────────────────────────
# Each entry: (alert_message, actor_dict, hazard_type)
# These are injected INTO the running simulation mid-task — the new simulator
# state is returned as the very next observation to all agents.
_SUDDEN_ALERT_POOL: List[Dict[str, Any]] = [
    {
        "id": "sa_ambulance_rear",
        "message": "🚨 SUDDEN ALERT: Ambulance approaching fast from BEHIND — immediately clear right lane!",
        "hazard_type": "ambulance_approach",
        "actor": {"type": "ambulance", "x": -12, "y": 1.5, "vx": 3.2, "vy": 0.0, "behavior": "emergency_pass", "lane": "right", "on_road": True},
        "fleet_action": "clear_right_lane",
        "probability": 0.18,
    },
    {
        "id": "sa_pothole_center",
        "message": "⚠️ SUDDEN ALERT: Large pothole discovered in CENTRE lane — steer around it!",
        "hazard_type": "pothole",
        "actor": {"type": "pothole", "x": 8, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True},
        "fleet_action": "avoid",
        "probability": 0.22,
    },
    {
        "id": "sa_cattle_road",
        "message": "🐄 SUDDEN ALERT: Cattle crossing! Several cows standing on the road — BRAKE immediately!",
        "hazard_type": "animal_crossing",
        "actor": {"type": "animal", "x": 10, "y": 0.2, "vx": -0.3, "vy": 0.2, "behavior": "sudden_cross", "lane": "center", "on_road": True},
        "fleet_action": "brake",
        "probability": 0.15,
    },
    {
        "id": "sa_flash_crowd",
        "message": "👥 SUDDEN ALERT: Flash crowd spilling onto the road (rally/procession) — SLOW DOWN!",
        "hazard_type": "pedestrian_surge",
        "actor": {"type": "pedestrian", "x": 9, "y": -0.5, "vx": -0.4, "vy": 0.6, "behavior": "sudden_cross", "lane": "left", "on_road": True},
        "fleet_action": "slow",
        "probability": 0.14,
    },
    {
        "id": "sa_speed_breaker",
        "message": "🚧 SUDDEN ALERT: Unmarked speed breaker / road hump — reduce speed before impact!",
        "hazard_type": "speed_breaker",
        "actor": {"type": "pothole", "x": 7, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True},
        "fleet_action": "slow",
        "probability": 0.20,
    },
    {
        "id": "sa_road_bridge",
        "message": "🌉 SUDDEN ALERT: Low road bridge / flyover entry — high vehicles must change lane!",
        "hazard_type": "infrastructure",
        "actor": {"type": "pothole", "x": 12, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True},
        "fleet_action": "change_lane",
        "probability": 0.12,
    },
    {
        "id": "sa_wrong_way",
        "message": "🔴 SUDDEN ALERT: Vehicle going WRONG WAY in your lane — swerve or brake hard!",
        "hazard_type": "wrong_way_vehicle",
        "actor": {"type": "car", "x": 6, "y": 0.0, "vx": -1.5, "behavior": "sudden_cross", "lane": "center", "on_road": True},
        "fleet_action": "emergency_brake",
        "probability": 0.10,
    },
    {
        "id": "sa_water_log",
        "message": "💧 SUDDEN ALERT: Waterlogged section ahead — loss of traction risk, reduce speed!",
        "hazard_type": "road_condition",
        "actor": {"type": "pothole", "x": 11, "y": 1.0, "vx": 0.0, "behavior": "static", "lane": "right", "on_road": True},
        "fleet_action": "slow",
        "probability": 0.16,
    },
]


class SuddenAlertEmitter:
    """Fires unpredictable mid-task alerts into the live simulation state.

    Called after every fleet step.  With configured probability, picks a random
    alert from the pool, injects the actor into both vehicles' simulators, and
    returns the alert dict so it propagates to ALL agent observations immediately.
    """

    def __init__(self, base_probability: float = 0.12, cooldown_steps: int = 6) -> None:
        self.base_probability = base_probability
        self.cooldown_steps = cooldown_steps
        self._steps_since_last: int = cooldown_steps  # allow first alert early

    def reset(self) -> None:
        self._steps_since_last = self.cooldown_steps

    def tick(self, step: int, difficulty: float) -> Optional[Dict[str, Any]]:
        """Return a sudden-alert dict or None.  Called once per fleet step."""
        self._steps_since_last += 1
        if self._steps_since_last < self.cooldown_steps:
            return None
        prob = self.base_probability + difficulty * 0.08
        if random.random() > prob:
            return None
        # Weight by each alert's own probability
        pool = _SUDDEN_ALERT_POOL
        weights = [a["probability"] for a in pool]
        alert = random.choices(pool, weights=weights, k=1)[0]
        self._steps_since_last = 0
        return dict(alert)  # shallow copy so caller can annotate


# ── Checkpoint definitions ────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    index: int
    name: str
    scenario_type: str
    difficulty: float
    description: str
    checkpoint_reward: float = 2.0
    max_steps: int = 25
    secondary_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "scenario_type": self.scenario_type,
            "difficulty": self.difficulty,
            "description": self.description,
            "checkpoint_reward": self.checkpoint_reward,
            "max_steps": self.max_steps,
        }

    def as_scenario_hint(self) -> str:
        secondaries = [e.get("message", "") for e in self.secondary_events if e.get("message")]
        extra = f"  Expect mid-segment: {'; '.join(secondaries)}" if secondaries else ""
        return f"[CP {self.index + 1}] {self.name}: {self.description}.{extra}"


# ─────────────────────────────────────────────────────────────────────────────
# CITY ROUTE — 10 authentic Indian road checkpoints
# Every segment has a primary hazard + 2-3 inline secondary alerts.
# SuddenAlertEmitter fires ADDITIONAL random alerts on top of these.
# ─────────────────────────────────────────────────────────────────────────────

CITY_ROUTE: List[Checkpoint] = [
    # CP 0 ─ Narrow Gully Entry ─────────────────────────────────────────────
    Checkpoint(
        index=0, difficulty=0.22, checkpoint_reward=2.0, max_steps=20,
        name="Narrow Gully Entry — Residential Lane",
        scenario_type="blind_spot_merge",
        description=(
            "One-lane gully with two-way traffic; parked autorickshaws reduce visibility; "
            "motorcycles squeeze past from both sides."
        ),
        secondary_events=[
            {"trigger_step": 4, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Child on bicycle darting out from a side gate!",
             "actor": {"type": "pedestrian", "x": 7, "y": 1.0, "vx": -0.6, "vy": 0.1, "behavior": "sudden_cross", "lane": "left", "on_road": True}},
            {"trigger_step": 10, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Villager pushing a handcart blocking half the lane!",
             "actor": {"type": "car", "x": 11, "y": -0.5, "vx": 0.05, "behavior": "static", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 1 ─ Market Bazaar ───────────────────────────────────────────────────
    Checkpoint(
        index=1, difficulty=0.28, checkpoint_reward=2.2, max_steps=22,
        name="Market Bazaar Street — Peak Hour",
        scenario_type="auto_cut_in",
        description=(
            "Chaotic bazaar street: autorickshaws weave continuously, street vendors "
            "spill onto the road, delivery bikes mount the pavement."
        ),
        secondary_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Unmarked speed breaker — slow down immediately!",
             "actor": {"type": "pothole", "x": 9, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True}},
            {"trigger_step": 13, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Person stepping off bus directly in front!",
             "actor": {"type": "pedestrian", "x": 8, "y": 0.8, "vx": -0.4, "vy": -0.2, "behavior": "sudden_cross", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 2 ─ Railway Gate ────────────────────────────────────────────────────
    Checkpoint(
        index=2, difficulty=0.38, checkpoint_reward=2.5, max_steps=26,
        name="Railway Gate Crossing — Signal Conflict",
        scenario_type="traffic_light_ambiguity",
        description=(
            "Level crossing: gate is closing, conflicting signals from traffic police "
            "and automatic lights; impatient drivers jumping the queue."
        ),
        secondary_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Bike emerging from blind spot behind the gate pillar!",
             "actor": {"type": "bike", "x": 9, "y": 2.0, "vx": 0.3, "behavior": "blind_spot_merge", "lane": "right", "on_road": True}},
            {"trigger_step": 14, "kind": "spawn_vehicle",
             "message": "🚨 SUDDEN ALERT: Ambulance approaching fast from behind — CLEAR THE LANE!",
             "actor": {"type": "ambulance", "x": -8, "y": 1.5, "vx": 2.5, "behavior": "emergency_pass", "lane": "right", "on_road": True},
             "hazard_type": "ambulance_approach"},
        ],
    ),
    # CP 3 ─ School Zone ─────────────────────────────────────────────────────
    Checkpoint(
        index=3, difficulty=0.40, checkpoint_reward=2.5, max_steps=26,
        name="School Zone Rush — Dismissal Time",
        scenario_type="pedestrian_crossing",
        description=(
            "Hundreds of children cross mid-lane simultaneously; school buses stop "
            "with no warning; ice-cream cart blocks the right shoulder."
        ),
        secondary_events=[
            {"trigger_step": 4, "kind": "spawn_vehicle",
             "message": "🐕 SUDDEN ALERT: Stray dog running across the road!",
             "actor": {"type": "animal", "x": 12, "y": 0.5, "vx": -0.8, "vy": -0.3, "behavior": "sudden_cross", "lane": "center", "on_road": True}},
            {"trigger_step": 11, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Deep pothole in left lane — avoid!",
             "actor": {"type": "pothole", "x": 10, "y": -1.5, "vx": 0.0, "behavior": "static", "lane": "left", "on_road": True}},
            {"trigger_step": 18, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Auto cutting in from school gate exit!",
             "actor": {"type": "auto", "x": 8, "y": -1.8, "vx": 0.2, "behavior": "cut_in", "lane": "left", "on_road": True}},
        ],
    ),
    # CP 4 ─ Flyover Entry ───────────────────────────────────────────────────
    Checkpoint(
        index=4, difficulty=0.48, checkpoint_reward=3.0, max_steps=28,
        name="Flyover Entry — Lane Merge Chaos",
        scenario_type="cut_in",
        description=(
            "Three lanes compress to one before the flyover ramp; aggressive merging "
            "from all sides; road narrows suddenly; motorcycles filter between lanes."
        ),
        secondary_events=[
            {"trigger_step": 6, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Truck aggressively merging — hold your lane or brake!",
             "actor": {"type": "car", "x": 10, "y": -2.0, "vx": 0.6, "behavior": "cut_in", "lane": "right", "on_road": True}},
            {"trigger_step": 15, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Waterlogged pothole on the ramp — steer right!",
             "actor": {"type": "pothole", "x": 9, "y": -0.5, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True}},
            {"trigger_step": 22, "kind": "spawn_vehicle",
             "message": "🌉 SUDDEN ALERT: Low bridge overhead — high vehicles change to left lane!",
             "actor": {"type": "pothole", "x": 14, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 5 ─ Hospital Gate ───────────────────────────────────────────────────
    Checkpoint(
        index=5, difficulty=0.55, checkpoint_reward=3.5, max_steps=30,
        name="Hospital Gate Corridor — Emergency Zone",
        scenario_type="ambulance_approach",
        description=(
            "Active ambulance corridor: multiple ambulances entering and exiting; "
            "anxious families crossing without looking; police directing traffic."
        ),
        secondary_events=[
            {"trigger_step": 6, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Bike zig-zagging out of hospital exit!",
             "actor": {"type": "bike", "x": 11, "y": 0.0, "vx": 0.2, "behavior": "zig_zag", "lane": "center", "on_road": True}},
            {"trigger_step": 15, "kind": "spawn_vehicle",
             "message": "🚔 SUDDEN ALERT: Police officer stopping all traffic — wait for signal!",
             "actor": {"type": "traffic_police", "x": 20, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True}},
            {"trigger_step": 22, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Unmarked speed hump at hospital exit!",
             "actor": {"type": "pothole", "x": 16, "y": 0.0, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 6 ─ Petrol-Pump Junction ────────────────────────────────────────────
    Checkpoint(
        index=6, difficulty=0.52, checkpoint_reward=3.0, max_steps=28,
        name="Petrol-Pump Junction — U-turn Hell",
        scenario_type="cut_in",
        description=(
            "Busy fuel-station crossing: vehicles making illegal U-turns, bikes "
            "refuelling and pulling out without checking, tanker truck blocking view."
        ),
        secondary_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Car making sudden illegal U-turn — brake or steer!",
             "actor": {"type": "car", "x": 8, "y": 0.5, "vx": -0.4, "vy": 0.5, "behavior": "sudden_cross", "lane": "center", "on_road": True}},
            {"trigger_step": 14, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Cattle meandering out of a side lane onto the road!",
             "actor": {"type": "animal", "x": 10, "y": 1.5, "vx": -0.2, "vy": 0.3, "behavior": "sudden_cross", "lane": "left", "on_road": True}},
        ],
    ),
    # CP 7 ─ Construction Zone ───────────────────────────────────────────────
    Checkpoint(
        index=7, difficulty=0.65, checkpoint_reward=4.0, max_steps=32,
        name="Construction Zone — Half-Road Blocked",
        scenario_type="adversarial",
        description=(
            "Active construction: one lane completely closed, flagmen waving "
            "contradictory signals, heavy machinery reversing without warning, "
            "rubble in the remaining lane."
        ),
        secondary_events=[
            {"trigger_step": 6, "kind": "spawn_vehicle",
             "message": "🚧 SUDDEN ALERT: JCB reversing onto road — STOP!",
             "actor": {"type": "car", "x": 10, "y": -1.0, "vx": -0.3, "behavior": "cut_in", "lane": "right", "on_road": True}},
            {"trigger_step": 14, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Rubble pile in centre lane — steer left immediately!",
             "actor": {"type": "pothole", "x": 9, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True}},
            {"trigger_step": 22, "kind": "spawn_vehicle",
             "message": "👥 SUDDEN ALERT: Construction workers crossing — slow down!",
             "actor": {"type": "pedestrian", "x": 8, "y": -0.5, "vx": -0.3, "vy": 0.4, "behavior": "sudden_cross", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 8 ─ Night Highway ───────────────────────────────────────────────────
    Checkpoint(
        index=8, difficulty=0.75, checkpoint_reward=5.0, max_steps=34,
        name="Night Highway — Low Visibility Sprint",
        scenario_type="adversarial",
        description=(
            "Unlit highway section at night: high-beam blinding from oncoming trucks, "
            "unlit bullock carts parked in lane, drunk driver swerving, "
            "road surface sudden dips."
        ),
        secondary_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "🔦 SUDDEN ALERT: Unlit bullock-cart parked in centre lane — BRAKE!",
             "actor": {"type": "car", "x": 12, "y": 0.0, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True}},
            {"trigger_step": 14, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Drunk driver swerving into your lane!",
             "actor": {"type": "car", "x": 8, "y": -1.5, "vx": 0.5, "vy": 0.4, "behavior": "zig_zag", "lane": "right", "on_road": True}},
            {"trigger_step": 23, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Sudden road dip / unmarked pothole — reduce speed!",
             "actor": {"type": "pothole", "x": 9, "y": 0.5, "vx": 0.0, "behavior": "ridge", "lane": "center", "on_road": True}},
        ],
    ),
    # CP 9 ─ Toll Plaza ──────────────────────────────────────────────────────
    Checkpoint(
        index=9, difficulty=0.82, checkpoint_reward=6.0, max_steps=36,
        name="Toll Plaza — FASTag Lane Scramble",
        scenario_type="adversarial",
        description=(
            "Toll plaza: aggressive lane-switching as drivers target the shortest queue; "
            "motorcycles filtering between booths; pedestrian toll workers directing "
            "vehicles; emergency lane being misused."
        ),
        secondary_events=[
            {"trigger_step": 5, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Truck cutting across three lanes to reach toll booth!",
             "actor": {"type": "car", "x": 12, "y": -2.0, "vx": 0.8, "behavior": "cut_in", "lane": "right", "on_road": True}},
            {"trigger_step": 12, "kind": "spawn_vehicle",
             "message": "⚠️ SUDDEN ALERT: Waterlogged stretch at toll gate — traction warning!",
             "actor": {"type": "pothole", "x": 10, "y": -0.5, "vx": 0.0, "behavior": "static", "lane": "center", "on_road": True}},
            {"trigger_step": 22, "kind": "spawn_vehicle",
             "message": "🚨 SUDDEN ALERT: Ambulance overtaking on emergency lane shoulder!",
             "actor": {"type": "ambulance", "x": -5, "y": 2.5, "vx": 3.0, "behavior": "emergency_pass", "lane": "left", "on_road": True},
             "hazard_type": "ambulance_approach"},
        ],
    ),
]

ROUTE_COMPLETION_BONUS = 20.0
MAX_RETRIES_PER_CHECKPOINT = 2


# ── Route state tracker ───────────────────────────────────────────────────────

@dataclass
class RouteState:
    """Tracks progress along the shared city route."""
    checkpoints: List[Checkpoint] = field(default_factory=lambda: list(CITY_ROUTE))
    current_index: int = 0
    steps_this_checkpoint: int = 0
    total_steps: int = 0
    checkpoint_retries: int = 0
    completed_checkpoints: List[int] = field(default_factory=list)
    failed_checkpoints: List[int] = field(default_factory=list)
    checkpoint_rewards: List[float] = field(default_factory=list)
    cumulative_penalty: float = 0.0
    route_aborted: bool = False
    route_completed: bool = False
    cumulative_route_reward: float = 0.0

    @property
    def current_checkpoint(self) -> Optional[Checkpoint]:
        if self.current_index < len(self.checkpoints):
            return self.checkpoints[self.current_index]
        return None

    @property
    def is_finished(self) -> bool:
        return self.route_completed or self.route_aborted

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoints)

    def progress_pct(self) -> float:
        return round(len(self.completed_checkpoints) / self.n_checkpoints, 3)

    def to_dict(self) -> Dict[str, Any]:
        cp = self.current_checkpoint
        return {
            "current_checkpoint_index": self.current_index,
            "current_checkpoint": cp.to_dict() if cp else None,
            "completed_checkpoints": self.completed_checkpoints,
            "failed_checkpoints": self.failed_checkpoints,
            "progress_pct": self.progress_pct(),
            "steps_this_checkpoint": self.steps_this_checkpoint,
            "total_steps": self.total_steps,
            "checkpoint_retries": self.checkpoint_retries,
            "route_completed": self.route_completed,
            "route_aborted": self.route_aborted,
            "cumulative_route_reward": round(self.cumulative_route_reward, 3),
            "cumulative_penalty": round(self.cumulative_penalty, 3),
            "remaining_checkpoints": [c.to_dict() for c in self.checkpoints[self.current_index + 1:]],
            "n_checkpoints": self.n_checkpoints,
        }


class RoutePlanner:
    """Controls a 10-checkpoint city route shared by all fleet vehicles.

    Shared by all fleet vehicles — when any vehicle clears a checkpoint the whole
    fleet advances together.  Each vehicle's backend loads the same checkpoint
    scenario so they face the same core hazard but from different positions.

    Penalty-and-continue mode (``continue_on_fail=True``):
      On checkpoint failure the agent takes a penalty but the route does NOT
      abort — it advances to the next checkpoint with reduced score.  This
      is the recommended training mode so the agent sees all 10 checkpoints
      every episode and learns from each one.
    """

    def __init__(self, continue_on_fail: bool = True) -> None:
        self.state = RouteState()
        self.continue_on_fail = continue_on_fail

    def reset(self) -> RouteState:
        self.state = RouteState()
        return self.state

    @property
    def current_checkpoint(self) -> Optional[Checkpoint]:
        return self.state.current_checkpoint

    def current_scenario_type(self) -> str:
        cp = self.state.current_checkpoint
        return cp.scenario_type if cp else "pedestrian_crossing"

    def current_difficulty(self) -> float:
        cp = self.state.current_checkpoint
        return cp.difficulty if cp else 0.25

    def current_max_steps(self) -> int:
        cp = self.state.current_checkpoint
        return cp.max_steps if cp else 22

    def get_secondary_events(self) -> List[Dict[str, Any]]:
        """Return the secondary mid-segment events for the current checkpoint."""
        cp = self.state.current_checkpoint
        return list(cp.secondary_events) if cp else []

    def record_step(self) -> None:
        self.state.steps_this_checkpoint += 1
        self.state.total_steps += 1

    def checkpoint_timed_out(self) -> bool:
        cp = self.state.current_checkpoint
        return cp is not None and self.state.steps_this_checkpoint >= cp.max_steps

    def get_route_hint(self) -> str:
        """Rich hint shown to the agent — includes current checkpoint and upcoming hazards."""
        cp = self.state.current_checkpoint
        if cp is None:
            return "ROUTE COMPLETE — you have reached the destination!"
        done = len(self.state.completed_checkpoints)
        remaining = self.state.n_checkpoints - done
        upcoming = [c.name for c in self.state.checkpoints[self.state.current_index + 1:self.state.current_index + 3]]
        upcoming_str = " → ".join(upcoming) if upcoming else "destination"
        return (
            f"[ROUTE {done + 1}/{self.state.n_checkpoints}] {cp.as_scenario_hint()} "
            f"| Step {self.state.steps_this_checkpoint}/{cp.max_steps} "
            f"| Retries: {self.state.checkpoint_retries}/{MAX_RETRIES_PER_CHECKPOINT} "
            f"| Ahead: {upcoming_str}"
        )

    # ── Checkpoint transitions ────────────────────────────────────────────────

    def on_checkpoint_success(self) -> Tuple[float, bool]:
        """Call when the agent clears the current checkpoint.

        Returns (reward_earned, route_completed).
        """
        cp = self.state.current_checkpoint
        if cp is None:
            return 0.0, True

        budget = cp.max_steps
        used = self.state.steps_this_checkpoint
        time_bonus = round(0.5 * max(0.0, (budget - used) / budget), 3)
        reward = cp.checkpoint_reward + time_bonus

        self.state.completed_checkpoints.append(cp.index)
        self.state.checkpoint_rewards.append(round(reward, 3))
        self.state.cumulative_route_reward += reward
        self.state.checkpoint_retries = 0
        self.state.current_index += 1
        self.state.steps_this_checkpoint = 0

        if self.state.current_index >= self.state.n_checkpoints:
            self.state.route_completed = True
            self.state.cumulative_route_reward += ROUTE_COMPLETION_BONUS
            logger.info("[Route] COMPLETED! Total reward: %.2f", self.state.cumulative_route_reward)
            return reward + ROUTE_COMPLETION_BONUS, True

        next_cp = self.state.current_checkpoint
        logger.info("[Route] CP %d cleared (+%.2f). Next → %s", cp.index, reward, next_cp.name)
        return reward, False

    def on_checkpoint_failure(self) -> Tuple[float, bool]:
        """Call when the agent fails a checkpoint (collision/stuck/timeout).

        In continue_on_fail mode: applies penalty and advances to next checkpoint.
        Otherwise: retries up to MAX_RETRIES, then aborts route.

        Returns (penalty, route_ended).
        """
        cp = self.state.current_checkpoint
        self.state.checkpoint_retries += 1
        self.state.failed_checkpoints.append(cp.index if cp else -1)
        self.state.steps_this_checkpoint = 0

        if self.continue_on_fail:
            # Penalty-and-continue: deduct and move on so agent sees all checkpoints
            penalty = -2.0
            self.state.cumulative_penalty += abs(penalty)
            self.state.current_index += 1
            self.state.checkpoint_retries = 0
            route_done = self.state.current_index >= self.state.n_checkpoints
            if route_done:
                self.state.route_completed = True  # completed (with penalties)
            logger.info("[Route] CP failed — penalty %.1f, continuing to next.", penalty)
            return penalty, route_done

        # Strict retry mode
        if self.state.checkpoint_retries >= MAX_RETRIES_PER_CHECKPOINT:
            self.state.route_aborted = True
            logger.info("[Route] ABORTED after %d retries on CP %d.", MAX_RETRIES_PER_CHECKPOINT, self.state.current_index)
            return -3.0, True

        logger.info("[Route] CP %d failed. Retry %d/%d.", self.state.current_index, self.state.checkpoint_retries, MAX_RETRIES_PER_CHECKPOINT)
        return -1.0, False
