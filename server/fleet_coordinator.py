"""Fleet coordinator for AutoDrive Gym â€” Theme 1: Fleet AI + Theme 2: Long-Horizon.

Full continuous flow
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Both vehicles share ONE city route (5 checkpoints end-to-end).  They travel
together: Vehicle 0 (~15 m ahead / primary role) and Vehicle 1 (follower).

Each step follows this sequence:

  1. OBSERVE   â€” each vehicle reads its own sensor snapshot
  2. NEGOTIATE â€” Vehicle 0 broadcasts its proposed action to Vehicle 1 and vice-
                 versa before anyone executes; each agent may adjust based on that
  3. OVERSIGHT â€” OversightAgent validates the joint action pair for conflicts,
                 ambulance corridors, following distance violations
  4. EXECUTE   â€” both vehicles execute their (possibly overridden) actions
  5. BROADCAST â€” ambulance / police / pothole alerts detected by either vehicle
                 are shared to the other immediately
  6. TRANSITION â€” when a vehicle's step triggers checkpoint clear, BOTH vehicles
                  advance to the next checkpoint together

Covers:
  â€¢ Theme 1 core       : two agents on shared road, seeing each other's actions
  â€¢ Fleet AI bonus     : OversightAgent monitors and explains every override
  â€¢ Halluminate bonus  : road actors have intents broadcast between agents
  â€¢ Theme 2            : shared RoutePlanner drives 5-checkpoint city journey
  â€¢ Theme 4            : self-improvement via per-vehicle failure profiling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .driving_backend import DrivingBackend
from .route_planner import RoutePlanner, CITY_ROUTE, SuddenAlertEmitter
from .scenario_generator import ScenarioGenerator

logger = logging.getLogger(__name__)

MAX_FLEET_SIZE = 3
SAFE_INTER_VEHICLE_GAP_M = 15.0


# â”€â”€ Negotiation helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _negotiate(proposing_id: str, proposing_action: Dict[str, Any],
               other_id: str, other_last_action: Dict[str, Any],
               proposing_role: str, hazard_near: bool) -> str:
    """Short negotiation message the proposing agent sends before execution."""
    act = proposing_action.get("action", "wait")
    val = float(proposing_action.get("value", 0.0))
    other_act = other_last_action.get("action", "wait")
    if act in ("brake", "wait") and hazard_near:
        return f"{proposing_id}â†’{other_id}: {'Braking hard' if val > 0.7 else 'Slowing'} â€” hazard ahead. Hold or reduce speed."
    elif act == "steer_left":
        return f"{proposing_id}â†’{other_id}: Steering left â€” shift right or hold lane."
    elif act == "steer_right":
        return f"{proposing_id}â†’{other_id}: Steering right â€” hold left lane."
    elif act == "change_lane_left":
        return f"{proposing_id}â†’{other_id}: Moving to left lane â€” right lane free for you."
    elif act == "change_lane_right":
        return f"{proposing_id}â†’{other_id}: Moving to right lane â€” left lane free for you."
    elif act == "accelerate" and other_act in ("brake", "wait"):
        return f"{proposing_id}â†’{other_id}: Hazard cleared â€” accelerating. Follow when safe."
    elif proposing_role == "primary":
        return f"{proposing_id}[primary]â†’{other_id}: action={act}({val:.1f}). Adjust following distance."
    return f"{proposing_id}â†’{other_id}: action={act}({val:.1f})."


# â”€â”€ Vehicle record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class FleetVehicle:
    vehicle_id: str
    backend: DrivingBackend
    scenario: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0
    is_done: bool = False
    role: str = "primary"
    last_action: Dict[str, Any] = field(default_factory=lambda: {"action": "wait", "value": 0.0})
    last_oversight_note: str = ""
    inbox: List[str] = field(default_factory=list)  # messages from other agents

    def position(self) -> float:
        return float(self.backend.simulator.ego.get("x", 0.0))

    def speed(self) -> float:
        return float(self.backend.simulator.ego.get("speed", 0.0))

    def lane(self) -> str:
        return str(self.backend.simulator.ego.get("lane", "center"))

    def nearest_hazard_distance(self) -> float:
        objects = self.backend.simulator.objects or []
        distances = [float(o.get("distance", 999)) for o in objects if o.get("on_road", True)]
        return min(distances) if distances else 999.0


# â”€â”€ Fleet coordinator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FleetCoordinator:
    """Shared-route multi-vehicle coordinator.

    All vehicles share ONE RoutePlanner so checkpoint transitions are
    synchronised.  When any vehicle clears a checkpoint the coordinator
    advances ALL vehicles to the next checkpoint scenario together.

    Typical usage::

        fc = FleetCoordinator()
        summary = fc.reset(n_vehicles=2)
        result  = fc.step_fleet({"vehicle_0": {"action":"brake","value":0.8},
                                 "vehicle_1": {"action":"wait","value":0.0}})
        # or step one vehicle at a time:
        fc.step("vehicle_0", {"action":"brake","value":0.8})
    """

    def __init__(self, llm=None) -> None:
        self.llm = llm
        self.vehicles: Dict[str, FleetVehicle] = {}
        self.fleet_id: str = ""
        self.route: RoutePlanner = RoutePlanner(continue_on_fail=True)
        self.shared_alerts: List[str] = []
        self.ambulance_corridor_active: bool = False
        self.session_step: int = 0
        self.sudden_alert_emitter: SuddenAlertEmitter = SuddenAlertEmitter(base_probability=0.12, cooldown_steps=6)
        self.sudden_alerts_fired: List[Dict[str, Any]] = []  # full history for metrics

    # â”€â”€ Lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(self, n_vehicles: int = 2, continue_on_fail: bool = True) -> Dict[str, Any]:
        """Initialise a shared-route fleet. Both vehicles start at Checkpoint 0."""
        n_vehicles = max(1, min(n_vehicles, MAX_FLEET_SIZE))
        self.fleet_id = str(uuid4())[:8]
        self.vehicles = {}
        self.shared_alerts = []
        self.ambulance_corridor_active = False
        self.session_step = 0
        self.sudden_alerts_fired = []
        self.route = RoutePlanner(continue_on_fail=continue_on_fail)
        self.route.reset()
        self.sudden_alert_emitter.reset()

        generator = ScenarioGenerator(self.llm)
        roles = ["primary", "follower", "escort"]
        cp = self.route.current_checkpoint

        for i in range(n_vehicles):
            vid = f"vehicle_{i}"
            backend = DrivingBackend()
            backend.reset()
            backend.simulator.ego["x"] = float(i) * -SAFE_INTER_VEHICLE_GAP_M
            fault = cp.scenario_type if cp else "auto_cut_in"
            diff = (cp.difficulty if cp else 0.25) + i * 0.04
            try:
                scenario = generator.generate({}, diff, fault_type_hint=fault).__dict__
                if cp and cp.secondary_events:
                    scenario["dynamic_events"] = (scenario.get("dynamic_events") or []) + list(cp.secondary_events)
            except Exception:
                scenario = _fallback_scenario(fault, diff, cp)
            backend.inject_scenario(scenario)
            self.vehicles[vid] = FleetVehicle(
                vehicle_id=vid, backend=backend, scenario=scenario,
                role=roles[i % len(roles)],
            )

        logger.info("[Fleet %s] %d vehicles at CP 0: %s", self.fleet_id, n_vehicles, cp.name if cp else "?")
        return self._fleet_summary()

    # â”€â”€ Stepping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def step_fleet(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Step ALL vehicles with negotiation + joint oversight in one call."""
        self.session_step += 1
        self.route.record_step()
        # ── Sudden alert injection ─────────────────────────────────────────────
        # Fire BEFORE negotiation so agents can react in this very step.
        cp = self.route.current_checkpoint
        diff = cp.difficulty if cp else 0.4
        sudden_alert = self.sudden_alert_emitter.tick(self.session_step, diff)
        if sudden_alert:
            sudden_alert["step"] = self.session_step
            sudden_alert["checkpoint"] = cp.name if cp else "?"
            self.sudden_alerts_fired.append(sudden_alert)
            # Inject the actor into every vehicle's simulator so state changes immediately
            for v in self.vehicles.values():
                if not v.is_done:
                    try:
                        v.backend.simulator.objects.append(dict(sudden_alert["actor"]))
                    except Exception:
                        pass
            # Broadcast to all agent inboxes
            self._broadcast_alert(sudden_alert["message"], sender_id="__sudden_alert__")
            logger.info(
                "[SUDDEN ALERT | Step %d | CP %s] %s",
                self.session_step, cp.name if cp else "?", sudden_alert["message"],
            )
        # 1. Negotiation â€” share proposed actions before anyone executes
        negotiation_log: List[str] = []
        for vid, proposed in actions.items():
            vehicle = self.vehicles.get(vid)
            if vehicle is None or vehicle.is_done:
                continue
            hazard_near = vehicle.nearest_hazard_distance() < 12.0
            for other_id, other_v in self.vehicles.items():
                if other_id == vid:
                    continue
                msg = _negotiate(vid, proposed, other_id, other_v.last_action, vehicle.role, hazard_near)
                other_v.inbox.append(msg)
                negotiation_log.append(msg)

        # 2. Oversight + execute
        vehicle_results: Dict[str, Dict[str, Any]] = {}
        checkpoint_cleared_by = None
        any_failed = True  # assume all failed until proven otherwise

        for vid, proposed in actions.items():
            vehicle = self.vehicles.get(vid)
            if vehicle is None or vehicle.is_done:
                vehicle_results[vid] = {"skipped": True, "done": True}
                continue

            final_action, oversight_note = self._oversight_check(vehicle, proposed, actions)
            vehicle.last_action = final_action
            vehicle.last_oversight_note = oversight_note

            cmd = vehicle.backend.execute(final_action["action"], final_action.get("value", 0.0))
            vehicle.backend.update()
            vehicle.step_count += 1

            validation = vehicle.backend.programmatic_checks()
            conflict = self._check_inter_vehicle_conflict(vid)
            if conflict:
                validation["fleet_conflict"] = True
                validation["fleet_conflict_note"] = conflict

            reward = self._compute_fleet_reward(validation, final_action, vehicle)
            vehicle.cumulative_reward += reward
            vehicle.history.append({"step": vehicle.step_count, "action": final_action, "reward": reward})
            self._update_shared_alerts(vehicle)

            done = bool(
                validation.get("collision") or validation.get("reached_goal")
                or validation.get("progress_restored") or validation.get("stuck")
                or self.route.checkpoint_timed_out()
            )
            if not validation.get("collision") and (
                validation.get("progress_restored") or validation.get("reached_goal")
            ):
                checkpoint_cleared_by = vid
                any_failed = False

            if done:
                vehicle.is_done = True

            obs = vehicle.backend.build_observation(
                steps_taken=vehicle.step_count,
                max_steps=self.route.current_max_steps() + 6,
                hint=self.route.get_route_hint(),
            )
            obs.update({
                "fleet_context": self._build_fleet_context(vid),
                "shared_alerts": list(self.shared_alerts),
                "negotiation_inbox": list(vehicle.inbox),
                "sudden_alert": sudden_alert,  # None or the alert dict fired this step
                "oversight_note": oversight_note,
                "was_overridden": (final_action != proposed),
                "validation": validation,
                "reward": round(reward, 3),
                "done": done,
                "vehicle_id": vid,
                "role": vehicle.role,
                "route_state": self.route.state.to_dict(),
            })
            vehicle.inbox.clear()
            vehicle_results[vid] = obs

        # 3. Checkpoint transition
        route_event = None
        if checkpoint_cleared_by:
            cp_reward, route_done = self.route.on_checkpoint_success()
            route_event = {
                "type": "checkpoint_cleared",
                "cleared_by": checkpoint_cleared_by,
                "reward": round(cp_reward, 3),
                "route_done": route_done,
                "next_checkpoint": self.route.current_checkpoint.to_dict() if self.route.current_checkpoint else None,
            }
            if not route_done:
                self._advance_all_vehicles_to_checkpoint()
        elif all(v.is_done for v in self.vehicles.values()):
            penalty, route_ended = self.route.on_checkpoint_failure()
            route_event = {"type": "checkpoint_failed", "penalty": round(penalty, 3), "route_ended": route_ended}
            if not route_ended:
                self._reset_all_vehicles_at_checkpoint()
                for v in self.vehicles.values():
                    v.is_done = False

        return {
            "fleet_id": self.fleet_id,
            "session_step": self.session_step,
            "vehicle_observations": vehicle_results,
            "negotiation_log": negotiation_log,
            "shared_alerts": list(self.shared_alerts),
            "sudden_alert": sudden_alert,
            "route_state": self.route.state.to_dict(),
            "route_event": route_event,
            "fleet_done": self.route.state.is_finished,
        }

    def step(self, vehicle_id: str, proposed_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: step a single vehicle (wraps step_fleet)."""
        return self.step_fleet({vehicle_id: proposed_action})

    # â”€â”€ Checkpoint transitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _advance_all_vehicles_to_checkpoint(self) -> None:
        cp = self.route.current_checkpoint
        if cp is None:
            return
        generator = ScenarioGenerator(self.llm)
        for i, (vid, vehicle) in enumerate(self.vehicles.items()):
            vehicle.backend.reset()
            vehicle.backend.simulator.ego["x"] = float(i) * -SAFE_INTER_VEHICLE_GAP_M
            vehicle.is_done = False
            diff = cp.difficulty + i * 0.04
            try:
                scenario = generator.generate({}, diff, fault_type_hint=cp.scenario_type).__dict__
                if cp.secondary_events:
                    scenario["dynamic_events"] = (scenario.get("dynamic_events") or []) + list(cp.secondary_events)
            except Exception:
                scenario = _fallback_scenario(cp.scenario_type, diff, cp)
            vehicle.backend.inject_scenario(scenario)
            vehicle.scenario = scenario
        logger.info("[Fleet] Advanced to CP %d: %s", cp.index, cp.name)

    def _reset_all_vehicles_at_checkpoint(self) -> None:
        for i, (vid, vehicle) in enumerate(self.vehicles.items()):
            vehicle.backend.reset()
            vehicle.backend.simulator.ego["x"] = float(i) * -SAFE_INTER_VEHICLE_GAP_M
            vehicle.backend.inject_scenario(vehicle.scenario)

    # â”€â”€ Oversight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _oversight_check(
        self, vehicle: FleetVehicle, proposed: Dict[str, Any],
        all_proposed: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        action = proposed.get("action", "wait")
        value = float(proposed.get("value", 0.0))

        if self.ambulance_corridor_active and vehicle.role != "primary":
            if action not in ("steer_left", "change_lane_left"):
                return ({"action": "steer_left", "value": 0.6},
                        "OVERSIGHT: Ambulance corridor â€” clear right lane.")

        if vehicle.role == "follower":
            primary = next((v for v in self.vehicles.values() if v.role == "primary"), None)
            if primary and vehicle.position() >= primary.position() - 5.0:
                if action == "accelerate" and value > 0.4:
                    return ({"action": "brake", "value": 0.3},
                            "OVERSIGHT: Too close to lead vehicle â€” reduce speed.")

        for other_id, other_proposed in all_proposed.items():
            if other_id == vehicle.vehicle_id:
                continue
            other_action = other_proposed.get("action", "wait")
            if action == "change_lane_left" and other_action == "change_lane_left":
                return ({"action": "wait", "value": 0.0},
                        f"OVERSIGHT: {other_id} also moving left â€” holding to avoid conflict.")
            if action == "change_lane_right" and other_action == "change_lane_right":
                return ({"action": "wait", "value": 0.0},
                        f"OVERSIGHT: {other_id} also moving right â€” holding to avoid conflict.")

        conflict = self._check_inter_vehicle_conflict(vehicle.vehicle_id)
        if conflict and action == "accelerate" and value > 0.5:
            return ({"action": "brake", "value": 0.6},
                    f"OVERSIGHT: Fleet conflict â€” braking. {conflict}")

        return proposed, "OVERSIGHT: Action approved."

    def _check_inter_vehicle_conflict(self, vehicle_id: str) -> str:
        this = self.vehicles.get(vehicle_id)
        if this is None:
            return ""
        for other_id, other in self.vehicles.items():
            if other_id == vehicle_id or other.is_done:
                continue
            gap = abs(this.position() - other.position())
            if this.lane() == other.lane() and gap < SAFE_INTER_VEHICLE_GAP_M * 0.5:
                return (f"{vehicle_id} and {other_id} same lane, gap={gap:.1f}m "
                        f"(min {SAFE_INTER_VEHICLE_GAP_M * 0.5:.0f}m).")
        return ""

    # â”€â”€ Shared alert broadcast â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _update_shared_alerts(self, vehicle: FleetVehicle) -> None:
        sim = vehicle.backend.simulator
        has_ambulance = any(o.get("type") == "ambulance" for o in sim.objects)
        has_police    = any(o.get("type") == "traffic_police" for o in sim.objects)

        if has_ambulance and not self.ambulance_corridor_active:
            self.ambulance_corridor_active = True
            self._broadcast_alert(
                f"FLEET ALERT [{vehicle.vehicle_id}]: Ambulance â€” ALL vehicles clear right lane!",
                vehicle.vehicle_id)

        if has_police:
            self._broadcast_alert(
                f"FLEET ALERT [{vehicle.vehicle_id}]: Police directing traffic â€” wait for signal.",
                vehicle.vehicle_id)

        if not has_ambulance and self.ambulance_corridor_active:
            any_amb = any(
                any(o.get("type") == "ambulance" for o in v.backend.simulator.objects)
                for v in self.vehicles.values()
            )
            if not any_amb:
                self.ambulance_corridor_active = False
                self._broadcast_alert("FLEET CLEAR: Ambulance corridor released.", vehicle.vehicle_id)
                self.shared_alerts = [a for a in self.shared_alerts if "Ambulance" not in a]

    def _broadcast_alert(self, message: str, sender_id: str) -> None:
        if message not in self.shared_alerts:
            self.shared_alerts.append(message)
        for vid, v in self.vehicles.items():
            if vid != sender_id and message not in v.inbox:
                v.inbox.append(message)

    # â”€â”€ Reward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _compute_fleet_reward(
        self, validation: Dict[str, Any], action: Dict[str, Any], vehicle: FleetVehicle
    ) -> float:
        reward = 0.5
        if validation.get("collision"):
            reward = 0.02
        elif validation.get("near_miss"):
            reward *= 0.4
        if validation.get("safe_distance"):
            reward += 0.15
        if validation.get("fleet_conflict"):
            reward *= 0.6
        if validation.get("progress_restored"):
            reward += 0.2
        if vehicle.role != "primary" and validation.get("incident_cleared"):
            reward += 0.10
        return round(max(0.01, min(0.99, reward)), 3)

    # â”€â”€ Context builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_fleet_context(self, this_vehicle_id: str) -> Dict[str, Any]:
        others = []
        for vid, v in self.vehicles.items():
            if vid == this_vehicle_id:
                continue
            others.append({
                "vehicle_id": vid,
                "role": v.role,
                "position_x": round(v.position(), 1),
                "speed_kmh": round(v.speed(), 1),
                "lane": v.lane(),
                "is_done": v.is_done,
                "last_action": v.last_action,
                "oversight_note": v.last_oversight_note,
                "nearest_hazard_m": round(v.nearest_hazard_distance(), 1),
            })
        return {
            "fleet_id": self.fleet_id,
            "fleet_size": len(self.vehicles),
            "session_step": self.session_step,
            "ambulance_corridor_active": self.ambulance_corridor_active,
            "other_vehicles": others,
            "current_checkpoint": (
                self.route.current_checkpoint.to_dict()
                if self.route.current_checkpoint else None
            ),
            "route_progress_pct": self.route.state.progress_pct(),
        }

    def _fleet_summary(self) -> Dict[str, Any]:
        cp = self.route.current_checkpoint
        return {
            "fleet_id": self.fleet_id,
            "n_vehicles": len(self.vehicles),
            "route": {
                "total_checkpoints": self.route.state.n_checkpoints,
                "current_checkpoint": cp.to_dict() if cp else None,
                "continue_on_fail": self.route.continue_on_fail,
            },
            "vehicles": [
                {
                    "vehicle_id": vid,
                    "role": v.role,
                    "scenario_type": v.scenario.get("type", ""),
                    "difficulty": v.scenario.get("difficulty", 0.0),
                    "starting_position_x": round(v.position(), 1),
                }
                for vid, v in self.vehicles.items()
            ],
        }

    # â”€â”€ Accessors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_fleet_status(self) -> Dict[str, Any]:
        cp = self.route.current_checkpoint
        return {
            "fleet_id": self.fleet_id,
            "session_step": self.session_step,
            "ambulance_corridor_active": self.ambulance_corridor_active,
            "shared_alerts": self.shared_alerts,
            "current_checkpoint": cp.to_dict() if cp else None,
            "route_state": self.route.state.to_dict(),
            "sudden_alerts_fired": self.sudden_alerts_fired,
            "vehicles": [
                {
                    "vehicle_id": vid,
                    "role": v.role,
                    "position_x": round(v.position(), 1),
                    "speed_kmh": round(v.speed(), 1),
                    "lane": v.lane(),
                    "step_count": v.step_count,
                    "cumulative_reward": round(v.cumulative_reward, 3),
                    "is_done": v.is_done,
                    "last_action": v.last_action,
                    "oversight_note": v.last_oversight_note,
                    "inbox": list(v.inbox),
                }
                for vid, v in self.vehicles.items()
            ],
            "fleet_done": self.route.state.is_finished,
        }

    def all_done(self) -> bool:
        return self.route.state.is_finished


# â”€â”€ Fallback scenario helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fallback_scenario(fault: str, diff: float, cp) -> Dict[str, Any]:
    from .constants import DEFAULT_SCENE_ENV, DEFAULT_VEHICLE_PROFILE
    secondary = list(cp.secondary_events) if cp and cp.secondary_events else []
    return {
        "name": f"fleet_{fault}",
        "type": fault,
        "difficulty": diff,
        "root_cause": f"Fleet scenario at {cp.name if cp else 'checkpoint'}",
        "correct_fix_description": "Navigate safely and coordinate with fleet.",
        "expected_behavior": ["brake", "wait"],
        "actors": [],
        "dynamic_events": secondary,
        "environment": dict(DEFAULT_SCENE_ENV),
        "vehicle_profile": dict(DEFAULT_VEHICLE_PROFILE),
    }