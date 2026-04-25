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
from .intent_engine import assign_intents
from .judge import LLMJudge
from .llm_client import LLMClient
from .belief_tracker import BeliefTracker
from .multi_agent_pipeline import MultiAgentPipeline
from .reward_tracker import RewardTracker
from .route_planner import RoutePlanner
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
        # Wire self-improvement into curriculum (Theme 4)
        self.curriculum.set_adversarial_designer(self.designer)
        # Multi-agent pipeline (Theme 1)
        self.pipeline = MultiAgentPipeline(self.llm)
        # Long-horizon route planner (Theme 2)
        self.route_planner: RoutePlanner | None = None
        self._route_mode: bool = False
        # Reward tracker for reward curves (judging)
        self.reward_tracker = RewardTracker()
        # Theory-of-Mind belief tracker (surfaced in observation for agents)
        self.belief_tracker = BeliefTracker()
        self._episode_num: int = 0
        self.scenario = None
        self.history = []
        self._step_count = 0
        self.max_steps = MAX_STEPS
        self._state = AutoDriveState(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: str | None = None) -> AutoDriveObservation:
        self.backend.reset()
        self.belief_tracker.reset()
        self._episode_num += 1
        difficulty = self.curriculum.get_difficulty()

        # ── Theme 2: Route mode activation ────────────────────────────────────
        self._route_mode = (task_id == "city_route")
        if self._route_mode:
            if self.route_planner is None:
                self.route_planner = RoutePlanner()
            else:
                self.route_planner.reset()
            incident_type = self.route_planner.current_scenario_type()
            difficulty = self.route_planner.current_difficulty()
        else:
            self.route_planner = None
            # ── Theme 4: Self-improvement check ───────────────────────────────
            if self.curriculum.should_self_improve():
                self_improve_scenario = self.curriculum.generate_self_improvement_scenario()
                if self_improve_scenario:
                    self.scenario = self_improve_scenario
                    self._inject_scenario_with_intents(self.scenario)
                    self.history = []
                    self._step_count = 0
                    episode_id = str(uuid4())
                    self.max_steps = int(MAX_STEPS + 6 * difficulty)
                    self._state = AutoDriveState(
                        episode_id=episode_id,
                        step_count=0,
                        incident_id=self.scenario.get("name", ""),
                        difficulty=difficulty,
                        incident_type=self.scenario.get("type", ""),
                        root_cause=self.scenario.get("root_cause", ""),
                        correct_fix=self.scenario.get("correct_fix_description", ""),
                        judge_persona=self.curriculum.get_judge_persona(),
                        curriculum_stats=self.curriculum.get_stats(),
                    )
                    self.reward_tracker.start_episode(
                        self._episode_num, episode_id,
                        self.scenario.get("type", "adversarial"),
                        difficulty, self.curriculum.get_tier_name(),
                    )
                    obs = self.backend.build_observation(
                        steps_taken=0, max_steps=self.max_steps,
                        hint=self._hint(),
                        metadata={"scenario": self.scenario, "judge_persona": self.curriculum.get_judge_persona()},
                    )
                    obs["scenario_type"] = self.scenario.get("type", "")
                    obs["judge_persona"] = self.curriculum.get_judge_persona()
                    obs["stage_scores"] = {}
                    obs["validation"] = {}
                    obs["resolution"] = {}
                    obs["zone_cues"] = self.scenario.get("zone_cues", {}) or {}
                    obs["belief_state"] = {}
                    result = self._to_observation(obs, reward=0.0, done=False)
                    result.self_improve_context = {
                        "triggered": True,
                        "weak_spots": self.curriculum.get_weak_spots(),
                        "trigger_count": self.curriculum.get_stats().get("self_improve_triggered", 0),
                    }
                    return result
            incident_type = task_id or self.curriculum.pick_fault_type()

        if incident_type == "adversarial":
            self.scenario = self.designer.design(self.curriculum.get_skill_profile(), difficulty)
        else:
            self.scenario = self.generator.generate(self.curriculum.get_skill_profile(), difficulty, fault_type_hint=incident_type).__dict__
        self._inject_scenario_with_intents(self.scenario)
        self.history = []
        self._step_count = 0
        self.max_steps = int(
            (self.route_planner.current_max_steps() if self._route_mode else MAX_STEPS)
            + 6 * difficulty
        )
        episode_id = str(uuid4())
        self._state = AutoDriveState(
            episode_id=episode_id,
            step_count=0,
            incident_id=self.scenario.get("name", ""),
            difficulty=difficulty,
            incident_type=self.scenario.get("type", ""),
            root_cause=self.scenario.get("root_cause", ""),
            correct_fix=self.scenario.get("correct_fix_description", ""),
            judge_persona=self.curriculum.get_judge_persona(),
            curriculum_stats=self.curriculum.get_stats(),
        )
        # Start reward tracking for this episode
        self.reward_tracker.start_episode(
            self._episode_num, episode_id, self.scenario.get("type", ""),
            difficulty, self.curriculum.get_tier_name(),
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
        obs["zone_cues"] = self.scenario.get("zone_cues", {}) or {}
        obs["belief_state"] = {}
        result = self._to_observation(obs, reward=0.0, done=False)
        # Add route state if in route mode
        if self._route_mode and self.route_planner:
            result.route_state = self.route_planner.state.to_dict()
        return result

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

        # Count only CONSECUTIVE trailing repetitions — not all occurrences.
        # (Bug fix: counting all history meant correct repeated braking killed reward to 0)
        consecutive_count = 0
        for _item in reversed(self.history):
            if _item.get("action") == action.action:
                consecutive_count += 1
            else:
                break
        # Only penalise after 2 identical actions in a row (oscillation starts at 3rd)
        repeat_penalty = -0.18 * max(0, consecutive_count - 1) if consecutive_count > 1 else 0.0

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

        reward = self._compute_reward(validation, judge_score, repeat_penalty, resolution_bonus, done, success, action, consecutive_count, before)

        # Educational feedback — what did the agent do wrong this step?
        current_stage = self.backend.simulator.current_stage
        last_mistake = self._identify_mistake(action, validation, current_stage, before)

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

        # ── Reward tracker ─────────────────────────────────────────────────────
        self.reward_tracker.record_step(
            self._step_count, action.action, action.value, reward,
            collision=bool(validation.get("collision", False)),
            near_miss=bool(validation.get("near_miss", False)),
            safe_distance=bool(validation.get("safe_distance", True)),
            progress_restored=bool(validation.get("progress_restored", False)),
        )

        # ── Route mode: checkpoint transitions ──────────────────────────────
        route_extra_reward = 0.0
        if self._route_mode and self.route_planner:
            self.route_planner.record_step()
            if done and success:
                cp_reward, _route_done = self.route_planner.on_checkpoint_success()
                # Fold a small fraction of the checkpoint reward into this step
                route_extra_reward = min(0.05, cp_reward * 0.005)
            elif done and not success:
                self.route_planner.on_checkpoint_failure()

        reward = min(1.0 - 1e-3, reward + route_extra_reward)

        if done:
            route_pct = self.route_planner.state.progress_pct() if self._route_mode and self.route_planner else 0.0
            self.reward_tracker.end_episode(success, route_progress_pct=route_pct)
            self.reward_tracker.save()

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
        # current_stage already computed above for _identify_mistake; reuse it here

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

        # Educational mistake feedback — helps agent understand what to improve
        obs["last_mistake"] = last_mistake

        # ── Theme 3.1: Zone cues (indirect signals — agent must infer zone) ───
        zone_cues = self.scenario.get("zone_cues", {}) or {}
        obs["zone_cues"] = zone_cues

        # ── Theme 3.2: Bayesian belief state (Theory of Mind) ─────────────────
        sensor_objects = obs.get("sensor_data", {}).get("objects", []) or []
        self.belief_tracker.update_from_sensor_objects(sensor_objects)
        obs["belief_state"] = self.belief_tracker.get_belief_state()

        # ── Theme 1: Run pipeline on current observation for intent + negotiation
        # Run inline so intent/negotiation traces enrich this observation.
        # The pipeline also receives zone_cues so the DecisionAgent can reason
        # about zone context from indirect signals.
        try:
            _pipe_action, pipeline_trace = self.pipeline.run(obs, self.history)
        except Exception as _pe:
            logger.debug("Pipeline run failed in step (%s); skipping trace", _pe)
            pipeline_trace = {}
        obs["pipeline_trace"]    = pipeline_trace
        obs["intent_context"]    = pipeline_trace.get("intent_inference", {})
        obs["negotiation_context"] = pipeline_trace.get("negotiation", {})

        result = self._to_observation(obs, reward=reward, done=done)
        # Attach route and self-improve context
        if self._route_mode and self.route_planner:
            result.route_state = self.route_planner.state.to_dict()
            if obs.get("hint") and self.route_planner:
                result.hint = self.route_planner.get_route_hint()
        curriculum_stats = self._state.curriculum_stats or {}
        result.self_improve_context = {
            "weak_spots": self.curriculum.get_weak_spots(),
            "trigger_count": curriculum_stats.get("self_improve_triggered", 0),
            "cooldown": curriculum_stats.get("self_improve_cooldown", 0),
        }
        return result

    def _compute_reward(self, validation: dict, judge_score: float, repeat_penalty: float, resolution_bonus: float, done: bool, success: bool, action: "AutoDriveAction | None" = None, consecutive_count: int = 0, before_obs: dict | None = None) -> float:
        """Reward shaping for Observe → Act → Reward → Learn cycle.

        The reward is strictly in (0, 1) and shaped to give the agent clear,
        immediate signals at every step — not just at episode end.

        Key design:
          - Correct braking during approach is always positive (no collapsed reward)
          - Stage transitions (approach→clearing→cleared) get an explicit bonus
          - Collision is heavily penalised but non-zero so the Q-update still runs
          - Oscillation (3+ same actions in a row) is gently penalised
          - Terminal failure penalty is soft (0.75x) not sharp (0.5x) so the
            agent still gets a gradient to learn from failed episodes
        """
        eps = 1e-3
        # map judge_score from [-1,1] → [0,1]
        base = max(-1.0, min(1.0, judge_score))
        reward = (base + 1.0) / 2.0

        # ── Multiplicative safety penalties ───────────────────────────────────
        if validation.get("collision"):
            reward *= 0.08   # heavy penalty but non-zero (preserves gradient)
        elif validation.get("near_miss"):
            reward *= 0.55
        elif validation.get("offroad"):
            reward *= 0.25

        # ── Additive bonuses for good behaviours (immediate feedback) ─────────
        if validation.get("safe_distance"):
            reward += 0.10   # clearly staying safe: +0.10
        if validation.get("signal_respected"):
            reward += 0.05

        # Stage-transition bonus — the key Observe→Act→Reward signal.
        # When the hazard clears (the agent survived the dangerous stage), give a
        # meaningful bonus so the Q-table learns that "braking during approach" leads
        # to good outcomes.
        if validation.get("incident_cleared") and validation.get("progress_restored"):
            reward += 0.20   # full success: survived + moving again
        elif validation.get("incident_cleared"):
            reward += 0.10   # partial: hazard gone, not yet moving

        # ── Oscillation penalty (soft) ────────────────────────────────────────
        # Only applies when hazard is still active (not after clearing).
        if not validation.get("incident_cleared"):
            repeat_factor = max(0.40, 1.0 + repeat_penalty)   # floor at 40% to keep gradient
            reward *= repeat_factor

        # ── Terminal outcome adjustment ────────────────────────────────────────
        # Use a soft penalty (0.75x) on failure — preserves learning signal in
        # failed episodes rather than collapsing all rewards to near-zero.
        if done and not success:
            reward *= 0.75

        # Resolution bonus (proportional, not tiny) — rewards finishing quickly
        reward += max(0.0, min(0.15, resolution_bonus * 0.02))

        # ── Over-braking penalty ──────────────────────────────────────────────
        # Case 1 (zone-based): low-density zone + hazard far + no danger
        # Case 2 (distance-based): hazard is far OR moving away — braking is
        #   unnecessary regardless of zone. This is the real-world Indian driving
        #   principle: if the obstacle is receding or far, keep moving.
        if action is not None and before_obs is not None:
            _zone = (self.scenario or {}).get("zone_cues", {}) or {}
            _density = _zone.get("pedestrian_density", "medium")
            _hd = float((before_obs or {}).get("hazard_distance", 999.0) or 999.0)
            _act = action.action
            _no_danger = not validation.get("near_miss") and not validation.get("collision")
            _moving_away = bool((before_obs or {}).get("hazard_moving_away", False))

            # Case 1: low-density zone, hazard distant
            if (_density in ("very_low", "none", "low")
                    and _hd > 20.0
                    and _act in ("brake", "wait")
                    and consecutive_count >= 2
                    and _no_danger):
                reward *= 0.72  # unnecessary caution — zone clear, hazard distant

            # Case 2: hazard distance > 15m AND moving away from ego
            # Applies regardless of zone type — this is a real-world physics signal.
            # At 15m+ with the hazard receding, braking repeated times wastes time.
            elif (_moving_away
                    and _hd > 15.0
                    and _act in ("brake", "wait")
                    and consecutive_count >= 2
                    and _no_danger):
                reward *= 0.78  # hazard is moving away — ease off brakes

            # Case 3: hazard > 20m and no on-road danger (general over-caution)
            elif (_hd > 20.0
                    and _act in ("brake", "wait")
                    and consecutive_count >= 3
                    and _no_danger
                    and not validation.get("safe_distance")):
                reward *= 0.82  # gently discourage prolonged braking far from any threat

        # Clamp into (eps, 1-eps) — never exactly 0 or 1
        reward = max(eps, min(1.0 - eps, reward))
        return round(reward, 3)

    def _identify_mistake(self, action: "AutoDriveAction", validation: dict, stage: str, before_obs: dict) -> str:
        """Return a short, specific educational note on what the agent did wrong.

        This surfaces actionable feedback in the observation so the agent (or
        developer debugging it) understands the cause of low reward — without
        giving away the optimal action directly.
        """
        hd = float(before_obs.get("hazard_distance", 999.0) or 999.0)
        act = action.action

        if validation.get("collision"):
            return f"Collision at {hd:.0f}m — needed earlier/harder braking before reaching {hd:.0f}m"
        if validation.get("near_miss") and act == "accelerate":
            return f"Near-miss: accelerated with hazard at {hd:.0f}m — should brake first"
        if stage in ("clearing", "cleared") and act in ("brake", "wait"):
            return f"Hazard cleared but still {act}ing — accelerate to restore progress"
        if stage == "approaching" and act == "accelerate" and hd < 15.0:
            return f"Accelerated toward hazard at {hd:.0f}m during approach — brake or wait"
        if validation.get("stuck"):
            return "Stuck: no forward progress — try change_lane or alternate between wait/accelerate"
        # Braking while hazard is actively moving away
        _moving_away = bool(before_obs.get("hazard_moving_away", False))
        if _moving_away and hd > 12.0 and act in ("brake", "wait"):
            return (
                f"Hazard is MOVING AWAY (now {hd:.0f}m and increasing) — "
                f"ease off brakes. Prepare to accelerate."
            )
        # Braking far from any hazard (general over-caution)
        if hd > 18.0 and act in ("brake", "wait") and not before_obs.get("active_alerts"):
            return (
                f"Hazard is {hd:.0f}m away with no active alerts — "
                f"no need to brake. Accelerate or maintain speed."
            )
        # Over-braking in a low-density / clear zone
        _zone = (self.scenario or {}).get("zone_cues", {}) or {}
        _density = _zone.get("pedestrian_density", "medium")
        if _density in ("very_low", "none", "low") and hd > 20.0 and act in ("brake", "wait"):
            return (f"Unnecessary braking — pedestrian density is '{_density}' and road is clear "
                    f"at {hd:.0f}m. Maintain or resume normal speed.")
        return ""

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

    def _inject_scenario_with_intents(self, scenario: dict) -> None:
        """Assign hidden intents to scenario actors, then inject into the backend.

        Intents are stored on the actor dicts (hidden_intent key) so the
        backend simulator can simulate intent-driven behaviors.  They are
        NEVER forwarded to the agent observation — only behavioral signals are.
        """
        actors = scenario.get("actors", [])
        if actors:
            assign_intents(actors)
        self.backend.inject_scenario(scenario)

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
            road_geometry=payload.get("road_geometry", {}),
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
            pipeline_trace=payload.get("pipeline_trace", {}),
            fleet_context=payload.get("fleet_context", {}),
            route_state=payload.get("route_state", {}),
            self_improve_context=payload.get("self_improve_context", {}),
            intent_context=payload.get("intent_context", {}),
            negotiation_context=payload.get("negotiation_context", {}),
            zone_cues=payload.get("zone_cues", {}),
            belief_state=payload.get("belief_state", {}),
        )

    @property
    def state(self) -> AutoDriveState:
        return self._state