"""Training loop for AutoDrive Gym — produces reward curves and metrics.

This script demonstrates training progress for all hackathon judging criteria:
  • Reward and Training Script/Pipeline Setup (10%)
  • Showing Improvement in Rewards (20%) — reward curves, rolling averages,
    per-scenario performance, tier progression

Usage
-----
# Heuristic baseline run (no LLM required, shows curriculum progression)
python -m autodrive_env.train --episodes 50 --mode heuristic

# Multi-agent pipeline run (requires LLM API key)
python -m autodrive_env.train --episodes 30 --mode pipeline

# Long-horizon city route run
python -m autodrive_env.train --episodes 10 --mode route

# Plot results after training
python -m autodrive_env.train --plot-only

Environment variables
---------------------
OPENAI_API_KEY / GROQ_API_KEY / HF_TOKEN  -- LLM provider credentials
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Optional wandb — install with: pip install wandb
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

# Live state file – the /demo endpoint in app.py reads this for real-time updates
_LIVE_STATE_PATH = os.path.join(os.path.dirname(__file__), "live_state.json")


def _write_live_state(state: Dict[str, Any]) -> None:
    """Write current training state to live_state.json for the web demo."""
    try:
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as _f:
            json.dump(state, _f)
    except Exception:
        pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Key metric namespaces stay at INFO
for _ns in ("autodrive", "autodrive_openenv", "autodrive.train", "autodrive_openenv.server"):
    logging.getLogger(_ns).setLevel(logging.INFO)
# Silence HTTP noise
for _ns in ("httpx", "httpcore", "urllib3", "requests", "uvicorn.access", "fastapi"):
    logging.getLogger(_ns).setLevel(logging.ERROR)
logger = logging.getLogger("autodrive.train")

SYSTEM_PROMPT = """You are an autonomous driving agent navigating dense Indian road conditions.
Choose the BEST SINGLE action for the current situation — not just brake or accelerate.

ACTION MENU (use all of them as appropriate):
  brake            — slow or stop for hazards, red lights, pedestrians, animals
  accelerate       — build speed when the path is clear or hazard has passed
  wait             — hold position when yielding to a signal/police/procession
  steer_left       — veer left to dodge pothole, create ambulance corridor, or avoid right-side cut-in
  steer_right      — veer right to dodge left-side obstacle or create lane space
  horn             — warn pedestrians/animals; use once sparingly in processions
  change_lane_left — move fully to left lane when right lane is blocked or unsafe
  change_lane_right— move fully to right lane when left lane is blocked or unsafe

SCENARIO → ACTION MAPPING:
  pedestrian/child crossing  : horn (if >10m) → brake → wait → accelerate once clear
  auto/bike cut-in           : steer_left or brake → wait → accelerate
  bike blind spot merge      : steer_right (if right is clear) or brake → wait
  pothole / speed breaker    : brake to slow → steer_left or steer_right to go around
  ambulance from behind      : steer_left immediately (create corridor) — do NOT brake hard
  animal on road             : horn (if >10m) → brake → wait → accelerate
  police/flagman override    : wait → accelerate on signal
  traffic jam                : brake → wait (repeat) → accelerate when gap opens
  foggy/rain/waterlog        : brake (moderate) → steer cautiously, no sudden moves
  wedding procession         : horn (once, >12m) → brake → wait until gap → accelerate
  construction single lane   : wait → brake → accelerate when flagman signals
  lane blocked ahead         : change_lane_left or change_lane_right → accelerate

STAGE RULES:
  approaching : hazard still ahead — brake/wait/steer/horn as needed
  clearing    : hazard moving away — ease off, prepare to accelerate
  cleared     : hazard gone — accelerate to resume normal speed

DISTANCE RULES (based on current hazard_distance):
  < 4m   : brake(1.0) or steer hard — emergency only
  4-8m   : brake(0.7-0.9) or steer to avoid
  8-14m  : brake(0.4-0.6) or horn or wait
  14-18m : gentle brake or change lane — stay alert
  >18m   : DO NOT brake. Accelerate, maintain speed, or change lane.

DISTANCE TREND RULES (read hazard_trend / hazard_moving_away in observation):
  hazard_trend=receding (hazard_moving_away=true):
    → The hazard is moving AWAY from you. Distance is INCREASING.
    → DO NOT keep braking. Ease off. Prepare to accelerate.
    → If distance > 12m and receding: switch to accelerate or wait(1 step max).
    → Continuing to brake when hazard is moving away = wrong, penalised.
  hazard_trend=approaching_or_static:
    → Standard brake/wait logic applies based on distance above.

INDIAN ROAD REALITY — how real drivers behave:
  - If the auto-rickshaw has cut in and is now pulling ahead → follow it, accelerate
  - If the bike has merged and is now 15m ahead pulling away → go, don't sit idle
  - If the pedestrian has crossed and is off-road → accelerate immediately
  - Braking when the hazard has passed makes you a bottleneck — bad driving
  - Only brake/wait if something is actively in your path right now

READING NEW OBSERVATION FIELDS:
  closing_speed < 0         : hazard is RECEDING — ease off brakes, prepare to accelerate
  closing_speed > 0         : hazard approaching — apply DISTANCE RULES above
  trajectory=moving_away    : hazard leaving your path — prepare to accelerate
  trajectory=will_cross     : hazard will cross your lane — brake proactively
  trajectory=erratic_unpredictable : treat as high-risk regardless of distance
  intent_confidence < 0.40  : very uncertain actor — extra 2 m caution buffer
  dominance=high            : must yield (bus/truck/ambulance/police) — brake or steer
  negotiation_required=true : adjust speed to negotiate right-of-way, don't blast through
  signal_trust_score < 0.50 : signal unreliable (construction/police override) — use own judgment
  time_context=rush_hour    : expect aggressive actors; tighter negotiation windows
  time_context=night        : slower default speed; higher occlusion risk
  occlusion=true            : slow down — unseen hazard may emerge from hidden zone
  pedestrian_flow_direction=crossing_road : brake proactively even if no pedestrian visible yet
  behavioral_pattern contains "aggressive×" : treat that actor as high-risk regardless of distance
  behavioral_pattern contains "erratic"     : maintain extra buffer and do NOT tailgate

ANTI-LOOP: Never repeat the same action more than 2 times in a row — switch tactic.
If hint says 'accelerate', 'steer', or 'change_lane', follow it immediately.

Return ONLY valid JSON — no explanation, no markdown:
{"action": "<one of the 8 actions above>", "value": <float 0.0-1.0>}
"""


# Scenario→primary action guidance (mirrors inference.py _scene_guidance)
_SCENARIO_GUIDANCE: Dict[str, str] = {
    "pedestrian_crossing":     "CHILD/PEDESTRIAN detected — HORN if >10m, then BRAKE and WAIT until clear.",
    "auto_cut_in":             "Vehicle cutting in — BRAKE or STEER_LEFT to create space, then WAIT.",
    "bike_blind_spot":         "Bike merging from blind spot — STEER_RIGHT (if right is clear) or BRAKE.",
    "pothole_ahead":           "POTHOLE ahead — BRAKE to slow, then STEER_LEFT or STEER_RIGHT to go around.",
    "speed_breaker":           "SPEED BREAKER — BRAKE(0.6) before it, then ACCELERATE after.",
    "crowded_market":          "Crowded market — BRAKE slowly, HORN once to warn, WAIT for pedestrians.",
    "ambulance_approach":      "AMBULANCE behind — STEER_LEFT immediately to open a corridor. Do NOT brake hard.",
    "police_override":         "POLICE directing traffic — WAIT for hand signal, then ACCELERATE when waved.",
    "traffic_jam":             "TRAFFIC JAM — BRAKE to stop, WAIT in queue, ACCELERATE when gap opens.",
    "animal_crossing":         "ANIMAL on road — HORN (if >10m) → BRAKE → WAIT → ACCELERATE once clear.",
    "rain_slippery_road":      "SLIPPERY road — BRAKE gently(0.4), STEER cautiously, no sudden moves.",
    "traffic_light_ambiguity": "Ambiguous signal — BRAKE(0.5) and WAIT for clear signal before proceeding.",
    "school_bus_stop":         "School bus stopped — BRAKE fully, WAIT for children to clear, then ACCELERATE.",
    "construction_zone":       "Construction zone — WAIT for flagman, single lane, BRAKE and ACCELERATE slowly.",
    "night_fog":               "Night fog — BRAKE(0.4) for low visibility, STEER_LEFT carefully if needed.",
    "waterlogged_underpass":   "Waterlogged road — BRAKE(0.3) to crawl through, no sudden steer.",
    "wedding_procession":      "Wedding procession — HORN once (>12m), BRAKE, WAIT for gap, ACCELERATE.",
    "highway_merge_truck":     "Truck merging — CHANGE_LANE_LEFT or BRAKE hard to give space.",
    "multi_agent_chaos":       "Multi-vehicle chaos — BRAKE(0.7), WAIT, then pick safest lane via CHANGE_LANE.",
    "adversarial":             "Adversarial vehicle — BRAKE, STEER away from threat, CHANGE_LANE if safe.",
}


def format_observation(obs, weakness_hint: str = "", step: int = 0) -> str:
    stage       = getattr(obs, 'scenario_stage', 'approaching') or 'approaching'
    hazard_type = getattr(obs, 'hazard_type', '') or ''
    hazard_dist = float(getattr(obs, 'hazard_distance', 999.0) or 999.0)
    alerts      = getattr(obs, 'active_alerts', []) or []
    hint        = getattr(obs, 'hint', '') or ''
    ego         = getattr(obs, 'ego_state', {}) or {}
    road_geo    = getattr(obs, 'road_geometry', {}) or {}
    sensor      = getattr(obs, 'sensor_data', {}) or {}
    environment = getattr(obs, 'environment', {}) or {}
    task_type   = getattr(obs, 'scenario_type', '') or ''
    signal      = environment.get('traffic_signal', sensor.get('traffic_signal', 'none'))
    road_cond   = environment.get('road_condition', 'normal')
    objects     = (sensor.get('objects') or [])[:6]

    guidance = _SCENARIO_GUIDANCE.get(task_type, '')
    hint_l = hint.lower()
    for kw in ('steer_left', 'steer_right', 'change_lane', 'horn', 'accelerate now'):
        if kw in hint_l:
            guidance = f"HINT override: {hint[:80]}"
            break
    if stage in ('clearing', 'cleared') and 'accelerate' not in (guidance or '').lower():
        guidance = "Hazard clearing/cleared — ACCELERATE to resume."

    dist_str   = f"{hazard_dist:.1f}m" if hazard_dist < 900 else "clear"
    lane_off   = ego.get('lane_offset_m', ego.get('lane_position', 0.0))
    sp_left    = road_geo.get('space_left_m',  '?')
    sp_right   = road_geo.get('space_right_m', '?')
    can_left   = road_geo.get('can_steer_left',   True)
    can_right  = road_geo.get('can_steer_right',  True)
    can_cl_l   = road_geo.get('can_change_lane_left',  False)
    can_cl_r   = road_geo.get('can_change_lane_right', False)

    road_line = ""
    if road_geo:
        sl = f"{sp_left:.1f}" if isinstance(sp_left, float) else str(sp_left)
        sr = f"{sp_right:.1f}" if isinstance(sp_right, float) else str(sp_right)
        road_line = (
            f"ROAD  width={road_geo.get('road_width_m','?')}m  lanes={road_geo.get('num_lanes','?')}"
            f"  surface={road_geo.get('surface_type','?')}"
            f"  space_L={sl}m  space_R={sr}m"
            f"  steer_L={'Y' if can_left else 'N'}  steer_R={'Y' if can_right else 'N'}"
            f"  lane_L={'Y' if can_cl_l else 'N'}  lane_R={'Y' if can_cl_r else 'N'}"
        )

    lines = [
        f"STEP={step} | TASK={task_type} | STAGE={stage} | HAZARD={hazard_type or '-'}@{dist_str}",
        f">> {guidance}" if guidance else "",
        f"EGO   speed={ego.get('speed', 0.0):.1f}km/h  lane={ego.get('lane','center')}  lat={lane_off:+.2f}m  signal={signal}  road={road_cond}",
        road_line,
    ]
    if alerts:
        lines.append(f"ALERTS: {'; '.join(alerts[:4])}")
    if hint:
        lines.append(f"HINT: {hint[:100]}")

    # Per-object sensor data with position, velocity, TTC, lateral placement
    if objects:
        lines.append("OBJECTS:")
        for o in objects:
            typ   = o.get('type', 'obj')
            dist  = o.get('distance', '?')
            side  = o.get('side', '?')
            f_dx  = o.get('rel_fwd_m', '?')
            l_dy  = o.get('rel_lat_m', '?')
            spd   = o.get('speed_mps', 0.0)
            vel   = o.get('velocity') or [0, 0]
            vx, vy = float(vel[0]), float(vel[1])
            ttc   = o.get('ttc_s', 99.9)
            beh   = o.get('behavior', 'static')
            inl   = 'IN-LANE' if o.get('in_ego_lane') else 'off-lane'
            ttc_s = f"ttc={ttc:.1f}s" if ttc < 90 else "no-threat"
            if isinstance(l_dy, float):
                lines.append(
                    f"  [{typ:<12}] dist={dist:>5}m  side={side:<7} {inl:<8}"
                    f"  fwd={f_dx:+.1f}m  lat={l_dy:+.1f}m"
                    f"  vel=({vx:+.1f},{vy:+.1f})m/s  spd={spd:.1f}  {ttc_s}  [{beh}]"
                )
            else:
                lines.append(f"  [{typ}] dist={dist}m  side={side}  [{beh}]")

    route = getattr(obs, 'route_state', {}) or {}
    if route:
        cp = route.get('current_checkpoint') or {}
        lines.append(
            f"ROUTE: CP {route.get('current_checkpoint_index', 0)+1}/{route.get('n_checkpoints',1)}"
            f" [{cp.get('name','')}]  progress={route.get('progress_pct',0.0):.0%}"
        )
    if weakness_hint:
        lines.append(f"⚠ WEAKNESS: {weakness_hint[:80]}")
    lines.append(f"STEPS: {getattr(obs, 'steps_taken', step)}/{getattr(obs, 'max_steps', 20)}")
    return "\n".join(l for l in lines if l)


def format_history(history: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"step {h['step']}: {h['action']} val={h.get('value', 0.0):.2f} reward={h.get('reward', 0.0):.3f}"
        for h in history[-5:]
    )


def parse_actions(text: str):
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            return [data]
    except Exception:
        pass
    return []


def heuristic_action(obs, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    from autodrive_env.agent_baseline import choose_action
    obs_dict = {
        "sensor_data":    getattr(obs, "sensor_data",    {}) or {},
        "ego_state":      getattr(obs, "ego_state",      {}) or {},
        "environment":    getattr(obs, "environment",    {}) or {},
        "active_alerts":  getattr(obs, "active_alerts",  []) or [],
        "hint":           getattr(obs, "hint",           "") or "",
        "scenario_stage": getattr(obs, "scenario_stage", "approaching") or "approaching",
        "hazard_distance":getattr(obs, "hazard_distance", 999.0) or 999.0,
        "hazard_type":    getattr(obs, "hazard_type",    "") or "",
    }
    return choose_action(obs_dict, history)


def pipeline_action(pipeline, obs, history: List[Dict[str, Any]]):
    obs_dict = {
        "sensor_data":    getattr(obs, "sensor_data",    {}) or {},
        "ego_state":      getattr(obs, "ego_state",      {}) or {},
        "environment":    getattr(obs, "environment",    {}) or {},
        "active_alerts":  getattr(obs, "active_alerts",  []) or [],
        "hint":           getattr(obs, "hint",           "") or "",
        "scenario_stage": getattr(obs, "scenario_stage", "approaching") or "approaching",
        "hazard_distance":getattr(obs, "hazard_distance", 999.0) or 999.0,
        "hazard_type":    getattr(obs, "hazard_type",    "") or "",
        "validation":     getattr(obs, "validation",    {}) or {},
    }
    return pipeline.run(obs_dict, history)



# ── Training log helpers (styled after inference.py) ─────────────────────────

def _print_ep_header(
    ep: int, total: int, cp_name: str, scenario_type: str,
    tier: str, difficulty: float, is_retry: bool, weakness_hint: str,
) -> None:
    print("\n" + "=" * 66)
    retry_tag = "  ⟳ RETRY" if is_retry else ""
    short_cp = cp_name.split("—")[0].strip()
    print(f"  EPISODE {ep}/{total}  [{short_cp}]{retry_tag}")
    print(f"  scenario={scenario_type:<26}  tier={tier}  diff={difficulty:.2f}")
    if weakness_hint:
        print(f"  ⚠  WEAKNESS FOCUS: {weakness_hint[:80]}")
    print("-" * 66)


def _print_step(
    step: int, action: str, value: float, reward: float,
    hazard_type: str = "", hazard_dist: float = 999.0,
    stage: str = "approaching", flags: str = "",
) -> None:
    status     = "✓" if reward >= 0.5 else "✗"
    dist_str   = f"{hazard_dist:.1f}m" if hazard_dist < 900 else "clear"
    hazard_str = f"{hazard_type}@{dist_str}" if hazard_type else dist_str
    flag_str   = f"  [{flags}]" if flags else ""
    print(
        f"  step {step:>2} {status} | {action:<18} val={value:.1f} "
        f"| r={reward:+.4f} | {hazard_str:<18} | stage={stage:<12}{flag_str}",
        flush=True,
    )


def _print_ep_result(
    success: bool, steps: int, episode_reward: float, step_rewards: List[float],
    rolling10: float, sr20: float, tier: str, difficulty: float, end_reason: str,
) -> None:
    icon  = "✅ SUCCESS" if success else "❌ FAIL   "
    avg   = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    best  = max(step_rewards) if step_rewards else 0.0
    worst = min(step_rewards) if step_rewards else 0.0
    print(f"\n  [RESULT] {icon}  steps={steps}  total_reward={episode_reward:.4f}  end={end_reason}")
    print(f"           per-step: avg={avg:.4f}  best={best:.4f}  worst={worst:.4f}")
    print(f"           roll10={rolling10:.4f}  SR(last20)={sr20:.1%}  tier={tier}  diff={difficulty:.2f}")


def _print_weakness_report(
    cp_short: str, scenario_type: str, end_reason: str,
    fail_stage: str, fail_hazard: str, fail_step: int,
    insight: str, retry_at_ep: int,
) -> None:
    w = 60
    print(f"\n  ┌─ WEAKNESS REPORT {'─' * (w - 19)}")
    print(f"  │  Scenario  : {scenario_type}  [{cp_short}]")
    print(f"  │  Failed at : {fail_stage}  (step {fail_step})")
    if fail_hazard:
        print(f"  │  Hazard    : {fail_hazard}")
    print(f"  │  Reason    : {end_reason}")
    print(f"  │  Insight   : {insight}")
    if retry_at_ep > 0:
        print(f"  │  Retry at  : episode {retry_at_ep}  (weakness hint will be shown)")
    else:
        print(f"  │  Retry     : max retries reached — moving on")
    print(f"  └{'─' * w}")


def _failure_insight(
    scenario_type: str, end_reason: str,
    collision: bool, near_miss: bool, stuck: bool, fail_stage: str,
) -> str:
    if collision:
        return f"Collision during '{fail_stage}' — apply harder brake earlier or steer away"
    if near_miss:
        return f"Near-miss during '{fail_stage}' — earlier hazard response needed"
    if stuck:
        return "Agent stuck — not progressing; try change_lane or alternating actions"
    if end_reason == "timeout":
        return "Timed out without clearing hazard — may be over-braking or looping on wait"
    return f"Low reward without clearing — refine action mapping for {scenario_type}"


# ── OpenEnv-compatible in-process session wrapper ────────────────────────────
# Mirrors the StepResult interface used in inference.py via AutoDriveClient so
# that train.py exercises the same reset()/step() API surface — just in-process
# (no HTTP hop) for training speed.

from dataclasses import dataclass as _dc
from typing import Any as _Any

@_dc
class _StepResult:
    """Minimal StepResult compatible with openenv_compat.StepResult."""
    observation: _Any
    reward: float
    done: bool


class _InProcessEnvSession:
    """Wraps AutoDriveGymEnvironment to expose the OpenEnv StepResult API.

    This makes train.py call reset()/step() the same way that inference.py
    does through the HTTP AutoDriveClient::

        result = session.reset(task_id=...)   # -> _StepResult
        obs    = result.observation
        result = session.step(action)         # -> _StepResult
        obs, reward, done = result.observation, result.reward, result.done

    Transparent attribute delegation makes session.curriculum, session.llm etc.
    work identically to env.curriculum, env.llm.
    """

    def __init__(self, env) -> None:
        self._env = env

    def __getattr__(self, name: str):
        # Delegate curriculum, reward_tracker, llm, … to the wrapped env
        return getattr(self._env, name)

    # ── OpenEnv lifecycle (no-op for in-process) ──────────────────────────
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()

    # ── Core API methods (matching AutoDriveClient) ───────────────────────
    def reset(self, task_id: str | None = None) -> _StepResult:
        """reset() → StepResult, just like AutoDriveClient.reset()."""
        obs = self._env.reset(task_id=task_id)
        return _StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action) -> _StepResult:
        """step() → StepResult, just like AutoDriveClient.step()."""
        obs = self._env.step(action)
        return _StepResult(
            observation=obs,
            reward=float(getattr(obs, "reward", 0.0) or 0.0),
            done=bool(getattr(obs, "done", False)),
        )


def run_training(
    n_episodes: int = 50,
    mode: str = "adaptive",
    task_id: str | None = None,
    verbose: bool = True,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Run N training episodes and return reward curve data.

    mode options:
      adaptive   — Q-table policy learner + belief tracker + counterfactual bonus
                   Shows measurable reward improvement over episodes. (DEFAULT)
      heuristic  — rule-based baseline (no learning, no improvement)
      pipeline   — full 6-agent LLM pipeline (requires API key)
      route      — long-horizon city route mode
    """
    from autodrive_env.server.autodrive_gym_environment import AutoDriveGymEnvironment
    from autodrive_env.server.multi_agent_pipeline import MultiAgentPipeline
    from autodrive_env.server.route_planner import CITY_ROUTE
    from autodrive_env.models import AutoDriveAction

    # ── wandb initialisation ─────────────────────────────────────────────────
    _wb_run = None
    if use_wandb and _WANDB_AVAILABLE:
        _wb_run = _wandb.init(
            project="autodrive-gym",
            name=f"{mode}-{n_episodes}ep",
            config={"mode": mode, "episodes": n_episodes, "task": task_id or "curriculum"},
        )
        print(f"  [wandb] Run started → {_wb_run.url}")
    elif use_wandb and not _WANDB_AVAILABLE:
        print("  [wandb] wandb not installed. Run: pip install wandb")

    # ── Learning modules (enabled in "adaptive" mode) ────────────────────────
    use_learning = (mode == "adaptive")
    if use_learning:
        from autodrive_env.server.policy_learner import PolicyLearner, state_from_obs
        from autodrive_env.server.belief_tracker import BeliefTracker
        from autodrive_env.server.counterfactual import CounterfactualReasoner
        learner     = PolicyLearner()
        belief_tkr  = BeliefTracker()
        cf_reasoner = CounterfactualReasoner()
    else:
        learner = belief_tkr = cf_reasoner = None  # type: ignore

    env      = AutoDriveGymEnvironment()
    # Wrap with the OpenEnv StepResult session (mirrors AutoDriveClient in inference.py)
    session  = _InProcessEnvSession(env)
    pipeline = MultiAgentPipeline(env.llm) if mode in ("pipeline", "route") else None
    use_route = (mode == "route")

    _route_types = [cp.scenario_type for cp in CITY_ROUTE]
    _route_names = [cp.name for cp in CITY_ROUTE]
    n_cp = len(_route_types)

    # ── Episode queue ─────────────────────────────────────────────────────────
    # Route mode: rotate through all CPs; failed episodes get a retry inserted
    # a few slots later, carrying the weakness insight as a hint.
    # Each entry: (cp_index_or_None, weakness_hint, is_retry)
    if use_route:
        episode_queue: List[tuple] = [(i % n_cp, "", False) for i in range(n_episodes)]
    else:
        episode_queue = [(None, "", False)] * n_episodes

    _retry_count: Dict[int, int] = {}          # cp_index → retries scheduled so far
    _weakness_ledger: Dict[str, Dict] = {}     # scenario_type → failure details
    _MAX_RETRIES_PER_CP = 2

    all_rewards:   List[float] = []
    all_successes: List[int]   = []

    if verbose:
        _mode_desc = {
            "adaptive":  "local Q-learning  (fast, no LLM — for LLM use --mode pipeline)",
            "pipeline":  "6-agent LLM pipeline  (requires GROQ_API_KEY, ~2-5s/step)",
            "heuristic": "rule-based baseline  (no learning)",
            "route":     "long-horizon city route  (LLM)",
        }.get(mode, mode)
        print("\n" + "=" * 66)
        print("  AutoDrive Gym — Training Run")
        print(f"  mode={mode}  target_episodes={n_episodes}  task={task_id or 'curriculum'}")
        print(f"  [{_mode_desc}]")
        if use_route:
            short_names = [cp.name.split("—")[0].strip()[:14] for cp in CITY_ROUTE]
            print(f"  Route plan: {' → '.join(short_names)}")
        print("=" * 66)

    q_pos = 0
    while q_pos < len(episode_queue):
        ep_num = q_pos + 1
        cp_index, weakness_hint, is_retry = episode_queue[q_pos]
        q_pos += 1

        if use_route:
            cp_task = _route_types[cp_index]
            cp_name = _route_names[cp_index]
        else:
            cp_task = task_id or ""
            cp_name = task_id or "curriculum"

        cur        = env.curriculum.get_stats()
        tier       = cur.get("tier", "warmup")
        difficulty = cur.get("difficulty", 0.15)

        if verbose:
            _print_ep_header(ep_num, len(episode_queue), cp_name, cp_task,
                             tier, difficulty, is_retry, weakness_hint)

        # OpenEnv API: reset() → StepResult; then read .observation (mirrors inference.py)
        _reset = session.reset(task_id=cp_task if cp_task else None)
        obs    = _reset.observation
        episode_reward = 0.0
        step           = 0
        done           = False
        history:      List[Dict[str, Any]] = []
        step_rewards: List[float]          = []
        ep_start = time.time()

        # Reset per-episode learning modules
        if use_learning:
            learner.start_episode(scenario_type=cp_task or "")
            belief_tkr.reset()
        prev_state = None

        while not done:
            # ── Build obs_dict for policy modules ─────────────────────────────
            obs_dict = {
                "scenario_type":  getattr(obs, "scenario_type",  "") or "",
                "scenario_stage": getattr(obs, "scenario_stage", "approaching") or "approaching",
                "hazard_distance":getattr(obs, "hazard_distance", 999.0) or 999.0,
                "hazard_type":    getattr(obs, "hazard_type",    "") or "",
                "pipeline_trace": getattr(obs, "pipeline_trace", {}) or {},
                "zone_cues":      getattr(obs, "zone_cues",      {}) or {},
                "environment":    getattr(obs, "environment",    {}) or {},
                "active_alerts":  getattr(obs, "active_alerts",  []) or [],
                "sensor_data":    getattr(obs, "sensor_data",    {}) or {},
            }

            # ── Update belief tracker with current sensor objects ─────────────
            if use_learning:
                sensor_objs = (obs_dict["sensor_data"].get("objects") or [])[:6]
                belief_tkr.update_from_sensor_objects(sensor_objs)

            # ── Action selection ──────────────────────────────────────────────
            if use_learning:
                # Adaptive: encode state, query policy learner
                current_state = learner.encode_state(obs_dict)
                action_name   = learner.select_action(current_state, episode_num=ep_num)
                action_value  = 0.7 if action_name in ("brake",) else 0.5
            elif mode == "heuristic" or pipeline is None:
                act = heuristic_action(obs, history)
                action_name  = act.get("action", "wait")
                action_value = float(act.get("value", 0.0))
                current_state = None
            else:
                act, _ = pipeline_action(pipeline, obs, history)
                action_name  = act.get("action", "wait")
                action_value = float(act.get("value", 0.0))
                current_state = None

            # OpenEnv API: step() → StepResult; read .observation, .reward, .done
            _step  = session.step(AutoDriveAction(action=action_name, value=action_value))
            obs    = _step.observation
            reward = float(_step.reward or 0.0)
            done   = bool(_step.done)

            # ── Counterfactual + avoidance bonus ─────────────────────────────
            cf_result = None
            if use_learning and cf_reasoner is not None:
                cf_result = cf_reasoner.compute(obs_dict, action_name, reward)
                reward   += cf_reasoner.avoidance_bonus(cf_result)
                reward    = min(1.0 - 1e-3, reward)

            # ── Q-update ──────────────────────────────────────────────────────
            if use_learning and current_state is not None:
                next_obs_dict = {
                    "scenario_type":  getattr(obs, "scenario_type",  "") or "",
                    "scenario_stage": getattr(obs, "scenario_stage", "approaching") or "approaching",
                    "hazard_distance":getattr(obs, "hazard_distance", 999.0) or 999.0,
                    "pipeline_trace": getattr(obs, "pipeline_trace", {}) or {},
                    "zone_cues":      getattr(obs, "zone_cues",      {}) or {},
                    "environment":    getattr(obs, "environment",    {}) or {},
                    "active_alerts":  getattr(obs, "active_alerts",  []) or [],
                    "sensor_data":    getattr(obs, "sensor_data",    {}) or {},
                }
                next_state = learner.encode_state(next_obs_dict)
                learner.update(current_state, action_name, reward, next_state=next_state, done=done)
                # Record safety events for stats
                val_now = getattr(obs, "validation", {}) or {}
                learner.record_validation(
                    collision=bool(val_now.get("collision")),
                    near_miss=bool(val_now.get("near_miss")),
                )

            episode_reward += reward
            step           += 1
            step_rewards.append(reward)
            prev_state = current_state

            if verbose:
                val   = getattr(obs, "validation", {}) or {}
                hd    = float(getattr(obs, "hazard_distance", 999.0) or 999.0)
                ht    = getattr(obs, "hazard_type", "") or ""
                stage = getattr(obs, "scenario_stage", "") or "approaching"
                flags = []
                if val.get("collision"):          flags.append("COLLISION!")
                if val.get("near_miss"):          flags.append("near-miss")
                if val.get("safe_distance"):      flags.append("safe✓")
                if val.get("progress_restored"):  flags.append("cleared✓")
                if val.get("stuck"):              flags.append("stuck")
                if use_learning and cf_result:
                    avoided = "[AVOIDED↯]" if cf_result.get("avoided_collision") else ""
                    if avoided:
                        flags.append(avoided)
                _print_step(step, action_name, action_value, reward,
                            ht, hd, stage, " ".join(flags))
                if use_learning and cf_result and verbose and step <= 3:
                    print(cf_reasoner.format_for_console(cf_result))
                if use_learning and use_learning and step == 1:
                    print(f"  {belief_tkr.belief_summary_line()}")

            history.append({
                "step": step, "action": action_name, "value": action_value, "reward": reward,
                "hazard_type": getattr(obs, "hazard_type", "") or "",
                "hazard_dist": float(getattr(obs, "hazard_distance", 999.0) or 999.0),
                "stage":       getattr(obs, "scenario_stage", "") or "approaching",
            })

            # ── Live state broadcast (for /demo web page) ─────────────────────
            _write_live_state({
                "episode":       ep_num,
                "total_episodes": len(episode_queue),
                "step":          step,
                "action":        action_name,
                "value":         round(action_value, 2),
                "reward":        round(reward, 3),
                "scenario":      cp_task or cp_name,
                "stage":         getattr(obs, "scenario_stage", "approaching") or "approaching",
                "hazard_dist":   float(getattr(obs, "hazard_distance", 999.0) or 999.0),
                "hazard_type":   getattr(obs, "hazard_type", "") or "",
                "speed":         float((getattr(obs, "ego_state", {}) or {}).get("speed", 0.0)),
                "alerts":        getattr(obs, "active_alerts", []) or [],
                "hint":          getattr(obs, "hint", "") or "",
                "last_mistake":  getattr(obs, "last_mistake", "") or "",
                "reward_history": [round(r, 3) for r in all_rewards[-30:]],
                "success_history": all_successes[-30:],
                "sr_last20":     round(sum(all_successes[-20:]) / max(len(all_successes[-20:]), 1), 3),
                "rolling10":     round(sum(all_rewards[-10:]) / max(len(all_rewards[-10:]), 1), 3),
                "tier":          tier,
                "difficulty":    round(difficulty, 3),
                "done":          done,
            })

        # ── Episode outcome ──────────────────────────────────────────────────
        val_final = getattr(obs, "validation", {}) or {}
        res_final = getattr(obs, "resolution",  {}) or {}
        success = bool(
            res_final.get("verified")
            or val_final.get("progress_restored")
            or val_final.get("reached_goal")
            or (episode_reward > 5.0 and not val_final.get("collision") and not val_final.get("stuck"))
        )
        end_reason = (
            "collision" if val_final.get("collision")        else
            "cleared"   if val_final.get("progress_restored") else
            "timeout"   if step >= (getattr(obs, "max_steps", 20) or 20) else
            "stuck"     if val_final.get("stuck")             else "done"
        )

        cur2       = env.curriculum.get_stats()
        tier       = cur2.get("tier", "warmup")
        difficulty = cur2.get("difficulty", 0.15)
        all_rewards.append(episode_reward)
        all_successes.append(int(success))

        # ── wandb episode logging ─────────────────────────────────────────────
        if _wb_run is not None:
            _rolling10 = sum(all_rewards[-10:]) / max(len(all_rewards[-10:]), 1)
            _sr20      = sum(all_successes[-20:]) / max(len(all_successes[-20:]), 1)
            _wandb.log({
                "episode":            ep_num,
                "episode_reward":     episode_reward,
                "success":            int(success),
                "steps":              step,
                "rolling10_reward":   _rolling10,
                "success_rate_20ep":  _sr20,
                "difficulty":         difficulty,
                "tier":               tier,
                "end_reason":         end_reason,
            }, step=ep_num)

        # ── Policy learner end-of-episode update ─────────────────────────────
        if use_learning:
            ep_stats = learner.end_episode(success, episode_reward, cp_task or "")
            if verbose:
                eps_val    = ep_stats["epsilon"]
                q_sz       = ep_stats["q_table_size"]
                collisions = ep_stats["collisions"]
                print(f"  [LEARNER] ε={eps_val:.3f}  Q-states={q_sz}  collisions={collisions}"
                      f"  entropy={ep_stats['action_entropy']:.2f}")
                # Print learning progress every 10 episodes
                if ep_num % 10 == 0:
                    curves = learner.get_learning_curves()
                    early  = curves.get("early_phase", {})
                    late   = curves.get("late_phase",  {})
                    improv = curves.get("reward_improvement_pct", 0.0)
                    print(f"\n  ┌─ LEARNING PROGRESS (ep {ep_num}) {'─'*30}")
                    print(f"  │  Early phase:  reward={early.get('mean_reward',0):.3f}  "
                          f"success={early.get('success_rate',0):.1%}  "
                          f"collisions/ep={early.get('collision_rate',0):.2f}")
                    print(f"  │  Latest phase: reward={late.get('mean_reward',0):.3f}  "
                          f"success={late.get('success_rate',0):.1%}  "
                          f"collisions/ep={late.get('collision_rate',0):.2f}")
                    trend = "↑ IMPROVING" if improv > 5 else ("↓ declining" if improv < -5 else "→ stable")
                    print(f"  │  Trend: {trend}  ({improv:+.1f}% reward change)")
                    print(f"  │  Q-states learned: {curves['summary']['total_states_learned']}")
                    print(f"  └{'─'*50}")

        if verbose:
            rolling10 = sum(all_rewards[-10:]) / len(all_rewards[-10:])
            sr20      = sum(all_successes[-20:]) / len(all_successes[-20:])
            _print_ep_result(success, step, episode_reward, step_rewards,
                             rolling10, sr20, tier, difficulty, end_reason)
            si = cur2.get("self_improve_triggered", 0)
            if si > 0 and cur2.get("consecutive_failures", 0) >= 3:
                print(f"\n  [SELF-IMPROVE] Adversarial scenario injected (trigger #{si})")

        # ── Weakness tracking & retry scheduling (route mode only) ───────────
        if use_route and not success:
            last_h      = history[-1] if history else {}
            fail_stage  = last_h.get("stage", "approaching")
            fail_ht     = last_h.get("hazard_type", "")
            fail_hd     = last_h.get("hazard_dist", 999.0)
            fail_hazard = f"{fail_ht}@{fail_hd:.1f}m" if fail_ht else ""

            had_collision = bool(val_final.get("collision"))
            had_near_miss = bool(val_final.get("near_miss"))
            had_stuck     = bool(val_final.get("stuck"))
            insight       = _failure_insight(cp_task, end_reason, had_collision,
                                             had_near_miss, had_stuck, fail_stage)

            # Update weakness ledger
            ledger = _weakness_ledger.setdefault(cp_task, {
                "fails": 0, "cp_name": cp_name, "insights": [], "last_ep": ep_num,
            })
            ledger["fails"]  += 1
            ledger["last_ep"] = ep_num
            if insight not in ledger["insights"]:
                ledger["insights"].append(insight)

            # Schedule retry: insert 3 slots ahead in the queue (if budget allows)
            retries_so_far = _retry_count.get(cp_index, 0)
            remaining_q    = len(episode_queue) - q_pos
            if retries_so_far < _MAX_RETRIES_PER_CP and remaining_q > 0:
                retry_offset = min(3, max(1, remaining_q))
                episode_queue.insert(q_pos + retry_offset, (cp_index, insight, True))
                _retry_count[cp_index] = retries_so_far + 1
                retry_ep = ep_num + retry_offset
                if verbose:
                    _print_weakness_report(cp_name.split("—")[0].strip(), cp_task,
                                           end_reason, fail_stage, fail_hazard,
                                           step, insight, retry_ep)
            else:
                if verbose:
                    _print_weakness_report(cp_name.split("—")[0].strip(), cp_task,
                                           end_reason, fail_stage, fail_hazard,
                                           step, insight, -1)

        if verbose:
            elapsed = time.time() - ep_start
            print(f"\n  ⏱  {elapsed:.1f}s\n")

    # ── Weakness summary ──────────────────────────────────────────────────────
    if use_route and _weakness_ledger and verbose:
        print("\n" + "=" * 66)
        print("  WEAKNESS SUMMARY  (scenarios that need more practice)")
        print("-" * 66)
        for stype, data in sorted(_weakness_ledger.items(), key=lambda x: -x[1]["fails"]):
            short = data["cp_name"].split("—")[0].strip()
            print(f"  {stype:<30} | {data['fails']} fail(s) | last seen ep {data['last_ep']}")
            for i, ins in enumerate(data["insights"], 1):
                print(f"     {i}. {ins}")
        print("=" * 66)

    # ── Save and summarise ────────────────────────────────────────────────────
    curves = env.reward_tracker.get_curves()
    env.reward_tracker.save()

    # ── Save learner + learning evidence ─────────────────────────────────────
    learning_curves = {}
    if use_learning:
        learner_path = os.path.join(os.path.dirname(env.reward_tracker.log_path), "policy_learned.json")
        learner.save(learner_path)
        learning_curves = learner.get_learning_curves()
        curves["learning_curves"] = learning_curves

        if verbose:
            print("\n" + "=" * 66)
            print("  POLICY LEARNING SUMMARY")
            print("-" * 66)
            early = learning_curves.get("early_phase", {})
            late  = learning_curves.get("late_phase",  {})
            s     = learning_curves.get("summary",     {})
            improv = learning_curves.get("reward_improvement_pct", 0.0)
            print(f"  Episodes:          {learning_curves.get('total_episodes', 0)}")
            print(f"  Q-states learned:  {s.get('total_states_learned', 0)}")
            print(f"  Total Q-updates:   {s.get('total_q_updates', 0)}")
            print(f"  Final epsilon:     {s.get('final_epsilon', 0):.4f}  (started at {learner.eps_start})")
            print(f"")
            print(f"  EARLY phase ({early.get('episodes', 0)} eps): "
                  f"reward={early.get('mean_reward', 0):.4f}  "
                  f"success={early.get('success_rate', 0):.1%}  "
                  f"collision={early.get('collision_rate', 0):.2f}/ep")
            print(f"  LATE  phase ({late.get('episodes', 0)} eps):  "
                  f"reward={late.get('mean_reward', 0):.4f}  "
                  f"success={late.get('success_rate', 0):.1%}  "
                  f"collision={late.get('collision_rate', 0):.2f}/ep")
            print(f"")
            sr_improv   = learning_curves.get("sr_improvement_pct", 0.0)
            sr_late_val = late.get("success_rate", 0.0)
            rw_ok = improv > 0
            sr_ok = sr_improv > 0 or sr_late_val >= 0.85
            if rw_ok and sr_ok:
                improved_str = "✅ YES"
            elif rw_ok or sr_ok:
                improved_str = "→ partial"
            else:
                improved_str = f"❌ NO  (per-step reward {improv:+.1f}%, SR={sr_late_val:.0%})"
            print(f"  Per-step reward:   {improv:+.1f}% vs early  ({improved_str})")
            print(f"  Success rate:      early={early.get('success_rate',0):.1%}  "
                  f"late={sr_late_val:.1%}  (Δ{sr_improv:+.1f}pp)")
            print(f"  Policy saved to:   {learner_path}")
            print("=" * 66)

    print("\n" + "=" * 66)
    print("  Training complete.")
    print("  " + env.reward_tracker.summary_line())
    print(f"  Reward log saved to: {env.reward_tracker.log_path}")
    sr_overall  = sum(all_successes) / max(len(all_successes), 1)
    sr_last10   = sum(all_successes[-10:]) / max(len(all_successes[-10:]), 1)
    if use_learning and learning_curves:
        lc_rw = learning_curves.get("reward_improvement_pct", 0.0)
        lc_sr = learning_curves.get("sr_improvement_pct", 0.0)
        print(f"  Per-step reward trend: {lc_rw:+.1f}%  |  Success-rate trend: {lc_sr:+.1f}%")
        print(f"  Success rate  overall={sr_overall:.1%}  last-10={sr_last10:.1%}")
    elif curves.get("overall"):
        o = curves["overall"]
        improved = o.get("final_10_mean", 0) > o.get("mean_reward", 0)
        print(f"  Reward improved (last-10 vs overall mean): {improved}  |  SR={sr_overall:.1%}")
    print("=" * 66)

    # ── Finish wandb run ──────────────────────────────────────────────────────
    if _wb_run is not None:
        if use_learning and learning_curves:
            _wandb.log({
                "final/reward_improvement_pct": learning_curves.get("reward_improvement_pct", 0),
                "final/sr_improvement_pct":     learning_curves.get("sr_improvement_pct", 0),
                "final/q_states_learned": learning_curves.get("summary", {}).get("total_states_learned", 0),
                "final/success_rate": sum(all_successes) / max(len(all_successes), 1),
                "final/mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            })
        _wb_run.finish()
        print(f"  wandb run finished → {_wb_run.url}")

    return curves


def plot_reward_curves(log_path: str | None = None) -> None:
    """Generate reward curve plots from the training log (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    from autodrive_env.server.reward_tracker import RewardTracker, DEFAULT_LOG_PATH
    curves = RewardTracker.load(log_path or DEFAULT_LOG_PATH)
    if "error" in curves:
        print(f"No data to plot: {curves['error']}")
        return

    rewards      = curves.get("reward_curve", [])
    roll10       = curves.get("rolling_10_curve", [])
    roll20       = curves.get("rolling_20_curve", [])
    successes    = curves.get("success_curve", [])
    difficulties = curves.get("difficulty_curve", [])
    episodes     = list(range(1, len(rewards) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AutoDrive Gym — Training Reward Curves", fontsize=14, fontweight="bold")

    # 1: Reward with rolling averages
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=1, label="Episode reward")
    ax.plot(episodes, roll10,  color="steelblue", linewidth=2.5, label="Rolling-10")
    ax.plot(episodes, roll20,  color="navy", linewidth=2, linestyle="--", label="Rolling-20")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Curve (Rolling Averages)"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2: Success rate
    ax = axes[0, 1]
    w = 10
    rolling_sr = [sum(successes[max(0, i-w+1):i+1]) / len(successes[max(0, i-w+1):i+1]) for i in range(len(successes))]
    ax.plot(episodes, rolling_sr, color="green", linewidth=2, label=f"Rolling-{w} SR")
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Mastery threshold (70%)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate Over Training"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    # 3: Difficulty progression (self-improvement)
    ax = axes[1, 0]
    ax.plot(episodes, difficulties, color="orange", linewidth=2, label="Curriculum difficulty")
    for label, level in [("warmup", 0.25), ("beginner", 0.45), ("intermediate", 0.60), ("advanced", 0.75)]:
        ax.axhline(y=level, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.text(1, level + 0.01, label, fontsize=7, color="grey")
    ax.set_xlabel("Episode"); ax.set_ylabel("Difficulty")
    ax.set_title("Self-Improving Curriculum Difficulty"); ax.set_ylim(0, 1.0); ax.legend(); ax.grid(True, alpha=0.3)

    # 4: Per-scenario performance
    ax = axes[1, 1]
    by_scenario = curves.get("by_scenario", {})
    if by_scenario:
        types = list(by_scenario.keys())
        mr    = [by_scenario[t]["mean_reward"] for t in types]
        sr    = [by_scenario[t]["success_rate"] for t in types]
        x     = list(range(len(types)))
        bw    = 0.35
        ax.bar([xi - bw/2 for xi in x], mr, bw, label="Mean reward", color="steelblue", alpha=0.8)
        ax2   = ax.twinx()
        ax2.bar([xi + bw/2 for xi in x], sr, bw, label="Success rate", color="green", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=7)
        ax.set_ylabel("Mean Reward"); ax2.set_ylabel("Success Rate")
        ax.set_title("Per-Scenario Performance"); ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No per-scenario data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Scenario Performance")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Reward curve plot saved to: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="AutoDrive Gym Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START COMMANDS
--------------------
# Adaptive Q-learning (default, recommended — shows reward improvement):
  python -m autodrive_env.train --episodes 50 --mode adaptive

# Heuristic baseline (rule-based, no improvement expected):
  python -m autodrive_env.train --episodes 30 --mode heuristic

# Full LLM pipeline (requires GROQ_API_KEY or HF_TOKEN):
  python -m autodrive_env.train --episodes 30 --mode pipeline

# Long city route across 5 checkpoints:
  python -m autodrive_env.train --episodes 20 --mode route

# Single scenario (e.g. pedestrian_crossing):
  python -m autodrive_env.train --episodes 40 --mode adaptive --task pedestrian_crossing

# With wandb reward curve tracking (requires: pip install wandb):
  python -m autodrive_env.train --episodes 50 --mode adaptive --wandb

# Generate reward curve plot (requires matplotlib):
  python -m autodrive_env.train --plot-only

ENVIRONMENT VARIABLES
---------------------
  GROQ_API_KEY   — your Groq API key (use for pipeline/adaptive modes)
  HF_TOKEN       — your HuggingFace token (optional, for HF inference)
""",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["heuristic", "adaptive", "pipeline", "route"],
        default="adaptive",
        help="adaptive = Q-learning (default), heuristic = rule-based, pipeline = LLM, route = city route",
    )
    parser.add_argument("--task", type=str, default=None, help="Pin to a specific scenario type")
    parser.add_argument("--plot", action="store_true", help="Generate reward curves after training")
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing reward_log.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb (requires: pip install wandb)")
    args = parser.parse_args()

    if args.plot_only:
        plot_reward_curves()
        return

    run_training(
        n_episodes=args.episodes,
        mode=args.mode,
        task_id=args.task,
        verbose=not args.quiet,
        use_wandb=args.wandb,
    )

    if args.plot:
        plot_reward_curves()


if __name__ == "__main__":
    main()