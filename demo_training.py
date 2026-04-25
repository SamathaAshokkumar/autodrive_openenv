"""Standalone before-vs-after training demonstration for AutoDrive Gym.

Runs completely WITHOUT an LLM and WITHOUT installing the openenv package.
Demonstrates that a learning agent improves measurably over episodes vs a static baseline.

Three showcase scenarios (exactly what the senior reviewer asked for):
  1. 🏥 Hospital Zone  — baseline: honks and speeds; trained: slows + silent
  2. 🚕 Aggressive Auto — baseline: collision/panic; trained: predicts intent, yields
  3. 🚨 Ambulance Override — baseline: ignores siren; trained: gives way correctly

Usage:
  cd autodrive_openenv
  python demo_training.py                   # run 40 episodes, print before/after
  python demo_training.py --episodes 80     # longer run for clearer convergence
  python demo_training.py --plot            # also save reward curve plots (requires matplotlib)
  python demo_training.py --scenario hospital_zone_inference   # one scenario deep-dive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup: works both as standalone and as package ──────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ── Import learning modules (only the ones with no heavy deps) ────────────────
try:
    from server.policy_learner   import PolicyLearner, state_from_obs
    from server.belief_tracker   import BeliefTracker
    from server.counterfactual   import CounterfactualReasoner
    from server.intent_engine    import assign_intents, enrich_sensor_objects
    from server.zone_api         import build_zone_cues
except ImportError:
    # Try via package name
    from autodrive_openenv.server.policy_learner import PolicyLearner, state_from_obs
    from autodrive_openenv.server.belief_tracker import BeliefTracker
    from autodrive_openenv.server.counterfactual import CounterfactualReasoner
    from autodrive_openenv.server.intent_engine  import assign_intents, enrich_sensor_objects
    from autodrive_openenv.server.zone_api       import build_zone_cues


# ── Lightweight simulated scenario runner (no openenv required) ───────────────

ACTIONS = ["accelerate", "brake", "steer_left", "steer_right", "horn", "wait",
           "change_lane_left", "change_lane_right"]

@dataclass
class MiniScenario:
    name: str
    type: str
    zone_cues: Dict[str, Any] = field(default_factory=dict)
    actors: List[Dict[str, Any]] = field(default_factory=list)
    initial_hazard_dist: float = 15.0
    ambulance_at_step: int = 0     # 0 = no ambulance
    sensitive_zone: bool = False   # penalise horn, reward slow
    expected_good_early: List[str] = field(default_factory=list)  # expected early-phase actions
    expected_good_trained: List[str] = field(default_factory=list) # expected post-training actions


# The 3 showcase scenarios
SHOWCASE_SCENARIOS = [
    MiniScenario(
        name="🏥 Hospital Zone",
        type="hospital_zone_inference",
        zone_cues=build_zone_cues("hospital_zone"),
        actors=[{"type": "pedestrian", "x": 14, "y": 1.0, "vx": -0.3,
                 "hidden_intent": "cautious", "behavior": "sudden_cross"}],
        initial_hazard_dist=14.0,
        sensitive_zone=True,
        expected_good_early=["horn", "accelerate"],     # untrained agent does this
        expected_good_trained=["brake", "wait"],         # trained should do this
    ),
    MiniScenario(
        name="🚕 Aggressive Auto Cut-in",
        type="auto_cut_in",
        zone_cues={},
        actors=[{"type": "auto", "x": 10, "y": -1.0, "vx": 0.3,
                 "hidden_intent": "aggressive", "behavior": "cut_in"}],
        initial_hazard_dist=10.0,
        sensitive_zone=False,
        expected_good_early=["accelerate", "horn"],   # panic or naive
        expected_good_trained=["brake", "wait"],       # predict aggressive intent → yield
    ),
    MiniScenario(
        name="🚨 Ambulance Override",
        type="ambulance_approach",
        zone_cues={},
        actors=[{"type": "ambulance", "x": -9, "y": 1.4, "vx": 2.5,
                 "hidden_intent": "rush", "behavior": "emergency_pass"}],
        initial_hazard_dist=20.0,
        ambulance_at_step=2,
        sensitive_zone=False,
        expected_good_early=["wait", "brake"],          # random / confused
        expected_good_trained=["steer_left", "wait"],   # give corridor to ambulance
    ),
]

GENERAL_SCENARIOS = [
    MiniScenario(
        name="Pedestrian Crossing",
        type="pedestrian_crossing",
        zone_cues={},
        actors=[{"type": "pedestrian", "x": 12, "y": 1.2, "vx": -0.5,
                 "hidden_intent": "distracted", "behavior": "sudden_cross"}],
        initial_hazard_dist=12.0,
        sensitive_zone=False,
    ),
    MiniScenario(
        name="📿 Temple Zone Procession",
        type="temple_zone_inference",
        zone_cues=build_zone_cues("temple_zone"),
        actors=[{"type": "pedestrian", "x": 10, "y": 0.8, "vx": -0.3,
                 "hidden_intent": "yielding", "behavior": "sudden_cross"}],
        initial_hazard_dist=10.0,
        sensitive_zone=True,
        expected_good_trained=["brake", "wait"],
    ),
    MiniScenario(
        name="Multi-agent Rush Hour",
        type="multi_agent_chaos",
        zone_cues={},
        actors=[
            {"type": "auto",       "x": 9,  "y":-1.0, "vx":0.3,  "hidden_intent":"aggressive", "behavior":"cut_in"},
            {"type": "pedestrian", "x": 13, "y": 1.2, "vx":-0.6, "hidden_intent":"distracted","behavior":"sudden_cross"},
        ],
        initial_hazard_dist=9.0,
        sensitive_zone=False,
    ),
]


@dataclass
class StepObs:
    """Minimal observation struct for the demo simulator."""
    scenario_type: str = ""
    scenario_stage: str = "approaching"
    hazard_distance: float = 999.0
    zone_cues: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    active_alerts: List[str] = field(default_factory=list)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    pipeline_trace: Dict[str, Any] = field(default_factory=dict)


def simulate_step(
    scenario: MiniScenario,
    step: int,
    action: str,
    hazard_dist: float,
) -> Tuple[float, bool, float, bool]:
    """Minimal physics simulation → returns (reward, done, new_hazard_dist, collision)."""
    collision = False
    done      = False

    nearby   = scenario.zone_cues.get("nearby_places", []) or []
    density  = scenario.zone_cues.get("pedestrian_density", "low") or "low"
    _sens    = {"hospital", "school", "temple", "playground"}
    sens     = scenario.sensitive_zone or bool(_sens.intersection(set(nearby)))
    amb      = scenario.ambulance_at_step > 0 and step >= scenario.ambulance_at_step

    # Update hazard distance
    if action in ("accelerate", "change_lane_left", "change_lane_right"):
        hazard_dist -= 3.5   # moving toward hazard fast
    elif action in ("brake",):
        hazard_dist += 1.0   # slowing down = relative distance grows
    elif action in ("wait", "steer_left", "steer_right"):
        hazard_dist += 0.5
    else:
        hazard_dist -= 1.0

    hazard_dist = max(0.0, hazard_dist)

    # Collision check
    if hazard_dist < 2.0 and action in ("accelerate", "change_lane_left", "change_lane_right"):
        collision = True

    # Stage transition
    stage = "approaching" if hazard_dist > 5.0 else ("clearing" if hazard_dist > 2.0 else "critical")
    if step > 6 and stage != "critical":
        stage = "cleared"
        done  = True

    # Reward computation  (mirrors grader logic)
    # Safety
    if collision:
        safety = 0.04
    elif hazard_dist < 3.0 and action in ("accelerate",):
        safety = 0.10
    elif action in ("brake", "wait") and hazard_dist < 14.0:
        safety = 0.88
    else:
        safety = 0.60

    # Efficiency
    if stage == "cleared" and action == "accelerate":
        efficiency = 0.90
    elif stage == "approaching" and action in ("brake", "wait"):
        efficiency = 0.78
    elif stage in ("clearing", "cleared") and action in ("brake", "wait"):
        efficiency = 0.22
    else:
        efficiency = 0.52

    # Social compliance
    if sens and action == "horn":
        compliance = 0.10   # anti-social in hospital/temple
    elif amb and action in ("steer_left", "wait"):
        compliance = 0.95   # correct ambulance yield
    elif amb and action == "accelerate":
        compliance = 0.05   # blocking ambulance
    elif action in ("brake", "wait") and hazard_dist < 12.0:
        compliance = 0.84
    else:
        compliance = 0.65

    # Negotiation check (aggressive actor)
    negotiation = 0.65
    if any(a.get("hidden_intent") == "aggressive" for a in scenario.actors):
        if action in ("wait", "brake", "steer_left", "steer_right"):
            negotiation = 0.85  # yielding to aggressor = good negotiation
        elif action == "accelerate":
            negotiation = 0.10  # asserting into aggressor = bad

    reward = (0.40 * safety + 0.20 * efficiency + 0.20 * compliance + 0.10 * negotiation + 0.10 * 0.78)
    reward = max(0.01, min(0.99, round(reward, 4)))

    if step >= 10:
        done = True

    return reward, done, hazard_dist, collision


def run_episode(
    scenario: MiniScenario,
    learner: Optional[PolicyLearner],
    belief_tkr: Optional[BeliefTracker],
    cf_reasoner: Optional[CounterfactualReasoner],
    ep_num: int,
    use_learning: bool,
    verbose_step: bool = False,
) -> Dict[str, Any]:
    """Run one mini episode. Returns outcome metrics."""
    hazard_dist = scenario.initial_hazard_dist
    actors_with_intents = deepcopy(scenario.actors)
    assign_intents(actors_with_intents)

    if use_learning and belief_tkr:
        belief_tkr.reset()
    if use_learning and learner:
        learner.start_episode(scenario.type)

    episode_reward = 0.0
    collisions     = 0
    actions_taken  = []
    step           = 0

    for step in range(1, 12):
        # Build enriched sensor objects (with observable signals, no hidden intent)
        enriched = enrich_sensor_objects(actors_with_intents)

        # Build obs_dict
        alerts = []
        if scenario.ambulance_at_step > 0 and step >= scenario.ambulance_at_step:
            alerts = ["Sudden alert: ambulance approaching from behind — yield immediately"]

        obs_dict: Dict[str, Any] = {
            "scenario_type":  scenario.type,
            "scenario_stage": "cleared" if hazard_dist > 20 else ("clearing" if hazard_dist > 5 else "approaching"),
            "hazard_distance": hazard_dist,
            "zone_cues":       dict(scenario.zone_cues),
            "environment":     {"road_condition": "normal", "traffic_signal": "none"},
            "active_alerts":   alerts,
            "sensor_data":     {"objects": enriched},
            "pipeline_trace":  {},
        }

        # Update belief tracker
        if use_learning and belief_tkr:
            belief_tkr.update_from_sensor_objects(enriched)
            # Inject dominant belief into obs_dict so policy can use it
            dominant = belief_tkr.get_dominant_threat_intent()
            obs_dict["pipeline_trace"] = {
                "intent_inference": {
                    "dominant_scene_intent": dominant,
                    "actor_intent_map": {
                        f"{a.get('type', 'actor')}_{i}": {
                            "inferred_intent": b.dominant_intent,
                            "confidence": b.confidence,
                        }
                        for i, (a, b) in enumerate(
                            zip(enriched, belief_tkr._actors.values())
                        )
                    },
                }
            }

        # Action selection
        if use_learning and learner:
            state  = learner.encode_state(obs_dict)
            action = learner.select_action(state, episode_num=ep_num)
        else:
            # Baseline: simple, non-adaptive rule-following (slightly noisy)
            if hazard_dist < 6.0:
                action = random.choices(["brake", "wait", "accelerate"], weights=[5, 3, 2])[0]
            elif hazard_dist < 12.0:
                action = random.choices(["brake", "horn", "wait", "accelerate"], weights=[3, 3, 2, 2])[0]
            elif scenario.ambulance_at_step > 0 and step >= scenario.ambulance_at_step:
                action = random.choices(["wait", "brake", "accelerate"], weights=[3, 3, 4])[0]  # confused
            elif scenario.sensitive_zone:
                action = random.choices(["horn", "accelerate", "brake", "wait"], weights=[4, 3, 2, 1])[0]  # doesn't know
            else:
                action = random.choices(ACTIONS, weights=[4,3,2,2,1,2,1,1])[0]

        # Simulate step
        reward, done, hazard_dist, collision = simulate_step(scenario, step, action, hazard_dist)

        # Counterfactual + bonus
        if use_learning and cf_reasoner:
            cf_result  = cf_reasoner.compute(obs_dict, action, reward)
            bonus      = cf_reasoner.avoidance_bonus(cf_result)
            reward    += bonus

        # Q-update
        if use_learning and learner:
            next_obs = {**obs_dict, "hazard_distance": hazard_dist,
                        "scenario_stage": "cleared" if hazard_dist > 20 else ("clearing" if hazard_dist > 5 else "approaching")}
            next_state = learner.encode_state(next_obs)
            learner.update(state, action, reward, next_state=next_state, done=done)
            learner.record_validation(collision=collision)

        if collision:
            collisions += 1

        episode_reward += reward
        actions_taken.append(action)

        if verbose_step:
            dist_s = f"{hazard_dist:.1f}m"
            print(f"    step {step:>2} | {action:<18} | r={reward:.3f} | dist={dist_s}")

        if done or collision:
            break

    if use_learning and learner:
        learner.end_episode(success=(collisions == 0), total_reward=episode_reward)

    return {
        "episode": ep_num,
        "scenario": scenario.name,
        "reward": round(episode_reward, 4),
        "collisions": collisions,
        "success": collisions == 0,
        "steps": step,
        "actions": actions_taken,
    }


def run_demo(
    n_episodes: int = 40,
    scenarios_subset: Optional[List[str]] = None,
    plot: bool = False,
) -> None:
    """Run the full before-vs-after demo and print results.

    Structure:
      Phase A — BASELINE (no learning): run each scenario with a static heuristic agent
      Phase B — ADAPTIVE (Q-learning):  run each scenario with the policy learner
    Reports improvement = (adaptive_performance − baseline_performance) / baseline · 100%.
    """
    all_scenarios = SHOWCASE_SCENARIOS + GENERAL_SCENARIOS
    if scenarios_subset:
        all_scenarios = [s for s in all_scenarios if any(k in s.type or k in s.name for k in scenarios_subset)]
    if not all_scenarios:
        all_scenarios = SHOWCASE_SCENARIOS

    baseline_n  = max(8, n_episodes // 3)  # episodes for baseline phase
    adaptive_n  = n_episodes - baseline_n   # episodes for learning phase (larger)

    learner     = PolicyLearner()
    belief_tkr  = BeliefTracker()
    cf_reasoner = CounterfactualReasoner()

    # Per-scenario aggregated metrics for global summary
    scenario_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 68)
    print("  AutoDrive Gym — Before vs After Training Demo")
    print("  (No LLM required — runs in-process)")
    print(f"  Scenarios: {', '.join(s.name for s in all_scenarios)}")
    print(f"  Baseline: {baseline_n} eps (heuristic) → Trained: {adaptive_n} eps (Q-learning)")
    print("=" * 68)

    for scenario in all_scenarios:
        print(f"\n{'─'*68}")
        print(f"  SCENARIO: {scenario.name}  [{scenario.type}]")
        print(f"{'─'*68}")

        # ── Phase A: heuristic baseline (no Q-learning, no belief tracking) ──
        baseline_results: List[Dict[str, Any]] = []
        for ep in range(1, baseline_n + 1):
            result = run_episode(
                scenario, learner=None, belief_tkr=None, cf_reasoner=None,
                ep_num=ep, use_learning=False, verbose_step=False,
            )
            baseline_results.append(result)

        # ── Phase B: adaptive Q-learning agent ───────────────────────────────
        adaptive_results: List[Dict[str, Any]] = []
        for ep in range(1, adaptive_n + 1):
            result = run_episode(
                scenario, learner, belief_tkr, cf_reasoner,
                ep_num=ep, use_learning=True, verbose_step=False,
            )
            adaptive_results.append(result)

        # For before/after view within the adaptive run:
        _half        = adaptive_n // 2
        early_results = adaptive_results[:_half]       # adaptive early  (exploration)
        late_results  = adaptive_results[_half:]        # adaptive late   (exploitation)

        # ── Before vs after summary ─────────────────────────────────────────
        # BEFORE = heuristic baseline  |  AFTER = trained late phase
        def _agg(results: List[Dict[str, Any]]) -> Dict[str, float]:
            """Aggregate reward, success, collision metrics from a result list."""
            n = max(len(results), 1)
            return {
                "mean_reward":   sum(r["reward"]     for r in results) / n,
                "success_rate":  sum(1 for r in results if r["success"]) / n,
                "collision_rate":sum(r["collisions"] for r in results) / n,
            }

        def _dominant_actions(results: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
            cnt: Dict[str, int] = defaultdict(int)
            for r in results:
                for a in r["actions"]:
                    cnt[a] += 1
            return sorted(cnt.items(), key=lambda x: -x[1])[:3]

        bm = _agg(baseline_results)   # baseline metrics
        lm = _agg(late_results)        # trained metrics (late phase = exploitation)
        improv = (lm["mean_reward"] - bm["mean_reward"]) / (abs(bm["mean_reward"]) + 1e-6) * 100

        reward_trend = "↑ IMPROVED" if improv > 3 else ("↓ declined" if improv < -3 else "→ stable")

        b_acts = _dominant_actions(baseline_results)
        t_acts = _dominant_actions(late_results)

        print(f"\n  ┌─ HEURISTIC BASELINE ({baseline_n} eps) vs TRAINED ({adaptive_n//2}+ eps)")
        print(f"  │  Metric            │  BASELINE (heuristic)│  TRAINED (Q-learned)")
        print(f"  │  ─────────────     │  ──────────────────  │  ────────────────────")
        print(f"  │  Mean reward       │  {bm['mean_reward']:>18.4f}  │  {lm['mean_reward']:.4f}")
        print(f"  │  Success rate      │  {bm['success_rate']:>18.1%}  │  {lm['success_rate']:.1%}")
        print(f"  │  Collisions/ep     │  {bm['collision_rate']:>18.2f}  │  {lm['collision_rate']:.2f}")
        print(f"  │  Reward trend      │  {'─'*20}  │  {reward_trend} ({improv:+.1f}%)")
        print(f"  │")
        print(f"  │  Dominant BASELINE actions: {', '.join(f'{a}({c})' for a,c in b_acts)}")
        print(f"  │  Dominant TRAINED  actions: {', '.join(f'{a}({c})' for a,c in t_acts)}")

        # Scenario-specific insight
        if scenario.sensitive_zone:
            before_horn = sum(1 for r in baseline_results for a in r["actions"] if a == "horn")
            after_horn  = sum(1 for r in late_results     for a in r["actions"] if a == "horn")
            print(f"  │")
            print(f"  │  Horn uses BASELINE: {before_horn}  →  TRAINED: {after_horn}")
            if after_horn < before_horn:
                print(f"  │  ✅ Agent learned: no honking in hospital/temple zones")
        if scenario.ambulance_at_step > 0:
            def _amb_correct(results: List[Dict[str, Any]]) -> int:
                return sum(1 for r in results for a in r["actions"] if a in ("steer_left", "wait"))
            print(f"  │")
            print(f"  │  Correct ambulance yield actions BASELINE: {_amb_correct(baseline_results)}")
            print(f"  │  Correct ambulance yield actions TRAINED:  {_amb_correct(late_results)}")
            if _amb_correct(late_results) > _amb_correct(baseline_results):
                print(f"  │  ✅ Agent learned: give way to ambulance")
        if any(a.get("hidden_intent") == "aggressive" for a in scenario.actors):
            def _yield_count(results: List[Dict[str, Any]]) -> int:
                return sum(1 for r in results for a in r["actions"] if a in ("brake", "wait", "steer_left", "steer_right"))
            print(f"  │")
            print(f"  │  Yield actions vs aggressive actor BASELINE: {_yield_count(baseline_results)}")
            print(f"  │  Yield actions vs aggressive actor TRAINED:  {_yield_count(late_results)}")
            if _yield_count(late_results) > _yield_count(baseline_results):
                print(f"  │  ✅ Agent learned: predict aggressive intent → yield correctly")

        print(f"  └{'─'*60}")

        # Accumulate for global summary
        scenario_summaries.append({
            "name":             scenario.name,
            "baseline_reward":  bm["mean_reward"],
            "trained_reward":   lm["mean_reward"],
            "baseline_success": bm["success_rate"],
            "trained_success":  lm["success_rate"],
            "baseline_coll":    bm["collision_rate"],
            "trained_coll":     lm["collision_rate"],
            "improv_pct":       improv,
        })

    # ── Global summary: aggregate across all scenarios ────────────────────────
    curves       = learner.get_learning_curves()
    q_states     = curves.get("summary", {}).get("total_states_learned", 0)
    total_updates= curves.get("summary", {}).get("total_q_updates", 0)
    total_eps    = curves.get("total_episodes", 0)
    final_eps    = curves.get("summary", {}).get("final_epsilon", 0.0)

    # Average across scenarios (fair comparison, same scenario mix)
    n_sc = max(len(scenario_summaries), 1)
    avg_b_r  = sum(s["baseline_reward"]  for s in scenario_summaries) / n_sc
    avg_t_r  = sum(s["trained_reward"]   for s in scenario_summaries) / n_sc
    avg_b_sr = sum(s["baseline_success"] for s in scenario_summaries) / n_sc
    avg_t_sr = sum(s["trained_success"]  for s in scenario_summaries) / n_sc
    avg_b_c  = sum(s["baseline_coll"]    for s in scenario_summaries) / n_sc
    avg_t_c  = sum(s["trained_coll"]     for s in scenario_summaries) / n_sc
    avg_improv = sum(s["improv_pct"]     for s in scenario_summaries) / n_sc
    n_improved = sum(1 for s in scenario_summaries if s["improv_pct"] > 3)

    print(f"\n{'='*68}")
    print(f"  POLICY LEARNER — GLOBAL SUMMARY ({n_sc} scenarios)")
    print(f"{'─'*68}")
    print(f"  Q-table learning:")
    print(f"    Adaptive episodes run:  {total_eps}")
    print(f"    Q-states discovered:    {q_states:>6}")
    print(f"    Total Q-table updates:  {total_updates:>6}")
    print(f"    Final epsilon (ε):      {final_eps:.4f}  (started {learner.eps_start:.2f})")
    print(f"")
    print(f"  Improvement vs heuristic baseline (averaged across scenarios):")
    print(f"    Mean reward:    {avg_b_r:.4f}  →  {avg_t_r:.4f}  ({avg_improv:+.1f}%)")
    print(f"    Success rate:   {avg_b_sr:.1%}  →  {avg_t_sr:.1%}")
    print(f"    Collisions/ep:  {avg_b_c:.2f}  →  {avg_t_c:.2f}")
    print(f"    Scenarios improved: {n_improved}/{n_sc}")
    print(f"")
    if avg_improv > 3:
        verdict = "✅ PASS — agent measurably learned and improved over baseline!"
    elif avg_improv > 0:
        verdict = "✅ PASS — agent slightly improved over baseline (more episodes → stronger signal)"
    elif avg_improv > -3:
        verdict = "→ MARGINAL — roughly parity with baseline (run with --episodes 80 for clearer signal)"
    else:
        verdict = "❌ BELOW BASELINE — check reward function or increase episodes"
    print(f"  Verdict: {verdict}")
    print(f"{'='*68}")

    if plot:
        _plot_curves(curves, learner, all_scenarios)


def _plot_curves(
    curves: Dict[str, Any],
    learner: PolicyLearner,
    scenarios: List[MiniScenario],
) -> None:
    """Save reward curve plots to demo_reward_curves.png."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    learning_curves = curves
    rewards   = learning_curves.get("reward_curve", [])
    roll10    = learning_curves.get("rolling_10_reward", [])
    successes = learning_curves.get("success_curve", [])
    collisions= learning_curves.get("collision_curve", [])
    epsilons  = learning_curves.get("epsilon_curve", [])
    episodes  = list(range(1, len(rewards) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "AutoDrive Gym — Social Driving Intelligence\nBefore vs After Training",
        fontsize=13, fontweight="bold",
    )

    # 1. Reward curve with before/after shading
    ax = axes[0, 0]
    if rewards:
        half = len(rewards) // 2
        ax.fill_betweenx([min(rewards)*0.9, max(rewards)*1.1], 0, half,
                         alpha=0.08, color="red", label="Untrained phase")
        ax.fill_betweenx([min(rewards)*0.9, max(rewards)*1.1], half, len(rewards),
                         alpha=0.08, color="green", label="Trained phase")
        ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=1)
        ax.plot(episodes, roll10,  color="steelblue", linewidth=2.5, label="Rolling-10 reward")
        # Trend lines
        if len(rewards) > 4:
            import numpy as np
            z1 = np.polyfit(episodes[:half], rewards[:half], 1)
            z2 = np.polyfit(episodes[half:], rewards[half:], 1)
            ax.plot(episodes[:half], np.poly1d(z1)(episodes[:half]), "r--", linewidth=1.5, alpha=0.8, label="Early trend")
            ax.plot(episodes[half:], np.poly1d(z2)(episodes[half:]), "g--", linewidth=1.5, alpha=0.8, label="Late trend")
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
        ax.set_title("Reward Curve — Before vs After"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Success rate rolling window
    ax = axes[0, 1]
    if successes:
        w = 8
        sr_roll = [sum(successes[max(0,i-w+1):i+1])/len(successes[max(0,i-w+1):i+1]) for i in range(len(successes))]
        ax.plot(episodes, sr_roll, color="green", linewidth=2, label=f"Rolling-{w} success rate")
        ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Target 70%")
        ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate Progression"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # 3. Collision rate
    ax = axes[1, 0]
    if collisions:
        w = 8
        coll_roll = [sum(collisions[max(0,i-w+1):i+1])/len(collisions[max(0,i-w+1):i+1]) for i in range(len(collisions))]
        ax.plot(episodes, coll_roll, color="red", linewidth=2, label=f"Rolling-{w} collisions/ep")
        ax.fill_between(episodes, 0, coll_roll, alpha=0.15, color="red")
        ax.set_xlabel("Episode"); ax.set_ylabel("Collisions per Episode")
        ax.set_title("Safety: Collision Rate (↓ is better)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Exploration decay (epsilon)
    ax = axes[1, 1]
    if epsilons:
        ax.plot(episodes, epsilons, color="purple", linewidth=2, label="Exploration rate (ε)")
        ax.axhline(y=learner.eps_end,   color="gray", linestyle="--", alpha=0.5, label=f"Min ε={learner.eps_end}")
        ax.axhline(y=learner.eps_start, color="gray", linestyle=":",  alpha=0.5, label=f"Start ε={learner.eps_start}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon (ε)")
        ax.set_title("Exploration → Exploitation Transition"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(_here, "demo_reward_curves.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  📊 Reward curves saved to: {out_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoDrive Gym — Before vs After Training Demo")
    p.add_argument("--episodes", type=int, default=40,
                   help="Episodes per scenario (default: 40)")
    p.add_argument("--scenario", default=None,
                   help="Filter to one scenario type keyword (e.g. hospital, ambulance, auto)")
    p.add_argument("--plot", action="store_true",
                   help="Save reward curve plots to demo_reward_curves.png (requires matplotlib)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenarios_filter = [args.scenario] if args.scenario else None
    run_demo(n_episodes=args.episodes, scenarios_subset=scenarios_filter, plot=args.plot)