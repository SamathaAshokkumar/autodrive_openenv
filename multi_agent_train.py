"""Dual-agent cooperative training script for AutoDrive Gym.

Two agents share the same environment and act on each other's outputs:

  PilotAgent     — full heuristic driver (balanced speed + safety).
                   Makes the primary decision independently from env state.

  CoPilotAgent   — conservative safety co-pilot.  Receives the env state
                   AND the Pilot's proposal before making its own judgment.
                   Can confirm, soften, or override the pilot's action.

  Coordinator    — arbitrates the two proposals into a final action.
                   Logs every agreement / conflict step for analysis.

The key "multi-agent" interaction:
  1. Environment → PilotAgent  → pilot_proposal
  2. Environment + pilot_proposal → CoPilotAgent → copilot_proposal
  3. Coordinator(pilot_proposal, copilot_proposal) → FINAL action
  4. FINAL action → env.step()

Usage
-----
  python -m autodrive_env.multi_agent_train --episodes 30 --plot
  python -m autodrive_env.multi_agent_train --episodes 30 --mode heuristic
  python -m autodrive_env.multi_agent_train --plot-only
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ── Re-use helpers from train.py ──────────────────────────────────────────────
from autodrive_env.train import (
    _InProcessEnvSession,
    _StepResult,
    _print_ep_result,
    _failure_insight,
    _print_weakness_report,
    plot_reward_curves,
    SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# PilotAgent — balanced heuristic (re-uses agent_baseline.choose_action)
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_dict(obs) -> Dict[str, Any]:
    """Convert AutoDriveObservation to the dict expected by choose_action()."""
    return {
        "sensor_data":    getattr(obs, "sensor_data",    {}) or {},
        "ego_state":      getattr(obs, "ego_state",      {}) or {},
        "environment":    getattr(obs, "environment",    {}) or {},
        "active_alerts":  getattr(obs, "active_alerts",  []) or [],
        "hint":           getattr(obs, "hint",           "") or "",
        "scenario_stage": getattr(obs, "scenario_stage", "approaching") or "approaching",
        "hazard_distance":getattr(obs, "hazard_distance", 999.0) or 999.0,
        "hazard_type":    getattr(obs, "hazard_type",    "") or "",
        "road_geometry":  getattr(obs, "road_geometry",  {}) or {},
    }


def pilot_action(obs, history: List[Dict]) -> Dict[str, Any]:
    """PilotAgent: full heuristic from agent_baseline — balanced speed + safety."""
    from autodrive_env.agent_baseline import choose_action
    act = choose_action(_obs_to_dict(obs), history)
    return {**act, "agent": "pilot"}


# ─────────────────────────────────────────────────────────────────────────────
# CoPilotAgent — safety-first conservative agent
# Receives env state + pilot's proposal before making its own judgment.
# ─────────────────────────────────────────────────────────────────────────────

def copilot_action(
    obs,
    pilot_proposal: Dict[str, Any],
    history: List[Dict],
) -> Dict[str, Any]:
    """CoPilotAgent: conservative safety layer.

    Sees the same environment the pilot sees, PLUS the pilot's proposed action.
    Returns its own assessment — which may agree, soften, or override pilot.
    """
    obs_dict  = _obs_to_dict(obs)
    sensor    = obs_dict.get("sensor_data", {}) or {}
    objects   = sensor.get("objects", []) or []
    min_dist  = min((float(o.get("distance", 999)) for o in objects), default=999.0)
    stage     = obs_dict.get("scenario_stage", "approaching") or "approaching"
    hazard_d  = float(obs_dict.get("hazard_distance", 999.0) or 999.0)
    alerts    = obs_dict.get("active_alerts", []) or []
    ego       = obs_dict.get("ego_state", {}) or {}
    speed     = float(ego.get("speed", 0.0))

    pilot_act = pilot_proposal.get("action", "wait")
    pilot_val = float(pilot_proposal.get("value", 0.0))

    # ── Rule C0: Immediate danger regardless of stage ─────────────────────────
    if min_dist < 3.0:
        return {"action": "brake", "value": 1.0, "agent": "copilot",
                "note": "emergency_brake_override"}

    # ── Rule C1: Verify clearing before confirming acceleration ───────────────
    # Co-pilot only agrees to accelerate in clearing once min_dist > 6 m
    if stage in ("clearing", "cleared") and min_dist > 6.0:
        if pilot_act == "accelerate":
            # Confirm, but soften slightly
            return {"action": "accelerate", "value": min(pilot_val, 0.4), "agent": "copilot",
                    "note": "clearing_confirmed"}
        # Pilot is braking/waiting even though stage cleared — nudge to move
        if pilot_act in ("brake", "wait") and min_dist > 7.0:
            return {"action": "accelerate", "value": 0.3, "agent": "copilot",
                    "note": "copilot_clearing_nudge"}

    # ── Rule C2: Distrust pilot's accelerate during approach with objects < 9 m
    if pilot_act == "accelerate" and min_dist < 9.0:
        return {"action": "brake", "value": 0.6, "agent": "copilot",
                "note": "copilot_brakes_unsafe_accelerate"}

    # ── Rule C3: Always brake before steer if object < 5 m ───────────────────
    if pilot_act in ("steer_left", "steer_right") and min_dist < 5.0:
        return {"action": "brake", "value": 0.9, "agent": "copilot",
                "note": "brake_before_steer_at_close_range"}

    # ── Rule C4: Alert → independent brake regardless of pilot ────────────────
    if alerts and pilot_act not in ("brake", "wait", "steer_left", "steer_right"):
        return {"action": "brake", "value": 0.7, "agent": "copilot",
                "note": "alert_detected_brake"}

    # ── Rule C5: Conservative close-range cap ────────────────────────────────
    if min_dist < 6.0 and pilot_act == "accelerate":
        return {"action": "wait", "value": 0.0, "agent": "copilot",
                "note": "copilot_waits_close_range"}

    # ── Default: co-pilot agrees with the pilot ────────────────────────────────
    return {"action": pilot_act, "value": pilot_val, "agent": "copilot",
            "note": "agrees_with_pilot"}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator — arbitrates and returns the final action + decision reason
# ─────────────────────────────────────────────────────────────────────────────

_SAFETY_RANK: Dict[str, int] = {
    "brake": 0, "wait": 1, "horn": 2,
    "steer_left": 3, "steer_right": 3,
    "change_lane_left": 4, "change_lane_right": 4,
    "accelerate": 5,
}


def coordinate(
    pilot: Dict[str, Any],
    copilot: Dict[str, Any],
    obs,
) -> Tuple[Dict[str, Any], str]:
    """Choose the final action and return (action_dict, reason_string)."""
    pa = pilot.get("action", "wait")
    ca = copilot.get("action", "wait")
    pv = max(0.0, min(1.0, float(pilot.get("value", 0.0))))
    cv = max(0.0, min(1.0, float(copilot.get("value", 0.0))))
    note = copilot.get("note", "")

    # Both agents agree
    if pa == ca:
        # Take the higher value (more decisive when unanimous)
        return {"action": pa, "value": max(pv, cv)}, f"🤝 AGREED"

    # Co-pilot issued a safety override (note is set by a hard rule)
    if note in ("emergency_brake_override", "brake_before_steer_at_close_range",
                "copilot_brakes_unsafe_accelerate", "alert_detected_brake"):
        return {"action": ca, "value": cv}, f"🛡 COPILOT_SAFETY({note})"

    # Pilot wants progress, co-pilot is cautious
    if pa == "accelerate" and ca in ("brake", "wait"):
        return {"action": ca, "value": cv}, f"🛡 COPILOT_WIN_CAUTION"

    # Co-pilot nudges forward (clearing) but pilot is still holding back
    if ca == "accelerate" and pa in ("brake", "wait") and note == "copilot_clearing_nudge":
        # Compromise: execute pilot's safe action but at reduced intensity
        return {"action": pa, "value": max(0.1, pv * 0.5)}, f"🔶 COMPROMISE(clearing_dispute)"

    # Co-pilot confirmed clearing → trust it
    if note == "clearing_confirmed" and ca == "accelerate":
        return {"action": ca, "value": cv}, f"✅ COPILOT_CLEARED"

    # Different lateral/lane actions — prefer the safer (lower rank) one
    pr = _SAFETY_RANK.get(pa, 5)
    cr = _SAFETY_RANK.get(ca, 5)
    if cr < pr:
        return {"action": ca, "value": cv}, f"🛡 COPILOT_SAFER_ACTION"

    # Default — go with the pilot's richer heuristic
    return {"action": pa, "value": pv}, f"✈  PILOT_DEFAULT(copilot={ca})"


# ─────────────────────────────────────────────────────────────────────────────
# Episode log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_dual_ep_header(
    ep: int, total: int, scenario_type: str,
    tier: str, difficulty: float,
) -> None:
    print("\n" + "=" * 76)
    print(f"  EPISODE {ep}/{total}  [dual-agent]")
    print(f"  scenario={scenario_type:<26}  tier={tier}  diff={difficulty:.2f}")
    print("  Pilot=heuristic(balanced)  CoPilot=conservative(safety-first)")
    print("-" * 76)


def _print_dual_step(
    step: int,
    pilot_act: str, pilot_val: float,
    copilot_act: str, copilot_val: float,
    final_act: str, final_val: float,
    reason: str,
    reward: float,
    hazard_type: str, hazard_dist: float,
    stage: str, flags: str,
) -> None:
    status    = "✓" if reward >= 0.5 else "✗"
    dist_str  = f"{hazard_dist:.1f}m" if hazard_dist < 900 else "clear"
    haz_str   = f"{hazard_type}@{dist_str}" if hazard_type else dist_str
    flag_str  = f" [{flags}]" if flags else ""
    agreed    = pilot_act == copilot_act
    sync_icon = "=" if agreed else "≠"
    print(
        f"  step {step:>2} {status} "
        f"| P={pilot_act:<18}({pilot_val:.1f})"
        f" {sync_icon} C={copilot_act:<18}({copilot_val:.1f})"
        f" | {reason:<34}"
        f" | FINAL={final_act:<18}({final_val:.1f})"
        f" r={reward:+.4f} {haz_str}{flag_str}",
        flush=True,
    )


def _print_dual_ep_result(
    success: bool, steps: int, episode_reward: float,
    step_rewards: List[float],
    agreements: int, conflicts: int,
    rolling10: float, sr20: float, tier: str, diff: float, end_reason: str,
) -> None:
    icon = "✅ SUCCESS" if success else "❌ FAIL   "
    avg   = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    best  = max(step_rewards) if step_rewards else 0.0
    worst = min(step_rewards) if step_rewards else 0.0
    agree_pct = agreements / steps * 100.0 if steps else 0.0
    print(f"\n  [RESULT] {icon}  steps={steps}  total_reward={episode_reward:.4f}  end={end_reason}")
    print(f"           per-step: avg={avg:.4f}  best={best:.4f}  worst={worst:.4f}")
    print(f"           agent-sync: agreements={agreements}  conflicts={conflicts}  ({agree_pct:.0f}% agree)")
    print(f"           roll10={rolling10:.4f}  SR(last20)={sr20:.1%}  tier={tier}  diff={diff:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_dual_agent_training(
    n_episodes: int = 30,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run N episodes with Pilot + CoPilot collaborating on each step."""
    from autodrive_env.server.autodrive_gym_environment import AutoDriveGymEnvironment
    from autodrive_env.models import AutoDriveAction

    env     = AutoDriveGymEnvironment()
    session = _InProcessEnvSession(env)        # OpenEnv StepResult API wrapper

    all_rewards:   List[float] = []
    all_successes: List[int]   = []

    if verbose:
        print("\n" + "=" * 76)
        print("  AutoDrive Gym — Dual-Agent Training Run")
        print(f"  target_episodes={n_episodes}  mode=dual_heuristic")
        print("  PilotAgent: balanced heuristic (choose_action)")
        print("  CoPilotAgent: safety-first conservative | sees pilot's proposal")
        print("  Coordinator: arbitrates conflicts with safety-priority rules")
        print("=" * 76)

    for ep_num in range(1, n_episodes + 1):
        cur        = session.curriculum.get_stats()
        tier       = cur.get("tier", "warmup")
        difficulty = cur.get("difficulty", 0.15)

        # ── Reset environment (OpenEnv API) ───────────────────────────────────
        _reset      = session.reset()
        obs         = _reset.observation

        scenario_type = getattr(obs, "scenario_type", "") or getattr(obs, "hazard_type", "") or "unknown"

        if verbose:
            _print_dual_ep_header(ep_num, n_episodes, scenario_type, tier, difficulty)

        episode_reward = 0.0
        step           = 0
        done           = False
        history:      List[Dict[str, Any]] = []
        step_rewards: List[float]          = []
        agreements    = 0
        conflicts     = 0
        ep_start      = time.time()

        while not done:
            # ── Stage 1: PilotAgent decides (reads env state only) ────────────
            pilot   = pilot_action(obs, history)

            # ── Stage 2: CoPilotAgent responds (reads env + pilot's proposal) ─
            copilot = copilot_action(obs, pilot, history)

            # ── Stage 3: Coordinator arbitrates ──────────────────────────────
            final, reason = coordinate(pilot, copilot, obs)
            final_name  = str(final.get("action", "wait"))
            final_value = float(final.get("value", 0.0))

            if pilot.get("action") == copilot.get("action"):
                agreements += 1
            else:
                conflicts += 1

            # ── Step environment (OpenEnv API: step() → StepResult) ───────────
            _step  = session.step(AutoDriveAction(action=final_name, value=final_value))
            obs    = _step.observation
            reward = float(_step.reward or 0.0)
            done   = bool(_step.done)

            episode_reward += reward
            step           += 1
            step_rewards.append(reward)

            if verbose:
                val    = getattr(obs, "validation", {}) or {}
                hd     = float(getattr(obs, "hazard_distance", 999.0) or 999.0)
                ht     = getattr(obs, "hazard_type", "") or ""
                stage  = getattr(obs, "scenario_stage", "") or "approaching"
                flags  = []
                if val.get("collision"):         flags.append("COLLISION!")
                if val.get("near_miss"):         flags.append("near-miss")
                if val.get("safe_distance"):     flags.append("safe✓")
                if val.get("progress_restored"): flags.append("cleared✓")
                if val.get("stuck"):             flags.append("stuck")

                _print_dual_step(
                    step,
                    pilot.get("action", "wait"),  float(pilot.get("value", 0.0)),
                    copilot.get("action", "wait"), float(copilot.get("value", 0.0)),
                    final_name, final_value,
                    reason, reward,
                    ht, hd, stage, " ".join(flags),
                )

            history.append({
                "step": step, "action": final_name, "value": final_value, "reward": reward,
                "pilot_action":   pilot.get("action"),
                "copilot_action": copilot.get("action"),
                "reason":         reason,
                "hazard_type":    getattr(obs, "hazard_type", "") or "",
                "hazard_dist":    float(getattr(obs, "hazard_distance", 999.0) or 999.0),
                "stage":          getattr(obs, "scenario_stage", "") or "approaching",
            })

        # ── Episode outcome ───────────────────────────────────────────────────
        val_final = getattr(obs, "validation", {}) or {}
        res_final = getattr(obs, "resolution",  {}) or {}
        success = bool(
            res_final.get("verified")
            or val_final.get("progress_restored")
            or val_final.get("reached_goal")
            or (episode_reward > 5.0
                and not val_final.get("collision")
                and not val_final.get("stuck"))
        )
        end_reason = (
            "collision" if val_final.get("collision")         else
            "cleared"   if val_final.get("progress_restored") else
            "timeout"   if step >= (getattr(obs, "max_steps", 20) or 20) else
            "stuck"     if val_final.get("stuck")             else "done"
        )

        cur2       = session.curriculum.get_stats()
        tier       = cur2.get("tier", "warmup")
        difficulty = cur2.get("difficulty", 0.15)
        all_rewards.append(episode_reward)
        all_successes.append(int(success))

        if verbose:
            rolling10 = sum(all_rewards[-10:]) / len(all_rewards[-10:])
            sr20      = sum(all_successes[-20:]) / len(all_successes[-20:])
            _print_dual_ep_result(
                success, step, episode_reward, step_rewards,
                agreements, conflicts,
                rolling10, sr20, tier, difficulty, end_reason,
            )
            si = cur2.get("self_improve_triggered", 0)
            if si > 0 and cur2.get("consecutive_failures", 0) >= 3:
                print(f"\n  [SELF-IMPROVE] Adversarial scenario injected (trigger #{si})")
            elapsed = time.time() - ep_start
            print(f"\n  ⏱  {elapsed:.1f}s\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    curves = session.reward_tracker.get_curves()
    session.reward_tracker.save()

    if verbose:
        total_agreements = sum(1 for h in
            [ep_h for ep_history in [history] for ep_h in ep_history]
            if ep_h.get("pilot_action") == ep_h.get("copilot_action"))
        print("\n" + "=" * 76)
        print("  Dual-Agent Training complete.")
        print("  " + session.reward_tracker.summary_line())
        print(f"  Reward log: {session.reward_tracker.log_path}")
        overall_sr = sum(all_successes) / len(all_successes) if all_successes else 0.0
        print(f"  Overall SR={overall_sr:.1%}  episodes={len(all_rewards)}")
        if curves.get("overall"):
            o = curves["overall"]
            improved = o.get("final_10_mean", 0) > o.get("mean_reward", 0)
            print(f"  Reward improved (last-10 vs overall mean): {improved}")
        print("=" * 76)

    return curves


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoDrive Gym — Dual-Agent Training (Pilot + CoPilot)",
    )
    parser.add_argument("--episodes",  type=int, default=30,
                        help="Number of training episodes (default: 30)")
    parser.add_argument("--plot",      action="store_true",
                        help="Save reward-curve plot after training")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot from existing log, skip training")
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    if args.plot_only:
        plot_reward_curves()
        return

    run_dual_agent_training(n_episodes=args.episodes, verbose=not args.quiet)

    if args.plot:
        plot_reward_curves()


if __name__ == "__main__":
    main()