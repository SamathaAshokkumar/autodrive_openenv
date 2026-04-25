"""
GRPO Training Script — AutoDrive Gym (Indian Road Conditions)
Mirrors the kube-sre-gym winner pattern: TRL + vLLM + LoRA.

The LLM fine-tunes on driving episodes — after training it actually knows
how to navigate Indian road conditions, not just call an external API.

==============================================================================
SETUP (two terminals — same as kube winner)
==============================================================================

  pip install -e ".[grpo]"          # install grpo extras

  # Terminal 1: Start the AutoDrive Gym server
  uvicorn autodrive_env.server.app:app --host 0.0.0.0 --port 8000

  # Terminal 2: Run GRPO training  (needs GPU)
  python train_grpo.py --episodes 50 --model-id Qwen/Qwen3-0.6B

==============================================================================
HuggingFace GPU Space (recommended: T4 16GB or A10G 24GB)
==============================================================================
  # Upload this repo to your HF Space, then in the Space terminal:
  pip install -e ".[grpo]"
  uvicorn autodrive_env.server.app:app --host 0.0.0.0 --port 8000 &
  python train_grpo.py --episodes 50 --vllm-mode colocate --push-to-hub --hub-repo YOUR_HF_NAME/autodrive-agent

==============================================================================
Model size guide
==============================================================================
  GPU VRAM    Recommended model           LoRA rank
  8  GB       Qwen/Qwen3-0.6B             r=8
  16 GB       Qwen/Qwen3-1.7B             r=16   ← HF T4 default
  24 GB       Qwen/Qwen3-4B               r=16
  40-80 GB    Qwen/Qwen2.5-7B-Instruct    r=32
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Critical for reusing GPU memory on single-GPU setups
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

try:
    from autodrive_env import AutoDriveGymEnvironment
    from autodrive_env.models import AutoDriveAction, AutoDriveObservation
except ImportError:
    from . import AutoDriveGymEnvironment
    from .models import AutoDriveAction, AutoDriveObservation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── TRL / vLLM compatibility patch (same as kube winner) ─────────────────────
_orig_vllm_gen = None

def _patch_vllm_generate(trainer: GRPOTrainer) -> None:
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, "vllm_generation"):
        return
    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped


def patch_trl_vllm_compat() -> None:
    _orig = GRPOTrainer.train

    def _patched(self, *a, **kw):
        _patch_vllm_generate(self)
        return _orig(self, *a, **kw)

    GRPOTrainer.train = _patched


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous driving agent in India.
Read the current road situation and respond with ONE JSON action.

VALID ACTIONS:
  brake          — slow/stop for hazards, pedestrians, animals, red signal
  accelerate     — build speed when road is clear or hazard has passed
  wait           — hold position for signals, processions, police hand signal
  steer_left     — dodge right-side hazard, create ambulance corridor
  steer_right    — dodge left-side hazard or create lane space
  horn           — warn pedestrian/animal; use sparingly in temple/school zones
  change_lane_left  — move fully to left lane when right is blocked
  change_lane_right — move fully to right lane when left is blocked

DISTANCE RULES:
  < 4m   : brake(1.0) or steer hard — emergency only
  4-8m   : brake(0.7-0.9) or steer to avoid
  8-14m  : brake(0.4-0.6) or horn or wait
  14-18m : gentle brake only — stay alert
  > 18m  : DO NOT brake. Accelerate or maintain speed.

DISTANCE TREND (read hazard_trend field in observation):
  hazard_trend=receding → hazard is moving AWAY. Ease off brakes immediately.
    If distance > 12m and receding: switch to accelerate.
    Braking when hazard is receding = penalised.
  hazard_trend=approaching_or_static → standard brake rules above apply.

INDIAN ROAD REALITY:
  - Auto cut in and now pulling ahead? Follow it, accelerate.
  - Bike merged and now 15m ahead? Go. Don't idle.
  - Pedestrian crossed and off-road? Accelerate immediately.
  - Only brake if something is ACTIVELY in your path RIGHT NOW.

READING NEW OBSERVATION FIELDS:
  closing_speed < 0         : hazard RECEDING — ease off, prepare to accelerate
  closing_speed > 0         : hazard approaching — apply distance rules above
  trajectory=will_cross     : hazard will cross your lane — brake proactively
  trajectory=moving_away    : hazard leaving path — prepare to accelerate
  trajectory=erratic_unpredictable : high-risk regardless of distance
  intent_confidence < 0.40  : uncertain actor — add 2 m extra buffer
  dominance=high            : must yield (bus/truck/ambulance/police)
  negotiation_required=true : adjust speed to negotiate, don't blast through
  signal_trust_score < 0.50 : signal unreliable — use own judgment
  time_context=rush_hour    : aggressive actors; tight negotiation windows
  time_context=night        : slow default speed; higher occlusion risk
  occlusion=true            : hidden hazard may emerge — slow down
  pedestrian_flow_direction=crossing_road : brake even if no pedestrian visible yet
  behavioral_pattern contains "aggressive×" : treat as high-risk regardless of distance

OUTPUT FORMAT (strict JSON, nothing else):
{
  "action": "<one of the 8 actions above>",
  "value": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explaining why>"
}

RULES:
- Near ambulance/emergency: brake or yield immediately
- Hazard distance < 8m: brake hard (value > 0.7)
- Hazard cleared (stage=cleared): accelerate to restore progress
- No Horn in temple/school/hospital silent zones
- Police hand signal overrides traffic light
- Zone cues tell you context — infer from them, they are not explicit labels
"""


# ── Observation formatter ─────────────────────────────────────────────────────

def format_observation(obs: AutoDriveObservation, history: list[dict]) -> str:
    """Convert structured observation into text the LLM can reason about."""
    parts: list[str] = []

    # ── Scene summary ────────────────────────────────────────────────────────
    if obs.command_output:
        parts.append(f"SITUATION: {obs.command_output}")

    if obs.active_alerts:
        parts.append("⚠️  ACTIVE ALERTS: " + " | ".join(obs.active_alerts))

    # ── Hazard info ──────────────────────────────────────────────────────────
    hd = obs.hazard_distance if obs.hazard_distance is not None else 999.0
    stage = obs.scenario_stage or "approaching"
    htype = obs.hazard_type or "unknown"
    parts.append(
        f"HAZARD: type={htype}  distance={hd:.1f}m  stage={stage}  status={obs.hazard_status or 'active'}"
    )

    # ── Sensor snapshot ──────────────────────────────────────────────────────
    sensor = obs.sensor_data or {}
    objects = sensor.get("objects", []) or []
    if objects:
        obj_lines = []
        for o in objects[:4]:  # top 4 nearest
            obj_lines.append(
                f"  {o.get('type','?')} at {o.get('distance',0):.1f}m "
                f"({'on-road' if o.get('on_road') else 'off-road'}) "
                f"behavior={o.get('behavior','?')}"
            )
        parts.append("NEARBY OBJECTS:\n" + "\n".join(obj_lines))

    # ── Ego state ────────────────────────────────────────────────────────────
    ego = obs.ego_state or {}
    if ego:
        parts.append(
            f"EGO: speed={ego.get('speed', 0):.1f}km/h  lane={ego.get('lane','?')}  "
            f"steering={ego.get('steering', 0):.2f}"
        )

    # ── Environment ──────────────────────────────────────────────────────────
    env = obs.environment or {}
    if env:
        parts.append(
            f"ENVIRONMENT: road={env.get('road_condition','normal')}  "
            f"visibility={env.get('visibility','clear')}  "
            f"signal={env.get('traffic_signal','none')}"
        )

    # ── Zone cues (indirect — agent must infer) ──────────────────────────────
    zc = obs.zone_cues or {}
    if zc:
        nearby = zc.get("nearby_places", [])
        signs = zc.get("visible_signs", [])
        cues = zc.get("ambient_cues", [])
        density = zc.get("pedestrian_density", "")
        tod = zc.get("time_of_day", "")
        zone_lines = []
        if nearby:
            zone_lines.append(f"  nearby={nearby}")
        if signs:
            zone_lines.append(f"  signs={signs}")
        if cues:
            zone_lines.append(f"  cues={cues[:2]}")  # keep concise
        if density:
            zone_lines.append(f"  pedestrian_density={density}")
        if tod:
            zone_lines.append(f"  time_of_day={tod}")
        if zone_lines:
            parts.append("ZONE CONTEXT (infer from this — no explicit zone label given):\n" + "\n".join(zone_lines))

    # ── Hint ─────────────────────────────────────────────────────────────────
    if obs.hint:
        parts.append(f"HINT: {obs.hint}")

    # ── Recent history (last 3 steps) ────────────────────────────────────────
    if history:
        recent = history[-3:]
        lines = ["RECENT ACTIONS:"]
        for h in recent:
            fb = h.get("feedback", "")
            lines.append(
                f"  step {h['step']}: {h['action']} (value={h['value']:.2f})  "
                f"reward={h['reward']:.3f}"
                + (f"  ← {fb}" if fb else "")
            )
        parts.append("\n".join(lines))

    parts.append(f"\nStep {obs.steps_taken}/{obs.max_steps}. What is your driving action?")
    return "\n\n".join(parts)


# ── Action parser ─────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    "brake", "accelerate", "wait", "steer_left", "steer_right",
    "horn", "change_lane_left", "change_lane_right",
}


def parse_action(text: str) -> AutoDriveAction:
    """Extract action+value from LLM response. Falls back gracefully."""
    if not text:
        return AutoDriveAction(action="brake", value=0.5)

    # Try JSON extraction (handles any surrounding text)
    json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = str(data.get("action", "brake")).strip().lower()
            value = float(data.get("value", 0.5))
            if action in VALID_ACTIONS:
                return AutoDriveAction(action=action, value=max(0.0, min(1.0, value)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Keyword fallback — check if any valid action word appears in the text
    text_lower = text.lower()
    for act in VALID_ACTIONS:
        if act in text_lower:
            return AutoDriveAction(action=act, value=0.7)

    return AutoDriveAction(action="brake", value=0.5)


# ── Single episode rollout ────────────────────────────────────────────────────

def rollout_once(
    trainer: GRPOTrainer,
    env: AutoDriveGymEnvironment,
    tokenizer: AutoTokenizer,
    max_turns: int,
) -> dict[str, Any]:
    """
    Run one complete driving episode.

    Pattern mirrors kube winner exactly:
      - prompt_ids + completion_ids grow across steps
      - episode total reward = sum of step rewards
      - GRPO assigns this reward to the full token sequence
    """
    obs: AutoDriveObservation = env.reset()

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    episode_history: list[dict] = []

    MAX_TOTAL_TOKENS = 3072  # prevent OOM during backward

    for turn in range(max_turns):
        if obs.done:
            break
        if len(completion_ids) >= MAX_TOTAL_TOKENS:
            break

        # Format observation into text
        obs_text = format_observation(obs, episode_history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        # Apply chat template
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        # Generate via vLLM (managed by GRPOTrainer)
        rollout = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout["prompt_ids"])
        completion_ids.extend(rollout["completion_ids"])
        logprobs.extend(rollout["logprobs"])

        completion_text = rollout.get("text") or tokenizer.decode(
            rollout["completion_ids"], skip_special_tokens=True
        )

        # Parse the LLM output to a driving action
        action = parse_action(completion_text)

        # Execute in environment
        try:
            result = env.step(action)
            reward = float(result.reward or 0.0)
            step_rewards.append(reward)

            episode_history.append({
                "step": turn + 1,
                "action": action.action,
                "value": action.value,
                "reward": reward,
                "feedback": result.hint or "",
            })

            # Save transcript for offline analysis / SFT later
            obs = result
        except Exception as exc:
            logger.warning("Step error at turn %d: %s", turn, exc)
            step_rewards.append(-0.3)
            break

    total_reward = sum(step_rewards) if step_rewards else -1.0
    success = bool(obs.done and total_reward > 0 and not (obs.validation or {}).get("collision"))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "success": success,
        "steps": len(step_rewards),
    }


# ── Reward functions (TRL convention — one per reward signal) ─────────────────

def reward_total(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("total_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_success(completions: list[str], **kwargs) -> list[float]:
    successes = kwargs.get("success")
    return [1.0 if s else 0.0 for s in successes] if successes else [0.0] * len(completions)


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for AutoDrive Gym")

    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B",
                   help="Model to fine-tune. 0.6B fits on T4 (16GB), 1.7B on A10G (24GB)")
    p.add_argument("--episodes", type=int, default=50,
                   help="Total training episodes (each = one driving scenario)")
    p.add_argument("--max-turns", type=int, default=20,
                   help="Max steps per episode (matches env MAX_STEPS)")
    p.add_argument("--num-generations", type=int, default=8,
                   help="G for GRPO advantage estimation (8 is minimum for stable signal)")
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens per LLM response (keep short — JSON action is ~60 tokens)")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank. Use 8 for 8GB GPU, 16 for 16GB, 32 for 40GB+")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate",
                   help="colocate=single GPU, server=separate vLLM process")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default=None,
                   help="e.g. your-hf-username/autodrive-agent")
    p.add_argument("--report-to", default="none",
                   choices=("none", "wandb", "tensorboard"))
    p.add_argument("--reward-log", default="reward_log.csv")
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="T=1.0 is optimal for GRPO exploration diversity")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    patch_trl_vllm_compat()
    args = parse_args()

    logger.info("=" * 64)
    logger.info("AutoDrive Gym — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 64)
    logger.info("Model:       %s", args.model_id)
    logger.info("Episodes:    %d", args.episodes)
    logger.info("Generations: %d", args.num_generations)
    logger.info("vLLM mode:   %s", args.vllm_mode)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Environment ───────────────────────────────────────────────────────────
    # Uses the local gym directly (no HTTP needed — faster rollouts)
    env = AutoDriveGymEnvironment()

    # ── Dataset (one entry per episode) ───────────────────────────────────────
    dataset = Dataset.from_dict(
        {"prompt": ["Navigate Indian road conditions safely."] * args.episodes}
    )

    # ── Output dir ────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_slug = args.model_id.replace("/", "-")
    out = Path(args.output_dir or f"outputs/autodrive-grpo-{model_slug}-{ts}")
    out.mkdir(parents=True, exist_ok=True)

    # ── Reward CSV logger ─────────────────────────────────────────────────────
    reward_log = out / args.reward_log
    episode_counter = [0]
    all_rewards: list[float] = []

    with open(reward_log, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "success", "steps", "timestamp"])

    def _log(total_r: float, success: bool, steps: int) -> None:
        episode_counter[0] += 1
        all_rewards.append(total_r)
        with open(reward_log, "a", newline="") as f:
            csv.writer(f).writerow([
                episode_counter[0], total_r, int(success), steps,
                datetime.now().isoformat(),
            ])
        n = len(all_rewards)
        m10 = sum(all_rewards[-10:]) / min(n, 10)
        logger.info(
            "Episode %3d: reward=%.3f  success=%s  steps=%d | "
            "mean(10)=%.3f  best=%.3f",
            episode_counter[0], total_r, "✅" if success else "❌", steps,
            m10, max(all_rewards),
        )

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(out),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",
        # DAPO improvements (same as kube winner)
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ── Rollout function (called by GRPOTrainer per GRPO step) ───────────────
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        total_rewards: list[float] = []
        successes: list[bool] = []

        for _ in prompts:
            ep = rollout_once(trainer, env, tokenizer, args.max_turns)
            all_prompt_ids.append(ep["prompt_ids"])
            all_completion_ids.append(ep["completion_ids"])
            all_logprobs.append(ep["logprobs"])
            total_rewards.append(ep["total_reward"])
            successes.append(ep["success"])
            _log(ep["total_reward"], ep["success"], ep["steps"])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "total_reward": total_rewards,
            "success": successes,
        }

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_total, reward_success],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting GRPO training — %d episodes × %d generations each",
                args.episodes, args.num_generations)
    try:
        trainer.train()
    finally:
        # Plot reward curve even if interrupted
        try:
            _plot_rewards(reward_log, out / "reward_plot.png")
        except Exception as e:
            logger.warning("Could not plot rewards: %s", e)

    trainer.save_model(str(out))
    logger.info("Model saved → %s", out)

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info("Pushed → https://huggingface.co/%s", args.hub_repo)

    logger.info("Done!")


# ── Reward plot helper ────────────────────────────────────────────────────────

def _plot_rewards(csv_path: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    episodes, rewards, = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))

    if not episodes:
        return

    window = min(10, len(episodes))
    rolling = [sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(rewards))]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, rewards, alpha=0.25, color="steelblue", marker="o", markersize=3, label="Per episode")
    ax.plot(episodes, rolling, color="steelblue", linewidth=2.5, label=f"Rolling avg ({window})")

    z = np.polyfit(episodes, rewards, 1)
    trend = np.poly1d(z)
    direction = "↑" if z[0] > 0 else "↓"
    ax.plot(episodes, trend(episodes), color="crimson", linewidth=1.5, linestyle="--",
            label=f"Trend {direction} {abs(z[0]):.3f}/ep")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("AutoDrive Gym — GRPO Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    stats = (f"Episodes: {len(episodes)} | "
             f"Final mean(10): {rolling[-1]:.3f} | "
             f"Best: {max(rewards):.3f}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Reward plot saved → %s", out_path)


if __name__ == "__main__":
    main()