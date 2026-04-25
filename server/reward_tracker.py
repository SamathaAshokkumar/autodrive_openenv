"""Reward tracker for AutoDrive Gym — generates reward curves for judging.

Tracks per-step and per-episode metrics throughout training and exports them
in a format suitable for:
  - JSON analysis / custom plots
  - Matplotlib reward curve visualisation (runs standalone)
  - The /metrics API endpoint

Covers the judging criterion:
  "Showing Improvement in Rewards (20%): observable evidence of training progress"
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "reward_log.json")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    action: str
    value: float
    reward: float
    collision: bool = False
    near_miss: bool = False
    safe_distance: bool = True
    progress_restored: bool = False


@dataclass
class EpisodeRecord:
    episode: int
    episode_id: str
    scenario_type: str
    difficulty: float
    tier: str
    success: bool
    total_reward: float
    steps: int
    mean_step_reward: float
    timestamp: float = field(default_factory=time.time)
    steps_data: List[StepRecord] = field(default_factory=list)
    pipeline_used: bool = False       # True if MultiAgentPipeline was active
    route_mode: bool = False
    route_progress_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("steps_data")  # keep summary JSON small
        return d


# ── Tracker ───────────────────────────────────────────────────────────────────

class RewardTracker:
    """Accumulates training metrics and can export reward curves.

    Usage::

        tracker = RewardTracker()
        tracker.start_episode(1, "pedestrian_crossing", 0.2, "warmup")
        tracker.record_step(1, "brake", 0.8, 0.72, collision=False, safe_distance=True)
        tracker.end_episode(success=True, cumulative_reward=4.3, steps=9)
        tracker.save()
        curves = tracker.get_curves()
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        self.episodes: List[EpisodeRecord] = []
        self._current: Optional[EpisodeRecord] = None
        self._by_scenario: Dict[str, List[float]] = defaultdict(list)
        self._by_tier: Dict[str, List[float]] = defaultdict(list)

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def start_episode(
        self,
        episode_num: int,
        episode_id: str,
        scenario_type: str,
        difficulty: float,
        tier: str,
        pipeline_used: bool = False,
        route_mode: bool = False,
    ) -> None:
        self._current = EpisodeRecord(
            episode=episode_num,
            episode_id=episode_id,
            scenario_type=scenario_type,
            difficulty=difficulty,
            tier=tier,
            success=False,
            total_reward=0.0,
            steps=0,
            mean_step_reward=0.0,
            pipeline_used=pipeline_used,
            route_mode=route_mode,
        )

    def record_step(
        self,
        step: int,
        action: str,
        value: float,
        reward: float,
        collision: bool = False,
        near_miss: bool = False,
        safe_distance: bool = True,
        progress_restored: bool = False,
    ) -> None:
        if self._current is None:
            return
        self._current.steps_data.append(StepRecord(
            step=step,
            action=action,
            value=value,
            reward=reward,
            collision=collision,
            near_miss=near_miss,
            safe_distance=safe_distance,
            progress_restored=progress_restored,
        ))
        self._current.total_reward += reward
        self._current.steps += 1

    def end_episode(
        self,
        success: bool,
        cumulative_reward: Optional[float] = None,
        route_progress_pct: float = 0.0,
    ) -> EpisodeRecord:
        if self._current is None:
            raise RuntimeError("end_episode called without start_episode")
        ep = self._current
        ep.success = success
        if cumulative_reward is not None:
            ep.total_reward = cumulative_reward
        ep.mean_step_reward = round(ep.total_reward / max(ep.steps, 1), 4)
        ep.route_progress_pct = route_progress_pct

        self.episodes.append(ep)
        self._by_scenario[ep.scenario_type].append(ep.total_reward)
        self._by_tier[ep.tier].append(ep.total_reward)
        self._current = None
        return ep

    # ── Data export ───────────────────────────────────────────────────────────

    def get_curves(self) -> Dict[str, Any]:
        """Return reward curve data for the /metrics endpoint and plotting."""
        if not self.episodes:
            return {"episodes": [], "summary": {}}

        rewards = [ep.total_reward for ep in self.episodes]
        successes = [int(ep.success) for ep in self.episodes]
        difficulties = [ep.difficulty for ep in self.episodes]

        rolling_10 = _rolling_avg(rewards, window=10)
        rolling_20 = _rolling_avg(rewards, window=20)

        # Per-scenario type success rates
        by_scenario: Dict[str, Dict[str, Any]] = {}
        for stype, ep_rewards in self._by_scenario.items():
            eps = [e for e in self.episodes if e.scenario_type == stype]
            sr = sum(1 for e in eps if e.success) / max(len(eps), 1)
            by_scenario[stype] = {
                "episodes": len(eps),
                "mean_reward": round(sum(ep_rewards) / len(ep_rewards), 4),
                "success_rate": round(sr, 3),
            }

        # Tier progression
        tier_order = ["warmup", "beginner", "intermediate", "advanced", "expert"]
        tier_progression = {
            t: {
                "episodes": len(self._by_tier[t]),
                "mean_reward": round(sum(self._by_tier[t]) / max(len(self._by_tier[t]), 1), 4),
            }
            for t in tier_order if self._by_tier[t]
        }

        # Pipeline impact (with vs without multi-agent)
        with_pipeline = [e.total_reward for e in self.episodes if e.pipeline_used]
        without_pipeline = [e.total_reward for e in self.episodes if not e.pipeline_used]

        return {
            "episode_count": len(self.episodes),
            "reward_curve": [round(r, 4) for r in rewards],
            "rolling_10_curve": [round(r, 4) for r in rolling_10],
            "rolling_20_curve": [round(r, 4) for r in rolling_20],
            "success_curve": successes,
            "difficulty_curve": [round(d, 3) for d in difficulties],
            "overall": {
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                "max_reward": round(max(rewards), 4),
                "min_reward": round(min(rewards), 4),
                "mean_success_rate": round(sum(successes) / len(successes), 3),
                "final_10_mean": round(sum(rewards[-10:]) / len(rewards[-10:]), 4),
            },
            "by_scenario": by_scenario,
            "tier_progression": tier_progression,
            "pipeline_impact": {
                "with_pipeline_mean": round(sum(with_pipeline) / max(len(with_pipeline), 1), 4),
                "without_pipeline_mean": round(sum(without_pipeline) / max(len(without_pipeline), 1), 4),
                "pipeline_episodes": len(with_pipeline),
            },
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def save(self) -> None:
        """Persist reward log to JSON."""
        try:
            curves = self.get_curves()
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as fh:
                json.dump(curves, fh, indent=2)
            logger.info("Reward log saved to %s (%d episodes).", self.log_path, len(self.episodes))
        except Exception as exc:
            logger.warning("Could not save reward log: %s", exc)

    @classmethod
    def load(cls, log_path: str = DEFAULT_LOG_PATH) -> Dict[str, Any]:
        """Load and return persisted reward curves (for /metrics endpoint)."""
        try:
            with open(log_path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {"error": "No reward log found. Run training first.", "episodes": []}
        except Exception as exc:
            return {"error": str(exc), "episodes": []}

    def summary_line(self) -> str:
        """One-line training summary for console output."""
        if not self.episodes:
            return "No episodes recorded."
        rewards = [ep.total_reward for ep in self.episodes]
        sr = sum(1 for e in self.episodes if e.success) / len(self.episodes)
        last10 = rewards[-10:]
        return (
            f"Episodes: {len(self.episodes)} | "
            f"Mean reward: {sum(rewards)/len(rewards):.3f} | "
            f"Last-10 mean: {sum(last10)/len(last10):.3f} | "
            f"Success rate: {sr:.1%}"
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _rolling_avg(values: List[float], window: int = 10) -> List[float]:
    result = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(round(sum(chunk) / len(chunk), 4))
    return result