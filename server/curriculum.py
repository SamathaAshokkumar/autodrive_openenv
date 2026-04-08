"""Curriculum controller for AutoDrive Gym.

Progressive difficulty design:
- Starts at warmup (difficulty ≈ 0.15) with only easy scenarios unlocked.
- Advances through 5 tiers as the agent demonstrates sustained success.
- Difficulty controls: which scenarios are unlocked, step budget, judge persona strictness.
- Mastery of a scenario type is tracked separately — weak spots always get extra exposure.
- Momentum: successive successes push difficulty up faster; failures slow it down.
"""

from collections import defaultdict
import random
from typing import Dict, List

from .constants import SCENARIO_TYPES

MASTERY_THRESHOLD = 0.70   # success-rate required to consider a scenario type mastered
MIN_EPISODES_PER_SCENARIO = 3  # min episodes before a scenario can be marked mastered

DIFFICULTY_TIERS = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 4,  "advance_rate": 0.55},
    {"name": "beginner",     "max_diff": 0.45, "min_episodes": 5,  "advance_rate": 0.60},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 6,  "advance_rate": 0.65},
    {"name": "advanced",     "max_diff": 0.75, "min_episodes": 8,  "advance_rate": 0.70},
    {"name": "expert",       "max_diff": 0.92, "min_episodes": 0,  "advance_rate": 1.00},
]


class CurriculumController:
    def __init__(self):
        self.history: Dict[str, List[bool]] = defaultdict(list)
        self.step_counts: Dict[str, List[int]] = defaultdict(list)
        self.episode_rewards: List[float] = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0
        self._graduated: set = set()
        self._recent_fault_types: List[str] = []
        self._consecutive_successes = 0
        self._consecutive_failures = 0

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, failure_type: str, success: bool, steps: int, reward: float):
        self.history[failure_type].append(success)
        self.step_counts[failure_type].append(steps)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self._recent_fault_types.append(failure_type)
        self._recent_fault_types = self._recent_fault_types[-6:]
        self._tier_episodes += 1

        if success:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

        self._maybe_advance_tier()
        self._check_mastery(failure_type)

    def _check_mastery(self, scenario_type: str):
        results = self.history[scenario_type]
        if len(results) >= MIN_EPISODES_PER_SCENARIO:
            recent = results[-10:]
            if sum(recent) / len(recent) >= MASTERY_THRESHOLD:
                self._graduated.add(scenario_type)

    # ── Tier advancement ──────────────────────────────────────────────────────

    def _recent_success_rate(self, window: int = 10) -> float:
        all_results = [r for results in self.history.values() for r in results[-window:]]
        return (sum(all_results) / len(all_results)) if all_results else 0.0

    def _maybe_advance_tier(self):
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self._tier_episodes < tier["min_episodes"]:
            return
        rate = self._recent_success_rate()
        # Momentum: 3 consecutive successes can push a tier advance even before min rate
        if self._consecutive_successes >= 4 and rate >= tier["advance_rate"] - 0.10:
            self._tier_index += 1
            self._tier_episodes = 0
            self._consecutive_successes = 0
        elif rate >= tier["advance_rate"]:
            self._tier_index += 1
            self._tier_episodes = 0

    # ── Difficulty ────────────────────────────────────────────────────────────

    def get_difficulty(self) -> float:
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self.episode_count < 3:
            return 0.15  # always start easy

        prev_max = DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"] if self._tier_index > 0 else 0.10
        rate = self._recent_success_rate()

        # Momentum: bump difficulty faster after consecutive successes
        momentum_boost = min(0.05 * self._consecutive_successes, 0.10)
        # Drag: slow difficulty after consecutive failures
        failure_drag = min(0.05 * self._consecutive_failures, 0.12)

        raw = prev_max + rate * (tier["max_diff"] - prev_max) + momentum_boost - failure_drag
        return round(min(tier["max_diff"], max(prev_max, raw)), 3)

    # ── Persona ───────────────────────────────────────────────────────────────

    def get_judge_persona(self) -> str:
        return "principal"  # Always use the strict principal judge

    # ── Scenario selection ────────────────────────────────────────────────────

    def should_use_adversarial(self) -> bool:
        return self.get_difficulty() >= 0.75 and len(self._graduated) >= 3

    def pick_fault_type(self) -> str | None:
        if self.should_use_adversarial():
            return "adversarial"
        difficulty = self.get_difficulty()
        unlocked = [
            name for name, meta in SCENARIO_TYPES.items()
            if meta["min_difficulty"] <= difficulty and name != "adversarial"
        ]
        if not unlocked:
            return "pedestrian_crossing"

        # Diversify: avoid repeating the same type from the last 3 episodes
        recent_window = self._recent_fault_types[-3:]
        diversified = [name for name in unlocked if name not in recent_window] or list(unlocked)

        # Always try untried scenarios first (exploration before exploitation)
        untried = [name for name in diversified if name not in self.history]
        if untried:
            return random.choice(untried)

        # Then focus on weak spots (mastery < threshold)
        weak = self.get_weak_spots()
        weak_available = [w for w in weak if w in diversified]
        if weak_available:
            # Limit repetition of the same weak type consecutively
            not_recently_weak = [w for w in weak_available if w not in recent_window]
            return random.choice(not_recently_weak or weak_available)

        # Graduate-weighted random: mastered types get fewer plays
        weights = [1 if name in self._graduated else 3 for name in diversified]
        return random.choices(diversified, weights=weights, k=1)[0]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    def get_skill_profile(self) -> Dict[str, float]:
        return {
            s: round(sum(results[-10:]) / len(results[-10:]), 2)
            for s, results in self.history.items() if results
        }

    def get_weak_spots(self) -> List[str]:
        return [s for s, rate in self.get_skill_profile().items() if rate < MASTERY_THRESHOLD]

    def get_stats(self) -> dict:
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "difficulty": round(self.get_difficulty(), 3),
            "judge_persona": self.get_judge_persona(),
            "consecutive_successes": self._consecutive_successes,
            "consecutive_failures": self._consecutive_failures,
            "skill_profile": self.get_skill_profile(),
            "weak_spots": self.get_weak_spots(),
            "graduated": sorted(self._graduated),
        }
