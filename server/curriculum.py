"""Curriculum controller for AutoDrive Gym."""

from collections import defaultdict
import random

from .constants import SCENARIO_TYPES

MASTERY_THRESHOLD = 0.7
DIFFICULTY_TIERS = [
    {"name": "warmup", "max_diff": 0.25, "min_episodes": 4, "advance_rate": 0.6},
    {"name": "beginner", "max_diff": 0.45, "min_episodes": 5, "advance_rate": 0.65},
    {"name": "intermediate", "max_diff": 0.65, "min_episodes": 6, "advance_rate": 0.7},
    {"name": "advanced", "max_diff": 0.85, "min_episodes": 8, "advance_rate": 0.75},
    {"name": "expert", "max_diff": 0.95, "min_episodes": 0, "advance_rate": 1.0},
]


class CurriculumController:
    def __init__(self):
        self.history = defaultdict(list)
        self.step_counts = defaultdict(list)
        self.episode_rewards = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0
        self._graduated = set()
        self._recent_fault_types = []

    def record(self, failure_type: str, success: bool, steps: int, reward: float):
        self.history[failure_type].append(success)
        self.step_counts[failure_type].append(steps)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self._recent_fault_types.append(failure_type)
        self._recent_fault_types = self._recent_fault_types[-5:]
        self._tier_episodes += 1
        self._maybe_advance_tier()
        recent = self.history[failure_type][-10:]
        if len(recent) >= 3 and sum(recent) / len(recent) >= MASTERY_THRESHOLD:
            self._graduated.add(failure_type)

    def _recent_success_rate(self, window: int = 10) -> float:
        all_results = [r for results in self.history.values() for r in results[-window:]]
        return (sum(all_results) / len(all_results)) if all_results else 0.0

    def _maybe_advance_tier(self):
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self._tier_episodes < tier["min_episodes"]:
            return
        if self._recent_success_rate() >= tier["advance_rate"]:
            self._tier_index += 1
            self._tier_episodes = 0

    def get_difficulty(self) -> float:
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self.episode_count < 3:
            return 0.15
        prev_max = DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"] if self._tier_index > 0 else 0.1
        rate = self._recent_success_rate()
        return min(tier["max_diff"], prev_max + rate * (tier["max_diff"] - prev_max))

    def get_judge_persona(self) -> str:
        difficulty = self.get_difficulty()
        if difficulty < 0.35:
            return "junior"
        if difficulty < 0.7:
            return "senior"
        return "principal"

    def get_tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    def get_skill_profile(self) -> dict:
        return {scenario: round(sum(results[-10:]) / len(results[-10:]), 2) for scenario, results in self.history.items() if results}

    def get_weak_spots(self) -> list[str]:
        return [scenario for scenario, rate in self.get_skill_profile().items() if rate < MASTERY_THRESHOLD]

    def should_use_adversarial(self) -> bool:
        return self.get_difficulty() >= 0.7 and len(self._graduated) >= 2

    def pick_fault_type(self) -> str | None:
        if self.should_use_adversarial():
            return "adversarial"
        difficulty = self.get_difficulty()
        unlocked = [name for name, meta in SCENARIO_TYPES.items() if meta["min_difficulty"] <= difficulty and name != "adversarial"]
        if not unlocked:
            return "pedestrian_crossing"

        # Avoid repeating any of the last few scenario families when alternatives exist.
        recent_window = self._recent_fault_types[-3:]
        diversified = [name for name in unlocked if name not in recent_window]
        if not diversified:
            diversified = list(unlocked)

        # First prefer unexplored scenario families to keep evaluation diverse.
        untried = [name for name in diversified if name not in self.history]
        if untried:
            return random.choice(untried)

        weak = [name for name in self.get_weak_spots() if name in diversified]
        if weak:
            weak_not_recent = [name for name in weak if name not in recent_window]
            return random.choice(weak_not_recent or weak)

        weights = [1 if name in self._graduated else 2 for name in diversified]
        return random.choices(diversified, weights=weights, k=1)[0] if diversified else unlocked[0]

    def get_stats(self) -> dict:
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "skill_profile": self.get_skill_profile(),
            "weak_spots": self.get_weak_spots(),
            "graduated": sorted(self._graduated),
        }
