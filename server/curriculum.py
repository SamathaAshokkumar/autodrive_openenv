"""Curriculum controller for AutoDrive Gym.

Progressive difficulty design:
- Starts at warmup (difficulty ≈ 0.15) with only easy scenarios unlocked.
- Advances through 5 tiers as the agent demonstrates sustained success.
- Difficulty controls: which scenarios are unlocked, step budget, judge persona strictness.
- Mastery of a scenario type is tracked separately — weak spots always get extra exposure.
- Momentum: successive successes push difficulty up faster; failures slow it down.
"""

from collections import defaultdict
import logging
import random
from typing import Dict, List, Optional, TYPE_CHECKING

from .constants import SCENARIO_TYPES

if TYPE_CHECKING:
    from .adversarial_designer import AdversarialDesigner

logger = logging.getLogger(__name__)

MASTERY_THRESHOLD = 0.70   # success-rate required to consider a scenario type mastered
MIN_EPISODES_PER_SCENARIO = 3  # min episodes before a scenario can be marked mastered

DIFFICULTY_TIERS = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 8,  "advance_rate": 0.62},
    {"name": "beginner",     "max_diff": 0.45, "min_episodes": 10, "advance_rate": 0.65},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 12, "advance_rate": 0.68},
    {"name": "advanced",     "max_diff": 0.75, "min_episodes": 15, "advance_rate": 0.72},
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
        # Theme 4: Self-Improvement — designer is injected externally
        self._adversarial_designer: Optional["AdversarialDesigner"] = None
        self._self_improve_cooldown = 0   # steps before next self-improvement trigger
        self._self_improve_triggered_count = 0

    def set_adversarial_designer(self, designer: "AdversarialDesigner") -> None:
        """Inject the AdversarialDesigner for self-improvement (Theme 4)."""
        self._adversarial_designer = designer

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
        if self._self_improve_cooldown > 0:
            self._self_improve_cooldown -= 1

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
        # Momentum: require 6 consecutive successes (up from 4) to early-advance
        if self._consecutive_successes >= 6 and rate >= tier["advance_rate"] - 0.08:
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

    # ── Self-improvement (Theme 4) ────────────────────────────────────────────

    def should_self_improve(self) -> bool:
        """True when the agent is stuck enough to benefit from a targeted challenge."""
        return (
            self._adversarial_designer is not None
            and self._consecutive_failures >= 3
            and self._self_improve_cooldown == 0
        )

    def generate_self_improvement_scenario(self) -> Optional[dict]:
        """Generate a targeted adversarial scenario focused on the agent's weak spots.

        Called automatically when the agent fails 3+ consecutive episodes.  This
        implements Theme 4 (Self-Improvement): the environment adapts to the agent's
        specific failure profile instead of replaying random scenarios.
        """
        if self._adversarial_designer is None:
            return None
        skill_profile = self.get_skill_profile()
        difficulty = self.get_difficulty()
        try:
            scenario = self._adversarial_designer.design(skill_profile, difficulty)
            self._self_improve_triggered_count += 1
            self._self_improve_cooldown = 3   # cool-down: 3 episodes before next trigger
            logger.info(
                "[Self-Improve] Generated targeted scenario after %d consecutive failures "
                "(trigger #%d). Weak spots: %s",
                self._consecutive_failures,
                self._self_improve_triggered_count,
                self.get_weak_spots(),
            )
            return scenario
        except Exception as exc:
            logger.warning("Self-improvement scenario generation failed: %s", exc)
            return None

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
            "self_improve_triggered": self._self_improve_triggered_count,
            "self_improve_cooldown": self._self_improve_cooldown,
        }