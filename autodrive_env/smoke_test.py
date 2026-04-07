"""Quick submission smoke test for AutoDrive Gym.

Verifies that the OpenEnv client can:
- connect
- reset
- step
- fetch state
"""

from __future__ import annotations

import argparse

from autodrive_env import AutoDriveAction, AutoDriveClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test AutoDrive Gym reset/step/state flow.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="OpenEnv server base URL")
    args = parser.parse_args()

    ctx = AutoDriveClient(base_url=args.base_url).sync()
    env = ctx.__enter__()
    try:
        reset_result = env.reset()
        obs = reset_result.observation
        print("RESET OK", obs.scenario_type, obs.steps_taken, obs.max_steps)

        step_result = env.step(AutoDriveAction(action="wait", value=0.0))
        print("STEP OK", step_result.reward, step_result.done, step_result.observation.steps_taken)

        state = env.state()
        print("STATE OK", state.episode_id, state.step_count, state.incident_type)
    finally:
        ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
