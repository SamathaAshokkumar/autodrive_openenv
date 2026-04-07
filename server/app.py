"""FastAPI application for AutoDrive Gym."""

import argparse
import logging
import os

from ..models import AutoDriveAction, AutoDriveObservation
from ..openenv_compat import OPENENV_AVAILABLE, create_app
from .autodrive_gym_environment import AutoDriveGymEnvironment

logger = logging.getLogger(__name__)

app = create_app(
    AutoDriveGymEnvironment,
    AutoDriveAction,
    AutoDriveObservation,
    env_name="autodrive_gym",
    max_concurrent_envs=1,
)


@app.get("/healthz")
async def healthz():
    try:
        env = AutoDriveGymEnvironment()
        return {
            "status": "ok",
            "openenv_available": OPENENV_AVAILABLE,
            "difficulty": env.curriculum.get_difficulty(),
            "judge_persona": env.curriculum.get_judge_persona(),
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "pedestrian_crossing", "difficulty": "easy", "description": "Vulnerable road user crosses suddenly"},
            {"id": "auto_cut_in", "difficulty": "easy", "description": "Auto-rickshaw cuts in unpredictably"},
            {"id": "bike_blind_spot", "difficulty": "medium", "description": "Bike merges from blind spot"},
            {"id": "pothole_ahead", "difficulty": "medium", "description": "Road surface hazard requires smooth avoidance"},
            {"id": "traffic_light_ambiguity", "difficulty": "medium", "description": "Conflicting signals require cautious response"},
            {"id": "adversarial", "difficulty": "hard", "description": "Multiple unpredictable agents in chaotic Indian traffic"},
        ],
        "action_schema": {
            "action": {"type": "string", "enum": ["accelerate", "brake", "steer_left", "steer_right", "horn", "wait", "change_lane_left", "change_lane_right"]},
            "value": {"type": "float", "description": "Intensity or steering magnitude"},
        },
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    if not OPENENV_AVAILABLE:
        raise ImportError("OpenEnv is not installed. Install openenv-core to run the AutoDrive server.")

    import uvicorn

    parser = argparse.ArgumentParser(description="AutoDrive Gym OpenEnv server")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--host", default=host)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
