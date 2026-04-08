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


_TASKS = [
    {"id": "pedestrian_crossing", "difficulty": "easy",   "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Vulnerable road user crosses suddenly"},
    {"id": "auto_cut_in",         "difficulty": "easy",   "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Auto-rickshaw cuts in unpredictably"},
    {"id": "bike_blind_spot",     "difficulty": "medium", "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Bike merges from blind spot"},
    {"id": "pothole_ahead",       "difficulty": "medium", "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Road surface hazard requires smooth avoidance"},
    {"id": "traffic_light_ambiguity", "difficulty": "medium", "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Conflicting signals require cautious response"},
    {"id": "adversarial",         "difficulty": "hard",   "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "Multiple unpredictable agents in chaotic Indian traffic"},
]

_ACTION_SCHEMA = {
    "action": {"type": "string", "enum": ["accelerate", "brake", "steer_left", "steer_right", "horn", "wait", "change_lane_left", "change_lane_right"]},
    "value": {"type": "float", "min": 0.0, "max": 1.0, "description": "Intensity or steering magnitude"},
}

# Pre-built test cases for each task used by /grader and /baseline ---------------
_GRADER_TEST_CASES = {
    "pedestrian_crossing":    {"obs": {"sensor_data": {"objects": [{"type": "pedestrian", "distance": 8.0}]}}, "action": {"action": "brake", "value": 0.9}, "result": {"safe_distance": True, "collision": False, "near_miss": False, "signal_respected": True}},
    "auto_cut_in":            {"obs": {"sensor_data": {"objects": [{"type": "auto",       "distance": 7.0}]}}, "action": {"action": "brake", "value": 0.7}, "result": {"safe_distance": True, "collision": False, "near_miss": False, "signal_respected": True}},
    "bike_blind_spot":        {"obs": {"sensor_data": {"objects": [{"type": "bike",       "distance": 6.0}]}}, "action": {"action": "wait",  "value": 0.0}, "result": {"safe_distance": True, "collision": False, "near_miss": False, "signal_respected": True}},
    "pothole_ahead":          {"obs": {"sensor_data": {"objects": [{"type": "pothole",    "distance": 10.0}]}}, "action": {"action": "brake", "value": 0.5}, "result": {"safe_distance": True, "collision": False, "near_miss": False, "signal_respected": True}},
    "traffic_light_ambiguity":{"obs": {"sensor_data": {"objects": [{"type": "traffic_police", "distance": 9.0}]}}, "action": {"action": "wait",  "value": 0.0}, "result": {"safe_distance": True, "collision": False, "near_miss": False, "signal_respected": True}},
    "adversarial":            {"obs": {"sensor_data": {"objects": [{"type": "car",        "distance": 5.0}, {"type": "bike", "distance": 6.0}]}}, "action": {"action": "brake", "value": 0.8}, "result": {"safe_distance": False, "collision": False, "near_miss": True, "signal_respected": True}},
}


@app.get("/tasks")
async def list_tasks():
    return {"tasks": _TASKS, "action_schema": _ACTION_SCHEMA}


@app.get("/grader")
async def grader_endpoint(task_id: str = "pedestrian_crossing"):
    """Return a strict principal-judge score in (0.02, 0.98) for the given task."""
    from .judge import HeuristicGrader
    grader = HeuristicGrader(persona="principal")
    test = _GRADER_TEST_CASES.get(task_id, _GRADER_TEST_CASES["pedestrian_crossing"])
    result = grader(test["obs"], test["action"], test["result"], {"type": task_id, "expected_behavior": ["brake", "wait"]}, [])
    return {
        "task_id": task_id,
        "persona": "principal",
        "score": result["score"],
        "dimensions": {
            "safety": result.get("safety"),
            "efficiency": result.get("efficiency"),
            "compliance": result.get("compliance"),
        },
        "feedback": result["feedback"],
    }


@app.get("/baseline")
async def baseline_scores():
    """Run the strict principal-judge HeuristicGrader against all tasks.

    All scores guaranteed in (0.02, 0.98).
    """
    from .judge import HeuristicGrader
    grader = HeuristicGrader(persona="principal")
    results = []
    for task in _TASKS:
        tid = task["id"]
        test = _GRADER_TEST_CASES.get(tid, _GRADER_TEST_CASES["pedestrian_crossing"])
        scored = grader(test["obs"], test["action"], test["result"], {"type": tid, "expected_behavior": ["brake", "wait"]}, [])
        results.append({
            "task_id": tid,
            "difficulty": task["difficulty"],
            "persona": "principal",
            "score": scored["score"],
            "dimensions": {
                "safety": scored.get("safety"),
                "efficiency": scored.get("efficiency"),
                "compliance": scored.get("compliance"),
            },
            "feedback": scored["feedback"],
        })
    return {"status": "ok", "baseline_scores": results}


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
