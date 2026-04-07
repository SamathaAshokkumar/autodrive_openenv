"""Rollout helpers for AutoDrive Gym.

Kept to mirror the kube project structure while `eval.py` remains the stable
judge-facing inference entrypoint.
"""

SYSTEM_PROMPT = """You are an autonomous driving agent for Indian road conditions.
Return one JSON action per turn: {\"action\": \"...\", \"value\": <float>}"""


def format_observation(obs) -> str:
    return f"EVENT: {getattr(obs, 'event_log', '')}\nSCENE: {getattr(obs, 'scene_summary', '')}\nSENSOR: {getattr(obs, 'sensor_data', {})}\nEGO: {getattr(obs, 'ego_state', {})}\nENV: {getattr(obs, 'environment', {})}\nSTEP: {getattr(obs, 'steps_taken', 0)}/{getattr(obs, 'max_steps', 20)}"


def format_history(history):
    return "\n".join(f"step {i+1}: {item}" for i, item in enumerate(history))


def parse_actions(text: str):
    import json
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'action' in data:
            return [data]
    except Exception:
        pass
    return []
