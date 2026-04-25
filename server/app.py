"""FastAPI application for AutoDrive Gym."""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Body
from fastapi.responses import HTMLResponse, StreamingResponse

from ..models import AutoDriveAction, AutoDriveObservation
from ..openenv_compat import OPENENV_AVAILABLE, create_app
from .autodrive_gym_environment import AutoDriveGymEnvironment
from .fleet_coordinator import FleetCoordinator
from .multi_agent_pipeline import MultiAgentPipeline
from .reward_tracker import RewardTracker
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Live state file written by train.py during training
_LIVE_STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "live_state.json")

app = create_app(
    AutoDriveGymEnvironment,
    AutoDriveAction,
    AutoDriveObservation,
    env_name="autodrive_gym",
    max_concurrent_envs=1,
)

# Singletons for new features (shared across requests)
_llm = LLMClient()
_fleet = FleetCoordinator(_llm)
_pipeline = MultiAgentPipeline(_llm)


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
    # Theme 2: Long-horizon multi-checkpoint city route
    {"id": "city_route",          "difficulty": "long_horizon", "grader": "autodrive_env.server.judge:HeuristicGrader", "description": "5-checkpoint city route: Market→Railway→School→Hospital→Highway (sparse reward)"},
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
            "safety":              result.get("safety"),
            "efficiency":          result.get("efficiency"),
            "compliance":          result.get("compliance"),
            "smoothness":          result.get("smoothness"),
            "negotiation":         result.get("negotiation"),
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
                "safety":      scored.get("safety"),
                "efficiency":  scored.get("efficiency"),
                "compliance":  scored.get("compliance"),
                "smoothness":  scored.get("smoothness"),
                "negotiation": scored.get("negotiation"),
            },
            "feedback": scored["feedback"],
        })
    return {"status": "ok", "baseline_scores": results}


# ── Theme 1: Multi-agent pipeline endpoint ────────────────────────────────────

@app.post("/pipeline/decide")
async def pipeline_decide(
    observation: Dict[str, Any] = Body(...),
    history: list = Body(default=[]),
):
    """Run the 6-stage multi-agent pipeline (Perception→Context→IntentInference→Negotiation→Decision→Oversight).

    Demonstrates Theme 1 multi-agent cooperation with intent modeling and
    negotiation. Clients can call this before submitting a /step action to get
    the full pipeline trace.
    """
    try:
        action, trace = _pipeline.run(observation, history)
        return {
            "status": "ok",
            "final_action": action,
            "pipeline_trace": trace,
            "stages": {
                "1_perception":       trace.get("perception", {}),
                "2_context":          trace.get("context", {}),
                "2.5_intent":         trace.get("intent_inference", {}),
                "3_negotiation":      trace.get("negotiation", {}),
                "3.5_decision":       trace.get("proposed_decision", {}),
                "4_oversight":        trace.get("oversight", {}),
            },
            "was_overridden": trace.get("was_overridden", False),
        }
    except Exception as exc:
        logger.error("Pipeline decide error: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


# ── Theme 1: Fleet AI endpoints ───────────────────────────────────────────────

@app.post("/fleet/reset")
async def fleet_reset(
    n_vehicles: int = 2,
    continue_on_fail: bool = True,
):
    """Start a new shared city-route fleet session (10 checkpoints).

    Both vehicles begin at CP 0 together and advance in sync.
    Sudden alerts fire mid-task into both agents' observations.
    """
    try:
        summary = _fleet.reset(n_vehicles=n_vehicles, continue_on_fail=continue_on_fail)
        return {"status": "ok", **summary}
    except Exception as exc:
        logger.error("Fleet reset error: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@app.post("/fleet/step")
async def fleet_step(
    vehicle_id: str = "vehicle_0",
    action: str = "brake",
    value: float = 0.8,
):
    """Step a single fleet vehicle (other vehicles hold their last action).

    Returns the vehicle observation + any sudden alert that fired this step.
    """
    try:
        result = _fleet.step(vehicle_id, {"action": action, "value": value})
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("Fleet step error: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@app.post("/fleet/step_all")
async def fleet_step_all(
    actions: Dict[str, Any] = Body(
        ...,
        example={"vehicle_0": {"action": "brake", "value": 0.8},
                 "vehicle_1": {"action": "steer_left", "value": 0.5}},
    ),
):
    """Step ALL fleet vehicles simultaneously (recommended for two-agent training).

    Runs the full negotiation → oversight → execute → broadcast → transition loop.
    Returns per-vehicle observations + sudden_alert (if any) + route_event.
    """
    try:
        result = _fleet.step_fleet(actions)
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("Fleet step_all error: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@app.get("/fleet/status")
async def fleet_status():
    """Return the current fleet state (all vehicles, shared alerts, oversight notes)."""
    return {"status": "ok", **_fleet.get_fleet_status()}


# ── Theme 2: Route status endpoint ────────────────────────────────────────────

@app.get("/route/status")
async def route_status():
    """Return long-horizon route progress.

    Start a route session via /reset?task_id=city_route.
    This endpoint shows checkpoint progress, remaining checkpoints, and sparse reward.
    """
    from .route_planner import CITY_ROUTE, ROUTE_COMPLETION_BONUS
    return {
        "status": "ok",
        "route": {
            "total_checkpoints": len(CITY_ROUTE),
            "completion_bonus": ROUTE_COMPLETION_BONUS,
            "checkpoints": [cp.to_dict() for cp in CITY_ROUTE],
        },
        "note": "Start a city route by calling /reset with task_id=city_route",
    }


# ── Reward curves / metrics ───────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Return reward curves and training metrics for the current session.

    Judging criterion: 'Showing Improvement in Rewards'.
    Data includes:
      - per-episode reward curve
      - rolling-10 and rolling-20 averages
      - success rate over time
      - per-scenario-type performance
      - tier progression
      - multi-agent pipeline impact
    """
    curves = RewardTracker.load()
    return {"status": "ok", "metrics": curves}


@app.get("/metrics/summary")
async def metrics_summary():
    """One-line training summary."""
    curves = RewardTracker.load()
    if "error" in curves:
        return {"status": "no_data", "message": curves["error"]}
    overall = curves.get("overall", {})
    return {
        "status": "ok",
        "episode_count": curves.get("episode_count", 0),
        "mean_reward": overall.get("mean_reward"),
        "final_10_mean_reward": overall.get("final_10_mean"),
        "mean_success_rate": overall.get("mean_success_rate"),
        "reward_improved": (
            overall.get("final_10_mean", 0) > overall.get("mean_reward", 0)
        ),
        "by_scenario": curves.get("by_scenario", {}),
        "tier_progression": curves.get("tier_progression", {}),
    }



# ── Live training demo ─────────────────────────────────────────────────────────

@app.get("/demo/stream")
async def demo_stream():
    """Server-Sent Events stream of real-time training state.

    train.py writes live_state.json at every step; this endpoint tails it and
    pushes updates to the browser at 4 Hz.  Open /demo for the visual dashboard.
    """
    async def generate():
        last_mtime = -1.0
        while True:
            try:
                mtime = os.path.getmtime(_LIVE_STATE_PATH)
                if mtime != last_mtime:
                    with open(_LIVE_STATE_PATH, encoding="utf-8") as _f:
                        data = json.load(_f)
                    last_mtime = mtime
                    yield f"data: {json.dumps(data)}\n\n"
            except (FileNotFoundError, json.JSONDecodeError):
                yield "data: {\"waiting\": true}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Live training dashboard — auto-updates as train.py runs."""
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>AutoDrive Gym — Live Training Demo</title>
<style>
  :root{--green:#22c55e;--red:#ef4444;--yellow:#eab308;--blue:#3b82f6;--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#e2e8f0;--muted:#94a3b8;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;padding:16px;}
  h1{text-align:center;font-size:1.4rem;margin-bottom:16px;color:var(--green);}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;}
  .card h2{font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;}
  .badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:.75rem;font-weight:600;}
  .badge-green{background:#14532d;color:var(--green);}
  .badge-yellow{background:#713f12;color:var(--yellow);}
  .badge-red{background:#7f1d1d;color:var(--red);}
  .badge-blue{background:#1e3a8a;color:#93c5fd;}
  .stat{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);}
  .stat:last-child{border-bottom:none;}
  .stat-val{font-size:1.1rem;font-weight:700;color:var(--text);}
  /* road canvas */
  #road-canvas{display:block;margin:0 auto;border-radius:8px;}
  /* reward chart */
  #reward-canvas{display:block;margin:0 auto;}
  /* mistake box */
  #mistake-box{min-height:40px;padding:8px;border-radius:6px;background:#1a1a2e;border:1px solid var(--border);font-size:.82rem;color:var(--yellow);line-height:1.4;}
  #hint-box{min-height:32px;padding:8px;border-radius:6px;background:#0d2137;border:1px solid #1e3a8a;font-size:.82rem;color:#93c5fd;line-height:1.4;}
  .waiting{text-align:center;color:var(--muted);padding:30px;font-size:.9rem;}
  #status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--muted);margin-right:6px;vertical-align:middle;}
  .dot-live{background:var(--green);box-shadow:0 0 6px var(--green);}
  .sr-bar{height:8px;border-radius:4px;background:var(--border);margin-top:6px;overflow:hidden;}
  .sr-fill{height:100%;border-radius:4px;background:var(--green);transition:width .4s;}
</style>
</head>
<body>
<h1>🚗 AutoDrive Gym — Live Training Dashboard</h1>

<div id="waiting" class="waiting">
  <p>Waiting for training to start…</p>
  <p style="margin-top:8px;font-size:.78rem;">Run: <code>python -m autodrive_env.train --mode adaptive</code></p>
</div>

<div id="main" style="display:none">
<div class="grid">

  <!-- Episode info -->
  <div class="card">
    <h2><span id="status-dot"></span> Training Status</h2>
    <div class="stat"><span>Episode</span><span class="stat-val" id="ep-val">—</span></div>
    <div class="stat"><span>Step</span><span class="stat-val" id="step-val">—</span></div>
    <div class="stat"><span>Scenario</span><span id="scenario-badge">—</span></div>
    <div class="stat"><span>Stage</span><span id="stage-badge">—</span></div>
    <div class="stat"><span>Tier</span><span id="tier-val" class="stat-val">—</span></div>
    <div class="stat"><span>Difficulty</span><span class="stat-val" id="diff-val">—</span></div>
  </div>

  <!-- Action & reward -->
  <div class="card">
    <h2>Agent Action &amp; Reward</h2>
    <div class="stat"><span>Action</span><span id="action-badge">—</span></div>
    <div class="stat"><span>Value</span><span class="stat-val" id="value-val">—</span></div>
    <div class="stat"><span>Step reward</span><span class="stat-val" id="reward-val">—</span></div>
    <div class="stat"><span>Rolling-10 reward</span><span class="stat-val" id="roll10-val">—</span></div>
    <div class="stat"><span>Success rate (last 20)</span><span class="stat-val" id="sr-val">—</span></div>
    <div class="sr-bar"><div class="sr-fill" id="sr-fill" style="width:0%"></div></div>
    <div style="margin-top:10px;font-size:.78rem;color:var(--muted)">Hazard: <span id="hazard-val">—</span></div>
  </div>

  <!-- Road visualization -->
  <div class="card" style="grid-column:span 2">
    <h2>Road Scene</h2>
    <canvas id="road-canvas" width="560" height="130"></canvas>
  </div>

  <!-- Reward curve -->
  <div class="card" style="grid-column:span 2">
    <h2>Reward Curve (last 30 episodes)</h2>
    <canvas id="reward-canvas" width="560" height="140"></canvas>
  </div>

  <!-- Feedback -->
  <div class="card" style="grid-column:span 2">
    <h2>Learning Feedback</h2>
    <div style="margin-bottom:6px;font-size:.75rem;color:var(--muted)">HINT</div>
    <div id="hint-box">—</div>
    <div style="margin-top:10px;margin-bottom:6px;font-size:.75rem;color:var(--muted)">MISTAKE DETECTED</div>
    <div id="mistake-box">—</div>
    <div style="margin-top:10px;font-size:.75rem;color:var(--muted)">ALERTS: <span id="alerts-val">none</span></div>
  </div>

</div><!-- grid -->
</div><!-- main -->

<script>
const ACTION_COLORS = {
  brake:'#ef4444', accelerate:'#22c55e', wait:'#eab308',
  steer_left:'#3b82f6', steer_right:'#8b5cf6',
  horn:'#f97316', change_lane_left:'#06b6d4', change_lane_right:'#14b8a6',
};
const STAGE_COLORS = {approaching:'#ef4444', clearing:'#eab308', cleared:'#22c55e'};

function badgeCls(action) {
  const c = ACTION_COLORS[action] || '#6b7280';
  return `style="background:${c}22;color:${c};border:1px solid ${c};padding:2px 8px;border-radius:9999px;font-size:.75rem;font-weight:600;"`;
}

function drawRoad(canvas, state) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  // Road background
  ctx.fillStyle = '#374151';
  ctx.fillRect(0, H*0.25, W, H*0.5);

  // Lane markings
  ctx.strokeStyle = '#ffffff33'; ctx.lineWidth = 2; ctx.setLineDash([20,15]);
  ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();
  ctx.setLineDash([]);

  // Kerbs
  ctx.fillStyle='#1f2937'; ctx.fillRect(0,0,W,H*0.25); ctx.fillRect(0,H*0.75,W,H*0.25);
  ctx.fillStyle='#d1fae522'; ctx.fillRect(0,0,W,H*0.03); ctx.fillRect(0,H*0.97,W,H*0.03);

  // Ego vehicle (always at left ~15% of canvas)
  const egoX = W * 0.13, egoY = H/2;
  ctx.save();
  ctx.fillStyle = '#3b82f6';
  ctx.beginPath();
  ctx.roundRect(egoX - 18, egoY - 12, 36, 24, 4);
  ctx.fill();
  ctx.fillStyle = '#93c5fd88';
  ctx.fillRect(egoX - 10, egoY - 10, 14, 8); // windshield
  ctx.fillStyle = '#fbbf24'; // headlights
  ctx.fillRect(egoX + 14, egoY - 8, 4, 5);
  ctx.fillRect(egoX + 14, egoY + 3, 4, 5);
  ctx.restore();

  // Hazard object
  const maxDist = 50.0;
  const hd = Math.min(state.hazard_dist || 999, maxDist);
  if (hd < maxDist) {
    const t = hd / maxDist; // 0=close, 1=far
    const hazX = egoX + 50 + t * (W * 0.72);
    const stageColor = STAGE_COLORS[state.stage] || '#ef4444';
    ctx.save();
    ctx.fillStyle = stageColor;
    ctx.beginPath();
    ctx.arc(hazX, egoY, 14, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const icons = {pedestrian:'🚶',bike:'🏍',auto:'🛺',car:'🚗',truck:'🚚',pothole:'⚠',ambulance:'🚑',animal:'🐄',traffic_police:'👮'};
    const icon = icons[state.hazard_type] || '⚠';
    ctx.font = '14px serif';
    ctx.fillText(icon, hazX, egoY);
    ctx.restore();

    // Distance label
    ctx.fillStyle = stageColor;
    ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(hd.toFixed(0)+'m', hazX, egoY - 22);
  }

  // Stage label
  ctx.fillStyle = STAGE_COLORS[state.stage] || '#94a3b8';
  ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('STAGE: ' + (state.stage||'').toUpperCase(), 8, H - 6);

  // Speed
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'right';
  ctx.fillText((state.speed||0).toFixed(0)+' km/h', W-8, H-6);
}

function drawRewardCurve(canvas, rewards, successes) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  if (!rewards || rewards.length < 2) {
    ctx.fillStyle = '#475569'; ctx.font='13px sans-serif'; ctx.textAlign='center';
    ctx.fillText('Waiting for episode data…', W/2, H/2);
    return;
  }

  const pad = {l:36,r:10,t:12,b:24};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  // Grid
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {
    const y = pad.t + (cH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+cW, y); ctx.stroke();
    const val = (1 - i/4).toFixed(1);
    ctx.fillStyle='#64748b'; ctx.font='9px sans-serif'; ctx.textAlign='right';
    ctx.fillText(val, pad.l-4, y+3);
  }

  const xStep = cW / (rewards.length - 1);
  const toY = v => pad.t + cH * (1 - Math.min(1, Math.max(0, v)));

  // Success background strips
  successes.forEach((s,i) => {
    if (s) {
      ctx.fillStyle = '#22c55e18';
      ctx.fillRect(pad.l + i*xStep - xStep/2, pad.t, xStep, cH);
    }
  });

  // Reward line
  ctx.beginPath(); ctx.strokeStyle='#3b82f6'; ctx.lineWidth=1.5;
  rewards.forEach((r,i) => {
    const x = pad.l + i*xStep, y = toY(r/10); // normalise roughly
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Rolling average line (window 5)
  const roll = rewards.map((_,i) => {
    const w = rewards.slice(Math.max(0,i-4),i+1);
    return w.reduce((a,b)=>a+b,0)/w.length;
  });
  ctx.beginPath(); ctx.strokeStyle='#22c55e'; ctx.lineWidth=2.5;
  roll.forEach((r,i) => {
    const x = pad.l + i*xStep, y = toY(r/10);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Legend
  ctx.font='9px sans-serif'; ctx.textAlign='left';
  ctx.fillStyle='#3b82f6'; ctx.fillText('● Episode reward', pad.l+4, pad.t+10);
  ctx.fillStyle='#22c55e'; ctx.fillText('● Rolling avg', pad.l+110, pad.t+10);
  ctx.fillStyle='#22c55e44'; ctx.fillRect(pad.l+220,pad.t+3,8,8);
  ctx.fillStyle='#22c55e'; ctx.fillText('success', pad.l+230,pad.t+10);
}

const es = new EventSource('/demo/stream');
es.onmessage = function(e) {
  const s = JSON.parse(e.data);
  if (s.waiting) return;

  document.getElementById('waiting').style.display = 'none';
  document.getElementById('main').style.display = '';
  document.getElementById('status-dot').className = 'dot-live';

  document.getElementById('ep-val').textContent = `${s.episode} / ${s.total_episodes}`;
  document.getElementById('step-val').textContent = s.step;
  document.getElementById('scenario-badge').innerHTML = `<span ${badgeCls(s.scenario)}>${s.scenario||'—'}</span>`;
  const sc = STAGE_COLORS[s.stage]||'#94a3b8';
  document.getElementById('stage-badge').innerHTML = `<span style="color:${sc};font-weight:700;">${s.stage||'—'}</span>`;
  document.getElementById('tier-val').textContent = s.tier || '—';
  document.getElementById('diff-val').textContent = (s.difficulty||0).toFixed(2);
  document.getElementById('action-badge').innerHTML = `<span ${badgeCls(s.action)}>${s.action||'—'}</span>`;
  document.getElementById('value-val').textContent = (s.value||0).toFixed(2);
  const rColor = s.reward >= 0.5 ? '#22c55e' : (s.reward >= 0.25 ? '#eab308' : '#ef4444');
  document.getElementById('reward-val').innerHTML = `<span style="color:${rColor}">${(s.reward||0).toFixed(3)}</span>`;
  document.getElementById('roll10-val').textContent = (s.rolling10||0).toFixed(3);
  const srPct = ((s.sr_last20||0)*100).toFixed(0)+'%';
  document.getElementById('sr-val').textContent = srPct;
  document.getElementById('sr-fill').style.width = srPct;
  document.getElementById('hazard-val').textContent = s.hazard_type ? `${s.hazard_type} @ ${(s.hazard_dist||0).toFixed(1)}m` : 'clear';
  document.getElementById('hint-box').textContent = s.hint || '—';
  document.getElementById('mistake-box').textContent = s.last_mistake || '✓ No mistake';
  document.getElementById('alerts-val').textContent = (s.alerts||[]).join('; ') || 'none';

  drawRoad(document.getElementById('road-canvas'), s);
  drawRewardCurve(document.getElementById('reward-canvas'), s.reward_history, s.success_history);
};
es.onerror = function() {
  document.getElementById('status-dot').className = '';
};
</script>
</body>
</html>"""
    return HTMLResponse(html)


def main(host: str = "0.0.0.0", port: int = 8000):
    if not OPENENV_AVAILABLE:
        raise ImportError("OpenEnv is not installed. Install openenv-core to run the AutoDrive server.")

    import uvicorn

    parser = argparse.ArgumentParser(description="AutoDrive Gym OpenEnv server")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--host", default=host)
    args = parser.parse_args()

    # Suppress noisy HTTP access logs — only show rewards, failures, alerts
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep AutoDrive-specific loggers at INFO so rewards/failures still show
    for _ns in ("autodrive", "autodrive_openenv.server", "__main__"):
        logging.getLogger(_ns).setLevel(logging.INFO)
    # Silence uvicorn access log (HTTP request lines)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()