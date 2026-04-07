---
title: autodrive
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# AutoDrive Gym

AutoDrive Gym is an OpenEnv-compatible autonomous driving environment built for Indian road conditions. It evaluates agent behavior on dynamic hazards such as pedestrian crossings, auto-rickshaw cut-ins, bike blind spots, potholes, speed breakers, crowded markets, and sudden mid-episode alerts like ambulances or traffic police overrides.

This project is designed for hackathon submission:
- local Python server support
- Docker packaging
- OpenEnv-compatible API
- evaluation with judge verification and structured reports

## Highlights

- OpenEnv-compatible `reset()` / `step()` interaction loop
- FastAPI server for local and containerized execution
- Hybrid evaluation agent:
  - LLM reasoning
  - memory over recent steps
  - phase-aware decision logic
  - safety and anti-stall guardrails
- Judge verification with final outcome:
  - `RESOLVED`
  - `UNRESOLVED`
- Indian-road scenario coverage:
  - pedestrian crossings
  - auto-rickshaw cut-ins
  - bike blind spots
  - potholes
  - speed breakers
  - crowded markets
  - ambulance approach
  - police override
  - traffic jam
  - animal crossing
  - rain-slippery road
- Mid-episode sudden hazards:
  - a primary scenario can remain active while a secondary surprise hazard appears later in the same episode

## Project Structure

```text
autodrive_env/
|- agent_baseline.py
|- client.py
|- Dockerfile
|- eval.py
|- inference_report.json
|- models.py
|- openenv.yaml
|- openenv_compat.py
|- pyproject.toml
|- README.md
|- train.py
|- __init__.py
`- server/
   |- app.py
   |- autodrive_gym_environment.py
   |- constants.py
   |- curriculum.py
   |- driving_actions.py
   |- driving_backend.py
   |- judge.py
   |- llm_client.py
   |- scenario_generator.py
   `- scenario_injectors.py
```

## OpenEnv Interface

The environment exposes the standard loop:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

The server returns a structured observation containing:
- scene summary
- sensor/object snapshot
- ego vehicle state
- environment state
- hazard type
- hazard status
- hazard distance
- scenario stage
- judge metadata

## Example Scenarios

Primary scenarios:
- `pedestrian_crossing`
- `auto_cut_in`
- `bike_blind_spot`
- `pothole_ahead`
- `speed_breaker`
- `crowded_market`

Sudden mid-episode hazards:
- `ambulance_approach`
- `animal_crossing`
- `police_override`
- `traffic_jam`
- `speed_breaker`

Example intended behavior:
- Episode starts as `pedestrian_crossing`
- Agent brakes and stabilizes
- Sudden ambulance appears from the rear
- Agent must adapt while the episode still remains a `pedestrian_crossing` scenario

## Local Setup

### 1. Create or activate your environment

Use your preferred Python 3.10+ environment.

### 2. Install dependencies

If you use `uv`:

```powershell
uv sync
```

Or with pip:

```powershell
pip install -e .
```

### 3. Start the server

From the `autodrive_env` directory:

```powershell
python -m autodrive_env.server.app
```

Health check:

```powershell
curl http://localhost:8000/healthz
```

## Model Configuration

### Ollama / local OpenAI-compatible API

```powershell
set LLM_BACKEND=openai
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
set LLM_MODEL=llama3
```

### Groq

```powershell
set LLM_PROVIDER=openai
set OPENAI_BASE_URL=https://api.groq.com/openai/v1
set OPENAI_API_KEY=your_groq_key
set OPENAI_MODEL=llama-3.1-8b-instant
```

## Hackathon Inference Contract

For hackathon scoring, use the root-level [inference.py](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/inference.py) script.

It follows the required contract:
- uses the OpenAI client for LLM calls
- reads:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- emits structured stdout in:
  - `[START]`
  - `[STEP]`
  - `[END]`

### Required variables

```powershell
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set HF_TOKEN=your_api_key
```

### Run compliant inference

```powershell
python inference.py --base-url http://localhost:8000 --episodes 6
```

## Evaluation

Run the evaluation script:

```powershell
python eval.py --base-url http://localhost:8000 --episodes 6
```

The evaluation prints:
- per-step actions
- rewards
- decision trace
- sudden alerts when injected mid-episode
- judge verification

It also writes:
- [inference_report.json](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/inference_report.json)

The report includes:
- task trace per episode
- success rate
- average reward
- stage grades
- weakness tracking
- auto-generated notes

## Docker

Build the image from the `autodrive_env` directory:

```powershell
docker build -t autodrive-gym .
```

Run it:

```powershell
docker run --rm -p 8000:8000 autodrive-gym
```

Run with LLM configuration:

### Ollama on host

```powershell
docker run --rm -p 8000:8000 `
  -e LLM_BACKEND=openai `
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 `
  -e LLM_API_KEY=ollama `
  -e LLM_MODEL=llama3 `
  autodrive-gym
```

### Groq

```powershell
docker run --rm -p 8000:8000 `
  -e API_BASE_URL=https://api.groq.com/openai/v1 `
  -e MODEL_NAME=llama-3.1-8b-instant `
  -e HF_TOKEN=your_groq_key `
  autodrive-gym
```

## OpenEnv Packaging

This repo includes:
- [openenv.yaml](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/openenv.yaml)
- a FastAPI runtime
- a Dockerfile

That makes it ready for OpenEnv packaging and hub deployment.

### Push to hub

After authenticating your OpenEnv CLI:

```powershell
openenv push
```

If your workflow uses a specific project or remote target, use the corresponding OpenEnv CLI flags required by your hackathon account.

## Recommended Submission Checklist

- OpenEnv server runs locally
- Docker image builds cleanly
- `/healthz` responds successfully
- `eval.py` produces a fresh `inference_report.json`
- README includes setup, run, eval, and Docker instructions
- `openenv push` is executed from an authenticated CLI session

## Environment Variables

Useful runtime knobs:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LLM_BACKEND`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_PROVIDER`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`

## Current Output Signals

During evaluation you will see:
- `trace: phase=...`
- `sudden alert: ...`
- `guardrail adjusted ...`

Example:

```text
sudden alert: ambulance approaching quickly from behind.
```

This means:
- the episode still belongs to its original scenario
- a secondary hazard was injected mid-episode
- the agent is being evaluated on how it adapts mid-episode
