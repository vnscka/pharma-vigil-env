from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
 
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
 
from models import Action, Observation, EnvState
from server.ade_environment import ADEEnvironment

app = FastAPI(title="pharma_vigil_env", version="0.1.0")

env = ADEEnvironment()
 
_baseline_cache: Optional[dict] = None


class ResetRequest(BaseModel):
    task_id: int
 
class GraderRequest(BaseModel):
    episode_id: str
    task_id: int
    action: Action


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: ResetRequest) -> Observation:
    if req.task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
    return env.reset(task_id=req.task_id)
 
 
@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }
 
 
@app.get("/state")
def state() -> EnvState:
    try:
        return env.state()
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
 
@app.get("/tasks")
def tasks():
    return [
        {
            "id": "task1",
            "name": "Severity classification",
            "difficulty": "easy",
        },
        {
            "id": "task2",
            "name": "Full triage",
            "difficulty": "medium",
        },
        {
            "id": "task3",
            "name": "Signal detection",
            "difficulty": "hard",
        },
    ]
 
 
@app.post("/grader")
def grader(req: GraderRequest):
    if req.task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
 
    dataset = env._datasets.get(req.task_id, [])
    sample = next((s for s in dataset if s["episode_id"] == req.episode_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="episode not found")
 
    from models import GroundTruth
    gt = env._build_ground_truth(sample, req.task_id)
 
    if req.task_id == 1:
        from server.graders import grade_task1
        raw = grade_task1(req.action, gt)
    elif req.task_id == 2:
        from server.graders import grade_task2
        raw = grade_task2(req.action, gt)
    else:
        from server.graders import grade_task3
        raw = grade_task3(req.action, gt)
 
    _, reward, _, info = _apply_reward_shaping(req.action, gt, raw, req.task_id)
 
    return {
        "total_score": reward,
        "subscores": info,
    }
 
 
@app.get("/baseline")
def baseline():
    global _baseline_cache

    if _baseline_cache is not None:
        return _baseline_cache

    print("BASELINE ENDPOINT HIT", flush=True)

    openai_key = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise HTTPException(status_code=503, detail="HF_TOKEN not set")

    try:
        process = subprocess.Popen(
            [sys.executable, "inference.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/app",
            env={**os.environ, "HF_TOKEN": openai_key, "OPENAI_API_KEY": openai_key},
        )

        output_lines = []

        for line in process.stdout:
            print(line, end="", flush=True)  # ✅ LIVE LOGS
            output_lines.append(line)

        process.wait()

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="inference timeout")

    if process.returncode != 0:
        raise HTTPException(status_code=500, detail="inference failed (check logs)")

    stdout = "".join(output_lines)
    scores = _parse_baseline_output(stdout)
    _baseline_cache = scores

    return scores
 
 
# ── Helpers ─────────────────────────────────────────

def _apply_reward_shaping(action: Action, gt, raw_score: float, task_id: int):
    reward = raw_score
    penalties = []

    if gt.severity in ("serious", "life_threatening") and action.severity == "non_serious":
        reward -= 0.30
        penalties.append("missed_serious_case")

    if action.escalate and not gt.escalate:
        reward -= 0.05
        penalties.append("false_escalation")

    if abs(action.causality - gt.causality) > 0.4:
        reward -= 0.10
        penalties.append("causality_mismatch")

    if task_id == 3 and action.is_signal is not None:
        if action.is_signal != gt.is_signal:
            reward -= 0.20
            penalties.append("signal_error")

    reward = max(0.0, min(1.0, reward))

    return None, reward, True, {"penalties": penalties}


def _parse_baseline_output(stdout: str) -> dict:
    scores = {"task1_score": None, "task2_score": None, "task3_score": None, "overall_mean": None}
    for line in stdout.splitlines():
        line = line.strip().lower()
        if "task1" in line:
            scores["task1_score"] = _extract_float(line)
        elif "task2" in line:
            scores["task2_score"] = _extract_float(line)
        elif "task3" in line:
            scores["task3_score"] = _extract_float(line)
        elif "overall" in line or "mean" in line:
            scores["overall_mean"] = _extract_float(line)
    return scores


def _extract_float(line: str) -> Optional[float]:
    import re
    match = re.search(r"(\d+\.\d+)", line)
    return float(match.group(1)) if match else None


@app.get("/metadata")
def metadata():
    return {
        "name": "pharmacovigilance-triage",
        "description": "AI evaluation environment for adverse drug event triage."
    }


@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "id": 1, "result": {}}


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()