from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
 
import sys
from pathlib import Path

import uvicorn
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from models import Action, Observation, EnvState
from server.ade_environment import ADEEnvironment

app = FastAPI(title="pharma_vigil_env", version="0.1.0")

env = ADEEnvironment()
 
_baseline_cache: Optional[dict] = None


class ResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None
 
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
    obs = env.reset(task_id=req.task_id)
    return obs
 
 
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
            "description": (
                "Binary ADE classification. Agent reads a sentence from medical "
                "literature and predicts severity label."
            ),
            "action_schema": {
                "severity": ["serious", "non_serious", "life_threatening", "fatal"],
                "causality": "float 0.0-1.0",
                "escalate": "bool",
                "rec_action": [
                    "monitor_only",
                    "request_followup",
                    "expedited_review",
                    "signal_team_review",
                    "urgent_regulatory_notification",
                ],
                "is_signal": "bool or null",
            },
        },
        {
            "id": "task2",
            "name": "Full triage",
            "difficulty": "medium",
            "description": (
                "Full pharmacovigilance triage on noisy CADEC patient forum posts. "
                "Agent predicts all 4 Action fields."
            ),
            "action_schema": {
                "severity": ["serious", "non_serious", "life_threatening", "fatal"],
                "causality": "float 0.0-1.0",
                "escalate": "bool",
                "rec_action": [
                    "monitor_only",
                    "request_followup",
                    "expedited_review",
                    "signal_team_review",
                    "urgent_regulatory_notification",
                ],
                "is_signal": "bool or null",
            },
        },
        {
            "id": "task3",
            "name": "Signal detection",
            "difficulty": "hard",
            "description": (
                "Agent receives a cluster of 5 related CADEC Lipitor reports and "
                "must decide if they constitute a new safety signal."
            ),
            "action_schema": {
                "severity": ["serious", "non_serious", "life_threatening", "fatal"],
                "causality": "float 0.0-1.0",
                "escalate": "bool",
                "rec_action": [
                    "monitor_only",
                    "request_followup",
                    "expedited_review",
                    "signal_team_review",
                    "urgent_regulatory_notification",
                ],
                "is_signal": "bool or null — required for task3",
            },
        },
    ]
 
 
@app.post("/grader")
def grader(req: GraderRequest):
    if req.task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
 
    # re-run the episode with the given episode_id to get ground truth
    dataset = env._datasets.get(req.task_id, [])
    sample = next((s for s in dataset if s["episode_id"] == req.episode_id), None)
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"episode_id '{req.episode_id}' not found in task{req.task_id} dataset",
        )
 
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
        "task_id": req.task_id,
        "episode_id": req.episode_id,
        "subscores": {
            "raw_score": raw,
            "penalties": info["penalties_applied"],
            "final_reward": reward,
        },
    }
 
 
@app.get("/baseline")
def baseline():
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache
 
    openai_key = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN not set — cannot run baseline",
        )
 
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/app",
            env={**os.environ, "HF_TOKEN": openai_key, "OPENAI_API_KEY": openai_key},
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="inference.py timed out after 10 min")
 
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"inference.py failed: {result.stderr[:500]}",
        )
 
    scores = _parse_baseline_output(result.stdout)
    _baseline_cache = scores
    return scores
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _apply_reward_shaping(action: Action, gt, raw_score: float, task_id: int):
    """Mirrors ADEEnvironment._compute_reward for use in /grader endpoint."""
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
            penalties.append("signal_detection_error")
 
    reward = round(max(0.0, min(1.0, reward)), 4)
    info = {"penalties_applied": penalties, "final_reward": reward}
    return None, reward, True, info
 
 
def _parse_baseline_output(stdout: str) -> dict:
    """Parse printed scores from inference.py output."""
    scores = {"task1_score": None, "task2_score": None, "task3_score": None, "overall_mean": None}
    for line in stdout.splitlines():
        line = line.strip().lower()
        if "task1" in line or "task 1" in line:
            scores["task1_score"] = _extract_float(line)
        elif "task2" in line or "task 2" in line:
            scores["task2_score"] = _extract_float(line)
        elif "task3" in line or "task 3" in line:
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

@app.get("/schema")
def schema():
    return {
        "action": {
            "severity": ["serious", "non_serious", "life_threatening", "fatal"],
            "causality": "float 0.0-1.0",
            "escalate": "bool",
            "rec_action": ["monitor_only", "request_followup", "expedited_review", "signal_team_review", "urgent_regulatory_notification"],
            "is_signal": "bool or null"
        },
        "observation": {
            "episode_id": "str",
            "task_id": "int",
            "report_text": "str",
            "drug_name": "str",
            "reported_symptoms": "list[str]"
        },
        "state": {
            "episode_id": "str",
            "task_id": "int",
            "step": "int",
            "done": "bool"
        }
    }

@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "id": 1, "result": {}}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()