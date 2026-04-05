import os
import json
import time
import requests
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(".env")

API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL        = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
BASE_URL     = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

EPISODES_EACH = 20
TASKS         = [1, 2, 3]
BENCHMARK     = "pharma_vigil_env"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ── Mandatory log helpers ─────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(task_id: int, obs: dict) -> str:
    if task_id == 1:
        return f"""You are a medical text classifier. Classify this sentence as serious or non_serious.

Sentence: {obs.get('report_text', '')}

RULE: If the sentence describes a drug CAUSING an adverse event or side effect → "serious"
If the sentence does NOT describe a drug causing an adverse event → "non_serious"

Respond ONLY with this JSON:
{{
  "severity": "serious" or "non_serious",
  "causality": 0.9 if serious else 0.1,
  "escalate": true if serious else false,
  "rec_action": "request_followup" if serious else "monitor_only",
  "is_signal": null
}}

No explanation. No markdown. Just the JSON."""

    elif task_id == 2:
        return f"""You are a pharmacovigilance safety reviewer. Triage this adverse drug event report.

Report: {obs.get('report_text', '')}
Drug: {obs.get('drug_name', 'UNKNOWN')}
Symptoms: {obs.get('reported_symptoms', [])}

Severity guide:
- non_serious: mild symptoms (headache, nausea, drowsiness)
- serious: significant symptoms (muscle pain, depression, memory issues, fatigue)
- life_threatening: severe events (breathing difficulty, cardiac issues, severe muscle damage)
- fatal: death

Respond ONLY with this JSON:
{{
  "severity": one of ["non_serious", "serious", "life_threatening", "fatal"],
  "causality": float 0.0-1.0 (how likely drug caused this),
  "escalate": true if serious or worse, false if non_serious,
  "rec_action": one of ["monitor_only", "request_followup", "expedited_review", "signal_team_review", "urgent_regulatory_notification"],
  "is_signal": null
}}

No explanation. No markdown. Just the JSON."""

    else:
        reports = obs.get("reports", [])
        reports_text = "\n\n".join(
            f"Report {i+1}: {r.get('report_text', '')[:300]}"
            for i, r in enumerate(reports)
        )
        return f"""You are a pharmacovigilance signal detection specialist.
Review these 5 patient reports about {obs.get('drug_name', 'LIPITOR')} and decide if they represent a new safety signal.

A TRUE SIGNAL means: all 5 reports describe the SAME specific symptom pattern.
NOISE means: the 5 reports describe DIFFERENT symptoms with no clear common pattern.

{reports_text}

Respond ONLY with this JSON:
{{
  "severity": one of ["non_serious", "serious", "life_threatening", "fatal"],
  "causality": float 0.0-1.0,
  "escalate": true if this IS a safety signal, false if NOT,
  "rec_action": "signal_team_review" if signal, "monitor_only" if not,
  "is_signal": true if safety signal, false if not
}}

No explanation. No markdown. ONE single JSON object only."""


# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":   depth += 1
            elif ch == "}": depth -= 1
            if depth == 0:
                raw = json.loads(text[start:i+1])
                break
        return {
            "severity":   raw.get("severity", "serious"),
            "causality":  max(0.0, min(1.0, float(raw.get("causality", 0.5)))),
            "escalate":   bool(raw.get("escalate", True)),
            "rec_action": raw.get("rec_action", "request_followup"),
            "is_signal":  raw.get("is_signal", None),
        }
    except Exception:
        return {
            "severity":   "serious",
            "causality":  0.5,
            "escalate":   True,
            "rec_action": "request_followup",
            "is_signal":  None,
        }


# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(task_id: int) -> tuple:
    try:
        resp = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        obs = resp.json()

        prompt     = build_prompt(task_id, obs)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=42,
            max_tokens=200,
        )
        raw_text   = completion.choices[0].message.content
        action     = parse_action(raw_text)
        action_str = json.dumps(action, separators=(",", ":"))

        step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()
        return result.get("reward", 0.0), action_str, None

    except Exception as e:
        return 0.0, "null", str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

all_scores = []

for task_id in TASKS:
    task_name = f"task{task_id}"
    rewards   = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL)

    for ep in range(1, EPISODES_EACH + 1):
        reward, action_str, error = run_episode(task_id)
        rewards.append(reward)
        log_step(step=ep, action=action_str, reward=reward, done=True, error=error)
        time.sleep(0.3)

    score   = round(sum(rewards) / len(rewards), 4)
    success = score > 0.0
    log_end(success=success, steps=EPISODES_EACH, score=score, rewards=rewards)
    all_scores.append(score)

overall = round(sum(all_scores) / len(all_scores), 4)
print(f"Task 1 score : {all_scores[0]}", flush=True)
print(f"Task 2 score : {all_scores[1]}", flush=True)
print(f"Task 3 score : {all_scores[2]}", flush=True)
print(f"Overall mean : {overall}",       flush=True)