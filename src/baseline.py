import os
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("../.env")

# Config
API_KEY = os.environ.get("OPENAI_API_KEY")
# if not API_KEY:
#     raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
EPISODES_EACH = 20
# TASKS = ["task1", "task2", "task3"]
TASKS = [1, 2, 3]

print(f"Model : {MODEL}")
print(f"URL : {BASE_URL}")
print(f"Episodes per task: {EPISODES_EACH}")

# Prompt builder
def build_prompt(task_id: str, obs: dict) -> str:
    # if task_id in ["task1", "task2"]:
    if task_id in [1, 2]:
        return f"""You are a pharmacovigilance safety reviewer. Read the adverse drug event report and triage it.

Report: {obs.get('report_text', '')}
Drug: {obs.get('drug_name', 'UNKNOWN')}
Reported symptoms: {obs.get('reported_symptoms', [])}

Respond ONLY with a JSON object with exactly these fields:
{{
  "severity": one of ["non_serious", "serious", "life_threatening", "fatal"],
  "causality": float between 0.0 and 1.0,
  "escalate": true or false,
  "rec_action": one of ["monitor_only", "request_followup", "expedited_review", "signal_team_review", "urgent_regulatory_notification"],
  "is_signal": null
}}

No explanation. No markdown. Just the JSON object."""

    else:  # task3
        reports = obs.get("reports", [])
        reports_text = "\n\n".join(
            f"Report {i+1}: {r.get('report_text', '')[:300]}"
            for i, r in enumerate(reports)
        )
        return f"""You are a pharmacovigilance signal detection specialist.
Review these 5 related adverse drug event reports and decide if they represent a new safety signal.

Drug: {obs.get('drug_name', 'LIPITOR')}

{reports_text}

Respond ONLY with a JSON object with exactly these fields:
{{
  "severity": one of ["non_serious", "serious", "life_threatening", "fatal"],
  "causality": float between 0.0 and 1.0,
  "escalate": true or false,
  "rec_action": one of ["monitor_only", "request_followup", "expedited_review", "signal_team_review", "urgent_regulatory_notification"],
  "is_signal": true or false
}}

No explanation. No markdown. Just the JSON object."""

print("Prompt builder defined.")

# Action parser
def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        raw = json.loads(text)
        return {
            "severity":  raw.get("severity", "serious"),
            "causality": max(0.0, min(1.0, float(raw.get("causality", 0.5)))),
            "escalate":  bool(raw.get("escalate", True)),
            "rec_action": raw.get("rec_action", "request_followup"),
            "is_signal": raw.get("is_signal", None),
        }
    except Exception:
        # safe fallback
        return {
            "severity":  "serious",
            "causality": 0.5,
            "escalate":  True,
            "rec_action": "request_followup",
            "is_signal": None,
        }

print("Parser defined.")

# Single episode runner
def run_episode(task_id: str) -> float:
    # reset
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    # call LLM
    prompt = build_prompt(task_id, obs)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=42,
        max_tokens=200,
    )
    raw = completion.choices[0].message.content
    action = parse_action(raw)

    # step
    step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
    step_resp.raise_for_status()
    result = step_resp.json()
    return result.get("reward", 0.0)

print("Episode runner defined.")





# Run baseline
# Make sure ENV_BASE_URL is set to live HF Space URL before running this cell
# e.g. os.environ['ENV_BASE_URL'] = 'https://your-hf-space.hf.space'

all_scores = []

for task_id in TASKS:
    scores = []
    print(f"\nRunning {task_id}...")
    for ep in range(EPISODES_EACH):
        try:
            reward = run_episode(task_id)
            scores.append(reward)
            print(f"  ep{ep+1:02d}: {reward:.4f}")
        except Exception as e:
            print(f"  ep{ep+1:02d}: ERROR — {e}")
            scores.append(0.0)
        time.sleep(0.3)

    mean = round(sum(scores) / len(scores), 4)
    all_scores.append(mean)
    print(f"  {task_id} mean: {mean}")

overall = round(sum(all_scores) / len(all_scores), 4)
print(f"Task 1 score : {all_scores[0]}")
print(f"Task 2 score : {all_scores[1]}")
print(f"Task 3 score : {all_scores[2]}")
print(f"Overall mean : {overall}")