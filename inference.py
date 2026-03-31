import os
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(".env")

# Config
API_KEY = os.environ.get("HF_TOKEN")
# if not API_KEY:
#     raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

client = OpenAI(
    api_key=os.environ.get("HF_TOKEN"),
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
)
MODEL = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

EPISODES_EACH = 20
# TASKS = ["task1", "task2", "task3"]
TASKS = [1,2,3]

print(f"Model : {MODEL}")
print(f"URL : {BASE_URL}")
print(f"Episodes per task: {EPISODES_EACH}")

# Prompt builder
def build_prompt(task_id, obs: dict) -> str:
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

    else:  # task 3
        reports = obs.get("reports", [])
        reports_text = "\n\n".join(
            f"Report {i+1}: {r.get('report_text', '')[:300]}"
            for i, r in enumerate(reports)
        )
        return f"""You are a pharmacovigilance signal detection specialist.
Review these 5 patient reports about {obs.get('drug_name', 'LIPITOR')} and decide if they represent a new safety signal.

A TRUE SIGNAL means: all 5 reports describe the SAME specific symptom pattern 
(e.g. all mention muscle pain, or all mention memory loss, or all mention liver issues).

NOISE means: the 5 reports describe DIFFERENT symptoms with no clear common pattern,
OR the symptoms are very common/mild (headache, nausea, general pain).

Look for a SPECIFIC repeated pattern across all 5. If symptoms vary widely → false.
If one clear symptom dominates all 5 reports → true.

{reports_text}

Respond ONLY with this JSON:
{{
  "severity": one of ["non_serious", "serious", "life_threatening", "fatal"],
  "causality": float 0.0-1.0,
  "escalate": true if this IS a safety signal, false if NOT,
  "rec_action": "signal_team_review" if signal, "monitor_only" if not,
  "is_signal": true if safety signal, false if not
}}

No explanation. No markdown. ONE single JSON object only — your combined decision across all 5 reports."""

print("Prompt builder defined.")

# Action parser
def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    # extract first JSON object only
    try:
        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    first_json = text[start:i+1]
                    break
        raw = json.loads(first_json)
        return {
            "severity":   raw.get("severity", "serious"),
            "causality":  max(0.0, min(1.0, float(raw.get("causality", 0.5)))),
            "escalate":   bool(raw.get("escalate", True)),
            "rec_action": raw.get("rec_action", "request_followup"),
            "is_signal":  raw.get("is_signal", None),
        }
    except Exception:
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