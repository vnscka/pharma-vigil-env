# pharma_vigil_env

**Pharmacovigilance ADE Triage Environment** — an OpenEnv environment where an AI agent triages adverse drug event (ADE) reports like a real FDA safety reviewer.

## Motivation

Regulatory agencies like the FDA process hundreds of thousands of adverse drug event reports annually. Manual triage is a bottleneck — reviewers must classify severity, estimate causality, decide escalation, and recommend follow-up action for each report. This environment simulates that real-world triage workflow, enabling AI agents to learn and be evaluated on this safety-critical task.

This domain has never appeared in OpenEnv before. Ground truth labels are derived from MedDRA codes — the same professional standard used by the FDA — making graders medically defensible.

---

## Environment Overview

| Property     | Value                           |
|--------------|---------------------------------|
| Domain       | Pharmacovigilance / Drug Safety |
| Tasks        | 3 (easy → medium → hard)        |
| Episode type | Single-turn                     |
| Reward range | [0.0, 1.0]                      |
| Datasets     | ADE Corpus v2 + CADEC v2        |
| SDK          | Docker                          |
---

## Action Space

The agent must output an `Action` for each episode:

| Field | Type | Values |
|---|---|---|
| `severity` | `str` | `non_serious`, `serious`, `life_threatening`, `fatal` |
| `causality` | `float` | 0.0 to 1.0 — how likely the drug caused the event |
| `escalate` | `bool` | Whether to escalate this report for review |
| `rec_action` | `str` | `monitor_only`, `request_followup`, `expedited_review`, `signal_team_review`, `urgent_regulatory_notification` |
| `is_signal` | `bool \| null` | Task 3 only — whether 5 reports constitute a safety signal |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str` | Unique episode identifier |
| `task_id` | `int` | Task number (1, 2, or 3) |
| `report_text` | `str \| null` | Free-text ADE report (Tasks 1 & 2) |
| `reports` | `list \| null` | List of 5 related reports (Task 3 only) |
| `drug_name` | `str` | Name of the drug |
| `patient_age` | `int \| null` | Patient age if available |
| `reported_symptoms` | `list[str]` | Extracted symptom list |
| `prior_actions_taken` | `list[str]` | Previous actions taken |

---

## Tasks

### Task 1 — Severity Classification (Easy)
- **Dataset:** ADE Corpus v2 — 150 balanced medical literature sentences
- **Agent goal:** Classify severity as `serious` or `non_serious`
- **Grader:** Exact match — 1.0 if correct, 0.0 if wrong
- **Why easy:** Clean formal text, binary classification, minimal ambiguity

### Task 2 — Full Triage (Medium)
- **Dataset:** CADEC v2 — 100 real patient forum posts with 3+ ADRs
- **Agent goal:** Predict all 4 action fields
- **Grader (weighted partial credit):**
  - Severity correctness: 0.4
  - Causality closeness: 0.3 (= 1 − |pred − gt|)
  - Escalation correctness: 0.2
  - Recommended action correctness: 0.1
- **Why medium:** Noisy informal patient text, confounders, polypharmacy

### Task 3 — Signal Detection (Hard)
- **Dataset:** 30 clusters of 5 CADEC Lipitor posts grouped by MedDRA codes
- **Agent goal:** Decide if 5 related reports constitute a new drug safety signal
- **Grader:** F1 on escalate/is_signal + 0.1 bonus for correct recommended action, clamped to [0, 1]
- **Why hard:** Requires reasoning across 5 documents simultaneously, distinguishing genuine patterns from noise

---

## Reward Function

Applied inside `step()` on top of the base grader score:

```python
# Asymmetric safety penalty — missing real ADE is worse than over-escalating
if gt.severity in ("serious", "life_threatening") and action.severity == "non_serious":
    reward -= 0.30

if action.escalate and not gt.escalate:
    reward -= 0.05

if abs(action.causality - gt.causality) > 0.4:
    reward -= 0.10

if task_id == 3 and action.is_signal != gt.is_signal:
    reward -= 0.20

reward = round(max(0.0, min(1.0, reward)), 4)
```

The asymmetric penalty (−0.30 for missing a serious case vs −0.05 for false positive) encodes the real-world principle that missing a genuine ADE is more dangerous than over-escalating.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns 200 |
| `/reset` | POST | Start episode, returns Observation |
| `/step` | POST | Accept Action, returns reward/obs/done/info |
| `/state` | GET | Returns current episode State |
| `/tasks` | GET | Task list + Action schema |
| `/grader` | POST | Score a completed episode with subscore breakdown |
| `/baseline` | GET | Run inference script, return scores |

---

## Baseline Scores

Evaluated using `llama-3.3-70b-versatile` via Groq (OpenAI-compatible), 20 episodes per task:

| Task | Score      |
|---|------------|
| Task 1 — Severity classification | 1.00       |
| Task 2 — Full triage | 0.878      |
| Task 3 — Signal detection | 0.385      |
| **Overall mean** | **0.7543** |

---

## Setup & Usage

### Requirements

```bash
pip install openenv-core fastapi uvicorn pydantic
```

### Run locally

```bash
git clone https://github.com/vnscka/pharma-vigil-env
cd pharma-vigil-env
docker build -t pharma-vigil-env .
docker run -p 8000:8000 pharma-vigil-env
```

### Run inference script

Set the required environment variables:

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_groq_api_key
```

Then run:

```bash
python inference.py
```

### Live HF Space

```
https://ishantk059-pharma-vigil-env.hf.space
```

---

## Project Structure

```
pharma_vigil_env/
├── data/
│   ├── task1_dataset.json
│   ├── task2_dataset.json
│   └── task3_dataset.json 
│
├── notebooks/
│   ├── build_datasets.ipynb
│   ├── explore_datasets.ipynb
│    
├── server/
|   ├── ade_environment.py    # Environment logic
│   ├── app.py                # FastAPI endpoints
|   ├── grader.py              # Grader logic for scoring actions
│   └── requirements.txt
│
├── inference.py              # Baseline inference script
├── models.py             # Pydantic models
├── openenv.yaml              # OpenEnv metadata
└── Dockerfile
```

---

## Datasets

- **ADE Corpus v2** — 23,516 medical literature sentences, binary ADE labels (Task 1)
- **CADEC v2** — 1,250 real patient forum posts with MedDRA annotations (Tasks 2 & 3)

Ground truth severity labels are derived from MedDRA codes using a clinical lookup table based on standard MedDRA SOC hierarchy — the same coding system used by the FDA for pharmacovigilance.

---

## Tags

`openenv` `pharmacovigilance` `healthcare` `safety` `FDA` `adverse-drug-events`
