from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
from models import Action, EnvState, GroundTruth, Observation, SingleReport, StepInfo
from server.graders import grade_task1, grade_task2, grade_task3

DATA_DIR = Path(__file__).parent.parent / "data"

TASK_FILES = {
    1: DATA_DIR / "task1_dataset.json",
    2: DATA_DIR / "task2_dataset.json",
    3: DATA_DIR / "task3_dataset.json"
}

VALID_SEVERITY = {"serious", "non_serious", "life_threatening", "fatal"}
VALID_ACTIONS = {"monitor_only", "request_followup", "expedited_review", "signal_team_review", "urgent_regulatory_notification"}

class ADEEnvironment:
    def __init__(self)->None:
        self._datasets: dict[int, list[dict]]={}
        self._load_datasets()

        self._task_id: int=1
        self._episode_id: str =""
        self._step: int=0
        self._done: bool=True
        self._current_obs: Optional[Observation]=None
        self._ground_truth: Optional[GroundTruth]=None
        self._last_reward: Optional[float]=None
        self._total_reward: float=0.0

    def _load_datasets(self)->None:
        for task_id, path in TASK_FILES.items():
            with open(path, "r", encoding="utf-8") as f:
                self._datasets[task_id] = json.load(f)

    def reset(self, task_id: int = 1, seed: Optional[int] = None) -> Observation:
        assert task_id in (1, 2, 3), f"task_id must be 1, 2 or 3 — got {task_id}"

        if seed is not None:
            random.seed(seed)

        self._task_id     = task_id
        self._step        = 0
        self._done        = False
        self._last_reward = None
        self._total_reward = 0.0

        sample = random.choice(self._datasets[task_id])
        self._episode_id  = sample["episode_id"]

        self._current_obs  = self._build_observation(sample, task_id)
        self._ground_truth = self._build_ground_truth(sample, task_id)

        return self._current_obs

    def step(self, action: Action) -> tuple[Optional[Observation], float, bool, dict]:
        assert not self._done, "Episode is done — call reset() first"
        assert self._current_obs is not None

        if action.severity not in VALID_SEVERITY:
            raise ValueError(f"Invalid severity '{action.severity}'. Must be one of {VALID_SEVERITY}")
        if action.rec_action not in VALID_ACTIONS:
            raise ValueError(f"Invalid rec_action '{action.rec_action}'. Must be one of {VALID_ACTIONS}")

        raw_score = self._call_grader(action)
        reward, penalties = self._compute_reward(raw_score, action)

        self._step        += 1
        self._last_reward  = reward
        self._total_reward += reward
        self._done         = True   

        info = StepInfo(
            episode_id       = self._episode_id,
            task_id          = self._task_id,
            raw_score        = raw_score,
            penalties_applied= penalties,
            final_reward     = reward,
        ).model_dump()

        return None, reward, self._done, info

    def state(self) -> EnvState:
        assert self._current_obs is not None, "Call reset() before state()"
        return EnvState(
            episode_id   = self._episode_id,
            task_id      = self._task_id,
            step         = self._step,
            done         = self._done,
            current_obs  = self._current_obs,
            last_reward  = self._last_reward,
            total_reward = self._total_reward,
        )

    def _call_grader(self, action: Action) -> float:
        gt = self._ground_truth
        if self._task_id == 1:
            return grade_task1(action, gt)
        elif self._task_id == 2:
            return grade_task2(action, gt)
        elif self._task_id == 3:
            return grade_task3(action, gt)
        return 0.0

    def _compute_reward(self, raw_score: float, action: Action) -> tuple[float, list[str]]:
        reward    = raw_score
        penalties = []
        gt        = self._ground_truth

        if (gt.severity in ("serious", "life_threatening")
                and action.severity == "non_serious"):
            reward -= 0.30
            penalties.append("missed_serious_case")

        if action.escalate and not gt.escalate:
            reward -= 0.05
            penalties.append("false_escalation")

        if abs(action.causality - gt.causality) > 0.4:
            reward -= 0.10
            penalties.append("causality_mismatch")

        if self._task_id == 3 and action.is_signal is not None:
            if action.is_signal != gt.is_signal:
                reward -= 0.20
                penalties.append("signal_detection_error")

        reward = round(max(0.01, min(0.99, reward)), 4)
        return reward, penalties

    def _build_observation(self, sample: dict, task_id: int) -> Observation:
        if task_id == 3:
            return Observation(
                episode_id        = sample["episode_id"],
                task_id           = task_id,
                drug_name         = sample["drug_name"],
                reports           = [
                    SingleReport(
                        report_text       = r["report_text"],
                        reported_symptoms = r.get("reported_symptoms", []),
                        drug_name         = r.get("drug_name", sample["drug_name"]),
                    )
                    for r in sample["reports"]
                ],
            )
        return Observation(
            episode_id        = sample["episode_id"],
            task_id           = task_id,
            report_text       = sample["report_text"],
            drug_name         = sample.get("drug_name", "UNKNOWN"),
            patient_age       = sample.get("patient_age"),
            reported_symptoms = sample.get("reported_symptoms", []),
            prior_actions_taken = sample.get("prior_actions_taken", []),
        )

    def _build_ground_truth(self, sample: dict, task_id: int) -> GroundTruth:
        if task_id == 3:
                return GroundTruth(
                    severity   = "serious",  
                    causality  = 0.8,        
                    escalate   = sample["gt_escalate"],
                    rec_action = sample["gt_recommended_action"],
                    is_signal  = sample.get("is_signal", False),
                )
        return GroundTruth(
            severity           = sample["gt_severity"],
            causality          = float(sample["gt_causality"]),
            escalate           = sample["gt_escalate"],
            rec_action = sample["gt_recommended_action"],
            is_signal          = sample.get("is_signal") if task_id == 3 else None,
        )
