"""
models.py — Shared data models for pharma-vigil-env
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class SingleReport(BaseModel):
    report_text: str
    reported_symptoms: list[str]
    drug_name: str

class Observation(BaseModel):
    episode_id: str
    task_id: int
    report_text: Optional[str] = None
    reports: Optional[list[SingleReport]] = None
    drug_name: str
    patient_age: Optional[int] = None
    reported_symptoms: list[str] = []
    prior_actions_taken: list[str] = []

class Action (BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    severity: str = Field(..., description="One of: serious | non_serious | life_threatening | fatal")
    causality: float = Field(..., ge=0.0, le=1.0, description="Causality confidence score 0,0 to 1.0")
    escalate: bool
    rec_action: str = Field(..., alias="rec_action", description="One of: monitor_only | request_followup | expedited_review | signal_team_review | urgent_regulatory_notification")
    is_signal : bool | None = None
    # Property alias for graders
    @property
    def recommended_action(self) -> str:
        return self.rec_action

class GroundTruth(BaseModel):
    severity: str
    causality: float
    escalate: bool
    rec_action: str
    is_signal: Optional[bool] = None
    # Property alias for graders
    @property
    def recommended_action(self) -> str:
        return self.rec_action

class StepInfo(BaseModel):
    episode_id: str
    task_id: int
    raw_score: float
    penalties_applied: list[str]
    final_reward: float

class EnvState(BaseModel):
    episode_id: str
    task_id: int
    step: int
    done: bool
    current_obs: Observation
    last_reward: Optional[float] = None
    total_reward: float = 0.0
