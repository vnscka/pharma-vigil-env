from __future__ import annotations
from models import Action, GroundTruth

SEVERITY_RANK = {
    "non_serious":      0,
    "serious":          1,
    "life_threatening": 2,
}
ACTION_RANK = {
    "close":            0,
    "monitor_only":     1,
    "request_followup": 2,
    "expedited_review": 3,
}

def _clamp(score: float) -> float:
    return round(max(0.01, min(0.99, score)), 4)

def grade_task1(action: Action, gt: GroundTruth) -> float:
    return _clamp(1.0 if action.severity == gt.severity else 0.0)

def grade_task2(action: Action, gt: GroundTruth) -> float:
    sev_pred = SEVERITY_RANK.get(action.severity, 0)
    sev_gt   = SEVERITY_RANK.get(gt.severity, 0)
    sev_diff = abs(sev_pred - sev_gt)
    if sev_diff == 0:
        sev_score = 1.0
    elif sev_diff == 1:
        sev_score = 0.5
    else:
        sev_score = 0.0
    caus_score = max(0.0, 1.0 - abs(action.causality - gt.causality))
    esc_score = 1.0 if action.escalate == gt.escalate else 0.0
    act_pred = ACTION_RANK.get(action.recommended_action, 0)
    act_gt   = ACTION_RANK.get(gt.recommended_action, 0)
    act_diff = abs(act_pred - act_gt)
    if act_diff == 0:
        act_score = 1.0
    elif act_diff == 1:
        act_score = 0.5
    else:
        act_score = 0.0
    return _clamp(round(
        0.4 * sev_score +
        0.3 * caus_score +
        0.2 * esc_score +
        0.1 * act_score,
        4
    ))

def grade_task3(action: Action, gt: GroundTruth) -> float:
    if action.is_signal is None:
        return _clamp(0.0)
    if action.is_signal == gt.is_signal:
        f1 = 1.0
    else:
        f1 = 0.0
    act_bonus = 0.0
    if action.recommended_action == gt.recommended_action:
        act_bonus = 0.15
    return _clamp(round(min(1.0, f1 * 0.85 + act_bonus), 4))