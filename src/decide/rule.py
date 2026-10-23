"""Pure decision rule. It never reads trajectory labels."""

from __future__ import annotations

FEATURE_KEYS = (
    "success_rate",
    "success_trend",
    "failure_phase_hist",
    "mean_wrist_err",
    "mean_finger_err",
    "mean_tip_err",
    "contact_fraction",
    "slip_fraction",
    "train_steps",
    "edit_evals",
    "intervention_deltas",
)

FORBIDDEN_KEYS = ("label", "class_label", "bias", "meta", "ground_truth", "injected")


def decision_features(evidence: dict) -> dict:
    unknown = set(FORBIDDEN_KEYS).intersection(evidence)
    if unknown:
        raise ValueError(f"decision evidence contains analysis fields: {sorted(unknown)}")
    return {key: evidence[key] for key in FEATURE_KEYS if key in evidence}


def select_action(
    sr0: float,
    sr_train: float,
    train_steps: int,
    sr_edit: float,
    edit_steps: int,
    margin: float = 0.08,
) -> str:
    """Pick train, edit, or defer from two probe outcomes.

    Gain is the success-rate increase per environment step. A probe that
    does not clear `margin` is treated as no evidence for that arm.
    """
    d_train = float(sr_train) - float(sr0)
    d_edit = float(sr_edit) - float(sr0)
    g_train = d_train / max(int(train_steps), 1)
    g_edit = d_edit / max(int(edit_steps), 1)
    if sr0 >= 0.8 and d_train < margin and d_edit < margin:
        return "defer"
    if d_train < margin and d_edit < margin:
        return "defer"
    if d_edit >= margin and g_edit > g_train:
        return "edit"
    if d_train >= margin:
        return "train"
    if d_edit >= margin:
        return "edit"
    return "defer"
