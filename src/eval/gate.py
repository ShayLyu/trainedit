"""Did train-versus-edit benefits actually separate?"""

from __future__ import annotations

import numpy as np


def _mean(records: list[dict], label: str, key: str) -> float:
    vals = [r[key] for r in records if r["label"] == label]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def assess_gate(records: list[dict]) -> dict:
    """`records` items have label, success_train, success_edit.

    The gate passes only when easy references are already solvable, hard
    references prefer more training, and biased references prefer an edit.
    """
    easy_train = _mean(records, "easy", "success_train")
    hard_train = _mean(records, "hard", "success_train")
    hard_edit = _mean(records, "hard", "success_edit")
    bias_labels = ("bias_trans", "bias_rot", "bias_closure")
    bias_train = float(np.nanmean([_mean(records, lab, "success_train") for lab in bias_labels]))
    bias_edit = float(np.nanmean([_mean(records, lab, "success_edit") for lab in bias_labels]))
    per_label = {}
    for lab in ("easy", "hard", *bias_labels):
        per_label[lab] = {
            "train": _mean(records, lab, "success_train"),
            "edit": _mean(records, lab, "success_edit"),
        }
    easy_ok = easy_train >= 0.6
    # A tie is allowed. Rejecting every edit leaves both arms as training, so
    # hard does not have to beat that arm. Editing has to be no better.
    hard_train_ok = hard_train >= 0.4
    hard_prefers_train = hard_train_ok and hard_edit <= hard_train + 0.05
    bias_prefers_edit = bias_edit >= bias_train + 0.2
    train_fixes_bias = bias_train >= 0.7
    edit_always = all(
        per_label[lab]["edit"] > per_label[lab]["train"] + 0.05
        for lab in ("easy", "hard", *bias_labels)
        if not np.isnan(per_label[lab]["edit"])
    )
    proceed = bool(easy_ok and hard_prefers_train and bias_prefers_edit and not train_fixes_bias and not edit_always)
    if proceed:
        reason = "差异化收益成立：困难参考更受益于训练，偏差参考更受益于编辑。"
    elif train_fixes_bias:
        reason = "继续训练已经克服注入偏差，调度机制的价值有限。"
    elif edit_always or (hard_train_ok and hard_edit > hard_train + 0.05 and bias_prefers_edit):
        reason = "编辑始终不差于训练，应转向改进编辑器而不是分配预算。"
    elif not hard_train_ok:
        reason = "困难参考继续训练没有达到可用成功率。"
    else:
        reason = "看门实验没有形成稳定的差异化收益。"
    return {
        "proceed": proceed,
        "reason": reason,
        "easy_train": easy_train,
        "hard_train": hard_train,
        "hard_edit": hard_edit,
        "bias_train": bias_train,
        "bias_edit": bias_edit,
        "per_label": per_label,
    }


def summarize_traces(traces: list) -> dict:
    labels = sorted({t.label for t in traces})
    by_label = {}
    for lab in labels:
        group = [t for t in traces if t.label == lab]
        by_label[lab] = {
            "n": len(group),
            "success": float(np.mean([t.final_success for t in group])),
            "steps": float(np.mean([t.steps_spent for t in group])),
            "gen_success": float(np.mean([t.gen_success for t in group])),
        }
    n = len(traces)
    return {
        "n": n,
        "success": float(np.mean([t.final_success for t in traces])) if n else 0.0,
        "steps": float(np.mean([t.steps_spent for t in traces])) if n else 0.0,
        "gen_success": float(np.mean([t.gen_success for t in traces])) if n else 0.0,
        "edited_fraction": float(np.mean([t.edited for t in traces])) if n else 0.0,
        "deferred_fraction": float(np.mean([t.deferred for t in traces])) if n else 0.0,
        "kept_fraction": float(np.mean([(not t.edited) and (not t.deferred) for t in traces])) if n else 0.0,
        "by_label": by_label,
    }


def hard_clean_mistakes(adaptive: list, train_only: list) -> dict:
    train_by_id = {t.traj_id: t for t in train_only}
    checked = 0
    mistakes = 0
    details = []
    for trace in adaptive:
        if trace.label != "hard":
            continue
        checked += 1
        baseline = train_by_id.get(trace.traj_id)
        abandoned = trace.deferred or trace.edited
        worse = baseline is not None and trace.final_success + 0.05 < baseline.final_success
        if abandoned and worse:
            mistakes += 1
            details.append(trace.traj_id)
    return {
        "n_hard": checked,
        "mistaken_fraction": (mistakes / checked) if checked else 0.0,
        "ids": details,
    }
