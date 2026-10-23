"""Build the three v1 reference classes from one nominal flanged lift."""

from __future__ import annotations

import numpy as np

from src.config import (
    BIAS_CLOSURE_SCALE,
    BIAS_TRANS_M,
    BIAS_PITCH_RAD,
    HARD_SLIDE_M,
    HARD_SLIDE_STEPS,
    CONTROL_DT,
    HOVER_OFFSET,
    LIFT_HEIGHT,
    N_APPROACH,
    N_CLOSE,
    N_HOLD,
    N_LIFT,
    OBJECT_POS,
)
from src.env.grasp_solve import nominal_keyframes
from src.env.model import HandModel
from src.refs.edit import apply_edit, finalize
from src.refs.schema import EditParams, Reference


def _phase_ids() -> tuple[np.ndarray, int]:
    parts = (
        np.full(N_APPROACH, 0, dtype=np.int64),
        np.full(N_CLOSE, 1, dtype=np.int64),
        np.full(N_LIFT, 2, dtype=np.int64),
        np.full(N_HOLD, 3, dtype=np.int64),
    )
    phase = np.concatenate(parts)
    return phase, N_APPROACH + N_CLOSE


def make_clean_reference(ref_id: str, label: str, profile: str, hand: HandModel) -> Reference:
    keys = nominal_keyframes(hand)
    grasp = keys["wrist_pos"]
    quat = keys["wrist_quat"]
    open_q = keys["finger_open"]
    closed = keys["finger_closed"]
    hover = grasp + np.asarray(HOVER_OFFSET, dtype=np.float64)
    phase, t_lift = _phase_ids()
    horizon = phase.shape[0]
    pos = np.zeros((horizon, 3))
    quats = np.zeros((horizon, 4))
    finger = np.zeros((horizon, 16))
    alphas = np.linspace(0.0, 1.0, N_APPROACH, endpoint=False)
    cursor = 0
    for a in alphas:
        pos[cursor] = hover * (1.0 - a) + grasp * a
        quats[cursor] = quat
        finger[cursor] = open_q
        cursor += 1
    if profile == "rush":
        # Finish the approach outside the cap. The slide back happens after close is filled.
        for i in range(N_APPROACH):
            pos[i, 0] += HARD_SLIDE_M * min(1.0, (i + 1) / max(N_APPROACH - HARD_SLIDE_STEPS, 1))
    for a in np.linspace(0.0, 1.0, N_CLOSE):
        pos[cursor] = grasp
        quats[cursor] = quat
        finger[cursor] = open_q * (1.0 - a) + closed * a
        cursor += 1
    if profile == "rush":
        # Slide to the grasp while the fingers close. The lift then starts centered;
        # the lagged wrist is still outside the cap. A constant shift is not a fix.
        for k in range(HARD_SLIDE_STEPS):
            u = (k + 1) / HARD_SLIDE_STEPS
            pos[N_APPROACH + k, 0] = HARD_SLIDE_M * (1.0 - u)
    for a in np.linspace(0.0, 1.0, N_LIFT):
        lifted = grasp.copy()
        lifted[2] = grasp[2] + LIFT_HEIGHT * a
        pos[cursor] = lifted
        quats[cursor] = quat
        finger[cursor] = closed
        cursor += 1
    for _ in range(N_HOLD):
        lifted = grasp.copy()
        lifted[2] = grasp[2] + LIFT_HEIGHT
        pos[cursor] = lifted
        quats[cursor] = quat
        finger[cursor] = closed
        cursor += 1

    obj = np.repeat(np.asarray(OBJECT_POS, dtype=np.float64)[None, :], horizon, axis=0)
    # The object target stays put until lift, then rises with the wrist.
    # Editing the hand reference must not change this array.
    for i in range(t_lift, horizon):
        obj[i, 2] = OBJECT_POS[2] + (pos[i, 2] - grasp[2])
    obj_quat = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), horizon, axis=0)
    ref = Reference(
        id=ref_id,
        label=label,
        wrist_pos=pos,
        wrist_quat=quats,
        finger_q=finger,
        finger_open=open_q.copy(),
        obj_pos=obj,
        obj_quat=obj_quat,
        wrist_lin_vel=np.zeros((horizon, 3)),
        wrist_ang_vel=np.zeros((horizon, 3)),
        finger_vel=np.zeros((horizon, 16)),
        fingertip_pos=np.zeros((horizon, 4, 3)),
        phase=phase,
        t_lift=t_lift,
        control_dt=CONTROL_DT,
        meta={"profile": profile},
    )
    return finalize(ref, hand)


def generate_dataset(n_easy: int, n_hard: int, n_bias: int, seed: int = 0, hand: HandModel | None = None) -> list[Reference]:
    """Replicates share the nominal geometry. Bias copies differ by sign.

    Labels are stored for the gate analysis. They are not decision features.
    """
    del seed  # geometry is deterministic; seed is reserved for training
    hand = hand or HandModel()
    refs: list[Reference] = []
    for i in range(n_easy):
        refs.append(make_clean_reference(f"easy-{i:02d}", "easy", "linear", hand))
    for i in range(n_hard):
        refs.append(make_clean_reference(f"hard-{i:02d}", "hard", "rush", hand))
    for i in range(n_bias):
        sign = 1.0 if i % 2 == 0 else -1.0
        base = make_clean_reference(f"bias-src-{i:02d}", "easy", "linear", hand)
        trans = np.zeros(3)
        if i % 2 == 0:
            trans[0] = sign * BIAS_TRANS_M
        else:
            trans[1] = sign * BIAS_TRANS_M
        edited = apply_edit(
            base,
            EditParams(trans, np.zeros(3), 1.0),
            hand,
            new_id=f"bias-trans-{i:02d}",
        )
        edited.label = "bias_trans"
        edited.meta = {"injected": edited.meta.get("profile"), "bias": "trans", "sign": sign}
        refs.append(edited)

        yaw = apply_edit(
            base,
            EditParams(np.zeros(3), np.array([0.0, sign * BIAS_PITCH_RAD, 0.0]), 1.0),
            hand,
            new_id=f"bias-rot-{i:02d}",
        )
        yaw.label = "bias_rot"
        yaw.meta = {"bias": "rot", "sign": sign}
        refs.append(yaw)

        close = apply_edit(
            base,
            EditParams(np.zeros(3), np.zeros(3), BIAS_CLOSURE_SCALE),
            hand,
            new_id=f"bias-closure-{i:02d}",
        )
        close.label = "bias_closure"
        close.meta = {"bias": "closure", "scale": BIAS_CLOSURE_SCALE}
        refs.append(close)
    return refs


def zero_action(obs: np.ndarray) -> np.ndarray:
    del obs
    return np.zeros(22)


def lead_action(obs: np.ndarray, bounds) -> np.ndarray:
    """Separability check: cancel the wrist lag with a velocity lead."""
    from src.config import LEAD_GAIN_S

    action = np.zeros(22)
    # ref linear velocity sits at obs[3+3+16+3+3 : +3] = [28:31]
    vel = obs[28:31]
    action[0:3] = np.clip(LEAD_GAIN_S * vel / bounds.trans_m, -1.0, 1.0)
    return action
