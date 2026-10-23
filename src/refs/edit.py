"""Structured edits on the pre-lift window, then consistent velocities and tips."""

from __future__ import annotations

import numpy as np

from src.env.model import HandModel
from src.math3d import apply_rotvec, quat_error, quat_mul, quat_normalize
from src.refs.schema import EditParams, Reference


def _finite_velocities(ref: Reference) -> None:
    dt = ref.control_dt
    t = ref.horizon
    lin = np.zeros((t, 3))
    ang = np.zeros((t, 3))
    fvel = np.zeros_like(ref.finger_q)
    for i in range(t - 1):
        lin[i] = (ref.wrist_pos[i + 1] - ref.wrist_pos[i]) / dt
        ang[i] = quat_error(ref.wrist_quat[i], ref.wrist_quat[i + 1]) / dt
        fvel[i] = (ref.finger_q[i + 1] - ref.finger_q[i]) / dt
    if t > 1:
        lin[-1] = lin[-2]
        ang[-1] = ang[-2]
        fvel[-1] = fvel[-2]
    ref.wrist_lin_vel = lin
    ref.wrist_ang_vel = ang
    ref.finger_vel = fvel


def _fingertips(ref: Reference, hand: HandModel) -> None:
    tips = np.zeros((ref.horizon, 4, 3))
    for i in range(ref.horizon):
        tips[i] = hand.fingertips(ref.wrist_pos[i], ref.wrist_quat[i], ref.finger_q[i])
    ref.fingertip_pos = tips


def finalize(ref: Reference, hand: HandModel) -> Reference:
    _finite_velocities(ref)
    _fingertips(ref, hand)
    return ref


def apply_edit(ref: Reference, edit: EditParams, hand: HandModel, new_id: str | None = None) -> Reference:
    """Apply a wrist SE(3) offset and a closure scale before lift.

    The lift continues vertically from the edited grasp pose. The object
    trajectory is copied unchanged. Velocities and fingertip targets are
    recomputed so the reference stays internally consistent.
    """
    t_lift = ref.t_lift
    pos = ref.wrist_pos.copy()
    quat = ref.wrist_quat.copy()
    finger = ref.finger_q.copy()
    delta_q = quat_normalize(apply_rotvec(np.array([1.0, 0.0, 0.0, 0.0]), edit.rotvec))
    for i in range(t_lift):
        pos[i] = pos[i] + edit.trans
        quat[i] = quat_normalize(quat_mul(delta_q, quat[i]))
        finger[i] = ref.finger_open + edit.closure_scale * (ref.finger_q[i] - ref.finger_open)
    finger = hand.clip_fingers(finger)
    if t_lift > 0:
        grasp_pos = pos[t_lift - 1].copy()
        grasp_quat = quat[t_lift - 1].copy()
        grasp_finger = finger[t_lift - 1].copy()
        z0 = ref.wrist_pos[t_lift - 1, 2]
        for i in range(t_lift, ref.horizon):
            pos[i] = grasp_pos.copy()
            pos[i, 2] = grasp_pos[2] + (ref.wrist_pos[i, 2] - z0)
            quat[i] = grasp_quat.copy()
            finger[i] = grasp_finger.copy()
    out = Reference(
        id=new_id or ref.id,
        label=ref.label,
        wrist_pos=pos,
        wrist_quat=quat,
        finger_q=finger,
        finger_open=ref.finger_open.copy(),
        obj_pos=ref.obj_pos.copy(),
        obj_quat=ref.obj_quat.copy(),
        wrist_lin_vel=np.zeros_like(ref.wrist_lin_vel),
        wrist_ang_vel=np.zeros_like(ref.wrist_ang_vel),
        finger_vel=np.zeros_like(ref.finger_vel),
        fingertip_pos=np.zeros_like(ref.fingertip_pos),
        phase=ref.phase.copy(),
        t_lift=ref.t_lift,
        control_dt=ref.control_dt,
        meta=dict(ref.meta),
    )
    return finalize(out, hand)


def yaw_edit(yaw: float) -> EditParams:
    return EditParams(np.zeros(3), np.array([0.0, 0.0, yaw]), 1.0)


def trans_edit(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> EditParams:
    return EditParams(np.array([x, y, z], dtype=np.float64), np.zeros(3), 1.0)
