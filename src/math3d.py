"""Small SO(3) helpers. Quaternions are MuJoCo wxyz."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle * 0.5), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)
    return q[1:4] / s * angle


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)])


def apply_rotvec(q: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    return quat_normalize(quat_mul(rotvec_to_quat(rotvec), q))


def quat_error(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Rotation vector, in the world frame, taking q_from to q_to."""
    return quat_to_rotvec(quat_mul(q_to, quat_inv(q_from)))


def apply_se2_pos(pos: np.ndarray, dx: float, dy: float, yaw: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.array(pos, copy=True)
    xy = pos[..., :2]
    out[..., 0] = c * xy[..., 0] - s * xy[..., 1] + dx
    out[..., 1] = s * xy[..., 0] + c * xy[..., 1] + dy
    return out


def apply_se2_quat(quat: np.ndarray, yaw: float) -> np.ndarray:
    yq = yaw_quat(yaw)
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim == 1:
        return quat_normalize(quat_mul(yq, quat))
    return np.stack([quat_normalize(quat_mul(yq, quat[i])) for i in range(len(quat))])


def apply_se2_lin_vel(vel: np.ndarray, yaw: float) -> np.ndarray:
    vel = np.asarray(vel, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.array(vel, copy=True)
    xy = vel[..., :2]
    out[..., 0] = c * xy[..., 0] - s * xy[..., 1]
    out[..., 1] = s * xy[..., 0] + c * xy[..., 1]
    return out
