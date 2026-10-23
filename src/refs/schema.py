"""Reference and edit records.

`label` and `meta` are for analysis after a decision. The allocator receives
only execution evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EditParams:
    trans: np.ndarray
    rotvec: np.ndarray
    closure_scale: float

    @staticmethod
    def identity() -> "EditParams":
        return EditParams(np.zeros(3), np.zeros(3), 1.0)

    def copy(self) -> "EditParams":
        return EditParams(self.trans.copy(), self.rotvec.copy(), float(self.closure_scale))

    def to_dict(self) -> dict:
        return {
            "trans": [float(v) for v in self.trans],
            "rotvec": [float(v) for v in self.rotvec],
            "closure_scale": float(self.closure_scale),
        }


@dataclass
class Reference:
    id: str
    label: str
    wrist_pos: np.ndarray
    wrist_quat: np.ndarray
    finger_q: np.ndarray
    finger_open: np.ndarray
    obj_pos: np.ndarray
    obj_quat: np.ndarray
    wrist_lin_vel: np.ndarray
    wrist_ang_vel: np.ndarray
    finger_vel: np.ndarray
    fingertip_pos: np.ndarray
    phase: np.ndarray
    t_lift: int
    control_dt: float
    meta: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.wrist_pos.shape[0])
