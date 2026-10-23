"""Nominal open and closed finger postures for the flanged lift."""

from __future__ import annotations

import numpy as np

from src.config import CLOSED_CLOSURE, OBJECT_POS, OPEN_CLOSURE
from src.env.model import HandModel


def q_from_closure(closure: float) -> np.ndarray:
    """Map one synergy coordinate onto the 16 LEAP joints.

    mcp_side stays at zero. Flexion joints share the synergy so a single
    scale edits the grasp shape.
    """
    q = np.zeros(16, dtype=np.float64)
    for i in range(4):
        q[i * 4 + 1] = 0.55 * closure
        q[i * 4 + 2] = 0.75 * closure
        q[i * 4 + 3] = 0.45 * closure
    return q


def nominal_keyframes(hand: HandModel | None = None) -> dict:
    hand = hand or HandModel()
    wrist_pos = np.array([OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2]], dtype=np.float64)
    wrist_quat = np.array([1.0, 0.0, 0.0, 0.0])
    finger_open = hand.clip_fingers(q_from_closure(OPEN_CLOSURE))
    finger_closed = hand.clip_fingers(q_from_closure(CLOSED_CLOSURE))
    return {
        "wrist_pos": wrist_pos,
        "wrist_quat": wrist_quat,
        "finger_open": finger_open,
        "finger_closed": finger_closed,
        "hand": hand,
    }
