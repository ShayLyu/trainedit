"""Unit tests that do not train a policy."""

from __future__ import annotations

import unittest

import numpy as np

from src.budget.ledger import Ledger
from src.config import BIAS_CLOSURE_SCALE
from src.decide.rule import decision_features, select_action
from src.eval.gate import assess_gate
from src.math3d import quat_mul, quat_normalize, quat_to_rotvec, rotvec_to_quat
from src.refs.edit import apply_edit
from src.refs.generate import generate_dataset
from src.refs.schema import EditParams


class MathTests(unittest.TestCase):
    def test_rotvec_roundtrip(self) -> None:
        rv = np.array([0.2, -0.1, 0.4])
        q = rotvec_to_quat(rv)
        back = quat_to_rotvec(q)
        np.testing.assert_allclose(back, rv, atol=1e-6)

    def test_quat_inverse_identity(self) -> None:
        q = quat_normalize(np.array([0.5, 0.1, -0.2, 0.3]))
        ident = quat_mul(q, np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q))
        np.testing.assert_allclose(ident, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


class LedgerTests(unittest.TestCase):
    def test_counts_by_kind(self) -> None:
        ledger = Ledger()
        ledger.add("train", 10)
        ledger.add("edit_eval", 4)
        ledger.add("train", 6)
        self.assertEqual(ledger.total, 20)
        self.assertEqual(ledger.by_kind()["train"], 16)


class RuleTests(unittest.TestCase):
    def test_choices(self) -> None:
        self.assertEqual(select_action(0.0, 0.9, 100, 0.1, 100), "train")
        self.assertEqual(select_action(0.0, 0.1, 100, 0.9, 80), "edit")
        self.assertEqual(select_action(0.95, 0.96, 50, 0.95, 50), "defer")
        self.assertEqual(select_action(0.2, 0.22, 40, 0.21, 40), "defer")

    def test_features_reject_labels(self) -> None:
        with self.assertRaises(ValueError):
            decision_features({"success_rate": 0.5, "label": "easy"})
        clean = decision_features({"success_rate": 0.5, "train_steps": 3, "unused": 1})
        self.assertEqual(clean, {"success_rate": 0.5, "train_steps": 3})


class GateTests(unittest.TestCase):
    def test_separating_gate_proceeds(self) -> None:
        records = []
        for lab, train, edit in (
            ("easy", 1.0, 1.0),
            ("hard", 0.8, 0.2),
            ("bias_trans", 0.0, 1.0),
            ("bias_rot", 0.0, 1.0),
            ("bias_closure", 0.1, 0.9),
        ):
            records.append({"label": lab, "success_train": train, "success_edit": edit})
        gate = assess_gate(records)
        self.assertTrue(gate["proceed"])

    def test_hard_tie_still_proceeds(self) -> None:
        records = []
        for lab, train, edit in (
            ("easy", 1.0, 1.0),
            ("hard", 1.0, 1.0),
            ("bias_trans", 0.0, 1.0),
            ("bias_rot", 0.0, 1.0),
            ("bias_closure", 0.2, 1.0),
        ):
            records.append({"label": lab, "success_train": train, "success_edit": edit})
        self.assertTrue(assess_gate(records)["proceed"])

    def test_training_away_bias_stops(self) -> None:
        records = [
            {"label": "easy", "success_train": 1.0, "success_edit": 1.0},
            {"label": "hard", "success_train": 0.8, "success_edit": 0.2},
            {"label": "bias_trans", "success_train": 0.9, "success_edit": 0.95},
            {"label": "bias_rot", "success_train": 0.9, "success_edit": 0.95},
            {"label": "bias_closure", "success_train": 0.85, "success_edit": 0.9},
        ]
        self.assertFalse(assess_gate(records)["proceed"])


class EditTests(unittest.TestCase):
    def test_edit_keeps_object_motion_and_matches_fk(self) -> None:
        refs = generate_dataset(1, 0, 0)
        ref = refs[0]
        from src.env.model import HandModel

        hand = HandModel()
        edited = apply_edit(ref, EditParams(np.array([0.03, 0.0, 0.0]), np.zeros(3), 1.0), hand)
        np.testing.assert_allclose(edited.obj_pos, ref.obj_pos)
        self.assertGreater(abs(edited.wrist_pos[0, 0] - ref.wrist_pos[0, 0]), 0.02)
        # Lift stays vertically above the edited grasp.
        grasp = edited.t_lift - 1
        np.testing.assert_allclose(edited.wrist_pos[-1, :2], edited.wrist_pos[grasp, :2], atol=1e-6)
        self.assertGreater(edited.wrist_pos[-1, 2], edited.wrist_pos[grasp, 2] + 0.05)
        # Velocities match a forward difference. Tips match FK.
        dt = edited.control_dt
        np.testing.assert_allclose(
            edited.wrist_lin_vel[0],
            (edited.wrist_pos[1] - edited.wrist_pos[0]) / dt,
            atol=1e-6,
        )
        tips = hand.fingertips(edited.wrist_pos[3], edited.wrist_quat[3], edited.finger_q[3])
        np.testing.assert_allclose(tips, edited.fingertip_pos[3], atol=1e-5)

    def test_closure_scale_restores(self) -> None:
        refs = generate_dataset(0, 0, 1)
        biased = next(ref for ref in refs if ref.label == "bias_closure")
        from src.env.model import HandModel

        hand = HandModel()
        restored = apply_edit(
            biased,
            EditParams(np.zeros(3), np.zeros(3), 1.0 / BIAS_CLOSURE_SCALE),
            hand,
        )
        clean = generate_dataset(1, 0, 0)[0]
        np.testing.assert_allclose(restored.finger_q, clean.finger_q, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
