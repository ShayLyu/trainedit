"""Single-hand grasp-and-lift environment.

The wrist command is filtered by a first-order lag. Fingers use position
actuators. A residual in [-1, 1] is scaled by the active residual bounds
and added to the reference before that lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.config import (
    CONTROL_DT,
    LIFT_HEIGHT,
    N_SUBSTEPS,
    PHYSICS_DT,
    SUCCESS_HEIGHT_RATIO,
    SUCCESS_HOLD_FRACTION,
    SUCCESS_XY,
    WRIST_LAG_TAU,
)
from src.env.model import FINGERS, HandModel
from src.math3d import apply_rotvec, apply_se2_lin_vel, apply_se2_pos, apply_se2_quat, quat_error
from src.refs.schema import Reference

PHASE_NAMES = ("approach", "close", "lift", "hold")


@dataclass
class StepInfo:
    success: bool
    failure_phase: int
    wrist_err: float
    finger_err: float
    tip_err: float
    obj_z: float
    obj_xy: float
    n_finger_contacts: int
    slip: bool
    phase: int


class GraspLiftEnv:
    def __init__(self, ref: Reference, bounds, seed: int = 0) -> None:
        self.hand = HandModel()
        self.model = self.hand.model
        self.data = self.hand.data
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.ref = ref
        self.horizon = int(ref.wrist_pos.shape[0])
        self.alpha = float(1.0 - np.exp(-PHYSICS_DT / WRIST_LAG_TAU))
        self._cache_world(0.0, 0.0, 0.0)
        self.t = 0
        self.filt_pos = np.zeros(3)
        self.filt_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._hold_flags: list[bool] = []
        self._phase_contact = [0, 0, 0, 0]
        self._phase_steps = [0, 0, 0, 0]
        self._lifted_during_lift = False
        self.initial_obj_z = 0.0
        self.initial_obj_xy = np.zeros(2)

    def set_reference(self, ref: Reference) -> None:
        self.ref = ref
        self.horizon = int(ref.wrist_pos.shape[0])

    def _cache_world(self, dx: float, dy: float, yaw: float) -> None:
        ref = self.ref
        self.w_pos = apply_se2_pos(ref.wrist_pos, dx, dy, yaw)
        self.w_quat = apply_se2_quat(ref.wrist_quat, yaw)
        self.w_finger = ref.finger_q.copy()
        self.w_obj = apply_se2_pos(ref.obj_pos, dx, dy, yaw)
        self.w_obj_quat = apply_se2_quat(ref.obj_quat, yaw)
        self.w_lin = apply_se2_lin_vel(ref.wrist_lin_vel, yaw)
        self.w_ang = ref.wrist_ang_vel.copy()
        self.w_fvel = ref.finger_vel.copy()
        self.w_tips = apply_se2_pos(ref.fingertip_pos, dx, dy, yaw)
        self.w_phase = ref.phase.copy()

    def reset(self, seed: int | None = None, obj_offset: tuple[float, float, float] | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if obj_offset is None:
            dx, dy, yaw = 0.0, 0.0, 0.0
        else:
            dx, dy, yaw = obj_offset
        self._cache_world(dx, dy, yaw)
        self.t = 0
        self.filt_pos = self.w_pos[0].copy()
        self.filt_quat = self.w_quat[0].copy()
        self._hold_flags = []
        self._phase_contact = [0, 0, 0, 0]
        self._phase_steps = [0, 0, 0, 0]
        self._lifted_during_lift = False
        self.hand.set_state(
            self.filt_pos,
            self.filt_quat,
            self.w_finger[0],
            obj_pos=self.w_obj[0],
            obj_quat=self.w_obj_quat[0],
        )
        self.initial_obj_z = float(self.w_obj[0, 2])
        self.initial_obj_xy = self.w_obj[0, :2].copy()
        return self._obs(np.zeros(self.bounds.action_dim))

    @property
    def obs_dim(self) -> int:
        return 3 + 3 + 16 + 3 + 3 + 3 + 4 + 1

    def _object_pos(self) -> np.ndarray:
        adr = self.hand.object_qadr
        return self.data.qpos[adr : adr + 3].copy()

    def _object_quat(self) -> np.ndarray:
        adr = self.hand.object_qadr
        return self.data.qpos[adr + 3 : adr + 7].copy()

    def _object_vel(self) -> np.ndarray:
        adr = self.hand.object_dadr
        return self.data.qvel[adr : adr + 3].copy()

    def _finger_contacts(self) -> int:
        names = set()
        for i in range(self.data.ncon):
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(self.data.contact[i].geom1)) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(self.data.contact[i].geom2)) or ""
            pair = (g1, g2)
            if not any(name.startswith("object") for name in pair):
                continue
            other = g2 if g1.startswith("object") else g1
            for finger in FINGERS:
                if other.startswith(finger):
                    names.add(finger)
        return len(names)

    def _command(self, action: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        pos = self.w_pos[index] + a[0:3] * self.bounds.trans_m
        quat = apply_rotvec(self.w_quat[index], a[3:6] * self.bounds.rot_rad)
        finger = self.hand.clip_fingers(self.w_finger[index] + a[6:22] * self.bounds.finger_rad)
        return pos, quat, finger

    def _obs(self, action: np.ndarray) -> np.ndarray:
        index = min(self.t, self.horizon - 1)
        obj = self._object_pos()
        wrist_err = self.filt_pos - self.w_pos[index]
        rot_err = quat_error(self.filt_quat, self.w_quat[index])
        finger = self.data.qpos[self.hand.finger_qadr]
        finger_err = finger - self.w_finger[index]
        obj_err = obj - self.w_obj[index]
        phase = np.zeros(4)
        phase[int(self.w_phase[index])] = 1.0
        time = np.array([index / max(self.horizon - 1, 1)])
        obs = np.concatenate(
            [
                wrist_err,
                rot_err,
                finger_err,
                obj_err,
                self._object_vel(),
                self.w_lin[index],
                phase,
                time,
            ]
        )
        return obs.astype(np.float64)

    def _height_ok(self, obj: np.ndarray) -> bool:
        return bool(obj[2] > self.initial_obj_z + LIFT_HEIGHT * SUCCESS_HEIGHT_RATIO)

    def _xy_ok(self, obj: np.ndarray) -> bool:
        return bool(np.linalg.norm(obj[:2] - self.initial_obj_xy) < SUCCESS_XY)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        index = min(self.t, self.horizon - 1)
        phase = int(self.w_phase[index])
        cmd_pos, cmd_quat, cmd_finger = self._command(action, index)
        for _ in range(N_SUBSTEPS):
            self.filt_pos = self.filt_pos + self.alpha * (cmd_pos - self.filt_pos)
            self.filt_quat = apply_rotvec(
                self.filt_quat, self.alpha * quat_error(self.filt_quat, cmd_quat)
            )
            self.data.mocap_pos[0] = self.filt_pos
            self.data.mocap_quat[0] = self.filt_quat
            self.data.ctrl[:] = cmd_finger
            mujoco.mj_step(self.model, self.data)

        obj = self._object_pos()
        ncon = self._finger_contacts()
        self._phase_steps[phase] += 1
        if ncon > 0:
            self._phase_contact[phase] += 1
        lifted = self._height_ok(obj) and self._xy_ok(obj)
        if phase == 2 and lifted:
            self._lifted_during_lift = True
        if phase == 3:
            self._hold_flags.append(lifted)

        wrist_err = float(np.linalg.norm(self.filt_pos - self.w_pos[index]))
        finger_err = float(np.linalg.norm(self.data.qpos[self.hand.finger_qadr] - self.w_finger[index]))
        tips = np.stack([self.data.site_xpos[i] for i in self.hand.tip_ids])
        tip_err = float(np.linalg.norm(tips - self.w_tips[index]))
        vel = self._object_vel()
        slip = bool(ncon > 0 and np.linalg.norm(vel[:2]) > 0.08 and not lifted)

        self.t += 1
        done = self.t >= self.horizon
        success = False
        failure_phase = -1
        if done:
            success = self._episode_success()
            failure_phase = -1 if success else self._failure_phase()

        height_gain = max(0.0, obj[2] - self.initial_obj_z)
        reward = (
            1.0 * np.exp(-wrist_err / 0.015)
            + 0.3 * np.exp(-finger_err / 0.4)
            + 1.5 * np.clip(height_gain / LIFT_HEIGHT, 0.0, 1.0)
            + (5.0 if success else 0.0)
            - 0.02 * float(np.dot(action, action))
            - (0.2 if slip else 0.0)
        )
        info = StepInfo(
            success=success,
            failure_phase=failure_phase,
            wrist_err=wrist_err,
            finger_err=finger_err,
            tip_err=tip_err,
            obj_z=float(obj[2]),
            obj_xy=float(np.linalg.norm(obj[:2] - self.initial_obj_xy)),
            n_finger_contacts=ncon,
            slip=slip,
            phase=phase,
        )
        return self._obs(action), float(reward), done, info

    def _episode_success(self) -> bool:
        if not self._hold_flags:
            return False
        return float(np.mean(self._hold_flags)) >= SUCCESS_HOLD_FRACTION

    def _failure_phase(self) -> int:
        if self._phase_steps[1] > 0 and self._phase_contact[1] == 0 and not self._lifted_during_lift:
            return 1
        if not self._lifted_during_lift:
            return 2
        return 3


def run_episode(env: GraspLiftEnv, act_fn, seed: int | None = None, obj_offset=None) -> dict:
    obs = env.reset(seed=seed, obj_offset=obj_offset)
    total_reward = 0.0
    info = None
    wrist_err = []
    finger_err = []
    tip_err = []
    contacts = []
    slips = []
    while True:
        action = np.asarray(act_fn(obs), dtype=np.float64)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        wrist_err.append(info.wrist_err)
        finger_err.append(info.finger_err)
        tip_err.append(info.tip_err)
        contacts.append(info.n_finger_contacts > 0)
        slips.append(info.slip)
        if done:
            break
    assert info is not None
    return {
        "success": bool(info.success),
        "failure_phase": int(info.failure_phase),
        "return": float(total_reward),
        "wrist_err": float(np.mean(wrist_err)),
        "finger_err": float(np.mean(finger_err)),
        "tip_err": float(np.mean(tip_err)),
        "contact_fraction": float(np.mean(contacts)),
        "slip_fraction": float(np.mean(slips)),
        "steps": env.horizon,
        "final_obj_z": info.obj_z,
        "final_obj_xy": info.obj_xy,
    }


def aggregate_episodes(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("no episodes")
    hist = {name: 0 for name in ("success",) + PHASE_NAMES}
    for ep in episodes:
        if ep["success"]:
            hist["success"] += 1
        else:
            hist[PHASE_NAMES[ep["failure_phase"]]] += 1
    n = len(episodes)
    return {
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "failure_phase_hist": {k: v / n for k, v in hist.items()},
        "mean_wrist_err": float(np.mean([ep["wrist_err"] for ep in episodes])),
        "mean_finger_err": float(np.mean([ep["finger_err"] for ep in episodes])),
        "mean_tip_err": float(np.mean([ep["tip_err"] for ep in episodes])),
        "contact_fraction": float(np.mean([ep["contact_fraction"] for ep in episodes])),
        "slip_fraction": float(np.mean([ep["slip_fraction"] for ep in episodes])),
        "steps": int(sum(ep["steps"] for ep in episodes)),
    }


# Control step used by reference velocities. Re-exported for callers that
# should not import config and the env together in a cycle.
REF_DT = CONTROL_DT
