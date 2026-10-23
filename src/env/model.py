"""LEAP-topology hand as capsule bodies.

Joint order matches the LEAP 16-DoF layout: index, middle, ring, thumb,
each with mcp_side, mcp, pip, dip. Collision geoms are capsules standing
in for the official meshes. Fingers are mounted around the palm so a top
pinch of a small box is reachable; this is a kinematic approximation of
the LEAP hand, not a CAD import.
"""

from __future__ import annotations

import mujoco
import numpy as np

import src.config as cfg

FINGERS = ("index", "middle", "ring", "thumb")
JOINT_SUFFIXES = ("mcp_side", "mcp", "pip", "dip")

# Mounts in the palm frame. +X of each finger points inward. Flexion is about
# local +Z, so the tip swings in the horizontal plane and closes under the cap.
_MOUNTS = (
    # yaw -90 deg: local +X points to world -Y
    ("index", (0.00, 0.066, 0.000), (np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4))),
    # yaw 180 deg: local +X points to world -X
    ("middle", (0.066, 0.00, 0.000), (0.0, 0.0, 0.0, 1.0)),
    # yaw 0: local +X points to world +X, base is on -X
    ("ring", (-0.066, 0.00, 0.000), (1.0, 0.0, 0.0, 0.0)),
    # yaw +90 deg: local +X points to world +Y
    ("thumb", (0.00, -0.066, 0.000), (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))),
)

# Straight reach leaves the tip just outside the 4 cm box. Flexion curls down.
_L_SIDE = 0.012
_L_MCP = 0.016
_L_PIP = 0.014
_L_DIP = 0.010
_L_TIP = 0.006


def joint_names() -> list[str]:
    return [f"{finger}_{suffix}" for finger in FINGERS for suffix in JOINT_SUFFIXES]


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _finger_xml(name: str, pos, quat) -> str:
    axis = "0 0 1"
    return f"""
    <body name="{name}_base" pos="{_fmt(pos)}" quat="{_fmt(quat)}">
      <joint name="{name}_mcp_side" type="hinge" axis="0 0 1" range="-0.6 0.6" damping="0.04"/>
      <geom name="{name}_side_geom" class="hand" type="capsule" fromto="0 0 0 {_L_SIDE} 0 0" size="0.0055"/>
      <body name="{name}_prox" pos="{_L_SIDE} 0 0">
        <joint name="{name}_mcp" type="hinge" axis="{axis}" range="0 1.5" damping="0.03"/>
        <geom name="{name}_mcp_geom" class="hand" type="capsule" fromto="0 0 0 {_L_MCP} 0 0" size="0.0055"/>
        <body name="{name}_mid" pos="{_L_MCP} 0 0">
          <joint name="{name}_pip" type="hinge" axis="{axis}" range="0 1.6" damping="0.025"/>
          <geom name="{name}_pip_geom" class="hand" type="capsule" fromto="0 0 0 {_L_PIP} 0 0" size="0.005"/>
          <body name="{name}_dist" pos="{_L_PIP} 0 0">
            <joint name="{name}_dip" type="hinge" axis="{axis}" range="0 1.4" damping="0.02"/>
            <geom name="{name}_dip_geom" class="hand" type="capsule" fromto="0 0 0 {_L_DIP} 0 0" size="0.0045"/>
            <site name="{name}_tip" pos="{_L_DIP + _L_TIP} 0 0" size="0.004"/>
          </body>
        </body>
      </body>
    </body>
    """


def build_xml() -> str:
    fingers = "\n".join(_finger_xml(n, p, q) for n, p, q in _MOUNTS)
    actuators = "\n".join(
        f'<position name="act_{name}" joint="{name}" kp="{cfg.FINGER_KP}" kv="{cfg.FINGER_KV}" '
        f'forcerange="-{cfg.FINGER_FORCE} {cfg.FINGER_FORCE}"/>'
        for name in joint_names()
    )
    ox, oy, oz = cfg.OBJECT_POS
    cap = cfg.CAP_HALF
    return f"""
<mujoco model="leap_grasp_lift">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.PHYSICS_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom condim="4" solref="0.01 1" solimp="0.92 0.98 0.001" friction="{cfg.FRICTION} 0.01 0.001"/>
    <default class="hand">
      <geom contype="2" conaffinity="1" rgba="0.25 0.42 0.75 1"/>
    </default>
    <default class="env">
      <geom contype="1" conaffinity="3"/>
    </default>
  </default>
  <worldbody>
    <body name="table" pos="0 0 -0.02">
      <geom name="table" class="env" type="box" size="0.35 0.35 0.02" rgba="0.35 0.35 0.35 1"/>
    </body>
    <!-- Stem plus a wider cap. Fingers close in the gap and lift the cap from below. -->
    <body name="object" pos="{ox} {oy} {oz}">
      <freejoint name="object_free"/>
      <geom name="object" class="env" type="box" size="0.012 0.012 0.006" pos="0 0 -0.014"
            mass="0.03" rgba="0.85 0.45 0.2 1"/>
      <geom name="object_cap" class="env" type="box" size="{cap} {cap} 0.006" pos="0 0 0.014"
            mass="0.03" rgba="0.75 0.35 0.15 1"/>
    </body>
    <body name="palm" mocap="true" pos="0 0 0.08">
      <inertial pos="0 0 0" mass="0.30" diaginertia="0.0008 0.0008 0.0004"/>
      <geom name="palm_geom" class="hand" type="box" size="0.02 0.02 0.008" pos="0 0 0.045"/>
      {fingers}
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml())


def _adr(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


class HandModel:
    """Kinematic access to the compiled model. One instance is not forked across threads."""

    def __init__(self) -> None:
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names()
        self.finger_qadr = np.array(
            [_adr(self.model.joint(name).qposadr) for name in self.joint_names], dtype=np.int32
        )
        self.finger_low = np.array(
            [float(self.model.joint(name).range[0]) for name in self.joint_names]
        )
        self.finger_high = np.array(
            [float(self.model.joint(name).range[1]) for name in self.joint_names]
        )
        self.object_qadr = _adr(self.model.joint("object_free").qposadr)
        self.object_dadr = _adr(self.model.joint("object_free").dofadr)
        self.palm_id = int(self.model.body("palm").id)
        self.tip_ids = [int(self.model.site(f"{name}_tip").id) for name in FINGERS]
        self.actuator_names = [f"act_{name}" for name in self.joint_names]

    def clip_fingers(self, finger_q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(finger_q, dtype=np.float64), self.finger_low, self.finger_high)

    def set_state(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        finger_q: np.ndarray,
        obj_pos: np.ndarray | None = None,
        obj_quat: np.ndarray | None = None,
    ) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[0] = np.asarray(wrist_pos, dtype=np.float64)
        self.data.mocap_quat[0] = np.asarray(wrist_quat, dtype=np.float64)
        self.data.qpos[self.finger_qadr] = self.clip_fingers(finger_q)
        if obj_pos is None:
            obj_pos = np.array(cfg.OBJECT_POS, dtype=np.float64)
        if obj_quat is None:
            obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.object_qadr : self.object_qadr + 3] = obj_pos
        self.data.qpos[self.object_qadr + 3 : self.object_qadr + 7] = obj_quat
        mujoco.mj_forward(self.model, self.data)

    def fingertips(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        finger_q: np.ndarray,
    ) -> np.ndarray:
        self.set_state(wrist_pos, wrist_quat, finger_q)
        return np.stack([self.data.site_xpos[i].copy() for i in self.tip_ids])

    def finger_targets_from_q(self, finger_q: np.ndarray) -> np.ndarray:
        return self.clip_fingers(finger_q)
