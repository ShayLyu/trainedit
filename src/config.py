"""TrainEdit v1 experiment parameters.

The protocol in docs/v1_protocol.md describes these values. If a number
changes, update both files.
"""

from __future__ import annotations

from dataclasses import dataclass


# One budget unit is one control step, not one physics substep.
PHYSICS_DT = 0.005
N_SUBSTEPS = 4
CONTROL_DT = PHYSICS_DT * N_SUBSTEPS  # 0.02 s

N_APPROACH = 16
N_CLOSE = 8
N_LIFT = 12
N_HOLD = 8

LIFT_HEIGHT = 0.08
SUCCESS_HEIGHT_RATIO = 0.75
SUCCESS_HOLD_FRACTION = 0.75
SUCCESS_XY = 0.07

OBJECT_POS = (0.0, 0.0, 0.02)
OBJECT_HALF = 0.02
OBJECT_MASS = 0.06
FRICTION = 2.0
# Cap half-width. Closed fingertips sit near 0.019 m, so this sets how far a
# constant wrist shift can be and still stay under the flange.
CAP_HALF = 0.034

# Fingers are position-controlled. The wrist mocap follows its command through
# a first-order lag. The hard reference slides in from the side while the
# fingers close; the lag is still outside the cap when the lift starts.
# A constant shift does not cancel that slide.
WRIST_LAG_TAU = 0.05
FINGER_KP = 2.5
FINGER_KV = 0.15
FINGER_FORCE = 2.0

# Open fingers clear the cap. Closed fingers rest in the gap underneath it.
OPEN_CLOSURE = 0.80
CLOSED_CLOSURE = 0.28
HOVER_OFFSET = (0.0, 0.0, 0.05)
HARD_SLIDE_M = 0.04
HARD_SLIDE_STEPS = 6


@dataclass(frozen=True)
class ResidualBounds:
    trans_m: float
    rot_rad: float
    finger_rad: float

    @property
    def action_dim(self) -> int:
        return 3 + 3 + 16


BOUNDS_SMALL = ResidualBounds(0.005, 0.05, 0.08)
BOUNDS_MEDIUM = ResidualBounds(0.015, 0.15, 0.20)
BOUNDS_LARGE = ResidualBounds(0.040, 0.50, 0.80)

# Injected biases lie outside the medium residual box and on the edit grid,
# so the gate measures allocation rather than search resolution.
# Rotation bias is pitch about +Y. A yaw about Z leaves this symmetric hand
# under the cap, so it is not a usable wrist-rotation failure.
BIAS_TRANS_M = 0.03
BIAS_PITCH_RAD = 0.45
BIAS_CLOSURE_SCALE = 0.20

# Feedforward lead (seconds) used only by the separability check.
LEAD_GAIN_S = 0.025

EVAL_EPISODES = 4
GEN_OFFSETS = (
    (0.01, 0.0, 0.0),
    (0.0, 0.01, 0.0),
    (0.0, 0.0, 0.25),  # dx, dy, yaw
)

PPO_HIDDEN = 128
PPO_N_STEPS = 512
PPO_EPOCHS = 2
PPO_MINIBATCH = 128
PPO_LR = 1.0e-3
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENT = 0.002
PPO_VF = 0.5
PPO_GRAD_NORM = 0.5
ACTION_L2 = 0.02

EDIT_ACCEPT_MARGIN = 0.1
DEFER_GAIN_MARGIN = 0.08
FIXED_EDIT_RATIO = 0.30

# Edit search grids. Closure scales are absolute multipliers of the
# open-to-current closing motion, so 1/0.20 restores a 0.20 injection.
# Pitch candidates are rotations about +Y, matching the injected bias.
TRANS_CANDIDATES = (0.0, 0.015, -0.015, 0.03, -0.03)
PITCH_CANDIDATES = (0.0, 0.2, -0.2, 0.45, -0.45)
CLOSURE_CANDIDATES = (1.0, 0.45, 1.0 / 0.45, 0.7, 1.6, 1.0 / 0.20)

CEM_SAMPLES = 6
CEM_ITERS = 2
CEM_ELITES = 2


@dataclass(frozen=True)
class ExperimentScale:
    name: str
    n_easy: int
    n_hard: int
    n_bias: int
    budget: int
    eval_episodes: int
    probe_train_steps: int
    sensitivity_budget: int
    seed: int


SCALE_REDUCED = ExperimentScale(
    name="reduced",
    n_easy=2,
    n_hard=2,
    n_bias=2,
    budget=10240,
    eval_episodes=4,
    probe_train_steps=5120,
    sensitivity_budget=5120,
    seed=0,
)

SCALE_FULL = ExperimentScale(
    name="full",
    n_easy=10,
    n_hard=10,
    n_bias=8,
    budget=32768,
    eval_episodes=8,
    probe_train_steps=4096,
    sensitivity_budget=4096,
    seed=0,
)
