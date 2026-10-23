"""Train-only, edit-then-train, and the four allocation baselines."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from src.budget.ledger import Ledger
from src.config import (
    BIAS_CLOSURE_SCALE,
    CLOSURE_CANDIDATES,
    EDIT_ACCEPT_MARGIN,
    FIXED_EDIT_RATIO,
    GEN_OFFSETS,
    PITCH_CANDIDATES,
    PPO_LR,
    TRANS_CANDIDATES,
)
from src.decide.rule import select_action
from src.env.grasp_lift import GraspLiftEnv, aggregate_episodes, run_episode
from src.refs.edit import apply_edit
from src.refs.schema import EditParams, Reference
from src.train.ppo import ResidualPolicy, make_policy, train_steps


@dataclass
class Trace:
    traj_id: str
    label: str
    curve: list[tuple[int, float]] = field(default_factory=list)
    final_success: float = 0.0
    steps_spent: int = 0
    actions: list[str] = field(default_factory=list)
    edited: bool = False
    deferred: bool = False
    edit: dict | None = None
    gen_success: float = 0.0
    ledger: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "label": self.label,
            "curve": self.curve,
            "final_success": self.final_success,
            "steps_spent": self.steps_spent,
            "actions": self.actions,
            "edited": self.edited,
            "deferred": self.deferred,
            "edit": self.edit,
            "gen_success": self.gen_success,
            "ledger": self.ledger,
        }


def _zero(_obs: np.ndarray) -> np.ndarray:
    return np.zeros(22, dtype=np.float64)


def _quat_pitch(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    sinp = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(sinp, -1.0, 1.0)))


def evaluate(env: GraspLiftEnv, ref: Reference, policy_fn, n_episodes: int, offsets=None) -> tuple[dict, int]:
    env.set_reference(ref)
    episodes = []
    for i in range(n_episodes):
        offset = None if offsets is None else offsets[i % len(offsets)]
        episodes.append(run_episode(env, policy_fn, seed=10_000 + i, obj_offset=offset))
    stats = aggregate_episodes(episodes)
    return stats, int(stats["steps"])


def _snap(value: float, candidates: tuple[float, ...]) -> float:
    return float(min(candidates, key=lambda item: abs(item - value)))


def _diagnose(env: GraspLiftEnv, ref: Reference, policy_fn) -> dict:
    """One execution used to propose a single edit per family."""
    env.set_reference(ref)
    obs = env.reset(seed=0)
    rel = []
    pitches = []
    contacts = []
    success = False
    phase = env.w_phase
    t_lift = ref.t_lift
    while True:
        action = policy_fn(obs)
        index = min(env.t, env.horizon - 1)
        if phase[index] == 1 or index == max(t_lift - 1, 0):
            obj = env._object_pos()
            rel.append(obj[:2] - env.w_pos[index, :2])
            pitches.append(_quat_pitch(env.w_quat[index]))
        obs, _reward, done, info = env.step(action)
        if phase[min(index, len(phase) - 1)] == 1:
            contacts.append(info.n_finger_contacts > 0)
        if done:
            success = bool(info.success)
            break
    rel_arr = np.asarray(rel) if rel else np.zeros((1, 2))
    mean_xy = np.clip(rel_arr.mean(axis=0), -0.04, 0.04)
    trans = np.zeros(3)
    axis = int(np.argmax(np.abs(mean_xy)))
    trans[axis] = _snap(float(mean_xy[axis]), TRANS_CANDIDATES)
    return {
        "trans": trans,
        "pitch": _snap(float(-np.mean(pitches) if pitches else 0.0), PITCH_CANDIDATES),
        "closure_scale": (1.0 / BIAS_CLOSURE_SCALE)
        if ((not contacts or float(np.mean(contacts)) < 0.2) or not success)
        else 1.0,
        "steps": env.horizon,
    }


def propose_edits(env: GraspLiftEnv, ref: Reference, policy_fn) -> tuple[list[EditParams], int]:
    diag = _diagnose(env, ref, policy_fn)
    edits = [
        EditParams(diag["trans"], np.zeros(3), 1.0),
        EditParams(np.zeros(3), np.array([0.0, diag["pitch"], 0.0]), 1.0),
        EditParams(np.zeros(3), np.zeros(3), float(diag["closure_scale"])),
    ]
    return edits, int(diag["steps"])


def _score_edit(env, ref, edit: EditParams, policy_fn, n_episodes: int) -> tuple[float, int, Reference]:
    edited = ref if _is_identity(edit) else apply_edit(ref, edit, env.hand, new_id=ref.id + "/edit")
    stats, steps = evaluate(env, edited, policy_fn, n_episodes)
    return float(stats["success_rate"]), steps, edited


def _is_identity(edit: EditParams) -> bool:
    return (
        float(np.linalg.norm(edit.trans)) < 1e-8
        and float(np.linalg.norm(edit.rotvec)) < 1e-8
        and abs(edit.closure_scale - 1.0) < 1e-8
    )


def greedy_search(env, ref: Reference, policy_fn, n_episodes: int) -> tuple[EditParams, Reference, float, int]:
    """Coordinate search over translation, pitch, and closure. Identity is included."""
    best = EditParams.identity()
    best_ref = ref
    best_sr, steps, _ = _score_edit(env, ref, best, policy_fn, n_episodes)
    total = steps

    def consider(edit: EditParams) -> None:
        nonlocal best, best_ref, best_sr, total
        sr, used, edited = _score_edit(env, ref, edit, policy_fn, n_episodes)
        total += used
        if sr > best_sr + 1e-6:
            best, best_ref, best_sr = edit.copy(), edited, sr

    for value in TRANS_CANDIDATES:
        if abs(value) < 1e-8:
            continue
        consider(EditParams(np.array([value, best.trans[1], 0.0]), best.rotvec.copy(), best.closure_scale))
        consider(EditParams(np.array([best.trans[0], value, 0.0]), best.rotvec.copy(), best.closure_scale))
    for pitch in PITCH_CANDIDATES:
        if abs(pitch) < 1e-8:
            continue
        consider(EditParams(best.trans.copy(), np.array([0.0, pitch, 0.0]), best.closure_scale))
    for scale in CLOSURE_CANDIDATES:
        if abs(scale - 1.0) < 1e-8:
            continue
        consider(EditParams(best.trans.copy(), best.rotvec.copy(), float(scale)))
    if best_sr < EDIT_ACCEPT_MARGIN:
        return EditParams.identity(), ref, best_sr, total
    return best, best_ref, best_sr, total


def _policy_call(policy: ResidualPolicy):
    return lambda obs: policy(obs, deterministic=True)


def _generalize(env, ref, policy_fn, ledger: Ledger) -> float:
    stats, steps = evaluate(env, ref, policy_fn, len(GEN_OFFSETS), offsets=list(GEN_OFFSETS))
    ledger.add("validate", steps)
    return float(stats["success_rate"])


def _fit_trace(trace: Trace, ledger: Ledger, success: float, curve, actions, edited, deferred, edit, gen) -> Trace:
    trace.curve = [(int(s), float(v)) for s, v in curve]
    trace.final_success = float(success)
    trace.steps_spent = int(ledger.total)
    trace.actions = actions
    trace.edited = edited
    trace.deferred = deferred
    trace.edit = None if edit is None else edit.to_dict()
    trace.gen_success = float(gen)
    trace.ledger = ledger.to_dict()
    return trace


def _train_budget(env, ref, bounds, seed: int, budget: int, n_eval: int, ledger: Ledger, kind: str = "train") -> tuple[ResidualPolicy, list, float]:
    env.set_reference(ref)
    net, opt = make_policy(env.obs_dim, bounds.action_dim, seed)

    def eval_fn(policy: ResidualPolicy):
        stats, steps = evaluate(env, ref, _policy_call(policy), n_eval)
        return float(stats["success_rate"]), steps

    curve, _steps = train_steps(env, net, opt, budget, eval_fn)
    # train_steps counts steps internally; mirror them into the ledger in one entry.
    spent = curve[-1][0] if curve else 0
    ledger.add(kind, spent)
    return ResidualPolicy(net), curve, float(curve[-1][1] if curve else 0.0)


def run_train_only(ref: Reference, bounds, budget: int, n_eval: int, seed: int) -> Trace:
    env = GraspLiftEnv(ref, bounds, seed=seed)
    ledger = Ledger()
    policy, curve, success = _train_budget(env, ref, bounds, seed, budget, n_eval, ledger, "train")
    gen = _generalize(env, ref, _policy_call(policy), ledger)
    trace = Trace(ref.id, ref.label)
    return _fit_trace(trace, ledger, success, curve, ["train"], False, False, None, gen)


def run_edit_then_train(ref: Reference, bounds, budget: int, n_eval: int, seed: int) -> Trace:
    env = GraspLiftEnv(ref, bounds, seed=seed)
    ledger = Ledger()
    edit, edited, sr_edit, search_steps = greedy_search(env, ref, _zero, n_episodes=2)
    ledger.add("edit_eval", search_steps)
    improved = (not _is_identity(edit)) and sr_edit >= EDIT_ACCEPT_MARGIN
    chosen = edited if improved else ref
    remain = max(0, budget - search_steps)
    actions = ["edit", "train"] if improved else ["train"]
    if remain > 0:
        policy, curve, success = _train_budget(env, chosen, bounds, seed, remain, n_eval, ledger, "train")
    else:
        policy = ResidualPolicy(make_policy(env.obs_dim, bounds.action_dim, seed)[0])
        curve = [(search_steps, sr_edit if improved else 0.0)]
        success = sr_edit if improved else 0.0
    curve = [(search_steps, sr_edit if improved else curve[0][1])] + [(search_steps + s, v) for s, v in curve]
    gen = _generalize(env, chosen, _policy_call(policy) if remain > 0 else _zero, ledger)
    trace = Trace(ref.id, ref.label)
    return _fit_trace(trace, ledger, success, curve, actions, improved, False, edit if improved else None, gen)


def run_uniform_preprocess(ref: Reference, bounds, budget: int, n_eval: int, seed: int) -> Trace:
    env = GraspLiftEnv(ref, bounds, seed=seed)
    ledger = Ledger()
    proposals, diag_steps = propose_edits(env, ref, _zero)
    ledger.add("edit_eval", diag_steps)
    base_sr, base_steps = evaluate(env, ref, _zero, 2)
    ledger.add("edit_eval", base_steps)
    best_edit = EditParams.identity()
    best_ref = ref
    best_sr = base_sr["success_rate"]
    for edit in proposals:
        sr, steps, edited = _score_edit(env, ref, edit, _zero, 2)
        ledger.add("edit_eval", steps)
        if sr > best_sr + EDIT_ACCEPT_MARGIN:
            best_edit, best_ref, best_sr = edit, edited, sr
    improved = not _is_identity(best_edit)
    spent = ledger.total
    remain = max(0, budget - spent)
    policy, curve, success = _train_budget(env, best_ref, bounds, seed, remain, n_eval, ledger, "train")
    curve = [(spent, best_sr)] + [(spent + s, v) for s, v in curve]
    gen = _generalize(env, best_ref, _policy_call(policy), ledger)
    trace = Trace(ref.id, ref.label)
    return _fit_trace(
        trace, ledger, success, curve, ["preprocess", "train"], improved, False, best_edit if improved else None, gen
    )


def run_fixed_ratio(ref: Reference, bounds, budget: int, n_eval: int, seed: int) -> Trace:
    env = GraspLiftEnv(ref, bounds, seed=seed)
    ledger = Ledger()
    edit_budget = int(budget * FIXED_EDIT_RATIO)
    edit, edited, sr_edit, search_steps = greedy_search(env, ref, _zero, n_episodes=2)
    ledger.add("edit_eval", search_steps)
    if ledger.total < edit_budget:
        extra = edit_budget - ledger.total
        # Burn the unused edit share on repeated validation so the ratio stays fixed.
        episodes = max(1, extra // ref.horizon)
        stats, steps = evaluate(env, edited, _zero, episodes)
        ledger.add("validate", steps)
        sr_edit = float(stats["success_rate"])
    improved = (not _is_identity(edit)) and sr_edit >= EDIT_ACCEPT_MARGIN
    chosen = edited if improved else ref
    remain = max(0, budget - ledger.total)
    policy, curve, success = _train_budget(env, chosen, bounds, seed, remain, n_eval, ledger, "train")
    curve = [(ledger.total - remain, sr_edit)] + [(ledger.total - remain + s, v) for s, v in curve]
    gen = _generalize(env, chosen, _policy_call(policy), ledger)
    trace = Trace(ref.id, ref.label)
    return _fit_trace(
        trace, ledger, success, curve, ["fixed_edit", "train"], improved, False, edit if improved else None, gen
    )


def run_adaptive(ref: Reference, bounds, budget: int, n_eval: int, seed: int, probe_steps: int) -> Trace:
    env = GraspLiftEnv(ref, bounds, seed=seed)
    ledger = Ledger()
    net, opt = make_policy(env.obs_dim, bounds.action_dim, seed)
    initial = copy.deepcopy(net.state_dict())
    sr0_stats, sr0_steps = evaluate(env, ref, _policy_call(ResidualPolicy(net)), max(2, n_eval // 2))
    ledger.add("validate", sr0_steps)
    sr0 = float(sr0_stats["success_rate"])

    def eval_fn(policy: ResidualPolicy):
        stats, steps = evaluate(env, ref, _policy_call(policy), max(2, n_eval // 2))
        return float(stats["success_rate"]), steps

    probe_curve, _ = train_steps(env, net, opt, probe_steps, eval_fn)
    probe_spent = probe_curve[-1][0] if probe_curve else 0
    ledger.add("probe_train", probe_spent)
    sr_train = float(probe_curve[-1][1]) if probe_curve else sr0
    trained = copy.deepcopy(net.state_dict())

    net.load_state_dict(initial)
    proposals, diag_steps = propose_edits(env, ref, _policy_call(ResidualPolicy(net)))
    ledger.add("probe_edit", diag_steps)
    best_edit = EditParams.identity()
    best_ref = ref
    best_sr = sr0
    edit_steps = diag_steps
    for edit in proposals:
        sr, steps, edited = _score_edit(env, ref, edit, _policy_call(ResidualPolicy(net)), 2)
        ledger.add("probe_edit", steps)
        edit_steps += steps
        if sr > best_sr:
            best_edit, best_ref, best_sr = edit, edited, sr
    action = select_action(sr0, sr_train, probe_spent, best_sr, edit_steps)
    remain = max(0, budget - ledger.total)
    if action == "defer":
        gen = _generalize(env, ref, _policy_call(ResidualPolicy(net)), ledger)
        trace = Trace(ref.id, ref.label)
        return _fit_trace(trace, ledger, sr0, [(ledger.total, sr0)], ["defer"], False, True, None, gen)
    if action == "edit":
        net.load_state_dict(initial)
        env.set_reference(best_ref)
        opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)

        def eval_edit(policy: ResidualPolicy):
            stats, steps = evaluate(env, best_ref, _policy_call(policy), max(2, n_eval // 2))
            return float(stats["success_rate"]), steps

        curve, _ = train_steps(env, net, opt, remain, eval_edit)
        ledger.add("train", curve[-1][0] if curve else 0)
        success = float(curve[-1][1]) if curve else best_sr
        gen = _generalize(env, best_ref, _policy_call(ResidualPolicy(net)), ledger)
        trace = Trace(ref.id, ref.label)
        return _fit_trace(
            trace,
            ledger,
            success,
            [(ledger.total - remain, best_sr)] + [(ledger.total - remain + s, v) for s, v in curve],
            ["probe", "edit", "train"],
            True,
            False,
            best_edit,
            gen,
        )
    net.load_state_dict(trained)
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)

    def eval_train(policy: ResidualPolicy):
        stats, steps = evaluate(env, ref, _policy_call(policy), max(2, n_eval // 2))
        return float(stats["success_rate"]), steps

    curve, _ = train_steps(env, net, opt, remain, eval_train)
    ledger.add("train", curve[-1][0] if curve else 0)
    success = float(curve[-1][1]) if curve else sr_train
    gen = _generalize(env, ref, _policy_call(ResidualPolicy(net)), ledger)
    trace = Trace(ref.id, ref.label)
    return _fit_trace(
        trace,
        ledger,
        success,
        probe_curve + [(probe_spent + s, v) for s, v in curve],
        ["probe", "train"],
        False,
        False,
        None,
        gen,
    )


def reference_features(env: GraspLiftEnv, ref: Reference) -> tuple[np.ndarray, int]:
    diag = _diagnose(env, ref, _zero)
    contact = 0.0 if diag["closure_scale"] > 1.0 else 1.0
    feat = np.array([diag["trans"][0], diag["trans"][1], diag["pitch"], contact], dtype=np.float64)
    return feat, int(diag["steps"])


def apply_editor(ref: Reference, feat: np.ndarray, scales: np.ndarray, hand) -> tuple[Reference, EditParams]:
    scale_t, scale_r, scale_c = [float(v) for v in scales]
    trans = np.clip(scale_t * feat[:2], -0.04, 0.04)
    pitch = float(np.clip(scale_r * feat[2], -0.5, 0.5))
    # feat[3] is 1 when the diagnostic already had contact. Low contact asks for more closure.
    closure = float(np.clip(1.0 + scale_c * (1.0 - feat[3]) * (1.0 / BIAS_CLOSURE_SCALE - 1.0), 0.4, 6.0))
    edit = EditParams(np.array([trans[0], trans[1], 0.0]), np.array([0.0, pitch, 0.0]), closure)
    if _is_identity(edit):
        return ref, edit
    return apply_edit(ref, edit, hand, new_id=ref.id + "/editor"), edit


def run_learned_editor(refs: list[Reference], bounds, budget: int, n_eval: int, seed: int, cem_samples: int, cem_iters: int) -> tuple[list[Trace], dict]:
    """One editor, three scalars, fit by CEM on zero-policy success. Cost is global."""
    if not refs:
        return [], {"editor_steps": 0, "scales": [1.0, 1.0, 1.0]}
    env = GraspLiftEnv(refs[0], bounds, seed=seed)
    ledger_global = Ledger()
    feats = []
    for ref in refs:
        feat, steps = reference_features(env, ref)
        feats.append(feat)
        ledger_global.add("editor_train", steps)
    rng = np.random.default_rng(seed)
    mean = np.array([1.0, 1.0, 1.0])
    std = np.array([0.25, 0.25, 0.35])

    def score(scales: np.ndarray) -> float:
        vals = []
        for ref, feat in zip(refs, feats):
            edited, _edit = apply_editor(ref, feat, scales, env.hand)
            stats, steps = evaluate(env, edited, _zero, 1)
            ledger_global.add("editor_train", steps)
            vals.append(stats["success_rate"])
        return float(np.mean(vals))

    for _ in range(cem_iters):
        samples = rng.normal(mean, std, size=(cem_samples, 3))
        samples = np.clip(samples, 0.0, 2.0)
        scored = sorted(((score(s), s) for s in samples), key=lambda item: item[0])
        elites = np.stack([s for _sr, s in scored[-2:]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 0.05
    editor_steps = ledger_global.total
    amortized = editor_steps / len(refs)
    traces = []
    for index, (ref, feat) in enumerate(zip(refs, feats)):
        edited, edit = apply_editor(ref, feat, mean, env.hand)
        ledger = Ledger()
        ledger.add("editor_train", int(round(amortized)))
        remain = max(0, budget - int(round(amortized)))
        policy, curve, success = _train_budget(
            env, edited, bounds, seed + index, remain, n_eval, ledger, "train"
        )
        curve = [(int(round(amortized)), curve[0][1] if curve else 0.0)] + [
            (int(round(amortized)) + s, v) for s, v in curve
        ]
        gen = _generalize(env, edited, _policy_call(policy), ledger)
        trace = Trace(ref.id, ref.label)
        traces.append(
            _fit_trace(
                trace,
                ledger,
                success,
                curve,
                ["editor", "train"],
                not _is_identity(edit),
                False,
                edit,
                gen,
            )
        )
    return traces, {"editor_steps": editor_steps, "scales": [float(v) for v in mean], "amortized_per_traj": amortized}


def run_sensitivity(ref: Reference, bounds_list: list, budget: int, n_eval: int, seed: int) -> list[dict]:
    rows = []
    for bounds in bounds_list:
        trace = run_train_only(ref, bounds, budget, n_eval, seed)
        rows.append(
            {
                "traj_id": ref.id,
                "label": ref.label,
                "trans_m": bounds.trans_m,
                "rot_rad": bounds.rot_rad,
                "finger_rad": bounds.finger_rad,
                "success": trace.final_success,
                "steps": trace.steps_spent,
            }
        )
    return rows
