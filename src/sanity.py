"""Zero-policy and lead-policy checks that the three reference classes separate."""

from __future__ import annotations

from src.config import BOUNDS_MEDIUM
from src.env.grasp_lift import GraspLiftEnv, run_episode
from src.refs.generate import generate_dataset, lead_action, zero_action


def _rate(ref, act_fn) -> float:
    env = GraspLiftEnv(ref, BOUNDS_MEDIUM, seed=0)
    episodes = [run_episode(env, act_fn, seed=i) for i in range(3)]
    return float(sum(ep["success"] for ep in episodes) / len(episodes))


def run_sanity(n_easy: int = 1, n_hard: int = 1, n_bias: int = 1) -> dict:
    refs = generate_dataset(n_easy, n_hard, n_bias, seed=0)
    by_label: dict[str, dict] = {}
    for ref in refs:
        zero = _rate(ref, zero_action)
        lead = _rate(ref, lambda obs: lead_action(obs, BOUNDS_MEDIUM))
        by_label.setdefault(ref.label, {"zero": [], "lead": []})
        by_label[ref.label]["zero"].append(zero)
        by_label[ref.label]["lead"].append(lead)
    summary = {
        lab: {
            "zero": float(sum(v["zero"]) / len(v["zero"])),
            "lead": float(sum(v["lead"]) / len(v["lead"])),
        }
        for lab, v in by_label.items()
    }
    easy_ok = summary.get("easy", {}).get("zero", 0.0) >= 0.99
    hard_zero = summary.get("hard", {}).get("zero", 1.0) <= 0.01
    hard_lead = summary.get("hard", {}).get("lead", 0.0) >= 0.99
    bias_zero = all(summary.get(lab, {}).get("zero", 1.0) <= 0.01 for lab in ("bias_trans", "bias_rot", "bias_closure"))
    ok = bool(easy_ok and hard_zero and hard_lead and bias_zero)
    return {"ok": ok, "by_label": summary, "n": len(refs)}
