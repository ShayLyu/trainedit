"""Run the reduced or full v1 budget-allocation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import (
    BOUNDS_LARGE,
    BOUNDS_MEDIUM,
    BOUNDS_SMALL,
    CEM_ITERS,
    CEM_SAMPLES,
    SCALE_FULL,
    SCALE_REDUCED,
)
from src.decide.methods import (
    run_adaptive,
    run_edit_then_train,
    run_fixed_ratio,
    run_learned_editor,
    run_sensitivity,
    run_train_only,
    run_uniform_preprocess,
)
from src.eval.gate import assess_gate, hard_clean_mistakes, summarize_traces
from src.refs.generate import generate_dataset
from src.sanity import run_sanity


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v1 reference-adaptation budget experiment")
    parser.add_argument("--scale", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--skip-allocation", action="store_true")
    args = parser.parse_args()
    scale = SCALE_REDUCED if args.scale == "reduced" else SCALE_FULL
    torch.set_num_threads(1)
    out_dir = Path("results")
    sanity = run_sanity(n_easy=1, n_hard=1, n_bias=1)
    print("sanity", json.dumps(sanity, ensure_ascii=False))
    if not sanity["ok"]:
        _dump(out_dir / "v1_summary.json", {"sanity": sanity, "stopped": True})
        raise SystemExit("sanity check failed: the three reference classes do not separate")
    if args.sanity_only:
        _dump(out_dir / "v1_summary.json", {"sanity": sanity})
        return

    refs = generate_dataset(scale.n_easy, scale.n_hard, scale.n_bias, seed=scale.seed)
    print(f"dataset {len(refs)} trajectories, budget {scale.budget}")
    train_traces = []
    edit_traces = []
    for index, ref in enumerate(refs):
        print(f"train-only {ref.id}", flush=True)
        train_traces.append(
            run_train_only(ref, BOUNDS_MEDIUM, scale.budget, scale.eval_episodes, scale.seed + index)
        )
        print(f"  success {train_traces[-1].final_success:.2f} steps {train_traces[-1].steps_spent}")
        print(f"edit-then-train {ref.id}", flush=True)
        edit_traces.append(
            run_edit_then_train(ref, BOUNDS_MEDIUM, scale.budget, scale.eval_episodes, scale.seed + index)
        )
        print(f"  success {edit_traces[-1].final_success:.2f} edited {edit_traces[-1].edited}")

    records = [
        {
            "label": train.label,
            "success_train": train.final_success,
            "success_edit": edit.final_success,
        }
        for train, edit in zip(train_traces, edit_traces)
    ]
    gate = assess_gate(records)
    print("gate", gate["reason"], "proceed" , gate["proceed"])

    sensitivity = []
    for ref in refs:
        if ref.label in ("hard", "bias_trans") and ref.id.endswith("00"):
            print(f"sensitivity {ref.id}", flush=True)
            sensitivity.extend(
                run_sensitivity(
                    ref,
                    [BOUNDS_SMALL, BOUNDS_MEDIUM, BOUNDS_LARGE],
                    scale.sensitivity_budget,
                    max(2, scale.eval_episodes // 2),
                    scale.seed,
                )
            )

    payload = {
        "scale": scale.name,
        "budget": scale.budget,
        "sanity": sanity,
        "gate": gate,
        "train_only": summarize_traces(train_traces),
        "edit_then_train": summarize_traces(edit_traces),
        "sensitivity": sensitivity,
        "train_only_traces": [t.to_dict() for t in train_traces],
        "edit_then_train_traces": [t.to_dict() for t in edit_traces],
    }
    if gate["proceed"] and not args.skip_allocation:
        methods = {}
        uniform, fixed, adaptive = [], [], []
        for index, ref in enumerate(refs):
            print(f"uniform {ref.id}", flush=True)
            uniform.append(
                run_uniform_preprocess(ref, BOUNDS_MEDIUM, scale.budget, scale.eval_episodes, scale.seed + index)
            )
            print(f"fixed {ref.id}", flush=True)
            fixed.append(
                run_fixed_ratio(ref, BOUNDS_MEDIUM, scale.budget, scale.eval_episodes, scale.seed + index)
            )
            print(f"adaptive {ref.id}", flush=True)
            adaptive.append(
                run_adaptive(
                    ref,
                    BOUNDS_MEDIUM,
                    scale.budget,
                    scale.eval_episodes,
                    scale.seed + index,
                    scale.probe_train_steps,
                )
            )
        print("learned editor", flush=True)
        editor, editor_info = run_learned_editor(
            refs,
            BOUNDS_MEDIUM,
            scale.budget,
            scale.eval_episodes,
            scale.seed,
            CEM_SAMPLES,
            CEM_ITERS,
        )
        methods = {
            "uniform_preprocess": summarize_traces(uniform),
            "fixed_ratio": summarize_traces(fixed),
            "adaptive": summarize_traces(adaptive),
            "learned_editor": summarize_traces(editor),
            "learned_editor_fit": editor_info,
            "hard_clean_mistakes": hard_clean_mistakes(adaptive, train_traces),
        }
        payload["methods"] = methods
        payload["adaptive_traces"] = [t.to_dict() for t in adaptive]
        payload["uniform_traces"] = [t.to_dict() for t in uniform]
        payload["fixed_traces"] = [t.to_dict() for t in fixed]
        payload["editor_traces"] = [t.to_dict() for t in editor]
    _dump(out_dir / "v1_summary.json", payload)
    print("wrote", out_dir / "v1_summary.json")


if __name__ == "__main__":
    main()
