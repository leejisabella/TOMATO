"""Run Particle Swarm Optimization on all 4 crops, multiple seeds.

Saves one OptimizerHistory JSON per (crop, seed) under results/pso/.

Usage:
    python run_pso.py                               # repo defaults
    python run_pso.py --from-best results/best_pso.json
    python run_pso.py --crop Tomato --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from optimization import GreenhouseProblem, ParticleSwarmOptimizer
from surrogate.api import load_predictor


CROPS = ("Tomato", "Cucumber", "Lettuce", "Pepper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10,
                   help="number of random seeds per crop (default: 10)")
    p.add_argument("--crop", type=str, default=None,
                   choices=CROPS, help="run only a single crop")
    p.add_argument("--swarm-size", type=int, default=50)
    p.add_argument("--max-iters", type=int, default=100,
                   help="per-seed iteration budget (default: 100)")
    p.add_argument("--patience-iters", type=int, default=10,
                   help="plateau patience in iterations (default: 10)")
    p.add_argument("--w", type=float, default=0.7)
    p.add_argument("--c1", type=float, default=1.5)
    p.add_argument("--c2", type=float, default=1.5)
    p.add_argument("--velocity-clamp-frac", type=float, default=0.2)
    p.add_argument("--init-velocity-frac", type=float, default=0.1)
    p.add_argument("--init-feasible-ratio", type=float, default=0.5)
    p.add_argument("--repair-mode", choices=["segment", "none"], default="segment")
    p.add_argument("--repair-steps", type=int, default=12)
    p.add_argument("--from-best", type=Path, default=None,
                   help="path to best_pso.json from tune_pso.py; if present, "
                        "overrides PSO hyperparameters above")
    p.add_argument("--out-dir", type=Path, default=Path("results") / "pso")
    return p.parse_args()


def _load_best_config(path: Path) -> dict:
    with path.open("r") as f:
        payload = json.load(f)
    cfg = payload.get("best_combined_config", {})
    if not cfg:
        raise ValueError(f"{path} has empty best_combined_config")
    return cfg


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    optimizer_kwargs = {
        "swarm_size": args.swarm_size,
        "w": args.w,
        "c1": args.c1,
        "c2": args.c2,
        "velocity_clamp_frac": args.velocity_clamp_frac,
        "init_velocity_frac": args.init_velocity_frac,
        "init_feasible_ratio": args.init_feasible_ratio,
        "repair_mode": args.repair_mode,
        "repair_steps": args.repair_steps,
    }
    if args.from_best is not None:
        best_cfg = _load_best_config(args.from_best)
        print(f"Loaded tuned config from {args.from_best}: {best_cfg}")
        optimizer_kwargs.update(best_cfg)
    print(f"Using hyperparameters: {optimizer_kwargs}")

    predictor = load_predictor()
    crops = (args.crop,) if args.crop else CROPS

    t0 = time.perf_counter()
    for crop in crops:
        problem = GreenhouseProblem(crop=crop, predictor=predictor)
        print(f"\n=== {crop}  (dim={problem.dim}, "
              f"n_constraints={len(problem.constraints)}) ===")

        for seed in range(args.seeds):
            t_run = time.perf_counter()
            pso = ParticleSwarmOptimizer(
                problem=problem,
                max_iters=args.max_iters,
                patience_iters=args.patience_iters,
                seed=seed,
                **optimizer_kwargs,
            )
            hist = pso.run()
            out_path = args.out_dir / f"{crop}_seed{seed}.json"
            hist.to_json(out_path)
            best = hist.best_feasible_so_far()
            elapsed = time.perf_counter() - t_run
            iters = len(hist.populations) - 1 if hist.populations is not None else 0
            print(f"  seed {seed:2d}: {iters:3d} iters ({len(hist.ys):4d} evals), "
                  f"best feasible y={best:6.3f}, "
                  f"converged={hist.converged}, {elapsed:5.1f}s "
                  f"→ {out_path.name}")

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
