"""Generate all comparison + diagnostic plots for the optimizer experiments.

Reads JSON histories from results/{bo,ga,pso,cmaes}/ and tuning summaries
from results/tuning_{bo,ga,pso,cmaes}.json (if present), and writes PNGs
into artifacts/plots/.

Plots (matching the approved plan §A–E):
  A. Per-algorithm convergence diagnostics
       1. convergence_<algo>_<crop>.png
       2. feasibility_over_time_<algo>_<crop>.png
       3. violation_<algo>_<crop>.png
  B. Cross-algorithm comparisons
       4. comparison_convergence_<crop>.png
       5. comparison_walltime_<crop>.png
       6. comparison_final_yield.png
       7. comparison_sample_efficiency.png
       8. comparison_convergence_status.png
  C. Hyperparameter tuning
       9. tuning_bo_acquisition.png
      10. tuning_bo_kernel.png
      11. tuning_bo_ninit.png
      12. tuning_ga_popsize.png
      13. tuning_ga_mutation.png
      14. tuning_ga_crossover.png
      15. tuning_pso_*.png
      16. tuning_cmaes_*.png
  D. GA inner-mechanics
      17. ga_snapshots_<crop>.png  (every 100th eval, 3-column grid)
  E. Solution-quality analysis
      18. optimal_config_heatmap.png
      19. cost_breakdown_<crop>.png
      20. sensitivity_<crop>.png

Usage:
    python plot_results.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimization import GreenhouseProblem
from optimization.crop_specs import DECISION_VARIABLES
from optimization.history import OptimizerHistory
from surrogate.api import load_predictor


CROPS = ("Tomato", "Cucumber", "Lettuce", "Pepper")
ALGORITHM_SPECS = (
    ("bo", "BayesianOptimization", "BO", "#1f77b4"),
    ("ga", "GeneticAlgorithm", "GA", "#d62728"),
    ("pso", "ParticleSwarmOptimization", "PSO", "#ff7f0e"),
    ("cmaes", "CMAES", "CMA-ES", "#9467bd"),
)
ALGO_DIR_TO_NAME = {algo_dir: algo_name for algo_dir, algo_name, _, _ in ALGORITHM_SPECS}
ALGO_COLORS = {algo_name: color for _, algo_name, _, color in ALGORITHM_SPECS}
ALGO_SHORT = {algo_name: short for _, algo_name, short, _ in ALGORITHM_SPECS}
ALGO_NAME_TO_DIR = {algo_name: algo_dir for algo_dir, algo_name, _, _ in ALGORITHM_SPECS}


# =============================================================================
# Loading + small helpers
# =============================================================================


def load_histories(results_dir: Path, algo_dir: str, crop: str) -> List[OptimizerHistory]:
    out: List[OptimizerHistory] = []
    folder = results_dir / algo_dir
    if not folder.exists():
        return out
    for path in sorted(folder.glob(f"{crop}_seed*.json")):
        try:
            out.append(OptimizerHistory.from_json(path))
        except Exception as e:
            print(f"  skipping {path.name}: {e}")
    return out


def available_algorithms(results_dir: Path) -> List[Tuple[str, str]]:
    """Algorithms with at least one results directory or tuning file present."""
    available: List[Tuple[str, str]] = []
    for algo_dir, algo_name, _, _ in ALGORITHM_SPECS:
        if (results_dir / algo_dir).exists() or (results_dir / f"tuning_{algo_dir}.json").exists():
            available.append((algo_dir, algo_name))
    return available


def algorithm_iteration_label(algo: str) -> str:
    if algo == "GeneticAlgorithm":
        return "generation"
    if algo in {"ParticleSwarmOptimization", "CMAES"}:
        return "iteration"
    return "evaluation"


def median_iqr(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack ragged curves at common x; return (x, median, q25, q75).

    Curves are forward-filled within each run so trailing NaNs propagate
    the last best-feasible value (or NaN if no feasible was ever found).
    """
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
        # Forward-fill: keep last non-nan
        last = np.nan
        for j in range(max_len):
            if np.isnan(arr[i, j]):
                arr[i, j] = last
            else:
                last = arr[i, j]
    x = np.arange(max_len)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    return x, med, q25, q75


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple right-aligned rolling mean (NaN-safe)."""
    if len(x) == 0:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.nanmean(x[lo : i + 1])
    return out


# =============================================================================
# Section A: per-algorithm convergence diagnostics
# =============================================================================


def plot_convergence(
    histories: List[OptimizerHistory], crop: str, out_dir: Path
) -> None:
    if not histories:
        return
    algo = histories[0].algorithm
    color = ALGO_COLORS.get(algo, "tab:blue")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = []
    for h in histories:
        y = np.asarray(h.best_feasible_y, dtype=float)
        ax.plot(y, color=color, alpha=0.18, linewidth=0.8)
        curves.append(y)

    x, med, q25, q75 = median_iqr(curves)
    ax.plot(x, med, color=color, linewidth=2.4,
            label=f"{ALGO_SHORT.get(algo, algo)} median")
    ax.fill_between(x, q25, q75, color=color, alpha=0.18, label="IQR")

    if algo == "BayesianOptimization":
        ax.set_xlabel("evaluation #")
    else:
        ax.set_xlabel(f"evaluation # (across {algorithm_iteration_label(algo)}s)")
    ax.set_ylabel("best-feasible yield  (kg/m²)")
    ax.set_title(f"{ALGO_SHORT.get(algo, algo)}  —  {crop}  convergence")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"convergence_{ALGO_SHORT.get(algo, algo).lower()}_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_feasibility_rate(
    histories: List[OptimizerHistory], crop: str, out_dir: Path
) -> None:
    if not histories:
        return
    algo = histories[0].algorithm
    color = ALGO_COLORS.get(algo, "tab:blue")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    window = 20
    rates = []
    for h in histories:
        feas = np.asarray(h.feasible, dtype=float)
        roll = rolling_mean(feas, window=window)
        ax.plot(roll, color=color, alpha=0.18)
        rates.append(roll)
    x, med, q25, q75 = median_iqr(rates)
    ax.plot(x, med, color=color, linewidth=2.4, label="median (window=20)")
    ax.fill_between(x, q25, q75, color=color, alpha=0.18)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel(f"rolling-mean feasibility rate  (window={window})")
    ax.set_title(f"{ALGO_SHORT.get(algo, algo)}  —  {crop}  feasibility over time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"feasibility_over_time_{ALGO_SHORT.get(algo, algo).lower()}_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_violation_over_iter(
    histories: List[OptimizerHistory], crop: str, out_dir: Path
) -> None:
    """Max constraint violation of the running incumbent vs iteration."""
    if not histories:
        return
    algo = histories[0].algorithm
    color = ALGO_COLORS.get(algo, "tab:blue")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = []
    for h in histories:
        # Running max-violation of the best-feasible-so-far, treating
        # pre-feasible iterations as the minimum max_violation seen.
        gs = np.asarray([np.max(np.maximum(0.0, g)) for g in h.gs])
        running_min = np.minimum.accumulate(gs)
        ax.plot(running_min, color=color, alpha=0.2, linewidth=0.8)
        curves.append(running_min)
    x, med, q25, q75 = median_iqr(curves)
    ax.plot(x, med, color=color, linewidth=2.4, label="median running min-violation")
    ax.fill_between(x, q25, q75, color=color, alpha=0.18)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("running min  max(0, g)  (violation magnitude)")
    ax.set_title(f"{ALGO_SHORT.get(algo, algo)}  —  {crop}  constraint violation")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"violation_{ALGO_SHORT.get(algo, algo).lower()}_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


# =============================================================================
# Section B: cross-algorithm comparisons
# =============================================================================


def plot_comparison_convergence(
    histories_by_algo: Dict[str, List[OptimizerHistory]],
    crop: str,
    out_dir: Path,
) -> None:
    present = [(algo, hs) for algo, hs in histories_by_algo.items() if hs]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, hists in present:
        curves = [np.asarray(h.best_feasible_y, dtype=float) for h in hists]
        x, med, q25, q75 = median_iqr(curves)
        col = ALGO_COLORS[algo]
        ax.plot(x, med, color=col, linewidth=2.4,
                label=f"{ALGO_SHORT[algo]} median")
        ax.fill_between(x, q25, q75, color=col, alpha=0.15)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("best-feasible yield (kg/m²)")
    ax.set_title(f"Cross-algorithm best-feasible yield  ({crop})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"comparison_convergence_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_comparison_walltime(
    histories_by_algo: Dict[str, List[OptimizerHistory]],
    crop: str,
    out_dir: Path,
) -> None:
    present = [(algo, hs) for algo, hs in histories_by_algo.items() if hs]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, hists in present:
        col = ALGO_COLORS[algo]
        for h in hists:
            t = np.asarray(h.wall_time_s)
            y = np.asarray(h.best_feasible_y, dtype=float)
            ax.plot(t, y, color=col, alpha=0.4, linewidth=1.0)
        ax.plot([], [], color=col, label=ALGO_SHORT[algo])  # legend handle
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("best-feasible yield (kg/m²)")
    ax.set_title(f"Cross-algorithm wall-clock convergence  ({crop})")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"comparison_walltime_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_final_yield_bars(
    results_dir: Path, out_dir: Path,
) -> None:
    """Per-crop grouped bar chart, one bar per available algorithm."""
    present_algos = [
        algo_name
        for algo_dir, algo_name in available_algorithms(results_dir)
        if any(load_histories(results_dir, algo_dir, crop) for crop in CROPS)
    ]
    if not present_algos:
        return

    means: Dict[str, Dict[str, float]] = {ALGO_SHORT[a]: {} for a in present_algos}
    stds: Dict[str, Dict[str, float]] = {ALGO_SHORT[a]: {} for a in present_algos}
    for crop in CROPS:
        for algo_dir, algo_full in available_algorithms(results_dir):
            hs = load_histories(results_dir, algo_dir, crop)
            if not hs:
                continue
            ys = np.array([h.best_feasible_so_far() for h in hs])
            ys = ys[np.isfinite(ys)]
            if len(ys) == 0:
                continue
            means[ALGO_SHORT[algo_full]][crop] = float(np.mean(ys))
            stds[ALGO_SHORT[algo_full]][crop] = float(np.std(ys))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CROPS))
    width = 0.8 / len(present_algos)
    for i, algo_full in enumerate(present_algos):
        algo = ALGO_SHORT[algo_full]
        vals = [means[algo].get(c, np.nan) for c in CROPS]
        errs = [stds[algo].get(c, 0.0) for c in CROPS]
        col = ALGO_COLORS[algo_full]
        offset = (i - (len(present_algos) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
               color=col, alpha=0.85, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(CROPS)
    ax.set_ylabel("final best-feasible yield (kg/m²)")
    ax.set_title("Final yield per crop  (mean ± std across seeds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "comparison_final_yield.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_sample_efficiency(
    results_dir: Path, out_dir: Path, fraction: float = 0.95,
) -> None:
    """# evals to reach `fraction` of best-found yield, per crop × algo."""
    present_algos = [
        algo_name
        for algo_dir, algo_name in available_algorithms(results_dir)
        if any(load_histories(results_dir, algo_dir, crop) for crop in CROPS)
    ]
    if not present_algos:
        return

    data: Dict[str, Dict[str, List[int]]] = {ALGO_SHORT[a]: {} for a in present_algos}
    for crop in CROPS:
        all_finals: List[float] = []
        for algo_dir, algo_full in available_algorithms(results_dir):
            for h in load_histories(results_dir, algo_dir, crop):
                v = h.best_feasible_so_far()
                if np.isfinite(v):
                    all_finals.append(v)
        if not all_finals:
            continue
        target = fraction * max(all_finals)
        for algo_dir, algo_full in available_algorithms(results_dir):
            evals_to_target: List[int] = []
            for h in load_histories(results_dir, algo_dir, crop):
                y = np.asarray(h.best_feasible_y, dtype=float)
                reached = np.where(y >= target)[0]
                if len(reached):
                    evals_to_target.append(int(reached[0]))
                else:
                    evals_to_target.append(len(y))  # never reached → penalize w/ full budget
            data[ALGO_SHORT[algo_full]][crop] = evals_to_target

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CROPS))
    width = 0.8 / len(present_algos)
    for i, algo_full in enumerate(present_algos):
        algo = ALGO_SHORT[algo_full]
        means = [np.mean(data[algo].get(c, [np.nan])) for c in CROPS]
        stds = [np.std(data[algo].get(c, [0])) for c in CROPS]
        col = ALGO_COLORS[algo_full]
        offset = (i - (len(present_algos) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=col, alpha=0.85, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(CROPS)
    ax.set_ylabel(f"# evaluations to reach {fraction:.0%} of best-found")
    ax.set_title(f"Sample efficiency  (lower is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "comparison_sample_efficiency.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_convergence_status(
    results_dir: Path, out_dir: Path,
) -> None:
    """Stacked bar: fraction of seeds that converged (plateau) vs hit-cap."""
    present_algos = [
        algo_name
        for algo_dir, algo_name in available_algorithms(results_dir)
        if any(load_histories(results_dir, algo_dir, crop) for crop in CROPS)
    ]
    if not present_algos:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CROPS))
    width = 0.8 / len(present_algos)
    for i, algo_full in enumerate(present_algos):
        algo_dir = ALGO_NAME_TO_DIR[algo_full]
        conv = []
        cap = []
        for crop in CROPS:
            hs = load_histories(results_dir, algo_dir, crop)
            n = len(hs) or 1
            n_conv = sum(1 for h in hs if h.converged)
            conv.append(n_conv / n)
            cap.append(1 - n_conv / n)
        col = ALGO_COLORS[algo_full]
        offset = (i - (len(present_algos) - 1) / 2.0) * width
        ax.bar(x + offset, conv, width,
               color=col, alpha=0.85, label=f"{ALGO_SHORT[algo_full]} plateau")
        ax.bar(x + offset, cap, width, bottom=conv,
               color=col, alpha=0.35, label=f"{ALGO_SHORT[algo_full]} hit cap",
               hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(CROPS)
    ax.set_ylabel("fraction of seeds")
    ax.set_title("Convergence status  (plateau-stop vs hit-cap)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "comparison_convergence_status.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


# =============================================================================
# Section C: hyperparameter tuning
# =============================================================================


def _tuning_plot(sweep_name: str, sweep_data: List[dict],
                 algo_color: str, title: str, out_path: Path,
                 label_keys: List[str]) -> None:
    """Generic tuning plot: convergence curves, one per config."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    n = len(sweep_data)
    for i, entry in enumerate(sweep_data):
        cfg = entry["config"]
        runs = entry["runs"]
        curves = [np.asarray(r["best_feasible_y"], dtype=float) for r in runs]
        if not curves:
            continue
        x, med, q25, q75 = median_iqr(curves)
        col = cmap(i / max(n - 1, 1))
        label_parts = [f"{k}={cfg[k]}" for k in label_keys if k in cfg]
        label = ", ".join(label_parts) if label_parts else str(cfg)
        ax.plot(x, med, color=col, linewidth=2.0, label=label)
        ax.fill_between(x, q25, q75, color=col, alpha=0.15)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("best-feasible yield (kg/m²)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_tuning_bo(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "tuning_bo.json"
    if not path.exists():
        return
    with path.open("r") as f:
        data = json.load(f)
    sweeps = data["sweeps"]
    if "kernel" in sweeps:
        _tuning_plot("kernel", sweeps["kernel"], ALGO_COLORS["BayesianOptimization"],
                     "BO — kernel sweep", out_dir / "tuning_bo_kernel.png",
                     ["kernel"])
    if "acquisition" in sweeps:
        _tuning_plot("acquisition", sweeps["acquisition"],
                     ALGO_COLORS["BayesianOptimization"],
                     "BO — acquisition function sweep",
                     out_dir / "tuning_bo_acquisition.png",
                     ["acquisition", "kappa"])
    if "n_init" in sweeps:
        _tuning_plot("n_init", sweeps["n_init"],
                     ALGO_COLORS["BayesianOptimization"],
                     "BO — initial-design size (n_init) sweep",
                     out_dir / "tuning_bo_ninit.png", ["n_init"])
    if "ucb_kappa" in sweeps:
        _tuning_plot("ucb_kappa", sweeps["ucb_kappa"],
                     ALGO_COLORS["BayesianOptimization"],
                     "BO — UCB κ sweep",
                     out_dir / "tuning_bo_ucb_kappa.png", ["kappa"])


def plot_tuning_ga(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "tuning_ga.json"
    if not path.exists():
        return
    with path.open("r") as f:
        data = json.load(f)
    sweeps = data["sweeps"]
    if "pop_size" in sweeps:
        _tuning_plot("pop_size", sweeps["pop_size"],
                     ALGO_COLORS["GeneticAlgorithm"],
                     "GA — population size sweep",
                     out_dir / "tuning_ga_popsize.png", ["pop_size"])
    if "pc" in sweeps:
        _tuning_plot("pc", sweeps["pc"], ALGO_COLORS["GeneticAlgorithm"],
                     "GA — crossover probability sweep",
                     out_dir / "tuning_ga_crossover.png", ["pc"])
    # Combined mutation plot: pm + eta_m on same panel layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, label_keys, title in (
        (axes[0], "pm", ["pm"], "pm sweep (mutation probability)"),
        (axes[1], "eta_m", ["eta_m"], "η_m sweep (mutation distribution)"),
    ):
        if key not in sweeps:
            continue
        cmap = plt.get_cmap("viridis")
        entries = sweeps[key]
        for i, entry in enumerate(entries):
            curves = [np.asarray(r["best_feasible_y"], dtype=float)
                      for r in entry["runs"]]
            if not curves:
                continue
            x, med, q25, q75 = median_iqr(curves)
            col = cmap(i / max(len(entries) - 1, 1))
            cfg = entry["config"]
            label = ", ".join(f"{k}={cfg[k]:.3g}" for k in label_keys if k in cfg)
            ax.plot(x, med, color=col, linewidth=2.0, label=label)
            ax.fill_between(x, q25, q75, color=col, alpha=0.15)
        ax.set_xlabel("evaluation #")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("best-feasible yield (kg/m²)")
    fig.tight_layout()
    fname = out_dir / "tuning_ga_mutation.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")

    if "eta_c" in sweeps:
        _tuning_plot("eta_c", sweeps["eta_c"], ALGO_COLORS["GeneticAlgorithm"],
                     "GA — SBX η_c (crossover distribution) sweep",
                     out_dir / "tuning_ga_eta_c.png", ["eta_c"])


def plot_tuning_pso(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "tuning_pso.json"
    if not path.exists():
        return
    with path.open("r") as f:
        data = json.load(f)
    sweeps = data["sweeps"]
    if "swarm_size" in sweeps:
        _tuning_plot("swarm_size", sweeps["swarm_size"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — swarm size sweep",
                     out_dir / "tuning_pso_swarm_size.png", ["swarm_size"])
    if "w" in sweeps:
        _tuning_plot("w", sweeps["w"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — inertia weight sweep",
                     out_dir / "tuning_pso_w.png", ["w"])
    if "c1" in sweeps:
        _tuning_plot("c1", sweeps["c1"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — cognitive coefficient sweep",
                     out_dir / "tuning_pso_c1.png", ["c1"])
    if "c2" in sweeps:
        _tuning_plot("c2", sweeps["c2"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — social coefficient sweep",
                     out_dir / "tuning_pso_c2.png", ["c2"])
    if "velocity_clamp_frac" in sweeps:
        _tuning_plot("velocity_clamp_frac", sweeps["velocity_clamp_frac"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — velocity clamp sweep",
                     out_dir / "tuning_pso_velocity_clamp.png",
                     ["velocity_clamp_frac"])
    if "repair_mode" in sweeps:
        _tuning_plot("repair_mode", sweeps["repair_mode"],
                     ALGO_COLORS["ParticleSwarmOptimization"],
                     "PSO — repair mode sweep",
                     out_dir / "tuning_pso_repair.png", ["repair_mode"])


def plot_tuning_cmaes(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "tuning_cmaes.json"
    if not path.exists():
        return
    with path.open("r") as f:
        data = json.load(f)
    sweeps = data["sweeps"]
    if "pop_size" in sweeps:
        _tuning_plot("pop_size", sweeps["pop_size"],
                     ALGO_COLORS["CMAES"],
                     "CMA-ES — population size sweep",
                     out_dir / "tuning_cmaes_pop_size.png", ["pop_size"])
    if "sigma0" in sweeps:
        _tuning_plot("sigma0", sweeps["sigma0"],
                     ALGO_COLORS["CMAES"],
                     "CMA-ES — initial step size sweep",
                     out_dir / "tuning_cmaes_sigma0.png", ["sigma0"])
    if "init_mode" in sweeps:
        _tuning_plot("init_mode", sweeps["init_mode"],
                     ALGO_COLORS["CMAES"],
                     "CMA-ES — initialization mode sweep",
                     out_dir / "tuning_cmaes_init_mode.png", ["init_mode"])
    if "repair_mode" in sweeps:
        _tuning_plot("repair_mode", sweeps["repair_mode"],
                     ALGO_COLORS["CMAES"],
                     "CMA-ES — repair mode sweep",
                     out_dir / "tuning_cmaes_repair.png", ["repair_mode"])


# =============================================================================
# Section D: GA inner-mechanics snapshots
# =============================================================================


def plot_ga_snapshots(
    results_dir: Path, crop: str, out_dir: Path,
    snapshot_eval_step: int = 100,
    n_snapshots: int = 5,
    seed: int = 0,
) -> None:
    """Snapshot the GA every `snapshot_eval_step` evaluations.

    Loads results/ga/<crop>_seed<seed>.json (default seed=0). Picks
    `n_snapshots` generations evenly spread across the run and plots each
    as a row of three panels: candidates, parents, performance.

    The 2-D projection is fixed at (light_intensity_lux, photoperiod_hours)
    — the two strongest electricity drivers.
    """
    path = results_dir / "ga" / f"{crop}_seed{seed}.json"
    if not path.exists():
        return
    hist = OptimizerHistory.from_json(path)
    if hist.populations is None or len(hist.populations) < 2:
        return

    pop_size = len(hist.populations[0])
    n_gens = len(hist.populations)

    # Generation step matching the eval cadence (1 gen = pop_size evals)
    gens_per_step = max(1, snapshot_eval_step // pop_size)
    candidate_gens = list(range(0, n_gens, gens_per_step))
    if candidate_gens[-1] != n_gens - 1:
        candidate_gens.append(n_gens - 1)
    if len(candidate_gens) > n_snapshots:
        idx = np.linspace(0, len(candidate_gens) - 1, n_snapshots).astype(int)
        candidate_gens = [candidate_gens[i] for i in idx]

    var_x = DECISION_VARIABLES.index("light_intensity_lux")
    var_y = DECISION_VARIABLES.index("photoperiod_hours")

    fig, axes = plt.subplots(
        nrows=len(candidate_gens), ncols=3,
        figsize=(13.5, 3.2 * len(candidate_gens)),
        squeeze=False,
    )
    fig.suptitle(
        f"GA inner mechanics  —  {crop} (seed={seed}, pop={pop_size})\n"
        f"projection: x = light_intensity_lux,  y = photoperiod_hours",
        fontsize=11, y=0.998,
    )

    # Determine overall y-range for the 'performance' column from the
    # running best-feasible curve.
    bfy = np.asarray(hist.best_feasible_y, dtype=float)

    for row, gen in enumerate(candidate_gens):
        pop = hist.populations[gen]
        fit = hist.fitnesses[gen]
        feas = hist.feasibility_masks[gen]
        # parent_indices[0] is empty (no parents at gen 0); from gen 1 on,
        # it points at gen-1's individuals that produced gen.
        if gen >= 1 and hist.parent_indices is not None:
            parent_idx = hist.parent_indices[gen]
            # Plot parents projected on the PREVIOUS generation's population
            prev_pop = hist.populations[gen - 1]
            prev_feas = hist.feasibility_masks[gen - 1]
        else:
            parent_idx = np.array([], dtype=int)
            prev_pop = pop
            prev_feas = feas

        eval_count = (gen + 1) * pop_size

        # ----- col 1: candidates (this generation) -----
        ax = axes[row, 0]
        feas_pts = pop[feas]
        infeas_pts = pop[~feas]
        if len(feas_pts) > 0:
            sc = ax.scatter(feas_pts[:, var_x], feas_pts[:, var_y],
                            c=fit[feas], cmap="viridis", s=40, marker="o",
                            edgecolor="k", linewidth=0.3)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                         label="yield" if row == 0 else "")
        if len(infeas_pts) > 0:
            ax.scatter(infeas_pts[:, var_x], infeas_pts[:, var_y],
                       facecolor="none", edgecolor="grey", s=40, marker="x",
                       linewidth=0.8, label="infeasible" if row == 0 else None)
        ax.set_title(f"gen {gen}  candidates" if row == 0 else f"gen {gen}")
        ax.grid(True, alpha=0.3)
        if row == len(candidate_gens) - 1:
            ax.set_xlabel("light_intensity_lux")
        ax.set_ylabel("photoperiod_hours")

        # ----- col 2: parents (gen-1 individuals that produced this gen) -----
        ax = axes[row, 1]
        # Faded full previous population
        ax.scatter(prev_pop[:, var_x], prev_pop[:, var_y],
                   color="lightgrey", s=25, alpha=0.5, edgecolor="none",
                   label="all gen-1" if row == 0 else None)
        if len(parent_idx) > 0:
            picked = prev_pop[parent_idx]
            picked_feas = prev_feas[parent_idx]
            ax.scatter(picked[picked_feas, var_x], picked[picked_feas, var_y],
                       color="red", s=35, marker="o", edgecolor="k",
                       linewidth=0.3, alpha=0.8,
                       label="parents (feas)" if row == 0 else None)
            ax.scatter(picked[~picked_feas, var_x], picked[~picked_feas, var_y],
                       color="red", s=35, marker="x", linewidth=0.9,
                       label="parents (infeas)" if row == 0 else None)
        ax.set_title("tournament-selected parents" if row == 0 else "")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7)
        if row == len(candidate_gens) - 1:
            ax.set_xlabel("light_intensity_lux")

        # ----- col 3: performance to date -----
        ax = axes[row, 2]
        ax.plot(bfy[: eval_count], color=ALGO_COLORS["GeneticAlgorithm"],
                linewidth=1.6)
        ax.axvline(eval_count, color="k", linewidth=0.8, linestyle="--",
                   alpha=0.6)
        ax.set_xlim(0, len(bfy))
        if np.any(np.isfinite(bfy)):
            ax.set_ylim(np.nanmin(bfy) - 0.5, np.nanmax(bfy) + 0.5)
        ax.set_title("best-feasible y vs evals" if row == 0 else "")
        ax.grid(True, alpha=0.3)
        if row == len(candidate_gens) - 1:
            ax.set_xlabel("evaluation #")
        ax.set_ylabel("yield (kg/m²)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fname = out_dir / f"ga_snapshots_{crop}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  → {fname.name}")


# =============================================================================
# Section E: solution-quality analysis
# =============================================================================


def _best_x_summary(results_dir: Path, algo_dir: str, crop: str) -> Optional[Tuple[np.ndarray, dict]]:
    hs = load_histories(results_dir, algo_dir, crop)
    if not hs:
        return None
    best_y = -np.inf
    best = None
    for h in hs:
        y = h.best_feasible_so_far()
        if np.isfinite(y) and y > best_y and h.final_x is not None:
            best_y = y
            best = (h.final_x, h.final_summary)
    return best


def plot_optimal_config_heatmap(results_dir: Path, out_dir: Path) -> None:
    """Normalize each variable into [0,1] within its bounds; show as heatmap."""
    predictor = load_predictor()
    rows: List[Tuple[str, np.ndarray]] = []
    row_labels: List[str] = []
    for crop in CROPS:
        problem = GreenhouseProblem(crop=crop, predictor=predictor)
        lo, hi = problem.bounds()
        for algo_dir, algo_full in available_algorithms(results_dir):
            best = _best_x_summary(results_dir, algo_dir, crop)
            if best is None:
                continue
            x, _ = best
            norm = (x - lo) / np.where(hi - lo > 1e-9, hi - lo, 1.0)
            rows.append((f"{crop} ({ALGO_SHORT[algo_full]})", norm))

    if not rows:
        return
    M = np.array([r[1] for r in rows])
    row_labels = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(13, max(3, 0.45 * len(rows))))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(DECISION_VARIABLES)))
    ax.set_xticklabels(DECISION_VARIABLES, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="normalized within bounds")
    ax.set_title("Best decision vector per (crop, algorithm)  "
                 "[normalized to box bounds]")
    fig.tight_layout()
    fname = out_dir / "optimal_config_heatmap.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_cost_breakdown(results_dir: Path, out_dir: Path) -> None:
    """For each crop: stacked bar showing $/m² breakdown for each algorithm's best solution."""
    present_algos = [
        algo_name
        for algo_dir, algo_name in available_algorithms(results_dir)
        if any(load_histories(results_dir, algo_dir, crop) for crop in CROPS)
    ]
    if not present_algos:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(CROPS))
    width = 0.8 / len(present_algos)

    # We'll need cost breakdowns. Recompute via problem.cost_breakdown_usd
    # so we don't depend on whatever was saved.
    predictor = load_predictor()

    for i, algo_full in enumerate(present_algos):
        algo_dir = ALGO_NAME_TO_DIR[algo_full]
        elec = []
        water = []
        fert = []
        for crop in CROPS:
            best = _best_x_summary(results_dir, algo_dir, crop)
            if best is None:
                elec.append(0); water.append(0); fert.append(0)
                continue
            x_best, summary = best
            problem = GreenhouseProblem(crop=crop, predictor=predictor)
            bd = problem.cost_breakdown_usd(x_best)
            # Convert farm-total $ to $/m²
            area = problem.area_m2
            elec.append(bd["electricity"] / area)
            water.append(bd["water"] / area)
            fert.append(bd["fertilizer"] / area)
        offset = (i - (len(present_algos) - 1) / 2.0) * width
        base = ALGO_COLORS[algo_full]
        ax.bar(x + offset, elec, width, label=f"{ALGO_SHORT[algo_full]} elec",
               color=base, alpha=0.85)
        ax.bar(x + offset, water, width, bottom=elec,
               label=f"{ALGO_SHORT[algo_full]} water",
               color=base, alpha=0.55, hatch="//")
        ax.bar(x + offset, fert, width,
               bottom=np.array(elec) + np.array(water),
               label=f"{ALGO_SHORT[algo_full]} fert",
               color=base, alpha=0.35, hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(CROPS)
    ax.set_ylabel("operating cost ($ / m² / cycle)")
    ax.set_title("Cost breakdown of best feasible solution per crop")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "cost_breakdown.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_sensitivity(
    results_dir: Path, crop: str, out_dir: Path,
    pert_fraction: float = 0.10,
) -> None:
    """One-at-a-time ±10% range perturbations around the best feasible solution
    for `crop`. Returns yield change vs perturbation per decision variable."""
    predictor = load_predictor()
    problem = GreenhouseProblem(crop=crop, predictor=predictor)
    lo, hi = problem.bounds()

    # Pick best across all available algorithms so the sensitivity is
    # anchored at the strongest candidate found overall.
    best_overall: Tuple[Optional[np.ndarray], float] = (None, -np.inf)
    for algo_dir, _ in available_algorithms(results_dir):
        b = _best_x_summary(results_dir, algo_dir, crop)
        if b is None:
            continue
        x, summary = b
        y = summary.get("yield_kg_per_m2", -np.inf)
        if np.isfinite(y) and y > best_overall[1]:
            best_overall = (x, float(y))
    x_star, y_star = best_overall
    if x_star is None:
        return

    spans = (hi - lo) * pert_fraction
    deltas = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # multiples of span
    n_vars = len(DECISION_VARIABLES)
    yield_grid = np.zeros((n_vars, len(deltas)))

    for j, var_name in enumerate(DECISION_VARIABLES):
        for k, d in enumerate(deltas):
            x_p = x_star.copy()
            x_p[j] = np.clip(x_p[j] + d * spans[j], lo[j], hi[j])
            yield_grid[j, k] = problem.objective(x_p)

    delta_pct = deltas * pert_fraction * 100

    # Bar chart of max yield range per variable, sorted descending
    sensitivity = yield_grid.max(axis=1) - yield_grid.min(axis=1)
    order = np.argsort(-sensitivity)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(range(n_vars), sensitivity[order],
            color="#1f77b4", alpha=0.85)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels([DECISION_VARIABLES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel(f"yield range across ±{pert_fraction*100:.0f}% box perturbation "
                  f"(kg/m²)")
    ax.set_title(f"Local sensitivity of best solution  ({crop}, y*={y_star:.2f} kg/m²)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"sensitivity_{crop}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


# =============================================================================
# Top-level
# =============================================================================


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "plots")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="skip Section C if you haven't run the tune_*.py scripts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    algos = available_algorithms(args.results)

    print("== Section A: per-algorithm convergence ==")
    for crop in CROPS:
        for algo_dir, _ in algos:
            hs = load_histories(args.results, algo_dir, crop)
            plot_convergence(hs, crop, args.out)
            plot_feasibility_rate(hs, crop, args.out)
            plot_violation_over_iter(hs, crop, args.out)

    print("\n== Section B: cross-algorithm comparisons ==")
    for crop in CROPS:
        histories_by_algo = {
            algo_name: load_histories(args.results, algo_dir, crop)
            for algo_dir, algo_name in algos
        }
        plot_comparison_convergence(histories_by_algo, crop, args.out)
        plot_comparison_walltime(histories_by_algo, crop, args.out)
    plot_final_yield_bars(args.results, args.out)
    plot_sample_efficiency(args.results, args.out)
    plot_convergence_status(args.results, args.out)

    if not args.skip_tuning:
        print("\n== Section C: hyperparameter tuning ==")
        plot_tuning_bo(args.results, args.out)
        plot_tuning_ga(args.results, args.out)
        plot_tuning_pso(args.results, args.out)
        plot_tuning_cmaes(args.results, args.out)

    print("\n== Section D: GA inner mechanics ==")
    for crop in CROPS:
        plot_ga_snapshots(args.results, crop, args.out)

    print("\n== Section E: solution-quality analysis ==")
    try:
        plot_optimal_config_heatmap(args.results, args.out)
        plot_cost_breakdown(args.results, args.out)
        for crop in CROPS:
            plot_sensitivity(args.results, crop, args.out)
    except ModuleNotFoundError as e:
        print(f"  skipping Section E because a model dependency is unavailable: {e}")

    print(f"\nAll plots saved to: {args.out}")


if __name__ == "__main__":
    main()
