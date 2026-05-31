""" Particle Swarm with Deb-style constraint handling.

Implementation follows Kochenderfer & Wheeler, *Algorithms for
Optimization* — Chapter 9 ("Population Methods"). 
The velocity update follows the canonical PSO rule

    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
    x <- x + v

while feasibility is handled using the same rule this repo already uses
for the genetic algorithm:
Constraint handling uses Deb's constrained-domination tournament:

    Deb (2000): "An efficient constraint handling method for
    genetic algorithms"

which orders individuals by:
  1. feasible beats infeasible
  2. among feasible points, higher objective beats lower
  3. among infeasible points, lower max-violation beats higher

Crossover and mutation are the standard Deb operators:
  - Simulated Binary Crossover (Deb & Agrawal 1995)
  - Polynomial Mutation (Deb & Goyal 1996)
Both respect the per-variable box bounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from optimization.history import OptimizerHistory
from optimization.problem import GreenhouseProblem


@dataclass
class Particle:
    """One PSO particle: current state plus its personal best."""

    x: np.ndarray
    v: np.ndarray
    x_best: np.ndarray
    y_best: float
    max_viol_best: float


def deb_feasibility_better(
    y_a: float,
    max_viol_a: float,
    y_b: float,
    max_viol_b: float,
    tol: float = 1e-9,
) -> bool:
    """Return True if candidate A is Deb-better than candidate B."""
    a_feas = max_viol_a <= tol
    b_feas = max_viol_b <= tol

    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    if a_feas and b_feas:
        return y_a >= y_b
    return max_viol_a <= max_viol_b


@dataclass
class ParticleSwarmOptimizer:
    problem: GreenhouseProblem
    swarm_size: int = 50
    max_iters: int = 100
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    velocity_clamp_frac: Optional[float] = 0.2
    init_velocity_frac: float = 0.1
    init_feasible_ratio: float = 0.5
    repair_mode: str = "segment"   # "segment" | "none"
    repair_steps: int = 12
    patience_iters: int = 10
    eps_rel: float = 1e-4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.dim = self.problem.dim
        self.lo, self.hi = self.problem.bounds()
        self.span = self.hi - self.lo
        self._rng = np.random.default_rng(self.seed)
        if self.swarm_size <= 0:
            raise ValueError("swarm_size must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.repair_mode not in {"segment", "none"}:
            raise ValueError("repair_mode must be 'segment' or 'none'")
        if self.velocity_clamp_frac is not None and self.velocity_clamp_frac <= 0.0:
            raise ValueError("velocity_clamp_frac must be positive when provided")
        if not (0.0 <= self.init_feasible_ratio <= 1.0):
            raise ValueError("init_feasible_ratio must be in [0, 1]")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _sample_position(self) -> np.ndarray:
        n_feasible = int(round(self.swarm_size * self.init_feasible_ratio))
        if getattr(self, "_n_initialized", 0) < n_feasible:
            try:
                x = self.problem.sample_random_feasible(rng=self._rng)
            except RuntimeError:
                x = self.problem.sample_random(rng=self._rng)
        else:
            x = self.problem.sample_random(rng=self._rng)
        self._n_initialized = getattr(self, "_n_initialized", 0) + 1
        return np.asarray(x, dtype=float)

    def _sample_velocity(self) -> np.ndarray:
        scale = self.init_velocity_frac * self.span
        v = self._rng.uniform(-scale, scale)
        return self._clip_velocity(v)

    def _clip_velocity(self, v: np.ndarray) -> np.ndarray:
        if self.velocity_clamp_frac is None:
            return np.asarray(v, dtype=float)
        vmax = self.velocity_clamp_frac * self.span
        return np.clip(np.asarray(v, dtype=float), -vmax, vmax)

    def _initial_swarm(self) -> list[Particle]:
        self._n_initialized = 0
        particles: list[Particle] = []
        for _ in range(self.swarm_size):
            x0 = self._sample_position()
            v0 = self._sample_velocity()
            particles.append(
                Particle(
                    x=x0.copy(),
                    v=v0,
                    x_best=x0.copy(),
                    y_best=float("-inf"),
                    max_viol_best=float("inf"),
                )
            )
        return particles

    # ------------------------------------------------------------------
    # Evaluation / repair helpers
    # ------------------------------------------------------------------

    def _evaluate_point(self, x: np.ndarray) -> tuple[float, np.ndarray, float, bool]:
        g = self.problem.constraint_values(x)
        g = np.asarray(g, dtype=float)
        box_low = float(np.maximum(0.0, self.lo - x).max())
        box_hi = float(np.maximum(0.0, x - self.hi).max())
        g_viol = float(np.maximum(0.0, g).max()) if g.size else 0.0
        max_viol = float(max(g_viol, box_low, box_hi))
        feasible = max_viol <= 1e-9
        y = float(self.problem.objective(x))
        return y, g, max_viol, feasible

    def _segment_repair(
        self,
        x_old: np.ndarray,
        x_new: np.ndarray,
        old_feasible: bool,
    ) -> np.ndarray:
        if self.repair_mode != "segment":
            return x_new
        if not old_feasible or self.problem.is_feasible(x_new):
            return x_new

        left = x_old.copy()
        lo_alpha, hi_alpha = 0.0, 1.0
        best = left
        for _ in range(self.repair_steps):
            mid_alpha = 0.5 * (lo_alpha + hi_alpha)
            trial = self.problem.clip_to_box(x_old + mid_alpha * (x_new - x_old))
            if self.problem.is_feasible(trial):
                best = trial
                lo_alpha = mid_alpha
            else:
                hi_alpha = mid_alpha
        return best

    def _global_best_index(
        self,
        ys: np.ndarray,
        max_viol: np.ndarray,
    ) -> int:
        best_idx = 0
        for i in range(1, len(ys)):
            if deb_feasibility_better(
                float(ys[i]), float(max_viol[i]),
                float(ys[best_idx]), float(max_viol[best_idx]),
            ):
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> OptimizerHistory:
        t0 = time.perf_counter()
        history = OptimizerHistory(
            algorithm="ParticleSwarmOptimization",
            crop=self.problem.crop,
            seed=int(self.seed) if self.seed is not None else -1,
            config={
                "swarm_size": self.swarm_size,
                "max_iters": self.max_iters,
                "w": self.w,
                "c1": self.c1,
                "c2": self.c2,
                "velocity_clamp_frac": self.velocity_clamp_frac,
                "init_velocity_frac": self.init_velocity_frac,
                "init_feasible_ratio": self.init_feasible_ratio,
                "repair_mode": self.repair_mode,
                "repair_steps": self.repair_steps,
                "patience_iters": self.patience_iters,
                "eps_rel": self.eps_rel,
            },
            populations=[],
            fitnesses=[],
            feasibility_masks=[],
            parent_indices=[],
        )

        swarm = self._initial_swarm()

        ys = np.empty(self.swarm_size, dtype=float)
        max_viol = np.empty(self.swarm_size, dtype=float)
        feas_mask = np.empty(self.swarm_size, dtype=bool)

        for i, particle in enumerate(swarm):
            y, g, viol, feas = self._evaluate_point(particle.x)
            ys[i] = y
            max_viol[i] = viol
            feas_mask[i] = feas
            particle.y_best = y
            particle.max_viol_best = viol
            history.log_eval(
                x=particle.x,
                y=y,
                g=g,
                feasible=feas,
                iter_index=0,
                wall_time_s=time.perf_counter() - t0,
            )

        best_idx = self._global_best_index(ys, max_viol)
        gbest_x = swarm[best_idx].x.copy()
        gbest_y = float(ys[best_idx])
        gbest_viol = float(max_viol[best_idx])

        history.populations.append(np.vstack([p.x for p in swarm]))
        history.fitnesses.append(ys.copy())
        history.feasibility_masks.append(feas_mask.copy())
        history.parent_indices.append(np.array([], dtype=int))

        plateau_run = 0

        for iteration in range(1, self.max_iters + 1):
            prev_best = history.best_feasible_so_far()

            for i, particle in enumerate(swarm):
                old_x = particle.x.copy()
                old_feasible = bool(feas_mask[i])
                r1 = self._rng.random(self.dim)
                r2 = self._rng.random(self.dim)

                particle.v = (
                    self.w * particle.v
                    + self.c1 * r1 * (particle.x_best - particle.x)
                    + self.c2 * r2 * (gbest_x - particle.x)
                )
                particle.v = self._clip_velocity(particle.v)

                x_new = self.problem.clip_to_box(particle.x + particle.v)
                x_new = self._segment_repair(old_x, x_new, old_feasible)
                particle.v = x_new - old_x
                particle.x = x_new

                y, g, viol, feas = self._evaluate_point(particle.x)
                ys[i] = y
                max_viol[i] = viol
                feas_mask[i] = feas

                if deb_feasibility_better(
                    y, viol, particle.y_best, particle.max_viol_best
                ):
                    particle.x_best = particle.x.copy()
                    particle.y_best = y
                    particle.max_viol_best = viol

                if deb_feasibility_better(y, viol, gbest_y, gbest_viol):
                    gbest_x = particle.x.copy()
                    gbest_y = y
                    gbest_viol = viol

                history.log_eval(
                    x=particle.x,
                    y=y,
                    g=g,
                    feasible=feas,
                    iter_index=iteration,
                    wall_time_s=time.perf_counter() - t0,
                )

            history.populations.append(np.vstack([p.x for p in swarm]))
            history.fitnesses.append(ys.copy())
            history.feasibility_masks.append(feas_mask.copy())
            history.parent_indices.append(np.array([], dtype=int))

            new_best = history.best_feasible_so_far()
            if np.isfinite(prev_best) and np.isfinite(new_best):
                rel_improve = (new_best - prev_best) / max(abs(prev_best), 1e-12)
                if rel_improve < self.eps_rel:
                    plateau_run += 1
                else:
                    plateau_run = 0
            else:
                plateau_run = 0

            if plateau_run >= self.patience_iters:
                history.converged = True
                break

        best_x = history.best_feasible_x()
        if best_x is None:
            best_idx = int(np.argmax(history.ys))
            best_x = history.xs[best_idx]
        history.final_x = best_x
        history.final_summary = self.problem.summary(best_x)
        return history
