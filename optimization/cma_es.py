"""Constrained CMA-ES in normalized [0, 1]^d space.

The algorithm maintains a multivariate Gaussian search distribution
over the unit-cube parameterization of the greenhouse problem:

    u ~ N(m, sigma^2 C),   u in [0, 1]^d

Candidate points are mapped back to the original greenhouse units only
for objective / constraint evaluation. Constraint handling uses the
same Deb-style feasibility rule already used by the GA and PSO:

  1. feasible beats infeasible
  2. among feasible points, higher objective beats lower
  3. among infeasible points, lower max-violation beats higher

This lets CMA-ES fit the same constrained black-box interface as the
other optimizers in the repo without introducing penalty weights as a
required hyperparameter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from optimization.history import OptimizerHistory
from optimization.problem import GreenhouseProblem


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


def _deb_argsort(fitness: np.ndarray, max_viol: np.ndarray) -> np.ndarray:
    """Argsort best→worst under Deb's constrained-feasibility rule."""
    feasible = max_viol <= 1e-9
    feas_idx = np.where(feasible)[0]
    infeas_idx = np.where(~feasible)[0]
    feas_order = feas_idx[np.argsort(-fitness[feas_idx])]
    infeas_order = infeas_idx[np.argsort(max_viol[infeas_idx])]
    return np.concatenate([feas_order, infeas_order])


@dataclass
class CMAESOptimizer:
    problem: GreenhouseProblem
    pop_size: Optional[int] = None
    max_iters: int = 100
    sigma0: float = 0.3
    init_mode: str = "feasible"     # "feasible" | "midpoint" | "random"
    repair_mode: str = "mean_segment"   # "mean_segment" | "none"
    repair_steps: int = 12
    patience_iters: int = 10
    eps_rel: float = 1e-4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.dim = self.problem.dim
        self.lo, self.hi = self.problem.bounds()
        self.span = self.hi - self.lo
        self._rng = np.random.default_rng(self.seed)

        if self.pop_size is None:
            self.pop_size = 4 + int(np.floor(3 * np.log(self.dim)))
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")
        if self.init_mode not in {"feasible", "midpoint", "random"}:
            raise ValueError("init_mode must be 'feasible', 'midpoint', or 'random'")
        if self.repair_mode not in {"mean_segment", "none"}:
            raise ValueError("repair_mode must be 'mean_segment' or 'none'")

        self.mu = self.pop_size // 2
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = float(1.0 / np.sum(self.weights ** 2))

        n = self.dim
        self.cc = float((4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n))
        self.cs = float((self.mu_eff + 2.0) / (n + self.mu_eff + 5.0))
        self.c1 = float(2.0 / (((n + 1.3) ** 2) + self.mu_eff))
        self.cmu = float(min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + (1.0 / self.mu_eff)) / (((n + 2.0) ** 2) + self.mu_eff),
        ))
        self.damps = float(
            1.0
            + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )
        self.chi_n = float(np.sqrt(n) * (1.0 - (1.0 / (4.0 * n)) + (1.0 / (21.0 * n * n))))

    # ------------------------------------------------------------------
    # Unit-cube helpers
    # ------------------------------------------------------------------

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.lo) / self.span

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u, dtype=float) * self.span

    def _clip_unit(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

    def _initial_mean(self) -> np.ndarray:
        if self.init_mode == "midpoint":
            return np.full(self.dim, 0.5, dtype=float)
        if self.init_mode == "random":
            return self._rng.uniform(0.0, 1.0, size=self.dim)
        try:
            x0 = self.problem.sample_random_feasible(rng=self._rng)
            return self._to_unit(x0)
        except RuntimeError:
            return np.full(self.dim, 0.5, dtype=float)

    # ------------------------------------------------------------------
    # Evaluation / repair helpers
    # ------------------------------------------------------------------

    def _evaluate_unit_point(
        self,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, float]:
        u = self._clip_unit(u)
        x = self._from_unit(u)
        g = np.asarray(self.problem.constraint_values(x), dtype=float)
        g_viol = float(np.maximum(0.0, g).max()) if g.size else 0.0
        feasible = g_viol <= 1e-9
        y = float(self.problem.objective(x))
        return u, g, g_viol, feasible, y

    def _segment_repair_from_mean(self, mean_u: np.ndarray, cand_u: np.ndarray) -> np.ndarray:
        if self.repair_mode != "mean_segment":
            return cand_u

        mean_x = self._from_unit(mean_u)
        cand_u = self._clip_unit(cand_u)
        cand_x = self._from_unit(cand_u)
        if not self.problem.is_feasible(mean_x) or self.problem.is_feasible(cand_x):
            return cand_u

        lo_alpha, hi_alpha = 0.0, 1.0
        best = mean_u.copy()
        for _ in range(self.repair_steps):
            mid_alpha = 0.5 * (lo_alpha + hi_alpha)
            trial_u = self._clip_unit(mean_u + mid_alpha * (cand_u - mean_u))
            if self.problem.is_feasible(self._from_unit(trial_u)):
                best = trial_u
                lo_alpha = mid_alpha
            else:
                hi_alpha = mid_alpha
        return best

    def _sample_offspring(
        self,
        mean_u: np.ndarray,
        sigma: float,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = self._rng.standard_normal(size=(self.pop_size, self.dim))
        y = z @ transform.T
        u = mean_u + sigma * y
        return u, z

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> OptimizerHistory:
        t0 = time.perf_counter()
        history = OptimizerHistory(
            algorithm="CMAES",
            crop=self.problem.crop,
            seed=int(self.seed) if self.seed is not None else -1,
            config={
                "pop_size": self.pop_size,
                "max_iters": self.max_iters,
                "sigma0": self.sigma0,
                "init_mode": self.init_mode,
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

        mean_u = self._initial_mean()
        sigma = float(self.sigma0)
        pc = np.zeros(self.dim, dtype=float)
        ps = np.zeros(self.dim, dtype=float)
        B = np.eye(self.dim)
        D = np.ones(self.dim, dtype=float)
        C = np.eye(self.dim)
        invsqrtC = np.eye(self.dim)
        counteval = 0

        plateau_run = 0

        for iteration in range(self.max_iters):
            prev_best = history.best_feasible_so_far()
            mean_old = mean_u.copy()

            transform = B @ np.diag(D)
            raw_u, _ = self._sample_offspring(mean_old, sigma, transform)

            pop_x = np.empty((self.pop_size, self.dim), dtype=float)
            pop_u = np.empty((self.pop_size, self.dim), dtype=float)
            fitness = np.empty(self.pop_size, dtype=float)
            max_viol = np.empty(self.pop_size, dtype=float)
            feasible = np.empty(self.pop_size, dtype=bool)

            for i in range(self.pop_size):
                cand_u = self._clip_unit(raw_u[i])
                cand_u = self._segment_repair_from_mean(mean_old, cand_u)
                cand_u, g, viol, feas, y = self._evaluate_unit_point(cand_u)

                pop_u[i] = cand_u
                pop_x[i] = self._from_unit(cand_u)
                fitness[i] = y
                max_viol[i] = viol
                feasible[i] = feas
                counteval += 1

                history.log_eval(
                    x=pop_x[i],
                    y=y,
                    g=g,
                    feasible=feas,
                    iter_index=iteration,
                    wall_time_s=time.perf_counter() - t0,
                )

            order = _deb_argsort(fitness, max_viol)
            selected_idx = order[: self.mu]
            selected_u = pop_u[selected_idx]

            mean_u = np.sum(self.weights[:, None] * selected_u, axis=0)
            y_k = (selected_u - mean_old) / max(sigma, 1e-12)
            y_w = np.sum(self.weights[:, None] * y_k, axis=0)

            ps = (1.0 - self.cs) * ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (
                invsqrtC @ y_w
            )
            norm_ps = float(np.linalg.norm(ps))
            hsig = float(
                norm_ps
                / np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (iteration + 1)))
                / self.chi_n
                < (1.4 + (2.0 / (self.dim + 1.0)))
            )
            pc = (1.0 - self.cc) * pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w

            rank_mu = np.zeros_like(C)
            for w_i, y_i in zip(self.weights, y_k):
                rank_mu += w_i * np.outer(y_i, y_i)

            C = (
                (1.0 - self.c1 - self.cmu) * C
                + self.c1 * (
                    np.outer(pc, pc)
                    + (1.0 - hsig) * self.cc * (2.0 - self.cc) * C
                )
                + self.cmu * rank_mu
            )

            sigma *= float(np.exp((self.cs / self.damps) * ((norm_ps / self.chi_n) - 1.0)))

            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            invsqrtC = B @ np.diag(1.0 / D) @ B.T

            history.populations.append(pop_x.copy())
            history.fitnesses.append(fitness.copy())
            history.feasibility_masks.append(feasible.copy())
            history.parent_indices.append(selected_idx.copy())

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
