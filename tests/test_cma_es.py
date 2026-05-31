"""Tests for the constrained CMA-ES optimizer."""
from __future__ import annotations

import numpy as np

from optimization.cma_es import CMAESOptimizer, deb_feasibility_better
from optimization.history import OptimizerHistory


def test_deb_feasibility_rule_prefers_feasible_point():
    assert deb_feasibility_better(1.0, 0.0, 10.0, 0.5)
    assert not deb_feasibility_better(10.0, 0.5, 1.0, 0.0)


def test_deb_feasibility_rule_prefers_lower_violation_when_both_infeasible():
    assert deb_feasibility_better(1.0, 0.2, 10.0, 0.8)
    assert not deb_feasibility_better(10.0, 0.8, 1.0, 0.2)


class _ToyProblem:
    crop = "Toy"
    dim = 2

    def __init__(self) -> None:
        self.lo = np.zeros(2)
        self.hi = np.ones(2)
        self.constraints = [None]

    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def clip_to_box(self, x):
        return np.minimum(np.maximum(x, self.lo), self.hi)

    @staticmethod
    def _y(x):
        return -((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)

    @staticmethod
    def _g(x):
        return np.array([x[0] + x[1] - 1.5])

    def objective(self, x):
        return float(self._y(x))

    def constraint_values(self, x):
        return self._g(x)

    def is_feasible(self, x, tol=1e-6):
        return bool((self._g(x) <= tol).all()) and bool(
            np.all(x >= self.lo - tol) and np.all(x <= self.hi + tol)
        )

    def sample_random(self, rng):
        return rng.uniform(self.lo, self.hi)

    def sample_random_feasible(self, max_tries=1000, rng=None, tol=1e-6):
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(max_tries):
            x = self.sample_random(rng)
            if self.is_feasible(x, tol=tol):
                return x
        raise RuntimeError("could not find feasible toy sample")

    def summary(self, x):
        return {"x": x.tolist(), "y": self.objective(x), "feasible": self.is_feasible(x)}


def test_cmaes_converges_on_toy_problem():
    problem = _ToyProblem()
    cma = CMAESOptimizer(
        problem=problem,                # type: ignore[arg-type]
        pop_size=20,
        max_iters=50,
        patience_iters=15,
        sigma0=0.25,
        seed=0,
    )
    hist = cma.run()
    assert isinstance(hist, OptimizerHistory)
    assert hist.final_x is not None
    assert np.linalg.norm(hist.final_x - np.array([0.7, 0.3])) < 0.15


def test_cmaes_final_solution_is_feasible():
    problem = _ToyProblem()
    cma = CMAESOptimizer(
        problem=problem,                # type: ignore[arg-type]
        pop_size=16,
        max_iters=40,
        patience_iters=12,
        sigma0=0.25,
        seed=1,
    )
    hist = cma.run()
    assert problem.is_feasible(hist.final_x)


def test_cmaes_logs_population_snapshots():
    problem = _ToyProblem()
    cma = CMAESOptimizer(
        problem=problem,                # type: ignore[arg-type]
        pop_size=12,
        max_iters=8,
        patience_iters=999,
        sigma0=0.2,
        seed=2,
    )
    hist = cma.run()
    assert hist.populations is not None and len(hist.populations) >= 1
    assert hist.fitnesses is not None and len(hist.fitnesses) == len(hist.populations)
    assert hist.feasibility_masks is not None
    assert hist.parent_indices is not None
