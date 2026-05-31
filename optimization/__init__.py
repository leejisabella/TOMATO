"""TOMATO constrained optimization problem definition.

Exports:
  GreenhouseProblem       — the per-crop constrained problem class
  make_all_problems       — convenience: 4 problems sharing one surrogate
  CROP_SPECS              — Appendix-A bounds table
  default_constraints     — milestone coupling-constraint list
  Constraint              — g(x) ≤ 0 constraint wrapper
"""

from optimization.constraints import (
    Constraint,
    DEFAULT_COST_BUDGET_USD_PER_M2,
    DEFAULT_ELECTRICITY_BUDGET_KWH_PER_M2,
    DEFAULT_WATER_BUDGET_L_PER_M2,
    VPD_LOWER_KPA,
    VPD_UPPER_KPA,
    default_constraints,
    evaluate_constraints,
    max_violation,
    vapor_pressure_deficit_kpa,
    violation,
)
from optimization.cost_model import (
    ELECTRICITY_PRICE_USD_PER_KWH,
    LAYER_AREA_M2,
    NUM_LAYERS,
    TOTAL_AREA_M2,
    electricity_cost_usd,
    electricity_kwh_per_m2_per_cycle,
    total_operating_cost_usd,
    total_yield_kg,
    water_cost_usd,
    water_l_per_m2_per_cycle,
)
from optimization.crop_specs import (
    CROP_SPECS,
    DECISION_VARIABLES,
    MAX_PEST_SEVERITY,
)
from optimization.bayesian import BayesianOptimizer
from optimization.cma_es import CMAESOptimizer
from optimization.genetic import GeneticAlgorithm
from optimization.history import OptimizerHistory
from optimization.particle_swarm import ParticleSwarmOptimizer
from optimization.problem import GreenhouseProblem, make_all_problems

__all__ = [
    "BayesianOptimizer",
    "CMAESOptimizer",
    "CROP_SPECS",
    "Constraint",
    "DECISION_VARIABLES",
    "DEFAULT_COST_BUDGET_USD_PER_M2",
    "DEFAULT_ELECTRICITY_BUDGET_KWH_PER_M2",
    "DEFAULT_WATER_BUDGET_L_PER_M2",
    "ELECTRICITY_PRICE_USD_PER_KWH",
    "GeneticAlgorithm",
    "GreenhouseProblem",
    "LAYER_AREA_M2",
    "MAX_PEST_SEVERITY",
    "NUM_LAYERS",
    "OptimizerHistory",
    "ParticleSwarmOptimizer",
    "TOTAL_AREA_M2",
    "VPD_LOWER_KPA",
    "VPD_UPPER_KPA",
    "default_constraints",
    "electricity_cost_usd",
    "electricity_kwh_per_m2_per_cycle",
    "evaluate_constraints",
    "make_all_problems",
    "max_violation",
    "total_operating_cost_usd",
    "total_yield_kg",
    "vapor_pressure_deficit_kpa",
    "violation",
    "water_cost_usd",
    "water_l_per_m2_per_cycle",
]
