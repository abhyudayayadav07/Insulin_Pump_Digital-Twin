"""Configuration for robustness evaluation and monitoring."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RobustnessConfig:
    glucose_min: float = 20.0
    glucose_max: float = 600.0
    max_prediction_error: float = 70.0
    max_hazard_score: float = 0.85
    min_sensor_integrity: float = 0.5
    min_controller_integrity: float = 0.5
    min_pump_integrity: float = 0.5
    min_communication_integrity: float = 0.5
    perturbation_levels: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    required_success_rate: float = 0.90
    random_seed: int = 42
