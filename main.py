"""
Main entry point for the Insulin Digital Twin framework.

Current scope:
- Initializes the DT1 ML/DL digital twin pipeline.
- Loads configuration.
- Provides a single entry point for future integration with:
    simulator -> DT1 -> DT2 -> attacks -> decision engine -> Omniverse.

The simulator, DT2, attack, decision-engine, and Omniverse modules can be
connected here as they are implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the global YAML configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping/object.")

    return config


def initialize_directories(config: dict[str, Any]) -> None:
    """Create project data/output directories if they do not already exist."""
    directories = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "results",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def initialize_dt1(config: dict[str, Any]) -> dict[str, Any]:
    """
    Initialize DT1 configuration.

    Model creation/training is intentionally kept outside main.py.
    The function provides a clean integration point for the DT1 module.
    """
    dt1_config = config.get("dt1", {})

    return {
        "enabled": dt1_config.get("enabled", True),
        "model": dt1_config.get("model", "lstm"),
        "sequence_length": dt1_config.get("sequence_length", 12),
        "prediction_horizon": dt1_config.get("prediction_horizon", 1),
        "checkpoint": dt1_config.get("checkpoint"),
    }


def run_pipeline(config: dict[str, Any]) -> None:
    """
    Run the complete digital-twin pipeline.

    At the current development stage, only system initialization is performed.
    As each module is implemented, this function becomes the orchestration
    point connecting the simulator, DT1, DT2, attacks, decision engine,
    evaluation, and Omniverse bridge.
    """
    print("=" * 60)
    print("        INSULIN DIGITAL TWIN FRAMEWORK")
    print("=" * 60)

    initialize_directories(config)

    dt1 = initialize_dt1(config)

    print("\n[1] System initialized")
    print(f"    Project root : {PROJECT_ROOT}")
    print(f"    DT1 enabled  : {dt1['enabled']}")
    print(f"    DT1 model    : {dt1['model']}")
    print(f"    Sequence     : {dt1['sequence_length']}")
    print(f"    Horizon      : {dt1['prediction_horizon']}")

    # ------------------------------------------------------------
    # FUTURE PIPELINE
    # ------------------------------------------------------------
    #
    # 1. simulator.simglucose_runner
    #       ↓
    # 2. simulator.data_logger
    #       ↓
    # 3. digital_twin.dt1_ml
    #       ↓
    # 4. digital_twin.dt2_hazard
    #       ↓
    # 5. attacks
    #       ↓
    # 6. decision_engine
    #       ↓
    # 7. evaluation
    #       ↓
    # 8. omniverse.omniverse_bridge
    #
    # These components will be added here progressively.
    # ------------------------------------------------------------

    print("\n[2] DT1 module ready for integration")
    print("[3] Simulator integration: pending")
    print("[4] DT2 integration: pending")
    print("[5] Attack simulation: pending")
    print("[6] Decision engine: pending")
    print("[7] Omniverse bridge: pending")

    print("\nFramework initialization completed.")


def main() -> None:
    """Parse command-line arguments and start the framework."""
    parser = argparse.ArgumentParser(
        description="Insulin Digital Twin framework entry point."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
