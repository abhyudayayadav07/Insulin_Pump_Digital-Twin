"""
SimGlucose runner for the Insulin Digital Twin framework.

Responsibilities:
1. Create and configure a SimGlucose simulation.
2. Run a virtual patient through the simulation.
3. Collect simulation results.
4. Return a standardized pandas DataFrame for downstream modules.

The runner is deliberately kept separate from DT1/DT2 so that the same
physiological simulation can feed both digital twins and attack scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd


class SimGlucoseRunner:
    """
    Wrapper around SimGlucose.

    The SimGlucose package is imported inside the run method so that the
    rest of the project can still be imported/tested when SimGlucose is not
    installed yet.
    """

    def __init__(
        self,
        patient_name: Optional[str] = None,
        scenario: Any = None,
        controller: Any = None,
        sample_time_minutes: int = 5,
        duration_hours: int = 24,
    ):
        self.patient_name = patient_name
        self.scenario = scenario
        self.controller = controller
        self.sample_time_minutes = sample_time_minutes
        self.duration_hours = duration_hours

    def run(self) -> pd.DataFrame:
        """
        Execute the SimGlucose simulation.

        Returns:
            pandas.DataFrame containing standardized simulation data.

        Raises:
            ImportError: If SimGlucose is not installed.
            ValueError: If required simulation configuration is missing.
        """
        try:
            from simglucose.simulation.env import sim
            from simglucose.simulation.scenario import CustomScenario
            from simglucose.simulation.user_interface import simulate
        except ImportError as exc:
            raise ImportError(
                "SimGlucose is not installed. Install it before running "
                "the physiological simulation."
            ) from exc

        if self.patient_name is None:
            raise ValueError(
                "A SimGlucose patient name must be supplied before running."
            )

        if self.scenario is None:
            raise ValueError(
                "A SimGlucose scenario must be supplied before running."
            )

        if self.controller is None:
            raise ValueError(
                "A controller must be supplied before running."
            )

        # SimGlucose APIs differ slightly across releases. This method keeps
        # the project-level interface stable while the exact simulator setup
        # is configured for the installed SimGlucose version.
        #
        # The actual simulation object is intentionally created here rather
        # than inside main.py.
        results = simulate(
            controller=self.controller,
            patient=self.patient_name,
            scenario=self.scenario,
        )

        return self.standardize_results(results)

    @staticmethod
    def standardize_results(results: Any) -> pd.DataFrame:
        """
        Convert SimGlucose output into the common project data schema.

        Expected downstream columns include:
            time, glucose, cgm, insulin, meal

        Additional columns returned by SimGlucose are preserved.
        """
        if isinstance(results, pd.DataFrame):
            df = results.copy()
        elif hasattr(results, "data"):
            df = pd.DataFrame(results.data)
        else:
            df = pd.DataFrame(results)

        # Normalize common SimGlucose column names.
        rename_map = {}

        for column in df.columns:
            normalized = str(column).strip().lower()

            if normalized in {"bg", "blood_glucose", "blood glucose"}:
                rename_map[column] = "glucose"
            elif normalized in {"cgm", "sensor_glucose", "sensor glucose"}:
                rename_map[column] = "cgm"
            elif normalized in {"insulin", "insulin_dose", "insulin dose"}:
                rename_map[column] = "insulin"
            elif normalized in {"meal", "carbs", "carbohydrate", "cho"}:
                rename_map[column] = "meal"
            elif normalized in {"timestamp", "datetime", "date_time"}:
                rename_map[column] = "time"

        df = df.rename(columns=rename_map)

        # Preserve the expected schema even if a particular signal is absent.
        for required_column in ("time", "glucose", "cgm", "insulin", "meal"):
            if required_column not in df.columns:
                df[required_column] = pd.NA

        ordered_columns = [
            "time",
            "glucose",
            "cgm",
            "insulin",
            "meal",
        ]

        remaining_columns = [
            column for column in df.columns
            if column not in ordered_columns
        ]

        return df[ordered_columns + remaining_columns]

    def save_results(
        self,
        results: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Save standardized simulation results as CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_path, index=False)
        return output_path


def run_simglucose(
    patient_name: str,
    scenario: Any,
    controller: Any,
    sample_time_minutes: int = 5,
    duration_hours: int = 24,
) -> pd.DataFrame:
    """
    Convenience function for running a SimGlucose experiment.

    This function is the simple interface that main.py or a future
    experiment manager can call.
    """
    runner = SimGlucoseRunner(
        patient_name=patient_name,
        scenario=scenario,
        controller=controller,
        sample_time_minutes=sample_time_minutes,
        duration_hours=duration_hours,
    )

    return runner.run()


if __name__ == "__main__":
    print(
        "SimGlucose runner module loaded. "
        "Use SimGlucoseRunner or run_simglucose() from the project pipeline."
    )
