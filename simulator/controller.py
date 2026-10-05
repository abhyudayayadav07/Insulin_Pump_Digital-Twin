"""
Controller module for the Insulin Digital Twin framework.

This module provides the insulin-delivery controller used by SimGlucose.
The controller is intentionally separated from the simulator runner so that
the same controller interface can later be connected to DT1, DT2, attack
simulation, and the Omniverse digital-twin environment.

The baseline controller follows SimGlucose's Controller/Action interface.
"""

from __future__ import annotations

from typing import Any, Optional


def _load_simglucose_api():
    """Load SimGlucose controller classes only when they are needed."""
    try:
        from simglucose.controller.base import Action, Controller
    except ImportError as exc:
        raise ImportError(
            "SimGlucose is not installed. Install it before using "
            "the insulin controller."
        ) from exc

    return Controller, Action


class BaselineInsulinController:
    """
    Project-level wrapper around the SimGlucose basal-bolus controller.

    The wrapper keeps the project interface independent from SimGlucose's
    internal implementation. It can later be replaced by a DT-aware,
    security-aware controller without changing the rest of the architecture.
    """

    def __init__(self, target_glucose: float = 140.0):
        self.target_glucose = float(target_glucose)
        self._controller = None

    def build(self):
        """
        Construct the SimGlucose basal-bolus controller.

        Returns:
            A SimGlucose-compatible controller object.
        """
        try:
            from simglucose.controller.basal_bolus_ctrller import BBController
        except ImportError as exc:
            raise ImportError(
                "SimGlucose is not installed or its basal-bolus controller "
                "is unavailable."
            ) from exc

        self._controller = BBController(target=self.target_glucose)
        return self._controller


class DTCompatibleController:
    """
    SimGlucose-compatible controller interface reserved for DT integration.

    This controller is useful once DT1/DT2 and the security decision engine
    are connected. It accepts the current observation and can receive a
    security response from the backend before producing the pump action.
    """

    def __init__(
        self,
        basal_rate: float = 0.0,
        target_glucose: float = 140.0,
    ):
        Controller, _ = _load_simglucose_api()

        self.ControllerBase = Controller
        self.Action = _load_simglucose_api()[1]

        self.basal_rate = float(basal_rate)
        self.target_glucose = float(target_glucose)
        self.security_state = "NORMAL"
        self.last_action = self.Action(basal=0.0, bolus=0.0)

    def policy(
        self,
        observation: Any,
        reward: float,
        done: bool,
        **kwargs: Any,
    ):
        """
        Produce an insulin action for the current simulation step.

        Inputs:
            observation: SimGlucose sensor observation.
            reward: Current simulator reward.
            done: Whether the simulation episode has ended.
            kwargs: Additional simulator information, including patient state
                    and sample time.

        Output:
            SimGlucose Action containing basal and bolus insulin rates.

        Note:
            The initial implementation provides a neutral baseline action.
            The final control policy will be connected to the selected
            controller/DT/security strategy after those modules are validated.
        """
        if done:
            return self.Action(basal=0.0, bolus=0.0)

        # A safe integration point for future security decisions.
        if self.security_state in {"BLOCK", "SAFE_MODE", "EMERGENCY_STOP"}:
            action = self.Action(basal=0.0, bolus=0.0)
        else:
            action = self.Action(
                basal=self.basal_rate,
                bolus=0.0,
            )

        self.last_action = action
        return action

    def reset(self):
        """Reset controller state at the beginning of a new simulation."""
        self.security_state = "NORMAL"
        self.last_action = self.Action(
            basal=self.basal_rate,
            bolus=0.0,
        )

    def update_security_state(self, state: str):
        """
        Update the controller with a decision-engine security state.

        This is the connection point that will later receive the output of
        evidence fusion/decision/safety_response.py.
        """
        allowed_states = {
            "NORMAL",
            "MONITOR",
            "BLOCK",
            "SAFE_MODE",
            "EMERGENCY_STOP",
        }

        normalized_state = str(state).upper()

        if normalized_state not in allowed_states:
            raise ValueError(
                f"Unsupported security state: {state}. "
                f"Expected one of {sorted(allowed_states)}."
            )

        self.security_state = normalized_state


def create_controller(
    controller_type: str = "baseline",
    target_glucose: float = 140.0,
):
    """
    Create the controller selected by configuration.

    Supported project-level controller types:
        baseline
        dt_compatible
    """
    controller_type = controller_type.lower()

    if controller_type == "baseline":
        return BaselineInsulinController(
            target_glucose=target_glucose
        ).build()

    if controller_type in {"dt_compatible", "digital_twin"}:
        return DTCompatibleController(
            target_glucose=target_glucose
        )

    raise ValueError(
        f"Unsupported controller type: {controller_type}. "
        "Use 'baseline' or 'dt_compatible'."
    )


if __name__ == "__main__":
    print(
        "Controller module loaded. "
        "Use create_controller() to construct a SimGlucose controller."
    )
