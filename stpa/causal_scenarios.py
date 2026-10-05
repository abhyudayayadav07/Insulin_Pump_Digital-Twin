"""
DT2 STPA — Causal Scenarios
===========================

Defines causal scenarios that can lead to Unsafe Control Actions (UCAs)
and consequently to hazards in the insulin-pump cyber-physical system.

Purpose
-------
This module connects the STPA reasoning chain:

    Hazards
       ↓
    Unsafe Control Actions (UCAs)
       ↓
    Causal Scenarios
       ↓
    Safety Constraints
       ↓
    Runtime hazard reasoning

A causal scenario describes a plausible system condition, interaction,
failure, or attack pathway through which an unsafe control action can occur.

This module is a knowledge-base representation. It does not claim that a
scenario is currently occurring and does not calculate attack probability.
Runtime evidence is handled by the DT2 state/probability/prediction layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class ScenarioCategory(str, Enum):
    """High-level categories of causal scenarios."""

    SENSOR_FAILURE = "sensor_failure"
    CONTROLLER_FAILURE = "controller_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    SOFTWARE_FAILURE = "software_failure"
    CONFIGURATION_ERROR = "configuration_error"
    CYBER_ATTACK = "cyber_attack"
    HUMAN_INTERACTION = "human_interaction"
    ENVIRONMENTAL = "environmental"
    MODEL_MISMATCH = "model_mismatch"


class ScenarioMechanism(str, Enum):
    """Mechanism by which a causal scenario can arise."""

    SENSOR_NOISE = "sensor_noise"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_SPOOFING = "sensor_spoofing"
    STALE_DATA = "stale_data"

    INCORRECT_LOGIC = "incorrect_logic"
    SOFTWARE_BUG = "software_bug"
    NUMERICAL_ERROR = "numerical_error"
    MODEL_ERROR = "model_error"

    COMMUNICATION_DELAY = "communication_delay"
    COMMUNICATION_LOSS = "communication_loss"
    MESSAGE_CORRUPTION = "message_corruption"
    COMMAND_INJECTION = "command_injection"
    REPLAY = "replay"

    ACTUATOR_STUCK = "actuator_stuck"
    ACTUATOR_OVERSHOOT = "actuator_overshoot"
    ACTUATOR_UNDERDELIVERY = "actuator_underdelivery"
    ACTUATOR_OVERDELIVERY = "actuator_overdelivery"

    UNAUTHORIZED_CONFIGURATION = "unauthorized_configuration"
    INCORRECT_CONFIGURATION = "incorrect_configuration"

    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CausalScenario:
    """
    Formal representation of one causal scenario.

    Parameters
    ----------
    scenario_id:
        Unique scenario identifier.
    name:
        Human-readable name.
    category:
        Broad scenario category.
    mechanism:
        Specific causal mechanism.
    description:
        Description of the causal pathway.
    initiating_condition:
        Condition that initiates the scenario.
    causal_chain:
        Ordered sequence of events leading toward the UCA.
    affected_component:
        Main component affected.
    ucaa_ids:
        IDs of related unsafe control actions.
    related_hazards:
        IDs of related hazards.
    observable_indicators:
        Signals that may provide runtime evidence.
    mitigation_points:
        Points where the system can detect or interrupt the scenario.
    severity:
        Qualitative scenario severity.
    enabled:
        Whether the scenario is active in the analysis.
    metadata:
        Additional information.
    """

    scenario_id: str
    name: str
    category: ScenarioCategory
    mechanism: ScenarioMechanism
    description: str
    initiating_condition: str
    causal_chain: List[str] = field(default_factory=list)
    affected_component: str = ""
    uca_ids: List[str] = field(default_factory=list)
    related_hazards: List[str] = field(default_factory=list)
    observable_indicators: List[str] = field(default_factory=list)
    mitigation_points: List[str] = field(default_factory=list)
    severity: str = "major"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not self.description.strip():
            raise ValueError("description must not be empty.")

        if not self.initiating_condition.strip():
            raise ValueError(
                "initiating_condition must not be empty."
            )

        if not self.affected_component.strip():
            raise ValueError(
                "affected_component must not be empty."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        data = asdict(self)
        data["category"] = self.category.value
        data["mechanism"] = self.mechanism.value
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "CausalScenario":
        """Create a causal scenario from a dictionary."""
        values = dict(data)

        category = values.get(
            "category",
            ScenarioCategory.SOFTWARE_FAILURE,
        )
        if not isinstance(category, ScenarioCategory):
            category = ScenarioCategory(str(category))

        mechanism = values.get(
            "mechanism",
            ScenarioMechanism.UNKNOWN,
        )
        if not isinstance(mechanism, ScenarioMechanism):
            mechanism = ScenarioMechanism(str(mechanism))

        return cls(
            scenario_id=str(values["scenario_id"]),
            name=str(values["name"]),
            category=category,
            mechanism=mechanism,
            description=str(values["description"]),
            initiating_condition=str(
                values["initiating_condition"]
            ),
            causal_chain=list(values.get("causal_chain", [])),
            affected_component=str(
                values.get("affected_component", "")
            ),
            uca_ids=list(values.get("uca_ids", [])),
            related_hazards=list(
                values.get("related_hazards", [])
            ),
            observable_indicators=list(
                values.get("observable_indicators", [])
            ),
            mitigation_points=list(
                values.get("mitigation_points", [])
            ),
            severity=str(values.get("severity", "major")),
            enabled=bool(values.get("enabled", True)),
            metadata=dict(values.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Default insulin-pump causal scenarios
# ---------------------------------------------------------------------------

DEFAULT_CAUSAL_SCENARIOS: List[CausalScenario] = [
    CausalScenario(
        scenario_id="CS01",
        name="CGM Sensor Bias Causes Unsafe Insulin Command",
        category=ScenarioCategory.SENSOR_FAILURE,
        mechanism=ScenarioMechanism.SENSOR_BIAS,
        description=(
            "A persistent sensor bias causes the controller to receive "
            "glucose measurements that differ from the patient's actual "
            "physiological state."
        ),
        initiating_condition=(
            "CGM measurement develops a persistent positive or negative bias."
        ),
        causal_chain=[
            "CGM develops systematic measurement bias",
            "Controller receives biased glucose value",
            "Controller estimates incorrect physiological state",
            "Controller computes inappropriate insulin command",
            "Pump applies the command",
            "Patient glucose trajectory moves toward an unsafe region",
        ],
        affected_component="CGM sensor",
        uca_ids=[
            "UCA02",
            "UCA07",
            "UCA08",
            "UCA09",
        ],
        related_hazards=[
            "H01",
            "H02",
            "H03",
            "H05",
            "H08",
        ],
        observable_indicators=[
            "cgm_glucose_residual",
            "glucose_cgm_disagreement",
            "unexpected_glucose_trajectory",
            "DT1_prediction_residual",
        ],
        mitigation_points=[
            "sensor plausibility validation",
            "sensor redundancy/consistency checking",
            "DT1 residual monitoring",
            "hazard-state monitoring",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS02",
        name="CGM Dropout Causes Delayed Control",
        category=ScenarioCategory.SENSOR_FAILURE,
        mechanism=ScenarioMechanism.SENSOR_DROPOUT,
        description=(
            "Missing CGM measurements prevent the controller from obtaining "
            "timely physiological information."
        ),
        initiating_condition=(
            "CGM samples become unavailable for longer than the permitted "
            "freshness interval."
        ),
        causal_chain=[
            "CGM data becomes unavailable",
            "Controller receives stale or missing state information",
            "Controller cannot safely update insulin command",
            "Required control action is delayed or omitted",
            "Glucose trajectory deteriorates",
        ],
        affected_component="CGM-to-controller interface",
        uca_ids=[
            "UCA01",
            "UCA04",
            "UCA10",
        ],
        related_hazards=[
            "H02",
            "H04",
            "H05",
        ],
        observable_indicators=[
            "missing_cgm",
            "sensor_age",
            "stale_measurement_duration",
            "control_update_delay",
        ],
        mitigation_points=[
            "data freshness check",
            "safe fallback controller",
            "sensor timeout handling",
        ],
        severity="major",
    ),
    CausalScenario(
        scenario_id="CS03",
        name="Sensor Spoofing Causes False Hypoglycemia",
        category=ScenarioCategory.CYBER_ATTACK,
        mechanism=ScenarioMechanism.SENSOR_SPOOFING,
        description=(
            "An attacker manipulates the glucose measurement so that the "
            "controller perceives glucose to be substantially different "
            "from the physiological state."
        ),
        initiating_condition=(
            "Unauthorized modification or injection of CGM measurements."
        ),
        causal_chain=[
            "Attacker manipulates CGM data",
            "Controller receives malicious glucose value",
            "Controller state estimate becomes incorrect",
            "Unsafe insulin command is generated",
            "Pump executes manipulated control decision",
            "Hazardous glucose trajectory develops",
        ],
        affected_component="CGM communication path",
        uca_ids=[
            "UCA02",
            "UCA09",
            "UCA14",
        ],
        related_hazards=[
            "H01",
            "H02",
            "H03",
            "H08",
        ],
        observable_indicators=[
            "sensor_attack_score",
            "CGM residual",
            "unexpected CGM rate",
            "DT1 residual",
            "command deviation",
        ],
        mitigation_points=[
            "message authentication",
            "sensor consistency checking",
            "DT1 prediction residual",
            "attack detection",
            "safety-state escalation",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS04",
        name="Controller Logic Error Produces Excess Insulin",
        category=ScenarioCategory.CONTROLLER_FAILURE,
        mechanism=ScenarioMechanism.INCORRECT_LOGIC,
        description=(
            "An error in controller logic causes an insulin command that "
            "exceeds what is appropriate for the current physiological state."
        ),
        initiating_condition=(
            "Controller logic produces an incorrect insulin dose or basal rate."
        ),
        causal_chain=[
            "Controller receives valid state information",
            "Controller logic computes incorrect control output",
            "Safety constraint is not enforced",
            "Pump receives excessive command",
            "Insulin overdelivery occurs",
            "Glucose falls toward a hazardous region",
        ],
        affected_component="Insulin controller",
        uca_ids=[
            "UCA02",
            "UCA07",
            "UCA08",
        ],
        related_hazards=[
            "H01",
            "H03",
            "H06",
        ],
        observable_indicators=[
            "commanded_insulin",
            "expected_insulin",
            "control_residual",
            "DT1 glucose trajectory",
        ],
        mitigation_points=[
            "command range checking",
            "independent safety monitor",
            "DT1 trajectory monitoring",
            "emergency stop authority",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS05",
        name="Controller Delay Causes Late Insulin Response",
        category=ScenarioCategory.CONTROLLER_FAILURE,
        mechanism=ScenarioMechanism.COMMUNICATION_DELAY,
        description=(
            "A delay in the control loop causes an otherwise appropriate "
            "insulin command to arrive after the relevant physiological "
            "control window."
        ),
        initiating_condition=(
            "Controller or communication latency exceeds the configured "
            "control-cycle tolerance."
        ),
        causal_chain=[
            "Physiological state changes",
            "Controller update is delayed",
            "Old state is used for control",
            "Insulin command arrives late",
            "Glucose trajectory continues deteriorating",
        ],
        affected_component="Control loop",
        uca_ids=[
            "UCA04",
            "UCA03",
        ],
        related_hazards=[
            "H02",
            "H04",
        ],
        observable_indicators=[
            "control_latency",
            "timestamp_difference",
            "command_age",
            "glucose_rate_of_change",
        ],
        mitigation_points=[
            "latency monitoring",
            "stale-command rejection",
            "fallback control",
        ],
        severity="major",
    ),
    CausalScenario(
        scenario_id="CS06",
        name="Pump Stuck-On Causes Prolonged Insulin Delivery",
        category=ScenarioCategory.ACTUATOR_FAILURE,
        mechanism=ScenarioMechanism.ACTUATOR_STUCK,
        description=(
            "The pump continues delivering insulin after the commanded "
            "delivery should have stopped."
        ),
        initiating_condition=(
            "Actuator fails to terminate insulin delivery."
        ),
        causal_chain=[
            "Controller commands insulin delivery",
            "Pump begins delivery",
            "Stop condition is reached",
            "Pump fails to stop",
            "Actual insulin exceeds commanded amount",
            "Hypoglycemia hazard develops",
        ],
        affected_component="Insulin pump actuator",
        uca_ids=[
            "UCA05",
            "UCA12",
        ],
        related_hazards=[
            "H01",
            "H03",
            "H07",
        ],
        observable_indicators=[
            "commanded_insulin",
            "delivered_insulin",
            "pump_delivery_residual",
            "glucose_rate_of_change",
        ],
        mitigation_points=[
            "independent delivery monitoring",
            "delivery timeout",
            "hardware/software safety stop",
            "DT1 trajectory monitoring",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS07",
        name="Pump Underdelivery Causes Persistent Hyperglycemia",
        category=ScenarioCategory.ACTUATOR_FAILURE,
        mechanism=ScenarioMechanism.ACTUATOR_UNDERDELIVERY,
        description=(
            "The pump delivers less insulin than commanded, causing the "
            "actual physiological response to diverge from the expected "
            "trajectory."
        ),
        initiating_condition=(
            "Pump delivery is partially or completely below the commanded dose."
        ),
        causal_chain=[
            "Controller issues insulin command",
            "Pump delivers insufficient insulin",
            "Actual insulin differs from commanded insulin",
            "Glucose remains elevated or rises",
            "Hyperglycemia hazard develops",
        ],
        affected_component="Insulin pump actuator",
        uca_ids=[
            "UCA01",
            "UCA06",
        ],
        related_hazards=[
            "H02",
            "H04",
            "H07",
        ],
        observable_indicators=[
            "commanded_insulin",
            "delivered_insulin",
            "delivery_residual",
            "predicted_vs_actual_glucose",
        ],
        mitigation_points=[
            "command-vs-delivery comparison",
            "trajectory residual monitoring",
            "pump fault detection",
        ],
        severity="major",
    ),
    CausalScenario(
        scenario_id="CS08",
        name="Communication Loss Causes Stale Control",
        category=ScenarioCategory.COMMUNICATION_FAILURE,
        mechanism=ScenarioMechanism.COMMUNICATION_LOSS,
        description=(
            "Loss of communication between sensing, controller, or pump "
            "components causes stale or missing control information."
        ),
        initiating_condition=(
            "One or more safety-critical communication links become unavailable."
        ),
        causal_chain=[
            "Communication link becomes unavailable",
            "Latest state or command is not delivered",
            "Component continues using stale information",
            "Control action becomes inappropriate or delayed",
            "Unsafe physiological/device state develops",
        ],
        affected_component="Safety-critical communication link",
        uca_ids=[
            "UCA01",
            "UCA04",
            "UCA14",
        ],
        related_hazards=[
            "H02",
            "H04",
            "H07",
            "H08",
        ],
        observable_indicators=[
            "packet_loss",
            "communication_timeout",
            "message_age",
            "command_age",
        ],
        mitigation_points=[
            "communication timeout",
            "safe fallback state",
            "stale-data rejection",
            "independent safety monitor",
        ],
        severity="major",
    ),
    CausalScenario(
        scenario_id="CS09",
        name="Replay Attack Reuses a Previous Insulin Command",
        category=ScenarioCategory.CYBER_ATTACK,
        mechanism=ScenarioMechanism.REPLAY,
        description=(
            "An attacker replays a previously valid control message under "
            "a different physiological context."
        ),
        initiating_condition=(
            "Previously valid command is replayed without adequate freshness "
            "or sequence validation."
        ),
        causal_chain=[
            "Valid control command is captured",
            "Attacker stores the command",
            "Physiological state changes",
            "Captured command is replayed",
            "Controller/pump accepts stale command",
            "Unsafe insulin action occurs",
        ],
        affected_component="Control communication interface",
        uca_ids=[
            "UCA03",
            "UCA14",
        ],
        related_hazards=[
            "H03",
            "H04",
            "H08",
        ],
        observable_indicators=[
            "duplicate_command_id",
            "sequence_number",
            "message_timestamp",
            "command_age",
            "attack_score",
        ],
        mitigation_points=[
            "freshness validation",
            "sequence numbers",
            "authenticated communication",
            "replay protection",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS10",
        name="Unauthorized Configuration Changes Safety Limits",
        category=ScenarioCategory.CYBER_ATTACK,
        mechanism=ScenarioMechanism.UNAUTHORIZED_CONFIGURATION,
        description=(
            "An unauthorized actor changes safety-critical pump or "
            "controller parameters."
        ),
        initiating_condition=(
            "Unauthorized access to the configuration interface."
        ),
        causal_chain=[
            "Attacker obtains configuration access",
            "Safety-critical parameter is modified",
            "Modified parameter passes to controller",
            "Controller operates under unsafe limit",
            "Unsafe control action becomes possible",
            "Patient/device hazard develops",
        ],
        affected_component="Pump configuration interface",
        uca_ids=[
            "UCA07",
            "UCA08",
            "UCA11",
        ],
        related_hazards=[
            "H01",
            "H02",
            "H03",
            "H04",
            "H08",
        ],
        observable_indicators=[
            "configuration_change",
            "unauthorized_access",
            "parameter_deviation",
            "audit_log_event",
        ],
        mitigation_points=[
            "authentication",
            "authorization",
            "configuration integrity checks",
            "audit logging",
            "parameter range validation",
        ],
        severity="critical",
    ),
    CausalScenario(
        scenario_id="CS11",
        name="Model Mismatch Causes Incorrect DT2 Hazard Evidence",
        category=ScenarioCategory.MODEL_MISMATCH,
        mechanism=ScenarioMechanism.MODEL_ERROR,
        description=(
            "Mismatch between the digital-twin model and the simulated or "
            "real physiological/device behavior causes misleading residual "
            "or trajectory evidence."
        ),
        initiating_condition=(
            "Digital Twin assumptions no longer adequately represent the "
            "current system state."
        ),
        causal_chain=[
            "System behavior changes",
            "Digital Twin model remains based on outdated assumptions",
            "Predicted trajectory diverges from observed behavior",
            "Residual increases",
            "DT2 may interpret model mismatch as hazard evidence",
            "Incorrect hazard state may be generated",
        ],
        affected_component="Digital Twin model",
        uca_ids=[
            "UCA09",
            "UCA13",
        ],
        related_hazards=[
            "H05",
            "H06",
            "H08",
        ],
        observable_indicators=[
            "persistent_DT1_residual",
            "model_drift",
            "prediction_error",
            "parameter_drift",
        ],
        mitigation_points=[
            "model validation",
            "residual persistence analysis",
            "model recalibration",
            "independent evidence sources",
        ],
        severity="major",
    ),
    CausalScenario(
        scenario_id="CS12",
        name="Stale Safety Evidence Delays Emergency Stop",
        category=ScenarioCategory.SOFTWARE_FAILURE,
        mechanism=ScenarioMechanism.STALE_DATA,
        description=(
            "The safety monitor uses outdated hazard evidence and therefore "
            "fails to recognize that a critical condition is currently present."
        ),
        initiating_condition=(
            "Hazard evidence is not updated within the required monitoring period."
        ),
        causal_chain=[
            "Hazard evidence becomes outdated",
            "Safety monitor continues using previous state",
            "Current hazard is not recognized",
            "Emergency stop is not issued",
            "Unsafe actuation continues",
        ],
        affected_component="Safety monitoring layer",
        uca_ids=[
            "UCA12",
        ],
        related_hazards=[
            "H01",
            "H03",
            "H06",
            "H08",
        ],
        observable_indicators=[
            "evidence_age",
            "last_update_time",
            "hazard_state_age",
            "monitoring_latency",
        ],
        mitigation_points=[
            "evidence freshness checks",
            "watchdog timer",
            "monitoring timeout",
            "safe-state transition",
        ],
        severity="critical",
    ),
]


class CausalScenarioRegistry:
    """Registry for causal scenarios."""

    def __init__(
        self,
        scenarios: Optional[Iterable[CausalScenario]] = None,
    ) -> None:
        self._scenarios: Dict[str, CausalScenario] = {}

        for scenario in scenarios or DEFAULT_CAUSAL_SCENARIOS:
            self.register(scenario)

    def register(self, scenario: CausalScenario) -> None:
        """Register one causal scenario."""
        if scenario.scenario_id in self._scenarios:
            raise ValueError(
                f"Scenario '{scenario.scenario_id}' is already registered."
            )

        self._scenarios[scenario.scenario_id] = scenario

    def get(self, scenario_id: str) -> CausalScenario:
        """Retrieve a scenario by ID."""
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown scenario '{scenario_id}'. "
                f"Available IDs: {list(self._scenarios)}"
            ) from exc

    def get_all(
        self,
        *,
        enabled_only: bool = False,
    ) -> List[CausalScenario]:
        """Return all registered scenarios."""
        scenarios = list(self._scenarios.values())

        if enabled_only:
            scenarios = [
                scenario
                for scenario in scenarios
                if scenario.enabled
            ]

        return scenarios

    def get_by_category(
        self,
        category: ScenarioCategory,
        *,
        enabled_only: bool = False,
    ) -> List[CausalScenario]:
        """Return scenarios belonging to one category."""
        return [
            scenario
            for scenario in self.get_all(enabled_only=enabled_only)
            if scenario.category == category
        ]

    def get_by_mechanism(
        self,
        mechanism: ScenarioMechanism,
        *,
        enabled_only: bool = False,
    ) -> List[CausalScenario]:
        """Return scenarios using a specific mechanism."""
        return [
            scenario
            for scenario in self.get_all(enabled_only=enabled_only)
            if scenario.mechanism == mechanism
        ]

    def get_by_uca(
        self,
        uca_id: str,
        *,
        enabled_only: bool = False,
    ) -> List[CausalScenario]:
        """Return scenarios that can lead to a specific UCA."""
        return [
            scenario
            for scenario in self.get_all(enabled_only=enabled_only)
            if uca_id in scenario.uca_ids
        ]

    def get_by_hazard(
        self,
        hazard_id: str,
        *,
        enabled_only: bool = False,
    ) -> List[CausalScenario]:
        """Return scenarios associated with a hazard."""
        return [
            scenario
            for scenario in self.get_all(enabled_only=enabled_only)
            if hazard_id in scenario.related_hazards
        ]

    def __len__(self) -> int:
        return len(self._scenarios)

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self._scenarios


def get_default_scenarios() -> List[CausalScenario]:
    """Return independent copies of the default scenarios."""
    return [
        CausalScenario.from_dict(scenario.to_dict())
        for scenario in DEFAULT_CAUSAL_SCENARIOS
    ]


def get_causal_scenario_registry() -> CausalScenarioRegistry:
    """Create a registry populated with the default scenarios."""
    return CausalScenarioRegistry(get_default_scenarios())


def scenarios_to_dict(
    scenarios: Iterable[CausalScenario],
) -> List[Dict[str, Any]]:
    """Convert scenarios into serializable dictionaries."""
    return [scenario.to_dict() for scenario in scenarios]


__all__ = [
    "ScenarioCategory",
    "ScenarioMechanism",
    "CausalScenario",
    "CausalScenarioRegistry",
    "DEFAULT_CAUSAL_SCENARIOS",
    "get_default_scenarios",
    "get_causal_scenario_registry",
    "scenarios_to_dict",
]
