"""
DT1 fusion component of the hybrid predictive digital twin.

This package combines evidence produced by the predictive DT1
components:

    DT1 ML prediction
          +
    GECO physiological prediction
          +
    model comparison
          +
    residual / CUSUM evidence
          +
    prediction fusion
          |
          v
    unified DT1 evidence

Planned components:
    model_comparison.py
        Compare ML and GECO predictions and quantify disagreement.

    prediction_fusion.py
        Combine compatible ML and GECO predictions.

    evidence_generator.py
        Convert DT1 outputs into structured evidence for the
        security / decision layers.

    dt1_output.py
        Canonical DT1 output container.

This package generates predictive/model-consistency evidence. It does
not directly control the insulin pump or make the final safety decision.
"""

__version__ = "0.1.0"

# These imports will be enabled as the corresponding modules are added:
#
# from .model_comparison import (
#     ModelComparison,
#     ModelComparisonResult,
# )
# from .prediction_fusion import (
#     PredictionFusion,
#     PredictionFusionResult,
# )
# from .evidence_generator import (
#     DT1EvidenceGenerator,
#     DT1Evidence,
# )
# from .dt1_output import DT1Output

__all__ = [
    "__version__",
]
