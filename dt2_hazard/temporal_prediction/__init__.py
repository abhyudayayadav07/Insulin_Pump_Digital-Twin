"""
Temporal prediction components for Digital Twin 2.

This package provides future-state prediction capabilities used by the
hazard-survival digital twin. The temporal prediction layer is intended
to support:

    - short-term near-future prediction
    - long-term/far-future prediction
    - multi-horizon trajectory generation
    - prediction uncertainty estimation
    - integration with reachability, hazard, survival, and mitigation layers

The prediction layer should remain an evidence-producing component. It
does not directly make insulin dosing or pump-control decisions.

Planned modules:
    sequence_builder.py
    temporal_predictor.py
    short_term_predictor.py
    long_term_predictor.py
    prediction_output.py
    prediction_uncertainty.py
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
