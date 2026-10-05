"""
CUSUM anomaly-monitoring component of DT1.

This package provides sequential monitoring of physiological/model
residuals for the hybrid predictive digital twin.

Planned components:
    cusum.py
        CUSUM statistic and sequential accumulation logic.

    cusum_config.py
        Configuration and monitoring thresholds.

    anomaly_detector.py
        Higher-level anomaly detection using CUSUM outputs.

The CUSUM layer is an evidence-generation component. It does not
directly make a clinical or pump-control decision.
"""

__version__ = "0.1.0"

# These imports will be enabled as the corresponding modules are added:
#
# from .cusum import CUSUM, CUSUMResult
# from .cusum_config import CUSUMConfig
# from .anomaly_detector import CUSUMAnomalyDetector, AnomalyResult

__all__ = [
    "__version__",
]
