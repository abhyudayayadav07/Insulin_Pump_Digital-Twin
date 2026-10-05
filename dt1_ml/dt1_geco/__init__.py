"""
GECO-based physiological modeling component of DT1.

This package provides the physiological-modeling side of the hybrid
predictive digital twin.

Planned components:
    geco_model.py
        Core GECO physiological model.

    parameter_estimation.py
        Patient/model parameter fitting.

    state_estimator.py
        Estimation of latent physiological states.

    trajectory_predictor.py
        Future glucose trajectory prediction using GECO.

    geco_residual.py
        Residual calculation between observed and GECO-predicted
        glucose trajectories.

The GECO component works alongside the DT1 ML/DL component:

    Observed Data
          |
          +--------------------+
          |                    |
          v                    v
       DT1-ML               DT1-GECO
     LSTM/GRU/             Physiological
    Transformer               Model
          |                    |
          v                    v
    ML Prediction        GECO Prediction
          |                    |
          +---------+----------+
                    |
                    v
             Residual Analysis
                    |
                    v
                  CUSUM
"""

__version__ = "0.1.0"

# These imports will be enabled as the corresponding modules are created.
# Keeping them commented initially prevents import errors while the
# package is being built incrementally.

# from .geco_model import GECOModel
# from .parameter_estimation import GECOParameterEstimator
# from .state_estimator import GECOStateEstimator
# from .trajectory_predictor import GECOTrajectoryPredictor
# from .geco_residual import GECORResidual

__all__ = [
    "__version__",
]