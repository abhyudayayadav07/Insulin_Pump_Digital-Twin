"""
Knowledge-guided prediction package for Digital Twin 2.

This layer incorporates domain knowledge and safety constraints into
temporal prediction. It is inspired by the knowledge-guided prediction
principle used by KnowSafe: predictions should not only fit observed
data, but should also remain consistent with reachable/safe state
constraints.

The components here produce training losses and validation evidence.
They do not directly generate insulin doses or control the pump.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
