"""
Safety-constraint package for Digital Twin 2.

Provides explicit physiological, trajectory, control, timing, and system
safety constraints used to verify predicted trajectories and control-related
states before evidence is passed to the higher-level decision layer.

These are engineering/research constraints and must be validated for the
intended experimental or clinical setting.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
