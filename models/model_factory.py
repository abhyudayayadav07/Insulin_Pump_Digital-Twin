"""
model_factory.py

Factory for constructing the DT1 prediction model selected in config.yaml.
"""

from __future__ import annotations

from typing import Any

from .gru import GRURegressor
from .lstm import LSTMRegressor
from .transformer import TransformerRegressor


def create_model(
    model_name: str,
    input_size: int,
    **kwargs: Any,
):
    """
    Create an LSTM, GRU, or Transformer model.

    Examples
    --------
    create_model("lstm", input_size=8)
    create_model("gru", input_size=8)
    create_model("transformer", input_size=8)
    """
    name = model_name.strip().lower()

    if name == "lstm":
        return LSTMRegressor(
            input_size=input_size,
            **kwargs,
        )

    if name == "gru":
        return GRURegressor(
            input_size=input_size,
            **kwargs,
        )

    if name in {"transformer", "transformer_encoder"}:
        return TransformerRegressor(
            input_size=input_size,
            **kwargs,
        )

    raise ValueError(
        f"Unknown DT1 model '{model_name}'. "
        "Supported models: lstm, gru, transformer."
    )


__all__ = [
    "create_model",
    "LSTMRegressor",
    "GRURegressor",
    "TransformerRegressor",
]
