"""
transformer.py - Transformer encoder for Digital Twin 1 (DT1).

This model uses temporal positional encoding and a Transformer encoder to
predict future glucose from a historical feature sequence.
"""

from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)

        # For odd d_model, the cosine vector is one element longer.
        encoding[:, 1::2] = torch.cos(
            position * div_term[:encoding[:, 1::2].shape[1]]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.encoding[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    """Transformer encoder regression model for glucose forecasting."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        if input_size <= 0 or d_model <= 0:
            raise ValueError("input_size and d_model must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size

        self.input_projection = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected x shape (batch, sequence_length, input_size), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {x.shape[-1]}."
            )

        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)

        # Final timestep summarizes the available historical context.
        x = self.norm(x[:, -1, :])
        return self.output_layer(x)

    def predict(
        self,
        x: torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        if device is not None:
            self.to(device)
            x = x.to(device)

        with torch.no_grad():
            output = self.forward(x)

        if was_training:
            self.train()

        return output

    def count_parameters(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )


def create_transformer_model(
    input_size: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    output_size: int = 1,
) -> TransformerRegressor:
    return TransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=output_size,
    )
