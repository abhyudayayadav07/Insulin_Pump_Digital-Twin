"""
gru.py - GRU model for Digital Twin 1 (DT1).
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class GRURegressor(nn.Module):
    """GRU regression model for future glucose prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        predict_all_steps: bool = False,
    ) -> None:
        super().__init__()

        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_size, hidden_size and num_layers must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.predict_all_steps = predict_all_steps
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(
            hidden_size * self.num_directions,
            output_size,
        )

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected x shape (batch, sequence_length, input_size), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {x.shape[-1]}."
            )

        sequence_output, _ = self.gru(x, hidden)

        if self.predict_all_steps:
            return self.output_layer(self.dropout(sequence_output))

        return self.output_layer(
            self.dropout(sequence_output[:, -1, :])
        )

    def predict(
        self,
        x: torch.Tensor,
        device: Optional[torch.device | str] = None,
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


def create_gru_model(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    output_size: int = 1,
    dropout: float = 0.1,
) -> GRURegressor:
    return GRURegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )
