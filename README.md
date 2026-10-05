# DT1 Models

Neural-network models for Digital Twin 1 glucose forecasting.

## Files

- `lstm.py` — LSTM regression model
- `gru.py` — GRU regression model
- `transformer.py` — Transformer encoder regression model
- `model_factory.py` — unified model factory
- `__init__.py` — package exports

## Input

Models expect:

`(batch_size, sequence_length, num_features)`

With the current DT1 configuration:

`sequence_length = 12`

## Output

Single-horizon glucose prediction:

`(batch_size, 1)`

The models are independent of the training, checkpointing, residual, and
threshold modules.
