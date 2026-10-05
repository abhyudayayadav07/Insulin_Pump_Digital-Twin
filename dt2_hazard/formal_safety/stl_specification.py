"""Lightweight Signal Temporal Logic (STL)-style specification support.

This is an engineering representation of common safety predicates. It is not
a full formal STL theorem prover. It provides atomic predicates and temporal
operators that can be used to express and monitor safety requirements.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Sequence


class STLTruth(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class STLSignal:
    times: Sequence[float]
    values: Sequence[float]

    def __post_init__(self):
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have equal length")
        if len(self.times) == 0:
            raise ValueError("signal cannot be empty")


@dataclass(frozen=True)
class AtomicPredicate:
    name: str
    predicate: Callable[[float], bool]

    def evaluate(self, value: float) -> bool:
        return bool(self.predicate(value))


def greater_equal(threshold: float, name: str = "ge") -> AtomicPredicate:
    return AtomicPredicate(name, lambda x: x >= threshold)


def less_equal(threshold: float, name: str = "le") -> AtomicPredicate:
    return AtomicPredicate(name, lambda x: x <= threshold)


def between(low: float, high: float, name: str = "between") -> AtomicPredicate:
    return AtomicPredicate(name, lambda x: low <= x <= high)


def globally(signal: STLSignal, predicate: AtomicPredicate) -> STLTruth:
    return STLTruth.SATISFIED if all(predicate.evaluate(v) for v in signal.values) else STLTruth.VIOLATED


def eventually(signal: STLSignal, predicate: AtomicPredicate) -> STLTruth:
    return STLTruth.SATISFIED if any(predicate.evaluate(v) for v in signal.values) else STLTruth.VIOLATED


def until(
    left_signal: STLSignal,
    right_signal: STLSignal,
    left_predicate: AtomicPredicate,
    right_predicate: AtomicPredicate,
) -> STLTruth:
    if len(left_signal.values) != len(right_signal.values):
        raise ValueError("signals must be aligned")
    for i, right_value in enumerate(right_signal.values):
        if right_predicate.evaluate(right_value):
            if all(left_predicate.evaluate(v) for v in left_signal.values[: i + 1]):
                return STLTruth.SATISFIED
    return STLTruth.VIOLATED


@dataclass(frozen=True)
class STLSpecification:
    name: str
    evaluator: Callable[[STLSignal], STLTruth]

    def evaluate(self, signal: STLSignal) -> STLTruth:
        return self.evaluator(signal)
