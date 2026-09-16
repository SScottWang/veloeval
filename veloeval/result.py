"""Metric return type.

Every metric returns a :class:`MetricResult`, never a bare float.  The point is
that an absent value carries *why* it is absent:

``ok``              the value is a real measurement
``not_applicable``  this metric does not apply to this method/dataset
``missing_input``   a required field is absent from ``adata`` (usually upstream)
``failed``          the computation raised

Collapsing these four into ``np.nan`` makes a results table impossible to read
and impossible to aggregate correctly.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Status = Literal["ok", "not_applicable", "missing_input", "failed"]

__all__ = [
    "MetricResult",
    "Status",
    "MissingInput",
    "NotApplicable",
    "metric",
    "ok",
    "not_applicable",
    "missing_input",
    "failed",
]


@dataclass(frozen=True)
class MetricResult:
    """One metric on one run.

    Parameters
    ----------
    name
        Metric identifier, e.g. ``"cbdir"``.
    value
        The scalar, or ``None`` when ``status != "ok"``.
    status
        Why the value is (or is not) there.  See module docstring.
    detail
        Human-readable explanation; empty when ``status == "ok"``.
    per_cell
        Optional per-cell array behind the scalar.  Not serialized to CSV;
        used by the plotting layer and for diagnosing low scores.
    """

    name: str
    value: float | None = None
    status: Status = "ok"
    detail: str = ""
    per_cell: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __float__(self) -> float:
        return float("nan") if self.value is None else float(self.value)

    def __bool__(self) -> bool:
        return self.status == "ok"


def ok(name: str, value: float, per_cell: np.ndarray | None = None) -> MetricResult:
    return MetricResult(name=name, value=float(value), status="ok", per_cell=per_cell)


def not_applicable(name: str, detail: str) -> MetricResult:
    return MetricResult(name=name, status="not_applicable", detail=detail)


def missing_input(name: str, *keys: str) -> MetricResult:
    return MetricResult(
        name=name, status="missing_input", detail="missing: " + ", ".join(keys)
    )


def failed(name: str, exc: BaseException) -> MetricResult:
    return MetricResult(
        name=name, status="failed", detail=f"{type(exc).__name__}: {exc}"
    )


class MissingInput(Exception):
    """Raised by accessors when a required ``adata`` field is absent."""

    def __init__(self, *keys: str) -> None:
        self.keys = keys
        super().__init__(", ".join(keys))


class NotApplicable(Exception):
    """Raised when a metric is not defined for this method or dataset."""


def metric(fn):
    """Wrap a metric body so it always returns a :class:`MetricResult`.

    The body may return a float, an ``(value, per_cell)`` tuple, or a
    ``MetricResult``; and may raise :class:`MissingInput` /
    :class:`NotApplicable` from anywhere, including deep inside accessors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> MetricResult:
        name = fn.__name__
        try:
            out = fn(*args, **kwargs)
        except MissingInput as exc:
            return missing_input(name, *exc.keys)
        except NotApplicable as exc:
            return not_applicable(name, str(exc))
        except Exception as exc:  # noqa: BLE001 - deliberate: status carries it
            return failed(name, exc)

        if isinstance(out, MetricResult):
            return out
        if isinstance(out, tuple):
            value, per_cell = out
            return ok(name, value, per_cell)
        return ok(name, out)

    return wrapper
