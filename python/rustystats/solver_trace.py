"""A fit's solver trace (``specs/out-of-core-fit.md`` §4.5).

Every result carries ``solver_trace``: the drivers' semantic steps in order. A driver
records a step at each decision it makes, never per pass over the rows, so an
in-memory fit and an out-of-core fit of the same data report the same trace whatever
their batching, thread count, target-encoding sort runs or speculation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

__all__ = ["SolverStep", "solver_steps"]


class SolverStep(NamedTuple):
    """One semantic step of a solver driver (§4.5's vocabulary).

    ``driver`` is ``"irls"``, ``"cd"``, ``"smooth"``, ``"monotone"``,
    ``"alpha_max"``, ``"cv"`` or ``"nb_theta"``; ``kind`` names the decision;
    ``evaluations`` counts the semantic evaluations it read in the driver's
    sequential order; ``accepted`` and ``status`` say what it decided.
    """

    driver: str
    kind: str
    evaluations: int
    accepted: bool
    status: str


def solver_steps(*parts: Iterable[tuple] | None) -> tuple[SolverStep, ...]:
    """The steps of several traces in order, from Rust's tuples or earlier steps."""
    return tuple(SolverStep(*step) for part in parts if part for step in part)
