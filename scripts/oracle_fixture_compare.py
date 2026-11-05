"""Compare a regenerated oracle fixture with the checked-in one (``--check``).

The fixture generators build their inputs with libm functions (``np.sin``, ``np.exp``)
whose last bits differ between platforms, so a regenerated fixture's floats can differ
from the checked-in ones in their last digits. ``--check`` therefore compares floats
with a tight relative tolerance and everything else (keys, lengths, strings, integers,
booleans, nulls, which floats are NaN or infinite) exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: Relative tolerance for a regenerated fixture's floats.
RTOL = 1e-10


def fixture_differences(
    expected: Any, generated: Any, *, rtol: float = RTOL, path: str = "$"
) -> list[str]:
    """Where ``generated`` differs from ``expected`` (empty when they match)."""
    if isinstance(expected, bool) or isinstance(generated, bool):
        return [] if expected is generated else [f"{path}: {generated!r} != {expected!r}"]
    if isinstance(expected, float) or isinstance(generated, float):
        if not isinstance(expected, int | float) or not isinstance(generated, int | float):
            return [f"{path}: {generated!r} != {expected!r}"]
        a, e = float(generated), float(expected)
        if math.isnan(a) or math.isnan(e) or math.isinf(a) or math.isinf(e):
            same = (math.isnan(a) and math.isnan(e)) or a == e
            return [] if same else [f"{path}: {a!r} != {e!r}"]
        if abs(a - e) <= rtol * max(abs(a), abs(e)):
            return []
        return [f"{path}: {a!r} != {e!r} (relative tolerance {rtol:g})"]
    if isinstance(expected, dict) and isinstance(generated, dict):
        if expected.keys() != generated.keys():
            missing = sorted(expected.keys() - generated.keys())
            extra = sorted(generated.keys() - expected.keys())
            return [f"{path}: keys differ (missing {missing}, extra {extra})"]
        differences: list[str] = []
        for key in sorted(expected):
            differences += fixture_differences(
                expected[key], generated[key], rtol=rtol, path=f"{path}.{key}"
            )
        return differences
    if isinstance(expected, list) and isinstance(generated, list):
        if len(expected) != len(generated):
            return [f"{path}: length {len(generated)} != {len(expected)}"]
        differences = []
        for index, (e, a) in enumerate(zip(expected, generated)):
            differences += fixture_differences(e, a, rtol=rtol, path=f"{path}[{index}]")
        return differences
    if type(expected) is not type(generated) or expected != generated:
        return [f"{path}: {generated!r} != {expected!r}"]
    return []


def fixture_file_differences(expected: Path, generated: Path) -> list[str]:
    """``fixture_differences`` of two fixture files."""
    return fixture_differences(
        json.loads(expected.read_text(encoding="utf-8")),
        json.loads(generated.read_text(encoding="utf-8")),
    )
