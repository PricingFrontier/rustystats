"""Shared pytest configuration for the RustyStats Python test-suite.

Inserts this directory onto ``sys.path`` so the deterministic data generators in
``_fixtures.py`` can be imported both as pytest fixtures (below) and directly
(``from _fixtures import make_freq_frame``) from any test module, regardless of
pytest's import mode.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from _fixtures import (
    make_freq_frame,
    make_overdispersed_counts,
    make_severity_frame,
)


@pytest.fixture
def freq_frame():
    """Poisson frequency frame with exposure independent of risk."""
    return make_freq_frame()


@pytest.fixture
def severity_frame():
    """Strictly positive Gamma severity frame (no exposure)."""
    return make_severity_frame()


@pytest.fixture
def overdispersed_counts():
    """Negative-Binomial counts with a known dispersion theta."""
    return make_overdispersed_counts()
