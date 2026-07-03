"""Executable tests for the numerical torture scenario catalog."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_numerical_torture.py"
spec = importlib.util.spec_from_file_location("run_numerical_torture", RUNNER)
assert spec is not None and spec.loader is not None
run_numerical_torture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_numerical_torture)
load_scenarios = run_numerical_torture.load_scenarios
run_scenario = run_numerical_torture.run_scenario


@pytest.mark.assurance
@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s["id"])
def test_numerical_torture_scenario(scenario):
    result = run_scenario(scenario)
    assert result["status"] == "passed", result
