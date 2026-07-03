"""Package-level import/export contracts."""

from __future__ import annotations

import importlib
import importlib.metadata

import rustystats


def test_package_version_falls_back_when_distribution_metadata_is_unavailable(monkeypatch):
    def missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)

    reloaded = importlib.reload(rustystats)
    assert reloaded.__version__ == "0.0.0"

    monkeypatch.undo()
    importlib.reload(rustystats)
