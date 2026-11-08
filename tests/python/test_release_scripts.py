"""The Release workflow's scripts: the version the code declares, and what PyPI serves."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# CI and the Release workflow run these scripts on 3.11+; the package still supports 3.10.
pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load("check_release_version")
verify = _load("verify_pypi_release")


def _project(root: Path, *, python: str = "1.2.3", bindings: str = "1.2.3", core: str = "1.2.3"):
    files = (
        ("pyproject.toml", "project", python),
        ("crates/rustystats/Cargo.toml", "package", bindings),
        ("crates/rustystats-core/Cargo.toml", "package", core),
    )
    for name, table, version in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'[{table}]\nversion = "{version}"\n', encoding="utf-8")
    return root


def _pypi(tmp_path: Path, *versions: str) -> Path:
    path = tmp_path / "pypi.json"
    path.write_text(json.dumps({"releases": {version: [] for version in versions}}), "utf-8")
    return path


def test_the_repository_declares_one_version() -> None:
    check.agreed_version(check.declared_versions(ROOT))


@pytest.mark.parametrize("field", ["python", "bindings", "core"])
def test_one_file_out_of_step_is_refused_naming_each_file(tmp_path, field: str) -> None:
    root = _project(tmp_path, **{field: "1.3.0"})
    with pytest.raises(check.ReleaseVersionError, match="disagree") as caught:
        check.agreed_version(check.declared_versions(root))
    for name in check._VERSION_FILES:
        assert name in str(caught.value)


@pytest.mark.parametrize("version", ["1.2", "1.2.3rc1", "01.2.3", "v1.2.3"])
def test_a_version_that_is_not_plain_release_form_is_refused(tmp_path, version: str) -> None:
    root = _project(tmp_path, python=version, bindings=version, core=version)
    with pytest.raises(check.ReleaseVersionError, match=r"a release is X\.Y\.Z"):
        check.agreed_version(check.declared_versions(root))


@pytest.mark.parametrize(
    ("version", "released", "message"),
    [
        ("1.2.3", ["1.2.3"], "already on PyPI"),
        ("1.2.2", ["1.2.3"], "not newer than 1.2.3"),
        ("1.9.0", ["1.10.0"], "not newer than 1.10.0"),
    ],
)
def test_a_version_that_is_not_new_is_refused(version, released, message) -> None:
    with pytest.raises(check.ReleaseVersionError, match=message):
        check.check_new_release(version, released)


def test_a_newer_version_passes_and_prereleases_are_not_compared() -> None:
    check.check_new_release("1.3.0", ["1.2.3", "1.3.0rc1", "2.0.0b1"])


def test_main_reports_the_version_for_later_jobs(tmp_path) -> None:
    root = _project(tmp_path)
    output = tmp_path / "output"
    pypi = _pypi(tmp_path, "1.2.2")

    code = check.main(
        ["--root", str(root), "--pypi-json", str(pypi), "--github-output", str(output)]
    )

    assert code == 0
    assert output.read_text("utf-8") == "version=1.2.3\n"


def test_main_without_pypi_only_checks_the_files_agree(tmp_path) -> None:
    assert check.main(["--root", str(_project(tmp_path))]) == 0
    assert check.main(["--root", str(_project(tmp_path, core="9.9.9"))]) == 1


def test_main_refuses_an_unbumped_version_and_reports_nothing(tmp_path, capsys) -> None:
    root = _project(tmp_path)
    output = tmp_path / "output"
    pypi = _pypi(tmp_path, "1.2.3")

    code = check.main(
        ["--root", str(root), "--pypi-json", str(pypi), "--github-output", str(output)]
    )

    assert code == 1
    assert not output.exists()
    assert capsys.readouterr().out.startswith("::error::rustystats 1.2.3 is already on PyPI")


# ── verify_pypi_release ─────────────────────────────────────────────────────


WHEEL = "rustystats-1.2.3-cp312-cp312-manylinux_2_17_x86_64.whl"
SDIST = "rustystats-1.2.3.tar.gz"


def _dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / WHEEL).write_bytes(b"wheel bytes")
    (dist / SDIST).write_bytes(b"sdist bytes")
    return dist


def _release_json(tmp_path: Path, files: dict[str, bytes]) -> Path:
    urls = [
        {"filename": name, "digests": {"sha256": hashlib.sha256(content).hexdigest()}}
        for name, content in files.items()
    ]
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"urls": urls}), "utf-8")
    return path


def test_pypi_serving_exactly_the_built_files_passes(tmp_path) -> None:
    release = _release_json(tmp_path, {WHEEL: b"wheel bytes", SDIST: b"sdist bytes"})
    assert verify.main(["--dist", str(_dist(tmp_path)), "--pypi-json", str(release)]) == 0


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({WHEEL: b"wheel bytes"}, "is not on PyPI"),
        ({WHEEL: b"other", SDIST: b"sdist bytes"}, "not the built"),
        (
            {WHEEL: b"wheel bytes", SDIST: b"sdist bytes", "extra.whl": b"x"},
            "not built by this run",
        ),
    ],
)
def test_any_difference_from_the_built_files_fails(tmp_path, capsys, files, message) -> None:
    release = _release_json(tmp_path, files)
    assert verify.main(["--dist", str(_dist(tmp_path)), "--pypi-json", str(release)]) == 1
    assert message in capsys.readouterr().out


def test_an_empty_dist_fails(tmp_path) -> None:
    (tmp_path / "dist").mkdir()
    release = _release_json(tmp_path, {})
    assert verify.main(["--dist", str(tmp_path / "dist"), "--pypi-json", str(release)]) == 1
