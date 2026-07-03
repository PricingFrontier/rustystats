#!/usr/bin/env python3
"""Generate deterministic external-oracle fixtures.

By default this writes generated fixtures to a temporary comparison directory.
Use ``--accept`` to update checked-in fixtures after reviewing the diff.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.genmod.families as smf

ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "tests" / "oracles" / "statsmodels"


def _round_array(values: np.ndarray) -> list[float]:
    return [float(f"{v:.16g}") for v in np.asarray(values, dtype=np.float64)]


def _base_columns(n: int = 96) -> dict[str, list[float]]:
    x1 = np.linspace(-1.8, 2.2, n)
    x2 = np.sin(np.linspace(0.0, 3.0, n))
    exposure = np.linspace(0.35, 1.75, n)
    return {
        "x1": _round_array(x1),
        "x2": _round_array(x2),
        "exposure": _round_array(exposure),
    }


def _fixture(
    *,
    case_id: str,
    family: str,
    statsmodels_family,
    y: np.ndarray,
    columns: dict[str, list[float]],
    exposure: str | None = None,
    var_power: float = 1.5,
    theta: float | None = None,
    params_atol: float = 5e-7,
    params_rtol: float = 5e-7,
) -> dict:
    x1 = np.asarray(columns["x1"], dtype=np.float64)
    x2 = np.asarray(columns["x2"], dtype=np.float64)
    X = sm.add_constant(np.column_stack([x1, x2]))
    offset = None
    if exposure is not None:
        offset = np.log(np.asarray(columns[exposure], dtype=np.float64))
    result = sm.GLM(y, X, family=statsmodels_family, offset=offset).fit()
    data_columns = dict(columns)
    data_columns["y"] = _round_array(y)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "oracle": "statsmodels",
        "oracle_version": sm.__version__,
        "data": {"columns": data_columns},
        "model": {
            "response": "y",
            "terms": {"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            "family": family,
            "var_power": var_power,
            "theta": theta,
            "exposure": exposure,
            "fit_kwargs": {"max_iter": 100, "tol": 1e-10},
        },
        "expected": {
            "params": _round_array(result.params),
            "deviance": float(f"{result.deviance:.16g}"),
            "fittedvalues": _round_array(result.fittedvalues),
        },
        "tolerances": {
            "params_atol": params_atol,
            "params_rtol": params_rtol,
            "deviance_atol": 1e-6,
            "deviance_rtol": 1e-6,
            "prediction_atol": 1e-6,
            "prediction_rtol": 1e-6,
        },
    }


def build_fixtures() -> dict[str, dict]:
    cols = _base_columns()
    x1 = np.asarray(cols["x1"], dtype=np.float64)
    x2 = np.asarray(cols["x2"], dtype=np.float64)
    exposure = np.asarray(cols["exposure"], dtype=np.float64)

    gaussian_mu = 1.25 + 0.45 * x1 - 0.2 * x2
    gaussian_y = gaussian_mu + 0.03 * np.cos(np.arange(len(x1)))

    poisson_mu = exposure * np.exp(-0.35 + 0.25 * x1 - 0.1 * x2)
    poisson_y = np.asarray(np.floor(poisson_mu + 0.35 * (np.arange(len(x1)) % 3)), dtype=float)

    p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.9 * x1 - 0.35 * x2)))
    binomial_y = (p > np.quantile(p, 0.48)).astype(float)
    # Break deterministic separation while keeping a fixed, reviewed pattern.
    binomial_y[::11] = 1.0 - binomial_y[::11]

    gamma_mu = np.exp(1.1 + 0.18 * x1 + 0.08 * x2)
    gamma_y = gamma_mu * (1.0 + 0.04 * np.sin(np.arange(len(x1))))

    tweedie_mu = np.exp(0.25 + 0.2 * x1 - 0.08 * x2)
    tweedie_y = tweedie_mu * (1.0 + 0.08 * np.cos(np.arange(len(x1)) / 3.0))
    tweedie_y[::9] = 0.0

    theta = 2.5
    nb_mu = np.exp(0.1 + 0.28 * x1 - 0.12 * x2)
    nb_y = np.asarray(np.floor(nb_mu + (np.arange(len(x1)) % 4 == 0)), dtype=float)

    return {
        "gaussian_identity.json": _fixture(
            case_id="statsmodels-gaussian-identity",
            family="gaussian",
            statsmodels_family=smf.Gaussian(),
            y=gaussian_y,
            columns=cols,
            params_atol=1e-8,
            params_rtol=1e-8,
        ),
        "poisson_exposure.json": _fixture(
            case_id="statsmodels-poisson-exposure",
            family="poisson",
            statsmodels_family=smf.Poisson(),
            y=poisson_y,
            columns=cols,
            exposure="exposure",
            params_atol=2e-6,
            params_rtol=2e-6,
        ),
        "binomial_logit.json": _fixture(
            case_id="statsmodels-binomial-logit",
            family="binomial",
            statsmodels_family=smf.Binomial(),
            y=binomial_y,
            columns=cols,
            params_atol=2e-6,
            params_rtol=2e-6,
        ),
        "gamma_log.json": _fixture(
            case_id="statsmodels-gamma-log",
            family="gamma",
            statsmodels_family=smf.Gamma(link=smf.links.Log()),
            y=gamma_y,
            columns=cols,
            params_atol=2e-6,
            params_rtol=2e-6,
        ),
        "tweedie_log.json": _fixture(
            case_id="statsmodels-tweedie-log",
            family="tweedie",
            statsmodels_family=smf.Tweedie(var_power=1.5, link=smf.links.Log()),
            y=tweedie_y,
            columns=cols,
            var_power=1.5,
            params_atol=5e-5,
            params_rtol=5e-5,
        ),
        "negative_binomial_log.json": _fixture(
            case_id="statsmodels-negative-binomial-log",
            family="negbinomial",
            statsmodels_family=smf.NegativeBinomial(alpha=1.0 / theta),
            y=nb_y,
            columns=cols,
            theta=theta,
            params_atol=5e-5,
            params_rtol=5e-5,
        ),
    }


def _write_fixtures(directory: Path, fixtures: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        (directory / name).write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fixtures = build_fixtures()
    if args.accept:
        _write_fixtures(ORACLE_DIR, fixtures)
        print(f"Wrote {len(fixtures)} oracle fixtures to {ORACLE_DIR.relative_to(ROOT)}.")
        return 0

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="rustystats-oracles-"))
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
    _write_fixtures(output_dir, fixtures)

    if args.check:
        failures: list[str] = []
        for name in fixtures:
            expected = ORACLE_DIR / name
            generated = output_dir / name
            if not expected.is_file():
                failures.append(f"missing checked-in fixture {expected.relative_to(ROOT)}")
            elif expected.read_text(encoding="utf-8") != generated.read_text(encoding="utf-8"):
                failures.append(f"fixture drift: {expected.relative_to(ROOT)}")
        if failures:
            print("Oracle fixture generation check failed:")
            for failure in failures:
                print(f" - {failure}")
            print(f"Generated comparison fixtures in {output_dir}")
            return 1
        print(f"Oracle fixture generation check passed for {len(fixtures)} fixtures.")
        return 0

    print(f"Wrote comparison fixtures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
