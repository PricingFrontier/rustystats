"""The validation report (``validation_report``) and its HTML page (``report_html``)."""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats import _html_report as view
from rustystats import _report
from rustystats._html_report import report_html
from rustystats.exceptions import ValidationError
from rustystats.formula import GLMModel

_MLFLOW_KEY = re.compile(r"^[A-Za-z0-9_\-. /]{1,250}$")


@pytest.fixture(autouse=True)
def _in_tmp_cwd(tmp_path, monkeypatch):
    """diagnostics() writes analysis/diagnostics.json to the working directory."""
    monkeypatch.chdir(tmp_path)


def frame(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    veh = rng.choice([f"V{i}" for i in range(6)], n)
    dens = rng.lognormal(5, 1, n)
    expo = rng.uniform(0.1, 1.0, n)
    eta = -1.5 + (age - 45) ** 2 / 4000 + (veh == "V3") * 0.4 + 0.05 * np.log(dens)
    return pl.DataFrame(
        {
            "age": age,
            "veh": veh,
            "dens": dens,
            "exp": expo,
            "w": rng.uniform(0.5, 1.5, n),
            "extra": rng.normal(size=n),
            "y": rng.poisson(np.exp(eta) * expo).astype(float),
        }
    )


TERMS = {
    "age": {"type": "ns", "df": 4},
    "veh": {"type": "categorical"},
    "dens": {"type": "linear"},
}


@pytest.fixture(scope="module")
def poisson():
    df = frame()
    train, test = df[:2200], df[2200:]
    model = rs.glm_dict(
        response="y", terms=TERMS, data=train, family="poisson", exposure="exp"
    ).fit()
    return model, train, test


@pytest.fixture
def report(poisson, tmp_path):
    model, train, test = poisson
    return model.validation_report(train, test, name="Freq <v1>", interactions=[("age", "veh")])


def _fit(df, family, terms=None, **kwargs):
    fit_kwargs = {k: kwargs.pop(k) for k in ("alpha", "l1_ratio", "cv") if k in kwargs}
    return rs.glm_dict(response="y", terms=terms or TERMS, data=df, family=family, **kwargs).fit(
        **fit_kwargs
    )


def _assert_page(report):
    text = json.dumps(report, allow_nan=False)
    html = report_html(report)
    assert report_html(json.loads(text)) == html
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    return html


def test_report_is_strict_json_and_page_rebuilds_from_it(report):
    html = _assert_page(report)
    assert "Freq &lt;v1&gt;" in html
    with pytest.raises(ValidationError, match="report_version"):
        report_html({**report, "report_version": 99})


def test_html_report_matches_validation_report(poisson, tmp_path):
    model, train, test = poisson
    html = model.html_report(train, test, name="Same page")
    assert "Same page" in html and "<section" in html


def test_public_export():
    assert rs.report_html is report_html


def test_summary_metrics_are_flat_and_mlflow_ready(report):
    sm = report["summary_metrics"]
    assert {"train_deviance", "test_deviance_explained", "test_gini", "test_ae"} <= set(sm)
    for key, value in sm.items():
        assert _MLFLOW_KEY.match(key) and isinstance(value, float) and np.isfinite(value)


def test_headline_numbers_match_independent_computations(poisson, report):
    model, train, test = poisson
    first, second = report["datasets"][0]["metrics"], report["datasets"][1]["metrics"]
    y, e = train["y"].to_numpy(), train["exp"].to_numpy()
    mu = np.asarray(model.fittedvalues)

    def deviance(y, mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / mu), 0.0)
        return float(2 * np.sum(t - (y - mu)))

    assert first["deviance"] == pytest.approx(deviance(y, mu), rel=1e-9)
    assert first["deviance"] == pytest.approx(model.deviance, rel=1e-9)
    assert first["null_deviance"] == pytest.approx(deviance(y, y.sum() / e.sum() * e), rel=1e-6)
    assert first["ae"] == pytest.approx(y.sum() / mu.sum(), rel=1e-9)
    assert first["deviance_explained"] == pytest.approx(
        1 - first["deviance"] / first["null_deviance"], rel=1e-12
    )
    yt, et = test["y"].to_numpy(), test["exp"].to_numpy()
    mut = model.predict(test)
    assert second["deviance"] == pytest.approx(deviance(yt, mut), rel=1e-9)
    assert second["null_deviance"] == pytest.approx(
        deviance(yt, yt.sum() / et.sum() * et), rel=1e-9
    )
    assert second["ae"] == pytest.approx(yt.sum() / mut.sum(), rel=1e-9)


def test_lorenz_and_gini_are_consistent(report):
    for d in report["datasets"]:
        lz = d["lorenz"]
        assert lz["x"][0] == 0 and lz["x"][-1] == pytest.approx(1.0)
        assert lz["y"][-1] == pytest.approx(1.0)
        area = np.trapezoid(lz["y"], lz["x"])
        assert 1 - 2 * area == pytest.approx(d["metrics"]["gini"], abs=0.01)


def test_deciles_tie_out_to_the_diagnostics_partition(poisson):
    model, train, test = poisson
    report = model.validation_report(train, test)
    diag = model.diagnostics(train_data=train, test_data=test)
    for ds, rows in zip(report["datasets"], (diag.train_test.train, diag.train_test.test)):
        for ours, theirs in zip(ds["deciles"], rows.ae_by_decile):
            assert ours["rows"] == theirs.n
            assert ours["mass"] == pytest.approx(theirs.exposure, rel=1e-3)
            assert ours["actual"] == pytest.approx(theirs.actual, rel=1e-3)
            assert ours["predicted"] == pytest.approx(theirs.predicted, rel=1e-3)


def test_factor_bands_tie_out_and_are_shared_across_datasets(poisson, report):
    model, train, test = poisson
    diag = model.diagnostics(
        train_data=train,
        test_data=test,
        categorical_factors=["veh"],
        continuous_factors=["age", "dens"],
    )
    age = next(f for f in report["features"] if f["feature"] == "age")
    bands = sorted(diag.train_test.train.continuous_diagnostics["age"], key=lambda b: b.band)
    assert [b["rows"] for b in age["bands"][0]] == [b.n for b in bands]
    assert len(age["bands"][0]) == len(age["bands"][1]) == len(age["labels"])
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    by_level = {b.level: b for b in diag.train_test.test.factor_diagnostics["veh"]}
    for label, ours in zip(veh["labels"], veh["bands"][1]):
        assert ours["rows"] == by_level[label].n
        assert ours["actual_total"] == pytest.approx(by_level[label].actual_total, rel=1e-3)


def test_z_scores_use_the_family_variance(report):
    phi = report["fit"]["credibility_dispersion"]
    assert phi >= 1.0  # a fixed-scale family never reports less noise than Poisson
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    b = veh["bands"][0][0]
    sd = np.sqrt(phi * b["expected_total"])
    assert b["z"] == pytest.approx((b["actual_total"] - b["expected_total"]) / sd, rel=1e-9)
    assert b["sd"] == pytest.approx(sd / b["expected_total"], rel=1e-9)


def test_fitted_effects_match_the_coefficients(poisson, report):
    model, _, _ = poisson
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    effect = veh["effect"]
    names = list(model.feature_names)
    params = np.asarray(model.params)
    bse = np.asarray(model.bse())
    assert effect["multiplicative"] and effect["kind"] == "relativity"
    for level, value, lo, hi in zip(effect["x"], effect["value"], effect["lo"], effect["hi"]):
        name = f"veh[T.{level}]"
        if name not in names:
            assert level == effect["base"] and value == pytest.approx(1.0)
            continue
        j = names.index(name)
        assert value == pytest.approx(np.exp(params[j]), rel=1e-9)
        assert lo == pytest.approx(np.exp(params[j] - 1.959964 * bse[j]), rel=1e-6)
        assert hi == pytest.approx(np.exp(params[j] + 1.959964 * bse[j]), rel=1e-6)


def test_continuous_effect_is_one_at_its_reference(report):
    dens = next(f for f in report["features"] if f["feature"] == "dens")
    e = dens["effect"]
    x = np.asarray(e["x"], dtype=float)
    y = np.log(np.asarray(e["value"], dtype=float))
    slope = np.polyfit(x, y, 1)[0]
    assert np.exp(slope * (e["base"] - x[0])) * e["value"][0] == pytest.approx(1.0, rel=1e-6)


def test_review_catches_a_misfit_and_passes_a_clean_fit():
    df = frame(8000, seed=2)
    rng = np.random.default_rng(3)
    # A strong effect left out of the model shows as out-of-line bands of that factor.
    effect = np.where(df["veh"].to_numpy() == "V2", 2.0, 1.0)
    mu = np.exp(-1.0) * df["exp"].to_numpy() * effect
    bad = df.with_columns(pl.Series("y", rng.poisson(mu).astype(float)))
    train, test = bad[:6000], bad[6000:]
    terms = {"dens": {"type": "linear"}}
    model = _fit(train, "poisson", terms=terms, exposure="exp")
    report = model.validation_report(train, test, categorical_factors=["veh"])
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    assert not veh["in_model"] and veh["score_test"]["significant"]
    flags = [f for f in report["review"] if f["check"] == "factor_ae" and f["status"] == "warning"]
    assert any(f["feature"] == "veh" for f in flags)
    assert any(f["check"] == "score_test" and f["feature"] == "veh" for f in report["review"])
    good = model.validation_report(train, test, thresholds={"factor_ae": None})
    assert not any(f["check"] == "factor_ae" and f["status"] == "warning" for f in good["review"])


def test_dispersion_flag_on_overdispersed_counts():
    rng = np.random.default_rng(7)
    n = 6000
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.negative_binomial(0.8, 0.8 / (0.8 + mu)).astype(float)
    df = pl.DataFrame({"x": x, "y": y})
    model = _fit(df, "poisson", terms={"x": {"type": "linear"}})
    report = model.validation_report(df)
    assert report["fit"]["dispersion"] > 1.2
    assert any(f["check"] == "dispersion" and f["status"] == "warning" for f in report["review"])


def test_train_only_report_says_so(poisson):
    model, train, _ = poisson
    report = model.validation_report(train)
    assert [d["label"] for d in report["datasets"]] == ["train"]
    assert any(f["check"] == "holdout_data" for f in report["review"])
    _assert_page(report)


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("poisson", {"exposure": "exp"}),
        ("quasipoisson", {"exposure": "exp"}),
        ("negbinomial", {"exposure": "exp", "theta": 1.5}),
        ("gamma", {}),
        ("gaussian", {}),
        ("tweedie", {}),
        ("binomial", {}),
    ],
)
def test_every_family_renders(family, kwargs):
    df = frame(1500, seed=4)
    rng = np.random.default_rng(5)
    if family == "gamma":
        df = df.with_columns(pl.Series("y", rng.gamma(2.0, 2.0, df.height) + 0.01))
    elif family == "gaussian":
        df = df.with_columns(pl.Series("y", rng.normal(size=df.height)))
    elif family == "tweedie":
        df = df.with_columns(
            pl.Series("y", rng.poisson(0.5, df.height) * rng.gamma(2.0, 2.0, df.height))
        )
    elif family == "binomial":
        df = df.with_columns(pl.Series("y", rng.integers(0, 2, df.height).astype(float)))
    train, test = df[:1100], df[1100:]
    model = _fit(train, family, **kwargs)
    report = model.validation_report(train, test)
    html = _assert_page(report)
    first = report["datasets"][0]["metrics"]
    if family != "negbinomial":  # the fit's own deviance uses the estimated theta too, but rounds
        assert first["deviance"] == pytest.approx(model.deviance, rel=1e-6)
    assert first["deviance_explained"] is not None
    if family in ("gaussian", "gamma"):
        assert report["datasets"][0]["qq"] is not None
        assert "Normal QQ plot" in html
    if family in ("poisson", "quasipoisson", "negbinomial"):
        assert report["datasets"][0]["count_distribution"] is not None


def test_weights_and_offset():
    df = frame(1800, seed=6).with_columns(pl.col("exp").log().alias("logexp"))
    train, test = df[:1300], df[1300:]
    weighted = _fit(train, "poisson", exposure="exp", weights="w")
    report = weighted.validation_report(train, test)
    first = report["datasets"][0]["metrics"]
    assert first["mass"] == pytest.approx((train["w"] * train["exp"]).sum(), rel=1e-9)
    assert first["actual"] == pytest.approx((train["w"] * train["y"]).sum(), rel=1e-9)
    _assert_page(report)
    with_offset = _fit(train, "poisson", offset="logexp")
    report2 = with_offset.validation_report(train, test)
    a = train["exp"].to_numpy()
    y = train["y"].to_numpy()
    mu0 = y.sum() / a.sum() * a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu0), 0.0)
    expected_null = 2 * np.sum(t - (y - mu0))
    assert report2["datasets"][0]["metrics"]["null_deviance"] == pytest.approx(
        expected_null, rel=1e-6
    )
    _assert_page(report2)


def test_loaded_model_renders_without_coefficients(poisson):
    model, train, test = poisson
    loaded = GLMModel.from_bytes(model.to_bytes())
    report = loaded.validation_report(train, test)
    assert report["model"]["deserialized"] is True
    assert report["coefficients"] == []
    html = _assert_page(report)
    assert "loaded from a saved file" in html


@pytest.mark.parametrize(
    "fit_kwargs", [{"alpha": 0.1, "l1_ratio": 0.0}, {"alpha": 0.05, "l1_ratio": 1.0}]
)
def test_regularised_fits_hide_confidence_bands(poisson, fit_kwargs):
    _, train, test = poisson
    model = _fit(train, "poisson", exposure="exp", **fit_kwargs)
    report = model.validation_report(train, test)
    assert report["fit"]["inference_valid"] is False
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    assert veh["effect"] is not None and "lo" not in veh["effect"]
    assert any(f["check"] == "inference" for f in report["review"])
    _assert_page(report)


def test_cross_validated_path_renders(poisson):
    _, train, test = poisson
    model = rs.glm_dict(
        response="y", terms=TERMS, data=train, family="poisson", exposure="exp"
    ).fit(regularization="ridge", cv=3)
    report = model.validation_report(train, test)
    html = _assert_page(report)
    assert "Regularisation" in html and 'aria-label="regularisation path"' in html


def test_smooth_and_other_term_types_render(poisson):
    _, train, test = poisson
    terms = {
        "age": {"type": "bs", "monotonicity": "increasing"},
        "dens": {"type": "bs"},
        "veh": {"type": "target_encoding"},
    }
    model = _fit(train, "poisson", terms=terms, exposure="exp")
    report = model.validation_report(train, test)
    html = _assert_page(report)
    if report["fit"]["smooth_terms"]:
        assert "Smooth terms" in html


def test_interactions_are_deduplicated_and_drawn():
    df = frame(4000, seed=8)
    train, test = df[:3000], df[3000:]
    model = rs.glm_dict(
        response="y",
        terms={"age": {"type": "linear"}, "veh": {"type": "categorical"}},
        interactions=[
            {"age": {"type": "linear"}, "veh": {"type": "categorical"}, "include_main": False}
        ],
        data=train,
        family="poisson",
        exposure="exp",
    ).fit()
    report = model.validation_report(train, test, interactions=[("age", "veh"), ("dens", "veh")])
    names = [p["name"] for p in report["interactions"]]
    assert len(names) == len(set(names))
    fitted = [p for p in report["interactions"] if p["in_model"]]
    assert len(fitted) == 1 and fitted[0]["effect"] is not None
    assert len(fitted[0]["cells"]) == 2
    n = fitted[0]["n_rows"] * fitted[0]["n_cols"]
    assert len(fitted[0]["cells"][0]) == n
    veh = next(f for f in report["features"] if f["feature"] == "veh")
    assert fitted[0]["name"] in veh["in_interactions"]
    html = _assert_page(report)
    assert "Fitted interaction" in html


def test_base_predictions_comparison(poisson):
    model, train, test = poisson
    base_tr = train.with_columns(pl.Series("base", np.asarray(model.fittedvalues) * 1.03))
    base_te = test.with_columns(pl.Series("base", model.predict(test) * 0.98))
    report = model.validation_report(base_tr, base_te, base_predictions="base")
    html = _assert_page(report)
    assert "Compared with the base model" in html
    assert report["base_predictions"]["test"] is not None


def test_hostile_names_are_escaped():
    rng = np.random.default_rng(9)
    n = 1200
    nasty = '<img src=x onerror="bad()">'
    cats = np.array([nasty, "</style>", 'a"b', "x&y"])[rng.integers(0, 4, n)]
    df = pl.DataFrame(
        {
            "<b>x": rng.normal(size=n),
            nasty: cats,
            "y": rng.poisson(1.0, n).astype(float),
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={"<b>x": {"type": "linear"}, nasty: {"type": "categorical"}},
        data=df,
        family="poisson",
    ).fit()
    html = report_html(model.validation_report(df, name="<script>x</script>"))
    assert "<script" not in html.lower()
    assert "<img" not in html
    assert html.count("</style>") == 1
    assert "&lt;img" in html


def test_ids_do_not_depend_on_labels(poisson):
    model, train, test = poisson
    html = report_html(model.validation_report(train, test))
    assert not re.search(r'id="[^"]*(veh|age|dens)', html)


def test_thresholds_are_validated_and_applied(poisson):
    model, train, test = poisson
    with pytest.raises(ValidationError, match="unknown threshold"):
        model.validation_report(train, thresholds={"nope": 1.0})
    with pytest.raises(ValidationError, match="non-negative"):
        model.validation_report(train, thresholds={"ae_train": -1.0})
    with pytest.raises(ValidationError, match="psi_warn"):
        model.validation_report(train, thresholds={"psi_warn": 0.5, "psi_critical": 0.1})
    report = model.validation_report(
        train, test, thresholds={"ae_test": 1e-9, "factor_z": None, "psi_warn": None}
    )
    assert report["thresholds"]["ae_test"] == 1e-9 and report["thresholds"]["psi_warn"] is None
    assert any(
        f["check"] == "overall_ae" and f["status"] == "warning" and f["dataset"] == "test"
        for f in report["review"]
    )


def test_bad_arguments_raise(poisson):
    model, train, test = poisson
    with pytest.raises(ValidationError, match="labels"):
        model.validation_report(train, test, labels=("same", "same"))
    with pytest.raises(ValidationError, match="not a column"):
        model.validation_report(train, categorical_factors=["no_such_column"])
    with pytest.raises(ValidationError, match="polars"):
        model.validation_report(train.to_pandas())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="report_version"):
        report_html({"not": "a report"})


def test_labels_name_the_datasets(poisson):
    model, train, test = poisson
    report = model.validation_report(train, test, labels=("fit 2024", "holdout 2025"))
    assert [d["label"] for d in report["datasets"]] == ["fit 2024", "holdout 2025"]
    assert "holdout_2025_deviance" in report["summary_metrics"]


def test_influence_identifies_the_outlier():
    rng = np.random.default_rng(11)
    n = 600
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.normal(scale=0.2, size=n)
    x[17], y[17] = 8.0, -5.0  # high leverage and far off the line
    df = pl.DataFrame({"x": x, "y": y})
    model = _fit(df, "gaussian", terms={"x": {"type": "linear"}})
    report = model.validation_report(df)
    influence = report["fit"]["influence"]
    assert influence["available"] and influence["top"][0]["row"] == 17


def test_gini_helper_and_small_utilities():
    assert _report._gini(np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.ones(2)) is None
    assert _report._psi([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)
    assert _report._psi([0.0, 0.0], [1.0, 1.0]) is None
    safe = _report.json_safe({"a": np.float64("nan"), "b": np.int64(2), "c": (1, np.bool_(True))})
    assert safe == {"a": None, "b": 2, "c": [1, True]}


def test_integer_coded_categorical_keeps_its_levels():
    rng = np.random.default_rng(12)
    n = 2000
    df = pl.DataFrame(
        {
            "power": rng.integers(4, 9, n),
            "x": rng.normal(size=n),
            "y": rng.poisson(0.5, n).astype(float),
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={"power": {"type": "categorical"}, "x": {"type": "linear"}},
        data=df,
        family="poisson",
    ).fit()
    report = model.validation_report(df)
    power = next(f for f in report["features"] if f["feature"] == "power")
    assert power["effect"] is not None and sorted(power["effect"]["x"]) == [
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    _assert_page(report)


def test_unfitted_factors_with_nulls_get_a_level_of_their_own(poisson):
    model, train, test = poisson
    add = pl.Series("blanks", [None if i % 7 == 0 else "a" for i in range(train.height)])
    extra = train.with_columns(add)
    test2 = test.with_columns(pl.Series("blanks", ["b"] * test.height))
    report = model.validation_report(extra, test2, categorical_factors=["blanks"])
    blanks = next(f for f in report["features"] if f["feature"] == "blanks")
    assert "None" in blanks["labels"]
    at = blanks["labels"].index("None")
    assert blanks["bands"][0][at]["rows"] == add.null_count()
    assert blanks["missing"][1]["rows"] == test.height  # "b" is a level train did not have
    assert any(f["check"] == "unseen_values" and f["feature"] == "blanks" for f in report["review"])
    _assert_page(report)


def test_high_cardinality_levels_are_pooled():
    rng = np.random.default_rng(13)
    n = 4000
    df = pl.DataFrame(
        {
            "code": [f"L{v}" for v in rng.integers(0, 60, n)],
            "x": rng.normal(size=n),
            "y": rng.poisson(0.5, n).astype(float),
        }
    )
    model = rs.glm_dict(
        response="y", terms={"x": {"type": "linear"}}, data=df, family="poisson"
    ).fit()
    report = model.validation_report(df, categorical_factors=["code"])
    code = next(f for f in report["features"] if f["feature"] == "code")
    assert code["labels"][-1] == "other levels" and len(code["labels"]) == 31
    assert sum(b["rows"] for b in code["bands"][0]) == n
    _assert_page(report)


def test_identity_link_with_offset_has_a_null_on_the_offset():
    rng = np.random.default_rng(14)
    n = 800
    off = rng.normal(size=n)
    df = pl.DataFrame({"x": rng.normal(size=n), "o": off, "y": off + rng.normal(size=n)})
    model = _fit(df, "gaussian", terms={"x": {"type": "linear"}}, offset="o")
    report = model.validation_report(df)
    y = df["y"].to_numpy()
    expected = np.sum((y - (np.mean(y - off) + off)) ** 2)
    assert report["datasets"][0]["metrics"]["null_deviance"] == pytest.approx(expected, rel=1e-6)


def test_effect_failure_leaves_the_effect_out(poisson, monkeypatch):
    model, train, test = poisson

    def refuse(*_args, **_kwargs):
        raise ValidationError("grid not supported")

    monkeypatch.setattr(_report, "_frame_on", refuse)
    monkeypatch.setattr(_report, "_iter_design_matrix_chunks", refuse)
    report = model.validation_report(train, test)
    assert all(f["effect"] is None for f in report["features"])
    html = _assert_page(report)
    assert "No fitted effects to show" in html
    assert report["fit"]["influence"]["available"] is False
    assert "could not be rebuilt" in report["fit"]["influence"]["reason"]
    assert any(f["check"] == "effect" for f in report["review"])


def test_an_unexpected_error_in_the_effect_grid_is_not_swallowed(poisson, monkeypatch):
    model, train, _ = poisson

    def broken(*_args, **_kwargs):
        raise RuntimeError("a bug, not an unsupported grid")

    monkeypatch.setattr(_report, "_frame_on", broken)
    with pytest.raises(RuntimeError, match="a bug"):
        model.validation_report(train)


def test_influence_is_skipped_on_very_wide_data(poisson, monkeypatch):
    model, train, _ = poisson
    monkeypatch.setattr(_report, "_INFLUENCE_FLOPS", 0.0)
    report = model.validation_report(train)
    assert report["fit"]["influence"]["available"] is False
    assert "left out" in report["fit"]["influence"]["reason"]
    assert "left out" in report_html(report)


def _strip(report):
    bare = json.loads(json.dumps(report))
    for d in bare["datasets"]:
        d.update(
            deciles=[],
            fit_bands=[],
            lorenz=None,
            rate_histogram=None,
            residual_histogram=None,
            qq=None,
            count_distribution=None,
            rate_percentiles={},
        )
    bare["coefficients"] = []
    bare["interactions"] = []
    bare["base_predictions"] = None
    fit = bare["fit"]
    fit["influence"] = {"available": False, "reason": "not held"}
    fit["coefficient_correlations"] = []
    fit["terms"] = []
    fit["solver_trace"] = []
    return bare


def test_page_survives_every_optional_section_missing(report):
    html = report_html(_strip(report))
    assert "No interaction pairs" in html and "not held" in html
    assert "No data." in html


def test_page_survives_no_factors(report):
    bare = _strip(report)
    bare["features"] = []
    assert "No fitted effects to show" in report_html(bare)


def test_name_argument_overrides_the_report_heading(report):
    assert "<title>Chosen</title>" in report_html(report, name="Chosen")


def test_formatters_on_edge_values():
    assert view._num(None) == "—"
    assert view._num(True) == "True"
    assert view._num(float("nan")) == "—"
    assert view._num("x<y") == "x&lt;y"
    assert view._num(0.0) == "0"
    assert view._num(1e-7).endswith("e-07")
    assert view._num(1234.5) == "1,234.5"
    assert view._pct(None) == "—" and view._pct("x") == "—"
    assert view._fixed(None) == "—" and view._fixed("x") == "—"
    assert view._amt(None) == "—" and view._amt(float("inf")) == "—"
    assert (
        view._mult(None) == "—" and view._mult(float("nan")) == "—" and view._mult(12.3) == "×12.3"
    )
    assert view._signed(None) == "—" and view._signed(float("nan")) == "—"
    assert view._column([1500.0, None]) == ["1,500", "—"]
    assert view._p(None) == "—" and view._p(float("nan")) == "—" and view._p(1e-9) == "<0.0001"
    assert view._stars(None) == "" and view._stars(0.0005) == "***" and view._stars(0.07) == "."
    assert view._label_of(2.5) == "2.5" and view._label_of("a") == "a"


def test_heat_tones():
    assert view._heat(None, 1.0, 0.3) == "tn"
    assert view._heat(1.0, 1.0, 0.3) == "tn"
    assert view._heat(1.5, 1.0, 0.3) == "tg5"
    assert view._heat(0.5, 1.0, 0.3) == "ty5"
    assert view._heat(0.2, 0.0, 1.0).startswith("tg")
    assert view._heat_scale([None], True) == 1.0
    assert view._heat_scale([0.5, 2.0], False) == 2.0
    assert view._heat_scale([1.0], True) == 0.05


def test_scales_handle_degenerate_ranges():
    assert view._ticks(float("nan"), 1.0) == [0.0]
    assert view._ticks(1.0, 1.0) == [1.0]
    flat = view._Lin(1.0, 1.0, 0.0, 10.0)
    assert flat.lo < 1.0 < flat.hi
    assert view._Lin(float("nan"), 1.0, 0.0, 10.0).lo <= 0.0
    wide = view._Log(1e-5, 1e5, 0.0, 100.0)
    assert 1.0 in wide.ticks and wide(1.0) == pytest.approx(50.0, abs=1.0)
    assert view._fv([None, float("inf"), 2]) == [2.0]


def test_kv_renders_nested_values():
    html = view._kv(
        {
            "a": {},
            "b": [],
            "c": [{"k": 1.5}, {"k": None}],
            "d": list(range(100)),
            "e": {"x": [1, 2]},
        }
    )
    assert "—" in html and "[]" in html and "… (100 values)" in html
    assert view._short_val({"long": "z" * 400}).endswith("…</span>")


def test_thresholds_must_be_a_dict():
    with pytest.raises(ValidationError, match="dict"):
        _report.resolve_thresholds(["ae_train"])  # type: ignore[arg-type]
    assert _report.json_safe(np.array([1.0, np.nan])) == [1.0, None]
    assert _report.json_safe(object()).startswith("<object")


def test_variance_takeaway_reads_the_trend(report):
    r = copy.deepcopy(report)
    bands = r["datasets"][0]["fit_bands"]
    for i, b in enumerate(bands):
        b["dispersion"] = 1.0 + i / 4.0
    assert "faster" in view._variance_takeaway(r, 0)
    for i, b in enumerate(bands):
        b["dispersion"] = 6.0 - i / 4.0
    assert "slower" in view._variance_takeaway(r, 0)
    r["datasets"][0]["fit_bands"] = bands[:2]
    assert view._variance_takeaway(r, 0).startswith("Variance check")


def test_takeaways_with_nothing_to_say(report):
    r = copy.deepcopy(report)
    r["coefficients"] = []
    assert view._coef_takeaway(r) == "Coefficients"
    r["features"] = []
    assert view._importance_takeaway(r) == "Factor importance"
    r["datasets"][0]["deciles"] = []
    assert view._lift_takeaway(r, 0) == "Calibration by decile"
    r["thresholds"]["factor_ae"] = None
    assert view._balance_rule(r) == "A/E by band"


def test_importance_folds_the_tail():
    features = [
        {
            "feature": f"f{i}",
            "in_model": True,
            "relative_importance": 12.0 - i,
            "significance": None,
        }
        for i in range(12)
    ]
    html = view.importance_chart({"features": features}, 500)
    assert "3 more" in html
    assert "12 factors carry" in view._importance_takeaway({"features": features})


def test_forest_uses_a_linear_axis_for_an_identity_link(poisson):
    _, train, _ = poisson
    rng = np.random.default_rng(15)
    df = train.with_columns(pl.Series("y", rng.normal(size=train.height)))
    report = _fit(df, "gaussian").validation_report(df)
    html = view.forest_chart(report, 900)
    assert "×" not in html and 'aria-label="coefficient forest plot"' in html
    assert view._heat_scale([0.2, -0.4], False) == 0.4


def test_page_without_valid_inference_notes_it(poisson):
    _, train, _ = poisson
    model = _fit(train, "poisson", exposure="exp", alpha=0.1, l1_ratio=0.0)
    report = model.validation_report(train)
    html = report_html(report)
    assert "not valid as usual" in html


def test_score_test_pair_and_missing_score_facts(report):
    r = copy.deepcopy(report)
    pair = r["interactions"][0]
    pair["score_test"] = {"pvalue": 0.01}
    pair["significance"] = {"p": 0.02}
    pair["gvif"] = 2.0
    html = report_html(r)
    assert "score test p 0.0100" in html
    f = r["features"][0]
    f["in_model"] = False
    f["score_test"] = None
    assert "not in the model" in view._factor_facts(f)


def test_double_lift_handles_a_missing_side(report):
    r = copy.deepcopy(report)
    assert view._double_lift_chart(r, 0, 500, 200) == view._none()
    r["base_predictions"] = {"train": None, "test": {"model_vs_base_deciles": []}}
    assert "No base comparison" in view._double_lift_chart(r, 0, 500, 200)
    assert view._double_lift_chart(r, 1, 500, 200) == view._none()
    assert view._base_card(r, {"train": None, "test": None}) == ""


# --------------------------------------------------------------------------- multinomial ------


def choice_frame(n: int = 6000, seed: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    veh = rng.choice(["V1", "V2", "V3", "V4"], n)
    dens = rng.lognormal(5, 1, n)
    eta = np.column_stack(
        [
            np.zeros(n),
            0.4 + 0.02 * (age - 45) + 0.5 * (veh == "V2"),
            -0.3 - 0.015 * (age - 45) + 0.6 * (veh == "V3") + 0.1 * np.log(dens),
        ]
    )
    p = np.exp(eta)
    p /= p.sum(axis=1, keepdims=True)
    picks = (p.cumsum(axis=1) > rng.uniform(size=(n, 1))).argmax(axis=1)
    return pl.DataFrame(
        {
            "age": age,
            "veh": veh,
            "dens": dens,
            "extra": rng.choice(["p", "q"], n),
            "w": rng.uniform(0.5, 1.5, n),
            "tier": np.array(["basic", "mid", "top"])[picks],
        }
    )


CHOICE_TERMS = {
    "age": {"type": "linear"},
    "veh": {"type": "categorical"},
    "dens": {"type": "bs", "df": 4},
}


@pytest.fixture(scope="module")
def multinomial():
    df = choice_frame()
    train, test = df[:4500], df[4500:]
    model = rs.multinomial_dict(
        response="tier", terms=CHOICE_TERMS, data=train, reference="basic"
    ).fit()
    return model, train, test


@pytest.fixture
def multi_report(multinomial):
    model, train, test = multinomial
    return model.validation_report(
        train,
        test,
        name="Tier <choice>",
        categorical_factors=["extra"],
        interactions=[("age", "veh")],
    )


def _view(report, label):
    return next(v for v in report["classes"] if v["class"] == label)


def test_multinomial_report_is_strict_json_and_page_rebuilds_from_it(multi_report):
    html = _assert_page(multi_report)
    assert "Tier &lt;choice&gt;" in html
    assert html.count('name="cl"') == 3 and 'aria-label="class"' in html


def test_multinomial_html_report_opens_on_a_non_reference_class(multinomial):
    model, train, test = multinomial
    html = model.html_report(train, test)
    assert 'id="cl-1" checked' in html  # basic, the reference, is class 0


def test_multinomial_metrics_tie_out_to_diagnostics(multinomial, multi_report):
    model, train, test = multinomial
    diag = model.diagnostics(train, test)
    first, second = multi_report["datasets"]
    assert first["metrics"]["deviance"] == pytest.approx(diag.train.deviance)
    assert second["metrics"]["accuracy"] == pytest.approx(diag.test.accuracy)
    assert second["metrics"]["log_loss"] == pytest.approx(diag.test.log_loss)
    assert first["class_matrix"]["confusion"] == diag.train.confusion_matrix.tolist()
    assert first["metrics"]["deviance_explained"] == pytest.approx(diag.mcfadden_pseudo_r2)
    assert np.sum(first["class_matrix"]["expected"]) == pytest.approx(first["metrics"]["mass"])


def test_multinomial_class_views_are_one_vs_rest(multinomial, multi_report):
    model, train, _ = multinomial
    p = model.predict_proba(train)
    k = model.classes_.index("mid")
    is_mid = (train["tier"] == "mid").to_numpy().astype(float)
    metrics = _view(multi_report, "mid")["datasets"][0]["metrics"]
    assert metrics["actual"] == pytest.approx(is_mid.sum())
    assert metrics["expected"] == pytest.approx(p[:, k].sum())
    assert metrics["ae"] == pytest.approx(is_mid.sum() / p[:, k].sum())
    mask = (train["veh"] == "V2").to_numpy()
    veh = next(f for f in _view(multi_report, "mid")["features"] if f["feature"] == "veh")
    band = veh["bands"][0][veh["labels"].index("V2")]
    assert band["rows"] == mask.sum()
    assert band["actual_total"] == pytest.approx(is_mid[mask].sum())
    assert band["expected_total"] == pytest.approx(p[mask, k].sum())
    sd = np.sqrt(np.sum(p[mask, k] * (1 - p[mask, k])))
    assert band["z"] == pytest.approx((is_mid[mask].sum() - p[mask, k].sum()) / sd)


def test_multinomial_effects_match_the_coefficients(multinomial, multi_report):
    model, _, _ = multinomial
    rows = {
        (r["class"], r["feature"]): r for r in model.coef_table("records") if r["class"] != "all"
    }
    veh = next(f for f in _view(multi_report, "top")["features"] if f["feature"] == "veh")
    effect = veh["effect"]
    assert effect["kind"] == "odds multiplier"
    for level, value, lo, hi in zip(effect["x"], effect["value"], effect["lo"], effect["hi"]):
        name = f"veh[T.{level}]"
        if ("top", name) not in rows:
            assert level == effect["base"] and value == pytest.approx(1.0)
            continue
        row = rows[("top", name)]
        assert value == pytest.approx(np.exp(row["estimate"]))
        assert lo == pytest.approx(np.exp(row["ci_lower"]), rel=1e-5)
        assert hi == pytest.approx(np.exp(row["ci_upper"]), rel=1e-5)


def test_multinomial_reference_class_has_no_effects_or_coefficients(multi_report):
    basic = _view(multi_report, "basic")
    assert basic["is_reference"] and basic["coefficients"] == []
    assert all(f["effect"] is None for f in basic["features"])
    assert "no coefficients of its own" in report_html(multi_report)


def test_multinomial_joint_tests_rank_the_factors(multi_report):
    features = _view(multi_report, "mid")["features"]
    shares = [f["relative_importance"] for f in features if f["in_model"]]
    assert sum(shares) == pytest.approx(100.0)
    assert shares == sorted(shares, reverse=True)
    extra = next(f for f in features if f["feature"] == "extra")
    assert not extra["in_model"] and extra["effect"] is None
    assert all(f["significance"]["p"] is not None for f in features if f["in_model"])


def test_multinomial_flags_name_the_class():
    df = choice_frame(9000, seed=21)
    # Leaving veh out of the model leaves its effect on the classes it moves.
    train, test = df[:7000], df[7000:]
    model = rs.multinomial_dict(
        response="tier", terms={"age": {"type": "linear"}}, data=train, reference="basic"
    ).fit()
    report = model.validation_report(train, test, categorical_factors=["veh"])
    flagged = [
        f for f in report["review"] if f["check"] == "factor_ae" and f["status"] == "warning"
    ]
    assert flagged and all(f["class"] in model.classes_ for f in flagged)
    assert any(f["feature"] == "veh" for f in flagged)
    assert any(f["class"] == flagged[0]["class"] for f in report["review"])
    html = report_html(report)
    assert f"veh · {flagged[0]['class']}" in html


def test_multinomial_overall_share_needs_credibility_to_warn(multinomial):
    model, train, test = multinomial
    small = test[:200]
    report = model.validation_report(train, small)
    assert not any(
        f["check"] == "overall_ae" and f["status"] == "warning" for f in report["review"]
    )
    strict = model.validation_report(train, small, thresholds={"ae_test": 1e-9, "factor_z": None})
    assert any(
        f["check"] == "overall_ae" and f["status"] == "warning" and f["class"] is not None
        for f in strict["review"]
    )


def test_multinomial_summary_metrics_are_flat_and_mlflow_ready(multi_report):
    sm = multi_report["summary_metrics"]
    assert {"test_deviance_explained", "test_accuracy", "test_ae_mid", "test_gini_top"} <= set(sm)
    for key, value in sm.items():
        assert _MLFLOW_KEY.match(key) and isinstance(value, float) and np.isfinite(value)


def test_multinomial_train_only(multinomial):
    model, train, _ = multinomial
    report = model.validation_report(train)
    assert len(report["datasets"]) == 1
    assert any(f["check"] == "holdout_data" for f in report["review"])
    _assert_page(report)


def test_multinomial_fitted_interactions_are_drawn():
    df = choice_frame(5000, seed=22)
    train, test = df[:4000], df[4000:]
    model = rs.multinomial_dict(
        response="tier",
        terms={"age": {"type": "linear"}, "veh": {"type": "categorical"}},
        interactions=[
            {"age": {"type": "linear"}, "veh": {"type": "categorical"}, "include_main": False}
        ],
        data=train,
        reference="basic",
    ).fit()
    report = model.validation_report(train, test, interactions=[("age", "veh"), ("dens", "veh")])
    pairs = _view(report, "mid")["interactions"]
    assert [p["in_model"] for p in pairs] == [True, False]
    assert len({p["name"] for p in pairs}) == 2
    assert pairs[0]["effect"] is not None and pairs[0]["coefficients"]
    assert _view(report, "basic")["interactions"][0]["effect"] is None
    assert sum(c["rows"] for c in pairs[0]["cells"][0]) == train.height
    html = _assert_page(report)
    assert "Fitted interaction" in html


def test_pairs_of_high_cardinality_keep_every_row():
    rng = np.random.default_rng(23)
    n = 3000
    df = pl.DataFrame(
        {
            "code": [f"L{v}" for v in rng.integers(0, 40, n)],
            "x": rng.normal(size=n),
            "tier": rng.choice(["a", "b", "c"], n),
        }
    )
    model = rs.multinomial_dict(
        response="tier", terms={"x": {"type": "linear"}}, data=df, reference="a"
    ).fit()
    report = model.validation_report(df, interactions=[("x", "code")])
    pair = _view(report, "b")["interactions"][0]
    assert "_Other" in pair["cols"]
    assert sum(c["rows"] for c in pair["cells"][0]) == n


def test_multinomial_loaded_model_renders(multinomial):
    from rustystats.multinomial import MultinomialModel

    model, train, test = multinomial
    loaded = MultinomialModel.from_bytes(model.to_bytes())
    report = loaded.validation_report(train, test)
    assert report["model"]["deserialized"] is True
    fresh = model.validation_report(train, test)
    assert report["datasets"][1]["metrics"]["deviance"] == pytest.approx(
        fresh["datasets"][1]["metrics"]["deviance"]
    )
    _assert_page(report)


def test_multinomial_regularised_fit_hides_confidence_bands():
    df = choice_frame(3000, seed=24)
    model = rs.multinomial_dict(
        response="tier", terms=CHOICE_TERMS, data=df, reference="basic"
    ).fit(alpha=0.01, l1_ratio=0.0)
    report = model.validation_report(df)
    assert report["fit"]["inference_valid"] is False
    veh = next(f for f in _view(report, "mid")["features"] if f["feature"] == "veh")
    assert veh["effect"] is not None and "lo" not in veh["effect"]
    html = _assert_page(report)
    assert "not valid as usual" in html and "Regularisation" in html


def test_multinomial_weights_and_availability():
    df = choice_frame(3000, seed=25).with_columns(
        (pl.col("age") < 70).alias("can_top"), pl.lit(True).alias("yes")
    )
    df = df.with_columns(
        pl.when(pl.col("can_top")).then(pl.col("tier")).otherwise(pl.lit("mid")).alias("tier")
    )
    model = rs.multinomial_dict(
        response="tier",
        terms=CHOICE_TERMS,
        data=df,
        reference="basic",
        weights="w",
        availability={"basic": "yes", "mid": "yes", "top": "can_top"},
    ).fit()
    report = model.validation_report(df)
    assert report["datasets"][0]["metrics"]["mass"] == pytest.approx(df["w"].sum())
    assert report["model"]["weights_column"] == "w"
    _assert_page(report)


def test_multinomial_alternative_coefficients_are_listed():
    rng = np.random.default_rng(26)
    n = 3000
    price = {c: rng.uniform(1, 5, n) for c in ("a", "b", "c")}
    utilities = np.column_stack([-0.5 * price[c] for c in ("a", "b", "c")])
    p = np.exp(utilities)
    p /= p.sum(axis=1, keepdims=True)
    picks = (p.cumsum(axis=1) > rng.uniform(size=(n, 1))).argmax(axis=1)
    df = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "price_a": price["a"],
            "price_b": price["b"],
            "price_c": price["c"],
            "tier": np.array(["a", "b", "c"])[picks],
        }
    )
    model = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        alternative_terms={
            "price": {
                "columns": {"a": "price_a", "b": "price_b", "c": "price_c"},
                "coefficient": "generic",
            }
        },
        data=df,
        reference="a",
    ).fit()
    report = model.validation_report(df)
    assert [c["feature"] for c in report["alternative_coefficients"]] == ["price"]
    assert "Coefficients shared by every class" in _assert_page(report)


def test_multinomial_arguments_are_validated(multinomial):
    model, train, test = multinomial
    with pytest.raises(ValidationError, match="labels"):
        model.validation_report(train, test, labels=("same", "same"))
    with pytest.raises(ValidationError, match="not a column"):
        model.validation_report(train, categorical_factors=["no_such_column"])
    with pytest.raises(ValidationError, match="polars"):
        model.validation_report(train.to_pandas())
    with pytest.raises(ValidationError, match="unknown threshold"):
        model.validation_report(train, thresholds={"nope": 1.0})


def test_tweedie_and_negbin_deviance_use_the_fitted_parameters():
    rng = np.random.default_rng(27)
    n = 1500
    df = pl.DataFrame({"x": rng.normal(size=n)}).with_columns(
        pl.Series("y", rng.poisson(1.0, n) * rng.gamma(2.0, 2.0, n))
    )
    model = _fit(df, "tweedie", terms={"x": {"type": "linear"}}, var_power=1.7)
    first = model.validation_report(df)["datasets"][0]["metrics"]
    assert first["deviance"] == pytest.approx(model.deviance, rel=1e-6)
    counts = df.with_columns(pl.Series("y", rng.poisson(1.0, n).astype(float)))
    nb = _fit(counts, "negbinomial", terms={"x": {"type": "linear"}}, theta=2.5)
    metrics = nb.validation_report(counts)["datasets"][0]["metrics"]
    assert metrics["deviance"] == pytest.approx(nb.deviance, rel=1e-6)
    assert metrics["null_deviance"] == pytest.approx(nb.null_deviance(), rel=1e-6)


def test_array_offset_reaches_the_train_null_deviance():
    df = frame(1500, seed=30)
    offset = np.log(df["exp"].to_numpy())
    model = _fit(df, "poisson", offset=offset)
    metrics = model.validation_report(df)["datasets"][0]["metrics"]
    assert metrics["null_deviance"] == pytest.approx(model.null_deviance(), rel=1e-10)


def test_out_of_core_fit_renders(tmp_path):
    df = frame(3000, seed=31)

    def open_batches(columns):
        selected = df.select(columns)
        for start in range(0, selected.height, 700):
            yield selected.slice(start, 700)

    model = rs.glm_dict(
        response="y",
        terms=TERMS,
        data=rs.BatchSource(open_batches, df.schema),
        family="poisson",
        exposure="exp",
    ).fit(out_of_core=rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=tmp_path))
    report = model.validation_report(df)
    assert report["datasets"][0]["metrics"]["deviance"] == pytest.approx(model.deviance, rel=1e-9)
    assert report["datasets"][0]["metrics"]["null_deviance"] == pytest.approx(
        model.null_deviance(), rel=1e-9
    )
    assert report["fit"]["influence"]["available"] is False
    _assert_page(report)
