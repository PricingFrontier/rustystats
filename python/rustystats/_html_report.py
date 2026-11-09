"""Render a validation report (``GLMModel.validation_report``) as one self-contained HTML page.

``report_html`` is a pure view of the report dict: it computes nothing that is not already in
the report, so a page can be rebuilt from the JSON logged next to the model. The page has no
scripts and loads nothing: charts are inline SVG (hover values from SVG ``<title>``), the tabs
and the dataset, class, factor and interaction selectors are CSS-only radio buttons, and detail
sits in ``<details>``. It opens offline, in a sandboxed viewer such as MLflow's, and prints.

The look is the t-boost model report's, so the two read as one family: a sticky bar of tabs with
one page-wide dataset switch, every chart drawn at the pixel width of its card. Colour roles:
blue is observed data, orange is the model; the selected dataset is drawn in colour and the
others in grey. Every label is HTML-escaped; element ids are built from indices, never labels.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from html import escape
from typing import Any, Protocol

from rustystats.exceptions import ValidationError

__all__ = ["report_html"]

_SUPPORTED_VERSIONS = (1,)

# Page geometry (CSS px). A chart is drawn at the inner width of its card, so one SVG unit is one
# CSS pixel on a desktop screen.
_CONTENT = 1200
_GAP = 16
_INNER = 2 * 16 + 2
_W_FULL = _CONTENT - _INNER
_W_HALF = (_CONTENT - _GAP) // 2 - _INNER
_RAIL = 240
_W_FACTOR = _CONTENT - _RAIL - _GAP - _INNER
_W_INTER = (_W_FACTOR + _INNER - _GAP) // 2 - _INNER
_H_ROW = 200  # chart height on the Overview: its three rows fit a 1440 x 900 screen
_FONT = 12.0
_RANKED_ROWS = 300
_OVERVIEW_ISSUES = 3
_TAB_NAMES = ("Overview", "Performance", "Factors", "Interactions", "Coefficients", "Fit", "Checks")
_CHECKS_TAB = _TAB_NAMES.index("Checks")
_BASIS_NAME = re.compile(r"^(bs|ns|s|ms)\(")

_LIGHT = (
    "--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;"
    "--grid:#e7e6e0;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--wash:rgba(11,11,11,.045);"
    "--data:#2a78d6;--model:#eb6834;--model-soft:#f6b89e;--other:#c3c2b9;"
    "--band:rgba(42,120,214,.14);--massbar:#d6d5ce;--tol:rgba(11,11,11,.05);"
    "--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--good-ink:#006300;--link:#1c5cab;"
    "--i0:#f2a17c;--i1:#e3703c;--i2:#b44c18;--i3:#6f2c0b;--hi:#c6ecc3;--lo:#fbeba6"
)
_DARK = (
    "--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8d86;"
    "--grid:#2c2c2a;--axis:#3d3d3a;--border:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);"
    "--data:#3987e5;--model:#d95926;--model-soft:#7a3a20;--other:#55544f;"
    "--band:rgba(57,135,229,.22);--massbar:#45443f;--tol:rgba(255,255,255,.06);"
    "--good-ink:#0ca30c;--link:#86b6ef;"
    "--i0:#b04c20;--i1:#d9662f;--i2:#f08f62;--i3:#f9c3a6;--hi:#2c4f30;--lo:#544a1b"
)

_CSS = """
:root{color-scheme:light;%(light)s}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;%(dark)s}}
:root[data-theme="dark"]{color-scheme:dark;%(dark)s}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.app{max-width:%(content)dpx;margin:0 auto;padding:0 16px 16px;box-sizing:content-box}
.top{padding:10px 0 0}
.top h1{font-size:20px;line-height:26px;margin:0;font-weight:650}
.top .sub{color:var(--ink2);font-size:13px;line-height:18px;margin:2px 0 0}
.bar{position:sticky;top:0;z-index:20;background:var(--page);display:flex;flex-wrap:wrap;
align-items:center;justify-content:space-between;gap:8px 16px;padding:8px 0;margin-bottom:12px;
border-bottom:1px solid var(--grid)}
.tabs{display:flex;flex-wrap:wrap;gap:2px}
.tabs label{display:block;padding:5px 12px;border:1px solid transparent;border-radius:7px;
cursor:pointer;color:var(--ink2);font-size:14px;line-height:20px;user-select:none}
.tabs label:hover{color:var(--ink);background:var(--wash)}
.switches{display:flex;flex-wrap:wrap;gap:8px}
.switch{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;
background:var(--surface);font-size:13px;line-height:18px}
.switch .cap{padding:5px 10px;color:var(--muted)}
.switch label{padding:5px 12px;cursor:pointer;color:var(--ink2);border-left:1px solid var(--border);
user-select:none}
.switch label:hover{color:var(--ink)}
.sr{position:absolute;opacity:0;width:1px;height:1px;margin:0;pointer-events:none}
section.tab{display:flex;flex-direction:column;gap:%(gap)dpx}
.row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:%(gap)dpx;align-items:stretch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:12px 16px;min-width:0;display:flex;flex-direction:column}
.card h3{font-size:15px;line-height:20px;margin:0;font-weight:650}
.card .what{font-size:12.5px;line-height:17px;color:var(--ink2);margin:2px 0 8px}
.card.solo>h3,.card.solo>.head{margin-bottom:8px}
.card .head{display:flex;align-items:center;justify-content:space-between;gap:12px}
.card h4{font-size:13px;line-height:18px;margin:14px 0 6px;color:var(--ink2);font-weight:600}
.card>details,.card>div>details{margin-top:8px}
@media (min-width:981px){.card h3,.card .what{white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}}
svg.chart{display:block;max-width:100%%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:%(font)dpx;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
svg .lbl{fill:var(--ink2)}
svg .strong{fill:var(--ink);font-weight:600}
svg .gl{stroke:var(--grid);stroke-width:1}
svg .ax{stroke:var(--axis);stroke-width:1}
svg .ref{stroke:var(--ink2);stroke-width:1;opacity:.55}
svg .guide{stroke:var(--axis);stroke-width:1;stroke-dasharray:4 3}
svg .data{stroke:var(--data);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .model{stroke:var(--model);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .faded{stroke:var(--model);stroke-width:2;fill:none;opacity:.3;stroke-linejoin:round;stroke-linecap:round}
svg .dfaded{stroke:var(--data);stroke-width:2;fill:none;opacity:.3;stroke-linejoin:round;stroke-linecap:round}
svg .ddot{fill:var(--data);stroke:var(--surface);stroke-width:1.5}
svg .mdot{fill:var(--model);stroke:var(--surface);stroke-width:1.5}
svg .dsl{stroke:var(--other);stroke-width:1.5;fill:none;stroke-linejoin:round}
svg .dsw{display:none;fill:var(--band);stroke:none}
svg .mb{fill:var(--massbar)}
svg .own,svg .keep{fill:var(--model)}
svg .via,svg .floor{fill:var(--model-soft)}
svg .drop{fill:var(--other)}
svg .mk{fill:var(--ink);stroke:var(--surface);stroke-width:2}
svg .tol{fill:var(--tol)}
svg .shapband{fill:var(--model);opacity:.14}
svg .ref1{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3;fill:none}
svg .ib0,svg .ib1,svg .ib2,svg .ib3{stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .ib0{stroke:var(--i0)}svg .ib1{stroke:var(--i1)}svg .ib2{stroke:var(--i2)}svg .ib3{stroke:var(--i3)}
svg .dash{stroke-dasharray:4 3}
svg .exl{stroke:var(--ink2);stroke-width:2;fill:none}
svg .acl{stroke:var(--ink2);stroke-width:2;stroke-dasharray:4 3;fill:none}
svg .ibw0{fill:var(--i0);opacity:.14}svg .ibw1{fill:var(--i1);opacity:.14}svg .ibw2{fill:var(--i2);opacity:.14}svg .ibw3{fill:var(--i3);opacity:.14}
svg .fade{opacity:.3}
svg .shapmean{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3;fill:none}
svg .hbar{fill:var(--model-soft)}
svg .flagdot{fill:var(--crit);stroke:var(--surface);stroke-width:1.5}
svg .neardot{fill:var(--warn);stroke:var(--surface);stroke-width:1.5}
svg .bandbox{fill:var(--band)}
svg .hit{fill:transparent}
svg .hit:hover{fill:var(--wash)}
table{border-collapse:collapse;width:100%%;font-size:13px;line-height:18px}
th,td{padding:5px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:baseline}
th{color:var(--ink2);font-weight:600;white-space:nowrap}
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.tw{overflow-x:auto}
.tw.scroll{max-height:440px;overflow-y:auto}
.tw.short{max-height:250px;overflow-y:auto}
.tw.scroll thead th,.tw.short thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.kv td:first-child{color:var(--ink2);width:32%%;padding-right:12px}
.tinted th:last-child,.tinted td:last-child{padding-right:8px}
td.g1{background:color-mix(in srgb,var(--hi) 20%%,transparent)}
td.g2{background:color-mix(in srgb,var(--hi) 40%%,transparent)}
td.g3{background:color-mix(in srgb,var(--hi) 60%%,transparent)}
td.g4{background:color-mix(in srgb,var(--hi) 80%%,transparent)}
td.g5{background:var(--hi)}
td.y1{background:color-mix(in srgb,var(--lo) 20%%,transparent)}
td.y2{background:color-mix(in srgb,var(--lo) 40%%,transparent)}
td.y3{background:color-mix(in srgb,var(--lo) 60%%,transparent)}
td.y4{background:color-mix(in srgb,var(--lo) 80%%,transparent)}
td.y5{background:var(--lo)}
details>summary{cursor:pointer;color:var(--ink2);font-size:13px;line-height:20px}
details[open]>summary{margin-bottom:6px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;line-height:18px;font-weight:600;
padding:0 8px 0 3px;border-radius:10px;border:1px solid var(--border);white-space:nowrap;color:var(--ink2)}
.badge i{font-style:normal;display:inline-block;width:14px;height:14px;border-radius:7px;color:#fff;
font-size:10px;line-height:14px;text-align:center}
.b-critical i{background:var(--crit)}.b-warning i{background:var(--warn);color:#0b0b0b}
.b-pass i{background:var(--good)}.b-info i{background:var(--muted)}
.counts{display:flex;gap:6px;flex-wrap:wrap}
.issues{list-style:none;margin:0;padding:0}
.issues li{display:grid;grid-template-columns:86px minmax(0,1fr);gap:10px;align-items:baseline;
padding:5px 0;border-top:1px solid var(--grid);font-size:13px;line-height:18px}
.issues li:first-child{border-top:0;padding-top:0}
.issues .badge{justify-self:start}
.issues .t{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.link{color:var(--link);cursor:pointer;font-size:13px;line-height:18px}
.link:hover{text-decoration:underline}
.more{margin-top:auto;padding-top:6px}
.ok{color:var(--good-ink);font-size:13px;margin:4px 0}
.note{color:var(--ink2);font-size:13px;margin:4px 0}
.muted{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.fxgrid{display:grid;grid-template-columns:%(rail)dpx minmax(0,1fr);gap:%(gap)dpx;align-items:start}
.rail{position:sticky;top:64px;max-height:calc(100vh - 134px);overflow-y:auto;padding:6px}
.rail .rl{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 8px;padding:7px 10px 8px;
border-radius:8px;cursor:pointer;user-select:none}
.rail .rl:hover{background:var(--wash)}
.rail .nm{font-size:13.5px;line-height:18px;font-weight:600;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.rail .pc{font-size:12px;line-height:18px;color:var(--ink2);font-variant-numeric:tabular-nums}
.rail .bars{grid-column:1/-1;display:flex;height:5px;border-radius:3px;overflow:hidden;
background:var(--grid)}
.rail .bars i{display:block;height:5px}
.rail .bars .o{background:var(--model)}.rail .bars .v{background:var(--model-soft)}
.rail .w{grid-column:1/-1;font-size:11.5px;line-height:15px;color:var(--muted);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.rail .cap{font-size:11.5px;line-height:15px;color:var(--muted);padding:12px 10px 4px}
.rail .unused .nm{color:var(--muted);font-weight:500}
.rail .pair .nm{white-space:normal;overflow-wrap:anywhere}
.ttgrid{display:grid;grid-template-columns:%(list)dpx minmax(0,1fr);gap:%(gap)dpx;align-items:start}
.tlist .tw{max-height:calc(100vh - 210px);overflow-y:auto}
.tlist .tw thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.tlist td label{display:block;cursor:pointer}.tlist td:first-child{white-space:nowrap}
.tview{position:sticky;top:64px}
@media (max-width:980px){.row,.fxgrid,.ttgrid{grid-template-columns:minmax(0,1fr)}
.rail,.tview{position:static;max-height:none}.tlist .tw{max-height:420px}}
@media print{:root{%(light)s}body{background:#fff}.bar,.rail,.tlist{display:none!important}
section.tab{display:flex!important;break-before:page}.fxp,.tv{display:block!important}
.row,.fxgrid,.ttgrid{grid-template-columns:minmax(0,1fr)}
.card{break-inside:avoid}.tw.scroll,.tw.short,.tlist .tw{max-height:none;overflow:visible}}
"""

_CSS_EXTRA = """
svg .whisker{stroke:var(--model);stroke-width:1.5;opacity:.75}
svg .nsdot{fill:var(--other);stroke:var(--surface);stroke-width:1.5}
svg .tn{fill:var(--wash)}
svg .tg1{fill:color-mix(in srgb,var(--hi) 25%,transparent)}
svg .tg2{fill:color-mix(in srgb,var(--hi) 45%,transparent)}
svg .tg3{fill:color-mix(in srgb,var(--hi) 65%,transparent)}
svg .tg4{fill:color-mix(in srgb,var(--hi) 85%,transparent)}
svg .tg5{fill:var(--hi)}
svg .ty1{fill:color-mix(in srgb,var(--lo) 25%,transparent)}
svg .ty2{fill:color-mix(in srgb,var(--lo) 45%,transparent)}
svg .ty3{fill:color-mix(in srgb,var(--lo) 65%,transparent)}
svg .ty4{fill:color-mix(in srgb,var(--lo) 85%,transparent)}
svg .ty5{fill:var(--lo)}
svg .tb1{fill:color-mix(in srgb,var(--data) 8%,transparent)}
svg .tb2{fill:color-mix(in srgb,var(--data) 16%,transparent)}
svg .tb3{fill:color-mix(in srgb,var(--data) 26%,transparent)}
svg .tb4{fill:color-mix(in srgb,var(--data) 38%,transparent)}
svg .tb5{fill:color-mix(in srgb,var(--data) 52%,transparent)}
.fxv>.fxp{display:flex;flex-direction:column;gap:16px}
.clc{display:contents}
svg .dbar{fill:var(--data)}
"""

# --------------------------------------------------------------------------- formatting -------


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return _e(v)
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or (ax >= 1e3 and x == round(x)):
        return f"{x:,.0f}"
    if ax >= 1e3:
        return f"{x:,.1f}"
    if ax < 1e-4:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def _pct(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x * 100:.{digits}f}%" if math.isfinite(x) else "—"


def _fixed(v: Any, digits: int = 3) -> str:
    """Fixed decimals, so 1 prints as 1.000 beside 0.993."""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.{digits}f}" if math.isfinite(x) else "—"


def _amt(v: Any) -> str:
    """An amount (exposure, expected count, support) with one decimal, so a column aligns."""
    if v is None:
        return "—"
    x = float(v)
    return f"{x:,.1f}" if math.isfinite(x) else "—"


def _column(values: Sequence[Any]) -> list[str]:
    """One format for a column of amounts: whole numbers when any passes 1,000, so the column
    aligns; ``_num`` otherwise."""
    big = any(v is not None and math.isfinite(float(v)) and abs(float(v)) >= 1000 for v in values)
    if not big:
        return [_num(v) for v in values]
    return ["—" if v is None or not math.isfinite(float(v)) else f"{float(v):,.0f}" for v in values]


def _mult(v: Any) -> str:
    """A multiplicative effect: ×0.80, ×1.96, ×12.3."""
    if v is None:
        return "—"
    x = float(v)
    if not math.isfinite(x):
        return "—"
    return f"×{x:.2f}" if x < 10 else f"×{x:.1f}"


def _signed(v: Any) -> str:
    """A z score with a true minus sign."""
    if v is None:
        return "—"
    x = float(v)
    return f"{x:+.1f}".replace("-", "−") if math.isfinite(x) else "—"


def _short(label: str, n: int = 14) -> str:
    return label if len(label) <= n else label[: max(n - 1, 1)] + "…"


def _text_w(s: str, size: float = _FONT) -> float:
    """Approximate rendered width of ``s`` in px (system-ui, digits and lower case)."""
    return len(s) * size * 0.56


# --------------------------------------------------------------------------- scales -----------


class _Scale(Protocol):
    ticks: list[float]

    def __call__(self, v: float) -> float: ...


_Fmt = Callable[[float], str]


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Nice ticks inside ``[lo, hi]``."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else [0.0]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    out: list[float] = []
    t = math.ceil(lo / step - 1e-9) * step
    while t <= hi + 1e-9 * step and len(out) < 50:
        out.append(round(t, 12))
        t += step
    return out


class _Lin:
    """Linear scale ``[lo, hi] -> [a, b]`` padded by ``pad`` of the span; ``zero`` floors
    non-negative data at 0."""

    def __init__(
        self,
        lo: float,
        hi: float,
        a: float,
        b: float,
        *,
        pad: float = 0.06,
        zero: bool = False,
        n: int = 4,
    ) -> None:
        if not (math.isfinite(lo) and math.isfinite(hi)):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
        span = hi - lo
        lo2, hi2 = lo - span * pad, hi + span * pad
        if zero and lo >= 0:
            lo2 = 0.0
        self.lo, self.hi, self.a, self.b = lo2, hi2, a, b
        self.ticks = _ticks(lo2, hi2, n)

    def __call__(self, v: float) -> float:
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)


# Tick sets for a log axis of multipliers, from fine to coarse; the finest set that puts no more
# than `max_ticks` ticks on the axis is used.
_LOG_TICKS = (
    (0.8, 0.9, 1.0, 1.1, 1.25),
    (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0),
    (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    (0.01, 0.1, 1.0, 10.0, 100.0),
    (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4),
)


class _Log:
    """Log scale for multipliers. The axis spans at least ``span`` either side of 1, so a small
    effect looks small."""

    def __init__(
        self, lo: float, hi: float, a: float, b: float, *, span: float = 1.12, max_ticks: int = 5
    ) -> None:
        lo = min(max(lo, 1e-9), 1 / span)
        hi = max(hi, span)
        llo, lhi = math.log(lo), math.log(hi)
        pad = (lhi - llo) * 0.05
        self.llo, self.lhi, self.a, self.b = llo - pad, lhi + pad, a, b
        # The finest tick set that spans at least 40% of the axis in at most `max_ticks` ticks;
        # failing that, the finest one thinned (every k-th tick from 1) to fit.
        whole: list[list[float]] = []
        thinned: list[list[float]] = []
        for tick_set in _LOG_TICKS:
            inside = [t for t in tick_set if self.llo <= math.log(t) <= self.lhi]
            fits = len(inside) <= max_ticks
            if not fits:
                step = math.ceil(len(inside) / max_ticks)
                at = inside.index(1.0) if 1.0 in inside else 0
                inside = [t for i, t in enumerate(inside) if (i - at) % step == 0]
            if len(inside) >= 2 and math.log(inside[-1] / inside[0]) >= 0.4 * (self.lhi - self.llo):
                (whole if fits else thinned).append(inside)
        self.ticks = (whole or thinned or [[1.0]])[0]

    def __call__(self, v: float) -> float:
        lv = math.log(max(v, 1e-12))
        return self.a + (lv - self.llo) / (self.lhi - self.llo) * (self.b - self.a)


def _fv(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        x = float(v)
        if math.isfinite(x):
            out.append(x)
    return out


def _mult_tick(t: float) -> str:
    return f"×{t:g}"


def _fixed2(t: float) -> str:
    return _fixed(t, 2)


def _pct0(t: float) -> str:
    return _pct(t, 0)


# --------------------------------------------------------------------------- SVG --------------


def _d(pts: Sequence[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


class _Svg:
    """An SVG drawn in CSS pixels."""

    def __init__(self, w: float, h: float, label: str) -> None:
        self.w, self.h, self.label = w, h, label
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, *, anchor: str = "start", cls: str = "") -> None:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{a}{c}>{_e(s)}</text>')

    def hline(self, x1: float, x2: float, y: float, cls: str) -> None:
        self.add(f'<line class="{cls}" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')

    def vline(self, x: float, y1: float, y2: float, cls: str) -> None:
        self.add(f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>')

    def path(self, pts: Sequence[tuple[float, float]], cls: str) -> None:
        if len(pts) > 1:
            self.add(f'<path class="{cls}" d="{_d(pts)}"/>')

    def hit(self, x: float, y: float, w: float, h: float, title: str) -> None:
        self.add(
            f'<rect class="hit" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}">'
            f"<title>{_e(title)}</title></rect>"
        )

    def group(self, cls: str, parts: list[str]) -> None:
        if parts:
            self.add(f'<g class="{cls}">' + "".join(parts) + "</g>" if cls else "".join(parts))

    def render(self) -> str:
        return (
            f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
            f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}" role="img" '
            f'aria-label="{_e(self.label)}">' + "".join(self.parts) + "</svg>"
        )


def _yaxis(svg: _Svg, scale: _Scale, x0: float, x1: float, fmt: _Fmt) -> None:
    """Gridlines, and tick labels right-aligned 6 px left of the plot."""
    for t in scale.ticks:
        y = scale(t)
        svg.hline(x0, x1, y, "gl")
        svg.text(x0 - 6, y + 4, fmt(t), anchor="end")


def _label_w(scale: _Scale, fmt: _Fmt) -> float:
    return max((_text_w(fmt(t)) for t in scale.ticks), default=20.0)


def _legend(svg: _Svg, x: float, y: float, items: Sequence[tuple[str, str, str]]) -> float:
    """``(kind, class, label)`` items left to right from ``x``; ``kind`` is line, dot or box.
    Returns where the legend ends."""
    for kind, cls, label in items:
        if kind == "line":
            svg.add(
                f'<line class="{cls}" x1="{x:.1f}" x2="{x + 16:.1f}" y1="{y - 4:.1f}" '
                f'y2="{y - 4:.1f}"/>'
            )
        elif kind == "dot":
            svg.add(f'<circle class="{cls}" cx="{x + 8:.1f}" cy="{y - 4:.1f}" r="4"/>')
        else:
            svg.add(
                f'<rect class="{cls}" x="{x + 3:.1f}" y="{y - 9:.1f}" width="10" height="10" '
                'rx="2"/>'
            )
        svg.text(x + 22, y, label, cls="lbl")
        x += 22 + _text_w(label) + 18
    return x


def _others(scored: list[dict[str, Any]], i: int) -> str:
    """The labels of every dataset but the ``i``-th."""
    return ", ".join(str(d["label"]) for j, d in enumerate(scored) if j != i)


def _ds_legend(
    svg: _Svg,
    x: float,
    y: float,
    n: int,
    items_for: Callable[[int], list[tuple[str, str, str]]],
    *,
    fit: float | None = None,
) -> None:
    """A legend for each of ``n`` datasets that names it and the others, shown with it by the
    page's dataset switch. With ``fit``, trailing items that would overflow that width go."""
    for i in range(n):
        items = items_for(i)
        while (
            fit is not None
            and len(items) > 1
            and (sum(40 + _text_w(label) for _, _, label in items) - 18 > fit)
        ):
            items.pop()
        g = _Svg(svg.w, svg.h, "")
        _legend(g, x, y, items)
        svg.group(f"ds ds-{i}", g.parts)


def _bar(
    cls: str, x: float, y_top: float, w: float, y_base: float, r: float = 2.0, title: str = ""
) -> str:
    """A column from the baseline up to ``y_top``, rounded at the data end only."""
    h = y_base - y_top
    if h <= 0.2 or w <= 0:
        return ""
    r = min(r, w / 2, h)
    d = (
        f"M{x:.1f},{y_base:.1f}V{y_top + r:.1f}Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f}"
        f"H{x + w - r:.1f}Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f}V{y_base:.1f}Z"
    )
    t = f"<title>{_e(title)}</title>" if title else ""
    return f'<path class="{cls}" d="{d}">{t}</path>' if t else f'<path class="{cls}" d="{d}"/>'


def _segments(
    svg: _Svg,
    pts: Sequence[tuple[float, float]],
    faded: Sequence[bool],
    cls: str = "model",
    faded_cls: str = "faded",
) -> None:
    """A joined line whose segments touching a faded point are drawn ``faded_cls``."""
    if len(pts) == 1:
        svg.add(f'<circle class="mdot" cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="4"/>')
        return
    run: list[tuple[float, float]] = [pts[0]] if pts else []
    state: bool | None = None
    for j in range(1, len(pts)):
        f = faded[j - 1] or faded[j]
        if state is None or f == state:
            run.append(pts[j])
        else:
            svg.path(run, faded_cls if state else cls)
            run = [pts[j - 1], pts[j]]
        state = f
    svg.path(run, faded_cls if state else cls)


def _xlabels(
    svg: _Svg,
    xs: Sequence[float],
    labels: Sequence[str],
    y: float,
    slot: float,
    *,
    max_chars: int = 16,
) -> None:
    """Category labels under a plot at a regular step, never overlapping; the last always."""
    n = len(xs)
    if n == 0:
        return
    texts = [_short(s, max_chars) for s in labels]
    widths = [_text_w(t) for t in texts]
    step = max(1, math.ceil((max(widths) + 10) / max(slot, 1e-6)))
    cand = list(range(0, n, step))
    if cand[-1] != n - 1:
        cand.append(n - 1)

    def clear(i: int, j: int) -> bool:
        return xs[j] - widths[j] / 2 >= xs[i] + widths[i] / 2 + 8

    placed: list[int] = []
    for i in cand:
        if i == n - 1:
            while placed and not clear(placed[-1], i):
                placed.pop()
            placed.append(i)
        elif not placed or clear(placed[-1], i):
            placed.append(i)
    for i in placed:
        svg.text(xs[i], y, texts[i], anchor="middle")


def _none(text: str = "No data.") -> str:
    return f'<p class="muted">{_e(text)}</p>'


# --------------------------------------------------------------------------- page kit ---------


def _table(
    cols: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric: Iterable[int] = (),
    row_class: Sequence[str] | None = None,
    scroll: str = "",
    cls: str = "",
    tones: Sequence[Sequence[str]] | None = None,
) -> str:
    """Rows of already-escaped cell HTML; ``scroll`` is ``scroll`` or ``short`` for a box;
    ``tones`` adds a class per cell (``_tones``)."""
    num = set(numeric)
    head = "".join(
        f'<th class="{"n" if i in num else ""}">{_e(c)}</th>' for i, c in enumerate(cols)
    )
    body = []
    for r_i, row in enumerate(rows):
        rc = f' class="{row_class[r_i]}"' if row_class and row_class[r_i] else ""
        tone = tones[r_i] if tones is not None and r_i < len(tones) else []
        tds = []
        for i, c in enumerate(row):
            classes = " ".join(
                x for x in ("n" if i in num else "", tone[i] if i < len(tone) else "") if x
            )
            tds.append(f'<td class="{classes}">{c}</td>')
        body.append(f"<tr{rc}>" + "".join(tds) + "</tr>")
    tc = f' class="{cls}"' if cls else ""
    return (
        f'<div class="tw{" " + scroll if scroll else ""}"><table{tc}><thead><tr>{head}</tr>'
        f"</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _kv(data: Any, depth: int = 0) -> str:
    """A nested dict as key/value tables; lists of dicts as tables (capped)."""
    if isinstance(data, dict):
        if not data:
            return '<span class="muted">—</span>'
        rows = "".join(
            f"<tr><td>{_e(k)}</td><td>{_kv(v, depth + 1)}</td></tr>" for k, v in data.items()
        )
        return f'<table class="kv">{rows}</table>'
    if isinstance(data, list):
        if not data:
            return '<span class="muted">[]</span>'
        if all(isinstance(v, dict) for v in data) and depth < 4:
            keys: list[str] = []
            for v in data:
                keys += [k for k in v if k not in keys]
            cells = [[_short_val(v.get(k)) for k in keys] for v in data[:_RANKED_ROWS]]
            return _table(keys, cells)
        text = ", ".join(_short_val(v) for v in data[:60])
        return text + (f" … ({len(data)} values)" if len(data) > 60 else "")
    return _short_val(data)


def _short_val(v: Any) -> str:
    if v is None:
        return '<span class="muted">None</span>'
    if isinstance(v, float):
        return _num(v)
    if isinstance(v, (dict, list)):
        text = json.dumps(v, default=str)
        return f'<span class="mono">{_e(text if len(text) <= 160 else text[:159] + "…")}</span>'
    return _e(v)


def _badge(status: str, text: str | None = None) -> str:
    icon = {"critical": "!", "warning": "!", "pass": "✓", "info": "i"}.get(status, "?")
    word = text if text is not None else {"info": "note"}.get(status, status)
    return f'<span class="badge b-{_e(status)}"><i aria-hidden="true">{icon}</i>{_e(word)}</span>'


def _card(
    title: str,
    body: str,
    *,
    what: str = "",
    extra: str = "",
    cls: str = "",
    titles: list[str] | None = None,
) -> str:
    """A card; ``titles`` gives one title per dataset, shown with it by the dataset switch
    (``title`` is then the hover text)."""
    shown = (
        "".join(f'<span class="ds ds-{i}">{_e(t)}</span>' for i, t in enumerate(titles))
        if titles
        else _e(title)
    )
    head = f'<h3 title="{_e(title)}">{shown}</h3>'
    if extra:
        head = f'<div class="head">{head}{extra}</div>'
    cap = f'<p class="what" title="{_e(what)}">{_e(what)}</p>' if what else ""
    classes = "card" + ("" if what else " solo") + (f" {cls}" if cls else "")
    return f'<div class="{classes}">{head}{cap}{body}</div>'


def _row(*cards: str) -> str:
    return '<div class="row">' + "".join(cards) + "</div>"


class _Radio:
    """A CSS-only radio group. Its inputs sit first in ``.app``; its labels may sit in ``.bar``
    or anywhere in ``.body``; content tagged ``{name} {name}-{i}`` shows only for option i."""

    def __init__(self, name: str, n: int, default: int = 0) -> None:
        self.name, self.n, self.default = name, max(n, 1), default

    def inputs(self) -> str:
        return "".join(
            f'<input class="sr" type="radio" name="{self.name}" id="{self.name}-{i}"'
            f"{' checked' if i == self.default else ''}>"
            for i in range(self.n)
        )

    def label(self, i: int, html: str, cls: str = "", title: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        t = f' title="{_e(title)}"' if title else ""
        return f'<label for="{self.name}-{i}"{c}{t}>{html}</label>'

    def tag(self, i: int) -> str:
        return f"{self.name} {self.name}-{i}"

    def css(self, selected: str) -> str:
        out = []
        for i in range(self.n):
            on = f"#{self.name}-{i}:checked"
            lab = f"label[for={self.name}-{i}]"
            out.append(f"{on}~.body .{self.name}:not(.{self.name}-{i}){{display:none}}")
            if selected:
                out.append(f"{on}~.bar {lab},{on}~.body {lab}{{{selected}}}")
            out.append(
                f"#{self.name}-{i}:focus-visible~.bar {lab},#{self.name}-{i}:focus-visible~"
                f".body {lab}{{outline:2px solid var(--data);outline-offset:-2px}}"
            )
        return "".join(out)


def _dataset_css(n: int) -> str:
    """The selected dataset's series in colour with its ±2σ band; its table rows washed."""
    out = []
    for i in range(n):
        on = f"#ds-{i}:checked~.body"
        out.append(
            f"{on} .dsl-{i}{{stroke:var(--data);stroke-width:2.5}}{on} .dsw-{i}{{display:inline}}"
            f"{on} .dsr-{i} td{{background:var(--wash)}}"
            f"{on} .dsr-{i} td:first-child{{box-shadow:-8px 0 0 var(--wash)}}"
            f"{on} .dsr-{i} td:last-child{{box-shadow:8px 0 0 var(--wash)}}"
        )
    return "".join(out)


# --------------------------------------------------------------------------- report access ----


def _views(r: dict[str, Any]) -> list[dict[str, Any]]:
    """What the page draws: ``r`` for a GLM, and for a multinomial report one one-vs-rest view
    per class with the shape of a GLM report, so every chart serves both."""
    if not r["model"].get("multiclass"):
        return [{**r, "class": None, "suffix": "", "is_reference": False}]
    model = {**r["model"], "link": "logit", "family_kind": "binomial", "exposure_column": None}
    return [
        {
            **r,
            "model": model,
            "class": c["class"],
            "suffix": f" · {c['class']}",
            "is_reference": c["is_reference"],
            "datasets": c["datasets"],
            "features": c["features"],
            "interactions": c["interactions"],
            "coefficients": c["coefficients"],
        }
        for c in r["classes"]
    ]


def _per_class(
    views: list[dict[str, Any]], cl: _Radio | None, render: Callable[[dict[str, Any]], str]
) -> str:
    """``render`` once per view, each shown with its class; layout-neutral in a grid."""
    if cl is None:
        return render(views[0])
    return "".join(f'<div class="clc {cl.tag(k)}">{render(v)}</div>' for k, v in enumerate(views))


def _scored(r: dict[str, Any]) -> list[dict[str, Any]]:
    return list(r["datasets"])


def _mass_name(r: dict[str, Any]) -> str:
    m = r["model"]
    if m.get("exposure_column"):
        return "exposure"
    return "weight" if m.get("weights_column") else "rows"


def _rate_name(r: dict[str, Any]) -> str:
    m = r["model"]
    if m["link"] == "logit":
        return "probability"
    return "rate per unit exposure" if m.get("exposure_column") else "mean response"


def _actual_name(r: dict[str, Any]) -> str:
    m = r["model"]
    if m.get("family_kind") == "poisson" and m.get("exposure_column"):
        return "claims"
    return "events" if m["link"] == "logit" else "actual"


def _feature(r: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((f for f in r["features"] if f["feature"] == name), None)


def _is_mult(r: dict[str, Any]) -> bool:
    return r["model"]["link"] in ("log", "logit")


def _effect_name(r: dict[str, Any]) -> str:
    return {"log": "relativity", "logit": "odds multiplier"}.get(r["model"]["link"], "effect")


def _stars(p: Any) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def _p(v: Any) -> str:
    if v is None:
        return "—"
    x = float(v)
    if not math.isfinite(x):
        return "—"
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"


def _heat(v: float | None, neutral: float, scale: float) -> str:
    """The tint class of a value: green above ``neutral``, yellow below, in five steps."""
    if v is None:
        return "tn"
    if neutral == 1.0:
        size = abs(math.log(max(v, 1e-9))) / (math.log(1.0 + scale) or 1.0)
    else:
        size = abs(v - neutral) / (scale or 1.0)
    step = min(max(1, math.ceil(5 * min(size, 1.0) ** 0.5)), 5)
    if abs(v - neutral) < 1e-12:
        return "tn"
    return f"tg{step}" if v > neutral else f"ty{step}"


def _heat_scale(values: Iterable[float | None], mult: bool) -> float:
    vals = _fv(values)
    if not vals:
        return 1.0
    if mult:
        return max(max(abs(math.log(max(v, 1e-9))) for v in vals), 0.05)
    return max(max(abs(v) for v in vals), 1e-9)


# --------------------------------------------------------------------------- charts -----------


def calibration_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Deciles of predicted rate: actual and predicted (top; the selected dataset) and A/E for
    every dataset (bottom; the selected one in colour with its ±2σ band)."""
    scored = _scored(r)
    idx = list(range(len(scored)))
    lifts = {i: scored[i]["deciles"] for i in idx}
    n = max((len(v) for v in lifts.values()), default=0)
    if n == 0:
        return _none()
    rates = _fv(b.get(key) for v in lifts.values() for b in v for key in ("actual", "predicted"))
    aes = _fv(b.get("ae") for v in lifts.values() for b in v)
    two_sd = min(2 * max(_fv(b.get("sd") for v in lifts.values() for b in v), default=0.0), 0.25)
    t0, gap, foot = 24.0, 28.0, 22.0
    usable = h - t0 - gap - foot
    top_h = round(usable * 0.56)
    b0 = t0 + top_h + gap
    b1 = h - foot
    ts = _Lin(0.0, max(rates or [1.0]), t0 + top_h, t0, zero=True, n=4)
    bs = _Lin(min(aes + [1 - two_sd, 0.95]), max(aes + [1 + two_sd, 1.05]), b1, b0, pad=0.06, n=3)

    def rate_fmt(v: float) -> str:
        return _num(v, 3)

    left = 6 + max(_label_w(ts, rate_fmt), _label_w(bs, _fixed2)) + 6
    right = w - 46
    bw = (right - left) / n
    xs = [left + (j + 0.5) * bw for j in range(n)]
    svg = _Svg(w, h, "calibration by decile of predicted rate")

    def items(i: int) -> list[tuple[str, str, str]]:
        label = str(scored[i]["label"])
        out = [("line", "data", f"actual ({label})"), ("line", "model", f"predicted ({label})")]
        if len(scored) > 1:
            out.append(("line", "dsl", f"A/E ({_others(scored, i)})"))
        return out

    _ds_legend(svg, left, 12, len(scored), items)
    _yaxis(svg, ts, left, right, rate_fmt)
    _yaxis(svg, bs, left, right, _fixed2)
    svg.hline(left, right, bs(1.0), "ref")
    svg.hline(left, right, t0 + top_h, "ax")
    svg.text(left, b0 - 8, "A/E by decile", cls="lbl")

    def clip(v: float) -> float:
        return bs(min(max(v, bs.lo), bs.hi))

    for i in idx:
        lift = lifts[i]
        g = _Svg(w, h, "")
        pre = [
            (xs[j], ts(b["predicted"]))
            for j, b in enumerate(lift)
            if b.get("predicted") is not None
        ]
        act = [(xs[j], ts(b["actual"])) for j, b in enumerate(lift) if b.get("actual") is not None]
        g.path(pre, "model")
        g.path(act, "data")
        for x, y in act:
            g.add(f'<circle class="ddot" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')
        if act:
            g.text(act[-1][0] + 8, act[-1][1] + 4, _num(lift[-1].get("actual"), 3), cls="lbl")
        svg.group(f"ds ds-{i}", g.parts)
        if any(b.get("sd") for b in lift):
            up = [(xs[j], clip(1 + 2 * (b.get("sd") or 0.0))) for j, b in enumerate(lift)]
            dn = [(xs[j], clip(1 - 2 * (b.get("sd") or 0.0))) for j, b in enumerate(lift)]
            svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
    for i in idx:
        pts = [(xs[j], clip(b["ae"])) for j, b in enumerate(lifts[i]) if b.get("ae") is not None]
        svg.path(pts, f"dsl dsl-{i}")
    svg.hline(left, right, b1, "ax")
    for j in range(n):
        svg.text(xs[j], b1 + 16, str(j + 1), anchor="middle")
    svg.text(right + 8, b1 + 16, "decile", cls="lbl")
    for j in range(n):
        lines = [f"decile {j + 1}"]
        for i in idx:
            if j < len(lifts[i]):
                b = lifts[i][j]
                lines.append(
                    f"{scored[i]['label']}: actual {_num(b.get('actual'), 4)}, predicted "
                    f"{_num(b.get('predicted'), 4)}, A/E {_fixed(b.get('ae'))}, "
                    f"z {_signed(b.get('z'))}"
                )
        svg.hit(left + j * bw, t0, bw, b1 - t0, "\n".join(lines))
    return svg.render()


def _bars_chart(
    rows: Sequence[tuple[str, float, str]],
    w: float,
    h: float | None = None,
    *,
    row_h: float = 22.0,
    fmt: Callable[[float], str] = _num,
    cls: str = "own",
) -> str:
    """Horizontal bars ``(name, value, hover)``, largest first."""
    if not rows:
        return _none()
    head = 6.0
    if h is None:
        h = head + row_h * len(rows)
    row_h = (h - head) / len(rows)
    name_w = min(max(_text_w(str(x[0])) for x in rows) + 14, w * 0.3)
    x0, x1 = name_w, w - 64
    peak = max(x[1] for x in rows) or 1.0
    svg = _Svg(w, h, "bars")
    for k, (name, value, hover) in enumerate(rows):
        y = head + k * row_h
        svg.text(
            name_w - 8,
            y + row_h / 2 + 4,
            _short(str(name), int(name_w / 7)),
            anchor="end",
            cls="lbl",
        )
        bw = max(value, 0.0) / peak * (x1 - x0)
        bar_h = min(row_h - 6, 14.0)
        svg.add(
            f'<rect class="{cls}" x="{x0:.1f}" y="{y + (row_h - bar_h) / 2:.1f}" width="{bw:.1f}" '
            f'height="{bar_h:.1f}" rx="2"/>'
        )
        svg.text(x0 + bw + 6, y + row_h / 2 + 4, fmt(value), cls="lbl")
        svg.hit(0, y, w, row_h, hover)
    return svg.render()


def importance_chart(r: dict[str, Any], w: float, h: float | None = None, *, top: int = 9) -> str:
    """Each fitted factor's share of the deviance the model explains, largest first."""
    feats = [f for f in r["features"] if f["in_model"] and f.get("relative_importance") is not None]
    rest = feats[top:]
    shown = [
        (
            str(f["feature"]),
            float(f["relative_importance"]) / 100.0,
            f"{f['feature']}: {_pct(float(f['relative_importance']) / 100.0)} of the explained "
            f"deviance; {_p((f.get('significance') or {}).get('p'))} joint p-value",
        )
        for f in feats[:top]
    ]
    if rest:
        share = sum(float(f["relative_importance"]) for f in rest) / 100.0
        shown.append((f"{len(rest)} more", share, f"{len(rest)} more factors"))
    return _bars_chart(shown, w, h, fmt=_pct)


def multiples_chart(r: dict[str, Any], w: float, h: float, *, count: int = 4) -> str:
    """The fitted effects of the most important factors side by side on one shared scale."""
    mult = _is_mult(r)
    panels = []
    for f in r["features"]:
        e = f.get("effect")
        if not f["in_model"] or not e or not e.get("value"):
            continue
        order = list(range(len(e["x"])))
        if f["type"] == "categorical":
            order.sort(key=lambda i: e["value"][i])
        panels.append((f, e, order))
        if len(panels) == count:
            break
    if not panels:
        return _none("No fitted effects to show.")
    vals = _fv(v for _, e, _ in panels for v in e["value"])
    top, bottom = 44.0, h - 20
    ys: _Scale = (
        _Log(min(vals), max(vals), bottom, top, span=1.25, max_ticks=4)
        if mult
        else _Lin(min(vals), max(vals), bottom, top, n=3)
    )

    def fmt(t: float) -> str:
        return _mult_tick(t) if mult else _num(t, 2)

    lead = _label_w(ys, fmt) + 10
    gap = 18.0
    pw = (w - lead - gap * (len(panels) - 1)) / len(panels)
    svg = _Svg(w, h, "largest fitted effects")
    for p, (f, e, order) in enumerate(panels):
        left = lead + p * (pw + gap)
        right = left + pw
        for tk in ys.ticks:
            svg.hline(left, right, ys(tk), "gl")
            if p == 0:
                svg.text(left - 6, ys(tk) + 4, fmt(tk), anchor="end")
        svg.hline(left, right, ys(1.0 if mult else 0.0), "ref")
        n = len(order)
        slot = (right - left) / n
        xs = [left + (j + 0.5) * slot for j in range(n)]
        svg.path([(xs[j], ys(e["value"][c])) for j, c in enumerate(order)], "model")
        live = [e["value"][c] for c in order]
        span = (
            f"{_mult(min(live))} to {_mult(max(live))}"
            if mult
            else f"{_num(min(live), 3)} to {_num(max(live), 3)}"
        )
        chars = int(pw / (_FONT * 0.6))
        svg.text(left, 13, _short(str(f["feature"]), chars), cls="strong")
        svg.text(left, 29, span, cls="lbl")
        half = max(int((pw / 2 - 6) / (_FONT * 0.56)), 3)
        first, last = e["x"][order[0]], e["x"][order[-1]]
        svg.text(left, bottom + 15, _short(_label_of(first), half))
        if n > 1:
            svg.text(right, bottom + 15, _short(_label_of(last), half), anchor="end")
        for j, c in enumerate(order):
            v = _mult(e["value"][c]) if mult else _num(e["value"][c], 4)
            svg.hit(
                xs[j] - slot / 2,
                top,
                slot,
                bottom - top,
                f"{f['feature']} {_label_of(e['x'][c])}: {v}",
            )
    return svg.render()


def _label_of(x: Any) -> str:
    return _num(x, 4) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def variance_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Dispersion by band of fitted value: flat near the overall value when V(μ) is right."""
    scored = _scored(r)
    bands = {i: d["fit_bands"] for i, d in enumerate(scored)}
    n = max((len(v) for v in bands.values()), default=0)
    if n == 0:
        return _none()
    vals = _fv(b.get("dispersion") for v in bands.values() for b in v)
    phi = r["fit"].get("dispersion")
    ceiling = max(vals + [float(phi or 1.0), 1.0])
    top, bottom = 24.0, h - 34
    ys = _Lin(0.0, ceiling, bottom, top, zero=True, n=4)
    left = _label_w(ys, _fixed2) + 14
    right = w - 8
    slot = (right - left) / n
    xs = [left + (j + 0.5) * slot for j in range(n)]
    svg = _Svg(w, h, "dispersion by band of fitted value")
    _yaxis(svg, ys, left, right, _fixed2)

    def items(i: int) -> list[tuple[str, str, str]]:
        out = [("line", "data", str(scored[i]["label"])), ("line", "guide", "overall φ")]
        if len(scored) > 1:
            out.append(("line", "dsl", _others(scored, i)))
        return out

    _ds_legend(svg, left, 12, len(scored), items)
    if phi is not None:
        svg.hline(left, right, ys(float(phi)), "guide")
    if r["fit"].get("fixed_scale"):
        svg.hline(left, right, ys(1.0), "ref")
    for i in bands:
        pts = [
            (xs[j], ys(b["dispersion"]))
            for j, b in enumerate(bands[i])
            if b.get("dispersion") is not None
        ]
        svg.path(pts, f"dsl dsl-{i}")
    svg.hline(left, right, bottom, "ax")
    for j in range(0, n, max(1, n // 5)):
        b = bands[max(bands)][j]
        svg.text(xs[j], bottom + 16, _num(b.get("fitted_mean"), 3), anchor="middle")
    svg.text(
        (left + right) / 2,
        bottom + 30,
        f"fitted {_rate_name(r)} (bands of equal rows)",
        anchor="middle",
        cls="lbl",
    )
    for j in range(n):
        lines = [f"band {j + 1}"]
        for i in bands:
            if j < len(bands[i]):
                b = bands[i][j]
                lines.append(
                    f"{scored[i]['label']}: dispersion {_fixed(b.get('dispersion'))}, fitted "
                    f"{_num(b.get('fitted_mean'), 4)}, A/E {_fixed(b.get('ae'))}"
                )
        svg.hit(xs[j] - slot / 2, top, slot, bottom - top, "\n".join(lines))
    return svg.render()


def fitted_ae_chart(r: dict[str, Any], w: float, h: float) -> str:
    """A/E by twenty bands of fitted value, for every dataset, ±2σ for the selected one."""
    scored = _scored(r)
    bands = {i: d["fit_bands"] for i, d in enumerate(scored)}
    n = max((len(v) for v in bands.values()), default=0)
    if n == 0:
        return _none()
    aes = _fv(b.get("ae") for v in bands.values() for b in v)
    two_sd = min(2 * max(_fv(b.get("sd") for v in bands.values() for b in v), default=0.0), 0.5)
    top, bottom = 24.0, h - 34
    ys = _Lin(
        min(aes + [1 - two_sd, 0.95]), max(aes + [1 + two_sd, 1.05]), bottom, top, pad=0.05, n=4
    )
    left = _label_w(ys, _fixed2) + 14
    right = w - 8
    slot = (right - left) / n
    xs = [left + (j + 0.5) * slot for j in range(n)]
    svg = _Svg(w, h, "A/E by band of fitted value")
    _yaxis(svg, ys, left, right, _fixed2)

    def items(i: int) -> list[tuple[str, str, str]]:
        out = [("line", "data", str(scored[i]["label"])), ("box", "bandbox", "±2σ")]
        if len(scored) > 1:
            out.append(("line", "dsl", _others(scored, i)))
        return out

    _ds_legend(svg, left, 12, len(scored), items)
    svg.hline(left, right, ys(1.0), "ref")

    def clip(v: float) -> float:
        return ys(min(max(v, ys.lo), ys.hi))

    for i in bands:
        up = [(xs[j], clip(1 + 2 * (b.get("sd") or 0.0))) for j, b in enumerate(bands[i])]
        dn = [(xs[j], clip(1 - 2 * (b.get("sd") or 0.0))) for j, b in enumerate(bands[i])]
        if up:
            svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
    for i in bands:
        svg.path(
            [(xs[j], clip(b["ae"])) for j, b in enumerate(bands[i]) if b.get("ae") is not None],
            f"dsl dsl-{i}",
        )
    svg.hline(left, right, bottom, "ax")
    for j in range(0, n, max(1, n // 5)):
        b = bands[max(bands)][j]
        svg.text(xs[j], bottom + 16, _num(b.get("fitted_mean"), 3), anchor="middle")
    svg.text(
        (left + right) / 2,
        bottom + 30,
        f"fitted {_rate_name(r)} (bands of equal rows)",
        anchor="middle",
        cls="lbl",
    )
    for j in range(n):
        lines = [f"band {j + 1}"]
        for i in bands:
            if j < len(bands[i]):
                b = bands[i][j]
                lines.append(
                    f"{scored[i]['label']}: A/E {_fixed(b.get('ae'))}, z {_signed(b.get('z'))}"
                )
        svg.hit(xs[j] - slot / 2, top, slot, bottom - top, "\n".join(lines))
    return svg.render()


def lorenz_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Lorenz curves for every dataset, the selected one in colour; Gini beside."""
    scored = _scored(r)
    left, top = 46.0, 10.0
    side = min(h - top - 30, w - left - 150)
    right, bottom = left + side, top + side
    svg = _Svg(w, h, "Lorenz curve")
    xs = _Lin(0.0, 1.0, left, right, pad=0.0, n=4)
    ys = _Lin(0.0, 1.0, bottom, top, pad=0.0, n=4)
    _yaxis(svg, ys, left, right, _pct0)
    for t in xs.ticks:
        svg.text(xs(t), bottom + 16, _pct(t, 0), anchor="middle")
    svg.add(
        f'<line class="ref" x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{top:.1f}"/>'
    )
    for i, d in enumerate(scored):
        lz = d.get("lorenz")
        if lz:
            svg.path([(xs(x), ys(y)) for x, y in zip(lz["x"], lz["y"])], f"dsl dsl-{i}")
    gx = right + 24
    svg.text(gx, top + 12, "Gini", cls="lbl")
    for j, d in enumerate(scored):
        y = top + 34 + j * 20
        svg.add(
            f'<line class="dsl dsl-{j}" x1="{gx:.1f}" x2="{gx + 16:.1f}" y1="{y - 4:.1f}" y2="{y - 4:.1f}"/>'
        )
        svg.text(gx + 22, y, f"{d['label']} {_fixed(d['metrics'].get('gini'), 3)}", cls="lbl")
    svg.text(
        (left + right) / 2,
        bottom + 32,
        f"share of {_mass_name(r)}, lowest predicted first",
        anchor="middle",
    )
    return svg.render()


def spread_chart(r: dict[str, Any], w: float) -> str:
    """The spread of the predictions: share of exposure by predicted rate, on shared bins."""
    scored = _scored(r)
    found = [(i, d["rate_histogram"]) for i, d in enumerate(scored) if d.get("rate_histogram")]
    if not found:
        return _none()
    edges = [float(e) for e in found[0][1]["edges"]]
    use_log = bool(found[0][1].get("log")) and edges[0] > 0
    share = {i: [float(m) / (sum(h["mass"]) or 1.0) for m in h["mass"]] for i, h in found}
    peak = max(max(v) for v in share.values()) or 1.0
    right, top = w - 8, 30.0
    bottom = top + 190.0
    h = bottom + 40
    ys = _Lin(0.0, peak, bottom, top, pad=0.05, zero=True, n=4)
    step = (ys.ticks[1] - ys.ticks[0]) * 100 if len(ys.ticks) > 1 else 1.0
    places = 0 if abs(step - round(step)) < 1e-9 else 1

    def y_label(v: float) -> str:
        return _pct(v, places)

    left = _label_w(ys, y_label) + 14
    xs: Callable[[float], float]
    if use_log:
        e0, e1 = math.log10(edges[0]), math.log10(edges[-1])

        def log_x(v: float) -> float:
            return left + (math.log10(max(v, 1e-300)) - e0) / (e1 - e0) * (right - left)

        xs = log_x
        cands = [
            m * 10.0**e for e in range(math.floor(e0) - 1, math.ceil(e1) + 1) for m in (1, 2, 5)
        ]
        ticks = [t for t in cands if e0 <= math.log10(t) <= e1]
    else:
        lin = _Lin(edges[0], edges[-1], left, right, pad=0.0, n=6)
        xs = lin
        ticks = [t for t in lin.ticks if edges[0] <= t <= edges[-1]]
    svg = _Svg(w, h, "spread of the predictions")
    _ds_legend(
        svg,
        left,
        12,
        len(scored),
        lambda i: (
            [("box", "hbar", str(scored[i]["label"]))]
            + ([("line", "dsl", _others(scored, i))] if len(found) > 1 else [])
        ),
    )
    _yaxis(svg, ys, left, right, y_label)
    svg.text(right, 12, f"share of {_mass_name(r)} per bin", anchor="end", cls="lbl")
    for t in ticks:
        svg.text(xs(t), bottom + 16, _num(t, 3), anchor="middle")
    svg.text(
        (left + right) / 2,
        bottom + 34,
        f"predicted {_rate_name(r)}" + (", log scale" if use_log else ""),
        anchor="middle",
        cls="lbl",
    )
    svg.hline(left, right, bottom, "ax")

    def outline(v: list[float]) -> list[tuple[float, float]]:
        return [
            (xs(edges[j] if e == 0 else edges[j + 1]), ys(s))
            for j, s in enumerate(v)
            for e in (0, 1)
        ]

    for i, _ in found:
        bars = [
            _bar(
                "hbar",
                xs(edges[j]) + 0.5,
                ys(v),
                max(xs(edges[j + 1]) - xs(edges[j]) - 1, 0.5),
                bottom,
                r=1.0,
            )
            for j, v in enumerate(share[i])
        ]
        svg.group(f"ds ds-{i}", bars)
    if len(found) > 1:
        for i, _ in found:
            svg.path(outline(share[i]), "dsl")
    for i, _ in found:
        g = _Svg(w, h, "")
        g.path(outline(share[i]), "model")
        svg.group(f"ds ds-{i}", g.parts)
    last = len(edges) - 2
    for j in range(len(edges) - 1):
        lo, hi = _num(edges[j], 3), _num(edges[j + 1], 3)
        span = f"≤ {hi}" if j == 0 else f"≥ {lo}" if j == last else f"{lo}–{hi}"
        lines = [f"predicted {_rate_name(r)} {span}"]
        lines += [
            f"{scored[i]['label']}: {_pct(share[i][j], 1)} of the {_mass_name(r)}" for i, _ in found
        ]
        svg.hit(xs(edges[j]), top, xs(edges[j + 1]) - xs(edges[j]), bottom - top, "\n".join(lines))
    return svg.render()


def histogram_chart(r: dict[str, Any], w: float, h: float) -> str:
    """The distribution of the deviance residuals, each dataset on its own bins."""
    scored = _scored(r)
    svg = _Svg(w, h, "deviance residuals")
    top, bottom = 24.0, h - 34
    any_bars = False
    for i, d in enumerate(scored):
        hist = d.get("residual_histogram")
        if not hist or not hist["edges"]:
            continue
        edges = [float(e) for e in hist["edges"]]
        total = sum(hist["mass"]) or 1.0
        share = [float(m) / total for m in hist["mass"]]
        ys = _Lin(0.0, max(share) or 1.0, bottom, top, pad=0.05, zero=True, n=3)
        left = _label_w(ys, _pct0) + 14
        right = w - 8
        xs = _Lin(edges[0], edges[-1], left, right, pad=0.0, n=8)
        g = _Svg(w, h, "")
        _yaxis(g, ys, left, right, _pct0)
        for t in xs.ticks:
            g.text(xs(t), bottom + 16, _num(t, 3), anchor="middle")
        for j, v in enumerate(share):
            g.add(
                _bar(
                    "hbar",
                    xs(edges[j]) + 0.5,
                    ys(v),
                    max(xs(edges[j + 1]) - xs(edges[j]) - 1, 0.5),
                    bottom,
                    r=1.0,
                )
            )
        if edges[0] < 0 < edges[-1]:
            g.vline(xs(0.0), top, bottom, "ref")
        g.hline(left, right, bottom, "ax")
        g.text(
            (left + right) / 2,
            bottom + 30,
            f"deviance residual ({d['label']})",
            anchor="middle",
            cls="lbl",
        )
        g.text(right, 12, f"share of {_mass_name(r)} per bin", anchor="end", cls="lbl")
        svg.group(f"ds ds-{i}", g.parts)
        any_bars = True
    return svg.render() if any_bars else _none()


def qq_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Standardised deviance residuals against the standard normal."""
    scored = _scored(r)
    with_qq = [(i, d["qq"]) for i, d in enumerate(scored) if d.get("qq")]
    if not with_qq:
        return _none()
    all_v = _fv(v for _, q in with_qq for key in ("theoretical", "sample") for v in q[key])
    lo, hi = max(min(all_v), -6.0), min(max(all_v), 6.0)
    top, bottom = 10.0, h - 34
    side = min(bottom - top, w - 80)
    left = 46.0
    right = left + side
    xs = _Lin(lo, hi, left, right, pad=0.0, n=5)
    ys = _Lin(lo, hi, top + side, top, pad=0.0, n=5)
    svg = _Svg(w, h, "normal QQ plot of the standardised deviance residuals")
    _yaxis(svg, ys, left, right, _fixed2)
    for t in xs.ticks:
        svg.text(xs(t), top + side + 16, _fixed(t, 1), anchor="middle")
    svg.add(
        f'<line class="ref" x1="{xs(lo):.1f}" y1="{ys(lo):.1f}" x2="{xs(hi):.1f}" y2="{ys(hi):.1f}"/>'
    )
    for i, q in with_qq:
        g = _Svg(w, h, "")
        pts = []
        for x, y in zip(q["theoretical"], q["sample"]):
            pts.append((xs(min(max(x, lo), hi)), ys(min(max(y, lo), hi))))
        for x, y in pts[::3]:
            g.add(f'<circle class="ddot" cx="{x:.1f}" cy="{y:.1f}" r="2.5"/>')
        svg.group(f"ds ds-{i}", g.parts)
    svg.text(
        (left + right) / 2, top + side + 32, "standard normal quantile", anchor="middle", cls="lbl"
    )
    return svg.render()


def balance_chart(r: dict[str, Any], w: float) -> str:
    """Every factor's A/E by band as a small chart on one shared scale; flagged bands marked."""
    scored = _scored(r)
    feats = [f for f in r["features"] if f["bands"]]
    if not scored or not feats:
        return _none()
    th = r["thresholds"]
    tol = th.get("factor_ae")
    fmin = th.get("factor_min_mass") or 0.0
    checked: list[dict[str, Any]] = []
    for f in feats:
        for bands in f["bands"]:
            total = sum(b["mass"] for b in bands) or 1.0
            checked += [b for b in bands if b["mass"] / total >= fmin and b.get("ae") is not None]
    vals = _fv(b["ae"] for b in checked)
    two_sd = min(2 * max(_fv(b.get("sd") for b in checked), default=0.0), 0.3)
    span = (tol or 0.1) * 1.3
    lo = max(min(vals + [1 - span, 1 - two_sd]), 0.4)
    hi = min(max(vals + [1 + span, 1 + two_sd]), 1.8)
    cols = max(1, min(len(feats), int((w + 16) // 186)))
    gap_x, gap_y, legend_h, name_h, plot_h = 16.0, 14.0, 28.0, 32.0, 56.0
    pw = (w - gap_x * (cols - 1)) / cols
    step = name_h + plot_h + gap_y
    rows = -(-len(feats) // cols)
    h = legend_h + rows * step - gap_y + 4
    svg = _Svg(w, h, "A/E by band for every factor")

    def items(i: int) -> list[tuple[str, str, str]]:
        out = [("line", "data", str(scored[i]["label"])), ("box", "bandbox", "±2σ")]
        if tol:
            out.append(("line", "guide", f"±{_pct(tol, 0)}"))
        out += [("dot", "flagdot", "flagged"), ("dot", "neardot", "near miss")]
        if len(scored) > 1:
            out.append(("line", "dsl", _others(scored, i)))
        return out

    _ds_legend(svg, 0, 12, len(scored), items, fit=w)
    chars = int(pw / (_FONT * 0.6))
    for n, f in enumerate(feats):
        px = (n % cols) * (pw + gap_x)
        py = legend_h + (n // cols) * step
        top, bottom = py + name_h, py + name_h + plot_h
        ys = _Lin(lo, hi, bottom, top, pad=0.04)

        def y_of(v: float, s: _Lin = ys) -> float:
            return s(min(max(v, s.lo), s.hi))

        svg.text(px, py + 12, _short(str(f["feature"]), chars), cls="strong")
        total0 = sum(b["mass"] for b in f["bands"][0]) or 1.0
        raw = [max(b["mass"] / total0, 0.004) for b in f["bands"][0]]
        x0, wid = [], []
        edge = px
        for v in raw:
            x0.append(edge)
            wid.append(pw * v / (sum(raw) or 1.0))
            edge += wid[-1]
        svg.hline(px, px + pw, ys(1.0), "ref")
        if tol:
            for g_ in (1 - tol, 1 + tol):
                if ys.lo <= g_ <= ys.hi:
                    svg.hline(px, px + pw, ys(g_), "guide")
        for i in range(len(scored)):
            bs = f["bands"][i]
            if any((b.get("sd") or 0) > 0 for b in bs):
                up = [
                    (x0[j] + e * wid[j], y_of(1 + 2 * (b.get("sd") or 0.0)))
                    for j, b in enumerate(bs)
                    for e in (0, 1)
                ]
                dn = [
                    (x0[j] + e * wid[j], y_of(1 - 2 * (b.get("sd") or 0.0)))
                    for j, b in enumerate(bs)
                    for e in (0, 1)
                ]
                svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
        for i in range(len(scored)):
            svg.path(
                [
                    (x0[j] + e * wid[j], y_of(b["ae"]))
                    for j, b in enumerate(f["bands"][i])
                    if b.get("ae") is not None
                    for e in (0, 1)
                ],
                f"dsl dsl-{i}",
            )
        labels = f["labels"]
        for i in range(len(scored)):
            bs = f["bands"][i]
            live = [j for j, b in enumerate(bs) if b.get("ae") is not None]
            if not live:
                continue
            g = _Svg(w, h, "")
            worst = max(live, key=lambda j: abs(bs[j].get("z") or 0.0))
            for j in live:
                mark = {"warning": "flagdot", "near_miss": "neardot"}.get(str(bs[j].get("flag")))
                if mark or j == worst:
                    g.add(
                        f'<circle class="{mark or "ddot"}" cx="{x0[j] + wid[j] / 2:.1f}" '
                        f'cy="{y_of(bs[j]["ae"]):.1f}" r="{4 if mark else 3}"/>'
                    )
            tail = f": {_fixed(bs[worst]['ae'], 2)}"
            room = max(chars - len(tail) - 6, 4)
            g.text(px, py + 27, "worst " + _short(str(labels[worst]), room) + tail, cls="lbl")
            svg.group(f"ds ds-{i}", g.parts)
        for j, lab in enumerate(labels):
            lines = [f"{f['feature']}: {lab}"]
            for i, d in enumerate(scored):
                b = f["bands"][i][j]
                if b.get("ae") is not None:
                    verdict = {"warning": " (flagged)", "near_miss": " (near miss)"}.get(
                        str(b.get("flag")), ""
                    )
                    lines.append(
                        f"{d['label']}: A/E {_fixed(b['ae'])}, z {_signed(b.get('z'))}" + verdict
                    )
            svg.hit(x0[j], top, max(wid[j], 1.0), plot_h, "\n".join(lines))
    return svg.render()


def factor_chart(r: dict[str, Any], f: dict[str, Any], w: float) -> str:
    """One factor on one x-axis: fitted effect with its 95% band, actual and expected rate, A/E
    of every dataset with ±2σ, and exposure. Continuous factors keep their bands at equal
    width, with the effect curve placed inside each band by its value."""
    scored = _scored(r)
    th = r["thresholds"]
    labels = f["labels"]
    n = len(labels)
    continuous = f["type"] == "continuous"
    eff = f.get("effect")
    mult = _is_mult(r)
    neutral = 1.0 if mult else 0.0
    at_level: dict[int, int] = {}
    order = list(range(n))
    if eff and not continuous:
        idx = {str(x): i for i, x in enumerate(eff["x"])}
        at_level = {k: idx[lab] for k, lab in enumerate(labels) if lab in idx}
        order.sort(key=lambda k: eff["value"][at_level[k]] if k in at_level else neutral)
    pos_of = {k: j for j, k in enumerate(order)}
    pd = f.get("partial_dependence") if f.get("in_interactions") or not eff else None
    pd_pts = _pd_points(f, pd, labels, continuous, eff) if pd else []
    rel_vals: list[float] = []
    ci_bounded = False
    if eff:
        rel_vals += _fv(eff["value"])
    rel_vals += _fv(v for _, v in pd_pts)
    has_rel = bool(rel_vals)
    t0, gap, foot = 22.0, 30.0, 26.0
    top_h, lvl_h, mid_h, mass_h = (180.0 if has_rel else 0.0), 120.0, 120.0, 48.0
    r0 = t0
    l0 = (r0 + top_h + gap) if has_rel else t0
    a0 = l0 + lvl_h + gap
    m0 = a0 + mid_h + gap
    h = m0 + mass_h + foot
    bands = f["bands"]
    lvl_vals = _fv(
        v
        for bs in bands
        for b in bs
        if b.get("rows")
        for v in (b.get("actual"), b.get("predicted"))
    )
    lvl_s = _Lin(0.0, max(lvl_vals or [1.0]), l0 + lvl_h, l0, zero=True, n=4)
    tol = th.get("factor_ae")
    aes = _fv(b.get("ae") for bs in bands for b in bs if b.get("rows"))
    two_sd = min(2 * max(_fv(b.get("sd") for bs in bands for b in bs), default=0.0), 0.3)
    ae_lo = max(min(aes + [1 - (tol or 0.1) * 1.3, 1 - two_sd]), 0.4)
    ae_hi = min(max(aes + [1 + (tol or 0.1) * 1.3, 1 + two_sd]), 1.8)
    ae_s = _Lin(ae_lo, ae_hi, a0 + mid_h, a0, pad=0.04, n=4)

    def rate_fmt(v: float) -> str:
        return _num(v, 3)

    def rel_fmt(v: float) -> str:
        return _mult_tick(v) if mult else _num(v, 2)

    rel_s: _Log | _Lin | None = None
    if has_rel:
        ci_lo, ci_hi = rel_vals[:], rel_vals[:]
        if eff and eff.get("lo"):
            low, high = min(_fv(eff["value"])), max(_fv(eff["value"]))
            span = (high / max(low, 1e-9)) ** 0.5 if mult else (high - low) * 0.5
            lim_lo, lim_hi = (
                (low / max(span, 1.25), high * max(span, 1.25))
                if mult
                else (low - span, high + span)
            )
            ci_lo = rel_vals + [max(v, lim_lo) for v in _fv(eff["lo"])]
            ci_hi = rel_vals + [min(v, lim_hi) for v in _fv(eff["hi"])]
            ci_bounded = True
        rel_s = (
            _Log(min(ci_lo), max(ci_hi), r0 + top_h, r0, max_ticks=5)
            if mult
            else _Lin(min(ci_lo), max(ci_hi), r0 + top_h, r0, n=4)
        )
    lw = max(
        _label_w(rel_s, rel_fmt) if rel_s is not None else 0.0,
        _label_w(ae_s, _fixed2),
        _label_w(lvl_s, rate_fmt),
    )
    left, right = 6 + lw + 6, w - 10
    slot = (right - left) / max(n, 1)
    centre = {k: left + (pos_of[k] + 0.5) * slot for k in range(n)}
    edges = f.get("edges") or []

    def curve_x(x: float) -> float:
        """Where a value of a continuous factor falls: its band, then linearly inside it."""
        j = max(min(bisect.bisect_right(edges, x) - 1, n - 1), 0)
        width = edges[j + 1] - edges[j]
        frac = (x - edges[j]) / width if width > 0 else 0.5
        return left + (j + min(max(frac, 0.0), 1.0)) * slot

    svg = _Svg(
        w, h, f"{f['feature']}: fitted effect, actual and expected, A/E by band and exposure"
    )
    if rel_s is not None:
        scale = rel_s
        _yaxis(svg, scale, left, right, rel_fmt)
        svg.hline(left, right, scale(neutral), "ref")
        title = f"fitted {_effect_name(r)}" + (" (log scale)" if mult else "")
        svg.text(left, r0 - 8, title, cls="lbl")
        keys: list[tuple[str, str, str]] = []
        if eff:
            keys.append(("line", "model", "main effect"))
            if eff.get("lo"):
                keys.append(("box", "shapband", "95% confidence"))
        if pd_pts:
            keys.append(("line", "shapmean", "partial dependence"))
        end = _legend(svg, left + _text_w(title) + 18, r0 - 8, keys)
        base = eff.get("base") if eff else None
        if base is not None:
            note = f"1.00 at {_label_of(base)}" if mult else f"0 at {_label_of(base)}"
            if end + _text_w(note) < right:
                svg.text(right, r0 - 8, note, anchor="end")
        if eff:
            if continuous:
                xs_v = [curve_x(float(v)) for v in eff["x"]]
                ys_v = list(eff["value"])
                lo_v = list(eff.get("lo") or [])
                hi_v = list(eff.get("hi") or [])
                order_v = list(range(len(xs_v)))
            else:
                order_v = [k for k in order if k in at_level]
                xs_v = [centre[k] for k in order_v]
                ys_v = [eff["value"][at_level[k]] for k in order_v]
                lo_v = [eff["lo"][at_level[k]] for k in order_v] if eff.get("lo") else []
                hi_v = [eff["hi"][at_level[k]] for k in order_v] if eff.get("hi") else []

            def bounded(v: float, s: _Log | _Lin = scale) -> float:
                if isinstance(s, _Log):
                    return s(min(max(v, math.exp(s.llo)), math.exp(s.lhi)))
                return s(min(max(v, s.lo), s.hi))

            if lo_v and hi_v and ci_bounded:
                if continuous:
                    upper = [(x, bounded(hv)) for x, hv in zip(xs_v, hi_v)]
                    lower = [(x, bounded(lv)) for x, lv in zip(xs_v, lo_v)]
                    svg.add(f'<path class="shapband" d="{_d(upper + lower[::-1])}Z"/>')
                else:
                    for x, lv, hv in zip(xs_v, lo_v, hi_v):
                        svg.add(
                            f'<line class="whisker" x1="{x:.1f}" x2="{x:.1f}" y1="{bounded(lv):.1f}" '
                            f'y2="{bounded(hv):.1f}"/>'
                        )
            if continuous:
                svg.path([(x, bounded(v)) for x, v in zip(xs_v, ys_v)], "model")
            else:
                svg.path([(x, bounded(v)) for x, v in zip(xs_v, ys_v)], "model")
                for x, v in zip(xs_v, ys_v):
                    svg.add(f'<circle class="mdot" cx="{x:.1f}" cy="{bounded(v):.1f}" r="3.5"/>')
        if pd_pts:
            if continuous:
                pts = sorted((curve_x(float(x)), scale(v)) for x, v in pd_pts)
            else:
                pts = sorted(
                    (centre[labels.index(str(x))], scale(v)) for x, v in pd_pts if str(x) in labels
                )
            svg.path(pts, "shapmean")
    _yaxis(svg, lvl_s, left, right, rate_fmt)
    svg.hline(left, right, l0 + lvl_h, "ax")
    for i in range(len(scored)):
        g = _Svg(w, h, "")
        title = f"actual and expected {_rate_name(r)} ({scored[i]['label']})"
        g.text(left, l0 - 8, title, cls="lbl")
        _legend(
            g,
            left + _text_w(title) + 18,
            l0 - 8,
            [("line", "data", "actual"), ("line", "model", "expected")],
        )
        at = [k for k in order if bands[i][k].get("rows")]
        act = [
            (centre[k], lvl_s(bands[i][k]["actual"]))
            for k in at
            if bands[i][k].get("actual") is not None
        ]
        pre = [
            (centre[k], lvl_s(bands[i][k]["predicted"]))
            for k in at
            if bands[i][k].get("predicted") is not None
        ]
        if len(at) <= 1:
            for x, y in pre:
                g.add(f'<circle class="mdot" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
            for x, y in act:
                g.add(f'<circle class="ddot" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        else:
            g.path(pre, "model")
            g.path(act, "data")
            if n <= 24:
                for x, y in act:
                    g.add(f'<circle class="ddot" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')
        svg.group(f"ds ds-{i}", g.parts)
    _yaxis(svg, ae_s, left, right, _fixed2)
    if tol:
        for g_ in (1 - tol, 1 + tol):
            if ae_s.lo <= g_ <= ae_s.hi:
                svg.hline(left, right, ae_s(g_), "guide")
    svg.hline(left, right, ae_s(1.0), "ref")
    for i in range(len(scored)):
        g = _Svg(w, h, "")
        others = f", {_others(scored, i)} in grey" if len(scored) > 1 else ""
        built_in = ""
        if f["in_model"] and not continuous and scored[i]["label"] == scored[0]["label"]:
            built_in = " (1.00 by construction for a fitted categorical)"
        g.text(
            left,
            a0 - 8,
            f"A/E by band: {scored[i]['label']} in colour" + others + built_in,
            cls="lbl",
        )
        svg.group(f"ds ds-{i}", g.parts)
    if tol:
        svg.text(right, a0 - 8, f"dashed: ±{_pct(tol, 0)}; shaded: ±2σ", anchor="end")

    def ae_y(v: float) -> float:
        return ae_s(min(max(v, ae_s.lo), ae_s.hi))

    def steps(i: int, sign: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for k in order:
            b = bands[i][k]
            if b.get("ae") is None:
                continue
            v = b["ae"] if sign == 0 else 1 + sign * 2 * (b.get("sd") or 0.0)
            pts += [(centre[k] - slot / 2, ae_y(v)), (centre[k] + slot / 2, ae_y(v))]
        return pts

    for i in range(len(scored)):
        up, dn = steps(i, 1.0), steps(i, -1.0)
        if up:
            svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
    for i in range(len(scored)):
        svg.path(steps(i, 0.0), f"dsl dsl-{i}")
        marks = _Svg(w, h, "")
        for k in order:
            b = bands[i][k]
            mark = {"warning": "flagdot", "near_miss": "neardot"}.get(str(b.get("flag")))
            if mark and b.get("ae") is not None:
                marks.add(
                    f'<circle class="{mark}" cx="{centre[k]:.1f}" cy="{ae_y(b["ae"]):.1f}" r="4"/>'
                )
        svg.group(f"ds ds-{i}", marks.parts)
    mass = [float(b["mass"]) for b in bands[0]]
    total = sum(mass) or 1.0
    shown = sorted(mass, reverse=True)
    cap = shown[0] or 1.0
    clipped = len(shown) > 1 and shown[0] > 3 * shown[1] > 0
    if clipped:
        cap = shown[1] * 1.5
    svg.text(left, m0 - 8, f"{_mass_name(r)}, {scored[0]['label']}", cls="lbl")
    bar_w = max(min(slot - 2, 26.0), 1.0)
    for k in order:
        v = min(mass[k], cap)
        svg.add(
            _bar("mb", centre[k] - bar_w / 2, m0 + mass_h - v / cap * mass_h, bar_w, m0 + mass_h)
        )
        if clipped and mass[k] > cap:
            note = f"{_pct(mass[k] / total, 0)} of {_mass_name(r)}"
            far = centre[k] + bar_w / 2 + 6 + _text_w(note) > right
            svg.text(
                centre[k] - bar_w / 2 - 6 if far else centre[k] + bar_w / 2 + 6,
                m0 + 12,
                note,
                anchor="end" if far else "start",
                cls="lbl",
            )
    svg.hline(left, right, m0 + mass_h, "ax")
    _xlabels(svg, [centre[k] for k in order], [labels[k] for k in order], m0 + mass_h + 16, slot)
    for k in order:
        lines = [f"{f['feature']}: {labels[k]}"]
        if k in at_level and eff:
            v = eff["value"][at_level[k]]
            text = _mult(v) if mult else _num(v, 4)
            if eff.get("lo"):
                a, b_ = eff["lo"][at_level[k]], eff["hi"][at_level[k]]
                text += f" (95% {_mult(a) if mult else _num(a, 3)} to {_mult(b_) if mult else _num(b_, 3)})"
            lines.append(f"{_effect_name(r)} {text}")
        lines.append(
            f"{_mass_name(r)} ({scored[0]['label']}) {_num(mass[k])}, {_pct(mass[k] / total, 2)}"
        )
        for i, d in enumerate(scored):
            b = bands[i][k]
            if b.get("rows"):
                lines.append(
                    f"{d['label']}: actual {_num(b.get('actual'), 4)}, expected "
                    f"{_num(b.get('predicted'), 4)}, A/E {_fixed(b.get('ae'))}, z {_signed(b.get('z'))}"
                )
        svg.hit(centre[k] - slot / 2, t0, slot, m0 + mass_h - t0, "\n".join(lines))
    return svg.render()


def _pd_points(
    f: dict[str, Any],
    pd: dict[str, Any],
    labels: Sequence[str],
    continuous: bool,
    eff: dict[str, Any] | None,
) -> list[tuple[Any, float]]:
    """The partial-dependence curve, rebased to the effect's reference when there is one."""
    grid = pd.get("grid_values") or []
    rel = pd.get("relativities")
    if not rel:
        return []
    pts = [(g, float(v)) for g, v in zip(grid, rel) if v is not None and float(v) > 0]
    if not pts or not eff or eff.get("base") is None:
        return pts
    base = eff["base"]
    if continuous:
        ref = min(pts, key=lambda p: abs(float(p[0]) - float(base)))[1]
    else:
        match = next((v for g, v in pts if str(g) == str(base)), None)
        ref = match if match is not None else 1.0
    return [(g, v / ref) for g, v in pts] if ref else pts


def forest_chart(r: dict[str, Any], w: float, *, limit: int = 40) -> str:
    """Point estimates with 95% intervals, largest |z| first, the rest in the table."""
    coefs = [c for c in r["coefficients"] if not _is_basis_name(str(c["feature"]))]
    if not coefs:
        return _none("No coefficients to show.")
    ranked = sorted(coefs, key=lambda c: -abs(c.get("z_value") or 0.0))[:limit]
    mult = _is_mult(r)
    rows = sorted(ranked, key=lambda c: coefs.index(c))
    row_h = 20.0
    head = 24.0
    foot = 26.0
    h = head + row_h * len(rows) + foot
    name_w = min(max(_text_w(str(c["feature"])) for c in rows) + 12, w * 0.32)
    x0, x1 = name_w + 6, w - 70
    values: list[float] = []
    for c in rows:
        ci = c.get("relativity_ci") if mult else c.get("conf_int")
        est = c.get("relativity") if mult else c.get("estimate")
        if est is not None:
            values.append(float(est))
        if ci:
            values += _fv(ci)
    if not values:
        return _none()
    neutral = 1.0 if mult else 0.0
    if mult:
        lo_v = max(min(values), 1e-6)
        xs: Any = _Log(min(lo_v, 1.0), max(max(values), 1.0), x0, x1, max_ticks=7)
        tick_fmt: Callable[[float], str] = _mult_tick
    else:
        xs = _Lin(min(values + [0.0]), max(values + [0.0]), x0, x1, n=6)

        def tick_fmt(v: float) -> str:
            return _num(v, 3)

    svg = _Svg(w, h, "coefficient forest plot")
    for t in xs.ticks:
        x = xs(t)
        svg.vline(x, head, h - foot, "gl")
        svg.text(x, h - foot + 16, tick_fmt(t), anchor="middle")
    svg.vline(xs(neutral), head, h - foot, "ref")
    svg.text(
        x0,
        12,
        _effect_name(r) + (" (log scale)" if mult else "") + ", with 95% interval",
        cls="lbl",
    )
    for k, c in enumerate(rows):
        y = head + (k + 0.5) * row_h
        name = str(c["feature"])
        svg.text(name_w - 2, y + 4, _short(name, int(name_w / 6.6)), anchor="end", cls="lbl")
        ci = c.get("relativity_ci") if mult else c.get("conf_int")
        est = c.get("relativity") if mult else c.get("estimate")
        sig = bool(c.get("significant"))
        if ci:
            a, b = float(ci[0]), float(ci[1])
            svg.add(
                f'<line class="whisker" x1="{xs(a):.1f}" x2="{xs(b):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
        if est is not None:
            cls = "mdot" if sig else "nsdot"
            svg.add(f'<circle class="{cls}" cx="{xs(float(est)):.1f}" cy="{y:.1f}" r="4"/>')
            svg.text(x1 + 8, y + 4, (_mult(float(est)) if mult else _num(float(est), 3)), cls="lbl")
        hover = f"{name}: estimate {_num(c.get('estimate'), 4)} (se {_num(c.get('std_error'), 3)}), z {_fixed(c.get('z_value'), 2)}, p {_p(c.get('p_value'))}"
        svg.hit(0, y - row_h / 2, w, row_h, hover)
    return svg.render()


def _is_basis_name(name: str) -> bool:
    return bool(_BASIS_NAME.match(name)) or name == "Intercept"


def heatmap_chart(
    r: dict[str, Any],
    inter: dict[str, Any],
    w: float,
    *,
    kind: str = "ae",
) -> str:
    """A grid over the pair's two axes: A/E per cell, exposure per cell or the fitted surface."""
    scored = _scored(r)
    rows, cols = inter["rows"], inter["cols"]
    n_r, n_c = inter["n_rows"], inter["n_cols"]
    th = r["thresholds"]
    if n_r == 0 or n_c == 0:
        return _none()
    label_w = min(max(_text_w(str(x)) for x in rows) + 8, 96.0)
    cell_w = max((w - label_w - 8) / n_c, 12.0)
    cell_h = 24.0
    head = 20.0
    h = head + n_r * cell_h + 22
    svg = _Svg(w, h, f"{inter['name']}: {kind}")
    for col in range(n_c):
        x = label_w + (col + 0.5) * cell_w
        svg.text(x, h - 6, _short(str(cols[col]), max(int(cell_w / 7), 3)), anchor="middle")
    for rr in range(n_r):
        y = head + (rr + 0.5) * cell_h
        svg.text(label_w - 6, y + 4, _short(str(rows[rr]), int(label_w / 7)), anchor="end")
    svg.text(0, 12, f"{inter['row_axis']} (rows) × {inter['col_axis']} (columns)", cls="lbl")

    def cell(g: _Svg, rr: int, cc: int, tone: str, text: str, hover: str, faded: bool) -> None:
        x = label_w + cc * cell_w
        y = head + rr * cell_h
        fade = " fade" if faded else ""
        g.add(
            f'<rect class="{tone}{fade}" x="{x + 1:.1f}" y="{y + 1:.1f}" '
            f'width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" rx="2"><title>{_e(hover)}</title></rect>'
        )
        if text and cell_w > 34:
            g.text(x + cell_w / 2, y + cell_h / 2 + 4, text, anchor="middle", cls="lbl")

    if kind == "effect":
        eff = inter.get("effect")
        if not eff:
            return _none("Fitted surface not available for this pair.")
        vals = [v for row in eff["values"] for v in row]
        mult = bool(eff["multiplicative"])
        scale = _heat_scale(vals, mult)
        neutral = 1.0 if mult else 0.0
        for rr in range(n_r):
            for cc in range(n_c):
                v = eff["values"][rr][cc]
                text = _fixed(v, 2)
                hover = f"{rows[rr]} × {cols[cc]}: {eff['kind']} {_num(v, 4)}"
                tone = _heat(v, neutral, (math.exp(scale) - 1) if mult else scale)
                cell(svg, rr, cc, tone, text, hover, False)
        return svg.render()
    fmin = th.get("factor_min_mass") or 0.0
    for i, d in enumerate(scored):
        g = _Svg(w, h, "")
        cells = inter["cells"][i]
        total = sum(c["mass"] for c in cells) or 1.0
        for rr in range(n_r):
            for cc in range(n_c):
                c = cells[rr * n_c + cc]
                share = c["mass"] / total
                hover = (
                    f"{rows[rr]} × {cols[cc]} ({d['label']}): A/E {_fixed(c.get('ae'))}, z "
                    f"{_signed(c.get('z'))}, {_pct(share, 2)} of {_mass_name(r)}"
                )
                if kind == "mass":
                    tone = f"tb{min(max(1, math.ceil(5 * share * n_r * n_c / 2.0)), 5)}"
                    cell(g, rr, cc, tone, _pct(share, 1), hover, False)
                else:
                    ae = c.get("ae")
                    cell(g, rr, cc, _heat(ae, 1.0, 0.3), _fixed(ae, 2), hover, share < fmin)
        svg.group(f"ds ds-{i}", g.parts)
    return svg.render()


def path_chart(r: dict[str, Any], w: float, h: float) -> str | None:
    """Cross-validated deviance against the penalty along a regularisation path."""
    path = r["fit"].get("regularization_path")
    if not isinstance(path, list):
        return None
    pts = sorted(
        (float(p["alpha"]), float(p["cv_deviance_mean"]))
        for p in path
        if p.get("cv_deviance_mean") is not None and float(p.get("alpha") or 0.0) > 0
    )
    if len(pts) < 2:
        return None
    la = [math.log10(a) for a, _ in pts]
    devs = [d for _, d in pts]
    top, bottom = 22.0, h - 34
    xs = _Lin(min(la), max(la), 60.0, w - 12, pad=0.02, n=6)
    ys = _Lin(min(devs), max(devs), bottom, top, pad=0.08, n=4)
    svg = _Svg(w, h, "regularisation path")
    _yaxis(svg, ys, 60.0, w - 12, lambda v: _num(v, 4))
    svg.text(60.0, 12, "cross-validated deviance", cls="lbl")
    svg.path([(xs(a), ys(d)) for a, d in zip(la, devs)], "model")
    svg.hline(60.0, w - 12, bottom, "ax")
    for t in xs.ticks:
        svg.text(xs(t), bottom + 16, f"1e{t:g}", anchor="middle")
    svg.text((60 + w - 12) / 2, bottom + 30, "penalty α (log scale)", anchor="middle", cls="lbl")
    chosen = (r["fit"]["summary"].get("regularization") or {}).get("alpha")
    if chosen and chosen > 0:
        svg.vline(xs(math.log10(float(chosen))), top, bottom, "guide")
    return svg.render()


def class_mix_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Actual and predicted share of each class, on the selected dataset."""
    scored = _scored(r)
    classes = [str(c) for c in r["model"]["classes"]]
    n = len(classes)
    vals = _fv(
        d["class_mix"][key][c] for d in scored for key in ("actual", "predicted") for c in classes
    )
    top, bottom = 26.0, h - 24
    ys = _Lin(0.0, max(vals or [1.0]), bottom, top, pad=0.05, zero=True, n=4)
    left = _label_w(ys, _pct0) + 14
    right = w - 8
    slot = (right - left) / n
    bar = min(slot * 0.32, 40.0)
    svg = _Svg(w, h, "actual and predicted class mix")
    _legend(svg, left, 12, [("box", "dbar", "actual"), ("box", "own", "predicted")])
    _yaxis(svg, ys, left, right, _pct0)
    svg.hline(left, right, bottom, "ax")
    xs = [left + (j + 0.5) * slot for j in range(n)]
    for i, d in enumerate(scored):
        mix = d["class_mix"]
        parts = []
        for j, c in enumerate(classes):
            parts.append(_bar("dbar", xs[j] - bar - 1, ys(mix["actual"][c]), bar, bottom))
            parts.append(_bar("own", xs[j] + 1, ys(mix["predicted"][c]), bar, bottom))
        svg.group(f"ds ds-{i}", parts)
    _xlabels(svg, xs, classes, bottom + 16, slot)
    for j, c in enumerate(classes):
        lines = [f"class {c}"]
        for d in scored:
            mix = d["class_mix"]
            lines.append(
                f"{d['label']}: actual {_pct(mix['actual'][c], 2)}, predicted "
                f"{_pct(mix['predicted'][c], 2)}"
            )
        svg.hit(xs[j] - slot / 2, top, slot, bottom - top, "\n".join(lines))
    return svg.render()


def _double_lift_chart(r: dict[str, Any], i: int, w: float, h: float) -> str:
    """Model and base A/E by deciles of the model-to-base ratio."""
    data = r.get("base_predictions")
    if not data:
        return _none()
    label = r["datasets"][i]["label"]
    side = data.get("train") if i == 0 else data.get("test")
    if not side:
        return _none(f"No base comparison on {label}.")
    deciles = side.get("model_vs_base_deciles") or []
    if not deciles:
        return _none()
    aes = _fv(d.get(key) for d in deciles for key in ("model_ae_ratio", "base_ae_ratio"))
    top, bottom = 22.0, h - 22
    ys = _Lin(min(aes + [0.9]), max(aes + [1.1]), bottom, top, pad=0.06, n=4)
    left = _label_w(ys, _fixed2) + 14
    right = w - 10
    slot = (right - left) / len(deciles)
    xs = [left + (j + 0.5) * slot for j in range(len(deciles))]
    svg = _Svg(w, h, "model against base A/E by decile of the ratio")
    _yaxis(svg, ys, left, right, _fixed2)
    svg.hline(left, right, ys(1.0), "ref")
    _legend(svg, left, 12, [("line", "model", "model A/E"), ("line", "data", "base A/E")])
    svg.path([(xs[j], ys(d["model_ae_ratio"])) for j, d in enumerate(deciles)], "model")
    svg.path([(xs[j], ys(d["base_ae_ratio"])) for j, d in enumerate(deciles)], "data")
    for j in range(len(deciles)):
        svg.text(xs[j], bottom + 16, str(j + 1), anchor="middle")
    return svg.render()


# --------------------------------------------------------------------------- takeaways --------


def _lift_takeaway(r: dict[str, Any], i: int) -> str:
    d = r["datasets"][i]
    bands = [b for b in d["deciles"] if b.get("predicted")]
    if len(bands) < 2:
        return "Calibration by decile"
    aes = _fv(b.get("ae") for b in bands)
    ratio = bands[-1]["predicted"] / bands[0]["predicted"]
    return (
        f"Top decile {ratio:.1f}× the bottom; {d['label']} decile A/E {min(aes):.2f}–{max(aes):.2f}"
    )


def _importance_takeaway(r: dict[str, Any]) -> str:
    feats = [f for f in r["features"] if f["in_model"] and f.get("relative_importance")]
    if not feats:
        return "Factor importance"
    words = (
        "One factor carries",
        "Two factors carry",
        "Three factors carry",
        "Four factors carry",
        "Five factors carry",
    )
    shares = [float(f["relative_importance"]) for f in feats]
    total = 0.0
    for k, share in enumerate(shares):
        total += share
        if total >= 80.0 or k == len(shares) - 1:
            word = words[k] if k < len(words) else f"{k + 1} factors carry"
            return f"{word} {total:.0f}% of the explained deviance"
    return "Factor importance"


def _variance_takeaway(r: dict[str, Any], i: int) -> str:
    bands = r["datasets"][i]["fit_bands"]
    phi = r["fit"].get("dispersion")
    vals = _fv(b.get("dispersion") for b in bands)
    tail = f"; dispersion {_fixed(phi, 2)}" if phi is not None else ""
    if len(vals) < 4:
        return f"Variance check{tail}"
    k = max(len(vals) // 4, 1)
    low, high = sum(vals[:k]) / k, sum(vals[-k:]) / k
    if low > 0 and high / low > 1.4:
        return f"Variance grows faster than the family assumes{tail}"
    if low > 0 and high / low < 0.7:
        return f"Variance grows slower than the family assumes{tail}"
    return f"Variance tracks the family's assumption{tail}"


def _balance_takeaway(r: dict[str, Any], i: int) -> str:
    flagged = sum(
        1 for f in r["features"] if any(b.get("flag") == "warning" for b in f["bands"][i])
    )
    label = r["datasets"][i]["label"]
    if flagged:
        return f"{flagged} factor{'s' if flagged != 1 else ''} with a band out of line on {label}"
    return f"Every factor balances on {label}"


def _balance_rule(r: dict[str, Any]) -> str:
    th = r["thresholds"]
    parts = []
    if th.get("factor_ae") is not None:
        parts.append(f"A/E by band; flagged past ±{_pct(th['factor_ae'], 0)}")
        if th.get("factor_z") is not None:
            parts.append(f"and {_num(th['factor_z'])}σ")
    return " ".join(parts) or "A/E by band"


def _coef_takeaway(r: dict[str, Any]) -> str:
    coefs = [c for c in r["coefficients"] if c["feature"] != "Intercept"]
    if not coefs:
        return "Coefficients"
    if not r["fit"]["inference_valid"]:
        return f"{len(coefs)} coefficients"
    sig = sum(1 for c in coefs if c.get("significant"))
    return f"{sig} of {len(coefs)} coefficients significant at 5%"


# --------------------------------------------------------------------------- page parts -------


def _open(tab: str) -> str:
    return f'<section class="tab tb tb-{_TAB_NAMES.index(tab)}" id="{tab.lower()}">'


def _header(r: dict[str, Any], base: dict[str, Any], name: str | None) -> str:
    m = r["model"]
    title = name or r.get("name") or f"{m['family']} GLM"
    used = sum(1 for f in base["features"] if f["in_model"])
    what = (
        f"multinomial logit, {len(m['classes'])} classes against {m['reference']}"
        if m.get("multiclass")
        else f"{m['family']}, {m['link']} link"
    )
    sub = [
        "generated " + str(r.get("generated_at") or "").replace("T", " ").replace("+00:00", " UTC"),
        what,
        f"{_num(m.get('n_params'))} coefficients on {_num(m.get('n_obs'))} rows",
        f"{used} factor{'s' if used != 1 else ''}",
        f"rustystats {m['rustystats_version']}",
    ]
    return (
        f'<header class="top"><h1>{_e(title)}</h1>'
        f'<p class="sub">{" · ".join(_e(x) for x in sub if x)}</p></header>'
    )


def _headline(r: dict[str, Any]) -> str:
    scored = _scored(r)
    mass = _column([d["metrics"].get("mass") for d in scored])
    if r["model"].get("multiclass"):
        cols = [
            "dataset",
            _mass_name(r),
            "deviance explained",
            "log loss",
            "accuracy",
            "macro F1",
        ]
        rows = [
            [
                _e(d["label"]),
                mass[i],
                _pct(d["metrics"].get("deviance_explained"), 2),
                _fixed(d["metrics"].get("log_loss")),
                _pct(d["metrics"].get("accuracy"), 1),
                _fixed(d["metrics"].get("macro_f1")),
            ]
            for i, d in enumerate(scored)
        ]
    else:
        cols = ["dataset", _mass_name(r), _actual_name(r), "A/E", "deviance explained", "Gini"]
        actual = _column([d["metrics"].get("actual") for d in scored])
        rows = [
            [
                _e(d["label"]),
                mass[i],
                actual[i],
                _fixed(d["metrics"].get("ae")),
                _pct(d["metrics"].get("deviance_explained"), 2),
                _fixed(d["metrics"].get("gini")),
            ]
            for i, d in enumerate(scored)
        ]
    return _table(
        cols, rows, numeric=range(1, len(cols)), row_class=[f"dsr-{i}" for i in range(len(scored))]
    )


def _scope(f: dict[str, Any]) -> str:
    scope = " · ".join(
        str(x) for x in (f.get("dataset"), f.get("feature"), f.get("class")) if x is not None
    )
    return f"{scope}: {f['message']}" if scope else str(f["message"])


def _issue_groups(r: dict[str, Any]) -> list[tuple[str, str, str]]:
    order = {"critical": 0, "warning": 1, "info": 2}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for f in r["review"]:
        if f["status"] != "pass":
            groups.setdefault((f["status"], f["check"]), []).append(f)
    out = []
    for (status, _), flagged in sorted(
        groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])
    ):
        full = "; ".join(_scope(f) for f in flagged)
        out.append((status, full, full))
    return out


def _counts(r: dict[str, Any]) -> str:
    n = {
        s: sum(f["status"] == s for f in r["review"])
        for s in ("critical", "warning", "info", "pass")
    }
    words = (
        ("critical", "critical", "critical"),
        ("warning", "warning", "warnings"),
        ("info", "note", "notes"),
        ("pass", "passed", "passed"),
    )
    return (
        '<span class="counts">'
        + "".join(
            _badge(s, f"{n[s]} {one if n[s] == 1 else many}") for s, one, many in words if n[s]
        )
        + "</span>"
    )


def _overview(
    r: dict[str, Any], views: list[dict[str, Any]], tabs: _Radio, cl: _Radio | None
) -> str:
    base = views[0]
    last = len(base["datasets"]) - 1
    groups = _issue_groups(r)
    lines = [
        f'<li>{_badge(s)}<span class="t" title="{_e(full)}">{_e(short)}</span></li>'
        for s, short, full in groups[:_OVERVIEW_ISSUES]
    ]
    rest = len(groups) - _OVERVIEW_ISSUES
    link = tabs.label(
        _CHECKS_TAB,
        (f"{rest} more under Checks →" if rest > 0 else "All checks and thresholds →"),
        cls="link",
    )
    issues = (
        f'<ul class="issues">{"".join(lines)}</ul>'
        if lines
        else '<p class="ok">No open findings.</p>'
    )
    review = _card("Review", issues + f'<div class="more">{link}</div>', extra=_counts(r))
    perf = _card("Performance", _headline(r))
    n = len(base["datasets"])

    def calibration(v: dict[str, Any]) -> str:
        return _card(
            _lift_takeaway(v, last) + v["suffix"],
            calibration_chart(v, _W_HALF, _H_ROW),
            titles=[_lift_takeaway(v, i) + v["suffix"] for i in range(n)],
            what="Deciles of predicted rate; A/E for every dataset, ±2σ shaded",
        )

    def effects(v: dict[str, Any]) -> str:
        against = f" against {r['model']['reference']}" if cl is not None else ""
        body = (
            _none("The reference class has no coefficients of its own.")
            if v.get("is_reference")
            else multiples_chart(v, _W_HALF, _H_ROW)
        )
        return _card(
            "The largest fitted effects" + v["suffix"],
            body,
            what=f"Main effects on one shared scale, as {_effect_name(v)}{against}",
        )

    imp_card = _card(
        _importance_takeaway(base),
        importance_chart(base, _W_HALF, _H_ROW),
        what=(
            "Each factor's share of the joint Wald statistics, across the classes"
            if cl is not None
            else "Each fitted factor's share of the deviance the model explains (joint Wald test)"
        ),
    )
    if cl is not None:
        fourth = _card(
            "Class mix",
            class_mix_chart(r, _W_HALF, _H_ROW),
            what="Actual and predicted share of each class, on the selected dataset",
        )
    else:
        fourth = _card(
            _variance_takeaway(base, last),
            variance_chart(base, _W_HALF, _H_ROW),
            titles=[_variance_takeaway(base, i) for i in range(n)],
            what="Observed variance over the family's V(μ) by band of fitted value: flat means V(μ) fits",
        )
    return (
        _open("Overview")
        + _row(perf, review)
        + _row(_per_class(views, cl, calibration), imp_card)
        + _row(_per_class(views, cl, effects), fourth)
        + "</section>"
    )


def _metrics_card(r: dict[str, Any]) -> str:
    scored = _scored(r)
    multi = bool(r["model"].get("multiclass"))
    keys = ("mass", "deviance", "null_deviance", "aic", "bic") + (
        () if multi else ("actual", "expected")
    )
    fmt = {k: _column([d["metrics"].get(k) for d in scored]) for k in keys}
    rows = []
    if multi:
        cols = [
            "dataset",
            _mass_name(r),
            "deviance",
            "null deviance",
            "deviance explained",
            "log loss",
            "accuracy",
            "top-2 accuracy",
            "balanced accuracy",
            "macro F1",
            "ECE",
        ]
        for i, d in enumerate(scored):
            m = d["metrics"]
            rows.append(
                [
                    _e(d["label"]),
                    fmt["mass"][i],
                    fmt["deviance"][i],
                    fmt["null_deviance"][i],
                    _pct(m.get("deviance_explained"), 2),
                    _fixed(m.get("log_loss")),
                    _pct(m.get("accuracy"), 1),
                    _pct(m.get("top_2_accuracy"), 1),
                    _pct(m.get("balanced_accuracy"), 1),
                    _fixed(m.get("macro_f1")),
                    _fixed(m.get("ece")),
                ]
            )
        what = "Deviance explained is against the class shares of the same rows (McFadden)"
    else:
        cols = [
            "dataset",
            "rows",
            _mass_name(r),
            "actual",
            "expected",
            "A/E",
            "deviance",
            "null deviance",
            "deviance explained",
            "Gini",
            "Gini / perfect",
            "AIC",
            "BIC",
        ]
        for i, d in enumerate(scored):
            m = d["metrics"]
            rows.append(
                [
                    _e(d["label"]),
                    _num(d["rows"]),
                    fmt["mass"][i],
                    fmt["actual"][i],
                    fmt["expected"][i],
                    _fixed(m.get("ae")),
                    fmt["deviance"][i],
                    fmt["null_deviance"][i],
                    _pct(m.get("deviance_explained"), 2),
                    _fixed(m.get("gini")),
                    _fixed(m.get("gini_normalised")),
                    fmt["aic"][i],
                    fmt["bic"][i],
                ]
            )
        what = (
            "Deviance explained is against the best constant model on the same rows, "
            "with the same exposure"
        )
    table = _table(
        cols, rows, numeric=range(1, len(cols)), row_class=[f"dsr-{i}" for i in range(len(scored))]
    )
    return _card("Metrics", table, what=what)


def _class_table(r: dict[str, Any], views: list[dict[str, Any]]) -> str:
    """Every class's one-vs-rest numbers, per dataset."""
    cols = [
        "class",
        "actual share",
        "predicted share",
        "A/E",
        "precision",
        "recall",
        "F1",
        "Gini",
        "deviance explained",
        "ECE",
    ]
    out = []
    for i in range(len(r["datasets"])):
        rows = []
        for v in views:
            m = v["datasets"][i]["metrics"]
            rows.append(
                [
                    _e(v["class"]) + (" (reference)" if v["is_reference"] else ""),
                    _pct(m.get("mean_actual"), 2),
                    _pct(m.get("mean_predicted"), 2),
                    _fixed(m.get("ae")),
                    _fixed(m.get("precision")),
                    _fixed(m.get("recall")),
                    _fixed(m.get("f1")),
                    _fixed(m.get("gini")),
                    _pct(m.get("deviance_explained"), 2),
                    _fixed(m.get("ece")),
                ]
            )
        out.append(
            f'<div class="ds ds-{i}">' + _table(cols, rows, numeric=range(1, len(cols))) + "</div>"
        )
    return _card("By class", "".join(out), what="One-vs-rest, on the selected dataset")


def _class_matrices(r: dict[str, Any]) -> str:
    out = []
    for i, d in enumerate(r["datasets"]):
        matrix = d["class_matrix"]
        labels = [str(c) for c in matrix["classes"]]
        heads = ["actual \\ predicted"] + labels
        numeric = range(1, len(labels) + 1)
        expected = [
            [_e(labels[a])] + [_amt(v) for v in row] for a, row in enumerate(matrix["expected"])
        ]
        confusion = [
            [_e(labels[a])] + [_amt(v) for v in row] for a, row in enumerate(matrix["confusion"])
        ]
        out.append(
            f'<div class="ds ds-{i}">'
            + _row(
                _card(
                    f"Expected counts · {d['label']}",
                    _table(heads, expected, numeric=numeric),
                    what="Sum of weight × P(column class), by actual class (row)",
                ),
                _card(
                    f"Most likely class · {d['label']}",
                    _table(heads, confusion, numeric=numeric),
                    what="Weighted count by actual class (row) and most likely class (column)",
                ),
            )
            + "</div>"
        )
    return "".join(out)


def _performance(r: dict[str, Any], views: list[dict[str, Any]], cl: _Radio | None) -> str:
    base = views[0]
    n = len(base["datasets"])
    last = n - 1

    def calibration(v: dict[str, Any]) -> str:
        dec_rows, dec_cls = [], []
        for i, d in enumerate(v["datasets"]):
            for b in d["deciles"]:
                dec_rows.append(
                    [
                        _e(d["label"]),
                        str(b["band"]),
                        _num(b["rows"]),
                        _pct(b.get("mass_share"), 1),
                        _num(b.get("actual"), 4),
                        _num(b.get("predicted"), 4),
                        _fixed(b.get("ae")),
                        _signed(b.get("z")),
                    ]
                )
                dec_cls.append(f"ds ds-{i}")
        table = _table(
            [
                "dataset",
                "decile",
                "rows",
                f"share of {_mass_name(v)}",
                "actual",
                "predicted",
                "A/E",
                "z",
            ],
            dec_rows,
            numeric=range(1, 8),
            row_class=dec_cls,
        )
        return _card(
            _lift_takeaway(v, last) + v["suffix"],
            calibration_chart(v, _W_FULL, 360)
            + f"<details><summary>Decile table</summary>{table}</details>",
            titles=[_lift_takeaway(v, i) + v["suffix"] for i in range(n)],
            what="Deciles of predicted rate: actual and predicted above, the A/E of each dataset below with ±2σ shaded",
        )

    def ranking(v: dict[str, Any]) -> str:
        lor = _card(
            "Lorenz curve" + v["suffix"],
            lorenz_chart(v, _W_HALF, 300),
            what=f"Share of actual against share of {_mass_name(v)}, lowest predicted first",
        )
        bal = _card(
            _balance_takeaway(v, last) + v["suffix"],
            balance_chart(v, _W_HALF),
            titles=[_balance_takeaway(v, i) + v["suffix"] for i in range(n)],
            what=_balance_rule(v),
        )
        return _row(lor, bal)

    def spread(v: dict[str, Any]) -> str:
        scored = _scored(v)
        keys = list((scored[0].get("rate_percentiles") or {}).keys())
        rows = [
            [_e(d["label"])] + [_num(x, 4) for x in (d.get("rate_percentiles") or {}).values()]
            for d in scored
        ]
        return _card(
            "Spread of the predictions" + v["suffix"],
            spread_chart(v, _W_FULL)
            + "<details><summary>Percentiles</summary>"
            + _table(["dataset"] + [f"p{q}" for q in keys], rows, numeric=range(1, len(keys) + 1))
            + "</details>",
            what=f"Share of {_mass_name(v)} by predicted {_rate_name(v)}, on bins shared by every dataset",
        )

    parts = [_metrics_card(r)]
    if cl is not None:
        parts += [_class_table(r, views), _class_matrices(r)]
    parts += [
        _per_class(views, cl, calibration),
        _per_class(views, cl, ranking),
        _per_class(views, cl, spread),
    ]
    if r.get("base_predictions"):
        parts.append(_base_card(r, r["base_predictions"]))
    return _open("Performance") + "".join(parts) + "</section>"


def _base_card(r: dict[str, Any], base: dict[str, Any]) -> str:
    head = [
        "dataset",
        "model A/E",
        "base A/E",
        "model loss",
        "base loss",
        "model Gini",
        "base Gini",
        "loss improvement",
    ]
    rows = []
    for key, d in zip(("train", "test"), r["datasets"]):
        side = base.get(key)
        if not side:
            continue
        mm, bm = side["model_metrics"], side["base_metrics"]
        rows.append(
            [
                _e(d["label"]),
                _fixed(mm.get("ae_ratio")),
                _fixed(bm.get("ae_ratio")),
                _num(mm.get("loss"), 5),
                _num(bm.get("loss"), 5),
                _fixed(mm.get("gini")),
                _fixed(bm.get("gini")),
                _pct((side.get("loss_improvement_pct") or 0.0) / 100.0, 2),
            ]
        )
    if not rows:
        return ""
    charts = "".join(
        f'<div class="ds ds-{i}">{_double_lift_chart(r, i, _W_FULL, 240)}</div>'
        for i in range(len(r["datasets"]))
    )
    return _card(
        "Compared with the base model",
        _table(head, rows, numeric=range(1, 8)) + charts,
        what="Model and base on the same rows; below, the A/E of each by decile of the model-to-base ratio",
    )


def _factors(r: dict[str, Any], views: list[dict[str, Any]], fx: _Radio, cl: _Radio | None) -> str:
    base = views[0]
    feats = base["features"]
    used = [f for f in feats if f["in_model"]]
    unused = [f for f in feats if not f["in_model"]]
    ordered = used + unused
    peak = max((float(f.get("relative_importance") or 0.0) for f in used), default=0.0) or 1.0
    by_name = {v["class"]: {str(f["feature"]): f for f in v["features"]} for v in views}
    rail = ['<div class="rail card">']

    def worst(f: dict[str, Any], i: int) -> str:
        live = [
            (j, b)
            for j, b in enumerate(f["bands"][i])
            if b.get("z") is not None and b.get("ae") is not None
        ]
        if not live:
            return ""
        j, b = max(live, key=lambda t: abs(t[1]["z"]))
        return f"worst {_short(str(f['labels'][j]), 12)}: {_fixed(b['ae'], 2)}, z {_signed(b['z'])}"

    def subs(name: str) -> str:
        out = []
        for k, v in enumerate(views):
            tag = "" if cl is None else f" {cl.tag(k)}"
            f = by_name[v["class"]][name]
            for i in range(len(v["datasets"])):
                out.append(f'<span class="w ds ds-{i}{tag}">{_e(worst(f, i))}</span>')
        return "".join(out)

    for j, f in enumerate(ordered):
        name = str(f["feature"])
        if j == len(used) and unused:
            rail.append('<div class="cap">Not in the model</div>')
        if f["in_model"]:
            share = float(f.get("relative_importance") or 0.0)
            width = share / peak * 100
            inner = (
                f'<span class="nm">{_e(name)}</span>'
                f'<span class="pc">{_pct(share / 100.0, 1)}</span>'
                f'<span class="bars"><i class="o" style="width:{width:.1f}%"></i></span>'
                + subs(name)
            )
            rail.append(fx.label(j, inner, cls="rl", title=name))
        else:
            st = f.get("score_test") or {}
            tag = f"p {_p(st.get('pvalue'))}" if st else "—"
            inner = f'<span class="nm">{_e(name)}</span><span class="pc">{tag}</span>' + subs(name)
            rail.append(fx.label(j, inner, cls="rl unused", title=name))
    rail.append("</div>")
    panes = []
    for j, f in enumerate(ordered):
        name = str(f["feature"])

        def body(v: dict[str, Any], name: str = name) -> str:
            this = by_name[v["class"]][name]
            flags = [
                fl
                for fl in r["review"]
                if fl["status"] != "pass"
                and fl.get("feature") == name
                and fl.get("class") in (None, v["class"])
            ]
            issues = "".join(
                f'<li>{_badge(fl["status"])}<span class="t" title="{_e(fl["message"])}">'
                + _e((f"{fl['dataset']}: " if fl.get("dataset") else "") + str(fl["message"]))
                + "</span></li>"
                for fl in flags
            )
            head = f'<ul class="issues" style="margin-bottom:10px">{issues}</ul>' if issues else ""
            return head + factor_chart(v, this, _W_FACTOR) + _factor_details(v, this)

        panes.append(
            f'<div class="fxp {fx.tag(j)}">'
            + _card(name, _per_class(views, cl, body), what=_factor_facts(f))
            + "</div>"
        )
    return (
        _open("Factors")
        + '<div class="fxgrid">'
        + "".join(rail)
        + '<div class="fxv">'
        + "".join(panes)
        + "</div></div></section>"
    )


def _factor_facts(f: dict[str, Any]) -> str:
    facts: list[str] = []
    if f["in_model"]:
        if f.get("relative_importance") is not None:
            facts.append(
                f"{_pct(float(f['relative_importance']) / 100.0, 1)} of the explained deviance"
            )
        sig = f.get("significance") or {}
        if sig.get("p") is not None:
            facts.append(f"joint test p {_p(sig['p'])}")
        if f.get("gvif") is not None:
            facts.append(f"GVIF {_num(f['gvif'], 3)}")
    else:
        st = f.get("score_test") or {}
        if st:
            facts.append(f"not in the model; score test p {_p(st.get('pvalue'))}")
        else:
            facts.append("not in the model")
    facts.append(f"{f['type']}, {len(f['labels'])} bands")
    if f.get("in_interactions"):
        facts.append("in " + ", ".join(str(x) for x in f["in_interactions"]))
    return " · ".join(facts)


def _factor_details(r: dict[str, Any], f: dict[str, Any]) -> str:
    scored = _scored(r)
    cols = ["band", "rows", _mass_name(r), "actual", "expected", "A/E", "z"]
    numbers = []
    for i in range(len(scored)):
        rows = [
            [
                _e(f["labels"][j]),
                _num(b["rows"]),
                _amt(b["mass"]),
                _amt(b["actual_total"]),
                _amt(b["expected_total"]),
                _fixed(b.get("ae")),
                _signed(b.get("z")),
            ]
            for j, b in enumerate(f["bands"][i])
            if b["rows"]
        ]
        numbers.append(
            f'<div class="ds ds-{i}">'
            + _table(cols, rows, numeric=range(1, 7), scroll="short")
            + "</div>"
        )
    out = "<details><summary>The numbers by band</summary>" + "".join(numbers) + "</details>"
    eff = f.get("effect")
    if eff and f["type"] == "categorical":
        name = eff["kind"]
        head = ["level", name] + (["95% low", "95% high"] if eff.get("lo") else [])
        rows2 = []
        for j, x in enumerate(eff["x"]):
            row = [_e(x), _num(eff["value"][j], 4)]
            if eff.get("lo"):
                row += [_num(eff["lo"][j], 4), _num(eff["hi"][j], 4)]
            rows2.append(row)
        out += (
            f"<details><summary>Fitted {_e(name)} by level (base: {_e(eff.get('base'))})</summary>"
            + _table(head, rows2, numeric=range(1, len(head)), scroll="short")
            + "</details>"
        )
    coefs = f.get("coefficients") or []
    if coefs and r["fit"]["inference_valid"]:
        rows3 = [
            [
                _e(c.get("term")),
                _num(c.get("estimate"), 4),
                _num(c.get("std_error"), 3),
                _fixed(c.get("z_value"), 2),
                _p(c.get("p_value")),
                _stars(c.get("p_value")),
            ]
            for c in coefs
        ]
        out += (
            "<details><summary>Coefficients</summary>"
            + _table(
                ["term", "estimate", "se", "z", "p", ""], rows3, numeric=range(1, 5), scroll="short"
            )
            + "</details>"
        )
    fd = next((x for x in r.get("factor_deviance") or [] if x["factor"] == f["feature"]), None)
    if fd:
        rows4 = [
            [
                _e(lv["level"]),
                _num(lv["n"]),
                _pct((lv.get("deviance_pct") or 0.0) / 100.0, 2),
                _num(lv.get("mean_deviance"), 4),
                _fixed(lv.get("ae_ratio")),
                "yes" if lv.get("problem") else "",
            ]
            for lv in fd.get("levels") or []
        ]
        out += (
            "<details><summary>Deviance by level (train)</summary>"
            + _table(
                ["level", "rows", "share of deviance", "mean deviance", "A/E", "problem"],
                rows4,
                numeric=range(1, 5),
                scroll="short",
            )
            + "</details>"
        )
    return out


def _interactions_tab(
    r: dict[str, Any], views: list[dict[str, Any]], ix: _Radio, cl: _Radio | None
) -> str:
    base = views[0]
    pairs = base.get("interactions") or []
    rail = ['<div class="rail card">']
    panes = []
    last = len(base["datasets"]) - 1
    th = r["thresholds"]
    if pairs:
        order = sorted(range(len(pairs)), key=lambda t: not pairs[t]["in_model"])
        n_fitted = sum(1 for p in pairs if p["in_model"])
        for j, at in enumerate(order):
            p = pairs[at]
            if j == n_fitted and n_fitted < len(pairs):
                rail.append('<div class="cap">Requested or detected, not in the model</div>')

            def line(v: dict[str, Any], at: int = at) -> str:
                cells = v["interactions"][at]["cells"][last]
                total = sum(c["mass"] for c in cells) or 1.0
                live = [
                    c
                    for c in cells
                    if c.get("ae") is not None
                    and c["mass"] / total >= (th.get("factor_min_mass") or 0.0)
                ]
                top = max(live, key=lambda c: abs(c.get("z") or 0.0), default=None)
                text = (
                    f"worst cell A/E {_fixed(top['ae'], 2)}, z {_signed(top.get('z'))}"
                    if top
                    else ""
                )
                return f'<span class="w">{_e(text)}</span>'

            inner = f'<span class="nm">{_e(p["name"])}</span>' + _per_class(views, cl, line)
            rail.append(ix.label(j, inner, cls="rl pair", title=str(p["name"])))

            def maps(v: dict[str, Any], at: int = at) -> str:
                q = v["interactions"][at]
                ae_map = _card(
                    "A/E by cell" + v["suffix"],
                    heatmap_chart(v, q, _W_INTER, kind="ae"),
                    what="Selected dataset; faded cells hold under the check's minimum share",
                )
                right = (
                    _card(
                        "Fitted interaction" + v["suffix"],
                        heatmap_chart(v, q, _W_INTER, kind="effect"),
                        what=f"The interaction term alone, as {q['effect']['kind']}, against the first cell",
                    )
                    if q.get("effect")
                    else _card(
                        "Exposure" + v["suffix"],
                        heatmap_chart(v, q, _W_INTER, kind="mass"),
                        what=f"Share of {_mass_name(v)} by cell",
                    )
                )
                return _row(ae_map, right)

            def coefs(v: dict[str, Any], at: int = at) -> str:
                rows = [
                    [
                        _e(c.get("term")),
                        _num(c.get("estimate"), 4),
                        _num(c.get("std_error"), 3),
                        _fixed(c.get("z_value"), 2),
                        _p(c.get("p_value")),
                    ]
                    for c in v["interactions"][at].get("coefficients") or []
                ]
                if not rows or not r["fit"]["inference_valid"]:
                    return ""
                return (
                    "<details><summary>Coefficients</summary>"
                    + _table(
                        ["term", "estimate", "se", "z", "p"],
                        rows,
                        numeric=range(1, 5),
                        scroll="short",
                    )
                    + "</details>"
                )

            facts = [p["pair_type"].replace("_", " ")]
            sig, st = p.get("significance") or {}, p.get("score_test") or {}
            if sig.get("p") is not None:
                facts.append(f"joint test p {_p(sig['p'])}")
            if st.get("pvalue") is not None:
                facts.append(f"score test p {_p(st['pvalue'])}")
            if p.get("gvif") is not None:
                facts.append(f"GVIF {_num(p['gvif'], 3)}")
            status = "in the model" if p["in_model"] else "not in the model"
            head = _card(
                p["name"],
                _per_class(views, cl, coefs) or '<span class="muted"></span>',
                what=f"{status} · " + " · ".join(facts),
            )
            panes.append(f'<div class="fxp {ix.tag(j)}">{head}{_per_class(views, cl, maps)}</div>')
        rail.append("</div>")
        body = (
            '<div class="fxgrid">'
            + "".join(rail)
            + '<div class="fxv">'
            + "".join(panes)
            + "</div></div>"
        )
    else:
        body = _card(
            "No interaction pairs",
            '<p class="note">The model has no fitted interactions and none were requested. Pass '
            "interactions=[(a, b), ...] to check a pair of factors.</p>",
        )
    return _open("Interactions") + body + _candidates(r) + "</section>"


def _candidates(r: dict[str, Any]) -> str:
    cands = r.get("interaction_candidates") or []
    if not cands:
        return ""
    rows = [
        [
            _e(c["factor1"]),
            _e(c["factor2"]),
            _num(c.get("interaction_strength"), 3),
            _p(c.get("pvalue")),
            _num(c.get("n_cells")),
        ]
        for c in cands
    ]
    return _card(
        "Detected interaction candidates",
        _table(
            ["factor 1", "factor 2", "strength", "p", "cells"],
            rows,
            numeric=(2, 3, 4),
            scroll="short",
        ),
        what="Pairs of factors whose residuals vary together; request one with interactions= to see its grid",
    )


def _coefficients_tab(r: dict[str, Any], views: list[dict[str, Any]], cl: _Radio | None) -> str:
    if not any(v["coefficients"] for v in views):
        return (
            _open("Coefficients")
            + _card(
                "Coefficients",
                '<p class="note">Not available: this model was loaded from a saved file, which '
                "does not hold standard errors. Report a freshly fitted model to see them.</p>",
            )
            + "</section>"
        )
    valid = r["fit"]["inference_valid"]

    def cards(v: dict[str, Any]) -> str:
        if v.get("is_reference"):
            return _card(
                f"Reference class{v['suffix']}",
                '<p class="note">The other classes are fitted against this one, so it has no '
                "coefficients of its own.</p>",
            )
        coefs = v["coefficients"]
        robust = any(c.get("robust_std_error") is not None for c in coefs)
        name = _effect_name(v)
        head = (
            ["term", "estimate", "se"]
            + (["robust se"] if robust else [])
            + ["z", "p", "", name, "95% low", "95% high"]
        )
        mult = _is_mult(v)
        rows = []
        for c in coefs:
            basis = bool(_BASIS_NAME.match(str(c["feature"])))
            ci = None if basis else (c.get("relativity_ci") if mult else c.get("conf_int"))
            est = None if basis else (c.get("relativity") if mult else c.get("estimate"))
            rows.append(
                [
                    _e(c["feature"]),
                    _num(c.get("estimate"), 4),
                    _num(c.get("std_error"), 3),
                    *([_num(c.get("robust_std_error"), 3)] if robust else []),
                    _fixed(c.get("z_value"), 2),
                    _p(c.get("p_value")),
                    _stars(c.get("p_value")) if valid else "",
                    _num(est, 4),
                    _num(ci[0], 4) if ci else "—",
                    _num(ci[1], 4) if ci else "—",
                ]
            )
        note = ""
        if not valid:
            status = r["fit"].get("inference_status") or "unavailable"
            note = (
                f'<p class="note">The standard errors here are not valid as usual ({_e(status)}): '
                "read z, p and the intervals with care.</p>"
            )
        table = _table(
            head, rows, numeric=[i for i in range(1, len(head)) if head[i] != ""], scroll="scroll"
        )
        limit = 40
        n_plot = sum(1 for c in coefs if not _is_basis_name(str(c["feature"])))
        base_rate = r["fit"].get("base_rate")
        against = f", against {r['model']['reference']}" if cl is not None else ""
        return _card(
            _coef_takeaway(v) + v["suffix"],
            note + forest_chart(v, _W_FULL, limit=limit),
            what=(
                f"Point estimate and 95% interval{against}, per unit for a continuous term; grey "
                "points are not significant at 5%; spline bases are on the Factors tab"
                + (f"; the {limit} with the largest |z|" if n_plot > limit else "")
            ),
        ) + _card(
            "All coefficients" + v["suffix"],
            table,
            what=(
                f"Estimates on the link scale, intercept included; base rate {_num(base_rate, 5)}"
                if base_rate is not None
                else "Estimates on the link scale"
            ),
        )

    parts = [_per_class(views, cl, cards)]
    shared = r.get("alternative_coefficients") or []
    if shared:
        rows = [
            [
                _e(c["feature"]),
                _num(c.get("estimate"), 4),
                _num(c.get("std_error"), 3),
                _fixed(c.get("z_value"), 2),
                _p(c.get("p_value")),
            ]
            for c in shared
        ]
        parts.append(
            _card(
                "Coefficients shared by every class",
                _table(["term", "estimate", "se", "z", "p"], rows, numeric=range(1, 5)),
                what="Alternative-specific attributes with one coefficient",
            )
        )
    return _open("Coefficients") + "".join(parts) + "</section>"


def _gof_table(r: dict[str, Any]) -> str:
    fit = r["fit"]
    summary = fit["summary"]
    d0 = r["datasets"][0]["metrics"]
    comparison = fit.get("model_comparison") or {}
    df_resid = summary.get("df_resid")
    dev = d0.get("deviance")
    rows = [
        ("Deviance", _num(dev)),
        ("Null deviance", _num(d0.get("null_deviance"))),
        (
            "Deviance / residual df",
            _num(dev / df_resid, 4) if dev is not None and df_resid else "—",
        ),
        ("Pearson φ (row level)", _num(fit.get("pearson_dispersion"), 4)),
        ("Dispersion (variance over V(μ))", _num(fit.get("dispersion"), 4)),
        ("Scale (used for standard errors)", _num(summary.get("scale"), 4)),
        ("Likelihood-ratio χ²", _num(comparison.get("likelihood_ratio_chi2"))),
        ("LR test df", _num(comparison.get("likelihood_ratio_df"))),
        ("LR test p", _p(comparison.get("likelihood_ratio_pvalue"))),
        ("Hosmer–Lemeshow p", _p((fit.get("calibration") or {}).get("hl_pvalue"))),
        ("Log-likelihood", _num(d0.get("log_likelihood"))),
        ("AIC", _num(d0.get("aic"))),
        ("BIC", _num(d0.get("bic"))),
    ]
    body = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in rows if v != "—")
    return f'<table class="kv">{body}</table>'


def _count_table(r: dict[str, Any]) -> str:
    out = []
    for i, d in enumerate(r["datasets"]):
        cd = d.get("count_distribution")
        if not cd:
            continue
        top = int(cd["top"])
        rows = []
        for k, (ob, ex) in enumerate(zip(cd["observed"], cd["expected"])):
            label = str(k) if k < top else f"≥ {top}"
            rows.append([label, _amt(ob), _amt(ex), _fixed(_safe_div(ob, ex))])
        out.append(
            f'<div class="ds ds-{i}">'
            + _table(["count", "observed", "expected", "ratio"], rows, numeric=(1, 2, 3))
            + "</div>"
        )
    return "".join(out)


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def _multinomial_fit_tab(r: dict[str, Any], views: list[dict[str, Any]], cl: _Radio | None) -> str:
    fit = r["fit"]
    summary = fit["summary"]
    conv_rows = [
        ("Converged", _e(summary.get("converged"))),
        ("Iterations", _num(summary.get("iterations"))),
        ("Solver", _e(fit.get("optimizer_route") or "—")),
        ("Solver status", _e(fit.get("solver_status") or "—")),
        ("Standard errors", _e(fit.get("inference_status") or "—")),
        ("Parameters", _num(summary.get("n_params"))),
        ("Residual df", _num(summary.get("df_resid"))),
    ]
    conv = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in conv_rows)
    parts = [
        _row(
            _card("Goodness of fit", _gof_table(r), what="On the train data"),
            _card("Convergence", f'<table class="kv">{conv}</table>', what="How the solver ran"),
        )
    ]

    def bands(v: dict[str, Any]) -> str:
        return _card(
            "A/E by band of fitted probability" + v["suffix"],
            fitted_ae_chart(v, _W_FULL, 260),
            what="Twenty bands of equal rows, ±2σ shaded; a bend shows a missing term",
        )

    parts.append(_per_class(views, cl, bands))
    corr_rows = [
        [_e(c["a"]), _e(c["b"]), _fixed(c["corr"], 3)] for c in fit["coefficient_correlations"]
    ]
    if corr_rows:
        parts.append(
            _card(
                "Largest correlations between estimates",
                _table(
                    ["coefficient", "coefficient", "correlation"],
                    corr_rows,
                    numeric=(2,),
                    scroll="short",
                ),
                what="Between coefficients of different terms",
            )
        )
    base = views[0]
    parts.append(_terms_card(base))
    for card in (_smooth_card(r), _regularisation_card(r)):
        if card:
            parts.append(card)
    return _open("Fit") + "".join(parts) + "</section>"


def _fit_tab(r: dict[str, Any]) -> str:
    fit = r["fit"]
    scored = _scored(r)
    last = len(scored) - 1
    summary = fit["summary"]
    parts: list[str] = []
    gof = _card("Goodness of fit", _gof_table(r), what="On the train data")
    conv_rows = [
        ("Converged", _e(summary.get("converged"))),
        ("Iterations", _num(summary.get("iterations"))),
        ("Solver", _e(fit.get("optimizer_route") or "—")),
        ("Solver status", _e(fit.get("solver_status") or "—")),
        ("Standard errors", _e(fit.get("inference_status") or "—")),
        ("Parameters", _num(summary.get("n_params"))),
        ("Residual df", _num(summary.get("df_resid"))),
    ]
    fd = fit.get("fit_diagnostics") or {}
    eta, fitted = fd.get("eta") or {}, fd.get("fitted_value") or {}
    if eta:
        conv_rows.append(
            (
                "Linear predictor (min, p50, max)",
                f"{_num(eta.get('min'))}, {_num(eta.get('p50'))}, {_num(eta.get('max'))}",
            )
        )
    if fitted:
        conv_rows.append(
            (
                "Fitted value (min, p50, max)",
                f"{_num(fitted.get('min'))}, {_num(fitted.get('p50'))}, {_num(fitted.get('max'))}",
            )
        )
    conv = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in conv_rows)
    trace = fit.get("solver_trace") or []
    trace_html = ""
    if trace:
        rows = [[_e(x[0]), _e(x[1]), _num(x[2]), _e(x[3]), _e(x[4])] for x in trace[:60]]
        trace_html = (
            "<details><summary>Solver trace</summary>"
            + _table(
                ["driver", "step", "evaluations", "accepted", "status"],
                rows,
                numeric=(2,),
                scroll="short",
            )
            + "</details>"
        )
    conv_card = _card(
        "Convergence", f'<table class="kv">{conv}</table>{trace_html}', what="How the solver ran"
    )
    parts.append(_row(gof, conv_card))
    var_card = _card(
        _variance_takeaway(r, last),
        variance_chart(r, _W_HALF, 260),
        titles=[_variance_takeaway(r, i) for i in range(len(scored))],
        what="Observed variance over the family's V(μ) by band of fitted value: flat means V(μ) fits",
    )
    link_card = _card(
        "A/E by band of fitted value",
        fitted_ae_chart(r, _W_HALF, 260),
        what="Twenty bands of equal rows, ±2σ shaded; a bend shows the link or a missing term",
    )
    parts.append(_row(var_card, link_card))
    dist_card = _card(
        "Deviance residuals",
        histogram_chart(r, _W_HALF, 260),
        what="Distribution over the central 99%; expect skew for counts and low means",
    )
    has_qq = any(d.get("qq") for d in scored)
    has_counts = any(d.get("count_distribution") for d in scored)
    if has_qq:
        second = _card(
            "Normal QQ plot",
            qq_chart(r, _W_HALF, 260),
            what="Standardised deviance residuals against a standard normal",
        )
    elif has_counts:
        second = _card(
            "Counts: observed against expected",
            _count_table(r),
            what="Claims per row under the fitted distribution; a shortfall at 0 or a heavy tail shows dispersion",
        )
    else:
        second = _card(
            "Residuals",
            '<p class="note">A normal QQ plot is not meaningful for this family; the variance check and A/E by band above take its place.</p>',
        )
    parts.append(_row(dist_card, second))
    gvif_rows = []
    for f in r["features"]:
        if f["in_model"] and f.get("gvif") is not None:
            df = max(int(f.get("df") or 1), 1)
            adj = float(f["gvif"]) ** (1.0 / df)
            gvif_rows.append(
                (
                    str(f["feature"]),
                    adj,
                    f"{f['feature']}: GVIF {_num(f['gvif'], 3)} on {df} column(s), GVIF^(1/df) {_num(adj, 3)}",
                )
            )
    gvif_rows.sort(key=lambda x: -x[1])
    vif_note = (
        f"Above {_num(r['thresholds'].get('gvif'))} is flagged."
        if r["thresholds"].get("gvif") is not None
        else ""
    )
    gvif_card = _card(
        "Collinearity",
        (
            _bars_chart(gvif_rows, _W_HALF, fmt=lambda v: _num(v, 3))
            if gvif_rows
            else '<p class="note">Not available for this model.</p>'
        ),
        what=f"GVIF^(1/df) per fitted factor; 1 is no inflation. {vif_note}",
    )
    corr = fit.get("coefficient_correlations") or []
    corr_rows = [[_e(c["a"]), _e(c["b"]), _fixed(c["corr"], 3)] for c in corr]
    corr_card = _card(
        "Largest correlations between estimates",
        _table(
            ["coefficient", "coefficient", "correlation"], corr_rows, numeric=(2,), scroll="short"
        )
        if corr_rows
        else '<p class="note">Not available for this model.</p>',
        what="Between coefficients of different terms",
    )
    parts.append(_row(gvif_card, corr_card))
    parts.append(_influence_card(r))
    parts.append(_terms_card(r))
    smooth = _smooth_card(r)
    if smooth:
        parts.append(smooth)
    reg = _regularisation_card(r)
    if reg:
        parts.append(reg)
    return _open("Fit") + "".join(parts) + "</section>"


def _influence_card(r: dict[str, Any]) -> str:
    inf = r["fit"].get("influence") or {}
    if not inf.get("available"):
        return _card(
            "Influence", f'<p class="note">Not available: {_e(inf.get("reason") or "unknown")}.</p>'
        )
    rows = [
        [
            _num(t["row"]),
            _num(t["y"], 4),
            _num(t["mu"], 4),
            _num(t["exposure"], 4),
            _num(t["leverage"], 3),
            _fixed(t["pearson"], 2),
            _num(t["cooks"], 3),
        ]
        for t in inf.get("top") or []
    ]
    table = _table(
        ["train row", "y", "fitted", "exposure", "leverage", "Pearson", "Cook's D"],
        rows,
        numeric=range(0, 7),
        scroll="short",
    )
    facts = (
        f"max leverage {_num(inf['max_leverage'], 3)} (rule of thumb {_num(inf['leverage_rule'], 3)}, "
        f"{_num(inf['high_leverage_rows'])} rows above); max Cook's D {_num(inf['max_cooks'], 3)}"
    )
    return _card("Influence", table, what=facts)


def _terms_card(r: dict[str, Any]) -> str:
    terms = r["fit"].get("terms") or []
    if not terms:
        return ""
    enc = {e["name"]: e for e in r.get("encodings") or []}
    rows = []
    for t in terms:
        factor_name = t["factors"][0] if len(t["factors"]) == 1 else None
        fdict = _feature(r, factor_name) if factor_name else None
        e = enc.get(t["term"]) or (enc.get(factor_name) if factor_name else None) or {}
        rows.append(
            [
                _e(t["term"]),
                _e(t["type"]),
                _num(t["columns"]),
                _p(((fdict or {}).get("significance") or {}).get("p")) if fdict else "—",
                _num(e.get("n_levels_train")) if e.get("n_levels_train") is not None else "—",
                _num(e.get("n_levels_test")) if e.get("n_levels_test") is not None else "—",
                _num(e.get("unseen_levels_test"))
                if e.get("unseen_levels_test") is not None
                else "—",
            ]
        )
    return _card(
        "Terms",
        _table(
            ["term", "type", "columns", "joint p", "levels (train)", "levels (test)", "unseen"],
            rows,
            numeric=(2, 3, 4, 5, 6),
            scroll="short",
        ),
        what="The model's terms and how each is encoded",
    )


def _smooth_card(r: dict[str, Any]) -> str:
    smooth = r["fit"].get("smooth_terms") or []
    if not smooth:
        return ""
    rows = [
        [
            _e(s["variable"]),
            _num(s["k"]),
            _fixed(s["edf"], 2),
            _num(s["lambda"], 3),
            _num(s["gcv"], 3),
            _num(s["chi2"], 3),
            _p(s["p_value"]),
        ]
        for s in smooth
    ]
    return _card(
        "Smooth terms",
        _table(
            ["term", "k", "edf", "λ", "GCV", "χ²", "p"], rows, numeric=range(1, 7), scroll="short"
        ),
        what="Effective degrees of freedom and the smoothing the fit chose",
    )


def _regularisation_card(r: dict[str, Any]) -> str:
    reg = r["fit"]["summary"].get("regularization")
    path = r["fit"].get("regularization_path")
    if not reg and not path:
        return ""
    chart = path_chart(r, _W_FULL, 240)
    facts = ""
    if reg:
        facts = (
            f"{_e(reg.get('type'))}, α {_num(reg.get('alpha'), 4)}, L1 ratio {_num(reg.get('l1_ratio'), 2)}"
            + (
                f", CV deviance {_num(reg.get('cv_deviance'), 5)}"
                if reg.get("cv_deviance") is not None
                else ""
            )
        )
    return _card("Regularisation", (chart or ""), what=facts or "The penalty the fit used")


def _checks(r: dict[str, Any], base: dict[str, Any]) -> str:
    order = {"critical": 0, "warning": 1, "info": 2, "pass": 3}
    flags = sorted(r["review"], key=lambda f: (order.get(f["status"], 9), f["check"]))
    th = r["thresholds"]

    def rows(items: list[dict[str, Any]]) -> str:
        body = [
            [
                _badge(f["status"]),
                _e(f["check"]),
                _e(
                    " · ".join(
                        str(x)
                        for x in (f.get("dataset"), f.get("feature"), f.get("class"))
                        if x is not None
                    )
                ),
                _e(f["message"]),
                _num(f["value"]) if not isinstance(f["value"], str) else _e(f["value"]),
                _num(f["threshold"]),
            ]
            for f in items
        ]
        return _table(
            ["status", "check", "scope", "finding", "value", "threshold"], body, numeric=(4, 5)
        )

    open_items = [f for f in flags if f["status"] != "pass"]
    passed = [f for f in flags if f["status"] == "pass"]
    th_rows = [
        [_e(k), "off" if v is None else _num(v), _e(r["threshold_descriptions"].get(k, ""))]
        for k, v in th.items()
    ]
    review = _card(
        "Review",
        (rows(open_items) if open_items else '<p class="ok">No open findings.</p>')
        + (
            f"<details><summary>{len(passed)} passed checks</summary>{rows(passed)}</details>"
            if passed
            else ""
        )
        + "<details><summary>Thresholds in force</summary>"
        + _table(["threshold", "value", "meaning"], th_rows, numeric=(1,))
        + "</details>",
        what="Flags for a reviewer, not a verdict; set the thresholds with the thresholds argument",
        extra=_counts(r),
    )
    return (
        _open("Checks")
        + review
        + _row(_data_card(base), _model_card(r))
        + _how_card(r)
        + "</section>"
    )


def _data_card(r: dict[str, Any]) -> str:
    th = r["thresholds"]
    scored = _scored(r)
    out = []
    feats = r["features"]
    any_missing = any(m["rows"] for f in feats for m in f["missing"])
    if not any_missing:
        out.append('<p class="ok">No missing or unseen values in any factor.</p>')
    else:
        rows = [
            [_e(f["feature"])]
            + [f"{_num(m['rows'])} ({_pct(m['mass_share'], 2)})" for m in f["missing"]]
            for f in feats
            if any(m["rows"] for m in f["missing"])
        ]
        out.append(
            "<h4>Missing or unseen values: rows (share of mass)</h4>"
            + _table(
                ["factor"] + [str(d["label"]) for d in scored],
                rows,
                numeric=range(1, len(scored) + 1),
            )
        )
    with_psi = [f for f in feats if f.get("psi") is not None]
    if with_psi:
        worst = max(with_psi, key=lambda f: f["psi"])
        warn = th.get("psi_warn")
        cls = "ok" if warn is None or worst["psi"] <= warn else "note"
        out.append(
            f'<p class="{cls}">Largest PSI {_num(worst["psi"], 3)} ({_e(worst["feature"])})'
            + (f"; warning above {_num(warn)}" if warn is not None else "")
            + ".</p>"
        )
        psi_rows = [
            [_e(f["feature"]), _num(f["psi"], 3)] for f in sorted(with_psi, key=lambda f: -f["psi"])
        ]
        out.append(
            "<details><summary>Population stability by factor</summary>"
            + _table(["factor", "PSI, train → test"], psi_rows, numeric=(1,))
            + "</details>"
        )
    return _card(
        "Data", "".join(out), what="PSI compares the supplied datasets; it is not a forecast"
    )


def _model_card(r: dict[str, Any]) -> str:
    m = r["model"]
    return _card(
        "Model",
        _kv(
            {
                "formula": m.get("formula"),
                "family": m.get("family"),
                "link": m.get("link"),
                "classes": ", ".join(str(c) for c in m["classes"]) if m.get("multiclass") else None,
                "reference class": m.get("reference"),
                "response": m.get("response"),
                "exposure column": m.get("exposure_column"),
                "weights column": m.get("weights_column"),
                "offset column": m.get("offset_column"),
                "loaded from a file": m.get("deserialized"),
                "report version": r["report_version"],
                "generated": r.get("generated_at"),
            }
        ),
    )


def _how_card(r: dict[str, Any]) -> str:
    items = [
        (
            "Deviance",
            "the family's total unit deviance of y against each row's fitted value, weighted by prior weight.",
        ),
        (
            "Deviance explained",
            "1 − deviance / null deviance; the null model is the best constant on the same rows, with the same exposure and offset.",
        ),
        (
            "Gini",
            "from the diagnostics: the Lorenz Gini of actual against predicted rate, exposure on the x-axis. Gini / perfect divides by the Gini of ranking by the actual rate.",
        ),
        (
            "A/E and z",
            "actual over expected in a band; z = (actual − expected) / σ, where σ² sums φ·weight·V(μ) over the band's rows and φ is the dispersion.",
        ),
        (
            "Dispersion",
            "Σ w (y − μ)² / Σ w V(μ) on the train data: the variance of a total against what the family assumes. The row-level Pearson φ weights rows by 1 / V(μ), so rows with a very small expected value dominate it.",
        ),
        (
            "Deciles and bands",
            "deciles of predicted rate with equal rows, the partition diagnostics uses. Factor bands: quantile bands of the train data per continuous factor, levels for a categorical; the same bands for every dataset.",
        ),
        (
            "Fitted effects",
            "the factor's main-effect term alone, relative to a reference value, with a 95% band from the fit's covariance. Partial dependence is drawn when the factor is also in an interaction.",
        ),
        (
            "Train A/E of a fitted categorical",
            "1 by construction: the score equations of a canonical-link fit balance each level. The check is the test data.",
        ),
        (
            "Factor importance",
            "each fitted factor's share of the sum of the joint Wald statistics of the fitted factors.",
        ),
        (
            "GVIF",
            "the generalised variance inflation factor of a term's columns, as GVIF^(1/df) so terms with several columns compare.",
        ),
        (
            "Hosmer–Lemeshow",
            "reported on the Fit tab, not flagged: on large data it rejects gaps too small to matter. The deciles and their z are the check.",
        ),
        (
            "Classes",
            "a multinomial model is reviewed class by class: each class's probability against whether the row was that class (one-vs-rest). Effects are odds against the reference class. Deviance explained on the Metrics card is McFadden's, against the class shares of the same rows.",
        ),
        (
            "PSI",
            "compares the train data with the test data on the factor's bands; shares floored at 0.0001.",
        ),
    ]
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in items)
    caveat = f'<p class="note">{_e(r.get("caveat"))}</p>' if r.get("caveat") else ""
    return _card(
        "How to read this report",
        caveat
        + f'<details><summary>Definitions</summary><table class="kv">{rows}</table></details>',
    )


# --------------------------------------------------------------------------- entry point ------


def report_html(report: dict[str, Any], *, name: str | None = None) -> str:
    """Render a validation report as one self-contained HTML page.

    ``report`` is the dictionary from ``GLMModel.validation_report``, or the same document read
    back from JSON. ``name`` overrides the report's own name in the heading. The page holds no
    scripts and loads nothing from the network, so it can be logged as an artifact (to MLflow,
    for example) and opened offline. Raises ``ValidationError`` for a report of another version.
    """
    version = report.get("report_version") if isinstance(report, dict) else None
    if version not in _SUPPORTED_VERSIONS:
        raise ValidationError(
            f"report_html renders report_version {_SUPPORTED_VERSIONS}, got {version!r}; "
            "build the report with validation_report from this rustystats."
        )
    r = report
    views = _views(r)
    base = views[0]
    multi = bool(r["model"].get("multiclass"))
    tabs = _Radio("tb", len(_TAB_NAMES))
    n_datasets = len(base["datasets"])
    ds = _Radio("ds", n_datasets, n_datasets - 1)
    default = next((k for k, v in enumerate(views) if not v["is_reference"]), 0)
    cl = _Radio("cl", len(views), default) if multi else None
    fx = _Radio("fx", len(base["features"]))
    ix = _Radio("ix", len(base.get("interactions") or []))
    switches = []
    if n_datasets > 1:
        switches.append(
            '<div class="switch" role="group" aria-label="dataset"><span class="cap">dataset</span>'
            + "".join(ds.label(i, _e(d["label"])) for i, d in enumerate(base["datasets"]))
            + "</div>"
        )
    if cl is not None:
        switches.append(
            '<div class="switch" role="group" aria-label="class"><span class="cap">class</span>'
            + "".join(cl.label(k, _e(v["class"])) for k, v in enumerate(views))
            + "</div>"
        )
    bar = (
        '<nav class="bar"><div class="tabs">'
        + "".join(tabs.label(i, _e(t)) for i, t in enumerate(_TAB_NAMES))
        + f'</div><div class="switches">{"".join(switches)}</div></nav>'
    )
    fit_tab = _multinomial_fit_tab(r, views, cl) if multi else _fit_tab(base)
    body = "".join(
        [
            _overview(r, views, tabs, cl),
            _performance(r, views, cl),
            _factors(r, views, fx, cl),
            _interactions_tab(r, views, ix, cl),
            _coefficients_tab(r, views, cl),
            fit_tab,
            _checks(r, base),
        ]
    )
    switch_on = "background:var(--wash);color:var(--ink);box-shadow:inset 0 -2px 0 var(--data)"
    rail_on = "background:var(--wash);box-shadow:inset 3px 0 0 var(--model)"
    css = (
        _CSS
        % {
            "light": _LIGHT,
            "dark": _DARK,
            "content": _CONTENT,
            "gap": _GAP,
            "font": _FONT,
            "rail": _RAIL,
            "list": 560,
        }
        + _CSS_EXTRA
        + tabs.css("background:var(--surface);color:var(--ink);border-color:var(--border)")
        + ds.css(switch_on)
        + (cl.css(switch_on) if cl is not None else "")
        + fx.css(rail_on)
        + ix.css(rail_on)
        + _dataset_css(n_datasets)
    )
    inputs = (
        tabs.inputs()
        + ds.inputs()
        + (cl.inputs() if cl is not None else "")
        + fx.inputs()
        + ix.inputs()
    )
    title = name or r.get("name") or f"{r['model']['family']} GLM validation"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{css}</style></head><body>"
        f'<div class="app">{inputs}{_header(r, base, name)}{bar}<main class="body">{body}</main>'
        "</div></body></html>"
    )
