"""
ONNX export for RustyStats GLM models.

Two export modes:

* **scoring** (default) -- The ONNX model takes a pre-built design matrix
  ``X`` of shape ``(batch, n_features)`` and produces predictions.  The
  intercept is handled internally.  This is the simplest, most portable
  format; the consumer builds the design matrix using the model's
  ``feature_names`` and stored preprocessing state.

* **full** -- The ONNX model takes raw feature values (continuous as
  float64, categorical as integer codes) and embeds the full
  preprocessing pipeline (one-hot encoding, piecewise-linear spline
  approximation, TE/FE lookups, interactions) inside the graph.

Protobuf serialization is implemented from scratch in Rust -- no external
dependencies beyond numpy are required.
"""

from __future__ import annotations

import json
import re
import numpy as np
from collections import OrderedDict, defaultdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rustystats.formula import GLMModel

from rustystats.export_pmml import _classify_feature


# ── helpers ──────────────────────────────────────────────────────────────────


def _inverse_link_name(link: str) -> str:
    """Map RustyStats link name to a description for metadata."""
    return {
        'log': 'exp',
        'logit': 'sigmoid',
        'identity': 'identity',
        'inverse': '1/x',
        'sqrt': 'square',
        'cloglog': 'cloglog_inv',
        'probit': 'probit_inv',
    }.get(link, link)


def _get_builder_attr(model: "GLMModel", attr: str, default=None):
    builder = getattr(model, '_builder', None)
    if builder is None:
        return default
    return getattr(builder, attr, default)


def _get_all_levels(model: "GLMModel", var: str) -> List[str]:
    builder = getattr(model, '_builder', None)
    if builder is None:
        return []
    cache = getattr(builder, '_cat_encoding_cache', {})
    cached = cache.get(f"{var}_True")
    if cached is not None:
        return list(cached.levels)
    return []


# ── Level 1: scoring mode ───────────────────────────────────────────────────

def _build_scoring_model(model: "GLMModel") -> bytes:
    """Build a Level-1 ONNX model: design matrix -> prediction.

    Uses the Rust protobuf serializer directly.
    """
    from rustystats._rustystats import build_onnx_glm_scoring_py

    names = model.feature_names
    params = np.asarray(model.params, dtype=np.float64)

    # Separate intercept from the rest
    intercept = 0.0
    coef_indices = []
    for i, nm in enumerate(names):
        if nm == 'Intercept':
            intercept = float(params[i])
        else:
            coef_indices.append(i)

    coefs = params[coef_indices] if coef_indices else np.zeros(0, dtype=np.float64)
    n_features = len(coef_indices)

    # Metadata
    meta_keys = [
        'feature_names',
        'inverse_link',
        'family',
        'formula',
    ]
    meta_values = [
        json.dumps([names[i] for i in coef_indices]),
        _inverse_link_name(model.link),
        model.family,
        getattr(model, 'formula', '') or '',
    ]

    return bytes(build_onnx_glm_scoring_py(
        coefs, intercept, n_features,
        model.link, model.family,
        meta_keys, meta_values,
    ))


# ── Level 2: full preprocessing mode ────────────────────────────────────────

class _GraphAccumulator:
    """Accumulate ONNX graph components as plain Python lists.

    Collects nodes, initializers, inputs and outputs as parallel lists
    that can be passed directly to the Rust ``serialize_onnx_graph_py``
    function for protobuf serialization.
    """

    def __init__(self):
        self._cnt = 0
        # Nodes
        self.node_ops: List[str] = []
        self.node_inputs: List[List[str]] = []
        self.node_outputs: List[List[str]] = []
        self.node_attr_names: List[List[str]] = []
        self.node_attr_types: List[List[str]] = []
        self.node_attr_ints: List[List[int]] = []
        self.node_attr_floats: List[List[float]] = []
        # Initializers
        self.init_names_f64: List[str] = []
        self.init_data_f64: List[List[float]] = []
        self.init_shapes_f64: List[List[int]] = []
        self.init_names_i64: List[str] = []
        self.init_data_i64: List[List[int]] = []
        self.init_shapes_i64: List[List[int]] = []

    def uid(self, prefix: str = "t") -> str:
        self._cnt += 1
        return f"{prefix}_{self._cnt}"

    def add_node(self, op: str, inputs: List[str], outputs: List[str],
                 attrs: Optional[List[Tuple[str, str, Any]]] = None):
        self.node_ops.append(op)
        self.node_inputs.append(list(inputs))
        self.node_outputs.append(list(outputs))
        a_names, a_types, a_ints, a_floats = [], [], [], []
        for name, atype, val in (attrs or []):
            a_names.append(name)
            a_types.append(atype)
            a_ints.append(int(val) if atype == "int" else 0)
            a_floats.append(float(val) if atype == "float" else 0.0)
        self.node_attr_names.append(a_names)
        self.node_attr_types.append(a_types)
        self.node_attr_ints.append(a_ints)
        self.node_attr_floats.append(a_floats)

    def add_init_f64(self, name: str, data):
        arr = np.asarray(data, dtype=np.float64)
        flat = arr.ravel()
        shape = list(arr.shape)
        self.init_names_f64.append(name)
        self.init_data_f64.append(flat.tolist())
        self.init_shapes_f64.append(shape)

    def add_init_i64(self, name: str, data):
        arr = np.asarray(data, dtype=np.int64)
        flat = arr.ravel()
        shape = list(arr.shape)
        self.init_names_i64.append(name)
        self.init_data_i64.append(flat.tolist())
        self.init_shapes_i64.append(shape)

    def slice_col(self, input_name: str, col_idx: int, out: str):
        """Slice a single column from the input tensor -> (batch, 1)."""
        s = self.uid("s")
        e = self.uid("e")
        a = self.uid("a")
        self.add_init_i64(s, np.array([col_idx]))
        self.add_init_i64(e, np.array([col_idx + 1]))
        self.add_init_i64(a, np.array([1]))
        self.add_node("Slice", [input_name, s, e, a], [out])


def _pwl_nodes(g: _GraphAccumulator, input_name: str, x_grid: np.ndarray,
               effects: np.ndarray, output_name: str):
    """Add piecewise-linear interpolation nodes to *g*.

    Given equally-spaced (x_grid, effects), builds nodes that:
      1. Clip x to [x_min, x_max]
      2. Scale to [0, n_segments]
      3. Floor -> segment index
      4. Gather left / right values
      5. Linear interpolation
    Result is shape (batch, 1).
    """
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    n_seg = len(x_grid) - 1
    step = (x_max - x_min) / n_seg if n_seg > 0 else 1.0

    mn = g.uid("xmin")
    mx = g.uid("xmax")
    g.add_init_f64(mn, np.array([x_min]))
    g.add_init_f64(mx, np.array([x_max]))
    cl = g.uid("clip")
    g.add_node("Clip", [input_name, mn, mx], [cl])

    sub = g.uid("sub")
    g.add_node("Sub", [cl, mn], [sub])
    st = g.uid("step")
    g.add_init_f64(st, np.array([step]))
    sc = g.uid("sc")
    g.add_node("Div", [sub, st], [sc])

    fl = g.uid("fl")
    g.add_node("Floor", [sc], [fl])
    smx = g.uid("smx")
    g.add_init_f64(smx, np.array([float(n_seg - 1)]))
    z = g.uid("z")
    g.add_init_f64(z, np.array([0.0]))
    ic = g.uid("ic")
    g.add_node("Clip", [fl, z, smx], [ic])

    ii = g.uid("ii")
    g.add_node("Cast", [ic], [ii], [("to", "int", 7)])  # INT64

    # Flatten + Squeeze to 1-D
    ft = g.uid("ft")
    g.add_node("Flatten", [ii], [ft], [("axis", "int", 1)])
    sq_ax = g.uid("sqax")
    g.add_init_i64(sq_ax, np.array([1]))
    i1 = g.uid("i1")
    g.add_node("Squeeze", [ft, sq_ax], [i1])

    y_name = g.uid("yv")
    g.add_init_f64(y_name, np.asarray(effects, dtype=np.float64))
    yl = g.uid("yl")
    g.add_node("Gather", [y_name, i1], [yl], [("axis", "int", 0)])

    one = g.uid("one")
    g.add_init_i64(one, np.array([1]))
    ip1 = g.uid("ip1")
    g.add_node("Add", [i1, one], [ip1])
    yr = g.uid("yr")
    g.add_node("Gather", [y_name, ip1], [yr], [("axis", "int", 0)])

    # Frac part
    ifl = g.uid("ifl")
    g.add_node("Cast", [i1], [ifl], [("to", "int", 11)])  # DOUBLE
    sf = g.uid("sf")
    g.add_node("Flatten", [sc], [sf], [("axis", "int", 1)])
    sq2 = g.uid("sq2")
    g.add_init_i64(sq2, np.array([1]))
    s1 = g.uid("s1")
    g.add_node("Squeeze", [sf, sq2], [s1])
    fr = g.uid("fr")
    g.add_node("Sub", [s1, ifl], [fr])

    # Interpolate: y_left + frac * (y_right - y_left)
    df = g.uid("df")
    g.add_node("Sub", [yr, yl], [df])
    pr = g.uid("pr")
    g.add_node("Mul", [fr, df], [pr])
    r1 = g.uid("r1")
    g.add_node("Add", [yl, pr], [r1])

    # Unsqueeze back to (batch, 1)
    uax = g.uid("uax")
    g.add_init_i64(uax, np.array([1]))
    g.add_node("Unsqueeze", [r1, uax], [output_name])


def _build_full_model(model: "GLMModel", n_grid_points: int = 200) -> bytes:
    """Build a Level-2 ONNX model with full preprocessing.

    The graph takes raw feature values as a single float64 tensor
    ``(batch, n_raw_vars)`` and produces predictions.
    Categorical variables are passed as integer codes.
    """
    from rustystats._rustystats import serialize_onnx_graph_py

    g = _GraphAccumulator()
    feature_names = model.feature_names
    params = np.asarray(model.params, dtype=np.float64)

    # ── Classify every design-matrix column ──
    features = []
    for i, (nm, coef) in enumerate(zip(feature_names, params)):
        info = _classify_feature(nm)
        info.update(index=i, coef=float(coef), name=nm)
        features.append(info)

    # Group related columns
    spline_groups: Dict[str, List[dict]] = defaultdict(list)
    cat_groups: Dict[str, Dict[str, dict]] = defaultdict(dict)
    intercept_coef = 0.0
    input_vars: OrderedDict = OrderedDict()  # var -> {"type": ...}
    col_outputs: List[str] = []  # tensor names contributing to eta

    def reg_cont(v):
        input_vars.setdefault(v, {"type": "continuous"})

    def reg_cat(v):
        if v not in input_vars:
            input_vars[v] = {"type": "categorical"}

    # First pass: group and register
    for feat in features:
        ft = feat['type']
        if ft == 'intercept':
            intercept_coef = feat['coef']
        elif ft == 'linear':
            reg_cont(feat['variable'])
        elif ft == 'categorical':
            cat_groups[feat['variable']][feat['level']] = feat
            reg_cat(feat['variable'])
        elif ft == 'spline':
            spline_groups[feat['variable']].append(feat)
            reg_cont(feat['variable'])
        elif ft == 'te':
            reg_cat(feat['variable'])
        elif ft == 'fe':
            reg_cat(feat['variable'])
        elif ft == 'constraint':
            reg_cont(feat['variable'])

    var_map = list(input_vars.keys())
    input_name = "input"

    # ── Build graph nodes for each term type ──

    # Linear terms
    for feat in features:
        if feat['type'] not in ('linear', 'constraint'):
            continue
        var = feat['variable']
        col_idx = var_map.index(var)
        coef = feat['coef']
        sl = g.uid("sl")
        g.slice_col(input_name, col_idx, sl)
        cn = g.uid("c")
        g.add_init_f64(cn, np.array([coef]))
        out = g.uid("lin")
        g.add_node("Mul", [sl, cn], [out])
        col_outputs.append(out)

    # Categorical terms (grouped by variable)
    for var, level_map in cat_groups.items():
        col_idx = var_map.index(var)
        levels = _get_all_levels(model, var)
        sl = g.uid("cat")
        g.slice_col(input_name, col_idx, sl)
        # Reference level is index 0; dummies for 1..n_levels-1
        for li in range(1, len(levels)):
            level_str = levels[li]
            feat = level_map.get(level_str)
            if feat is None:
                continue
            coef = feat['coef']
            idx_c = g.uid("li")
            g.add_init_f64(idx_c, np.array([float(li)]))
            eq = g.uid("eq")
            g.add_node("Equal", [sl, idx_c], [eq])
            ca = g.uid("ca")
            g.add_node("Cast", [eq], [ca], [("to", "int", 11)])  # DOUBLE
            cn = g.uid("cc")
            g.add_init_f64(cn, np.array([coef]))
            out = g.uid("cp")
            g.add_node("Mul", [ca, cn], [out])
            col_outputs.append(out)

    # Spline terms (grouped by variable -> piecewise linear)
    for var, feats in spline_groups.items():
        col_idx = var_map.index(var)
        sl = g.uid("spl")
        g.slice_col(input_name, col_idx, sl)

        fitted_splines = _get_builder_attr(model, '_fitted_splines', {})
        spline = fitted_splines.get(var)
        if spline is not None:
            info = spline.get_knot_info()
            bk = info.get('boundary_knots', [0.0, 1.0])
            x_grid = np.linspace(float(bk[0]), float(bk[1]), n_grid_points)
            basis, _ = spline.transform(x_grid)
            feats_sorted = sorted(feats, key=lambda f: f['basis_idx'])
            coefs = np.array([f['coef'] for f in feats_sorted])
            effects = basis @ coefs
            out = g.uid("seff")
            _pwl_nodes(g, sl, x_grid, effects, out)
            col_outputs.append(out)
        else:
            # Fallback: treat as linear
            for feat in feats:
                cn = g.uid("sc")
                g.add_init_f64(cn, np.array([feat['coef']]))
                out = g.uid("sp")
                g.add_node("Mul", [sl, cn], [out])
                col_outputs.append(out)

    # Target-encoding terms
    for feat in features:
        if feat['type'] != 'te':
            continue
        var = feat['variable']
        col_idx = var_map.index(var)
        coef = feat['coef']

        sl = g.uid("te")
        g.slice_col(input_name, col_idx, sl)

        te_all = _get_builder_attr(model, '_te_stats', {})
        te_info = te_all.get(var, {})
        prior = te_info.get('prior', 0.0)
        prior_weight = te_info.get('prior_weight', 1.0)
        level_stats = te_info.get('stats', {})
        exposure_weighted = te_info.get('used_exposure_weighted', False)
        levels = _get_all_levels(model, var)
        if not levels:
            levels = sorted(level_stats.keys())

        lookup = np.full(len(levels), prior, dtype=np.float64)
        for li, lv in enumerate(levels):
            tup = level_stats.get(lv)
            if tup is not None:
                if exposure_weighted:
                    sc_, se_ = tup
                    lookup[li] = (sc_ + prior * prior_weight) / (se_ + prior_weight)
                else:
                    st_, cnt_ = tup
                    lookup[li] = (st_ + prior * prior_weight) / (cnt_ + prior_weight)

        lut = g.uid("telut")
        g.add_init_f64(lut, lookup)

        ii = g.uid("tei")
        g.add_node("Cast", [sl], [ii], [("to", "int", 7)])
        ft_ = g.uid("tef")
        g.add_node("Flatten", [ii], [ft_], [("axis", "int", 1)])
        sq_ax = g.uid("sqax")
        g.add_init_i64(sq_ax, np.array([1]))
        sq = g.uid("tesq")
        g.add_node("Squeeze", [ft_, sq_ax], [sq])
        ga = g.uid("teg")
        g.add_node("Gather", [lut, sq], [ga], [("axis", "int", 0)])
        uax = g.uid("uax")
        g.add_init_i64(uax, np.array([1]))
        us = g.uid("teu")
        g.add_node("Unsqueeze", [ga, uax], [us])

        cn = g.uid("tec")
        g.add_init_f64(cn, np.array([coef]))
        out = g.uid("tep")
        g.add_node("Mul", [us, cn], [out])
        col_outputs.append(out)

    # Frequency-encoding terms
    for feat in features:
        if feat['type'] != 'fe':
            continue
        var = feat['variable']
        col_idx = var_map.index(var)
        coef = feat['coef']

        sl = g.uid("fe")
        g.slice_col(input_name, col_idx, sl)

        fe_all = _get_builder_attr(model, '_fe_stats', {})
        fe_info = fe_all.get(var, {})
        level_counts = fe_info.get('level_counts', {})
        max_count = fe_info.get('max_count', 1)
        levels = _get_all_levels(model, var)
        if not levels:
            levels = sorted(level_counts.keys())

        lookup = np.zeros(len(levels), dtype=np.float64)
        for li, lv in enumerate(levels):
            cnt = level_counts.get(lv, 0)
            lookup[li] = cnt / max_count if max_count > 0 else 0.0

        lut = g.uid("felut")
        g.add_init_f64(lut, lookup)

        ii = g.uid("fei")
        g.add_node("Cast", [sl], [ii], [("to", "int", 7)])
        ft_ = g.uid("fef")
        g.add_node("Flatten", [ii], [ft_], [("axis", "int", 1)])
        sq_ax = g.uid("sqax")
        g.add_init_i64(sq_ax, np.array([1]))
        sq = g.uid("fesq")
        g.add_node("Squeeze", [ft_, sq_ax], [sq])
        ga = g.uid("feg")
        g.add_node("Gather", [lut, sq], [ga], [("axis", "int", 0)])
        uax = g.uid("uax")
        g.add_init_i64(uax, np.array([1]))
        us = g.uid("feu")
        g.add_node("Unsqueeze", [ga, uax], [us])

        cn = g.uid("fec")
        g.add_init_f64(cn, np.array([coef]))
        out = g.uid("fep")
        g.add_node("Mul", [us, cn], [out])
        col_outputs.append(out)

    # ── Sum all contributions ──
    if not col_outputs:
        zn = g.uid("zero")
        g.add_init_f64(zn, np.array([[0.0]]))
        sum_name = zn
    elif len(col_outputs) == 1:
        sum_name = col_outputs[0]
    else:
        current = col_outputs[0]
        for idx, nxt in enumerate(col_outputs[1:]):
            out = g.uid("sum") if idx == len(col_outputs) - 2 else g.uid("ps")
            g.add_node("Add", [current, nxt], [out])
            current = out
        sum_name = current

    # Add intercept
    bn = g.uid("intercept")
    g.add_init_f64(bn, np.array([intercept_coef]))
    eta = g.uid("eta")
    g.add_node("Add", [sum_name, bn], [eta])

    # Inverse link
    output_name = "prediction"
    link = model.link
    if link == 'log':
        g.add_node("Exp", [eta], [output_name])
    elif link == 'logit':
        g.add_node("Sigmoid", [eta], [output_name])
    elif link == 'identity':
        g.add_node("Identity", [eta], [output_name])
    elif link == 'inverse':
        g.add_node("Reciprocal", [eta], [output_name])
    elif link == 'sqrt':
        g.add_node("Mul", [eta, eta], [output_name])
    else:
        g.add_node("Exp", [eta], [output_name])

    # ── Metadata ──
    cat_level_maps = {}
    for var, info in input_vars.items():
        if info['type'] == 'categorical':
            levels = _get_all_levels(model, var)
            if levels:
                cat_level_maps[var] = levels

    meta_keys = [
        'input_names',
        'input_types',
        'inverse_link',
        'family',
    ]
    meta_values = [
        json.dumps(list(input_vars.keys())),
        json.dumps({v: d['type'] for v, d in input_vars.items()}),
        _inverse_link_name(model.link),
        model.family,
    ]
    if cat_level_maps:
        meta_keys.append('cat_level_maps')
        meta_values.append(json.dumps(cat_level_maps))

    doc = f"RustyStats GLM: {model.family}, link={model.link}"

    # DT_DOUBLE = 11
    onnx_bytes = serialize_onnx_graph_py(
        g.node_ops, g.node_inputs, g.node_outputs,
        g.node_attr_names, g.node_attr_types, g.node_attr_ints, g.node_attr_floats,
        g.init_names_f64, g.init_data_f64, g.init_shapes_f64,
        g.init_names_i64, g.init_data_i64, g.init_shapes_i64,
        [input_name], [11], [[-1, len(var_map)]],
        [output_name], [11], [[-1, 1]],
        8, 18, "RustyStats", doc,
        meta_keys, meta_values,
    )
    return bytes(onnx_bytes)


# ── Public API ───────────────────────────────────────────────────────────────

def to_onnx(
    model: "GLMModel",
    path: Optional[str] = None,
    n_grid_points: int = 200,
    mode: str = "scoring",
) -> bytes:
    """Export a fitted GLMModel to ONNX format.

    No external dependencies are required -- protobuf serialization is
    handled by the Rust backend.

    Parameters
    ----------
    model : GLMModel
        A fitted RustyStats GLM model.
    path : str, optional
        If given, write the ONNX model to this file path.
    n_grid_points : int, default 200
        Grid resolution for piecewise-linear spline approximation
        (only used in ``"full"`` mode).
    mode : {"scoring", "full"}, default "scoring"
        * ``"scoring"`` -- The ONNX model takes a pre-built design
          matrix ``X`` of shape ``(batch, n_features)`` (without
          intercept column) and produces predictions.
        * ``"full"`` -- The ONNX model takes raw feature values and
          embeds all preprocessing inside the graph.

    Returns
    -------
    bytes
        Raw ONNX protobuf bytes.  Can be loaded with
        ``onnxruntime.InferenceSession(onnx_bytes)`` or written to disk.
    """
    if mode == "full":
        onnx_bytes = _build_full_model(model, n_grid_points=n_grid_points)
    else:
        onnx_bytes = _build_scoring_model(model)

    if path is not None:
        with open(path, 'wb') as f:
            f.write(onnx_bytes)

    return onnx_bytes
