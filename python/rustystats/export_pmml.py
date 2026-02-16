"""
PMML 4.4 export for RustyStats GLM models.

Exports fitted GLMModel instances as PMML GeneralRegressionModel XML.
Supports linear, categorical, spline, target encoding, frequency encoding,
expression, constraint, and interaction terms.

Spline basis functions are collapsed to piecewise-linear NormContinuous
derived fields evaluated on a configurable grid.
"""

from __future__ import annotations

import re
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rustystats.formula import GLMModel


# ── PMML family / link mapping ──────────────────────────────────────────────

_FAMILY_MAP = {
    "gaussian": "normal",
    "poisson": "poisson",
    "binomial": "binomial",
    "gamma": "gamma",
    "tweedie": "tweedie",
    "negativebinomial": "negbin",
    "quasipoisson": "poisson",
    "quasibinomial": "binomial",
    "inversegaussian": "igauss",
}

_LINK_MAP = {
    "identity": "identity",
    "log": "log",
    "logit": "logit",
    "inverse": "power",
    "sqrt": "power",
    "cloglog": "cloglog",
    "probit": "probit",
}


# ── Feature-name parsing ────────────────────────────────────────────────────

_RE_CATEGORICAL = re.compile(r'^(.+?)\[T\.(.+)\]$')
_RE_SPLINE = re.compile(r'^(bs|ns|ms)\((.+?),\s*(\d+)/(\d+)(?:,\s*(.+))?\)$')
_RE_TE = re.compile(r'^TE\((.+)\)$')
_RE_FE = re.compile(r'^FE\((.+)\)$')
_RE_EXPRESSION = re.compile(r'^I\((.+)\)$')
_RE_POS = re.compile(r'^pos\((.+)\)$')
_RE_NEG = re.compile(r'^neg\((.+)\)$')
_RE_POWER = re.compile(r'^(\w+)\s*\*\*\s*(\d+(?:\.\d+)?)$')


def _split_interaction(name: str) -> List[str]:
    """Split feature name on ':' while respecting parentheses / brackets."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in name:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ':' and depth == 0:
            parts.append(''.join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _classify(name: str) -> dict:
    """Classify a single feature-name component (not an interaction)."""
    if name == 'Intercept':
        return {'type': 'intercept'}

    m = _RE_CATEGORICAL.match(name)
    if m:
        return {'type': 'categorical', 'variable': m.group(1), 'level': m.group(2)}

    m = _RE_SPLINE.match(name)
    if m:
        return {
            'type': 'spline', 'spline_type': m.group(1),
            'variable': m.group(2),
            'basis_idx': int(m.group(3)), 'basis_total': int(m.group(4)),
            'flags': m.group(5),
        }

    m = _RE_TE.match(name)
    if m:
        return {'type': 'te', 'variable': m.group(1)}

    m = _RE_FE.match(name)
    if m:
        return {'type': 'fe', 'variable': m.group(1)}

    m = _RE_EXPRESSION.match(name)
    if m:
        return {'type': 'expression', 'expr': m.group(1)}

    m = _RE_POS.match(name)
    if m:
        return {'type': 'constraint', 'variable': m.group(1), 'sign': 'pos'}

    m = _RE_NEG.match(name)
    if m:
        return {'type': 'constraint', 'variable': m.group(1), 'sign': 'neg'}

    return {'type': 'linear', 'variable': name}


def _classify_feature(name: str) -> dict:
    """Classify a full feature name, including interactions."""
    components = _split_interaction(name)
    if len(components) > 1:
        return {
            'type': 'interaction',
            'components': [{**_classify(c), 'name': c} for c in components],
        }
    return _classify(name)


# ── PMML Exporter ───────────────────────────────────────────────────────────

class PMMLExporter:
    """Build a PMML 4.4 GeneralRegressionModel from a fitted ``GLMModel``."""

    PMML_VERSION = "4.4"
    PMML_NS = "http://www.dmg.org/PMML-4_4"

    def __init__(self, model: "GLMModel", n_grid_points: int = 200):
        self.model = model
        self.n_grid = n_grid_points
        self._pcnt = 0  # parameter name counter

        # Accumulated PMML structures
        self._raw_inputs: OrderedDict[str, dict] = OrderedDict()
        self._derived_fields: OrderedDict[str, ET.Element] = OrderedDict()
        self._parameters: List[Tuple[str, str, float]] = []       # (pname, label, beta)
        self._factors: OrderedDict[str, bool] = OrderedDict()
        self._covariates: OrderedDict[str, bool] = OrderedDict()
        self._pp_cells: List[Tuple[str, str, str]] = []           # (pname, predictor, value)

        self._analyze()

    # ── helpers ──────────────────────────────────────────────────────────

    def _pname(self) -> str:
        name = f"p{self._pcnt}"
        self._pcnt += 1
        return name

    def _response_name(self) -> str:
        formula = getattr(self.model, 'formula', '') or ''
        if '~' in formula:
            return formula.split('~')[0].strip()
        return 'y'

    def _get_all_levels(self, var: str) -> List[str]:
        builder = getattr(self.model, '_builder', None)
        if builder is None:
            return []
        cache = getattr(builder, '_cat_encoding_cache', {})
        cached = cache.get(f"{var}_True")
        if cached is not None:
            return list(cached.levels)
        return []

    def _get_builder_attr(self, attr: str, default=None):
        builder = getattr(self.model, '_builder', None)
        if builder is None:
            return default
        return getattr(builder, attr, default)

    # ── analysis pass ────────────────────────────────────────────────────

    def _analyze(self):
        names = self.model.feature_names
        params = self.model.params

        features = []
        for i, (nm, coef) in enumerate(zip(names, params)):
            info = _classify_feature(nm)
            info.update(index=i, coef=float(coef), name=nm)
            features.append(info)

        # Collect groups before processing
        spline_groups: Dict[str, List[dict]] = defaultdict(list)
        cat_groups: Dict[str, Dict[str, dict]] = defaultdict(dict)
        interaction_spline_groups: Dict[str, List[dict]] = defaultdict(list)
        interactions_simple: List[dict] = []

        for feat in features:
            ft = feat['type']
            if ft == 'intercept':
                self._emit_intercept(feat)
            elif ft == 'linear':
                self._emit_linear(feat)
            elif ft == 'categorical':
                cat_groups[feat['variable']][feat['level']] = feat
            elif ft == 'spline':
                spline_groups[feat['variable']].append(feat)
            elif ft == 'te':
                self._emit_te(feat)
            elif ft == 'fe':
                self._emit_fe(feat)
            elif ft == 'expression':
                self._emit_expression(feat)
            elif ft == 'constraint':
                self._emit_linear(feat)
            elif ft == 'interaction':
                comps = feat['components']
                has_spline = any(c['type'] == 'spline' for c in comps)
                if has_spline:
                    # Group spline interactions by (non-spline key, spline var)
                    key = self._spline_interaction_key(feat)
                    interaction_spline_groups[key].append(feat)
                else:
                    interactions_simple.append(feat)

        # Process grouped terms
        for var, level_map in cat_groups.items():
            self._emit_categorical(var, level_map)

        for var, feats in spline_groups.items():
            self._emit_spline_group(var, feats)

        for feat in interactions_simple:
            self._emit_interaction(feat)

        for key, feats in interaction_spline_groups.items():
            self._emit_spline_interaction_group(key, feats)

    # ── spline interaction grouping key ──────────────────────────────────

    @staticmethod
    def _spline_interaction_key(feat: dict) -> str:
        """Build a grouping key for spline-interaction features.

        Groups features that share the same non-spline components and the same
        spline variable so their basis coefficients can be collapsed together.
        """
        non_spline = []
        spline_var = None
        for c in feat['components']:
            if c['type'] == 'spline':
                spline_var = c['variable']
            else:
                non_spline.append(c['name'])
        return '|'.join(non_spline) + '||' + (spline_var or '')

    # ── emitters: single terms ───────────────────────────────────────────

    def _emit_intercept(self, feat: dict):
        pn = self._pname()
        self._parameters.append((pn, 'Intercept', feat['coef']))

    def _emit_linear(self, feat: dict):
        var = feat.get('variable', feat['name'])
        pn = self._pname()
        self._parameters.append((pn, feat['name'], feat['coef']))
        self._raw_inputs.setdefault(var, {'optype': 'continuous', 'dataType': 'double'})
        self._covariates[var] = True
        self._pp_cells.append((pn, var, '1'))

    def _emit_categorical(self, var: str, level_map: Dict[str, dict]):
        all_levels = self._get_all_levels(var)
        info = {'optype': 'categorical', 'dataType': 'string'}
        if all_levels:
            info['levels'] = all_levels
        self._raw_inputs[var] = info
        self._factors[var] = True

        for level, feat in level_map.items():
            pn = self._pname()
            self._parameters.append((pn, feat['name'], feat['coef']))
            self._pp_cells.append((pn, var, level))

    def _emit_spline_group(self, var: str, feats: List[dict]):
        """Collapse spline basis columns into a piecewise-linear derived field."""
        fitted_splines = self._get_builder_attr('_fitted_splines', {})
        spline = fitted_splines.get(var)
        if spline is None:
            for f in feats:
                self._emit_linear(f)
            return

        info = spline.get_knot_info()
        bk = info.get('boundary_knots', [0.0, 1.0])
        x_grid = np.linspace(float(bk[0]), float(bk[1]), self.n_grid)

        basis, _ = spline.transform(x_grid)
        feats_sorted = sorted(feats, key=lambda f: f['basis_idx'])
        coefs = np.array([f['coef'] for f in feats_sorted])
        effects = basis @ coefs

        derived = f"spline_effect_{var}"
        self._add_norm_continuous(derived, var, x_grid, effects)

        pn = self._pname()
        self._parameters.append((pn, f"spline({var})", 1.0))
        self._covariates[derived] = True
        self._pp_cells.append((pn, derived, '1'))
        self._raw_inputs.setdefault(var, {'optype': 'continuous', 'dataType': 'double'})

    def _emit_te(self, feat: dict):
        var = feat['variable']
        te_all = self._get_builder_attr('_te_stats', {})
        te_info = te_all.get(var, {})

        prior = te_info.get('prior', 0.0)
        prior_weight = te_info.get('prior_weight', 1.0)
        level_stats = te_info.get('stats', {})
        exposure_weighted = te_info.get('used_exposure_weighted', False)

        mapping: Dict[str, float] = {}
        for level, tup in level_stats.items():
            if exposure_weighted:
                sum_claims, sum_exposure = tup
                mapping[level] = (sum_claims + prior * prior_weight) / (sum_exposure + prior_weight)
            else:
                sum_target, count = tup
                mapping[level] = (sum_target + prior * prior_weight) / (count + prior_weight)

        derived = f"TE_{var}"
        self._add_map_values(derived, var, mapping, default=prior)
        self._raw_inputs.setdefault(var, {'optype': 'categorical', 'dataType': 'string'})

        pn = self._pname()
        self._parameters.append((pn, feat['name'], feat['coef']))
        self._covariates[derived] = True
        self._pp_cells.append((pn, derived, '1'))

    def _emit_fe(self, feat: dict):
        var = feat['variable']
        fe_all = self._get_builder_attr('_fe_stats', {})
        fe_info = fe_all.get(var, {})

        level_counts = fe_info.get('level_counts', {})
        max_count = fe_info.get('max_count', 1)

        mapping: Dict[str, float] = {}
        for level, cnt in level_counts.items():
            mapping[level] = cnt / max_count if max_count > 0 else 0.0

        derived = f"FE_{var}"
        self._add_map_values(derived, var, mapping, default=0.0)
        self._raw_inputs.setdefault(var, {'optype': 'categorical', 'dataType': 'string'})

        pn = self._pname()
        self._parameters.append((pn, feat['name'], feat['coef']))
        self._covariates[derived] = True
        self._pp_cells.append((pn, derived, '1'))

    def _emit_expression(self, feat: dict):
        expr = feat['expr']
        derived = feat['name']

        m = _RE_POWER.match(expr)
        if m:
            var, power = m.group(1), m.group(2)
            self._add_power_derived(derived, var, power)
            self._raw_inputs.setdefault(var, {'optype': 'continuous', 'dataType': 'double'})
        else:
            tokens = re.findall(r'\b([a-zA-Z_]\w*)\b', expr)
            for v in tokens:
                self._raw_inputs.setdefault(v, {'optype': 'continuous', 'dataType': 'double'})
            self._add_extension_derived(derived, expr)

        pn = self._pname()
        self._parameters.append((pn, feat['name'], feat['coef']))
        self._covariates[derived] = True
        self._pp_cells.append((pn, derived, '1'))

    # ── emitters: interactions ───────────────────────────────────────────

    def _emit_interaction(self, feat: dict):
        comps = feat['components']
        has_cat = any(c['type'] == 'categorical' for c in comps)
        has_linear = any(c['type'] in ('linear', 'constraint') for c in comps)
        has_te = any(c['type'] == 'te' for c in comps)
        has_fe = any(c['type'] == 'fe' for c in comps)

        pn = self._pname()
        self._parameters.append((pn, feat['name'], feat['coef']))

        for comp in comps:
            ct = comp['type']
            if ct == 'categorical':
                var, level = comp['variable'], comp['level']
                self._pp_cells.append((pn, var, level))
                self._factors.setdefault(var, True)
                if var not in self._raw_inputs:
                    lvls = self._get_all_levels(var)
                    info = {'optype': 'categorical', 'dataType': 'string'}
                    if lvls:
                        info['levels'] = lvls
                    self._raw_inputs[var] = info

            elif ct in ('linear', 'constraint'):
                var = comp.get('variable', comp['name'])
                self._pp_cells.append((pn, var, '1'))
                self._covariates.setdefault(var, True)
                self._raw_inputs.setdefault(var, {'optype': 'continuous', 'dataType': 'double'})

            elif ct == 'te':
                derived = f"TE_{comp['variable']}"
                self._pp_cells.append((pn, derived, '1'))
                self._covariates.setdefault(derived, True)

            elif ct == 'fe':
                derived = f"FE_{comp['variable']}"
                self._pp_cells.append((pn, derived, '1'))
                self._covariates.setdefault(derived, True)

            elif ct == 'expression':
                derived = comp['name']
                self._pp_cells.append((pn, derived, '1'))
                self._covariates.setdefault(derived, True)

    def _emit_spline_interaction_group(self, key: str, feats: List[dict]):
        """Collapse spline basis columns for a specific interaction group.

        For example, Region[T.B]:bs(VehAge, 1/5) ... Region[T.B]:bs(VehAge, 5/5)
        gets collapsed into a single piecewise-linear derived field
        ``int_spline_VehAge_Region_B`` and gated by Region=B in PPMatrix.
        """
        # Parse the key to recover non-spline components and spline variable
        parts = key.split('||')
        non_spline_key = parts[0]
        spline_var = parts[1] if len(parts) > 1 else ''

        fitted_splines = self._get_builder_attr('_fitted_splines', {})
        spline = fitted_splines.get(spline_var)

        if spline is None:
            # Fallback: emit each feature as individual terms
            for feat in feats:
                self._emit_interaction(feat)
            return

        info = spline.get_knot_info()
        bk = info.get('boundary_knots', [0.0, 1.0])
        x_grid = np.linspace(float(bk[0]), float(bk[1]), self.n_grid)
        basis, _ = spline.transform(x_grid)

        # Get the spline basis coefficients sorted by basis index
        spline_feats = []
        for feat in feats:
            for c in feat['components']:
                if c['type'] == 'spline':
                    spline_feats.append((c['basis_idx'], feat['coef']))
                    break
        spline_feats.sort(key=lambda x: x[0])
        coefs = np.array([c for _, c in spline_feats])
        effects = basis @ coefs

        # Create a unique derived-field name
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', non_spline_key)
        derived = f"int_spline_{spline_var}_{safe_key}"
        self._add_norm_continuous(derived, spline_var, x_grid, effects)

        # Now emit one parameter gated by the non-spline components + covariate
        pn = self._pname()
        self._parameters.append((pn, f"{non_spline_key}:spline({spline_var})", 1.0))
        self._covariates[derived] = True
        self._pp_cells.append((pn, derived, '1'))
        self._raw_inputs.setdefault(spline_var, {'optype': 'continuous', 'dataType': 'double'})

        # Gate by non-spline components (categorical levels, continuous vars)
        sample = feats[0]
        for comp in sample['components']:
            ct = comp['type']
            if ct == 'spline':
                continue
            if ct == 'categorical':
                var, level = comp['variable'], comp['level']
                self._pp_cells.append((pn, var, level))
                self._factors.setdefault(var, True)
                if var not in self._raw_inputs:
                    lvls = self._get_all_levels(var)
                    raw = {'optype': 'categorical', 'dataType': 'string'}
                    if lvls:
                        raw['levels'] = lvls
                    self._raw_inputs[var] = raw
            elif ct in ('linear', 'constraint'):
                var = comp.get('variable', comp['name'])
                self._pp_cells.append((pn, var, '1'))
                self._covariates.setdefault(var, True)
                self._raw_inputs.setdefault(var, {'optype': 'continuous', 'dataType': 'double'})

    # ── derived-field builders ───────────────────────────────────────────

    def _add_norm_continuous(self, name: str, source: str,
                             x_grid: np.ndarray, effects: np.ndarray):
        df = ET.Element('DerivedField', {
            'name': name, 'optype': 'continuous', 'dataType': 'double',
        })
        nc = ET.SubElement(df, 'NormContinuous', {
            'field': source, 'outliers': 'asExtremeValues',
        })
        for x, e in zip(x_grid, effects):
            ET.SubElement(nc, 'LinearNorm', {
                'orig': f'{float(x):.10g}', 'norm': f'{float(e):.10g}',
            })
        self._derived_fields[name] = df

    def _add_map_values(self, name: str, source: str,
                        mapping: Dict[str, float], default: float = 0.0):
        df = ET.Element('DerivedField', {
            'name': name, 'optype': 'continuous', 'dataType': 'double',
        })
        mv = ET.SubElement(df, 'MapValues', {
            'outputColumn': 'encoded', 'defaultValue': f'{default:.10g}',
        })
        ET.SubElement(mv, 'FieldColumnPair', {
            'field': source, 'column': 'original',
        })
        tbl = ET.SubElement(mv, 'InlineTable')
        for level in sorted(mapping):
            row = ET.SubElement(tbl, 'row')
            o = ET.SubElement(row, 'original')
            o.text = str(level)
            e = ET.SubElement(row, 'encoded')
            e.text = f'{mapping[level]:.10g}'
        self._derived_fields[name] = df

    def _add_power_derived(self, name: str, var: str, power: str):
        df = ET.Element('DerivedField', {
            'name': name, 'optype': 'continuous', 'dataType': 'double',
        })
        ap = ET.SubElement(df, 'Apply', {'function': 'pow'})
        ET.SubElement(ap, 'FieldRef', {'field': var})
        ET.SubElement(ap, 'Constant', {'dataType': 'double'}).text = str(power)
        self._derived_fields[name] = df

    def _add_extension_derived(self, name: str, expr: str):
        df = ET.Element('DerivedField', {
            'name': name, 'optype': 'continuous', 'dataType': 'double',
        })
        ET.SubElement(df, 'Extension', {
            'name': 'rustystats_expression', 'value': expr,
        })
        self._derived_fields[name] = df

    # ── XML generation ───────────────────────────────────────────────────

    def export(self) -> str:
        """Return the PMML XML document as a string."""
        root = ET.Element('PMML', {
            'version': self.PMML_VERSION,
            'xmlns': self.PMML_NS,
        })

        self._xml_header(root)
        self._xml_data_dictionary(root)
        if self._derived_fields:
            self._xml_transformation_dictionary(root)
        self._xml_grm(root)

        rough = ET.tostring(root, encoding='unicode', xml_declaration=False)
        dom = minidom.parseString(rough)
        xml_str = dom.toprettyxml(indent='  ')
        # Replace the minidom declaration with a clean one
        lines = xml_str.split('\n')
        if lines and lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
        return '\n'.join(line for line in lines if line.strip())

    def _xml_header(self, root: ET.Element):
        hdr = ET.SubElement(root, 'Header', {
            'copyright': 'RustyStats',
            'description': f'GLM ({self.model.family}, link={self.model.link})',
        })
        ET.SubElement(hdr, 'Application', {'name': 'RustyStats'})
        ts = ET.SubElement(hdr, 'Timestamp')
        ts.text = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    def _xml_data_dictionary(self, root: ET.Element):
        resp = self._response_name()
        n_fields = len(self._raw_inputs) + 1
        dd = ET.SubElement(root, 'DataDictionary', {'numberOfFields': str(n_fields)})
        ET.SubElement(dd, 'DataField', {
            'name': resp, 'optype': 'continuous', 'dataType': 'double',
        })
        for var, info in self._raw_inputs.items():
            attrs = {'name': var, 'optype': info['optype'], 'dataType': info['dataType']}
            elem = ET.SubElement(dd, 'DataField', attrs)
            for lv in info.get('levels', []):
                ET.SubElement(elem, 'Value', {'value': str(lv)})

    def _xml_transformation_dictionary(self, root: ET.Element):
        td = ET.SubElement(root, 'TransformationDictionary')
        for elem in self._derived_fields.values():
            td.append(elem)

    def _xml_grm(self, root: ET.Element):
        family_key = self.model.family.lower().split('(')[0].strip()
        pmml_dist = _FAMILY_MAP.get(family_key, 'normal')
        pmml_link = _LINK_MAP.get(self.model.link, 'log')

        attrs = {
            'modelType': 'generalizedLinear',
            'functionName': 'regression',
            'algorithmName': 'IRLS',
            'distributionName': pmml_dist,
            'linkFunction': pmml_link,
            'modelName': 'RustyStats_GLM',
        }

        if pmml_link == 'power':
            if self.model.link == 'inverse':
                attrs['linkParameter'] = '-1'
            elif self.model.link == 'sqrt':
                attrs['linkParameter'] = '0.5'

        offset_spec = getattr(self.model, '_offset_spec', None)
        if isinstance(offset_spec, str):
            attrs['offsetVariable'] = offset_spec

        grm = ET.SubElement(root, 'GeneralRegressionModel', attrs)

        # MiningSchema
        resp = self._response_name()
        ms = ET.SubElement(grm, 'MiningSchema')
        ET.SubElement(ms, 'MiningField', {'name': resp, 'usageType': 'target'})
        for var in self._raw_inputs:
            ET.SubElement(ms, 'MiningField', {'name': var, 'usageType': 'active'})

        # ParameterList
        pl = ET.SubElement(grm, 'ParameterList')
        for pn, label, _ in self._parameters:
            ET.SubElement(pl, 'Parameter', {'name': pn, 'label': label})

        # FactorList
        if self._factors:
            fl = ET.SubElement(grm, 'FactorList')
            for var in self._factors:
                ET.SubElement(fl, 'Predictor', {'name': var})

        # CovariateList
        if self._covariates:
            cl = ET.SubElement(grm, 'CovariateList')
            for var in self._covariates:
                ET.SubElement(cl, 'Predictor', {'name': var})

        # PPMatrix
        ppm = ET.SubElement(grm, 'PPMatrix')
        for pn, pred, val in self._pp_cells:
            ET.SubElement(ppm, 'PPCell', {
                'value': str(val), 'predictorName': pred, 'parameterName': pn,
            })

        # ParamMatrix
        pm = ET.SubElement(grm, 'ParamMatrix')
        for pn, _, beta in self._parameters:
            ET.SubElement(pm, 'PCell', {
                'parameterName': pn, 'beta': f'{beta:.10g}',
            })


# ── Public API ───────────────────────────────────────────────────────────────

def to_pmml(
    model: "GLMModel",
    path: Optional[str] = None,
    n_grid_points: int = 200,
) -> str:
    """Export a fitted GLMModel to PMML 4.4 XML.

    Parameters
    ----------
    model : GLMModel
        A fitted RustyStats GLM model.
    path : str, optional
        If given, write the PMML XML to this file path.
    n_grid_points : int, default 200
        Number of grid points used to approximate spline effect curves
        as piecewise-linear ``NormContinuous`` elements.

    Returns
    -------
    str
        The PMML XML document as a string.
    """
    exporter = PMMLExporter(model, n_grid_points=n_grid_points)
    xml = exporter.export()

    if path is not None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(xml)

    return xml
