"""
Interaction detection for diagnostics.

_InteractionDetector identifies potential factor interactions using
residual-based analysis (post-fit) for DiagnosticsComputer.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    f_cdf_py as _f_cdf,
)

from rustystats.diagnostics.types import InteractionCandidate
from rustystats.diagnostics.utils import discretize, validate_factor_in_data
from rustystats.constants import DEFAULT_MAX_INTERACTION_FACTORS

if TYPE_CHECKING:
    import polars as pl


class _InteractionDetector:
    """Detects potential interactions using residual-based analysis.
    
    Uses pearson residuals from a fitted model to identify factor pairs
    whose combined effect explains residual variance beyond their
    individual effects.
    """
    
    def __init__(
        self,
        pearson_residuals: np.ndarray,
        feature_names: List[str],
    ):
        self.pearson_residuals = pearson_residuals
        self.feature_names = feature_names
    
    def detect_interactions(
        self,
        data: "pl.DataFrame",
        factor_names: List[str],
        max_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
        min_correlation: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
    ) -> List[InteractionCandidate]:
        """Detect potential interactions using greedy residual-based approach."""
        # First, rank factors by residual association
        factor_scores = []
        
        for name in factor_names:
            validate_factor_in_data(name, data)
            
            values = data[name].to_numpy()
            
            # Check if categorical or continuous
            if values.dtype == object or str(values.dtype).startswith('str'):
                score = self._compute_eta_squared(values.astype(str))
            else:
                values = values.astype(np.float64)
                valid_mask = ~np.isnan(values) & ~np.isinf(values)
                if np.sum(valid_mask) < 10:
                    continue
                score = abs(np.corrcoef(values[valid_mask], self.pearson_residuals[valid_mask])[0, 1])
            
            if score >= min_correlation:
                factor_scores.append((name, score))
        
        # Sort and take top factors
        factor_scores.sort(key=lambda x: -x[1])
        top_factors = [name for name, _ in factor_scores[:max_factors]]
        
        if len(top_factors) < 2:
            return []
        
        # Check pairwise interactions
        candidates = []
        
        for i in range(len(top_factors)):
            for j in range(i + 1, len(top_factors)):
                name1, name2 = top_factors[i], top_factors[j]
                
                values1 = data[name1].to_numpy()
                values2 = data[name2].to_numpy()
                
                # Discretize both factors
                bins1 = discretize(values1, 5)
                bins2 = discretize(values2, 5)
                
                # Compute interaction strength
                candidate = self._compute_interaction_strength(
                    name1, bins1, name2, bins2, min_cell_count
                )
                
                if candidate is not None:
                    # Add current_terms and recommendation
                    terms1 = self._get_factor_terms(name1)
                    terms2 = self._get_factor_terms(name2)
                    candidate.current_terms = terms1 + terms2 if (terms1 or terms2) else None
                    
                    candidate.recommendation = self._generate_interaction_recommendation(
                        name1, name2, terms1, terms2, values1, values2
                    )
                    candidates.append(candidate)
        
        # Sort by strength and return top candidates
        candidates.sort(key=lambda x: -x.interaction_strength)
        return candidates[:max_candidates]
    
    def _get_factor_terms(self, name: str) -> List[str]:
        """Get all model terms that include this factor."""
        return [fn for fn in self.feature_names if name in fn]
    
    def _generate_interaction_recommendation(
        self,
        name1: str,
        name2: str,
        terms1: List[str],
        terms2: List[str],
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> str:
        """Generate a recommendation for how to model an interaction."""
        is_cat1 = values1.dtype == object or str(values1.dtype).startswith('str')
        is_cat2 = values2.dtype == object or str(values2.dtype).startswith('str')
        
        has_spline1 = any('bs(' in t or 'ns(' in t or 's(' in t for t in terms1)
        has_spline2 = any('bs(' in t or 'ns(' in t or 's(' in t for t in terms2)
        has_poly1 = any('I(' in t and '**' in t for t in terms1)
        has_poly2 = any('I(' in t and '**' in t for t in terms2)
        
        if is_cat1 and is_cat2:
            return f"Consider C({name1}):C({name2}) interaction term"
        elif is_cat1 and not is_cat2:
            if has_spline2:
                return f"Consider C({name1}):{name2} or separate splines by {name1} level"
            else:
                return f"Consider C({name1}):{name2} interaction term"
        elif not is_cat1 and is_cat2:
            if has_spline1:
                return f"Consider {name1}:C({name2}) or separate splines by {name2} level"
            else:
                return f"Consider {name1}:C({name2}) interaction term"
        else:
            if has_spline1 or has_spline2 or has_poly1 or has_poly2:
                return f"Consider {name1}:{name2} or tensor product spline"
            else:
                return f"Consider {name1}:{name2} interaction or joint spline"
    
    def _compute_eta_squared(self, categories: np.ndarray) -> float:
        """Compute eta-squared for categorical association with residuals."""
        unique_levels = np.unique(categories)
        overall_mean = np.mean(self.pearson_residuals)
        ss_total = np.sum((self.pearson_residuals - overall_mean) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        ss_between = 0.0
        for level in unique_levels:
            mask = categories == level
            level_resid = self.pearson_residuals[mask]
            level_mean = np.mean(level_resid)
            ss_between += len(level_resid) * (level_mean - overall_mean) ** 2
        
        return ss_between / ss_total
    
    def _compute_interaction_strength(
        self,
        name1: str,
        bins1: np.ndarray,
        name2: str,
        bins2: np.ndarray,
        min_cell_count: int,
    ) -> Optional[InteractionCandidate]:
        """Compute interaction strength between two discretized factors."""
        # Create interaction cells
        cell_ids = bins1 * 1000 + bins2
        unique_cells = np.unique(cell_ids)
        
        # Filter cells with sufficient data
        valid_cells = []
        cell_residuals = []
        
        for cell_id in unique_cells:
            mask = cell_ids == cell_id
            if np.sum(mask) >= min_cell_count:
                valid_cells.append(cell_id)
                cell_residuals.append(self.pearson_residuals[mask])
        
        if len(valid_cells) < 4:
            return None
        
        # Compute variance explained by cells
        all_resid = np.concatenate(cell_residuals)
        overall_mean = np.mean(all_resid)
        ss_total = np.sum((all_resid - overall_mean) ** 2)
        
        if ss_total == 0:
            return None
        
        ss_model = sum(
            len(r) * (np.mean(r) - overall_mean) ** 2
            for r in cell_residuals
        )
        
        r_squared = ss_model / ss_total
        
        # F-test p-value
        df_model = len(valid_cells) - 1
        df_resid = len(all_resid) - len(valid_cells)
        
        if df_model > 0 and df_resid > 0:
            f_stat = (ss_model / df_model) / ((ss_total - ss_model) / df_resid)
            pvalue = 1 - _f_cdf(f_stat, float(df_model), float(df_resid))
        else:
            pvalue = float('nan')
        
        return InteractionCandidate(
            factor1=name1,
            factor2=name2,
            interaction_strength=float(r_squared),
            pvalue=float(pvalue),
            n_cells=len(valid_cells),
        )
