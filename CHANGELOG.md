# Changelog

## Unreleased

### Added

- Native `multinomial_dict` baseline-category multinomial logit for mutually
  exclusive class outcomes such as insurance product-tier conversion.
- Shared-covariate and wide-format alternative-specific terms, including generic
  and class-specific alternative coefficients for tier price, richness, limits,
  or deductible columns.
- Row weights, class weights, availability masks, class-specific utility
  offsets, ridge regularization, summaries, diagnostics, tier-mix reports,
  vector-intercept calibration objects, price-change scenarios, and pickle
  serialization for multinomial models.
- `examples/tier_conversion_multinomial.py`, a complete train/holdout pricing
  workflow with held-out log loss, tier-specific price/richness terms,
  availability, calibration, and a premium price scenario.
- `benchmarks/bench_multinomial.py`, a dense native multinomial benchmark harness
  with quick/full grids, RSS sampling, Hessian sizing, guard checks, and
  prediction-throughput reporting.

### Deferred

- Multinomial lasso/elastic net, cross-validation, target encoding, automatic
  smooth penalties, symmetric reference-invariant ridge, and PMML/ONNX export
  remain reserved for later native support and fail explicitly where applicable.
