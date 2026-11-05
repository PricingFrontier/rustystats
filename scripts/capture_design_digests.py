"""Record the design digests that ``tests/python/test_frozen_design.py`` pins (§14 T1).

For every term-kind case (``tests/python/_design_cases.TERM_KINDS``) and every golden case
without target encoding, the script builds X, y and the column names with
``InteractionBuilder.build_design_matrix_from_parsed`` from the case's portable input
frame (floats on a dyadic grid, ``_design_cases.portable``) and records SHA-256 digests of
the input frame, y, the names and X. A design that evaluates a transcendental function
(``_design_cases.evaluates_transcendentals``) has its X stored in the ``.npz`` beside the
record instead, for a comparison within a few ulps. Run it against the v0.9.0 build,
whose builder is the oracle:

    git archive v0.9.0 | tar -x -C /tmp/v090    # then build its extension there
    PYTHONPATH=/tmp/v090/python python scripts/capture_design_digests.py \\
        tests/golden/design_digests_v0_9_0.json --source "v0.9.0 (git archive)"
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))

import numpy as np
import polars as pl
from _design_cases import (
    DIGEST_SCHEMA,
    array_key,
    design_record,
    design_specs,
    evaluates_transcendentals,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True, help="the build recorded, e.g. its tag")
    args = parser.parse_args()

    import rustystats as rs
    from rustystats.formula import dict_to_parsed_formula
    from rustystats.interactions import InteractionBuilder

    designs = {}
    arrays = {}
    for name, (data, spec) in design_specs().items():
        parsed = dict_to_parsed_formula(
            response=spec["response"],
            terms=spec["terms"],
            interactions=spec.get("interactions"),
            intercept=spec.get("intercept", True),
        )
        y, x, names = InteractionBuilder(data).build_design_matrix_from_parsed(parsed)
        designs[name] = design_record(data, spec, name, y, x, names)
        if evaluates_transcendentals(spec):
            arrays[array_key(name)] = np.ascontiguousarray(x, dtype=np.float64)
        print(f"recorded {name}: X {x.shape}")
    document = {
        "schema": DIGEST_SCHEMA,
        "source": args.source,
        "rustystats_version": rs.__version__,
        "recorded": datetime.date.today().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "machine": platform.machine(),
        "designs": designs,
    }
    args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    archive = args.output.with_suffix(".npz")
    np.savez_compressed(archive, **arrays)
    print(f"wrote {args.output}: {len(designs)} designs; {archive}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
