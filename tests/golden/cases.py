"""The fixed dataset, the fixed parameters and how a result is flattened.

Statistical bugs do not raise. A wrong denominator, a dropped row, an off-by-one
in a degrees-of-freedom term — all of those return a number that looks perfectly
reasonable, and no assertion in the existing test suite would notice. The golden
baseline exists for exactly that failure mode: it pins the actual numbers each
method produces on one dataset that never changes.

Rules for this directory:

* ``baseline_survey.csv`` is a fixture, not test data to be regenerated on a
  whim. Changing it invalidates every baseline at once.
* Every case must be deterministic. Bootstrap methods pass an explicit seed.
* Regenerating a baseline is an explicit act: run ``regenerate.py`` and read the
  diff before committing it. A baseline updated without looking is worse than no
  baseline, because it launders a regression into the repository.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

GOLDEN_DIR = Path(__file__).resolve().parent
DATASET_PATH = GOLDEN_DIR / "baseline_survey.csv"

# Editing the fixture silently invalidates every baseline at once, and the
# baselines would still compare green against whatever the new data produces.
# The digest makes that mistake loud.
DATASET_SHA256 = "09b6558354a40e50212225777c2bbcb9433c88e083c6ceced07a4d3debefc461"

FACTORS = {
    "维度A": ["A1", "A2", "A3", "A4"],
    "维度B": ["B1", "B2", "B3", "B4"],
    "维度C": ["C1", "C2", "C3", "C4"],
}
ALL_ITEMS = [item for items in FACTORS.values() for item in items]

# method key -> parameters. Every one of the engine's methods must appear here.
CASES: dict[str, dict[str, Any]] = {
    "descriptive": {"cols": ["A1", "A2", "A3", "A4", "X", "Y"]},
    "frequency": {"cols": ["Group", "A1"]},
    "reliability": {"cols": FACTORS["维度A"]},
    "validity": {"cols": ALL_ITEMS},
    "correlation": {"cols": ["X", "W", "M1", "M2", "Y"]},
    "efa": {"cols": ALL_ITEMS, "n_factors": 3},
    "cfa": {"factors": FACTORS},
    "regression": {"y": "Y", "x_cols": ["X", "M1", "M2"]},
    "moderation": {"x": "X", "y": "Y", "m": "W"},
    "parallel_mediation": {
        "x": "X",
        "y": "Y",
        "m_cols": ["M1", "M2"],
        # Bootstrap is only reproducible with a pinned seed and count.
        "n_boot": 200,
        "seed": 517,
        "ci": 0.95,
    },
    "sem": {
        "factors": FACTORS,
        "paths": [("维度A", "维度B"), ("维度B", "维度C")],
    },
}

# Methods whose baseline cannot be produced in this environment. Each entry
# needs a reason, and the test suite fails if a method is in neither CASES'
# baselines nor here — a method must never quietly lose its cover.
PENDING: dict[str, str] = {}


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def baseline_path(method: str) -> Path:
    return GOLDEN_DIR / f"{method}.json"


# Tables longer than this are summarised instead of stored row by row. The
# long ones are all per-respondent derived scores (180 rows of factor scores,
# for example); their inputs — loadings, eigenvalues, coefficients — are pinned
# exactly in the short tables, so storing every row adds little detection power
# and a lot of unreadable diff.
MAX_PINNED_ROWS = 30


def _table(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    records = [
        {str(key): normalize(item) for key, item in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    if len(records) <= MAX_PINNED_ROWS:
        return {"columns": columns, "rows": records}

    # Head and tail make a failure diagnosable; the per-column aggregate makes
    # a change in the middle of the table impossible to miss.
    aggregates: dict[str, Any] = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        aggregates[str(column)] = {
            "count": int(series.size),
            "sum": round(float(series.sum()), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
        }
    return {
        "columns": columns,
        "row_count": len(records),
        "head": records[:3],
        "tail": records[-3:],
        "column_aggregates": aggregates,
    }


def normalize(value: Any) -> Any:
    """Turn an engine result into plain JSON, keeping every number.

    DataFrames become explicit ``columns`` plus ``rows`` so that a renamed or
    reordered column is a visible difference, not a silent one.
    """

    if isinstance(value, pd.DataFrame):
        return _table(value)
    if isinstance(value, pd.Series):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize(item) for item in value]
    if hasattr(value, "item") and hasattr(value, "dtype"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return {"__float__": repr(value)}
    return value


def fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    """The part of a result the baseline is responsible for.

    ``display`` is included, and not only for cosmetic reasons: the engine puts
    real statistics in there as pre-rendered text. A linear regression's model
    summary line — R², adjusted R², the F value with its degrees of freedom and
    p — exists nowhere else in the result. Leaving ``display`` out let an
    off-by-one in the residual degrees of freedom pass the baseline unnoticed.
    """

    return {
        "method": result.get("method"),
        "label": result.get("label"),
        "meta": normalize(result.get("meta") or {}),
        "tables": [
            {
                "key": table.get("key"),
                "title": table.get("title"),
                **normalize(table["data"]),
            }
            for table in result.get("tables") or []
        ],
        "display": normalize(result.get("display") or {}),
    }
