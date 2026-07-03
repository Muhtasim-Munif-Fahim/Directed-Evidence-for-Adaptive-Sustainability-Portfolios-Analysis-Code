"""Shared result schema and deterministic claim classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "config" / "study_design.json").read_text(encoding="utf-8")
)
PROTOCOL_VERSION = PROTOCOL["protocol_version"]

RESULT_COLUMNS = [
    "model_id",
    "family_id",
    "family",
    "estimator",
    "source",
    "target",
    "coefficient",
    "std_error",
    "ci_low",
    "ci_high",
    "p_raw",
    "q_bh",
    "observations",
    "countries",
    "diagnostic_gate",
    "diagnostic_reason",
    "direction_consistent",
    "claim_status",
    "protocol_version",
    "effect_scale",
    "coefficient_std",
    "ci_low_std",
    "ci_high_std",
    "expected_direction",
]


def classify_claim(
    *,
    diagnostic_gate: str,
    p_raw: float | None,
    q_bh: float | None,
    direction_consistent: bool | None,
    promotion_pass: bool,
) -> str:
    """Apply the frozen strict claim hierarchy."""
    if diagnostic_gate == "fail":
        return "not_interpretable"
    if diagnostic_gate not in {"pass", "not_applicable"}:
        raise ValueError(f"Invalid diagnostic gate: {diagnostic_gate}")

    p_value = np.nan if p_raw is None else float(p_raw)
    q_value = np.nan if q_bh is None else float(q_bh)
    direction_ok = bool(direction_consistent)

    if (
        direction_ok
        and np.isfinite(q_value)
        and q_value < 0.05
        and promotion_pass
    ):
        return "supported"
    if direction_ok and np.isfinite(p_value) and p_value < 0.05:
        return "suggestive"
    return "unsupported"


def ensure_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a result frame with the full public contract in stable order."""
    result = frame.copy()
    defaults: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_gate": "not_applicable",
        "diagnostic_reason": "",
        "direction_consistent": False,
        "claim_status": "unsupported",
        "effect_scale": "raw",
        "coefficient_std": np.nan,
        "ci_low_std": np.nan,
        "ci_high_std": np.nan,
        "expected_direction": np.nan,
    }
    for column in RESULT_COLUMNS:
        if column not in result:
            result[column] = defaults.get(column, np.nan)
    invalid = set(result["claim_status"].dropna()) - set(
        PROTOCOL["claim_rules"]["allowed_statuses"]
    )
    if invalid:
        raise ValueError(f"Invalid claim statuses: {sorted(invalid)}")
    return result[RESULT_COLUMNS + [c for c in result.columns if c not in RESULT_COLUMNS]]


def bh_adjust(frame: pd.DataFrame, p_column: str = "p_raw") -> pd.DataFrame:
    """Apply Benjamini-Hochberg to one already-selected family."""
    from statsmodels.stats.multitest import multipletests

    result = frame.copy()
    valid = result[p_column].notna()
    result["q_bh"] = np.nan
    if valid.any():
        result.loc[valid, "q_bh"] = multipletests(
            result.loc[valid, p_column].astype(float).to_numpy(),
            method="fdr_bh",
        )[1]
    return result
