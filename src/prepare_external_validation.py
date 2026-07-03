"""Prepare independent policy-input and outcome variables for validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SDR_FILE = RAW / "SDR2025-data.xlsx"
GHED_FILE = RAW / "who_ghed" / "GHED_data.xlsx"
WDI_FILE = RAW / "world_bank" / "world_bank_policy_validation_long.csv"

GOALS = [f"goal{i}" for i in range(1, 18)]
GHED_VARIABLES = [
    "che_gdp",
    "che_pc_usd",
    "gghed_gdp",
    "gghed_gge",
    "gghed_pc_usd",
    "gghed_che",
    "ext_che",
    "ext_pc_usd",
    "oop_pc_usd",
]


def load_sdr_goals() -> pd.DataFrame:
    """SDR goal scores for the FULL country universe (no balance filter).

    The external-validation panel is deliberately built on the whole SDR
    country universe, not the balanced 114-country network panel: the point
    of triangulation is independent data with its own (broader) coverage.
    Restricting to the balanced panel compresses the variance of enrollment
    and mortality (its members are richer, near-ceiling) and is reported only
    as a sensitivity subset.
    """
    frame = pd.read_excel(
        SDR_FILE,
        sheet_name="Backdated SDG Index",
        usecols=["id", "year", *GOALS],
    )
    frame = frame[~frame["id"].astype(str).str.startswith("_")].copy()
    frame = frame[frame["year"].between(2000, 2024)]
    print(f"SDR universe: {frame['id'].nunique()} countries")
    return frame.sort_values(["id", "year"])


def balanced_ids(sdr: pd.DataFrame) -> list[str]:
    complete = sdr.dropna(subset=GOALS)
    counts = complete.groupby("id")["year"].nunique()
    ids = sorted(counts[counts == 25].index)
    if len(ids) < 100:
        raise ValueError(
            f"Balanced panel has only {len(ids)} countries (minimum 100 required)"
        )
    return ids


def load_ghed() -> pd.DataFrame:
    columns = ["code", "location", "region", "income", "year", *GHED_VARIABLES]
    frame = pd.read_excel(GHED_FILE, sheet_name="Data", usecols=columns)
    frame = frame.rename(columns={"code": "id", "location": "country"})
    frame = frame[frame["year"].between(2000, 2024)]
    return frame.sort_values(["id", "year"])


def load_wdi() -> pd.DataFrame:
    frame = pd.read_csv(WDI_FILE)
    wide = frame.pivot_table(
        index=["iso3", "year"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"iso3": "id"})


def coverage_table(frame: pd.DataFrame, external_columns: list[str]) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for column in external_columns:
        valid = frame.dropna(subset=[column])
        rows.append(
            {
                "variable": column,
                "observations": len(valid),
                "coverage_pct": round(100 * len(valid) / total, 2),
                "countries": valid["id"].nunique(),
                "first_year": int(valid["year"].min()) if len(valid) else None,
                "last_year": int(valid["year"].max()) if len(valid) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["coverage_pct", "variable"],
        ascending=[False, True],
    )


def main() -> None:
    if not GHED_FILE.exists():
        raise FileNotFoundError(f"Download WHO GHED first: {GHED_FILE}")
    if not WDI_FILE.exists():
        raise FileNotFoundError(
            f"Run download_world_bank.py before this script: {WDI_FILE}"
        )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    sdr = load_sdr_goals()
    ghed = load_ghed()
    wdi = load_wdi()

    ghed.to_csv(PROCESSED / "who_ghed_selected.csv", index=False)
    wdi.to_csv(PROCESSED / "world_bank_policy_validation_wide.csv", index=False)

    universe = sorted(sdr["id"].unique())
    base = wdi[wdi["id"].isin(universe)].copy()
    merged = base.merge(
        ghed.drop(columns=["country"], errors="ignore"),
        on=["id", "year"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(sdr, on=["id", "year"], how="left", validate="one_to_one")
    merged.to_csv(PROCESSED / "external_validation_panel.csv", index=False)

    # sensitivity subset: balanced 114-country SDR network panel members only
    bal = balanced_ids(sdr)
    merged[merged["id"].isin(bal)].to_csv(
        PROCESSED / "external_validation_panel_balanced114.csv", index=False
    )
    print(f"Full validation panel: {merged['id'].nunique()} countries; "
          f"balanced subset: {len(bal)} countries")

    external = [column for column in merged.columns if column not in {"id", "year", *GOALS}]
    coverage = coverage_table(merged, external)
    coverage.to_csv(PROCESSED / "external_validation_coverage.csv", index=False)

    summary = {
        "rows": len(merged),
        "countries": int(merged["id"].nunique()),
        "years": [int(merged["year"].min()), int(merged["year"].max())],
        "ghed_source": "https://apps.who.int/nha/database/Home/IndicatorsDownload/en",
        "world_bank_source": "https://api.worldbank.org/v2/",
    }
    (PROCESSED / "external_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
