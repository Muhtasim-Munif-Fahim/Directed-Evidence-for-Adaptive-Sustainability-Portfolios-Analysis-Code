"""Build reproducible source, mapping, unit, missingness, and coverage audits."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUT = ROOT / "outputs" / "data_audit"

SDR = RAW / "SDR2025-data.xlsx"
GHED = RAW / "who_ghed" / "GHED_data.xlsx"
WDI = RAW / "world_bank" / "world_bank_policy_validation_long.csv"
WDI_META = RAW / "world_bank" / "world_bank_policy_validation_metadata.json"
INCOME = RAW / "wb_income_classification.csv"
REGION = RAW / "wb_region_classification.csv"

GOALS = ["goal1", "goal3", "goal4", "goal7", "goal8", "goal10", "goal13", "goal16"]
UNITS = {
    "goal1": "SDR normalized score, 0-100",
    "goal3": "SDR normalized score, 0-100",
    "goal4": "SDR normalized score, 0-100",
    "goal7": "SDR normalized score, 0-100",
    "goal8": "SDR normalized score, 0-100",
    "goal10": "SDR normalized score, 0-100",
    "goal13": "SDR normalized score, 0-100",
    "goal16": "SDR normalized score, 0-100",
    "education_expenditure_pct_gdp": "percent of GDP",
    "primary_enrollment_gross_pct": "percent, gross enrollment",
    "secondary_enrollment_gross_pct": "percent, gross enrollment",
    "life_expectancy_years": "years",
    "under5_mortality_per_1000": "deaths per 1,000 live births",
    "poverty_headcount_3usd_2021ppp_pct": "percent of population",
    "che_gdp": "percent of GDP",
    "che_pc_usd": "current US dollars per capita",
    "gghed_gdp": "percent of GDP",
    "gghed_gge": "percent of general government expenditure",
    "gghed_pc_usd": "current US dollars per capita",
    "gghed_che": "percent of current health expenditure",
    "ext_che": "percent of current health expenditure",
    "ext_pc_usd": "current US dollars per capita",
    "oop_pc_usd": "current US dollars per capita",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_manifest() -> pd.DataFrame:
    sources = [
        ("sdr", SDR, "Sustainable Development Report 2025 workbook"),
        ("who_ghed", GHED, "WHO Global Health Expenditure Database"),
        ("wdi", WDI, "World Bank API extract"),
        ("wdi_metadata", WDI_META, "World Bank API acquisition metadata"),
        ("income_mapping", INCOME, "World Bank income classification"),
        ("region_mapping", REGION, "World Bank region classification"),
    ]
    rows = []
    for source_id, path, description in sources:
        rows.append(
            {
                "source_id": source_id,
                "description": description,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
                "zip_signature_valid": (
                    zipfile.is_zipfile(path)
                    if path.exists() and path.suffix.lower() == ".xlsx"
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def load_sdr_core() -> pd.DataFrame:
    frame = pd.read_excel(
        SDR,
        sheet_name="Backdated SDG Index",
        usecols=["id", "Country", "year", "indexreg_", *GOALS],
    )
    frame = frame[
        ~frame["id"].astype(str).str.startswith("_")
        & frame["year"].between(2000, 2024)
    ].copy()
    complete = frame.dropna(subset=GOALS)
    valid_ids = complete.groupby("id")["year"].nunique()
    core = complete[complete["id"].isin(valid_ids[valid_ids == 25].index)].copy()
    if len(core) != 3800 or core["id"].nunique() != 152:
        raise AssertionError(
            f"SDR core must be 152 x 25; got {core['id'].nunique()} x "
            f"{core['year'].nunique()} ({len(core)} rows)"
        )
    if core.duplicated(["id", "year"]).any():
        raise AssertionError("Duplicate SDR country-year keys")
    return core.sort_values(["id", "year"])


def audit_mappings(core: pd.DataFrame) -> pd.DataFrame:
    income = pd.read_csv(INCOME).rename(columns={"iso3": "id"})
    region = pd.read_csv(REGION).rename(columns={"iso3": "id"})
    ids = core[["id", "Country"]].drop_duplicates()
    mapped = ids.merge(
        income[["id", "income_level", "country_name"]],
        on="id",
        how="left",
        suffixes=("_sdr", "_income"),
    ).merge(
        region[["id", "region", "country_name"]],
        on="id",
        how="left",
        suffixes=("", "_region"),
    )
    mapped["income_mapped"] = mapped["income_level"].notna()
    mapped["region_mapped"] = mapped["region"].notna()
    return mapped


def coverage_audit(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PROCESSED / "external_validation_panel.csv")
    if panel.duplicated(["id", "year"]).any():
        raise AssertionError("Duplicate external-validation country-year keys")
    if set(panel[["id", "year"]].itertuples(index=False, name=None)) != set(
        core[["id", "year"]].itertuples(index=False, name=None)
    ):
        raise AssertionError("External panel keys do not exactly match SDR core")

    variables = [c for c in panel.columns if c not in {"id", "year"}]
    rows = []
    by_country = []
    for variable in variables:
        valid = panel.dropna(subset=[variable])
        rows.append(
            {
                "variable": variable,
                "unit": UNITS.get(variable, "source-defined; see codebook"),
                "observations": len(valid),
                "missing": int(panel[variable].isna().sum()),
                "coverage_pct": 100 * len(valid) / len(panel),
                "countries": valid["id"].nunique(),
                "first_year": valid["year"].min() if len(valid) else None,
                "last_year": valid["year"].max() if len(valid) else None,
            }
        )
        counts = valid.groupby("id")["year"].nunique()
        for country_id in core["id"].drop_duplicates():
            by_country.append(
                {
                    "id": country_id,
                    "variable": variable,
                    "observed_years": int(counts.get(country_id, 0)),
                    "coverage_pct": 100 * int(counts.get(country_id, 0)) / 25,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(by_country)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = load_sdr_core()
    sources = source_manifest()
    mappings = audit_mappings(core)
    variables, country_variable = coverage_audit(core)

    sources.to_csv(OUT / "source_manifest.csv", index=False)
    mappings.to_csv(OUT / "country_mapping_audit.csv", index=False)
    variables.to_csv(OUT / "variable_coverage.csv", index=False)
    country_variable.to_csv(OUT / "country_variable_coverage.csv", index=False)

    summary = {
        "sdr_rows": len(core),
        "sdr_countries": int(core["id"].nunique()),
        "sdr_years": int(core["year"].nunique()),
        "unique_country_year_keys": not core.duplicated(["id", "year"]).any(),
        "income_mapping_pct": 100 * mappings["income_mapped"].mean(),
        "region_mapping_pct": 100 * mappings["region_mapped"].mean(),
        "source_files_present": bool(sources["exists"].all()),
        "xlsx_files_valid": bool(
            sources["zip_signature_valid"].dropna().astype(bool).all()
        ),
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
