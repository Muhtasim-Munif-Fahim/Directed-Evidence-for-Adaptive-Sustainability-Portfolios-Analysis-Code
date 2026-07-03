"""Download prespecified World Bank indicators through the official API."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "world_bank"
START_YEAR = 2000
END_YEAR = 2024

INDICATORS = {
    "SE.XPD.TOTL.GD.ZS": "education_expenditure_pct_gdp",
    "SE.PRM.ENRR": "primary_enrollment_gross_pct",
    "SE.SEC.ENRR": "secondary_enrollment_gross_pct",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SH.DYN.MORT": "under5_mortality_per_1000",
    "SI.POV.DDAY": "poverty_headcount_3usd_2021ppp_pct",
}


def fetch_json(url: str, attempts: int = 5) -> object:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SDG-Time-Series-Revision/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def country_codes() -> set[str]:
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    payload = fetch_json(url)
    records = payload[1]
    return {
        row["id"]
        for row in records
        if row.get("region", {}).get("id") not in {"", "NA"}
        and len(row.get("id", "")) == 3
    }


def download_indicator(indicator: str, valid_codes: set[str]) -> tuple[list[dict], dict]:
    query = urllib.parse.urlencode(
        {
            "format": "json",
            "date": f"{START_YEAR}:{END_YEAR}",
            "per_page": 20000,
        }
    )
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?{query}"
    payload = fetch_json(url)
    metadata, observations = payload
    rows = []
    for item in observations:
        iso3 = item.get("countryiso3code", "")
        if iso3 not in valid_codes:
            continue
        rows.append(
            {
                "iso3": iso3,
                "country": item["country"]["value"],
                "year": int(item["date"]),
                "indicator": indicator,
                "variable": INDICATORS[indicator],
                "indicator_name": item["indicator"]["value"],
                "value": item["value"],
            }
        )
    metadata = {
        "indicator": indicator,
        "variable": INDICATORS[indicator],
        "api_url": url,
        "last_updated": metadata.get("lastupdated"),
        "rows": len(rows),
    }
    return rows, metadata


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_codes = country_codes()
    all_rows: list[dict] = []
    metadata: list[dict] = []
    for indicator in INDICATORS:
        rows, info = download_indicator(indicator, valid_codes)
        all_rows.extend(rows)
        metadata.append(info)
        print(f"{indicator}: {len(rows)} country-year rows")

    frame = pd.DataFrame(all_rows).sort_values(["iso3", "year", "indicator"])
    frame.to_csv(OUT_DIR / "world_bank_policy_validation_long.csv", index=False)
    (OUT_DIR / "world_bank_policy_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {len(frame):,} rows for {frame['iso3'].nunique()} countries "
        f"to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()

