"""Full reporting of the WDI education -> child-health triangulation.

Stanford reviewer ask: report the external-validation regression completely --
exact series codes, sample, specification, natural-unit effect sizes, and
robustness to standard development controls.

Focal analysis claim: in independent World Development Indicators data, a
lagged increase in gross secondary enrollment (SE.SEC.ENRR) predicts a
subsequent reduction in under-5 mortality (SH.DYN.MORT).

Rows reported:
  1. baseline      annual-change spec exactly as the primary pipeline:
                   d(U5MR)_t ~ lag d(enroll) + lag d(U5MR) + year dummies,
                   SEs clustered by country (differencing removes country
                   levels; no additional country fixed effects)
  2. + country FE  adds country dummies to the change equation (country trends)
  3. + health exp  adds lag d(gghed_gdp)  [WHO GHED, government health
                   expenditure % GDP -- local data]
  4. + educ exp    adds lag d(education_expenditure_pct_gdp) [SE.XPD.TOTL.GD.ZS]
  5. + GDP pc      adds lag d(log GDP per capita) [NY.GDP.PCAP.KD] (fetched and
                   cached if absent locally; row skipped with a note if offline)
  6. + fertility   adds lag d(fertility) [SP.DYN.TFRT.IN] (same caching rule)
  7. joint         all available controls together
  8. LP-style      5-year cumulative response U5MR_{t+5}-U5MR_{t-1} on
                   d(enroll)_t, country+year FE, Driscoll-Kraay SEs -- the
                   specification that mirrors the SDR local projections.

Natural units: U5MR is deaths per 1,000 live births; enrollment in percentage
points -> effect per 10pp = 10 x coefficient.

Outputs:
  outputs/network/external_validation_full.json
  outputs/network/external_validation_full.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from result_contract import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "external_validation_panel.csv"
RAW_WB = ROOT / "data" / "raw" / "world_bank"
OUT_DIR = ROOT / "outputs" / "network"

EXTRA_SERIES = {
    "NY.GDP.PCAP.KD": "gdp_pc_constant_usd",
    "SP.DYN.TFRT.IN": "fertility_rate",
}


def fetch_extra_series() -> pd.DataFrame | None:
    """Load (or download once and cache) GDP per capita and fertility."""
    cache = RAW_WB / "extra_controls.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        import requests
        frames = []
        for code, name in EXTRA_SERIES.items():
            url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                   f"?format=json&date=2000%3A2024&per_page=20000")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()[1]
            rows = [{"iso3": d["countryiso3code"], "year": int(d["date"]),
                     name: d["value"]} for d in data if d["countryiso3code"]]
            frames.append(pd.DataFrame(rows).dropna(subset=["iso3"]))
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["iso3", "year"], how="outer")
        out.to_csv(cache, index=False)
        print(f"Fetched and cached extra controls -> {cache}")
        return out
    except Exception as err:
        print(f"NOTE: could not fetch GDP/fertility controls ({err}); "
              "those robustness rows will be skipped.")
        return None


def run_spec(df: pd.DataFrame, extra_terms: list[str], label: str,
             country_fe: bool = False) -> dict | None:
    terms = ["lag_d_enroll", "lag_d_u5mr"] + extra_terms
    formula = "d_u5mr ~ " + " + ".join(terms) + " + C(year)"
    if country_fe:
        formula += " + C(id)"
    cols = ["id", "year", "d_u5mr"] + terms
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n_c = d["id"].nunique()
    if n_c < 20 or len(d) < 100:
        return None
    res = smf.ols(formula, data=d).fit(cov_type="cluster",
                                       cov_kwds={"groups": d["id"]})
    b = float(res.params["lag_d_enroll"])
    ci = res.conf_int().loc["lag_d_enroll"]
    return {
        "specification": label,
        "coefficient": round(b, 4),
        "se": round(float(res.bse["lag_d_enroll"]), 4),
        "ci_low": round(float(ci.iloc[0]), 4),
        "ci_high": round(float(ci.iloc[1]), 4),
        "p": float(f"{float(res.pvalues['lag_d_enroll']):.4g}"),
        "effect_per_10pp": round(10 * b, 3),
        "n_obs": int(res.nobs), "n_countries": int(n_c),
        "country_fe": country_fe,
        "controls": extra_terms,
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    extra = fetch_extra_series()
    if extra is not None:
        df = df.merge(extra, left_on=["id", "year"], right_on=["iso3", "year"],
                      how="left")
        df["log_gdp_pc"] = np.log(df["gdp_pc_constant_usd"])

    df = df.sort_values(["id", "year"])
    g = df.groupby("id")
    df["d_u5mr"] = g["under5_mortality_per_1000"].diff()
    df["lag_d_u5mr"] = g["under5_mortality_per_1000"].diff().groupby(df["id"]).shift(1)
    df["d_enroll"] = g["secondary_enrollment_gross_pct"].diff()
    df["lag_d_enroll"] = df.groupby("id")["d_enroll"].shift(1)
    for var, new in [("gghed_gdp", "lag_d_health_exp"),
                     ("education_expenditure_pct_gdp", "lag_d_educ_exp"),
                     ("log_gdp_pc", "lag_d_log_gdp"),
                     ("fertility_rate", "lag_d_fertility")]:
        if var in df.columns:
            df[new] = df.groupby("id")[var].diff().groupby(df["id"]).shift(1)

    rows = []
    rows.append(run_spec(df, [], "1. baseline (year dummies, country-clustered SE)"))
    rows.append(run_spec(df, [], "2. + country fixed effects (trend in changes)",
                         country_fe=True))
    rows.append(run_spec(df, ["lag_d_health_exp"],
                         "3. + govt health expenditure %GDP (GHED)"))
    rows.append(run_spec(df, ["lag_d_educ_exp"],
                         "4. + education expenditure %GDP (SE.XPD.TOTL.GD.ZS)"))
    if "lag_d_log_gdp" in df.columns:
        rows.append(run_spec(df, ["lag_d_log_gdp"],
                             "5. + log GDP per capita (NY.GDP.PCAP.KD)"))
    if "lag_d_fertility" in df.columns:
        rows.append(run_spec(df, ["lag_d_fertility"],
                             "6. + fertility rate (SP.DYN.TFRT.IN)"))
    joint = [c for c in ["lag_d_health_exp", "lag_d_educ_exp",
                         "lag_d_log_gdp", "lag_d_fertility"] if c in df.columns]
    rows.append(run_spec(df, joint, "7. all available controls jointly"))

    # sensitivity: balanced 114-country SDR network-panel members only.
    # Their enrollment/mortality series are near-ceiling/low-variance, so
    # attenuation here reflects variance compression, not contradiction.
    bal_file = ROOT / "data" / "processed" / "external_validation_panel_balanced114.csv"
    if bal_file.exists():
        bal_ids = set(pd.read_csv(bal_file, usecols=["id"])["id"])
        rows.append(run_spec(df[df["id"].isin(bal_ids)], [],
                             "7b. balanced-114 SDR subsample (sensitivity)"))

    # 8. LP-style 5-year cumulative response with country+year FE and DK SEs
    lp_row = None
    try:
        from linearmodels.panel import PanelOLS
        t = df.copy()
        t["resp_h5"] = (t.groupby("id")["under5_mortality_per_1000"].shift(-5)
                        - t.groupby("id")["under5_mortality_per_1000"].shift(1))
        d = t[["id", "year", "resp_h5", "d_enroll", "lag_d_u5mr",
               "lag_d_enroll"]].replace([np.inf, -np.inf], np.nan).dropna()
        if d["id"].nunique() >= 20 and len(d) >= 100:
            di = d.set_index(["id", "year"])
            res = PanelOLS(di["resp_h5"],
                           di[["d_enroll", "lag_d_u5mr", "lag_d_enroll"]],
                           entity_effects=True, time_effects=True,
                           drop_absorbed=True).fit(cov_type="kernel", bandwidth=5)
            b = float(res.params["d_enroll"])
            se = float(res.std_errors["d_enroll"])
            lp_row = {
                "specification": ("8. LP-style: U5MR(t+5)-U5MR(t-1) on "
                                   "d(enroll), country+year FE, DK SE"),
                "coefficient": round(b, 4), "se": round(se, 4),
                "ci_low": round(b - 1.96 * se, 4),
                "ci_high": round(b + 1.96 * se, 4),
                "p": float(f"{float(res.pvalues['d_enroll']):.4g}"),
                "effect_per_10pp": round(10 * b, 3),
                "n_obs": int(res.nobs),
                "n_countries": int(d["id"].nunique()),
                "country_fe": True, "controls": ["lag_d_u5mr", "lag_d_enroll"],
            }
    except Exception as err:
        print(f"LP-style row failed: {err}")
    if lp_row:
        rows.append(lp_row)

    rows = [r for r in rows if r is not None]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_DIR / "external_validation_full.csv", index=False)

    out = {
        "protocol_version": PROTOCOL_VERSION,
        "series_codes": {
            "source": "SE.SEC.ENRR (school enrollment, secondary, % gross)",
            "outcome": "SH.DYN.MORT (under-5 mortality per 1,000 live births)",
            "controls": {
                "health_expenditure": "WHO GHED gghed_gdp (govt health exp %GDP)",
                "education_expenditure": "SE.XPD.TOTL.GD.ZS",
                "gdp_per_capita": "NY.GDP.PCAP.KD (fetched)" if extra is not None
                                   else "NY.GDP.PCAP.KD (UNAVAILABLE offline)",
                "fertility": "SP.DYN.TFRT.IN (fetched)" if extra is not None
                              else "SP.DYN.TFRT.IN (UNAVAILABLE offline)",
            },
        },
        "fe_statement": ("Baseline operates on annual changes (first "
                          "differences), which removes country level effects; "
                          "it includes year dummies and clusters SEs by "
                          "country. Row 2 adds country dummies (country-"
                          "specific trends). Row 8 mirrors the SDR local "
                          "projection: 5-year cumulative outcome, country and "
                          "year fixed effects, Driscoll-Kraay SEs."),
        "reconciliation": ("The SDR local-projection family tests "
                            "WomenEducation (sdg5_edat, stock of attainment) "
                            "-> U5Mortality and finds it suggestive; the WDI "
                            "triangulation uses gross secondary enrollment "
                            "(SE.SEC.ENRR, a flow measure of schooling) from "
                            "an independent data pipeline. Both proxy the "
                            "education -> child-health mechanism; agreement "
                            "in direction across measures and datasets is the "
                            "point of the triangulation, not identity of the "
                            "indicator."),
        "rows": rows,
    }
    (OUT_DIR / "external_validation_full.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(tbl[["specification", "coefficient", "ci_low", "ci_high", "p",
               "effect_per_10pp", "n_countries", "n_obs"]].to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'external_validation_full.json'} and .csv")


if __name__ == "__main__":
    main()
