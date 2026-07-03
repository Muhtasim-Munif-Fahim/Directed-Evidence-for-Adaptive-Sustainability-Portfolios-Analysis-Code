"""Income-group-stratified local projections (reviewer heterogeneity check).

Gemini reviewer's top ask: do the supported/suggestive dynamic linkages, and
their 4-5 year accrual profile, hold across development stages, or is the
pooled estimate driven by one income stratum?

Design: the five supported/suggestive pairs are re-estimated separately for
World Bank income strata (Data/wb_income_classification.csv; the CURRENT
classification, so strata membership is time-invariant -- an acknowledged
simplification). Four-way strata leave Driscoll-Kraay inference underpowered
for several pairs, so the prespecified split is binary:
    lower  = LIC + LMIC,   higher = UMIC + HIC.
Same specification as the primary LP (cumulative y_{t+h}-y_{t-1}, country and
year fixed effects, Driscoll-Kraay kernel SEs, control lags = 2). For the two
SUPPORTED pairs the full IRF path h=0..10 per stratum is saved for sensitivity
inspection; for suggestive pairs only the confirmatory h*=5 estimate.

Outputs:
  outputs/network/lp_income_strata.json
  outputs/network/lp_income_strata_paths.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from run_panel_local_projections import (
    CTRL_LAGS, H_STAR, HORIZONS, INDICATOR_LABELS, build_lp_frame,
    load_pair_panel,
)
from result_contract import PROTOCOL_VERSION

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
OUT_DIR = ROOT / "outputs" / "network"

MIN_STRATUM_COUNTRIES = 25

FOCAL = [
    ("n_sdg1_wpc",       "n_sdg3_u5mort", "supported"),
    ("n_sdg6_sanita",    "n_sdg3_u5mort", "supported"),
    ("n_sdg3_u5mort",    "n_sdg4_second", "suggestive"),
    ("n_sdg5_edat",      "n_sdg3_u5mort", "suggestive"),
    ("n_sdg7_cleanfuel", "n_sdg3_u5mort", "suggestive"),
]

STRATA = {"lower (LIC+LMIC)": {"LIC", "LMIC"},
          "higher (UMIC+HIC)": {"UMIC", "HIC"}}


def fit_h(lp: pd.DataFrame, h: int, ctrl: list[str]) -> dict | None:
    cols = ["id", "year", f"resp_h{h}", "ds", *ctrl]
    d = lp[cols].dropna().copy()
    n_c = d["id"].nunique()
    if n_c < MIN_STRATUM_COUNTRIES or len(d) < 100:
        return None
    d = d.set_index(["id", "year"])
    try:
        res = PanelOLS(d[f"resp_h{h}"], d[["ds", *ctrl]],
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type="kernel",
                                               bandwidth=max(h, 1))
    except Exception:
        return None
    b = float(res.params["ds"])
    se = float(res.std_errors["ds"])
    return {"horizon": h, "beta": round(b, 4), "se": round(se, 4),
            "ci_low": round(b - 1.96 * se, 4),
            "ci_high": round(b + 1.96 * se, 4),
            "p": float(f"{float(res.pvalues['ds']):.4g}"),
            "n_obs": int(res.nobs), "n_countries": int(n_c)}


def main() -> None:
    income = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "wb_income_classification.csv")
    inc_map = dict(zip(income["iso3"], income["income_level"]))

    ctrl = [f"dy_l{l}" for l in range(1, CTRL_LAGS + 1)] + \
           [f"ds_l{l}" for l in range(1, CTRL_LAGS + 1)]

    results: dict = {}
    path_rows: list[dict] = []

    for s, t, status in FOCAL:
        sl, tl = INDICATOR_LABELS[s], INDICATOR_LABELS[t]
        label = f"{sl}->{tl}"
        print(f"\n{label} [{status}]")
        panel = load_pair_panel(s, t)
        panel = panel.assign(income=panel["id"].map(inc_map))
        pair_res = {"status_pooled": status, "strata": {}}

        for sname, groups in STRATA.items():
            sub = panel[panel["income"].isin(groups)]
            n_c = sub["id"].nunique()
            if n_c < MIN_STRATUM_COUNTRIES:
                pair_res["strata"][sname] = {
                    "n_countries": int(n_c),
                    "verdict": f"underpowered (<{MIN_STRATUM_COUNTRIES} countries)"}
                print(f"  {sname}: underpowered ({n_c} countries)")
                continue
            lp = build_lp_frame(sub, s, t)
            horizons = HORIZONS if status == "supported" else [H_STAR]
            est = {h: fit_h(lp, h, ctrl) for h in horizons}
            h5 = est.get(H_STAR)
            pair_res["strata"][sname] = {
                "n_countries": int(n_c),
                "h_star": h5 if h5 else "not estimable",
            }
            if h5:
                print(f"  {sname}: n={n_c}  beta5={h5['beta']:+.4f} "
                      f"[{h5['ci_low']:+.4f},{h5['ci_high']:+.4f}] p={h5['p']}")
            for h, e in est.items():
                if e:
                    path_rows.append({"pair": label, "stratum": sname, **e})

        # sign consistency verdict at h*
        betas = [v["h_star"]["beta"] for v in pair_res["strata"].values()
                 if isinstance(v, dict) and isinstance(v.get("h_star"), dict)]
        if len(betas) == 2:
            pair_res["sign_consistent_across_strata"] = bool(
                np.sign(betas[0]) == np.sign(betas[1]))
        results[label] = pair_res

    out = {
        "protocol_version": PROTOCOL_VERSION,
        "stratification": ("World Bank income groups (current classification, "
                            "time-invariant); binary split LIC+LMIC vs UMIC+HIC "
                            "prespecified for power"),
        "specification": ("identical to primary LP: cumulative response, "
                           "country+year FE, Driscoll-Kraay kernel SE, "
                           f"{CTRL_LAGS} control lags, h*={H_STAR}"),
        "min_stratum_countries": MIN_STRATUM_COUNTRIES,
        "pairs": results,
    }
    (OUT_DIR / "lp_income_strata.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    pd.DataFrame(path_rows).to_csv(OUT_DIR / "lp_income_strata_paths.csv",
                                   index=False)
    print(f"\nWrote {OUT_DIR / 'lp_income_strata.json'}")
    print(f"Wrote {OUT_DIR / 'lp_income_strata_paths.csv'}")


if __name__ == "__main__":
    main()
