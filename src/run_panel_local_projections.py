"""Panel local projections (Jorda 2005) for theory-driven SDG indicator linkages.

Why this method:
  Local projections estimate the dynamic (impulse) response of a target to a change
  in a source by running a separate regression at each horizon h, rather than
  iterating a fitted VAR. They are robust to misspecification of the long-run
  dynamics, impose no instrument structure (so none of the small-T GMM instrument
  proliferation), and -- with Driscoll-Kraay standard errors -- are robust to the
  cross-country dependence and serial correlation that pervade macro SDG panels.

Specification, per pair (source s -> target y), for h = 0, 1, ..., H:
    y_{i,t+h} - y_{i,t-1}
        = a_i + lambda_t
          + beta_h * (s_{i,t} - s_{i,t-1})
          + sum_{l=1..L} phi_l * (y_{i,t-l} - y_{i,t-l-1})
          + sum_{l=1..L} psi_l * (s_{i,t-l} - s_{i,t-l-1})
          + e_{i,t+h}

beta_h is the cumulative response of the target (in SDR score points) to a unit
improvement in the source. The sequence {beta_h} is the impulse-response function.
Entity + time fixed effects; Driscoll-Kraay (kernel) covariance with bandwidth H.

A linkage is flagged "supported" if the IRF band excludes zero in the expected
direction for at least two consecutive horizons within h <= 5.

Outputs:
  outputs/network/lp_irf_coefficients.csv  one row per (pair, horizon)
  outputs/network/lp_summary.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from result_contract import PROTOCOL_VERSION, bh_adjust

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
DATA_FILE = ROOT / "data" / "raw" / "SDR2025-data.xlsx"
OUT_DIR = ROOT / "outputs" / "network"

HORIZONS = list(range(0, 11))  # h = 0..10 years
H_STAR = 5        # prespecified confirmatory horizon (5-year cumulative response)
CTRL_LAGS = 2
MIN_COUNTRIES = 30

INDICATOR_LABELS = {
    "n_sdg1_wpc": "SDG1-Poverty", "n_sdg2_undernsh": "SDG2-Undernourishment",
    "n_sdg2_stunting": "SDG2-Stunting", "n_sdg3_u5mort": "SDG3-U5Mortality",
    "n_sdg3_uhc": "SDG3-UHC", "n_sdg3_lifee": "SDG3-LifeExpectancy",
    "n_sdg4_second": "SDG4-SecondaryEnrol", "n_sdg5_edat": "SDG5-WomenEducation",
    "n_sdg5_lfpr": "SDG5-FemaleLabour", "n_sdg6_water": "SDG6-Water",
    "n_sdg6_sanita": "SDG6-Sanitation", "n_sdg7_elecac": "SDG7-Electricity",
    "n_sdg7_cleanfuel": "SDG7-CleanFuel", "n_sdg8_adjgrowthi": "SDG8-Growth",
    "n_sdg10_gini": "SDG10-Gini", "n_sdg11_pm25": "SDG11-AirPollution",
    "n_sdg11_slums": "SDG11-Slums", "n_sdg13_co2gcp": "SDG13-CO2",
    "n_sdg16_cpi": "SDG16-Corruption", "n_sdg16_homicides": "SDG16-Homicides",
}

# (source, target, expected_sign, cluster, claimed_by)
PAIRS = [
    ("n_sdg1_wpc",        "n_sdg4_second",     +1, "A", "Laumann,Swain"),
    ("n_sdg1_wpc",        "n_sdg3_u5mort",     +1, "A", "Laumann"),
    ("n_sdg4_second",     "n_sdg8_adjgrowthi", +1, "A", "Laumann,Swain"),
    ("n_sdg3_u5mort",     "n_sdg4_second",     +1, "A", "Laumann"),
    ("n_sdg3_u5mort",     "n_sdg8_adjgrowthi", +1, "A", "Laumann"),
    ("n_sdg8_adjgrowthi", "n_sdg1_wpc",        +1, "A", "Laumann,Swain"),
    ("n_sdg5_edat",       "n_sdg3_u5mort",     +1, "B", "Laumann"),
    ("n_sdg5_edat",       "n_sdg2_undernsh",   +1, "B", "Laumann"),
    ("n_sdg4_second",     "n_sdg5_lfpr",       +1, "B", "Laumann,Swain"),
    ("n_sdg5_lfpr",       "n_sdg8_adjgrowthi", +1, "B", "Laumann"),
    ("n_sdg6_water",      "n_sdg3_u5mort",     +1, "C", "Bennich"),
    ("n_sdg6_sanita",     "n_sdg3_u5mort",     +1, "C", "Bennich"),
    ("n_sdg6_water",      "n_sdg2_undernsh",   +1, "C", "Bennich"),
    ("n_sdg6_sanita",     "n_sdg2_stunting",   +1, "C", "Bennich"),
    ("n_sdg16_cpi",       "n_sdg8_adjgrowthi", +1, "D", "Laumann"),
    ("n_sdg16_cpi",       "n_sdg4_second",     +1, "D", "Laumann"),
    ("n_sdg16_homicides", "n_sdg8_adjgrowthi", +1, "D", "Laumann"),
    ("n_sdg8_adjgrowthi", "n_sdg13_co2gcp",   -1, "E", "Laumann"),
    ("n_sdg13_co2gcp",    "n_sdg11_pm25",     +1, "E", "Laumann"),
    ("n_sdg13_co2gcp",    "n_sdg11_slums",    +1, "E", "Laumann"),
    ("n_sdg7_elecac",     "n_sdg8_adjgrowthi", +1, "E", "Bennich"),
    ("n_sdg7_elecac",     "n_sdg4_second",    +1, "E", "Bennich"),
    ("n_sdg7_cleanfuel",  "n_sdg3_u5mort",    +1, "E", "Bennich"),
    ("n_sdg2_undernsh",   "n_sdg3_u5mort",    +1, "F", "Bennich,Swain"),
    ("n_sdg2_stunting",   "n_sdg4_second",    +1, "F", "Bennich"),
    ("n_sdg8_adjgrowthi", "n_sdg2_undernsh",  +1, "F", "Swain"),
    ("n_sdg3_uhc",        "n_sdg3_u5mort",    +1, "G", "Laumann"),
    ("n_sdg3_uhc",        "n_sdg3_lifee",     +1, "G", "Laumann"),
    ("n_sdg10_gini",      "n_sdg3_u5mort",    +1, "H", "Laumann"),
    ("n_sdg10_gini",      "n_sdg4_second",    +1, "H", "Laumann"),
    ("n_sdg8_adjgrowthi", "n_sdg10_gini",     +1, "H", "Laumann"),
]


def load_pair_panel(source: str, target: str) -> pd.DataFrame:
    frame = pd.read_excel(
        DATA_FILE, sheet_name="Backdated SDG Index",
        usecols=["id", "year", source, target],
    )
    frame = frame[~frame["id"].astype(str).str.startswith("_")]
    frame = frame[frame["year"].between(2000, 2024)].copy()
    complete = frame.dropna(subset=[source, target])
    counts = complete.groupby("id")["year"].nunique()
    ids = counts[counts == 25].index
    return complete[complete["id"].isin(ids)].sort_values(["id", "year"])


def build_lp_frame(panel: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """Construct differenced LP design with control lags, per country."""
    parts = []
    for cid, unit in panel.groupby("id"):
        u = unit.sort_values("year").copy()
        u["dy"] = u[target].diff()
        u["ds"] = u[source].diff()
        for lag in range(1, CTRL_LAGS + 1):
            u[f"dy_l{lag}"] = u["dy"].shift(lag)
            u[f"ds_l{lag}"] = u["ds"].shift(lag)
        # cumulative response LHS for each horizon: y_{t+h} - y_{t-1}
        for h in HORIZONS:
            u[f"resp_h{h}"] = u[target].shift(-h) - u[target].shift(1)
        parts.append(u)
    return pd.concat(parts, ignore_index=True)


def run_pair_lp(source: str, target: str) -> list[dict]:
    panel = load_pair_panel(source, target)
    n = panel["id"].nunique()
    if n < MIN_COUNTRIES:
        return []
    lp = build_lp_frame(panel, source, target)
    ctrl = [f"dy_l{l}" for l in range(1, CTRL_LAGS + 1)] + \
           [f"ds_l{l}" for l in range(1, CTRL_LAGS + 1)]
    rows = []
    for h in HORIZONS:
        cols = ["id", "year", f"resp_h{h}", "ds", *ctrl]
        d = lp[cols].dropna().copy()
        if d["id"].nunique() < MIN_COUNTRIES or len(d) < 100:
            continue
        d = d.set_index(["id", "year"])
        y = d[f"resp_h{h}"]
        X = d[["ds", *ctrl]]
        try:
            mod = PanelOLS(y, X, entity_effects=True, time_effects=True,
                           drop_absorbed=True)
            res = mod.fit(cov_type="kernel", bandwidth=max(h, 1))
            b = float(res.params["ds"])
            se = float(res.std_errors["ds"])
            p = float(res.pvalues["ds"])
            rows.append({
                "source": source, "target": target, "horizon": h,
                "beta": b, "std_error": se,
                "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                "p_raw": p, "n_obs": int(res.nobs),
                "n_countries": int(d.index.get_level_values("id").nunique()),
            })
        except Exception as err:
            rows.append({
                "source": source, "target": target, "horizon": h,
                "beta": np.nan, "std_error": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p_raw": np.nan, "n_obs": 0,
                "n_countries": 0, "error": str(err)[:120],
            })
    return rows


def pair_confirmatory(pair_rows: pd.DataFrame, exp_sign: int) -> dict:
    """Per-pair confirmatory statistics for the prespecified horizon H_STAR.

    The confirmatory test is the cumulative response at H_STAR years; its p-value
    enters a Benjamini-Hochberg family across all pairs (applied in main). The
    consecutive-significant-horizon count and the early peak are retained only as
    descriptive response-shape diagnostics.
    """
    early = pair_rows[pair_rows["horizon"] <= 5].sort_values("horizon")
    sig_dir = []
    for r in early.itertuples():
        if pd.isna(r.ci_low):
            sig_dir.append(False)
            continue
        excludes_zero = (r.ci_low > 0) or (r.ci_high < 0)
        right_sign = np.sign(r.beta) == np.sign(exp_sign)
        sig_dir.append(bool(excludes_zero and right_sign))
    best = run = 0
    for s in sig_dir:
        run = run + 1 if s else 0
        best = max(best, run)
    early_valid = early.dropna(subset=["beta"])
    peak = early_valid.loc[early_valid["beta"].abs().idxmax()] \
        if len(early_valid) else None

    hrow = pair_rows[(pair_rows["horizon"] == H_STAR)].dropna(subset=["beta"])
    if len(hrow):
        hr = hrow.iloc[0]
        beta_star = float(hr["beta"])
        p_star = float(hr["p_raw"])
        sign_ok = bool(np.sign(beta_star) == np.sign(exp_sign))
    else:
        beta_star = p_star = None
        sign_ok = False
    return {
        "horizon_star": H_STAR,
        "beta_star": beta_star,
        "p_star": p_star,
        "sign_ok_star": sign_ok,
        "max_consecutive_sig": int(best),
        "peak_horizon": int(peak["horizon"]) if peak is not None else None,
        "peak_beta": float(peak["beta"]) if peak is not None else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []

    for i, (s, t, exp, cluster, claimed) in enumerate(PAIRS):
        sl, tl = INDICATOR_LABELS.get(s, s), INDICATOR_LABELS.get(t, t)
        print(f"Pair {i+1:02d}/{len(PAIRS)}: {sl} -> {tl}")
        rows = run_pair_lp(s, t)
        if not rows:
            print("  skipped (insufficient panel)")
            continue
        for r in rows:
            r.update({"cluster": cluster, "claimed_by": claimed,
                      "expected_sign": exp, "source_label": sl,
                      "target_label": tl, "pair_label": f"{sl}->{tl}"})
        all_rows.extend(rows)
        pdf = pd.DataFrame(rows)
        rec = pair_confirmatory(pdf, exp)
        summaries.append({
            "pair": f"{sl}->{tl}", "cluster": cluster, "claimed_by": claimed,
            "expected_sign": exp, **rec,
        })
        print(f"  beta(h={H_STAR})={rec['beta_star']}  p={rec['p_star']}")

    coeffs = pd.DataFrame(all_rows)
    coeffs["protocol_version"] = PROTOCOL_VERSION
    coeffs.to_csv(OUT_DIR / "lp_irf_coefficients.csv", index=False)

    # --- Benjamini-Hochberg across the pair family at the confirmatory horizon ---
    summ_df = pd.DataFrame(summaries)
    est = summ_df[summ_df["p_star"].notna()].copy().rename(columns={"p_star": "p_raw"})
    est = bh_adjust(est)  # adds q_bh on the H_STAR p-values
    qmap = dict(zip(est["pair"], est["q_bh"]))
    for s in summaries:
        q = qmap.get(s["pair"])
        s["q_bh_star"] = float(q) if q is not None and pd.notna(q) else None
        p = s["p_star"]
        if p is None:
            s["status"] = "not_interpretable"
        elif s["q_bh_star"] is not None and s["q_bh_star"] < 0.05 and s["sign_ok_star"]:
            s["status"] = "supported"
        elif p < 0.05 and s["sign_ok_star"]:
            s["status"] = "suggestive"
        else:
            s["status"] = "unsupported"

    sup = [s for s in summaries if s["status"] == "supported"]
    sug = [s for s in summaries if s["status"] == "suggestive"]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "Panel local projections (Jorda 2005), Driscoll-Kraay SE",
        "horizons": HORIZONS, "control_lags": CTRL_LAGS,
        "confirmatory_horizon": H_STAR,
        "multiplicity": f"Benjamini-Hochberg across {len(est)} estimable pairs "
                        f"at horizon {H_STAR}",
        "n_pairs": len(summaries),
        "supported": len(sup), "suggestive": len(sug),
        "supported_pairs": sup, "suggestive_pairs": sug,
        "all_pairs": summaries,
    }
    (OUT_DIR / "lp_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Confirmatory horizon: {H_STAR} yr;  BH across {len(est)} estimable pairs")
    print(f"Pairs analysed: {len(summaries)}")
    print(f"Supported IRFs (q<0.05, expected sign): {len(sup)}")
    print(f"Suggestive IRFs (p<0.05, expected sign): {len(sug)}")
    for label, items in [("Supported", sup), ("Suggestive", sug)]:
        if items:
            print(f"\n{label}:")
            for s in items:
                print(f"  {s['pair']:<46} beta(h{H_STAR})={s['beta_star']:+.4f} "
                      f"p={s['p_star']:.4g} q={s['q_bh_star']:.4g} "
                      f"peak h={s['peak_horizon']}")


if __name__ == "__main__":
    main()
