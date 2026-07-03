"""Pesaran (2007) CIPS panel unit-root test for the 17 SDG goal composites.

Justifies the first-differencing used by the Granger network and local projections.
For each goal we run, per country i, the cross-sectionally augmented Dickey-Fuller
(CADF) regression (intercept case, one augmentation lag):

  dy_{it} = a_i + b_i y_{i,t-1} + c_i ybar_{t-1} + d_i dybar_t
            + e_i dy_{i,t-1} + f_i dybar_{t-1} + u_{it}

where ybar_t is the cross-sectional mean. The CADF_i statistic is the t-ratio on
b_i; the panel CIPS statistic is the (truncated) cross-country average. We report
CIPS for the goal composites in levels and in first differences. The 5% critical
value comes from Pesaran (2007) Table II(b), intercept-without-trend case: for
this panel (N=114, T=25) the tabulated values bracket the sample at -2.20
(T=20) and -2.16 (T=30), N-rows >= 100 (values nearly invariant in N); linear
interpolation at T=25 gives -2.18. CIPS above it fails to reject a unit root;
below it rejects. (The previously used -2.11 is the large-T value, T~50-70,
and was flagged in review as inappropriate for T=25.)

Output: outputs/network/cips_unit_root.csv, cips_summary.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
DATA_FILE = ROOT / "data" / "raw" / "SDR2025-data.xlsx"
OUT = ROOT / "outputs" / "network"

GOALS = [f"goal{i}" for i in range(1, 18)]
GOAL_NAMES = {
    "goal1": "No Poverty", "goal2": "Zero Hunger", "goal3": "Health",
    "goal4": "Education", "goal5": "Gender Equality", "goal6": "Clean Water",
    "goal7": "Clean Energy", "goal8": "Decent Work/Growth", "goal9": "Industry/Innovation",
    "goal10": "Reduced Inequality", "goal11": "Sustainable Cities",
    "goal12": "Responsible Consumption", "goal13": "Climate Action",
    "goal14": "Life Below Water", "goal15": "Life on Land",
    "goal16": "Peace/Institutions", "goal17": "Partnerships",
}

# Pesaran (2007) Table II(b), intercept case, 5% level. Tabulated: -2.20 at
# (N>=100, T=20), -2.16 at (N>=100, T=30); linear interpolation at T=25 for
# this N=114 panel -> -2.18. Do NOT use the large-T value (-2.11, T~50-70).
CV_5PCT = -2.18
TRUNC_LO, TRUNC_HI = -6.19, 2.61  # CADF truncation bounds (intercept case)


def load_balanced_panel() -> pd.DataFrame:
    frame = pd.read_excel(
        DATA_FILE, sheet_name="Backdated SDG Index", usecols=["id", "year", *GOALS]
    )
    frame = frame[~frame["id"].astype(str).str.startswith("_")].copy()
    frame = frame[frame["year"].between(2000, 2024)]
    complete = frame.dropna(subset=GOALS)
    counts = complete.groupby("id")["year"].nunique()
    ids = counts[counts == 25].index
    return complete[complete["id"].isin(ids)].sort_values(["id", "year"])


def cadf_tstat(y: np.ndarray, ybar: np.ndarray) -> float | None:
    """CADF t-ratio on the lagged level for one country series (intercept, P=1)."""
    dy = np.diff(y)
    dybar = np.diff(ybar)
    y_lag = y[:-1]
    ybar_lag = ybar[:-1]
    # align with one augmentation lag of dy/dybar
    n = len(dy)
    dep = dy[1:]
    X = np.column_stack([
        np.ones(n - 1),
        y_lag[1:],          # b_i : coefficient of interest
        ybar_lag[1:],
        dybar[1:],
        dy[:-1],            # dy_{t-1}
        dybar[:-1],         # dybar_{t-1}
    ])
    if X.shape[0] < X.shape[1] + 2 or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    try:
        res = sm.OLS(dep, X).fit()
        t = float(res.tvalues[1])
        if not np.isfinite(t):
            return None
        return float(np.clip(t, TRUNC_LO, TRUNC_HI))
    except Exception:
        return None


def cips_for_variable(wide: pd.DataFrame) -> tuple[float, int]:
    """wide: index=year, columns=country, values=series. Returns (CIPS, N_used)."""
    ybar = wide.mean(axis=1).to_numpy()
    stats = []
    for col in wide.columns:
        y = wide[col].to_numpy()
        if np.isnan(y).any():
            continue
        t = cadf_tstat(y, ybar)
        if t is not None:
            stats.append(t)
    if len(stats) < 30:
        return np.nan, len(stats)
    return float(np.mean(stats)), len(stats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_balanced_panel()
    print(f"Panel: {panel['id'].nunique()} countries x 25 years")

    rows = []
    for g in GOALS:
        wide = panel.pivot(index="year", columns="id", values=g)
        cips_lvl, n_lvl = cips_for_variable(wide)
        cips_dif, n_dif = cips_for_variable(wide.diff().iloc[1:])
        rows.append({
            "goal": g, "goal_name": GOAL_NAMES[g],
            "cips_level": round(cips_lvl, 3), "cips_diff": round(cips_dif, 3),
            "level_unit_root_not_rejected": bool(cips_lvl > CV_5PCT),
            "diff_stationary": bool(cips_dif < CV_5PCT),
            "n_countries": n_lvl,
        })
        print(f"  {GOAL_NAMES[g]:<26} CIPS level={cips_lvl:6.2f}  diff={cips_dif:6.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cips_unit_root.csv", index=False)

    n_level_ur = int(df["level_unit_root_not_rejected"].sum())
    n_diff_stat = int(df["diff_stationary"].sum())
    summary = {
        "method": "Pesaran (2007) CIPS panel unit-root test (intercept, 1 lag)",
        "critical_value_5pct": CV_5PCT,
        "critical_value_source": (
            "Pesaran (2007) Table II(b), intercept case, 5% level; linear "
            "interpolation at T=25 between tabulated -2.20 (T=20) and -2.16 "
            "(T=30), N rows >= 100 (values nearly invariant in N)."
        ),
        "n_goals": len(GOALS),
        "goals_unit_root_in_levels": n_level_ur,
        "goals_stationary_in_differences": n_diff_stat,
        "conclusion": (
            f"In levels, the unit-root null is not rejected at 5% for "
            f"{n_level_ur}/{len(GOALS)} goal composites; in first differences it is "
            f"rejected for {n_diff_stat}/{len(GOALS)}. The composites are treated as "
            f"I(1) and all causal inference is conducted on first differences."
        ),
    }
    (OUT / "cips_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Levels: unit root not rejected for {n_level_ur}/{len(GOALS)} goals")
    print(f"Diffs:  stationary for {n_diff_stat}/{len(GOALS)} goals")


if __name__ == "__main__":
    main()
