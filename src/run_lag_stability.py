"""Lag-order robustness for the DH network.

Two questions reviewers raised about the fixed K=2 choice:
  1. Is the *driver-receiver ranking* (not just the edge set) stable across lags?
     -> Spearman correlation of goal net influence at K=1 and K=3 vs the K=2 primary.
  2. Is K=2 defensible on information criteria? -> for every estimable
     country x ordered-pair regression we fit K=1,2,3 on a common sample and record
     the AIC- and BIC-preferred lag, then report the distribution.

Writes outputs/network/network_lag_stability.json. Read-only on the primary.
"""
from __future__ import annotations

import collections
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from run_panel_granger_network import (
    GOALS, MIN_COUNTRY_OBS, compute_edges, load_balanced_panel,
)

NET = Path(__file__).resolve().parents[1] / "outputs" / "network"


def net_influence(edges: pd.DataFrame) -> pd.Series:
    sig = edges[edges["bh_significant"]]
    out_deg = sig.groupby("source").size().reindex(GOALS, fill_value=0)
    in_deg = sig.groupby("target").size().reindex(GOALS, fill_value=0)
    return (out_deg - in_deg).reindex(GOALS)


def aic_bic_for_unit(unit: pd.DataFrame, x: str, y: str, kmax: int = 3):
    """AIC/BIC at K=1..kmax on a common sample (fair comparison)."""
    df = unit.sort_values("year")[[y, x]]
    dy, dx = df[y].diff(), df[x].diff()
    cols = {}
    for lag in range(1, kmax + 1):
        cols[f"y{lag}"] = dy.shift(lag).values
        cols[f"x{lag}"] = dx.shift(lag).values
    full = pd.DataFrame(cols)
    full["y"] = dy.values
    full = full.dropna()
    if len(full) < MIN_COUNTRY_OBS:
        return None
    yv = full["y"]
    out = {}
    for k in range(1, kmax + 1):
        xcols = [f"y{l}" for l in range(1, k + 1)] + [f"x{l}" for l in range(1, k + 1)]
        X = sm.add_constant(full[xcols])
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            return None
        r = sm.OLS(yv, X).fit()
        out[k] = (r.aic, r.bic)
    return out


def main() -> None:
    panel = load_balanced_panel()

    # --- 1. driver-receiver ranking stability across K ---
    print("computing networks at K=1,2,3 ...")
    net = {k: net_influence(compute_edges(panel, k)) for k in (1, 2, 3)}
    rho1 = float(stats.spearmanr(net[2], net[1]).correlation)
    rho3 = float(stats.spearmanr(net[2], net[3]).correlation)

    # --- 2. AIC/BIC-preferred lag distribution ---
    print("computing AIC/BIC lag selection over country x pair regressions ...")
    aic_sel, bic_sel = [], []
    for x, y in permutations(GOALS, 2):
        for _, unit in panel.groupby("id"):
            o = aic_bic_for_unit(unit, x, y)
            if o is None:
                continue
            ks = sorted(o)
            aic_sel.append(min(ks, key=lambda k: o[k][0]))
            bic_sel.append(min(ks, key=lambda k: o[k][1]))
    n = len(aic_sel)
    aic_d = collections.Counter(aic_sel)
    bic_d = collections.Counter(bic_sel)
    share = lambda d, k: round(100 * d.get(k, 0) / n, 1)

    summary = {
        "net_influence_spearman_K2_vs_K1": round(rho1, 3),
        "net_influence_spearman_K2_vs_K3": round(rho3, 3),
        "n_unit_regressions": n,
        "aic_share_pct": {f"K={k}": share(aic_d, k) for k in (1, 2, 3)},
        "bic_share_pct": {f"K={k}": share(bic_d, k) for k in (1, 2, 3)},
        "aic_median_lag": int(np.median(aic_sel)),
        "bic_median_lag": int(np.median(bic_sel)),
    }
    NET.joinpath("network_lag_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
