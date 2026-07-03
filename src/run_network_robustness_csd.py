"""Cross-sectional-dependence (CSD) robustness for the DH network.

Addresses the convergent reviewer concern: the primary network uses only
per-country intercepts, whereas the local projections use time fixed effects.
We re-estimate the one-sided DH network at K=2 under two CSD controls and
compare to the primary:

  * DEMEAN : year-demean the first differences (time-fixed-effect equivalent;
             homogeneous unit factor loading).
  * CCE    : augment each country regression with contemporaneous cross-sectional
             averages of the differenced regressand and regressor (Pesaran 2006;
             heterogeneous, country-specific factor loadings).

We report, for each control: edge count, synergy/trade-off split, the
driver-receiver net-influence rank correlation vs the primary, and the size of
the "robust core" (edges significant in BOTH primary and the control). We also
report the Pesaran (2004) CD statistic on the differenced goal series before and
after year-demeaning. Read-only with respect to the primary pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_panel_granger_network import (
    GOAL_NAMES, GOALS, K_LAGS, compute_edges, load_balanced_panel,
)

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "outputs" / "network"


def pesaran_cd(wide: pd.DataFrame) -> tuple[float, float]:
    corr = wide.corr()
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    rho = corr.values[iu]
    rho = rho[np.isfinite(rho)]
    t = wide.shape[0]
    cd = np.sqrt(2.0 * t / (n * (n - 1))) * rho.sum()
    return float(cd), float(2.0 * stats.norm.sf(abs(cd)))


def _diff(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.sort_values(["id", "year"]).copy()
    for g in GOALS:
        d[g] = d.groupby("id")[g].diff()
    return d


def _year_demean_diff(panel: pd.DataFrame) -> pd.DataFrame:
    d = _diff(panel)
    for g in GOALS:
        d[g] = d[g] - d.groupby("year")[g].transform("mean")
    return d


def _diff_with_csa(panel: pd.DataFrame) -> pd.DataFrame:
    """Differenced panel plus contemporaneous cross-sectional averages per goal."""
    d = _diff(panel)
    for g in GOALS:
        d[f"{g}_csa"] = d.groupby("year")[g].transform("mean")
    return d


def _net_influence(edges: pd.DataFrame) -> pd.Series:
    sig = edges[edges["bh_significant"]]
    out_deg = sig.groupby("source").size().reindex(GOALS, fill_value=0)
    in_deg = sig.groupby("target").size().reindex(GOALS, fill_value=0)
    s = (out_deg - in_deg).reindex(GOALS)
    s.index = [GOAL_NAMES[g] for g in GOALS]
    return s


def _edge_set(edges: pd.DataFrame) -> set[tuple[str, str]]:
    sig = edges[edges["bh_significant"]]
    return set(map(tuple, sig[["source", "target"]].values))


def main() -> None:
    panel = load_balanced_panel()
    names = [GOAL_NAMES[g] for g in GOALS]

    # --- primary (one-sided) network from canonical outputs ---
    prim_edges = pd.read_csv(NET / "granger_edges.csv")
    prim_set = set(map(tuple, prim_edges[prim_edges["bh_significant"]][["source", "target"]].values))
    prim_cent = pd.read_csv(NET / "granger_centrality.csv").set_index("goal_name")
    prim_net = prim_cent["net_influence"].reindex(names)
    prim_sig = len(prim_set)

    # --- Pesaran CD before/after year-demeaning ---
    draw = _diff(panel)
    ddem = _year_demean_diff(panel)
    cd_rows = []
    for g in GOALS:
        wb = draw.pivot(index="year", columns="id", values=g).dropna(how="all")
        wa = ddem.pivot(index="year", columns="id", values=g).dropna(how="all")
        cb, pb = pesaran_cd(wb)
        ca, pa = pesaran_cd(wa)
        cd_rows.append({"goal": GOAL_NAMES[g], "cd_before": round(cb, 1),
                        "p_before": pb, "cd_after": round(ca, 1), "p_after": pa})
    cd_df = pd.DataFrame(cd_rows)

    # --- DEMEAN network ---
    print("\nEstimating year-demeaned (time-FE) network (K=2)...")
    dem_edges = compute_edges(ddem, K_LAGS, already_differenced=True)
    dem_set = _edge_set(dem_edges)
    dem_net = _net_influence(dem_edges)

    # --- CCE network ---
    print("Estimating CCE-augmented network (K=2)...")
    dcsa = _diff_with_csa(panel)
    cce_edges = compute_edges(dcsa, K_LAGS, already_differenced=True, cce=True)
    cce_set = _edge_set(cce_edges)
    cce_net = _net_influence(cce_edges)

    def split(edges):
        sig = edges[edges["bh_significant"]]
        return (int((sig["direction"] == "synergy").sum()),
                int((sig["direction"] == "trade_off").sum()))

    summary = {
        "method": "CSD robustness for one-sided DH network, K=2",
        "primary": {"edges": prim_sig,
                    "synergy": int((prim_edges[prim_edges.bh_significant].direction == "synergy").sum()),
                    "tradeoff": int((prim_edges[prim_edges.bh_significant].direction == "trade_off").sum())},
        "demean": {"edges": len(dem_set), "synergy": split(dem_edges)[0],
                   "tradeoff": split(dem_edges)[1],
                   "spearman_vs_primary": round(float(stats.spearmanr(prim_net, dem_net).correlation), 3),
                   "robust_core_with_primary": len(prim_set & dem_set)},
        "cce": {"edges": len(cce_set), "synergy": split(cce_edges)[0],
                "tradeoff": split(cce_edges)[1],
                "spearman_vs_primary": round(float(stats.spearmanr(prim_net, cce_net).correlation), 3),
                "robust_core_with_primary": len(prim_set & cce_set)},
        "robust_core_all_three": len(prim_set & dem_set & cce_set),
        "cd_mean_abs_before": round(float(cd_df["cd_before"].abs().mean()), 1),
        "cd_mean_abs_after": round(float(cd_df["cd_after"].abs().mean()), 1),
        "cd_share_sig_before": round(float((cd_df["p_before"] < 0.05).mean()), 3),
        "cd_share_sig_after": round(float((cd_df["p_after"] < 0.05).mean()), 3),
        "net_influence_table": [
            {"goal": nm, "primary": int(prim_net[nm]), "demean": int(dem_net[nm]),
             "cce": int(cce_net[nm])} for nm in names
        ],
        "cd_by_goal": cd_rows,
    }
    NET.joinpath("network_csd_robustness.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("net_influence_table", "cd_by_goal")}, indent=2))
    print("\nNet influence  primary | demean | CCE  (sorted by CCE):")
    tbl = sorted(summary["net_influence_table"], key=lambda r: r["cce"], reverse=True)
    for r in tbl:
        print(f"  {r['goal']:<26} {r['primary']:+d}   {r['demean']:+d}   {r['cce']:+d}")


if __name__ == "__main__":
    main()
