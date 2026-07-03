"""Benjamini--Yekutieli (dependence-robust) FDR sensitivity for the directed
Granger network (technical-review item #7).

Reads the frozen per-pair p-values in outputs/network/granger_edges.csv and
re-applies false-discovery control with the BY correction, which is valid under
arbitrary dependence among the 272 tests (shared countries, goals, common
shocks). BH (the primary) is exact only under independence or positive
dependence; BY is the conservative bound. We report how many of the BH edges
survive BY, whether the synergy/trade-off balance and net-degree ranking are
preserved, and write a small JSON summary for the evidence registry.

No regressions are re-estimated: BY is applied to the identical p-value family
that BH used, so this is a pure multiplicity-robustness check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "network"


def net_degree(edges: pd.DataFrame) -> pd.Series:
    out = edges.groupby("source_name").size()
    inn = edges.groupby("target_name").size()
    goals = set(out.index) | set(inn.index)
    return pd.Series({g: int(out.get(g, 0)) - int(inn.get(g, 0)) for g in goals})


def main() -> None:
    df = pd.read_csv(OUT_DIR / "granger_edges.csv")
    est = df[df["p_raw"].notna()].copy()
    est["q_by"] = multipletests(
        est["p_raw"].astype(float).to_numpy(), method="fdr_by"
    )[1]

    bh = est["q_bh"] < 0.05
    by = est["q_by"] < 0.05

    bh_edges = est[bh]
    by_edges = est[by]

    nb = net_degree(bh_edges)
    ny = net_degree(by_edges)
    allg = sorted(set(nb.index) | set(ny.index))
    comp = pd.DataFrame(
        {"BH": [nb.get(g, 0) for g in allg], "BY": [ny.get(g, 0) for g in allg]},
        index=allg,
    )
    rho = float(comp["BH"].corr(comp["BY"], method="spearman"))

    by_dir = by_edges["direction"].value_counts().to_dict()
    summary = {
        "family_size": int(len(est)),
        "bh_edges": int(bh.sum()),
        "by_edges": int(by.sum()),
        "by_subset_of_bh": int((by & bh).sum()),
        "by_only": int((by & ~bh).sum()),
        "bh_share_surviving_by": round(100 * (by & bh).sum() / bh.sum(), 1),
        "bh_density": round(int(bh.sum()) / len(est), 3),
        "by_density": round(int(by.sum()) / len(est), 3),
        "by_synergy": int(by_dir.get("synergy", 0)),
        "by_tradeoff": int(by_dir.get("trade_off", 0)),
        "netdegree_spearman_bh_vs_by": round(rho, 3),
    }

    # Persist the augmented edge table and the summary.
    est.sort_values("p_raw").to_csv(OUT_DIR / "granger_edges_by.csv", index=False)
    with open(OUT_DIR / "by_fdr_sensitivity.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
