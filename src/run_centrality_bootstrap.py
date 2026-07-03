"""Net-influence robustness: effect-size weighting + country bootstrap.

Reviewer concern (Technical #12, and our own lag finding): net influence by raw
out/in DEGREE treats a marginal edge like a strong one and is unstable to lag
choice and CSD treatment. Here we:

  1. recompute net influence with an EFFECT-SIZE weight (each edge weighted by the
     absolute cross-country median of sum_k beta, i.e. its magnitude in SDR points),
     and test whether the WEIGHTED ranking is lag-stable (Spearman across K=1,2,3);
  2. run a country bootstrap (resample countries with replacement, recompute the
     FDR-screened network and net influence) to report, for each goal, the
     probability it is a net driver and a top-tier driver.

Per-country Wald statistics are cached once per K so the bootstrap only re-aggregates
(fast). Read-only on the primary pipeline.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_panel_granger_network import (
    GOAL_NAMES, GOALS, country_wald, load_balanced_panel,
)

NET = Path(__file__).resolve().parents[1] / "outputs" / "network"
B = 500
T_EFF = 24


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = (p[order] * m / np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def cache_stats(panel: pd.DataFrame, k: int) -> dict:
    """For each ordered pair, {country_id: (W_i, b_sum)}."""
    cache = {}
    units = {cid: u for cid, u in panel.groupby("id")}
    for x, y in permutations(GOALS, 2):
        d = {}
        for cid, u in units.items():
            o = country_wald(u, x, y, k)
            if o is not None:
                d[cid] = o
        cache[(x, y)] = d
    return cache


def dh_z(w_list: list[float], n_eff: int, k: int) -> float:
    w_bar = float(np.mean(w_list))
    scale = np.sqrt((n_eff / (2 * k)) * ((T_EFF - 2 * k - 5) / (T_EFF - k - 3)))
    adj = ((T_EFF - 2 * k - 3) / (T_EFF - 2 * k - 1)) * w_bar - k
    return float(scale * adj)


def build_network(cache_k: dict, k: int, country_counts: Counter, sd: dict) -> pd.DataFrame:
    rows = []
    for (x, y), d in cache_k.items():
        ws, bs = [], []
        for cid, mult in country_counts.items():
            if cid in d:
                ws.extend([d[cid][0]] * mult)
                bs.extend([d[cid][1]] * mult)
        if len(ws) < 30:
            continue
        z = dh_z(ws, len(ws), k)
        rows.append((x, y, z, float(stats.norm.sf(z)), float(np.median(bs))))
    df = pd.DataFrame(rows, columns=["source", "target", "z", "p", "sign"])
    df["q"] = bh(df["p"].values)
    df["sig"] = df["q"] < 0.05
    # standardized effect-size weight: coefficient on z-scored first differences,
    # |median sum-beta| * SD(d source)/SD(d target), comparable across goals.
    df["w"] = df["sign"].abs() * df["source"].map(sd) / df["target"].map(sd)
    return df


def net_degree(df: pd.DataFrame) -> pd.Series:
    s = df[df["sig"]]
    out = s.groupby("source").size().reindex(GOALS, fill_value=0)
    inn = s.groupby("target").size().reindex(GOALS, fill_value=0)
    return (out - inn).reindex(GOALS)


def net_weight(df: pd.DataFrame) -> pd.Series:
    s = df[df["sig"]]
    out = s.groupby("source")["w"].sum().reindex(GOALS, fill_value=0.0)
    inn = s.groupby("target")["w"].sum().reindex(GOALS, fill_value=0.0)
    return (out - inn).reindex(GOALS)


def main() -> None:
    panel = load_balanced_panel()
    countries = sorted(panel["id"].unique())
    full = Counter(countries)

    # pooled SD of each goal's first differences (for standardized effect sizes)
    dpanel = panel.sort_values(["id", "year"])
    sd = {g: float(dpanel.groupby("id")[g].diff().std()) for g in GOALS}

    print("caching per-country Wald stats for K=1,2,3 ...")
    caches = {k: cache_stats(panel, k) for k in (1, 2, 3)}

    # --- lag stability: degree vs standardized effect-size-weighted ---
    deg = {k: net_degree(build_network(caches[k], k, full, sd)) for k in (1, 2, 3)}
    wt = {k: net_weight(build_network(caches[k], k, full, sd)) for k in (1, 2, 3)}
    sp = lambda a, b: round(float(stats.spearmanr(a, b).correlation), 3)
    lag = {
        "degree_spearman_K2_vs_K1": sp(deg[2], deg[1]),
        "degree_spearman_K2_vs_K3": sp(deg[2], deg[3]),
        "weighted_spearman_K2_vs_K1": sp(wt[2], wt[1]),
        "weighted_spearman_K2_vs_K3": sp(wt[2], wt[3]),
    }

    # --- country bootstrap at K=2 (weighted net influence) ---
    print(f"bootstrapping K=2 network, B={B} ...")
    rng = np.random.default_rng(0)
    driver = Counter()
    top5 = Counter()
    vals = {g: [] for g in GOALS}
    for _ in range(B):
        samp = Counter(rng.choice(countries, size=len(countries), replace=True))
        nw = net_weight(build_network(caches[2], 2, samp, sd))
        ranked = nw.sort_values(ascending=False)
        t5 = set(ranked.head(5).index)
        for g in GOALS:
            vals[g].append(float(nw[g]))
            if nw[g] > 0:
                driver[g] += 1
        for g in t5:
            top5[g] += 1

    boot = {}
    for g in GOALS:
        arr = np.array(vals[g])
        boot[GOAL_NAMES[g]] = {
            "p_net_driver": round(driver[g] / B, 3),
            "p_top5_driver": round(top5[g] / B, 3),
            "weighted_net_median": round(float(np.median(arr)), 2),
            "ci90": [round(float(np.percentile(arr, 5)), 2),
                     round(float(np.percentile(arr, 95)), 2)],
        }

    summary = {"lag_stability": lag,
               "bootstrap_B": B,
               "bootstrap_by_goal": boot}
    NET.joinpath("network_centrality_bootstrap.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(lag, indent=2))
    print("\nGoal: P(net driver) | P(top-5) | weighted net [90% CI]")
    for g in sorted(GOALS, key=lambda g: -boot[GOAL_NAMES[g]]["p_net_driver"]):
        b = boot[GOAL_NAMES[g]]
        print(f"  {GOAL_NAMES[g]:<26} {b['p_net_driver']:.2f}  {b['p_top5_driver']:.2f}  "
              f"{b['weighted_net_median']:+.2f} [{b['ci90'][0]:+.2f},{b['ci90'][1]:+.2f}]")


if __name__ == "__main__":
    main()
