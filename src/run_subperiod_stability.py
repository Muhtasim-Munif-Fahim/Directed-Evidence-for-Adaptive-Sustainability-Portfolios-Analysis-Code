"""Sub-period stability of the DH Granger network (structural-break check).

Reviewer concern (Technical report): is the directed architecture a stable
system feature or an artifact of a particular era (MDG->SDG transition, COVID)?
We re-estimate the network on sub-periods and compare with the primary
(2000-2024, K=2, 84 BH-significant edges).

Feasibility arithmetic (T = years in window, t_eff = T - 1 first differences):
  - per-country usable rows = t_eff - K; the protocol guard requires >= 10
    (MIN_COUNTRY_OBS), relaxed to 8 for the short 2015-2024 window (flagged);
  - the DH(2012) finite-sample standardisation needs t_eff - 2K - 5 > 0.
Hence:
  2000-2014 (t_eff=14): K=2 feasible (rows 12, num 5).
  2000-2019 (t_eff=19): K=2 feasible (rows 17, num 10).
  2015-2024 (t_eff=9):  K=2 INFEASIBLE (num = 9-4-5 = 0). K=1 feasible only
                        with the obs guard relaxed to 8 (rows 8, num 2) -- flagged.
  2020-2024 (t_eff=4):  INFEASIBLE at any K (num = 4-2-5 < 0 even for K=1).

Outputs:
  outputs/network/subperiod_stability.json
  outputs/network/subperiod_edges_<label>.csv (one per feasible split)
"""
from __future__ import annotations

import json
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import run_panel_granger_network as net
from run_panel_granger_network import GOALS, GOAL_NAMES, load_balanced_panel
from result_contract import PROTOCOL_VERSION, bh_adjust

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "network"
MIN_COUNTRIES = 30

SPLITS = [
    # label, year_lo, year_hi, K, min_country_obs, note
    ("2000_2014", 2000, 2014, 2, 10, "pre-SDG-adoption era"),
    ("2015_2024", 2015, 2024, 1, 8,
     "SDG era; K=1 with obs guard relaxed 10->8 (T too short for K=2)"),
    ("2000_2019", 2000, 2019, 2, 10, "pre-COVID"),
    ("2020_2024", 2020, 2024, None, None,
     "INFEASIBLE: t_eff=4 < 2K+6 for any K>=1 (DH finite-sample term <= 0)"),
]


def dh_pair(panel: pd.DataFrame, x: str, y: str, k: int, t_eff: int) -> dict:
    """DH test for one pair with subperiod-correct finite-sample standardisation."""
    total = panel["id"].nunique()
    w_list, b_list = [], []
    for _, unit in panel.groupby("id"):
        out = net.country_wald(unit, x, y, k)
        if out is not None:
            w_list.append(out[0])
            b_list.append(out[1])
    n_eff = len(w_list)
    if n_eff < MIN_COUNTRIES:
        return {"source": x, "target": y, "n_countries": n_eff,
                "w_bar": np.nan, "z_bar_tilde": np.nan, "p_raw": np.nan,
                "edge_sign": np.nan, "estimable": False}
    w_bar = float(np.mean(w_list))
    num = t_eff - 2 * k - 5
    den = t_eff - k - 3
    scale = np.sqrt((n_eff / (2 * k)) * (num / den))
    adj = ((t_eff - 2 * k - 3) / (t_eff - 2 * k - 1)) * w_bar - k
    z = float(scale * adj)
    return {"source": x, "target": y, "n_countries": n_eff,
            "w_bar": w_bar, "z_bar_tilde": z,
            "p_raw": float(stats.norm.sf(z)),
            "edge_sign": float(np.median(b_list)), "estimable": True}


def compute_subperiod(panel: pd.DataFrame, k: int, t_eff: int) -> pd.DataFrame:
    rows = [dh_pair(panel, x, y, k, t_eff) for x, y in permutations(GOALS, 2)]
    est = pd.DataFrame(rows)
    est = est[est["estimable"]].copy()
    est = bh_adjust(est)
    est["bh_significant"] = est["q_bh"] < 0.05
    est["direction"] = np.where(est["edge_sign"] >= 0, "synergy", "trade_off")
    est["source_name"] = est["source"].map(GOAL_NAMES)
    est["target_name"] = est["target"].map(GOAL_NAMES)
    return est


def net_influence(sig: pd.DataFrame) -> pd.Series:
    out = sig.groupby("source").size().reindex(GOALS, fill_value=0)
    inn = sig.groupby("target").size().reindex(GOALS, fill_value=0)
    return (out - inn).reindex(GOALS)


def main() -> None:
    panel = load_balanced_panel()
    primary = pd.read_csv(OUT_DIR / "granger_edges.csv")
    psig = primary[primary["bh_significant"]].copy()
    prim_edges = {(r.source, r.target) for r in psig.itertuples()}
    prim_sign = {(r.source, r.target): r.direction for r in psig.itertuples()}
    ni_primary = net_influence(psig)

    results = {}
    for label, lo, hi, k, min_obs, note in SPLITS:
        if k is None:
            results[label] = {"feasible": False, "note": note}
            print(f"\n[{label}] {note}")
            continue
        t_eff = (hi - lo + 1) - 1
        print(f"\n[{label}] K={k}, t_eff={t_eff}, min_obs={min_obs} -- estimating...")
        net.MIN_COUNTRY_OBS = min_obs  # protocol guard (relaxation flagged in note)
        sub = panel[panel["year"].between(lo, hi)]
        est = compute_subperiod(sub, k, t_eff)
        net.MIN_COUNTRY_OBS = 10
        est["protocol_version"] = PROTOCOL_VERSION
        est.sort_values("p_raw").to_csv(
            OUT_DIR / f"subperiod_edges_{label}.csv", index=False)

        sig = est[est["bh_significant"]].copy()
        sub_edges = {(r.source, r.target) for r in sig.itertuples()}
        sub_sign = {(r.source, r.target): r.direction for r in sig.itertuples()}
        shared = prim_edges & sub_edges
        union = prim_edges | sub_edges
        sign_agree = (sum(1 for e in shared if sub_sign[e] == prim_sign[e])
                      / len(shared)) if shared else float("nan")
        rho = float(stats.spearmanr(ni_primary, net_influence(sig)).correlation)
        # power-matched comparison: top-84 pairs by p in the subperiod, ignoring BH
        top84 = set(map(tuple, est.nsmallest(len(prim_edges), "p_raw")
                        [["source", "target"]].itertuples(index=False, name=None)))
        overlap84 = len(prim_edges & top84) / len(prim_edges)

        results[label] = {
            "feasible": True, "note": note, "K": k, "t_eff": t_eff,
            "min_country_obs": min_obs,
            "estimable_pairs": int(len(est)),
            "bh_significant_edges": int(len(sig)),
            "synergy": int((sig["direction"] == "synergy").sum()),
            "trade_off": int((sig["direction"] == "trade_off").sum()),
            "shared_with_primary": len(shared),
            "retained_share_of_primary": round(len(shared) / len(prim_edges), 3),
            "jaccard": round(len(shared) / len(union), 3) if union else None,
            "sign_agreement_on_shared": round(sign_agree, 3),
            "net_influence_spearman_vs_primary": round(rho, 3),
            "top84_pairs_overlap_with_primary": round(overlap84, 3),
            "top10_edges": [
                f"{r.source_name} -> {r.target_name} (z={r.z_bar_tilde:.2f}, "
                f"q={r.q_bh:.3g}, {r.direction})"
                for r in sig.nlargest(10, "z_bar_tilde").itertuples()
            ],
        }
        r = results[label]
        print(f"  edges={r['bh_significant_edges']} "
              f"({r['synergy']} syn/{r['trade_off']} trade)  "
              f"retained={r['retained_share_of_primary']:.2f}  "
              f"sign-agree={r['sign_agreement_on_shared']:.2f}  "
              f"NI-Spearman={r['net_influence_spearman_vs_primary']:.2f}  "
              f"top84-overlap={r['top84_pairs_overlap_with_primary']:.2f}")

    summary = {"protocol_version": PROTOCOL_VERSION,
               "primary_edges": len(prim_edges),
               "splits": results}
    (OUT_DIR / "subperiod_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'subperiod_stability.json'}")


if __name__ == "__main__":
    main()
