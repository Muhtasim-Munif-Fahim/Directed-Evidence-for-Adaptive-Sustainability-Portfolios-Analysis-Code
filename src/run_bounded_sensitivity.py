"""Bounded-score (ceiling/floor) sensitivity of the SDG DH-Granger network.

Reviewer concern (robustness): SDR goal scores are bounded on [0, 100]. Near the
ceiling, a country that is already "maxed out" on goal x can only move down or stay,
which could *mechanically* manufacture negative (trade-off) co-movement with another
goal that is still improving -- inflating the count of trade-off edges. We address
this in two read-only steps that DO NOT modify the primary pipeline.

  1. CEILING AUDIT. For each goal, the share of country-year scores above 90 and
     above 95 in the balanced panel. We then ask whether the 44 trade-off edges are
     concentrated on goals with high ceiling shares, by comparing the mean ceiling
     share of the goals involved in trade-off edges vs. synergy edges.

  2. LOGIT RE-ESTIMATION. The bounded score is mapped to an unbounded scale via the
     logit transform: p = clip(score/100, 0.01, 0.99); x = log(p / (1 - p)). On this
     scale the ceiling no longer compresses variance, so any ceiling-driven mechanical
     trade-off should weaken. We re-run the FULL Dumitrescu-Hurlin network at K=2 on
     the logit panel (first differences of logit scores), reusing the SAME per-country
     Wald machinery, the SAME DH(2012) Z-bar-tilde standardization (T_EFF=24, one-sided
     upper-tail p), the SAME min-30-country guard, and the SAME BH q<0.05 screen. We
     compare against the primary network: edges retained/lost (Jaccard), sign agreement
     among shared edges, synergy/trade-off counts, the change in trade-off share, and
     the Spearman correlation of net influence (out-degree minus in-degree).

Reuse (imported, never re-implemented): GOALS, GOAL_NAMES, load_balanced_panel,
country_wald, MIN_COUNTRY_OBS from run_panel_granger_network; bh_adjust from
result_contract. The logit panel is first-differenced here and fed to country_wald
with already_differenced=True so the exact same regression/guards/Wald path is used.

Outputs (read-only; new files only):
  outputs/network/bounded_sensitivity.json   ceiling audit + logit-vs-primary comparison
  outputs/network/bounded_edges.csv          full logit-network edge list (one row/pair)
"""
from __future__ import annotations

import json
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_panel_granger_network import (
    GOALS, GOAL_NAMES, country_wald, load_balanced_panel,
)
from result_contract import PROTOCOL_VERSION, bh_adjust

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "network"
PRIMARY_EDGES = OUT_DIR / "granger_edges.csv"

K_LAGS = 2
T_EFF = 24            # effective length after first-differencing (25 - 1)
MIN_COUNTRIES = 30    # DH estimability guard, identical to the primary pipeline


# --------------------------------------------------------------------------- #
# Step 1: ceiling audit
# --------------------------------------------------------------------------- #
def ceiling_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-goal share of country-year scores above 90 and above 95."""
    rows = []
    for g in GOALS:
        s = panel[g].to_numpy(dtype=float)
        rows.append({
            "goal": g,
            "goal_name": GOAL_NAMES[g],
            "n_obs": int(s.size),
            "mean_score": round(float(np.mean(s)), 2),
            "share_gt90": round(float(np.mean(s > 90)), 4),
            "share_gt95": round(float(np.mean(s > 95)), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2: logit-transformed DH network
# --------------------------------------------------------------------------- #
def logit_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Map each bounded goal score to the unbounded logit scale.

    p = clip(score / 100, 0.01, 0.99); x = log(p / (1 - p)).
    The clip keeps the transform finite for the rare exact 0 / 100 scores.
    """
    out = panel[["id", "year"]].copy()
    for g in GOALS:
        p = np.clip(panel[g].to_numpy(dtype=float) / 100.0, 0.01, 0.99)
        out[g] = np.log(p / (1.0 - p))
    return out.sort_values(["id", "year"])


def dh_z(w_bar: float, n_eff: int, k: int) -> float:
    """DH(2012) finite-sample standardized Z-bar-tilde (matches primary code)."""
    scale = np.sqrt((n_eff / (2 * k)) * ((T_EFF - 2 * k - 5) / (T_EFF - k - 3)))
    adj = ((T_EFF - 2 * k - 3) / (T_EFF - 2 * k - 1)) * w_bar - k
    return float(scale * adj)


def dumitrescu_hurlin_logit(panel_d: pd.DataFrame, x: str, y: str, k: int) -> dict:
    """DH non-causality x -> y on a first-differenced logit panel.

    panel_d already holds first differences of the logit scores, so we pass
    already_differenced=True to reuse the exact primary country_wald path
    (variance floor, rank/condition guards, Wald>500 drop, MIN_COUNTRY_OBS).
    """
    total = panel_d["id"].nunique()
    w_list, b_list = [], []
    for _, unit in panel_d.groupby("id"):
        out = country_wald(unit, x, y, k, already_differenced=True)
        if out is not None:
            w_list.append(out[0])
            b_list.append(out[1])

    n_eff = len(w_list)
    if n_eff < MIN_COUNTRIES:
        return {"source": x, "target": y, "n_countries": n_eff,
                "n_excluded": total - n_eff, "w_bar": np.nan,
                "z_bar_tilde": np.nan, "p_raw": np.nan,
                "edge_sign": np.nan, "estimable": False}

    w_bar = float(np.mean(w_list))
    z_tilde = dh_z(w_bar, n_eff, k)
    p_raw = float(stats.norm.sf(z_tilde))      # one-sided upper tail
    return {
        "source": x, "target": y, "n_countries": n_eff,
        "n_excluded": total - n_eff, "w_bar": w_bar,
        "z_bar_tilde": z_tilde, "p_raw": p_raw,
        "edge_sign": float(np.median(b_list)), "estimable": True,
    }


def compute_logit_edges(panel: pd.DataFrame, k: int) -> pd.DataFrame:
    """Full DH pass over all directed pairs on the logit panel, BH-screened."""
    lp = logit_panel(panel)
    # first-difference the logit scores once, per country
    diff = lp.copy()
    diff[GOALS] = lp.groupby("id")[GOALS].diff()

    rows = [dumitrescu_hurlin_logit(diff, x, y, k)
            for x, y in permutations(GOALS, 2)]
    est = pd.DataFrame(rows)
    est = est[est["estimable"]].copy()
    est = bh_adjust(est)
    est["bh_significant"] = est["q_bh"] < 0.05
    est["direction"] = np.where(est["edge_sign"] >= 0, "synergy", "trade_off")
    est["source_name"] = est["source"].map(GOAL_NAMES)
    est["target_name"] = est["target"].map(GOAL_NAMES)
    return est


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def net_influence(edges_sig: pd.DataFrame) -> pd.Series:
    """out-degree minus in-degree per goal over BH-significant edges."""
    out = edges_sig.groupby("source").size().reindex(GOALS, fill_value=0)
    inn = edges_sig.groupby("target").size().reindex(GOALS, fill_value=0)
    return (out - inn).reindex(GOALS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_balanced_panel()

    # ---- Step 1: ceiling audit ------------------------------------------- #
    audit = ceiling_audit(panel)
    share90 = dict(zip(audit["goal"], audit["share_gt90"]))
    share95 = dict(zip(audit["goal"], audit["share_gt95"]))

    primary = pd.read_csv(PRIMARY_EDGES)
    psig = primary[primary["bh_significant"]].copy()
    p_trade = psig[psig["direction"] == "trade_off"]
    p_syn = psig[psig["direction"] == "synergy"]

    def goal_set(df: pd.DataFrame) -> set[str]:
        return set(df["source"]).union(set(df["target"]))

    trade_goals = goal_set(p_trade)
    syn_goals = goal_set(p_syn)

    def mean_share(goals: set[str], tbl: dict) -> float:
        return round(float(np.mean([tbl[g] for g in goals])), 4) if goals else float("nan")

    # Per-edge mean ceiling share (average of source & target shares), so the
    # comparison is at the edge level, not just the distinct-goal level.
    def edge_mean_share(df: pd.DataFrame, tbl: dict) -> float:
        if df.empty:
            return float("nan")
        vals = [(tbl[s] + tbl[t]) / 2.0
                for s, t in zip(df["source"], df["target"])]
        return round(float(np.mean(vals)), 4)

    ceiling_summary = {
        "by_goal": audit.sort_values("share_gt90", ascending=False)
                        .to_dict("records"),
        "trade_off_vs_synergy": {
            "distinct_goal_level": {
                "trade_off_mean_share_gt90": mean_share(trade_goals, share90),
                "synergy_mean_share_gt90": mean_share(syn_goals, share90),
                "trade_off_mean_share_gt95": mean_share(trade_goals, share95),
                "synergy_mean_share_gt95": mean_share(syn_goals, share95),
            },
            "edge_level": {
                "trade_off_mean_share_gt90": edge_mean_share(p_trade, share90),
                "synergy_mean_share_gt90": edge_mean_share(p_syn, share90),
                "trade_off_mean_share_gt95": edge_mean_share(p_trade, share95),
                "synergy_mean_share_gt95": edge_mean_share(p_syn, share95),
            },
        },
    }

    # ---- Step 2: logit network ------------------------------------------- #
    print(f"\nRe-estimating DH network (K={K_LAGS}) on logit-transformed panel...")
    logit_est = compute_logit_edges(panel, K_LAGS)
    logit_est["protocol_version"] = PROTOCOL_VERSION
    logit_est = logit_est.sort_values("p_raw")
    logit_est.to_csv(OUT_DIR / "bounded_edges.csv", index=False)

    lsig = logit_est[logit_est["bh_significant"]].copy()

    # edge-set comparison
    primary_edges = {(r.source, r.target) for r in psig.itertuples()}
    logit_edges = {(r.source, r.target) for r in lsig.itertuples()}
    shared = primary_edges & logit_edges
    union = primary_edges | logit_edges
    jaccard = len(shared) / len(union) if union else float("nan")

    primary_sign = {(r.source, r.target): r.direction for r in psig.itertuples()}
    logit_sign = {(r.source, r.target): r.direction for r in lsig.itertuples()}
    sign_agree = (sum(1 for ed in shared if logit_sign[ed] == primary_sign[ed])
                  / len(shared)) if shared else float("nan")

    n_syn_p = int((psig["direction"] == "synergy").sum())
    n_trade_p = int((psig["direction"] == "trade_off").sum())
    n_syn_l = int((lsig["direction"] == "synergy").sum())
    n_trade_l = int((lsig["direction"] == "trade_off").sum())
    trade_share_p = n_trade_p / len(psig) if len(psig) else float("nan")
    trade_share_l = n_trade_l / len(lsig) if len(lsig) else float("nan")

    # net-influence Spearman
    ni_p = net_influence(psig)
    ni_l = net_influence(lsig)
    ni_spearman = round(float(stats.spearmanr(ni_p.values, ni_l.values).correlation), 3)

    comparison = {
        "primary": {
            "n_edges": int(len(psig)),
            "synergy": n_syn_p, "trade_off": n_trade_p,
            "trade_off_share": round(trade_share_p, 3),
        },
        "logit": {
            "n_edges": int(len(lsig)),
            "synergy": n_syn_l, "trade_off": n_trade_l,
            "trade_off_share": round(trade_share_l, 3),
            "estimable_pairs": int(len(logit_est)),
        },
        "edges_retained": len(shared),
        "edges_lost_from_primary": len(primary_edges - logit_edges),
        "edges_new_in_logit": len(logit_edges - primary_edges),
        "jaccard": round(jaccard, 3),
        "retained_share_of_primary": round(len(shared) / len(primary_edges), 3)
        if primary_edges else float("nan"),
        "sign_agreement_on_shared": round(sign_agree, 3),
        "trade_off_share_change": round(trade_share_l - trade_share_p, 3),
        "net_influence_spearman": ni_spearman,
    }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "Bounded-score (ceiling/floor) sensitivity of DH(2012) Granger network",
        "transform": "logit: p=clip(score/100, 0.01, 0.99); x=log(p/(1-p))",
        "lag_order_K": K_LAGS,
        "ceiling_audit": ceiling_summary,
        "logit_vs_primary": comparison,
    }
    (OUT_DIR / "bounded_sensitivity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ---- readable summary ------------------------------------------------ #
    print("\n" + "=" * 68)
    print("CEILING AUDIT (share of country-year scores above threshold)")
    print(f"{'goal':<26}{'mean':>7}{'>90':>9}{'>95':>9}")
    for r in audit.sort_values("share_gt90", ascending=False).itertuples():
        print(f"  {r.goal_name:<24}{r.mean_score:>7.1f}{r.share_gt90:>9.3f}{r.share_gt95:>9.3f}")

    dl = ceiling_summary["trade_off_vs_synergy"]["distinct_goal_level"]
    el = ceiling_summary["trade_off_vs_synergy"]["edge_level"]
    print("\nMean ceiling share -- trade-off vs synergy edges:")
    print(f"  distinct-goal  >90: trade={dl['trade_off_mean_share_gt90']:.3f}  "
          f"synergy={dl['synergy_mean_share_gt90']:.3f}")
    print(f"  distinct-goal  >95: trade={dl['trade_off_mean_share_gt95']:.3f}  "
          f"synergy={dl['synergy_mean_share_gt95']:.3f}")
    print(f"  edge-level     >90: trade={el['trade_off_mean_share_gt90']:.3f}  "
          f"synergy={el['synergy_mean_share_gt90']:.3f}")
    print(f"  edge-level     >95: trade={el['trade_off_mean_share_gt95']:.3f}  "
          f"synergy={el['synergy_mean_share_gt95']:.3f}")

    c = comparison
    print("\n" + "=" * 68)
    print("LOGIT NETWORK vs PRIMARY")
    print(f"  primary edges : {c['primary']['n_edges']}  "
          f"({c['primary']['synergy']} syn / {c['primary']['trade_off']} trade, "
          f"trade share {c['primary']['trade_off_share']:.3f})")
    print(f"  logit edges   : {c['logit']['n_edges']}  "
          f"({c['logit']['synergy']} syn / {c['logit']['trade_off']} trade, "
          f"trade share {c['logit']['trade_off_share']:.3f})")
    print(f"  retained      : {c['edges_retained']}  "
          f"(lost {c['edges_lost_from_primary']}, new {c['edges_new_in_logit']})")
    print(f"  Jaccard       : {c['jaccard']:.3f}")
    print(f"  sign agreement (shared) : {c['sign_agreement_on_shared']:.3f}")
    print(f"  trade-off share change  : {c['trade_off_share_change']:+.3f}")
    print(f"  net-influence Spearman  : {c['net_influence_spearman']:.3f}")
    print("\nWrote:")
    print(f"  {OUT_DIR / 'bounded_sensitivity.json'}")
    print(f"  {OUT_DIR / 'bounded_edges.csv'}")


if __name__ == "__main__":
    main()
