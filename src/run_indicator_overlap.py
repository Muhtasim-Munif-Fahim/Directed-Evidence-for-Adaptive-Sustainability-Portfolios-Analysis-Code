"""Indicator-overlap audit: are DH network edges mechanical artifacts of
indicators shared between SDR goal composites?

Reviewer concern (DeepSeek, Stanford, Technical report): if the same underlying
indicator contributes to two goal composites, a Granger edge between those goals
could be partly mechanical. We audit the SDR 2025 Codebook:

  1. EXACT overlap: indicator codes (IndCode) assigned to more than one SDG.
  2. NAME-LEVEL near-duplicates: identical or highly similar indicator labels
     appearing under different goals (same source variable re-used), via
     normalised exact match and difflib ratio > 0.85.

If any goal pairs are linked by shared/near-duplicate indicators, we re-screen
the network EXCLUDING those directed pairs from the BH family (exclusion only
changes the multiplicity family, so the existing per-pair p-values in
granger_edges.csv are reused -- no re-estimation needed) and report the effect
on edge counts and the net-influence ranking.

Outputs:
  outputs/network/indicator_overlap.json
  outputs/network/indicator_overlap_matrix.csv  (17x17 shared-indicator counts)
"""
from __future__ import annotations

import difflib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_panel_granger_network import GOALS, GOAL_NAMES, DATA_FILE
from result_contract import PROTOCOL_VERSION, bh_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "network"


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def net_influence(sig: pd.DataFrame) -> pd.Series:
    out = sig.groupby("source").size().reindex(GOALS, fill_value=0)
    inn = sig.groupby("target").size().reindex(GOALS, fill_value=0)
    return (out - inn).reindex(GOALS)


def main() -> None:
    cb = pd.read_excel(DATA_FILE, sheet_name="Codebook")
    cb = cb.dropna(subset=["IndCode", "SDG"]).copy()
    cb["SDG"] = cb["SDG"].astype(int)
    cb = cb[cb["SDG"].between(1, 17)]
    cb["name_norm"] = cb["Indicator"].map(norm_name)
    n_ind = len(cb)
    print(f"Codebook: {n_ind} indicator rows across {cb['SDG'].nunique()} goals")

    # 1. exact IndCode overlap across goals
    code_goals = cb.groupby("IndCode")["SDG"].agg(lambda s: sorted(set(s)))
    exact_dup = {c: g for c, g in code_goals.items() if len(g) > 1}

    # 2. identical normalised names across goals
    name_goals = cb.groupby("name_norm")["SDG"].agg(lambda s: sorted(set(s)))
    name_dup = {n: g for n, g in name_goals.items() if len(g) > 1}

    # 3. fuzzy near-duplicates across goals (ratio > 0.85)
    near = []
    rows = cb[["IndCode", "SDG", "name_norm", "Indicator"]].drop_duplicates(
        "IndCode").to_dict("records")
    for a, b in itertools.combinations(rows, 2):
        if a["SDG"] == b["SDG"]:
            continue
        if a["name_norm"] == b["name_norm"]:
            continue  # already counted as exact-name duplicate
        r = difflib.SequenceMatcher(None, a["name_norm"], b["name_norm"]).ratio()
        if r > 0.85:
            near.append({"ind_a": a["IndCode"], "goal_a": int(a["SDG"]),
                         "name_a": a["Indicator"],
                         "ind_b": b["IndCode"], "goal_b": int(b["SDG"]),
                         "name_b": b["Indicator"], "similarity": round(r, 3)})

    # overlap matrix (unordered goal pairs -> count of shared/near-shared indicators)
    mat = pd.DataFrame(0, index=GOALS, columns=GOALS)
    flagged_pairs: set[tuple[str, str]] = set()

    def flag(g_list: list[int]) -> None:
        for ga, gb in itertools.combinations(sorted(set(g_list)), 2):
            a, b = f"goal{ga}", f"goal{gb}"
            mat.loc[a, b] += 1
            mat.loc[b, a] += 1
            flagged_pairs.add((a, b))
            flagged_pairs.add((b, a))

    for g in exact_dup.values():
        flag(g)
    for g in name_dup.values():
        flag(g)
    for d in near:
        flag([d["goal_a"], d["goal_b"]])

    mat.to_csv(OUT_DIR / "indicator_overlap_matrix.csv")

    result: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "n_indicators": n_ind,
        "exact_code_duplicates": {c: g for c, g in exact_dup.items()},
        "exact_name_duplicates": {n: g for n, g in name_dup.items()},
        "near_duplicates_ratio_gt_0.85": near,
        "n_flagged_goal_pairs_unordered": len(flagged_pairs) // 2,
    }

    # network sensitivity: drop flagged DIRECTED pairs from the BH family
    edges = pd.read_csv(OUT_DIR / "granger_edges.csv")
    psig = edges[edges["bh_significant"]].copy()
    if flagged_pairs:
        keep = ~edges.apply(lambda r: (r["source"], r["target"]) in flagged_pairs,
                            axis=1)
        sub = edges[keep][["source", "target", "p_raw", "edge_sign",
                           "z_bar_tilde"]].copy()
        sub = bh_adjust(sub)
        sub["bh_significant"] = sub["q_bh"] < 0.05
        sub["direction"] = np.where(sub["edge_sign"] >= 0, "synergy", "trade_off")
        ssig = sub[sub["bh_significant"]]
        prim_kept = psig[~psig.apply(
            lambda r: (r["source"], r["target"]) in flagged_pairs, axis=1)]
        rho = float(stats.spearmanr(net_influence(psig),
                                    net_influence(ssig)).correlation)
        result["low_overlap_network"] = {
            "directed_pairs_excluded": int((~keep).sum()),
            "primary_edges_on_flagged_pairs": int(len(psig) - len(prim_kept)),
            "bh_significant_edges": int(len(ssig)),
            "synergy": int((ssig["direction"] == "synergy").sum()),
            "trade_off": int((ssig["direction"] == "trade_off").sum()),
            "net_influence_spearman_vs_primary": round(rho, 3),
        }
    else:
        result["low_overlap_network"] = (
            "Not needed: no indicator is shared between goal composites "
            "(SDR assigns each indicator to exactly one goal); no goal pair "
            "is mechanically linked at the composite-construction level.")

    (OUT_DIR / "indicator_overlap.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"\nExact IndCode duplicates across goals: {len(exact_dup)}")
    print(f"Exact name duplicates across goals:    {len(name_dup)}")
    print(f"Fuzzy near-duplicates (>0.85):         {len(near)}")
    for d in near[:15]:
        print(f"  [{d['similarity']:.2f}] SDG{d['goal_a']} '{d['name_a']}'  ~  "
              f"SDG{d['goal_b']} '{d['name_b']}'")
    if flagged_pairs:
        print(f"\nFlagged unordered goal pairs: {len(flagged_pairs) // 2}")
        print(json.dumps(result["low_overlap_network"], indent=2))
    else:
        print("\nNo mechanically linked goal pairs -> no network re-screen needed.")
    print(f"\nWrote {OUT_DIR / 'indicator_overlap.json'}")
    print(f"Wrote {OUT_DIR / 'indicator_overlap_matrix.csv'}")


if __name__ == "__main__":
    main()
