"""Dumitrescu-Hurlin (2012) panel Granger non-causality network across 17 SDGs.

Why this method (replaces the small-T GMM core):
  The SDR panel is a MACRO panel: N ~ 114 countries, T = 25 years. Arellano-Bond /
  Blundell-Bond GMM is built for small-T micro panels; with T = 25 its instrument
  count explodes (Roodman 2009), producing the degenerate/near-null results we saw.
  Dumitrescu & Hurlin (2012, Economic Modelling) is the canonical Granger
  non-causality test for exactly this regime: it allows fully heterogeneous
  coefficients across countries, is valid for moderate-large T, and yields a
  standardised Z-bar statistic per directed pair.

For each directed pair (x -> y) we run, for every country i, the individual VAR-style
regression
    y_{i,t} = a_i + sum_{k=1..K} g_{i,k} y_{i,t-k} + sum_{k=1..K} b_{i,k} x_{i,t-k} + e_{i,t}
and test H0_i: b_{i,1} = ... = b_{i,K} = 0 with a Wald statistic W_i.
The panel statistic is W_bar = mean_i(W_i), standardised to the finite-sample
Z-bar-tilde of DH(2012). p-values are one-sided (upper tail): non-causality is
rejected only for large positive Z-bar-tilde.

Edges with BH-FDR q < 0.05 form the directed SDG interaction network.
Edge sign (synergy vs trade-off) is the cross-country mean of sum_k b_{i,k}:
all SDR scores are 0-100 with higher = better, so a positive sum means an
improvement in x Granger-predicts an improvement in y (synergy).

Outputs:
  outputs/network/granger_edges.csv        one row per directed pair
  outputs/network/granger_centrality.csv   per-goal driver/receiver centrality
  outputs/network/granger_network_summary.json
"""

from __future__ import annotations

import json
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import statsmodels.api as sm

from result_contract import PROTOCOL_VERSION, bh_adjust

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
DATA_FILE = ROOT / "data" / "raw" / "SDR2025-data.xlsx"
OUT_DIR = ROOT / "outputs" / "network"

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

K_LAGS = 2          # lag order for the underlying VAR
MIN_COUNTRY_OBS = 10  # minimum usable rows per country regression


def load_balanced_panel() -> pd.DataFrame:
    """17-goal balanced panel: countries complete on all goals for all 25 years."""
    frame = pd.read_excel(
        DATA_FILE, sheet_name="Backdated SDG Index", usecols=["id", "year", *GOALS]
    )
    frame = frame[~frame["id"].astype(str).str.startswith("_")].copy()
    frame = frame[frame["year"].between(2000, 2024)]
    complete = frame.dropna(subset=GOALS)
    counts = complete.groupby("id")["year"].nunique()
    ids = counts[counts == 25].index
    balanced = complete[complete["id"].isin(ids)].copy()
    n = balanced["id"].nunique()
    if n < 100:
        raise ValueError(f"Balanced panel has only {n} countries (need >= 100)")
    print(f"Balanced panel: {n} countries x 25 years = {len(balanced)} rows")
    return balanced.sort_values(["id", "year"])


def country_wald(unit: pd.DataFrame, x: str, y: str, k: int,
                 already_differenced: bool = False,
                 cce: bool = False) -> tuple[float, float] | None:
    """Per-country Wald stat for H0: all k lags of x have zero coefficient.

    SDG scores are persistent/trending (near-I(1)); testing Granger causality on
    levels of such series yields spurious causality. We therefore difference both
    series to (approximate) stationarity and test causality in year-over-year
    changes -- consistent with the local-projection specification.

    Returns (W_i, sum_of_x_lag_coefficients) or None if not estimable.
    """
    needed = [y, x] + ([f"{y}_csa", f"{x}_csa"] if cce else [])
    df = unit.sort_values("year")[needed] if "year" in unit.columns else unit[needed]
    if already_differenced:
        # series are already first-differenced (and possibly year-demeaned upstream)
        dy = df[y]
        dx = df[x]
    else:
        dy = df[y].diff()
        dx = df[x].diff()
    cols = {}
    for lag in range(1, k + 1):
        cols[f"y_l{lag}"] = dy.shift(lag).values
        cols[f"x_l{lag}"] = dx.shift(lag).values
    design = pd.DataFrame(cols)
    design["y"] = dy.values
    if cce:
        # CCE augmentation: contemporaneous cross-sectional averages of the
        # (differenced) regressand and regressor proxy the common factor with a
        # country-specific loading (Pesaran 2006), unlike year-demeaning which
        # imposes a homogeneous unit loading.
        design["csa_y"] = df[f"{y}_csa"].values
        design["csa_x"] = df[f"{x}_csa"].values
    design = design.dropna()
    if len(design) < MIN_COUNTRY_OBS:
        return None

    x_cols = [f"x_l{lag}" for lag in range(1, k + 1)]
    y_cols = [f"y_l{lag}" for lag in range(1, k + 1)]
    extra = ["csa_y", "csa_x"] if cce else []
    # variance floor: near-constant (e.g. imputed-flat) differenced series give
    # degenerate regressions and explosive Wald stats -- drop such units.
    if design[[*y_cols, *x_cols, "y"]].std().min() < 1e-4:
        return None
    X = sm.add_constant(design[y_cols + x_cols + extra])
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None
    # guard against near-singular designs that explode the Wald statistic
    if np.linalg.cond(X.values) > 1e6:
        return None
    try:
        res = sm.OLS(design["y"], X).fit()
        # joint test that the k x-lags are zero -> chi2 with k df
        wald = res.wald_test(
            np.column_stack([
                np.zeros((k, 1 + k)),              # const + k y-lags unrestricted
                np.eye(k),                         # k x-lags restricted to 0
                np.zeros((k, len(extra))),         # CCE cross-sectional averages free
            ]),
            scalar=True,
            use_f=False,
        )
        w_i = float(np.asarray(wald.statistic).ravel()[0])
        b_sum = float(res.params[x_cols].sum())
        # a chi2(k) Wald above this is numerically degenerate, not real signal
        if not np.isfinite(w_i) or w_i > 500:
            return None
        return w_i, b_sum
    except Exception:
        return None


def dumitrescu_hurlin(panel: pd.DataFrame, x: str, y: str, k: int,
                      already_differenced: bool = False,
                      cce: bool = False) -> dict:
    """Panel Granger non-causality test for x -> y."""
    total = panel["id"].nunique()
    w_list, b_list = [], []
    for _, unit in panel.groupby("id"):
        out = country_wald(unit, x, y, k, already_differenced=already_differenced,
                           cce=cce)
        if out is not None:
            w_list.append(out[0])
            b_list.append(out[1])

    n_eff = len(w_list)
    if n_eff < 30:
        return {"source": x, "target": y, "n_countries": n_eff,
                "n_excluded": total - n_eff,
                "w_bar": np.nan, "z_bar_tilde": np.nan, "p_raw": np.nan,
                "edge_sign": np.nan, "estimable": False}

    w_bar = float(np.mean(w_list))
    t = 24  # effective length after first-differencing (25 - 1)

    # DH(2012) finite-sample standardised statistic Z-bar-tilde
    num = (t - 2 * k - 5)
    den = (t - k - 3)
    scale = np.sqrt((n_eff / (2 * k)) * (num / den))
    adj = ((t - 2 * k - 3) / (t - 2 * k - 1)) * w_bar - k
    z_tilde = float(scale * adj)
    # One-sided upper-tail p-value. Under H0 of non-causality Z-bar-tilde ~ N(0,1)
    # and the test rejects only for large POSITIVE values (W_bar >> k). The earlier
    # two-sided form 2*(1-Phi(|z|)) over-penalised genuine positive-z edges and could
    # in principle admit anti-causal (z<0) pairs; the canonical DH(2012) rejection
    # region is the upper tail.
    p_raw = float(stats.norm.sf(z_tilde))

    return {
        "source": x, "target": y, "n_countries": n_eff,
        "n_excluded": total - n_eff,
        "w_bar": w_bar, "z_bar_tilde": z_tilde, "p_raw": p_raw,
        # robust (median) cross-country coefficient: sign = synergy vs trade-off
        "edge_sign": float(np.median(b_list)), "estimable": True,
    }


def compute_edges(panel: pd.DataFrame, k: int,
                  already_differenced: bool = False,
                  cce: bool = False) -> pd.DataFrame:
    """Full DH pass over all directed pairs at lag k, with BH-FDR and edge signs."""
    rows = [dumitrescu_hurlin(panel, x, y, k, already_differenced=already_differenced,
                              cce=cce)
            for x, y in permutations(GOALS, 2)]
    est = pd.DataFrame(rows)
    est = est[est["estimable"]].copy()
    est = bh_adjust(est)
    est["bh_significant"] = est["q_bh"] < 0.05
    est["direction"] = np.where(est["edge_sign"] >= 0, "synergy", "trade_off")
    est["source_name"] = est["source"].map(GOAL_NAMES)
    est["target_name"] = est["target"].map(GOAL_NAMES)
    return est


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_balanced_panel()

    print(f"\nRunning Dumitrescu-Hurlin (K={K_LAGS}) over {len(GOALS) * (len(GOALS) - 1)} directed pairs...")
    estimable = compute_edges(panel, K_LAGS)
    estimable["protocol_version"] = PROTOCOL_VERSION
    estimable = estimable.sort_values("p_raw")
    estimable.to_csv(OUT_DIR / "granger_edges.csv", index=False)

    # --- Directed network from BH-significant edges ---
    sig = estimable[estimable["bh_significant"]].copy()
    G = nx.DiGraph()
    G.add_nodes_from(GOALS)
    for r in sig.itertuples():
        G.add_edge(r.source, r.target, weight=abs(r.z_bar_tilde),
                   sign=r.direction, q=r.q_bh)

    # Centrality: drivers (out) vs receivers (in)
    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())
    try:
        btw = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        btw = {n: 0.0 for n in G.nodes}

    cent_rows = []
    for g in GOALS:
        cent_rows.append({
            "goal": g, "goal_name": GOAL_NAMES[g],
            "out_degree": out_deg.get(g, 0),     # how many goals it drives
            "in_degree": in_deg.get(g, 0),       # how many goals drive it
            "net_influence": out_deg.get(g, 0) - in_deg.get(g, 0),
            "betweenness": round(btw.get(g, 0.0), 4),
            "role": ("driver" if out_deg.get(g, 0) > in_deg.get(g, 0)
                     else "receiver" if in_deg.get(g, 0) > out_deg.get(g, 0)
                     else "balanced"),
        })
    centrality = pd.DataFrame(cent_rows).sort_values(
        "net_influence", ascending=False
    )
    centrality.to_csv(OUT_DIR / "granger_centrality.csv", index=False)

    n_sig = int(sig.shape[0])
    n_syn = int((sig["direction"] == "synergy").sum())
    n_trade = int((sig["direction"] == "trade_off").sum())

    top_drivers = centrality.nlargest(5, "net_influence")[
        ["goal_name", "out_degree", "in_degree", "net_influence"]
    ].to_dict("records")
    top_receivers = centrality.nsmallest(5, "net_influence")[
        ["goal_name", "out_degree", "in_degree", "net_influence"]
    ].to_dict("records")

    # --- M3: country-exclusion summary on the primary (K=2) edges ---
    excl = estimable["n_excluded"]
    exclusion = {
        "panel_countries": int(panel["id"].nunique()),
        "median_excluded_per_pair": int(excl.median()),
        "max_excluded_per_pair": int(excl.max()),
        "mean_excluded_per_pair": round(float(excl.mean()), 1),
    }

    # --- M1/M2: lag-order sensitivity (K=1, K=3) vs primary K=2 ---
    primary_edges = {(r.source, r.target) for r in sig.itertuples()}
    primary_sign = {(r.source, r.target): r.direction for r in sig.itertuples()}
    lag_sensitivity = {}
    for k_alt in (1, 3):
        print(f"  lag-sensitivity: recomputing network at K={k_alt}...")
        e_alt = compute_edges(panel, k_alt)
        sig_alt = {(r.source, r.target) for r in
                   e_alt[e_alt["bh_significant"]].itertuples()}
        sign_alt = {(r.source, r.target): r.direction for r in e_alt.itertuples()}
        shared = primary_edges & sig_alt
        sign_agree = (sum(1 for ed in shared if sign_alt.get(ed) == primary_sign[ed])
                      / len(shared)) if shared else float("nan")
        lag_sensitivity[f"K={k_alt}"] = {
            "n_significant_edges": len(sig_alt),
            "retained_share_of_primary": round(len(shared) / len(primary_edges), 3)
            if primary_edges else float("nan"),
            "sign_agreement_on_shared": round(sign_agree, 3),
        }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "Dumitrescu-Hurlin (2012) panel Granger non-causality",
        "lag_order_K": K_LAGS,
        "panel": {"countries": int(panel["id"].nunique()), "years": 25},
        "directed_pairs_tested": int(estimable.shape[0]),
        "bh_significant_edges": n_sig,
        "synergy_edges": n_syn,
        "trade_off_edges": n_trade,
        "network_density": round(nx.density(G), 4),
        "country_exclusion": exclusion,
        "lag_sensitivity": lag_sensitivity,
        "top_driver_goals": top_drivers,
        "top_receiver_goals": top_receivers,
        "strongest_edges": [
            {
                "edge": f"{r.source_name} -> {r.target_name}",
                "z_bar_tilde": round(r.z_bar_tilde, 3),
                "q_bh": round(r.q_bh, 5),
                "direction": r.direction,
            }
            for r in sig.nlargest(15, "z_bar_tilde").itertuples()
        ],
    }
    (OUT_DIR / "granger_network_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(f"Directed pairs tested:  {estimable.shape[0]}")
    print(f"BH-significant edges:   {n_sig}  ({n_syn} synergy, {n_trade} trade-off)")
    print(f"Network density:        {nx.density(G):.3f}")
    print("\nTop driver goals (net influence = out - in):")
    for d in top_drivers:
        print(f"  {d['goal_name']:<28} out={d['out_degree']:>2} in={d['in_degree']:>2} "
              f"net={d['net_influence']:+d}")
    print("\nTop receiver goals:")
    for d in top_receivers:
        print(f"  {d['goal_name']:<28} out={d['out_degree']:>2} in={d['in_degree']:>2} "
              f"net={d['net_influence']:+d}")
    print("\nStrongest 10 edges:")
    for e in summary["strongest_edges"][:10]:
        print(f"  {e['edge']:<48} z={e['z_bar_tilde']:>7.2f} q={e['q_bh']:.4g} [{e['direction']}]")


if __name__ == "__main__":
    main()
