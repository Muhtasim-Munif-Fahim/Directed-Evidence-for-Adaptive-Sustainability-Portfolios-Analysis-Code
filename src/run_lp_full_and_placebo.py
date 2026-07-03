"""Full 31-pair LP results table + lead-placebo and reverse-direction tests.

Reviewer asks:
  (DeepSeek, Technical) Show ALL 31 prespecified pairs -- including unsupported
  and not-interpretable ones -- with effect sizes, CIs, p, q and status, so the
  evidence screen is auditable.
  (Technical) For the supported pathways, probe timing/simultaneity: a LEAD
  placebo (the h=5 cumulative outcome change regressed on the NEXT year's source
  change; a significant 'effect' of the future would flag timing problems) and
  the REVERSE-direction local projection (target -> source).

Reuses the primary LP machinery (run_panel_local_projections): same pair list,
panel construction, specification (cumulative y_{t+h}-y_{t-1}, country+year FE,
Driscoll-Kraay kernel SEs), H_STAR=5 and BH family.

Outputs:
  outputs/network/lp_full_results.csv   exactly 31 rows
  outputs/network/lp_placebo.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from run_panel_local_projections import (
    CTRL_LAGS, H_STAR, INDICATOR_LABELS, MIN_COUNTRIES, PAIRS,
    build_lp_frame, load_pair_panel,
)
from result_contract import PROTOCOL_VERSION

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "network"

SUPPORTED = [
    ("n_sdg1_wpc", "n_sdg3_u5mort"),      # Poverty -> U5 mortality
    ("n_sdg6_sanita", "n_sdg3_u5mort"),   # Sanitation -> U5 mortality
]


def lp_at_h5(lp: pd.DataFrame, regressor: str, ctrl: list[str]) -> dict | None:
    cols = ["id", "year", f"resp_h{H_STAR}", regressor, *ctrl]
    d = lp[cols].dropna().copy()
    if d["id"].nunique() < MIN_COUNTRIES or len(d) < 100:
        return None
    d = d.set_index(["id", "year"])
    mod = PanelOLS(d[f"resp_h{H_STAR}"], d[[regressor, *ctrl]],
                   entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="kernel", bandwidth=max(H_STAR, 1))
    b = float(res.params[regressor])
    se = float(res.std_errors[regressor])
    return {"beta5": round(b, 4), "se": round(se, 4),
            "ci_low": round(b - 1.96 * se, 4), "ci_high": round(b + 1.96 * se, 4),
            "p": float(f"{float(res.pvalues[regressor]):.4g}"),
            "n_obs": int(res.nobs),
            "n_countries": int(d.index.get_level_values('id').nunique())}


def main() -> None:
    # ---------- Part 1: full 31-pair table ---------------------------------- #
    summ = json.loads((OUT_DIR / "lp_summary.json").read_text(encoding="utf-8"))
    by_pair = {s["pair"]: s for s in summ["all_pairs"]}
    coeffs = pd.read_csv(OUT_DIR / "lp_irf_coefficients.csv")
    h5 = coeffs[coeffs["horizon"] == H_STAR].set_index("pair_label")

    rows = []
    for s, t, exp, cluster, claimed in PAIRS:
        sl, tl = INDICATOR_LABELS.get(s, s), INDICATOR_LABELS.get(t, t)
        label = f"{sl}->{tl}"
        rec = {
            "pair": label, "source_code": s.replace("n_", ""),
            "target_code": t.replace("n_", ""), "cluster": cluster,
            "claimed_by": claimed, "expected_sign": exp,
        }
        info = by_pair.get(label)
        if info is None or info.get("p_star") is None:
            panel = load_pair_panel(s, t)
            n = panel["id"].nunique()
            rec.update({
                "n_countries": int(n), "n_obs": None, "beta5": None,
                "se": None, "ci_low": None, "ci_high": None, "p_raw": None,
                "q_bh": None, "peak_horizon": None,
                "status": "not_interpretable",
                "reason": (f"balanced pair panel has {n} countries "
                           f"(< {MIN_COUNTRIES} required)" if n < MIN_COUNTRIES
                           else "confirmatory-horizon regression not estimable"),
            })
        else:
            h = h5.loc[label] if label in h5.index else None
            rec.update({
                "n_countries": int(h["n_countries"]) if h is not None else None,
                "n_obs": int(h["n_obs"]) if h is not None else None,
                "beta5": round(float(info["beta_star"]), 4),
                "se": round(float(h["std_error"]), 4) if h is not None else None,
                "ci_low": round(float(h["ci_low"]), 4) if h is not None else None,
                "ci_high": round(float(h["ci_high"]), 4) if h is not None else None,
                "p_raw": float(f"{info['p_star']:.4g}"),
                "q_bh": (float(f"{info['q_bh_star']:.4g}")
                         if info.get("q_bh_star") is not None else None),
                "peak_horizon": info.get("peak_horizon"),
                "status": info["status"], "reason": "",
            })
        rows.append(rec)

    full = pd.DataFrame(rows)
    assert len(full) == 31, f"expected 31 rows, got {len(full)}"
    full["protocol_version"] = PROTOCOL_VERSION
    full.to_csv(OUT_DIR / "lp_full_results.csv", index=False)
    print(f"lp_full_results.csv: {len(full)} rows")
    print(full["status"].value_counts().to_string())

    # ---------- Part 2: lead placebo + reverse LP for supported pairs ------- #
    placebo: dict = {"h_star": H_STAR, "tests": {}}
    ctrl = [f"dy_l{l}" for l in range(1, CTRL_LAGS + 1)] + \
           [f"ds_l{l}" for l in range(1, CTRL_LAGS + 1)]

    for s, t in SUPPORTED:
        sl, tl = INDICATOR_LABELS[s], INDICATOR_LABELS[t]
        label = f"{sl}->{tl}"
        print(f"\nPlacebo/reverse for {label} ...")
        panel = load_pair_panel(s, t)

        # primary frame + within-country lead of the source change
        lp = build_lp_frame(panel, s, t)
        lp["ds_lead1"] = lp.groupby("id")["ds"].shift(-1)

        baseline = lp_at_h5(lp, "ds", ctrl)
        lead = lp_at_h5(lp, "ds_lead1", ctrl)

        # joint: current + lead together (does the future add signal?)
        joint = None
        cols = ["id", "year", f"resp_h{H_STAR}", "ds", "ds_lead1", *ctrl]
        d = lp[cols].dropna().copy()
        if d["id"].nunique() >= MIN_COUNTRIES and len(d) >= 100:
            d = d.set_index(["id", "year"])
            res = PanelOLS(d[f"resp_h{H_STAR}"], d[["ds", "ds_lead1", *ctrl]],
                           entity_effects=True, time_effects=True,
                           drop_absorbed=True).fit(cov_type="kernel",
                                                   bandwidth=max(H_STAR, 1))
            joint = {
                "beta_current": round(float(res.params["ds"]), 4),
                "p_current": float(f"{float(res.pvalues['ds']):.4g}"),
                "beta_lead": round(float(res.params["ds_lead1"]), 4),
                "p_lead": float(f"{float(res.pvalues['ds_lead1']):.4g}"),
            }

        # reverse direction: target -> source, same machinery
        lp_rev = build_lp_frame(panel.rename(columns={s: "__s", t: "__t"})
                                .rename(columns={"__s": t, "__t": s}), s, t)
        # note: swapping column names makes 'source'=old target, 'target'=old source
        reverse = lp_at_h5(lp_rev, "ds", ctrl)

        placebo["tests"][label] = {
            "baseline_current_source": baseline,
            "lead_placebo_only": lead,
            "joint_current_plus_lead": joint,
            "reverse_direction": reverse,
        }
        print(f"  baseline beta5={baseline['beta5']} p={baseline['p']}")
        print(f"  lead-only beta5={lead['beta5'] if lead else None} "
              f"p={lead['p'] if lead else None}")
        if joint:
            print(f"  joint: current b={joint['beta_current']} p={joint['p_current']} "
                  f"| lead b={joint['beta_lead']} p={joint['p_lead']}")
        print(f"  reverse beta5={reverse['beta5'] if reverse else None} "
              f"p={reverse['p'] if reverse else None}")

    placebo["protocol_version"] = PROTOCOL_VERSION
    placebo["interpretation_rule"] = (
        "Timing is causal-consistent if the lead coefficient is small/non-"
        "significant while the current coefficient holds; a large significant "
        "reverse-direction beta indicates feedback/simultaneity and the pair "
        "should be described as a reinforcing loop rather than a one-way lever.")
    (OUT_DIR / "lp_placebo.json").write_text(json.dumps(placebo, indent=2),
                                             encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'lp_full_results.csv'}")
    print(f"Wrote {OUT_DIR / 'lp_placebo.json'}")


if __name__ == "__main__":
    main()
