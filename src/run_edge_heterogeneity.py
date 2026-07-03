"""Per-edge cross-country heterogeneity for the 84 BH-significant edges.

Reviewer ask (Technical report): for each retained edge report the fraction of
countries whose summed lag coefficient shares the edge's sign (sign consensus),
and the cross-country IQR, so readers can see whether an edge reflects a broad
pattern or a minority of countries.

Output: outputs/network/edge_heterogeneity.csv
  source, target, n_countries, share_sign_consensus, b_median, b_q25, b_q75
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_panel_granger_network import (
    GOALS, K_LAGS, country_wald, load_balanced_panel,
)

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "outputs" / "network"


def main() -> None:
    panel = load_balanced_panel()
    units = {cid: u for cid, u in panel.groupby("id")}
    edges = pd.read_csv(NET / "granger_edges.csv")
    sig = edges[edges["bh_significant"]].copy()

    rows = []
    for i, r in enumerate(sig.itertuples(), 1):
        bs = []
        for u in units.values():
            out = country_wald(u, r.source, r.target, K_LAGS)
            if out is not None:
                bs.append(out[1])
        bs = np.array(bs)
        med = float(np.median(bs))
        consensus = float(np.mean(np.sign(bs) == np.sign(med))) if med != 0 \
            else float("nan")
        rows.append({
            "source": r.source, "target": r.target,
            "n_countries": len(bs),
            "share_sign_consensus": round(consensus, 3),
            "b_median": round(med, 4),
            "b_q25": round(float(np.percentile(bs, 25)), 4),
            "b_q75": round(float(np.percentile(bs, 75)), 4),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(sig)} edges done")

    out = pd.DataFrame(rows)
    out.to_csv(NET / "edge_heterogeneity.csv", index=False)
    low = out[out["share_sign_consensus"] < 0.7]
    print(f"\n{len(out)} edges; median consensus "
          f"{out['share_sign_consensus'].median():.2f}; "
          f"{len(low)} edges below 70% consensus")
    print(f"Wrote {NET / 'edge_heterogeneity.csv'}")


if __name__ == "__main__":
    main()
