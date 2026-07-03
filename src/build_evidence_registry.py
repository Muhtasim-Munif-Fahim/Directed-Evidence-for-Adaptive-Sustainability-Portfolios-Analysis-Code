"""Build a machine-readable registry from analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from result_contract import PROTOCOL_VERSION, RESULT_COLUMNS, ensure_contract


ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "outputs" / "network"
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "evidence"


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return ensure_contract(pd.read_csv(path))


def granger_registry_frame() -> pd.DataFrame:
    path = NETWORK / "granger_edges.csv"
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    data = pd.read_csv(path)
    frame = pd.DataFrame({
        "model_id": "granger_" + data["source"].astype(str) + "_to_" + data["target"].astype(str),
        "family_id": "panel_granger_network",
        "family": "panel_granger_network",
        "estimator": "dh_panel_granger_first_difference",
        "source": data["source"],
        "target": data["target"],
        "coefficient": data["edge_sign"],
        "std_error": pd.NA,
        "ci_low": pd.NA,
        "ci_high": pd.NA,
        "p_raw": data["p_raw"],
        "q_bh": data["q_bh"],
        "observations": pd.NA,
        "countries": data["n_countries"],
        "diagnostic_gate": data["estimable"].map({True: "pass", False: "fail"}),
        "diagnostic_reason": "",
        "direction_consistent": data["bh_significant"],
        "claim_status": data["bh_significant"].map({True: "supported", False: "unsupported"}),
        "protocol_version": data.get("protocol_version", PROTOCOL_VERSION),
        "effect_scale": "summed_lag_sign",
        "expected_direction": pd.NA,
        "source_name": data.get("source_name"),
        "target_name": data.get("target_name"),
        "direction": data.get("direction"),
    })
    return ensure_contract(frame)


def local_projection_registry_frame() -> pd.DataFrame:
    path = NETWORK / "lp_full_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    data = pd.read_csv(path)
    frame = pd.DataFrame({
        "model_id": "lp_" + data["source_code"].astype(str) + "_to_" + data["target_code"].astype(str),
        "family_id": "local_projections",
        "family": "local_projections",
        "estimator": "panel_local_projection_driscoll_kraay",
        "source": data["source_code"],
        "target": data["target_code"],
        "coefficient": data["beta5"],
        "std_error": data["se"],
        "ci_low": data["ci_low"],
        "ci_high": data["ci_high"],
        "p_raw": data["p_raw"],
        "q_bh": data["q_bh"],
        "observations": data["n_obs"],
        "countries": data["n_countries"],
        "diagnostic_gate": data["status"].eq("not_interpretable").map({True: "fail", False: "pass"}),
        "diagnostic_reason": data.get("reason", ""),
        "direction_consistent": data["status"].isin(["supported", "suggestive"]),
        "claim_status": data["status"],
        "protocol_version": data.get("protocol_version", PROTOCOL_VERSION),
        "effect_scale": "five_year_cumulative_response",
        "expected_direction": data["expected_sign"],
        "pathway": data["pair"],
        "cluster": data.get("cluster"),
        "claimed_by": data.get("claimed_by"),
    })
    return ensure_contract(frame)


def build_registry() -> pd.DataFrame:
    frames = [
        granger_registry_frame(),
        local_projection_registry_frame(),
        load_optional(TABLES / "external_validation_results.csv"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_registry(registry: pd.DataFrame) -> dict:
    if registry.empty:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "rows": 0,
            "families": {},
        }

    families = {}
    for family, group in registry.groupby("family", dropna=False):
        families[str(family)] = {
            "rows": int(len(group)),
            "supported": int((group["claim_status"] == "supported").sum()),
            "suggestive": int((group["claim_status"] == "suggestive").sum()),
            "not_interpretable": int((group["claim_status"] == "not_interpretable").sum()),
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "rows": int(len(registry)),
        "families": families,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    registry.to_csv(OUT / "result_registry.csv", index=False)

    claims = registry[
        (registry["claim_status"].isin(["supported", "suggestive"]))
        & (registry["source"] != registry["target"])
    ].copy()
    claims.to_csv(OUT / "promotable_claims.csv", index=False)

    summary = summarize_registry(registry)
    (OUT / "analysis_registry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
