"""Run prespecified indicator and policy-input validation families."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from result_contract import PROTOCOL_VERSION, bh_adjust, classify_claim, ensure_contract


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "external_validation_panel.csv"
OUT_DIR = ROOT / "outputs" / "tables"

TESTS = [
    ("corridor_validation", "primary enrollment -> life expectancy", "primary_enrollment_gross_pct", "life_expectancy_years", 1),
    ("corridor_validation", "primary enrollment -> under-5 mortality", "primary_enrollment_gross_pct", "under5_mortality_per_1000", -1),
    ("corridor_validation", "secondary enrollment -> life expectancy", "secondary_enrollment_gross_pct", "life_expectancy_years", 1),
    ("corridor_validation", "secondary enrollment -> under-5 mortality", "secondary_enrollment_gross_pct", "under5_mortality_per_1000", -1),
    ("corridor_validation", "life expectancy -> poverty headcount", "life_expectancy_years", "poverty_headcount_3usd_2021ppp_pct", -1),
    ("corridor_validation", "under-5 mortality -> poverty headcount", "under5_mortality_per_1000", "poverty_headcount_3usd_2021ppp_pct", 1),
    ("policy_input_mechanisms", "education expenditure -> primary enrollment", "education_expenditure_pct_gdp", "primary_enrollment_gross_pct", 1),
    ("policy_input_mechanisms", "education expenditure -> secondary enrollment", "education_expenditure_pct_gdp", "secondary_enrollment_gross_pct", 1),
    ("policy_input_mechanisms", "government health expenditure (% GDP) -> life expectancy", "gghed_gdp", "life_expectancy_years", 1),
    ("policy_input_mechanisms", "government health expenditure (% GDP) -> under-5 mortality", "gghed_gdp", "under5_mortality_per_1000", -1),
    ("policy_input_mechanisms", "government health expenditure per capita -> life expectancy", "gghed_pc_usd", "life_expectancy_years", 1),
    ("policy_input_mechanisms", "government health expenditure per capita -> under-5 mortality", "gghed_pc_usd", "under5_mortality_per_1000", -1),
]


def annual_changes(frame: pd.DataFrame, variables: set[str]) -> pd.DataFrame:
    result = frame.sort_values(["id", "year"]).copy()
    for variable in variables:
        result[f"d_{variable}"] = result.groupby("id")[variable].diff()
        result[f"lag_d_{variable}"] = result.groupby("id")[f"d_{variable}"].shift(1)
    return result


def fit_test(frame: pd.DataFrame, test: tuple, sample: str) -> dict:
    family, pathway, source, target, expected_sign = test
    source_term = f"lag_d_{source}"
    target_term = f"d_{target}"
    target_lag = f"lag_d_{target}"
    analysis = frame[["id", "year", source_term, target_term, target_lag]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    countries = int(analysis["id"].nunique())
    observations = int(len(analysis))
    gate = "pass" if countries >= 20 and observations >= 100 else "fail"
    reason = "" if gate == "pass" else f"minimum sample failed: {countries} countries, {observations} observations"

    formula = f"{target_term} ~ {source_term} + {target_lag} + C(year)"
    if gate == "fail":
        return {
            "model_id": f"validation_{sample}_{source}_to_{target}",
            "family_id": f"{family}_{sample}",
            "family": family,
            "estimator": "annual_change_ols_cluster_country",
            "source": source,
            "target": target,
            "pathway": pathway,
            "coefficient": np.nan,
            "std_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_raw": np.nan,
            "observations": observations,
            "countries": countries,
            "diagnostic_gate": gate,
            "diagnostic_reason": reason,
            "expected_direction": expected_sign,
            "direction_consistent": False,
            "effect_scale": "annual_change",
            "coefficient_std": np.nan,
            "ci_low_std": np.nan,
            "ci_high_std": np.nan,
            "sample": sample,
            "model_formula": formula,
        }

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["id"]}
    )
    coefficient = float(model.params[source_term])
    interval = model.conf_int().loc[source_term]
    source_sd = float(analysis[source_term].std())
    target_sd = float(analysis[target_term].std())
    scale = source_sd / target_sd if target_sd > 0 else np.nan
    return {
        "model_id": f"validation_{sample}_{source}_to_{target}",
        "family_id": f"{family}_{sample}",
        "family": family,
        "estimator": "annual_change_ols_cluster_country",
        "source": source,
        "target": target,
        "pathway": pathway,
        "coefficient": coefficient,
        "std_error": float(model.bse[source_term]),
        "ci_low": float(interval.iloc[0]),
        "ci_high": float(interval.iloc[1]),
        "p_raw": float(model.pvalues[source_term]),
        "observations": int(model.nobs),
        "countries": countries,
        "diagnostic_gate": gate,
        "diagnostic_reason": reason,
        "expected_direction": expected_sign,
        "direction_consistent": bool(np.sign(coefficient) == expected_sign),
        "effect_scale": "annual_change",
        "coefficient_std": coefficient * scale,
        "ci_low_std": float(interval.iloc[0]) * scale,
        "ci_high_std": float(interval.iloc[1]) * scale,
        "sample": sample,
        "model_formula": formula,
        "r_squared": float(model.rsquared),
        "evidence_role": "secondary" if "poverty" in target else "primary",
    }


def add_multiplicity_and_claims(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, family in frame.groupby("family_id", sort=False):
        pieces.append(bh_adjust(family))
    result = pd.concat(pieces, ignore_index=True)

    sensitivity_sign = {
        (row.source, row.target): np.sign(row.coefficient)
        for row in result[result["sample"] == "pre2020"].itertuples()
        if pd.notna(row.coefficient)
    }
    statuses = []
    stability = []
    for row in result.itertuples():
        if row.sample == "full":
            stable = bool(
                pd.notna(row.coefficient)
                and sensitivity_sign.get((row.source, row.target)) == np.sign(row.coefficient)
            )
        else:
            stable = bool(row.direction_consistent)
        stability.append(stable)
        statuses.append(
            classify_claim(
                diagnostic_gate=row.diagnostic_gate,
                p_raw=row.p_raw,
                q_bh=row.q_bh,
                direction_consistent=row.direction_consistent,
                promotion_pass=stable,
            )
        )
    result["sensitivity_direction_consistent"] = stability
    result["claim_status"] = statuses
    result["protocol_version"] = PROTOCOL_VERSION
    return ensure_contract(result)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variables = {test[2] for test in TESTS} | {test[3] for test in TESTS}
    frame = pd.read_csv(
        DATA_FILE,
        usecols=["id", "year", *sorted(variables)],
        engine="python",
    )
    changed = annual_changes(frame, variables)
    rows = []
    for sample, sample_frame in [
        ("full", changed),
        ("pre2020", changed[changed["year"] <= 2019]),
    ]:
        rows.extend(fit_test(sample_frame, test, sample) for test in TESTS)

    results = add_multiplicity_and_claims(pd.DataFrame(rows))
    results.to_csv(OUT_DIR / "external_validation_results.csv", index=False)

    primary = results[results["sample"] == "full"]
    summary = {
        family: {
            "tests": int(len(group)),
            "supported": int((group["claim_status"] == "supported").sum()),
            "suggestive": int((group["claim_status"] == "suggestive").sum()),
            "not_interpretable": int((group["claim_status"] == "not_interpretable").sum()),
        }
        for family, group in primary.groupby("family")
    }
    summary["protocol_version"] = PROTOCOL_VERSION
    (OUT_DIR / "external_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(primary[[
        "family", "pathway", "coefficient_std", "p_raw", "q_bh",
        "countries", "sensitivity_direction_consistent", "claim_status"
    ]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
