# Directed Evidence for Adaptive Sustainability Portfolios

Reproducibility code for:

**Beyond Sustainable Development Goal (SDG) Accelerators: Directed Evidence for
Adaptive Sustainability Portfolios**

The analysis estimates directed goal-to-goal linkages with
Dumitrescu-Hurlin panel Granger non-causality tests on first-differenced SDG
goal series, applies false-discovery control across 272 directed pairs, and
estimates dynamic magnitudes for 31 theory-derived indicator linkages using
panel local projections with Driscoll-Kraay standard errors.

## Repository Scope

This repository contains the public analysis code needed to reproduce the
statistical outputs and machine-readable result registry. It intentionally does
not track large public datasets or generated outputs.

The source basis is the analysis-run code used for the Earth's Future review
package. This public copy excludes only non-analysis helpers such as manuscript
QA, figure rendering, LaTeX table generation, supplement assembly, and file-sync
utilities. Repository-local data paths replace the private project layout so
the scripts read public inputs from `data/raw/`.

Tracked:

- `src/`: acquisition, audit, analysis, robustness, validation, and registry scripts.
- `config/study_design.json`: frozen analysis and claim-status contract.
- `data/SOURCES.md`: exact public input source instructions.
- `tests/`: public-release checks that do not require raw data.

Not tracked:

- raw public datasets under `data/raw/`
- derived panels under `data/processed/`
- generated result files under `outputs/`

## Data

Place public input files in these paths before running the full pipeline:

```text
data/raw/SDR2025-data.xlsx
data/raw/wb_income_classification.csv
data/raw/wb_region_classification.csv
data/raw/who_ghed/GHED_data.xlsx
data/raw/world_bank/world_bank_policy_validation_long.csv
data/raw/world_bank/world_bank_policy_validation_metadata.json
```

See `data/SOURCES.md` for source links and acquisition notes. World Bank WDI
inputs can be downloaded with:

```bash
python src/download_world_bank.py
```

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Reproducible Run Order

Run from the repository root after placing the public input files:

```bash
python src/download_world_bank.py
python src/prepare_external_validation.py
python src/audit_data.py
python src/run_panel_unit_root.py
python src/run_panel_granger_network.py
python src/run_by_fdr_sensitivity.py
python src/run_network_robustness_csd.py
python src/run_lag_stability.py
python src/run_centrality_bootstrap.py
python src/run_subperiod_stability.py
python src/run_bounded_sensitivity.py
python src/run_indicator_overlap.py
python src/run_panel_local_projections.py
python src/run_lp_full_and_placebo.py
python src/run_lp_income_strata.py
python src/run_external_validation.py
python src/run_external_validation_full.py
python src/run_edge_heterogeneity.py
python src/run_selection_table.py
python src/build_evidence_registry.py
```

`src/build_evidence_registry.py` writes the standardized registry to
`outputs/evidence/result_registry.csv`, promotable claims to
`outputs/evidence/promotable_claims.csv`, and a compact registry summary to
`outputs/evidence/analysis_registry_summary.json`.

## Public-Release Checks

These checks verify the repository structure, compile the Python source, and
scan tracked files for local/private path fragments:

```bash
python -m unittest discover -s tests -v
```

These checks do not require the raw datasets. Full statistical reproduction
requires the data files listed above.

## License

Code is released under the MIT License.
