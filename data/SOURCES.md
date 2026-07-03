# Data Sources

This repository contains reproducibility code only. Large public input files and
derived analysis outputs are intentionally not tracked in Git.

Create the following local files before running the full pipeline:

```text
data/raw/SDR2025-data.xlsx
data/raw/wb_income_classification.csv
data/raw/wb_region_classification.csv
data/raw/who_ghed/GHED_data.xlsx
data/raw/world_bank/world_bank_policy_validation_long.csv
data/raw/world_bank/world_bank_policy_validation_metadata.json
```

## Sustainable Development Report 2025

- Local file: `data/raw/SDR2025-data.xlsx`
- Public download:
  `https://dashboards.sdgindex.org/static/downloads/files/SDR2025-data.xlsx`
- Required workbook sheets: `Backdated SDG Index`, `SDR2025 Data`
- Use: 2000-2024 SDG goal scores and constituent indicators.

## World Bank Classifications

- Local files:
  - `data/raw/wb_income_classification.csv`
  - `data/raw/wb_region_classification.csv`
- Source: World Bank country and lending group metadata:
  `https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups`
- Required columns are read by the scripts as `id`, `income_level`, and
  `region` where applicable.

## World Development Indicators

- Local file after acquisition:
  `data/raw/world_bank/world_bank_policy_validation_long.csv`
- Metadata after acquisition:
  `data/raw/world_bank/world_bank_policy_validation_metadata.json`
- Public API: `https://api.worldbank.org/v2/`
- Acquisition command:

```bash
python src/download_world_bank.py
```

## WHO Global Health Expenditure Database

- Local file: `data/raw/who_ghed/GHED_data.xlsx`
- Public download page:
  `https://apps.who.int/nha/database/Home/IndicatorsDownload/en`
- Required sheet: `Data`

The code treats all input datasets as public third-party sources. The local
file paths above are the expected locations for reproducible execution.
