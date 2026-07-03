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
- Public 2025 SDR data archive:
  `https://sdgtransformationcenter.org/sustainable-development-report`
- Public 2025 SDR data hub item:
  `https://sdg-transformation-center-sdsn.hub.arcgis.com/datasets/sdsn::sustainable-development-report-2025-with-indicators/explore`
- Public ArcGIS REST service:
  `https://services7.arcgis.com/IyvyFk20mB7Wpc95/arcgis/rest/services/Sustainable_Development_Report_2025_(with_indicators)/FeatureServer`
- Exact workbook used: `SDR2025-data.xlsx`
- SHA-256:
  `D506CB01758E8743DEEAF3BE03D648AC8FD9C02EC21A42119ABEDCE7AFF4D056`
- Note: use the official archive and Hub item above as the current public access
  route for the 2025 SDR data.
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
- DataBank source page:
  `https://databank.worldbank.org/source/world-development-indicators`
- Public API country endpoint:
  `https://api.worldbank.org/v2/country?format=json&per_page=400`
- Public API indicator endpoint pattern:
  `https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&date=2000:2024&per_page=20000`
- Indicators used:
  `SE.XPD.TOTL.GD.ZS`, `SE.PRM.ENRR`, `SE.SEC.ENRR`, `SP.DYN.LE00.IN`,
  `SH.DYN.MORT`, and `SI.POV.DDAY`
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
