# Data

Raw data are not stored in this repository. Download them from the original sources and put the files into `data/raw/` with the exact names below. All series used in the analysis end on 30 June 2026.

## 1. Planned stabilisation outages

- **Source:** Energy Map, DiXi Group
- **Page:** https://energy-map.info/uk/datasets/0f8f9882-1fb2-47c6-81dc-31fbad914f16
- **Title on the site:** «Інформація щодо застосування заходів обмеження споживання електроенергії»
- **Format:** CSV, Ukrainian version
- **File name:** `2026_07_01_electricity_outages.csv`

## 2. Electricity import and export

- **Source:** Energy Map, DiXi Group, hourly import and export dataset
- **Website:** https://energy-map.info
- **Format:** CSV, Ukrainian version
- **File name:** `2026_07_01_import_eksport_pohodynno.csv`
- **Use:** context charts in notebook 01 only; the paper does not use this dataset

## 3. Massive strikes on critical infrastructure

- **Source:** Energy Map, DiXi Group
- **Page:** https://energy-map.info/uk/datasets/12f3148d-841a-478d-b9ed-72bf0764b286
- **Title on the site:** «Інформація щодо масованих ударів по критичній інфраструктурі України»
- **Format:** CSV, Ukrainian version
- **File name:** `2026_07_01_udary_krytychna_infrastruktura.csv`

## 4. Weather

- **Source:** Open-Meteo Historical Weather API, licence CC BY 4.0
- **Download link:**
  ```
  https://archive-api.open-meteo.com/v1/archive?latitude=50.45&longitude=30.52&start_date=2023-01-01&end_date=2026-06-30&hourly=temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m&timezone=Europe%2FKyiv&format=csv
  ```
- **File name:** `open-meteo-50.44N30.48E169m.csv`

## Known issues in the raw files

They are handled in `notebooks/01_data_cleaning_eda.ipynb`:

- 328 rows of the outage file, from 1 April 00:00 to 14 April 15:00, 2026, contain spreadsheet formulas instead of the date and hour. The timestamps are restored from the row order.
- The hour 03:00–04:00 on the last Sunday of October is duplicated because of the switch to winter time.
- From 10 December 2025 to 28 January 2026 outage hours were aggregated from distribution system operators and do not reflect the real hourly profile. These hours are excluded from hourly analysis.
- The Open-Meteo file comes with a fixed UTC+2 offset for the whole period, without summer time, see `utc_offset_seconds` in its first rows. The outage and trade files use Kyiv wall-clock time. The notebook converts the weather timestamps to Kyiv time; without this the weather would be shifted by one hour from late March to late October.
- Opened in Excel, the CSV files may look broken because they are UTF-8 encoded. Python reads them correctly.
