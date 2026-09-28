# Data

| File | Description | Source |
|---|---|---|
| `raw/Tourism.csv` | Country-year tourism and economic indicators (arrivals, receipts, GDP, inflation, and more), 1999 onward | [Tourism and Economic Impact (Kaggle)](https://www.kaggle.com/datasets/bushraqurban/tourism-and-economic-impact) |
| `raw/Climate.csv` | Annual mean surface temperature by country, with a 5-year smoothed value | [Berkeley Earth, Climate Change: Earth Surface Temperature Data (Kaggle)](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data) |
| `processed/cleaned_data.csv` | Merged, cleaned country-year panel (145 countries, 2000 to 2020) with engineered features: growth rates, receipts per arrival, temperature change and `trend_score` | Built by the cleaning section of the notebook |

`trend_score = 0.6 × arrivals_growth + 0.4 × rpa_growth` (growth in arrivals and in receipts per arrival).

Please check each source's license before reusing the raw data.
