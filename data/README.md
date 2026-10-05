# NOAA observations

Source: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/ . Schema and units: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/readme.txt . Frozen manifest contains station/year HTTPS URLs and SHA-256 for 27 real files, 2016–2024. NOAA may revise historical observations; hash mismatches must be investigated explicitly. NOAA-produced data are public domain in the United States; see https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf . Attribute NOAA USCRN/NCEI.

Missing sentinels (including -99 for soil moisture) become NaN, physically impossible moisture/RH and negative rainfall/radiation are excluded. Calendar reindexing precedes lags. No synthetic study observations, irrigation labels, crop identities or field-management facts are created. Raw source copies stay ignored locally.
