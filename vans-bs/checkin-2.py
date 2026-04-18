import pandas as pd
import yearData as yd

data = pd.read_csv("./vans-listening-data_2021-2026.csv")

data24 = yd.getYearData(data, 2024)
data25 = yd.getYearData(data, 2025)

data24.to_csv("vans-data-2024.csv", index=False)
data25.to_csv("vans-data-2025.csv", index=False)