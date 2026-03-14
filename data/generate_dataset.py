import numpy as np
import pandas as pd
import os

# ── Reproducibility ─────────────────────────────────────
np.random.seed(42)
n = 5000

# ── Generate Features ───────────────────────────────────
med_inc    = np.random.lognormal(mean=1.5, sigma=0.6, size=n).clip(0.5, 15)
house_age  = np.random.uniform(1, 52, n)
ave_rooms  = np.random.normal(5.5, 2.0, n).clip(1, 15)
ave_bedrms = (ave_rooms * np.random.uniform(0.15, 0.35, n)).clip(0.5, 5)
population = np.random.lognormal(6, 1, n).clip(3, 35000)
ave_occup  = np.random.lognormal(1.1, 0.4, n).clip(1, 10)
latitude   = np.random.uniform(32.5, 42.0, n)
longitude  = np.random.uniform(-124.5, -114.0, n)

# ── Generate Target (Price) ──────────────────────────────
# Price is mostly driven by income + location
price = (
    0.5  * med_inc
  - 0.002 * house_age
  + 0.05  * ave_rooms
  - 0.04  * ave_bedrms
  - 0.0001 * population
  - 0.1   * ave_occup
  - 0.05  * (latitude - 37)
  + np.random.normal(0, 0.3, n)   # noise
).clip(0.2, 5.5)

# ── Build DataFrame ─────────────────────────────────────
df = pd.DataFrame({
    "MedInc"    : med_inc.round(4),
    "HouseAge"  : house_age.round(1),
    "AveRooms"  : ave_rooms.round(4),
    "AveBedrms" : ave_bedrms.round(4),
    "Population": population.round(0),
    "AveOccup"  : ave_occup.round(4),
    "Latitude"  : latitude.round(4),
    "Longitude" : longitude.round(4),
    "Price"     : price.round(4)
})

# ── Save CSV ─────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(__file__), "housing_data.csv")
df.to_csv(save_path, index=False)

print(f"✅ Dataset generated  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"💾 Saved → {save_path}")
print(f"\nPreview:")
print(df.head())
print(f"\nBasic Stats:")
print(df.describe().round(2))
