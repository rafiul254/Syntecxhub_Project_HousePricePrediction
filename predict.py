import joblib
import pandas as pd

# ── Load Artifacts ───────────────────────────────────────
model    = joblib.load("models/gradient_boosting.pkl")
scaler   = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

print("=" * 55)
print("   🏠 House Price Predictor — Syntecxhub")
print("=" * 55)
print(f"\n✅ Model loaded!")
print(f"📌 Features: {features}\n")

# ── Sample Input ─────────────────────────────────────────
sample = {
    "OverallQual" : 7,        # Overall quality (1-10)
    "GrLivArea"   : 1800,     # Above ground living area (sqft)
    "GarageCars"  : 2,        # Garage capacity (cars)
    "GarageArea"  : 500,      # Garage area (sqft)
    "TotalBsmtSF" : 900,      # Total basement area (sqft)
    "1stFlrSF"    : 1000,     # First floor area (sqft)
    "FullBath"    : 2,        # Full bathrooms
    "TotRmsAbvGrd": 7,        # Total rooms above ground
    "YearBuilt"   : 2005,     # Year house was built
    "YearRemodAdd": 2010,     # Year remodeled
}

# ── Filter to Model Features ─────────────────────────────
input_data   = {f: sample.get(f, 0) for f in features}
input_df     = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
predicted    = model.predict(input_scaled)[0]

print("── Input Details ──")
for k, v in input_data.items():
    print(f"  {k:<20} : {v}")

print(f"\n{'='*55}")
print(f"  💰 Predicted Sale Price : ${predicted:,.0f}")
print(f"{'='*55}")
