import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split
from sklearn.linear_model      import LinearRegression, Ridge, Lasso
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import mean_squared_error, r2_score
from sklearn.impute            import SimpleImputer

warnings.filterwarnings("ignore")

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

STEP = 0
def step(msg):
    global STEP; STEP += 1
    print(f"\n{'='*60}\n  STEP {STEP}: {msg}\n{'='*60}")


step("Load Kaggle Dataset")

df = pd.read_csv("data/train.csv")
print(f"✅ Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nFirst 3 rows:\n{df.head(3)}")


step("Exploratory Data Analysis")

print(f"\nTarget column: SalePrice")
print(df["SalePrice"].describe().round(2))

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\n── Top 10 Missing Value Columns ──")
print(missing.head(10))

# Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("House Sale Price Distribution", fontsize=14, fontweight="bold")
axes[0].hist(df["SalePrice"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Original Distribution")
axes[0].set_xlabel("Sale Price ($)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(df["SalePrice"]), bins=50, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].set_title("Log Transformed Distribution")
axes[1].set_xlabel("log(SalePrice + 1)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("plots/01_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → 01_price_distribution.png")

# Missing value heatmap
if len(missing) > 0:
    plt.figure(figsize=(12, 5))
    missing_pct = (missing / len(df) * 100).head(20)
    plt.barh(missing_pct.index[::-1], missing_pct.values[::-1],
             color="#e74c3c", edgecolor="white", alpha=0.85)
    plt.xlabel("Missing %")
    plt.title("Top 20 Columns with Missing Values", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/00_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved → 00_missing_values.png")


step("Data Cleaning & Preprocessing")

# Drop Id column
df.drop(columns=["Id"], inplace=True, errors="ignore")

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target from feature lists
if "SalePrice" in num_cols:
    num_cols.remove("SalePrice")

print(f"Numerical features  : {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# Fill missing values
# Numerical → fill with median
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical → fill with most frequent
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print(f"\n✅ Missing values filled")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

# Encode categorical columns using Label Encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"✅ Categorical columns encoded")


step("Feature Engineering")

# Outlier removal on SalePrice using IQR
Q1, Q3 = df["SalePrice"].quantile(0.25), df["SalePrice"].quantile(0.75)
IQR    = Q3 - Q1
before = len(df)
df     = df[
    (df["SalePrice"] >= Q1 - 1.5*IQR) &
    (df["SalePrice"] <= Q3 + 1.5*IQR)
]
print(f"✅ Outliers removed → {before - len(df)} rows | {len(df):,} remaining")

# Feature selection → top 15 correlated features with SalePrice
all_features   = num_cols + cat_cols
corr_matrix    = df[all_features + ["SalePrice"]].corr()
corr_with_price = corr_matrix["SalePrice"].abs().drop("SalePrice")
selected_features = corr_with_price.nlargest(15).index.tolist()

print(f"\n✅ Top 15 Selected Features:")
for i, f in enumerate(selected_features, 1):
    print(f"   {i:>2}. {f:<25} (corr: {corr_with_price[f]:.3f})")

# Correlation Heatmap (selected features only)
plt.figure(figsize=(12, 10))
heatmap_df = df[selected_features + ["SalePrice"]]
corr = heatmap_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=np.triu(np.ones_like(corr, dtype=bool)),
            linewidths=0.5, vmin=-1, vmax=1)
plt.title("Top Features Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Saved → 02_correlation_heatmap.png")

# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Boxplots — Outlier Detection", fontsize=13, fontweight="bold")
for ax, col in zip(axes, ["SalePrice", "GrLivArea", "OverallQual"]):
    if col in df.columns:
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        ax.set_title(col)
plt.tight_layout()
plt.savefig("plots/06_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → 06_boxplots.png")


step("Train / Test Split + Scaling")

X = df[selected_features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set : {X_train.shape[0]:,} samples")
print(f"Test set     : {X_test.shape[0]:,} samples")
print("✅ StandardScaler applied")

step("Train Multiple Models")

MODELS = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=10.0),
    "Lasso Regression"  : Lasso(alpha=0.01),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results        = []
trained_models = {}

for name, model in MODELS.items():
    print(f"\n  Training: {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test,  y_pred_test)

    results.append({
        "Model"     : name,
        "Train RMSE": round(train_rmse, 2),
        "Test RMSE" : round(test_rmse,  2),
        "Train R²"  : round(train_r2,   4),
        "Test R²"   : round(test_r2,    4),
    })
    trained_models[name] = model
    print(f"  ✅ Test R²: {test_r2:.4f} | RMSE: ${test_rmse:,.0f}")

results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False)
print(f"\n── Model Comparison ──\n{results_df.to_string(index=False)}")

step("Model Comparison Visualization")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467bd"]

axes[0].barh(results_df["Model"], results_df["Test R²"], color=colors)
axes[0].set_title("R² Score (higher = better)")
axes[0].set_xlabel("R² Score")
axes[0].set_xlim(0, 1)

axes[1].barh(results_df["Model"], results_df["Test RMSE"], color=colors)
axes[1].set_title("RMSE in $ (lower = better)")
axes[1].set_xlabel("RMSE ($)")

plt.tight_layout()
plt.savefig("plots/07_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → 07_model_comparison.png")

# Best model — Actual vs Predicted
best_name  = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
y_pred     = best_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Best Model: {best_name}", fontsize=14, fontweight="bold")

axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", s=15)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title(f"Actual vs Predicted (R²={results_df.iloc[0]['Test R²']:.3f})")
axes[0].legend()

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, color="#DD8452", s=15)
axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residuals ($)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("plots/03_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → 03_model_evaluation.png")

# Feature importance (Random Forest)
if "Random Forest" in trained_models:
    rf          = trained_models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=selected_features)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    colors_fi = ["#2ecc71" if v > importances.median() else "#4C72B0"
                 for v in importances.values]
    plt.barh(importances.index, importances.values,
             color=colors_fi, edgecolor="white")
    plt.title("Feature Importance — Random Forest",
              fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/04_feature_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved → 04_feature_coefficients.png")

step("Save Models & Artifacts")

for name, model in trained_models.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{fname}.pkl")
    print(f"✅ Saved → models/{fname}.pkl")

joblib.dump(scaler,            "models/scaler.pkl")
joblib.dump(selected_features, "models/features.pkl")
joblib.dump(results_df,        "models/model_results.pkl")
print("✅ Saved → scaler, features, model_results")


step(f"Example Predictions — {best_name}")

sample        = X_test.iloc[:5].copy()
sample_scaled = scaler.transform(sample)
preds         = best_model.predict(sample_scaled)
actuals       = y_test.iloc[:5].values

print(f"\n{'#':<4} {'Actual':>12} {'Predicted':>12} {'Error %':>10}")
print("-" * 42)
for i, (act, pred) in enumerate(zip(actuals, preds), 1):
    err = abs(act - pred) / act * 100
    print(f"{i:<4} ${act:>10,.0f}  ${pred:>10,.0f}  {err:>9.1f}%")

print(f"\n{'='*60}")
print(f"  ✅ PROJECT COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  Test R²    : {results_df.iloc[0]['Test R²']}")
print(f"  Test RMSE  : ${results_df.iloc[0]['Test RMSE']:,.0f}")
print(f"{'='*60}")
