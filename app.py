import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4C72B0;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown('<p class="main-header">🏠 House Price Predictor</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Syntecxhub ML Internship — Project 1 '
    '| Powered by Machine Learning</p>',
    unsafe_allow_html=True
)

# ── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_files = {
        "Linear Regression" : "models/linear_regression.pkl",
        "Ridge Regression"  : "models/ridge_regression.pkl",
        "Lasso Regression"  : "models/lasso_regression.pkl",
        "Random Forest"     : "models/random_forest.pkl",
        "Gradient Boosting" : "models/gradient_boosting.pkl",
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)

    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    return loaded, scaler, features

model_loaded = False
try:
    models, scaler, features = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}\n\n"
             f"Please run `house_price_prediction.py` first!")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

if model_loaded:
    selected_model = st.sidebar.selectbox(
        "🤖 Select Model", list(models.keys())
    )
else:
    selected_model = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    "Predicts house sale prices using ML models "
    "trained on the Kaggle Housing dataset.\n\n"
    "**Intern:** Rafi Ul Islam\n\n"
    "**Company:** Syntecxhub"
)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict",
    "📊 Model Performance",
    "📈 Data Insights"
])

with tab1:
    st.subheader("🏡 Enter House Details")

    if not model_loaded:
        st.warning("Run `house_price_prediction.py` first to train and save the models.")
    else:
        st.info(f"📌 Model uses **{len(features)} features**: "
                f"`{', '.join(features)}`")

        # Default slider config for known Kaggle features
        SLIDER_CONFIG = {
            "OverallQual"  : (1,    10,    7,    1),
            "GrLivArea"    : (300,  5000,  1500, 50),
            "GarageCars"   : (0,    4,     2,    1),
            "GarageArea"   : (0,    1500,  500,  10),
            "TotalBsmtSF"  : (0,    3000,  900,  10),
            "1stFlrSF"     : (300,  4000,  1000, 10),
            "FullBath"     : (0,    4,     2,    1),
            "TotRmsAbvGrd" : (2,    14,    7,    1),
            "YearBuilt"    : (1872, 2010,  1990, 1),
            "YearRemodAdd" : (1950, 2010,  2000, 1),
            "MasVnrArea"   : (0,    1600,  100,  10),
            "Fireplaces"   : (0,    4,     1,    1),
            "BsmtFinSF1"   : (0,    2000,  400,  10),
            "LotFrontage"  : (20,   300,   70,   1),
            "WoodDeckSF"   : (0,    800,   100,  10),
            "OpenPorchSF"  : (0,    500,   50,   5),
            "LotArea"      : (1000, 50000, 8000, 100),
            "BsmtUnfSF"    : (0,    2000,  400,  10),
            "2ndFlrSF"     : (0,    2000,  500,  10),
            "BedroomAbvGr" : (0,    8,     3,    1),
        }

        # Dynamically build sliders from saved features
        user_inputs = {}
        chunks = [features[i:i+3] for i in range(0, len(features), 3)]

        for chunk in chunks:
            cols = st.columns(3)
            for col, feat in zip(cols, chunk):
                with col:
                    if feat in SLIDER_CONFIG:
                        mn, mx, default, step_val = SLIDER_CONFIG[feat]
                        user_inputs[feat] = st.slider(
                            feat, mn, mx, default, step_val
                        )
                    else:
                        user_inputs[feat] = st.number_input(
                            feat, value=0.0, step=1.0
                        )

        st.markdown("---")

        if st.button("🔮 Predict House Price", use_container_width=True):
            try:
                input_df     = pd.DataFrame(
                    [[user_inputs[f] for f in features]],
                    columns=features
                )
                input_scaled = scaler.transform(input_df)
                prediction   = models[selected_model].predict(input_scaled)[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Predicted House Price</h2>
                    <h1>${prediction:,.0f}</h1>
                    <p>Model used: <b>{selected_model}</b></p>
                </div>
                """, unsafe_allow_html=True)

                # All models side by side
                st.markdown("### 🤖 All Models Comparison")
                comp_cols = st.columns(len(models))
                for i, (name, mdl) in enumerate(models.items()):
                    pred = mdl.predict(input_scaled)[0]
                    with comp_cols[i]:
                        st.metric(label=name, value=f"${pred:,.0f}")

            except Exception as e:
                st.error(f"Prediction error: {e}")



with tab2:
    st.subheader("📊 Model Performance Comparison")

    if os.path.exists("models/model_results.pkl"):
        results_df = joblib.load("models/model_results.pkl")

        st.dataframe(results_df, use_container_width=True)

        # Bar charts
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467bd"]

        axes[0].barh(results_df["Model"], results_df["Test R²"],
                     color=colors[:len(results_df)])
        axes[0].set_title("R² Score (higher = better)")
        axes[0].set_xlabel("R² Score")
        axes[0].set_xlim(0, 1)
        for i, v in enumerate(results_df["Test R²"]):
            axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

        axes[1].barh(results_df["Model"], results_df["Test RMSE"],
                     color=colors[:len(results_df)])
        axes[1].set_title("RMSE — $ (lower = better)")
        axes[1].set_xlabel("RMSE ($)")
        for i, v in enumerate(results_df["Test RMSE"]):
            axes[1].text(v + 100, i, f"${v:,.0f}", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📈 Saved Plots")
        plot_cols = st.columns(2)
        for i, path in enumerate([
            "plots/03_model_evaluation.png",
            "plots/07_model_comparison.png",
            "plots/04_feature_coefficients.png",
            "plots/02_correlation_heatmap.png",
        ]):
            if os.path.exists(path):
                with plot_cols[i % 2]:
                    st.image(path, use_container_width=True)
    else:
        st.warning("⚠️ Run `house_price_prediction.py` first.")


with tab3:
    st.subheader("📈 Dataset Insights")

    csv_path = "data/train.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples",  f"{len(df):,}")
        c2.metric("Total Features", f"{df.shape[1]-1}")
        c3.metric("Avg Price",      f"${df['SalePrice'].mean():,.0f}")
        c4.metric("Max Price",      f"${df['SalePrice'].max():,.0f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💰 Price Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["SalePrice"], bins=50,
                    color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Sale Price ($)")
            ax.set_ylabel("Count")
            ax.set_title("Sale Price Distribution")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**🔗 Top Feature Correlations with Price**")
            num_df   = df.select_dtypes(include=[np.number])
            corr     = num_df.corr()["SalePrice"].drop("SalePrice")
            corr_top = corr.abs().nlargest(10)
            corr_top = corr_top.sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            colors_bar = ["#2ecc71" if corr[i] > 0 else "#e74c3c"
                          for i in corr_top.index]
            ax.barh(corr_top.index, corr_top.values,
                    color=colors_bar, edgecolor="white")
            ax.set_xlabel("Absolute Correlation")
            ax.set_title("Top 10 Features vs SalePrice")
            plt.tight_layout()
            st.pyplot(fig)

        # Saved plots
        st.markdown("---")
        st.markdown("### 📊 EDA Plots")
        eda_cols = st.columns(2)
        for i, path in enumerate([
            "plots/01_price_distribution.png",
            "plots/06_boxplots.png",
            "plots/00_missing_values.png",
            "plots/05_pairplot.png",
        ]):
            if os.path.exists(path):
                with eda_cols[i % 2]:
                    st.image(path, use_container_width=True)

        # Raw data preview
        with st.expander("🔍 View Raw Dataset (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("⚠️ `data/train.csv` not found!")


# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "🏠 House Price Predictor | Syntecxhub ML Internship 2026 | Rafi Ul Islam"
    "</p>",
    unsafe_allow_html=True
)
