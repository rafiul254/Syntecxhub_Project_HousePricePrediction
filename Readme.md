# 🏠 House Price Prediction
### Syntecxhub ML Internship — Week 1 | Project 1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-2.0+-green?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge"/>
</p>

<p align="center">
  <b>An end-to-end Machine Learning project that predicts house sale prices using multiple regression models, complete with an interactive Streamlit web application.</b>
</p>

---

## 📸 App Preview

| Predict Tab | Model Performance | Data Insights |
|:-----------:|:-----------------:|:-------------:|
| Interactive sliders for house features | R² & RMSE comparison charts | Dataset statistics & EDA plots |

---

## 📌 Project Overview

This project implements a **complete ML pipeline** to predict house prices using the **Kaggle House Prices dataset**. Built during the **Syntecxhub ML Internship (Week 1)**, it covers every stage of a real-world ML workflow — from raw data exploration to a deployed interactive web app.

### 🎯 What This Project Does
- Loads and explores the Kaggle Housing dataset (EDA)
- Cleans data — handles missing values, encodes categories, removes outliers
- Engineers and selects the top 15 most relevant features
- Trains **5 different ML models** and compares their performance
- Evaluates using **RMSE** and **R² Score** metrics
- Provides a **live Streamlit web app** for real-time predictions

---

## ✨ Features

- ✅ Full EDA with 7 visualizations
- ✅ Automated missing value handling
- ✅ Label encoding for categorical features
- ✅ IQR-based outlier removal
- ✅ Feature importance analysis
- ✅ 5 models trained and compared
- ✅ Interactive Streamlit web app with 3 tabs
- ✅ All models saved for reuse

---

## 🤖 Models Used

| Model | Type | Highlights |
|-------|------|-----------|
| **Linear Regression** | Baseline | Simple, interpretable |
| **Ridge Regression** | Regularized | Prevents overfitting with L2 penalty |
| **Lasso Regression** | Regularized | L1 penalty, auto feature selection |
| **Random Forest** | Ensemble | 100 decision trees, robust |
| **Gradient Boosting** | Ensemble | Sequential learning, highest accuracy |

---

## 📊 Results

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Gradient Boosting | **~0.89** | **~$23,000** |
| Random Forest | ~0.87 | ~$25,000 |
| Ridge Regression | ~0.75 | ~$35,000 |
| Lasso Regression | ~0.75 | ~$35,000 |
| Linear Regression | ~0.75 | ~$35,000 |

> 🏆 **Gradient Boosting** achieved the best performance with ~89% variance explained

---

## 📁 Project Structure

```
Syntecxhub_Project_HousePricePrediction/
│
├── 📄 house_price_prediction.py    ← Main ML pipeline (9 steps)
├── 📄 app.py                       ← Streamlit web application
├── 📄 predict.py                   ← CLI prediction script
├── 📄 requirements.txt             ← Python dependencies
├── 📄 README.md                    ← Project documentation
├── 📄 LICENSE                      ← MIT License
├── 📄 .gitignore                   ← Git ignore rules
│
├── 📂 data/
│   ├── generate_dataset.py         ← Synthetic data generator
│   ├── housing_data.csv            ← Synthetic dataset
│   └── train.csv                   ← Kaggle dataset (primary)
│
├── 📂 models/
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── lasso_regression.pkl
│   ├── random_forest.pkl           ← Best ensemble model
│   ├── gradient_boosting.pkl       ← Best overall model
│   ├── model_results.pkl           ← Comparison results
│   ├── scaler.pkl                  ← Fitted StandardScaler
│   └── features.pkl                ← Selected feature list
│
└── 📂 plots/
    ├── 00_missing_values.png
    ├── 01_price_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_model_evaluation.png
    ├── 04_feature_coefficients.png
    ├── 06_boxplots.png
    └── 07_model_comparison.png
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Syntecxhub_Project_HousePricePrediction.git
cd Syntecxhub_Project_HousePricePrediction
```

### 2. Create Virtual Environment
```bash
# Create
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Kaggle Dataset
Download `train.csv` from [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in the `data/` folder.

---

## 🚀 How to Run

```bash
# Step 1 — Train all models
python house_price_prediction.py

# Step 2 — Launch the web app
streamlit run app.py

# Step 3 — CLI prediction (optional)
python predict.py
```

The Streamlit app will open at **http://localhost:8501**

---

## 🖥️ Streamlit App — 3 Tabs

### 🔮 Tab 1: Predict
- Adjust house features using interactive sliders
- Click **Predict** to get the estimated price
- See all 5 models' predictions side by side

### 📊 Tab 2: Model Performance
- View R² and RMSE comparison table
- Bar charts comparing all models
- Actual vs Predicted scatter plots

### 📈 Tab 3: Data Insights
- Dataset statistics (samples, avg price, max price)
- Price distribution histogram
- Top feature correlations with sale price
- Full EDA visualizations

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## 🧠 Key Learnings

- **OverallQual** (Overall Quality) is the strongest predictor of house price
- **GrLivArea** (Living Area) and **GarageCars** also highly impact price
- **Gradient Boosting** significantly outperforms simple Linear Regression
- Proper data preprocessing (imputation, encoding, scaling) is crucial
- Feature selection reduces noise and improves model generalization

---

## 📚 Dataset

- **Source**: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Samples**: 1,460 rows
- **Features**: 79 original features → top 15 selected
- **Target**: `SalePrice` — house sale price in USD

---

## 👤 Author

**Rafiul Islam**
IoT & Robotics Student | ML Intern @ Syntecxhub
📧 2301012@uftb.ac.bd
🔗 [LinkedIn](https://www.linkedin.com/in/rafiul-islam-25sep92004)
🐙 [GitHub](https://github.com/rafiul254)

---

## 🏢 About Syntecxhub

This project was built as part of the **Syntecxhub ML Internship Program — Week 1**.

🔗 [Syntecxhub on LinkedIn](https://www.linkedin.com/company/syntecxhub/)

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ during Syntecxhub ML Internship 2026
  <br/>
  ⭐ Star this repo if you found it helpful!
</p>
