# California Housing Prices — End-to-End ML Project

> **This README was AI-generated.**

An end-to-end machine learning project following **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron. The goal is to predict **median house values** for California districts using the 1990 Census dataset.

---

## Project Overview

This project walks through the complete machine learning workflow:

1. **Data Fetching & Exploration** — Automatically download and extract the California housing dataset; perform initial statistical analysis and visualization.
2. **Stratified Train/Test Split** — Split the data using `StratifiedShuffleSplit` on income categories to ensure representative sampling.
3. **Exploratory Data Analysis** — Compute Pearson correlation coefficients and visualize them using Seaborn heatmaps.
4. **Feature Engineering** — Create ratio features (`bedroom_ratio`, `rooms_per_house`, `people_per_house`) and analyze their correlation with the target variable.
5. **Preprocessing Pipeline** — Build a modular `ColumnTransformer` pipeline including:
   - `SimpleImputer` (median strategy for numerical, most frequent for categorical)
   - `FunctionTransformer` for log-transformation and ratio computation
   - Custom `ClusterSimilarity` transformer using K-Means + RBF kernel for geolocation features
   - `OneHotEncoder` for categorical features
   - `StandardScaler` for feature normalization
6. **Model Training & Cross-Validation** — Evaluate `LinearRegression`, `DecisionTreeRegressor`, and `RandomForestRegressor` using 10-fold cross-validation.
7. **Hyperparameter Tuning** — Fine-tune `RandomForestRegressor` using `RandomizedSearchCV` over preprocessing and model hyperparameters simultaneously.
8. **Final Evaluation & Model Export** — Evaluate the tuned model on the held-out test set and serialize the full pipeline using `joblib`.

---

## Results Summary

| Stage | Model | RMSE |
| :--- | :--- | :--- |
| Cross-Validation (10-fold) | Linear Regression | ~$68,193 |
| Cross-Validation (10-fold) | Decision Tree | ~$67,018 |
| Cross-Validation (10-fold) | Random Forest | ~$47,124 |
| After Tuning (5-fold CV) | Random Forest (tuned) | ~$41,685 |
| **Final Test Set** | **Random Forest (tuned)** | **~$41,598** |

The tuned Random Forest achieved a **~$5,500 improvement** over the untuned baseline and showed **no overfitting** (test RMSE ≈ validation RMSE).

---

## Project Structure

```
California_housing_prices/
├── california_housinng_prices.ipynb   # Main Jupyter notebook
├── my_california_house_value_predictor.pkl  # Serialized final model (not tracked by Git)
├── data/                              # Downloaded dataset (not tracked by Git)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## Setup & Installation

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook california_housinng_prices.ipynb
```

The notebook will automatically download the dataset on first run.

---

## Key Concepts Covered

- **Stratified Sampling** to prevent sampling bias
- **Correlation Analysis** and feature engineering with ratio attributes
- **Custom Scikit-Learn Transformers** (`BaseEstimator` + `TransformerMixin`)
- **K-Means Clustering + RBF Kernel** for geolocation feature engineering
- **Scikit-Learn Pipelines** (`make_pipeline`, `ColumnTransformer`)
- **Cross-Validation** for unbiased model evaluation
- **Hyperparameter Tuning** with `RandomizedSearchCV`
- **Model Serialization** with `joblib`

---

## Reference

- **Book**: *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron
- **Chapter**: 2 — End-to-End Machine Learning Project
- **Dataset**: California Housing Prices (1990 Census)
