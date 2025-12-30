# Fraud Detection: E-commerce & Bank Transactions

## Project Overview
This project focuses on improving the detection of fraudulent transactions for both e-commerce and bank credit datasets. By combining advanced machine learning models, feature engineering, and explainable AI techniques, the project aims to:

- Detect fraud accurately while balancing false positives and negatives.
- Integrate geolocation analysis and transaction patterns.
- Provide actionable business insights based on model predictions.

## Business Context
Fraudulent transactions lead to financial loss and reduce customer trust. Accurate fraud detection is critical for:

- E-commerce platforms
- Banking and credit institutions

By leveraging machine learning and SHAP explainability, Adey Innovations Inc. can proactively prevent fraud and improve transaction security.

## Datasets
1. **Fraud_Data.csv** – E-commerce transactions  
   - Key features: `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`  

2. **creditcard.csv** – Bank credit transactions  
   - Key features: `Time`, `V1`–`V28` (anonymized), `Amount`, `Class`  

3. **IpAddress_to_Country.csv** – Maps IP addresses to countries for geolocation analysis

## Project Structure
fraud-detection/
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and feature-engineered datasets
├── notebooks/
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ ├── modeling.ipynb
│ ├── shap-explainability.ipynb
│ └── README.md
├── src/ # Reusable scripts and functions
├── tests/
├── models/ # Saved model artifacts
├── scripts/
├── requirements.txt
├── README.md
└── .gitignore

## Key Features Engineered
- **Time-based features:** `hour_of_day`, `day_of_week`, `time_since_signup`
- **Transaction frequency & velocity:** Number of transactions per user within a time window
- **Geolocation-based features:** Country mapping from IP addresses

## Data Preprocessing
- Handling missing values
- Duplicate removal
- Data type corrections
- Encoding categorical features
- Scaling numerical features
- Handling class imbalance using **SMOTE** for training set

## Modeling
- **Baseline model:** Logistic Regression
- **Ensemble model:** XGBoost
- Hyperparameter tuning (`n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`)
- Evaluation metrics:
  - F1-Score
  - AUC-PR
  - Confusion Matrix
- Cross-validation with **Stratified K-Folds (k=5)**

## Model Explainability
- **SHAP analysis:**
  - Global feature importance
  - Decision plots for TP, FP, FN
  - Force plots (interactive) for individual predictions
- **Business recommendations** derived from top SHAP features

## How to Run
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd fraud-detection
Install dependencies:

pip install -r requirements.txt


Open notebooks in Jupyter or VSCode.

Follow the notebook sequence:

eda-fraud-data.ipynb → EDA for Fraud_Data.csv

eda-creditcard.ipynb → EDA for creditcard.csv

feature-engineering.ipynb → Feature engineering & preprocessing

modeling.ipynb → Model training, evaluation, comparison

shap-explainability.ipynb → Model explainability & business insights

References

Kaggle Credit Card Fraud Dataset

Kaggle IEEE Fraud Detection

SHAP Documentation: https://github.com/slundberg/shap