Fraud Detection Project — Adey Innovations Inc.
1. Project Overview

This project aims to improve fraud detection for e-commerce and bank credit transactions using machine learning. It focuses on detecting fraudulent transactions accurately while balancing false positives and false negatives. The project uses transaction patterns, geolocation analysis, and time-based features to improve detection performance.

2. Business Context

Fraud detection is crucial in fintech and e-commerce:

Prevents financial losses

Maintains trust with customers and financial institutions

Supports real-time transaction monitoring

The challenge is managing the trade-off between security (catching fraud) and user experience (avoiding false positives).

3. Data Description
E-commerce Dataset (Fraud_Data.csv)

user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class (target)

Imbalanced dataset: far fewer fraudulent transactions

Bank Dataset (creditcard.csv)

Time, V1–V28 (PCA features), Amount, Class (target)

Imbalanced dataset, typical for fraud detection

IP-to-Country Mapping (IpAddress_to_Country.csv)

lower_bound_ip_address, upper_bound_ip_address, country

4. Project Structure
fraud-detection/
├── data/
│   ├── raw/                     # Original datasets (ignored in git)
│   └── processed/               # Cleaned and feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                          # Optional helper scripts
├── models/                       # Saved trained models
├── requirements.txt
├── README.md
└── .gitignore

5. Key Steps Implemented

Data Cleaning & Preprocessing

Missing value handling, duplicates removal, data type corrections

Exploratory Data Analysis (EDA)

Univariate and bivariate analysis

Class imbalance visualization

Feature Engineering

Time-based features: time_since_signup, hour_of_day, day_of_week

Transaction frequency/velocity

IP-to-country geolocation mapping

Data Transformation

Standard scaling of numeric features

One-hot encoding of categorical features

Class Imbalance Handling

SMOTE applied to training set only

Modeling & Evaluation

Logistic Regression baseline

XGBoost/Random Forest ensemble models

Stratified train-test split and cross-validation

Model Explainability (SHAP)

Global feature importance

Force plots for TP, FP, FN

Decision plots for top features

6. Setup Instructions

Clone the repository:

git clone <your_repo_link>
cd fraud-detection


Install dependencies:

pip install -r requirements.txt


Launch Jupyter Notebook:

jupyter notebook


Open and run the notebooks in order:

eda-fraud-data.ipynb

eda-creditcard.ipynb

feature-engineering.ipynb

modeling.ipynb

shap-explainability.ipynb

7. Notes

All raw data is excluded from the repository (see .gitignore)

Processed datasets are saved in data/processed/

SMOTE and feature engineering are applied only to training data

SHAP plots require a working JavaScript-enabled notebook kernel