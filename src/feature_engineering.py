# src/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_fraud_data(fraud_df, ip_country_df):
    """
    Preprocess the Fraud_Data.csv dataset:
    - Convert timestamps
    - Feature engineering: time_since_signup, hour_of_day, day_of_week
    - Merge with IP geolocation
    """
    # Convert timestamps
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    
    # Feature engineering
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek

    # Merge IP geolocation
    # (Assume IP converted to int as in notebook)
    # fraud_df['ip_address_int'] = ...
    # fraud_df = merge_ip_country(fraud_df, ip_country_df)

    return fraud_df

def scale_and_encode(X_train, X_test, categorical_features=[], numeric_features=[]):
    """
    Apply standard scaling to numeric features and one-hot encoding to categorical features.
    Returns transformed train and test sets.
    """
    # Scaling numeric features
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # One-hot encoding categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_features]), index=X_train.index)
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_features]), index=X_test.index)
    
    # Drop original categorical features and join encoded
    X_train = X_train.drop(columns=categorical_features).join(X_train_encoded)
    X_test = X_test.drop(columns=categorical_features).join(X_test_encoded)
    
    return X_train, X_test
