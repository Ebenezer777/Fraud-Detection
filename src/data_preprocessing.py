# src/data_preprocessing.py
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(fraud_file, credit_file, ip_file):
    fraud_df = pd.read_csv(fraud_file)
    credit_df = pd.read_csv(credit_file)
    ip_df = pd.read_csv(ip_file)
    return fraud_df, credit_df, ip_df

def clean_fraud_data(fraud_df):
    # Convert timestamps
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    
    # Drop duplicates
    fraud_df = fraud_df.drop_duplicates()
    
    # Fill missing values if any
    fraud_df = fraud_df.fillna({
        'age': fraud_df['age'].median(),
        'sex': 'U',
        'browser': 'Unknown',
        'source': 'Unknown'
    })
    
    return fraud_df

def add_time_features(fraud_df):
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    return fraud_df

def handle_class_imbalance(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def scale_numeric_features(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler

def encode_categorical_features(X_train, X_test, cat_cols):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(
        encoder.fit_transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )
    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )
    # Drop original categorical cols and join encoded
    X_train = X_train.drop(columns=cat_cols).join(X_train_encoded)
    X_test = X_test.drop(columns=cat_cols).join(X_test_encoded)
    return X_train, X_test, encoder
