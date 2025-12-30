import pandas as pd

def add_time_features(fraud_df):
    try:
        fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
        fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
        fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
        return fraud_df
    except Exception as e:
        print(f"Error adding time features: {e}")
        raise

def merge_ip_country(fraud_df, ip_df):
    try:
        if 'ip_address' not in fraud_df.columns or 'lower_bound_ip_address' not in ip_df.columns:
            raise ValueError("Required columns missing for IP-country merge")
        fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
        merged_df = pd.merge_asof(
            fraud_df.sort_values('ip_address'),
            ip_df.sort_values('lower_bound_ip_address'),
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        return merged_df
    except Exception as e:
        print(f"Error merging IP addresses with country: {e}")
        raise
