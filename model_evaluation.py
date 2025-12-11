import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import boto3, io

bucket = 'ml-autoscaling-dataset'
s3 = boto3.client('s3')
prefix_ml = 'ml_predictive/'

ml_key = 'ml_predictive/ml_asg_combined_i-047ec14915c9a2b37.csv' # change to your instance ID
print(f"Using fixed ML file: {ml_key}")
merged_df = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=ml_key)['Body'].read()))

# Ensure CPU_Usage exists
if 'CPU_Usage' not in merged_df.columns:
    if 'CPU_User' in merged_df.columns and 'CPU_System' in merged_df.columns:
        merged_df['CPU_Usage'] = merged_df['CPU_User'] + merged_df['CPU_System']
        print("[+] Derived CPU_Usage from CPU_User + CPU_System")
    elif 'CPU_User' in merged_df.columns:
        merged_df['CPU_Usage'] = merged_df['CPU_User']
        print("Used CPU_User as CPU_Usage")
    else:
        raise KeyError("CPU_Usage column not found or derivable.")

# === Clean and separate ===
merged_df['Model'] = merged_df['Model'].fillna('unknown').str.lower()
merged_df = merged_df[merged_df['Model'].isin(['prophet', 'lstm'])]
merged_df = merged_df.dropna(subset=['CPU_Usage', 'Predicted_CPU'])

# Split by model
prophet_df = merged_df[merged_df['Model'] == 'prophet']
lstm_df = merged_df[merged_df['Model'] == 'lstm']

# === Calculate RMSE ===
rmse_prophet = np.sqrt(mean_squared_error(prophet_df['CPU_Usage'], prophet_df['Predicted_CPU']))
rmse_lstm = np.sqrt(mean_squared_error(lstm_df['CPU_Usage'], lstm_df['Predicted_CPU']))

# === Plot ===
plt.figure(figsize=(12, 6))

plt.plot(prophet_df['Timestamp'], prophet_df['CPU_Usage'], label='Actual CPU', color='black', alpha=0.6)
plt.plot(prophet_df['Timestamp'], prophet_df['Predicted_CPU'], label=f'Prophet Prediction (RMSE={rmse_prophet:.2f})', linestyle='--', color='blue')
plt.plot(lstm_df['Timestamp'], lstm_df['Predicted_CPU'], label=f'LSTM Prediction (RMSE={rmse_lstm:.2f})', linestyle='--', color='green')

plt.axhline(70, color='red', linestyle='--', label='Threshold (70%)')
plt.title("Forecasting Accuracy: Prophet vs LSTM")
plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()