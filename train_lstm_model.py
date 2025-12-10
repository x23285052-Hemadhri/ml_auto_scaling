import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta, timezone
import pickle
import os

# === CONFIGURATION ===
REGION = 'us-east-2' # Replace with your AWS region
BUCKET_NAME = 'ml-autoscaling-dataset'
MODEL_KEY = 'ml_predictive/lstm_cpu_model.h5'
SCALER_KEY = 'ml_predictive/lstm_scaler.pkl'
INSTANCE_ID = ' i-047ec14915c9a2b37' # Replace with your instance ID
PERIOD = 60
DURATION_MINS = 180
SEQUENCE_LENGTH = 10

# === AWS CLIENTS ===
s3 = boto3.client('s3', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)

# === METRICS CONFIGURATION ===
metrics_to_fetch = {
    'cpu_usage_user': 'CPU_User',
    'cpu_usage_system': 'CPU_System',
    'mem_used_percent': 'Memory_Usage',
    'disk_used_percent': 'Disk_Usage',
    'net_bytes_recv': 'Network_In',
    'net_bytes_sent': 'Network_Out'
}

dimension_map = {
    'cpu_usage_user': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'cpu_usage_system': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'mem_used_percent': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}],
    'disk_used_percent': [
        {'Name': 'InstanceId', 'Value': INSTANCE_ID},
        {'Name': 'device', 'Value': 'nvme0n1p1'},
        {'Name': 'path', 'Value': '/'},
        {'Name': 'fstype', 'Value': 'xfs'}
    ],
    'net_bytes_recv': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'interface', 'Value': 'ens5'}],
    'net_bytes_sent': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'interface', 'Value': 'ens5'}]
}

# === TIME RANGE ===
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(minutes=DURATION_MINS)

# === FETCH METRIC FUNCTION ===
def fetch_metric(metric_name):
    dims = dimension_map[metric_name]
    try:
        response = cloudwatch.get_metric_statistics(
            Namespace='CWAgent',
            MetricName=metric_name,
            Dimensions=dims,
            StartTime=start_time,
            EndTime=end_time,
            Period=PERIOD,
            Statistics=['Average']
        )
        datapoints = response.get('Datapoints', [])
        if not datapoints:
            print(f"No data for {metric_name}")
            return pd.DataFrame()
        return pd.DataFrame([
            {'Timestamp': dp['Timestamp'], metrics_to_fetch[metric_name]: dp['Average']}
            for dp in datapoints
        ])
    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame()

# === COLLECT METRICS ===
final_df = pd.DataFrame()
for metric in metrics_to_fetch:
    df = fetch_metric(metric)
    if not df.empty:
        final_df = df if final_df.empty else pd.merge(final_df, df, on='Timestamp', how='outer')

if final_df.empty:
    print("No data fetched. Exiting.")
    exit()

final_df = final_df.sort_values('Timestamp').dropna()

# === PREPARE FEATURES ===
features = list(metrics_to_fetch.values())
data = final_df[features].copy()

# === NORMALIZE ===
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === SEQUENCE BUILD ===
X, y = [], []
for i in range(len(data_scaled) - SEQUENCE_LENGTH):
    X.append(data_scaled[i:i + SEQUENCE_LENGTH])
    y.append(data_scaled[i + SEQUENCE_LENGTH][0])  # Predict CPU_User

X = np.array(X)
y = np.array(y)

print(f"Training data prepared: {X.shape[0]} samples, {X.shape[2]} features")

# === LSTM MODEL ===
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16)

# === SAVE MODEL & SCALER ===
model.save('lstm_cpu_model.h5')
with open('lstm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# === UPLOAD TO S3 ===
s3.upload_file('lstm_cpu_model.h5', BUCKET_NAME, MODEL_KEY)
s3.upload_file('lstm_scaler.pkl', BUCKET_NAME, SCALER_KEY)

print(f"LSTM model and scaler uploaded to s3://{BUCKET_NAME}/{MODEL_KEY}")
