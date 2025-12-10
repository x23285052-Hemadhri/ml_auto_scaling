import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import pickle
import os
import sys

# === CONFIGURATION ===
REGION = 'us-east-2'
BUCKET_NAME = 'ml-autoscaling-dataset'
ASG_NAME = 'ML-ASG'
LSTM_MODEL_KEY = 'ml_predictive/lstm_cpu_model.h5'
SCALER_KEY = 'ml_predictive/lstm_scaler.pkl'
PROPHET_MODEL_KEY = 'ml_predictive/prophet_cpu_model.pkl'
LOG_FILE = 'ml_scaling_logs.csv'
LOG_KEY = 'ml_predictive/ml_scaling_logs.csv'
INSTANCE_ID = 'i-047ec14915c9a2b37' # Replace with your instance ID
SEQUENCE_LENGTH = 10
THRESHOLD_UP = 70
THRESHOLD_DOWN = 70
DURATION_MINS = 30
PERIOD = 60
COOLDOWN_SECONDS = 300
MODEL_NAME = 'hybrid'

print(f"[INFO] Starting ML Predictive Auto-Scaling Controller ({MODEL_NAME.upper()}) for instance {INSTANCE_ID}")

# === AWS CLIENTS ===
s3 = boto3.client('s3', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
asg = boto3.client('autoscaling', region_name=REGION)

# === DOWNLOAD MODELS ===
try:
    s3.download_file(BUCKET_NAME, LSTM_MODEL_KEY, 'lstm_cpu_model.h5')
    s3.download_file(BUCKET_NAME, SCALER_KEY, 'lstm_scaler.pkl')
    s3.download_file(BUCKET_NAME, PROPHET_MODEL_KEY, 'prophet_cpu_model.pkl')
    
    lstm_model = load_model('lstm_cpu_model.h5')
    with open('lstm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('prophet_cpu_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    
    print("LSTM and Prophet models loaded.")
except Exception as e:
    print(f"Failed to load model or scaler: {e}")
    sys.exit(1)

# === METRICS ===
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

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(minutes=DURATION_MINS)

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
        return pd.DataFrame([
            {'Timestamp': dp['Timestamp'], metrics_to_fetch[metric_name]: dp['Average']}
            for dp in response.get('Datapoints', [])
        ])
    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame()

# === FETCH DATA ===
final_df = pd.DataFrame()
for metric in metrics_to_fetch:
    df = fetch_metric(metric)
    if not df.empty:
        final_df = df if final_df.empty else pd.merge(final_df, df, on='Timestamp', how='outer')

if final_df.empty:
    print("No data collected.")
    sys.exit(0)

final_df = final_df.sort_values('Timestamp').dropna()
data = final_df[list(metrics_to_fetch.values())].copy()

# === LSTM PREDICTION ===
data_scaled = scaler.transform(data)
if len(data_scaled) < SEQUENCE_LENGTH:
    print("[!] Not enough data for LSTM sequence.")
    sys.exit(0)

X_input = data_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(metrics_to_fetch))
lstm_pred = lstm_model.predict(X_input)[0][0] * 100

# === PROPHET PREDICTION ===
prophet_df = final_df[['Timestamp', 'CPU_User']].rename(columns={'Timestamp': 'ds', 'CPU_User': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
future = pd.DataFrame({'ds': [prophet_df['ds'].max() + timedelta(minutes=1)]})
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast['yhat'].iloc[0]

# === COMBINE ===
final_pred = 0.7 * lstm_pred + 0.3 * prophet_pred
print(f"Hybrid CPU Prediction → LSTM: {lstm_pred:.2f}%, Prophet: {prophet_pred:.2f}%, Final: {final_pred:.2f}%")

# === SCALING ===
cooldown_file = 'last_scale_time.txt'
last_scale_time = None
now = datetime.utcnow()
if os.path.exists(cooldown_file):
    with open(cooldown_file, 'r') as f:
        last_scale_time = datetime.fromisoformat(f.read().strip())

should_scale = False
action = 'no_action'
if last_scale_time is None or (now - last_scale_time).total_seconds() > COOLDOWN_SECONDS:
    if final_pred > THRESHOLD_UP:
        print("[ML Decision] CPU > threshold → scale up")
        asg.set_desired_capacity(AutoScalingGroupName=ASG_NAME, DesiredCapacity=2, HonorCooldown=False)
        action = 'scale_up'
        should_scale = True
    elif final_pred < THRESHOLD_DOWN:
        print("[ML Decision] CPU < threshold → scale down")
        asg.set_desired_capacity(AutoScalingGroupName=ASG_NAME, DesiredCapacity=1, HonorCooldown=False)
        action = 'scale_down'
        should_scale = True
    else:
        print("[ML Decision] CPU stable → no action")
else:
    print("[Cooldown] Skipping scaling due to recent action.")

if should_scale:
    with open(cooldown_file, 'w') as f:
        f.write(now.isoformat())

cloudwatch.put_metric_data(
    Namespace='MLAutoScaling',
    MetricData=[
        {
            'MetricName': 'PredictedCPU',
            'Dimensions': [{'Name': 'Model', 'Value': 'HYBRID'}],
            'Timestamp': datetime.utcnow(),
            'Value': final_pred,
            'Unit': 'Percent'
        },
        {
            'MetricName': 'Action',
            'Dimensions': [{'Name': 'Model', 'Value': 'HYBRID'}],
            'Timestamp': datetime.utcnow(),
            'Value': 1 if action == 'scale_up' else 0,
            'Unit': 'Count'
        }
    ]
)

# === LOG ===
log_entry = pd.DataFrame.from_records([{
    'Timestamp': now.replace(second=0, microsecond=0),
    'Predicted_CPU': final_pred,
    'Action': action,
    'Model': MODEL_NAME
}])
header_needed = not os.path.exists(LOG_FILE)
log_entry.to_csv(LOG_FILE, mode='a', header=header_needed, index=False)
try:
    s3.upload_file(LOG_FILE, BUCKET_NAME, LOG_KEY)
    print(f"Uploaded log to s3://{BUCKET_NAME}/{LOG_KEY}")
except Exception as e:
    print(f"[!] Failed to upload log: {e}")

print("Hybrid ML Predictive Auto-Scaling Controller complete.")