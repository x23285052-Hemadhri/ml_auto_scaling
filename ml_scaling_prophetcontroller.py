import boto3
import pandas as pd
import pickle
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import io
import sys
import os

# === CONFIGURATION ===
REGION = 'us-east-2' # Replace with your AWS region
BUCKET_NAME = 'ml-autoscaling-dataset'
ASG_NAME = 'ML-ASG'
MODEL_KEY = 'ml_predictive/prophet_cpu_model.pkl'
LOG_FILE = 'ml_scaling_logs.csv'
LOG_KEY = 'ml_predictive/ml_scaling_logs.csv'
THRESHOLD_UP = 70
THRESHOLD_DOWN = 70
PERIOD = 60
DURATION_MINS = 30
FORECAST_MINUTES = 10
COOLDOWN_SECONDS = 300
MODEL_NAME = 'prophet'

INSTANCE_ID = 'i-047ec14915c9a2b37' # Replace with your instance ID
INSTANCE_TYPE = 't3.micro'

print(f"[INFO] Starting ML Predictive Auto-Scaling Controller ({MODEL_NAME.upper()}) for instance {INSTANCE_ID}")

# === AWS Clients ===
s3 = boto3.client('s3', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
asg = boto3.client('autoscaling', region_name=REGION)

# === Load Prophet Model from S3 ===
try:
    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    model = pickle.load(io.BytesIO(model_obj['Body'].read()))
    print("Prophet model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# === Fetch Metrics from CloudWatch ===
metrics_to_fetch = {
    'cpu_usage_user': 'CPU_User',
    'cpu_usage_system': 'CPU_System'
}
dimension_map = {
    'cpu_usage_user': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'cpu_usage_system': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}]
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

final_df = pd.DataFrame()
for metric in metrics_to_fetch:
    df = fetch_metric(metric)
    if not df.empty:
        final_df = df if final_df.empty else pd.merge(final_df, df, on='Timestamp', how='outer')

if final_df.empty:
    print(" No metrics collected.")
    sys.exit(0)

final_df = final_df.sort_values('Timestamp')
print(f"Collected {len(final_df)} metric points.")

# === Forecasting with Prophet ===
df = final_df[['Timestamp', 'CPU_User']].rename(columns={'Timestamp': 'ds', 'CPU_User': 'y'})
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

future = pd.concat([
    df[['ds']],
    pd.DataFrame({'ds': [df['ds'].max() + timedelta(minutes=i) for i in range(1, FORECAST_MINUTES + 1)]})
])

forecast = model.predict(future)
forecast['yhat'] = forecast['yhat'].clip(lower=0, upper=100)
predicted_cpu = forecast['yhat'].iloc[-1]
print(f"Predicted CPU Utilization: {predicted_cpu:.2f}%")

# === Cooldown Logic ===
cooldown_file = 'last_scale_prophet.txt'
last_scale_time = None
now = datetime.utcnow()

if os.path.exists(cooldown_file):
    with open(cooldown_file, 'r') as f:
        last_scale_time = datetime.fromisoformat(f.read().strip())

should_scale = False
action = 'no_action'

if last_scale_time is None or (now - last_scale_time).total_seconds() > COOLDOWN_SECONDS:
    if predicted_cpu > THRESHOLD_UP:
        print("[ML Decision] CPU > threshold → scale up")
        asg.set_desired_capacity(AutoScalingGroupName=ASG_NAME, DesiredCapacity=2, HonorCooldown=False)
        action = 'scale_up'
        should_scale = True
    elif predicted_cpu < THRESHOLD_DOWN:
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
            'Dimensions': [{'Name': 'Model', 'Value': 'PROPHET'}],  
            'Timestamp': datetime.utcnow(),
            'Value': predicted_cpu,
            'Unit': 'Percent'
        },
        {
            'MetricName': 'Action',
            'Dimensions': [{'Name': 'Model', 'Value': 'PROPHET'}],
            'Timestamp': datetime.utcnow(),
            'Value': 1 if action == 'scale_up' else 0,
            'Unit': 'Count'
        }
    ]
)

# === Logging Forecast to CSV and S3 ===
forecast_log = forecast[['ds', 'yhat']].copy()
forecast_log.rename(columns={'ds': 'Timestamp', 'yhat': 'Predicted_CPU'}, inplace=True)
forecast_log['Timestamp'] = pd.to_datetime(forecast_log['Timestamp']).dt.round('min')
forecast_log['Action'] = forecast_log['Predicted_CPU'].apply(
    lambda x: 'scale_up' if x > THRESHOLD_UP else 'scale_down' if x < THRESHOLD_DOWN else 'no_action'
)
forecast_log['Model'] = MODEL_NAME
forecast_log = forecast_log.tail(FORECAST_MINUTES)

header_needed = not os.path.exists(LOG_FILE)
forecast_log.to_csv(LOG_FILE, mode='a', header=header_needed, index=False)

try:
    s3.upload_file(LOG_FILE, BUCKET_NAME, LOG_KEY)
    print(f"Uploaded log to s3://{BUCKET_NAME}/{LOG_KEY}")
except Exception as e:
    print(f"Failed to upload log: {e}")

print("ML Predictive Auto-Scaling Controller complete.")
