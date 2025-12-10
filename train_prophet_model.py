import boto3
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import pickle
import io
import sys

# ===== CONFIG =====
REGION = 'us-east-2' # Replace with your AWS region
BUCKET_NAME = 'ml-autoscaling-dataset'
ASG_NAME = 'ML-ASG'
BASELINE_INSTANCE_ID = ' i-047ec14915c9a2b37'  # Replace with your instance ID
MODEL_KEY = 'ml_predictive/prophet_cpu_model.pkl'
PERIOD = 60                  
DURATION_MINS = 180          

# ===== AWS CLIENTS =====
s3 = boto3.client('s3', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
asg = boto3.client('autoscaling', region_name=REGION)
ec2 = boto3.client('ec2', region_name=REGION)

# ===== Determine Instance to Use =====
try:
    response = ec2.describe_instances(
        InstanceIds=[BASELINE_INSTANCE_ID],
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )
    if response['Reservations']:
        INSTANCE_ID = BASELINE_INSTANCE_ID
        print(f"[INFO] Using baseline instance for training: {INSTANCE_ID}")
    else:
        asg_info = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
        INSTANCE_ID = asg_info['AutoScalingGroups'][0]['Instances'][0]['InstanceId']
        print(f"[WARN] Baseline instance not found. Using active ASG instance: {INSTANCE_ID}")
except Exception as e:
    print(f"[ERROR] Could not determine instance ID: {e}")
    sys.exit(1)

# ===== METRICS CONFIGURATION =====
metrics_to_fetch = {
    'cpu_usage_user': 'CPU_User',
    'cpu_usage_system': 'CPU_System',
    'cpu_usage_idle': 'CPU_Idle',
    'mem_used_percent': 'Memory_Usage'
}

dimension_map = {
    'cpu_usage_user': [
        {'Name': 'InstanceId', 'Value': INSTANCE_ID},
        {'Name': 'cpu', 'Value': 'cpu-total'}
    ],
    'cpu_usage_system': [
        {'Name': 'InstanceId', 'Value': INSTANCE_ID},
        {'Name': 'cpu', 'Value': 'cpu-total'}
    ],
    'cpu_usage_idle': [
        {'Name': 'InstanceId', 'Value': INSTANCE_ID},
        {'Name': 'cpu', 'Value': 'cpu-total'}
    ],
    'mem_used_percent': [
        {'Name': 'InstanceId', 'Value': INSTANCE_ID}
    ]
}

# ===== TIME RANGE =====
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(minutes=DURATION_MINS)

print(f"[INFO] Collecting CWAgent metrics from {start_time} to {end_time}")

# ===== METRIC FETCH FUNCTION =====
def fetch_metric(metric_name):
    try:
        dims = dimension_map[metric_name]
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
            print(f"[!] Skipping {metric_name} — no data returned.")
            return pd.DataFrame()

        df = pd.DataFrame([
            {'Timestamp': dp['Timestamp'], metrics_to_fetch[metric_name]: dp['Average']}
            for dp in datapoints
        ])
        return df
    except Exception as e:
        print(f"[!] Error fetching {metric_name}: {e}")
        return pd.DataFrame()

# ===== COLLECT MULTIPLE METRICS =====
final_df = pd.DataFrame()
for metric in metrics_to_fetch.keys():
    df = fetch_metric(metric)
    if not df.empty:
        final_df = df if final_df.empty else pd.merge(final_df, df, on='Timestamp', how='outer')

if final_df.empty:
    print("[ERROR] No metrics collected. Training aborted.")
    sys.exit(0)

final_df = final_df.sort_values('Timestamp')
print(f"[INFO] Collected {len(final_df)} data points for Prophet training.")

# ===== TRAIN PROPHET MODEL =====
try:
    train_df = final_df[['Timestamp', 'CPU_User']].rename(columns={'Timestamp': 'ds', 'CPU_User': 'y'})
    train_df['ds'] = pd.to_datetime(train_df['ds']).dt.tz_localize(None)
    train_df['y'] = train_df['y'].clip(lower=0, upper=100).rolling(window=3, min_periods=1).mean()

    print(f"[INFO] Training Prophet model with {len(train_df)} points...")
    model = Prophet(interval_width=0.95, daily_seasonality=False)
    model.fit(train_df)

    with open('prophet_cpu_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    s3.upload_file('prophet_cpu_model.pkl', BUCKET_NAME, MODEL_KEY)
    print(f"[SUCCESS] Uploaded new model to s3://{BUCKET_NAME}/{MODEL_KEY}")

except Exception as e:
    print(f"[ERROR] Training or upload failed: {e}")
    sys.exit(1)

print("[INFO] Prophet retraining complete.")
