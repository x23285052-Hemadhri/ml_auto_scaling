import boto3
import pandas as pd
import io
from datetime import datetime, timedelta

# === CONFIGURATION ===
INSTANCE_ID = 'i-047ec14915c9a2b37' # Replace with your instance ID
INSTANCE_TYPE = 't3.micro'
ML_ASG_NAME = 'ML-ASG'
REGION = 'us-east-2'
BUCKET_NAME = 'ml-autoscaling-dataset'
S3_KEY = f'ml_predictive/ml_asg_combined_{INSTANCE_ID}.csv'
PERIOD = 60
DURATION_MINS = 600

cloudwatch = boto3.client('cloudwatch', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)

# === METRIC DEFINITIONS ===
metrics_to_fetch = {
    'cpu_usage_user': 'CPU_User',
    'cpu_usage_system': 'CPU_System',
    'cpu_usage_idle': 'CPU_Idle',
    'mem_used_percent': 'Memory_Usage',
    'disk_used_percent': 'Disk_Usage',
    'net_bytes_recv': 'Network_In',
    'net_bytes_sent': 'Network_Out'
}

dimension_map = {
    'cpu_usage_user': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'cpu_usage_system': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'cpu_usage_idle': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'cpu', 'Value': 'cpu-total'}],
    'mem_used_percent': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}],
    'disk_used_percent': [
        {'Name': 'path', 'Value': '/'},
        {'Name': 'InstanceId', 'Value': INSTANCE_ID},
        {'Name': 'device', 'Value': 'nvme0n1p1'},
        {'Name': 'fstype', 'Value': 'xfs'}
    ],
    'net_bytes_recv': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'interface', 'Value': 'ens5'}],
    'net_bytes_sent': [{'Name': 'InstanceId', 'Value': INSTANCE_ID}, {'Name': 'interface', 'Value': 'ens5'}]
}

end_time = datetime.utcnow()
start_time = end_time - timedelta(minutes=DURATION_MINS)

# === Fetch Instance-Level Metrics ===
def fetch_instance_metrics(metric_name):
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
            print(f"No data for {metric_name}")
            return pd.DataFrame()
        df = pd.DataFrame([
            {'Timestamp': dp['Timestamp'], metrics_to_fetch[metric_name]: dp['Average']}
            for dp in datapoints
        ])
        print(f"{metric_name} — {len(df)} points")
        return df
    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame()

instance_df = pd.DataFrame()
for metric in metrics_to_fetch.keys():
    df = fetch_instance_metrics(metric)
    if not df.empty:
        instance_df = df if instance_df.empty else pd.merge(instance_df, df, on='Timestamp', how='outer')

if instance_df.empty:
    print("No instance metrics collected.")
    exit()

instance_df['InstanceId'] = INSTANCE_ID
instance_df['InstanceType'] = INSTANCE_TYPE
instance_df['Source'] = 'ML-Predictive'
instance_df['Timestamp'] = pd.to_datetime(instance_df['Timestamp']).dt.round('min')
instance_df = instance_df.sort_values('Timestamp')

# === Fetch ASG Group-Level Metrics ===
def fetch_asg_group_metrics(asg_name):
    asg_metrics = [
        'GroupDesiredCapacity',
        'GroupInServiceInstances',
        'GroupPendingInstances',
        'GroupTerminatingInstances',
        'GroupTotalInstances'
    ]
    group_df = pd.DataFrame()

    for metric in asg_metrics:
        try:
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/AutoScaling',
                MetricName=metric,
                Dimensions=[{'Name': 'AutoScalingGroupName', 'Value': asg_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=PERIOD,
                Statistics=['Average']
            )
            datapoints = response.get('Datapoints', [])
            if not datapoints:
                print(f"[!] No data for {metric}")
                continue
            df = pd.DataFrame([
                {'Timestamp': dp['Timestamp'], metric: dp['Average']}
                for dp in datapoints
            ])
            group_df = df if group_df.empty else pd.merge(group_df, df, on='Timestamp', how='outer')
            print(f"{metric} — {len(df)} points")
        except Exception as e:
            print(f"Error fetching {metric}: {e}")

    if group_df.empty:
        print("No ASG group metrics collected.")
        return None

    group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp']).dt.round('min')
    return group_df.sort_values('Timestamp')

asg_df = fetch_asg_group_metrics(ML_ASG_NAME)

# === Merge Instance & ASG Metrics ===
if asg_df is not None:
    merged_df = pd.merge_asof(
        instance_df.sort_values('Timestamp'),
        asg_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest'
    )
else:
    merged_df = instance_df

# === Optional: Merge Scaling Logs ===
try:
    scaling_key = f'ml_predictive/ml_scaling_logs.csv'
    scaling_obj = s3.get_object(Bucket=BUCKET_NAME, Key=scaling_key)
    scaling_df = pd.read_csv(io.BytesIO(scaling_obj['Body'].read()))
    
    # Normalize both timestamps
    scaling_df['Timestamp'] = pd.to_datetime(scaling_df['Timestamp']).dt.tz_localize(None).dt.round('min')
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp']).dt.tz_localize(None)

    merged_df = pd.merge_asof(
        merged_df.sort_values('Timestamp'),
        scaling_df[['Timestamp', 'Predicted_CPU', 'Action', 'Model']].sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=1)
    )
    print("Scaling log merged.")
except Exception as e:
    print(f"Scaling log not merged: {e}")
# === Final Output ===
if merged_df is None or merged_df.empty:
    print("No final merged data. Exiting.")
    exit()

merged_df = merged_df.sort_values('Timestamp')

# === Save and Upload ===
timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
local_copy = f'ml_asg_combined_{INSTANCE_ID}_{timestamp_str}.csv'
csv_file = f"/tmp/ml_asg_combined_{INSTANCE_ID}.csv"

merged_df.to_csv(local_copy, index=False)
merged_df.to_csv(csv_file, index=False)

s3.upload_file(csv_file, BUCKET_NAME, S3_KEY)

print(f"Uploaded to S3: s3://{BUCKET_NAME}/{S3_KEY}")
print(f"Local copy saved as: {local_copy}")
print("ML predictive scaling metrics export complete.")
