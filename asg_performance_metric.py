import boto3
import pandas as pd
from datetime import datetime, timedelta

# === CONFIGURATION ===
REGION = 'us-east-2'  # Change to your region
ASG_NAME = 'Traditional-ASG'
BUCKET_NAME = 'ml-autoscaling-dataset'
S3_PREFIX = 'asg_cpu_reactive/'
INSTANCE_ID = 'i-0dc7dc255dc8f5953'
S3_KEY = f'asg_cpu_reactive/asg_combined_{INSTANCE_ID}.csv'
PERIOD = 60
DURATION_MINS = 600  # Last 10 hours of data

# === Initialize AWS Clients ===
asg = boto3.client('autoscaling', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
s3 = boto3.client('s3')


# === Metrics to Collect (Instance-Level) ===
metrics_to_fetch = {
    'cpu_usage_user': 'CPU_User',
    'cpu_usage_system': 'CPU_System',
    'cpu_usage_idle': 'CPU_Idle',
    'mem_used_percent': 'Memory_Usage',
    'disk_used_percent': 'Disk_Usage',
    'net_bytes_recv': 'Network_In',
    'net_bytes_sent': 'Network_Out'
}
# === Helper Function to Fetch Instance Metrics ===
def fetch_metrics_for_instance(instance_id):
    dimension_map = {
        'cpu_usage_user': [
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'cpu', 'Value': 'cpu-total'}
        ],
        'cpu_usage_system': [
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'cpu', 'Value': 'cpu-total'}
        ],
        'cpu_usage_idle': [
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'cpu', 'Value': 'cpu-total'}
        ],
        'mem_used_percent': [
            {'Name': 'InstanceId', 'Value': instance_id}
        ],
        'disk_used_percent': [
            {'Name': 'path', 'Value': '/'},
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'device', 'Value': 'nvme0n1p1'},
            {'Name': 'fstype', 'Value': 'xfs'}
        ],
        'net_bytes_recv': [
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'interface', 'Value': 'ens5'}
        ],
        'net_bytes_sent': [
            {'Name': 'InstanceId', 'Value': instance_id},
            {'Name': 'interface', 'Value': 'ens5'}
        ]
    }

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=DURATION_MINS)
    final_df = pd.DataFrame()

    for metric_name in metrics_to_fetch.keys():
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
                print(f"Skipping {metric_name} — no data for {instance_id}")
                continue

            df = pd.DataFrame([
                {'Timestamp': dp['Timestamp'], metrics_to_fetch[metric_name]: dp['Average']}
                for dp in datapoints
            ])

            if final_df.empty:
                final_df = df
            else:
                final_df = pd.merge(final_df, df, on='Timestamp', how='outer')

            print(f"Collected {len(df)} points for {metric_name} on {instance_id}")

        except Exception as e:
            print(f"Error fetching {metric_name} for {instance_id}: {e}")

    if final_df.empty:
        print(f"No data collected for instance {instance_id}")
        return None

    final_df['InstanceId'] = instance_id
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
    final_df = final_df.sort_values('Timestamp')
    return final_df

# === Fetch Instance-Level Metrics for a Single Instance ===
combined_df = pd.DataFrame()

df = fetch_metrics_for_instance(instance_id=INSTANCE_ID)
if df is not None:
    combined_df = pd.concat([combined_df, df], ignore_index=True)


# === Fetch Group-Level ASG Metrics ===
def fetch_asg_group_metrics(asg_name):
    asg_metrics = [
        'GroupDesiredCapacity',
        'GroupInServiceInstances',
        'GroupPendingInstances',
        'GroupTerminatingInstances',
        'GroupTotalInstances'
    ]

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=DURATION_MINS)
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
                print(f"No data for {metric}")
                continue

            df = pd.DataFrame([
                {'Timestamp': dp['Timestamp'], metric: dp['Average']}
                for dp in datapoints
            ])

            if group_df.empty:
                group_df = df
            else:
                group_df = pd.merge(group_df, df, on='Timestamp', how='outer')

            print(f"Collected {len(df)} points for {metric}")

        except Exception as e:
            print(f"Error fetching ASG metric {metric}: {e}")

    if group_df.empty:
        print("No ASG group metrics collected.")
        return None

    group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'])
    group_df = group_df.sort_values('Timestamp')
    return group_df


# === Merge Group and Instance Metrics ===
asg_group_df = fetch_asg_group_metrics(ASG_NAME)
if asg_group_df is not None and not combined_df.empty:
    merged_df = pd.merge_asof(
        combined_df.sort_values('Timestamp'),
        asg_group_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest'
    )
else:
    merged_df = combined_df if not combined_df.empty else asg_group_df

if merged_df is None or merged_df.empty:
    print("No combined data available. Exiting.")
    exit()

# === Save and Upload to S3 ===
csv_file = f"/tmp/{ASG_NAME}_combined_metrics.csv"
merged_df.to_csv(csv_file, index=False)

s3.upload_file(csv_file, BUCKET_NAME, S3_KEY)



print(f"Uploaded combined instance + group metrics to s3://{BUCKET_NAME}/{S3_KEY}")
print("[INFO] ASG performance data collection complete.")

