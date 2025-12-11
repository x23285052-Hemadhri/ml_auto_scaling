import boto3, io, pandas as pd, numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
bucket = 'ml-autoscaling-dataset'
ml_key = 'ml_predictive/ml_asg_combined_i-047ec14915c9a2b37.csv' # change to you instance ID
asg_key = 'asg_cpu_reactive/asg_combined_i-0dc7dc255dc8f5953.csv' # change to your instance ID
s3 = boto3.client('s3')

# === Load and parse ===
print(f"Using fixed ML file: {ml_key}")
ml_data = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=ml_key)['Body'].read()))

print(f"Using fixed ASG file: {asg_key}")
asg_data = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=asg_key)['Body'].read()))

# === Parse Time ===
for df in [asg_data, ml_data]:
    df.rename(columns={'Timestamp': 'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], utc=True).dt.tz_localize(None)

# === CPU Usage Inference ===
def infer_cpu_usage(df, name):
    if 'CPU_Usage' not in df.columns:
        if 'CPU_User' in df.columns and 'CPU_System' in df.columns:
            df['CPU_Usage'] = df['CPU_User'] + df['CPU_System']
        elif 'CPU_User' in df.columns:
            df['CPU_Usage'] = df['CPU_User']
        else:
            df['CPU_Usage'] = df.select_dtypes(include='number').iloc[:, -1]
            print(f"[WARN] Fallback CPU column used for {name}")
    return df

asg_data = infer_cpu_usage(asg_data, 'ASG')
ml_data = infer_cpu_usage(ml_data, 'ML')

# === Add missing columns
for col in ['Model', 'Predicted_CPU', 'Action', 'GroupInServiceInstances']:
    for df in [asg_data, ml_data]:
        if col not in df.columns:
            df[col] = None

# === Filter ML data to ASG time range
start_time, end_time = asg_data['Time'].min(), asg_data['Time'].max()
ml_data = ml_data[(ml_data['Time'] >= start_time) & (ml_data['Time'] <= end_time)]
print(f"ML data filtered to ASG range: {start_time} → {end_time}")
print(f"ML rows remaining after filter: {len(ml_data)}")

# === Labeling
asg_data['Source'] = 'Traditional ASG'
ml_data['Model'] = ml_data.get('Model', 'unknown').fillna('unknown')
ml_data['Source'] = ml_data['Model'].str.upper().apply(lambda m: f"ML: {m}")
ml_data = ml_data[ml_data['Model'].str.lower().isin(['lstm', 'prophet'])]
ml_data = ml_data.drop_duplicates(subset=['Time', 'Model', 'Predicted_CPU', 'Action'])

# === Drop rows without valid CPU Usage
asg_data = asg_data.dropna(subset=['Time', 'CPU_Usage'])
ml_data = ml_data.dropna(subset=['Time', 'CPU_Usage'])

# === Combine
combined = pd.concat([
    asg_data[['Time', 'CPU_Usage', 'Source', 'Model', 'Predicted_CPU', 'Action', 'GroupInServiceInstances']],
    ml_data[['Time', 'CPU_Usage', 'Source', 'Model', 'Predicted_CPU', 'Action', 'GroupInServiceInstances']],
], ignore_index=True).sort_values('Time')

# === Metrics ===
def compute_metrics(df, label):
    return {
        'Label': label,
        'Avg_CPU': df['CPU_Usage'].mean(),
        'Std': df['CPU_Usage'].std(),
        'SLA_Violation': (df['CPU_Usage'] > 80).mean() * 100,
        'Cost': (len(df) / 60) * 0.0104
    }

results = pd.DataFrame([
    compute_metrics(group, label)
    for label, group in combined.groupby('Source')
])

# === ML Latency Estimation ===
def estimate_latency(df):
    if 'Action' not in df.columns or 'Predicted_CPU' not in df.columns:
        return np.nan
    df = df.dropna(subset=['Time', 'Predicted_CPU']).sort_values('Time')
    spikes = df[df['Predicted_CPU'] > 70]['Time']
    actions = df[df['Action'] == 'scale_up']['Time']
    latencies = []
    for spike_time in spikes:
        future_actions = actions[actions > spike_time]
        if not future_actions.empty:
            latencies.append((future_actions.iloc[0] - spike_time).total_seconds())
    return np.mean(latencies) if latencies else np.nan

ml_latency = (
    ml_data
    .groupby('Model', group_keys=False)
    .apply(estimate_latency)
    .reset_index()
    .rename(columns={0: 'ScaleUp_Latency(s)'})
)
ml_latency['Label'] = ml_latency['Model'].str.upper().apply(lambda m: f"ML: {m}")
ml_latency = ml_latency[['Label', 'ScaleUp_Latency(s)']]

# === ASG Latency Estimation ===
def estimate_asg_latency_from_instances(df, cpu_col='CPU_Usage', group_col='GroupInServiceInstances', threshold=70):
    df = df.dropna(subset=['Time', cpu_col, group_col]).sort_values('Time')
    df['ScaleUp'] = df[group_col].diff().fillna(0)
    spikes = df[df[cpu_col] > threshold]
    scale_ups = df[df['ScaleUp'] > 0]
    latencies = []
    for spike_time in spikes['Time']:
        match = scale_ups[scale_ups['Time'] > spike_time]
        if not match.empty:
            latency = (match.iloc[0]['Time'] - spike_time).total_seconds()
            if latency >= 0:
                latencies.append(latency)
    return np.mean(latencies) if latencies else np.nan

asg_latency = estimate_asg_latency_from_instances(asg_data)

# === Merge Latencies ===
results = results.merge(ml_latency, on='Label', how='left')
results.loc[results['Label'] == 'Traditional ASG', 'ScaleUp_Latency(s)'] = asg_latency

# === Final Summary ===
print("\n===== Auto-Scaling Performance Summary =====")
print(results.to_string(index=False, float_format="%.2f"))

# === Plotting ===
plt.figure(figsize=(12,6))
for name, group in combined.groupby('Source'):
    plt.plot(group['Time'], group['CPU_Usage'], label=name)

plt.axhline(y=70, color='red', linestyle='--', label='Scale Threshold (70%)')
plt.title("CPU Utilization: Traditional vs ML-Based Scaling (Prophet vs LSTM)")
plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()