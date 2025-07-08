import pandas as pd
import matplotlib.pyplot as plt

# --- SETTINGS ---
file_a = r"E:\STARS\fullframewithsquareNS.csv"
file_b = r"E:\STARS\fullframeNS.csv"
threshold = 0.05

# --- LOAD CSVs ---
df_a = pd.read_csv(file_a)
df_b = pd.read_csv(file_b)

# Ensure sorted and same shape
df_a = df_a.sort_values('Frame').reset_index(drop=True)
df_b = df_b.sort_values('Frame').reset_index(drop=True)

# Columns to compare
keys = ['RightHeel_Visibility',
    'LeftHeel_Visibility',
    'RightHeel_Presence',
    'LeftHeel_Presence']

# Create difference DataFrame (B - A)
df_diff = df_b[keys] - df_a[keys]
df_diff['Frame'] = df_a['Frame']  # Keep frame index for plotting

# --- THRESHOLD ANALYSIS ---
print(f"\nThreshold for meaningful difference: ±{threshold:.2f}")

for key in keys:
    diffs = df_diff[key]
    count_b_better = (diffs > threshold).sum()
    count_a_better = (diffs < -threshold).sum()
    count_neutral = ((diffs.abs() <= threshold)).sum()

    print(f"\n{key}:")
    print(f"  Crop B better (> +{threshold}): {count_b_better} frames")
    print(f"  Crop A better (< -{threshold}): {count_a_better} frames")
    print(f"  No significant difference (≤ ±{threshold}): {count_neutral} frames")

# --- PLOT ---
plt.figure(figsize=(14, 10))
for i, key in enumerate(keys):
    plt.subplot(2, 2, i + 1)
    plt.plot(df_diff['Frame'], df_diff[key], label=f'{key} (B - A)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axhline(y=threshold, color='green', linestyle=':', label=f'+{threshold} threshold')
    plt.axhline(y=-threshold, color='red', linestyle=':', label=f'-{threshold} threshold')
    plt.title(f'Difference in {key} (B - A)')
    plt.xlabel('Frame')
    plt.ylabel('Difference')
    plt.legend()

plt.tight_layout()
plt.show()