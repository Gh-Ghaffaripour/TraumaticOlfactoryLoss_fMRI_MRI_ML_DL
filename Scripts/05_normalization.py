import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Configuration
# ============================
base_folder   = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")
prefix        = "git_combine_"

input_file  = os.path.join(output_folder, f"{prefix}subR_missing_out_corr.csv")
output_file = os.path.join(output_folder, f"{prefix}subR_missing_out_corr_norm.csv")

# =============================================
# Normalization Script
# =============================================
df = pd.read_csv(input_file)

# Identify numerical columns (exclude Label & Subject_ID)
numerical_columns = [
    col for col in df.select_dtypes(include=['number']).columns 
    if col not in ("Label", "Subject_ID")
]

# Standardize features using Z‑score
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Verify Standardization
print("Mean after scaling (~0):\n", df[numerical_columns].mean().round(3))
print("Std dev after scaling (~1):\n", df[numerical_columns].std().round(3))

# Save normalized data
df.to_csv(output_file, index=False)

# Visualization: Seaborn histplots with KDE for first 6 features
sample_features = numerical_columns[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, feature in zip(axes, sample_features):
    sns.histplot(df[feature], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()

# Presence checks
print("Label present in DataFrame:", "Label" in df.columns)
print("Subject_ID present in DataFrame:", "Subject_ID" in df.columns)

print(f"✅ Standardization completed. Normalized dataset saved to {output_file}")
