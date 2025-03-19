import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# ============================
# Configuration
# ============================
base_folder = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")

prefix = "git_combine_"
# Input & Output Files
input_file = os.path.join(output_folder, f"{prefix}subR_missing_out_corr.csv")
output_file = os.path.join(output_folder, f"{prefix}subR_missing_out_corr_norm.csv")
# =============================================
# Normalization Script
# ==============================================
# Load Data
df = pd.read_csv(input_file)

# Identify numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ["Label", "Subject_ID"]]

# Standardize features using Z-score
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Verify Standardization
print("Mean values after scaling (should be ~0):\n", df[numerical_columns].mean().round(3))
print("Std deviation after scaling (should be ~1):")
print(df[numerical_columns].std().round(3))

# Save standardized data
df.to_csv(output_file, index=False)

# Visualization: Distributions after scaling
sample_features = numerical_columns[:6]  # Modify number if needed
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(sample_features):
    df[feature].hist(bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()

print(f"Standardization completed. Normalized dataset saved to {output_file}")
