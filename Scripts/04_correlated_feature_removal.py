import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ============================
# Configuration
# ============================
base_folder = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")

prefix = "git_combine_"

# Input & Output Files
input_file = os.path.join(output_folder, f"{prefix}subR_missing_out.csv")
output_file = os.path.join(output_folder, f"{prefix}subR_missing_out_corr.csv")

# =============================================
# Correlated Feature Removal Script
# ==============================================
# Load Data
df = pd.read_csv(input_file)

# Extract Subject_ID and Label
subject_id_column = df["Subject_ID"]
label_column = df["Label"]

# Drop Subject_ID and Label for correlation calculation
df_features = df.drop(columns=["Subject_ID", "Label"])

# Compute correlation matrix
corr_matrix = df_features.corr()

# Set correlation threshold
threshold = 0.8

# Compute correlation with Label
label_corr = df_features.corrwith(df["Label"]).abs()

# Identify highly correlated features
highly_correlated = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            drop_feature = feature_i if label_corr[feature_i] < label_corr[feature_j] else feature_j
            highly_correlated.add(drop_feature)

# Drop highly correlated features
df_reduced_corr = df_features.drop(columns=highly_correlated)

# Add back Subject_ID and Label
df_reduced_corr.insert(0, "Subject_ID", df["Subject_ID"])
df_reduced_corr["Label"] = label_column

# Save the resulting dataset
df_reduced_corr.to_csv(output_file, index=False)

# Visualization: Feature Correlation After Filtering
plt.figure(figsize=(12, 8))
sns.heatmap(df_reduced_corr.drop(columns=["Subject_ID"]).corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix After Feature Filtering")
plt.tight_layout()
plt.show()

print(f"Feature selection completed and saved to {output_file}")
print(f"Dropped features due to high correlation: {highly_correlated}")
