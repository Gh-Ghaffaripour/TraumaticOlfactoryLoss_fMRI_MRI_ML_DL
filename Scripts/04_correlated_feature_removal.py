import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
base_folder   = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")
prefix        = "git_combine_"

input_file  = os.path.join(output_folder, f"{prefix}subR_missing_out.csv")
output_file = os.path.join(output_folder, f"{prefix}subR_missing_out_corr.csv")

# =============================================
# Correlated Feature Removal
# =============================================
df = pd.read_csv(input_file)

# Separate metadata
subject_id = df["Subject_ID"]
label      = df["Label"]
features   = df.drop(columns=["Subject_ID", "Label"])

# Compute original correlation matrix
corr_orig = features.corr()

# Plot BEFORE filtering
plt.figure(figsize=(10, 8))
sns.heatmap(corr_orig, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Original Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Identify highly correlated features
threshold = 0.8
label_corr = features.corrwith(label).abs()
highly_correlated = set()

for i in range(len(corr_orig.columns)):
    for j in range(i):
        if abs(corr_orig.iloc[i, j]) > threshold:
            f_i = corr_orig.columns[i]
            f_j = corr_orig.columns[j]
            drop = f_i if label_corr[f_i] < label_corr[f_j] else f_j
            highly_correlated.add(drop)

# Drop and rebuild DataFrame
df_reduced = features.drop(columns=highly_correlated)
df_reduced.insert(0, "Subject_ID", subject_id)
df_reduced["Label"] = label

# Compute post‑filter correlation
corr_filtered = df_reduced.drop(columns=["Subject_ID", "Label"]).corr()

# Plot AFTER filtering
plt.figure(figsize=(10, 8))
sns.heatmap(corr_filtered, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Filtered Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Save and print summary
df_reduced.to_csv(output_file, index=False)
print(f"Feature selection complete — saved to {output_file}")
print("Dropped features due to high correlation:", highly_correlated)
print("Final DataFrame shape:", df_reduced.shape)
print("Subject_ID present:", "Subject_ID" in df_reduced.columns)
print("Label present:", "Label" in df_reduced.columns)
