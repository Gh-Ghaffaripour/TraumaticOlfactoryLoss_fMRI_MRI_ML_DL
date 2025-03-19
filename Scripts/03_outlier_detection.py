import pandas as pd
import numpy as np

# ============================
# Configuration
# ============================
base_folder = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")

prefix = "git_combine_"

# Input & Output Files
input_file = os.path.join(output_folder, f"{prefix}subR_missing.csv")
output_file = os.path.join(output_folder, f"{prefix}subR_missing_out.csv")


# =============================================
# Outlier Detection Script
# ==============================================
# Load the dataset
df = pd.read_csv(input_file, dtype={'Subject_ID': str})

# Separate Subject_ID and Label
df_numeric = df.drop(columns=['Subject_ID', 'Label'])

# Detect outliers in each feature using the IQR method
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
outliers_condition = (df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))
outlier_counts = outliers_condition.sum(axis=1)

# Identify subjects as outliers if they have outliers in >70% of features
threshold = 0.7 * df_numeric.shape[1]
outlier_subjects = outlier_counts[outlier_counts >= threshold].index

# Remove outlier subjects if they are less than 10% of total subjects
if len(outlier_subjects) < 0.1 * len(df):
    df_cleaned = df.drop(index=outlier_subjects)
else:
    df_cleaned = df.copy()

# Re‑order columns so ID & Label come first
ordered_cols = ['Subject_ID', 'Label'] + [c for c in df_cleaned.columns if c not in ['Subject_ID', 'Label']]
df_cleaned = df_cleaned[ordered_cols]

# Save the cleaned dataset
df_cleaned.to_csv(output_file, index=False)

print(f"Outlier removal completed. Cleaned dataset saved to {output_file}")
