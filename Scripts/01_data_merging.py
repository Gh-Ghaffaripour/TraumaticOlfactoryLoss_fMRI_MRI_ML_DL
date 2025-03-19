import os
import pandas as pd
import re

# =====================================
# Configuration 
# =====================================
base_folder = r"D:\image_group_data\team44"
output_folder = os.path.join(base_folder, "Analysis")

structural_file = os.path.join(output_folder, "structural_subR.csv")
graph_file = os.path.join(output_folder, "graph_subR.csv")
combined_file = os.path.join(output_folder, "git_combine_subR.csv")

os.makedirs(output_folder, exist_ok=True)

# =====================================
# Functions for ID normalization
# =====================================

def normalize_struct_id(sub_id: str) -> str:
    match = re.search(r'(\d+)$', sub_id)
    if match:
        digits = match.group(1).lstrip('0')
        return digits if digits else '0'
    return sub_id

def normalize_graph_id(sub_id: str) -> str:
    return sub_id.lstrip('0') or sub_id

# =====================================
# Main merging logic
# =====================================

def load_and_merge_data(structural_path, graph_path):
    df_struct = pd.read_csv(structural_path, dtype={"Subject_ID": str})
    df_graph = pd.read_csv(graph_path, dtype={"Subject_ID": str})

    df_struct["Normalized_ID"] = df_struct["Subject_ID"].apply(normalize_struct_id)
    df_graph["Normalized_ID"] = df_graph["Subject_ID"].apply(normalize_graph_id)

    df_combined = pd.merge(
        df_struct,
        df_graph,
        on=["Normalized_ID", "Label"],
        how="inner",
        suffixes=("_struct", "_graph")
    )

    print(f"Number of rows after merge: {len(df_combined)}")
    merged_preview = df_combined[
        ["Subject_ID_struct", "Subject_ID_graph", "Normalized_ID", "Label"]
    ].head(10)
    print("Preview of merged IDs:\n", merged_preview)

    # Clean up dataframe
    df_combined = df_combined.drop(columns=["Subject_ID_struct", "Subject_ID_graph"])
    df_combined = df_combined.rename(columns={"Normalized_ID": "Subject_ID"})
    df_combined["Subject_ID"] = df_combined["Subject_ID"].astype(str)

    return df_combined

# =====================================
# Execute and save
# =====================================

if __name__ == "__main__":
    merged_df = load_and_merge_data(structural_file, graph_file)
    merged_df.to_csv(combined_file, index=False)
    print(f"Merged data saved at: {combined_file}")
