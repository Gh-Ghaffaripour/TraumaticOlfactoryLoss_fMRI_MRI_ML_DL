# =========================
# Imports
# =========================
from pathlib import Path
from typing import Dict, Tuple, List, Set
import pandas as pd
import numpy as np
import json, hashlib, datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
from scipy.stats import levene, ttest_ind


# =========================
# Configuration: paths and directories
# =========================
BASE_DIR = Path(r"C:\Users\Growth fire\Desktop\group44manuscript\pipline_v1")

# Raw inputs
GRAPH_FILE   = BASE_DIR / "graph_clean_subID.csv"
STRUCT_FILE  = BASE_DIR / "structural_clean_subID.csv"

# Combined dataset (produced in Step 1)
COMBINED_FILE = BASE_DIR / "combined_clean_subID.csv"

# Outputs from the feature-cleaning step
OUT_STRUCT_CLEAN = BASE_DIR / "git_structural_cleaned.csv"
OUT_GRAPH_CLEAN  = BASE_DIR / "git_graph_cleaned.csv"
OUT_COMB_CLEAN   = BASE_DIR / "git_combine_cleaned.csv"

# One-time persisted split (locked by manifest)
STRUCT_TRAIN = BASE_DIR / "git_structural_train.csv"
STRUCT_TEST  = BASE_DIR / "git_structural_test.csv"
GRAPH_TRAIN  = BASE_DIR / "git_graph_train.csv"
GRAPH_TEST   = BASE_DIR / "git_graph_test.csv"
COMB_TRAIN   = BASE_DIR / "git_combine_train.csv"
COMB_TEST    = BASE_DIR / "git_combine_test.csv"

# Outlier removal outputs (bounds learned on TRAIN, decision applied to TRAIN and TEST)
STRUCT_TRAIN_OLRM = BASE_DIR / "git_structural_train_outliers_removed.csv"
STRUCT_TEST_OLRM  = BASE_DIR / "git_structural_test_outliers_removed.csv"
GRAPH_TRAIN_OLRM  = BASE_DIR / "git_graph_train_outliers_removed.csv"
GRAPH_TEST_OLRM   = BASE_DIR / "git_graph_test_outliers_removed.csv"
COMB_TRAIN_OLRM   = BASE_DIR / "git_combine_train_outliers_removed.csv"
COMB_TEST_OLRM    = BASE_DIR / "git_combine_test_outliers_removed.csv"

# Correlation-pruned outputs (feature set learned on TRAIN, mirrored to TEST)
STRUCT_TRAIN_CORR = BASE_DIR / "git_structural_train_corr_pruned.csv"
STRUCT_TEST_CORR  = BASE_DIR / "git_structural_test_corr_pruned.csv"
GRAPH_TRAIN_CORR  = BASE_DIR / "git_graph_train_corr_pruned.csv"
GRAPH_TEST_CORR   = BASE_DIR / "git_graph_test_corr_pruned.csv"
COMB_TRAIN_CORR   = BASE_DIR / "git_combine_train_corr_pruned.csv"
COMB_TEST_CORR    = BASE_DIR / "git_combine_test_corr_pruned.csv"

# RobustScaler outputs (scaler fit on TRAIN after correlation pruning; applied to TEST)
STRUCT_TRAIN_SCALED = BASE_DIR / "git_structural_train_robust_scaled.csv"
STRUCT_TEST_SCALED  = BASE_DIR / "git_structural_test_robust_scaled.csv"
GRAPH_TRAIN_SCALED = BASE_DIR / "git_graph_train_robust_scaled.csv"
GRAPH_TEST_SCALED  = BASE_DIR / "git_graph_test_robust_scaled.csv"
COMB_TRAIN_SCALED  = BASE_DIR / "git_combine_train_robust_scaled.csv"
COMB_TEST_SCALED   = BASE_DIR / "git_combine_test_robust_scaled.csv"

# Feature selection outputs — Welch t-test pass (BH-FDR controlled)
STRUCT_TRAIN_TPASS = BASE_DIR / "git_structural_train_ttest_pass.csv"
STRUCT_TEST_TPASS  = BASE_DIR / "git_structural_test_ttest_pass.csv"
GRAPH_TRAIN_TPASS  = BASE_DIR / "git_graph_train_ttest_pass.csv"
GRAPH_TEST_TPASS   = BASE_DIR / "git_graph_test_ttest_pass.csv"
COMB_TRAIN_TPASS   = BASE_DIR / "git_combine_train_ttest_pass.csv"
COMB_TEST_TPASS    = BASE_DIR / "git_combine_test_ttest_pass.csv"

# Feature selection outputs — Random Forest top-20 (by impurity importance)
STRUCT_TRAIN_RF20 = BASE_DIR / "git_structural_train_rf_top20.csv"
STRUCT_TEST_RF20  = BASE_DIR / "git_structural_test_rf_top20.csv"
GRAPH_TRAIN_RF20  = BASE_DIR / "git_graph_train_rf_top20.csv"
GRAPH_TEST_RF20   = BASE_DIR / "git_graph_test_rf_top20.csv"
COMB_TRAIN_RF20   = BASE_DIR / "git_combine_train_rf_top20.csv"
COMB_TEST_RF20    = BASE_DIR / "git_combine_test_rf_top20.csv"

# Feature selection outputs — Final RFE top-10
STRUCT_TRAIN_RFE10 = BASE_DIR / "git_structural_train_rfe_top10.csv"
STRUCT_TEST_RFE10  = BASE_DIR / "git_structural_test_rfe_top10.csv"
GRAPH_TRAIN_RFE10  = BASE_DIR / "git_graph_train_rfe_top10.csv"
GRAPH_TEST_RFE10   = BASE_DIR / "git_graph_test_rfe_top10.csv"
COMB_TRAIN_RFE10   = BASE_DIR / "git_combine_train_rfe_top10.csv"
COMB_TEST_RFE10    = BASE_DIR / "git_combine_test_rfe_top10.csv"

# Post–feature-selection (curated) outputs — RFE10 after interactive/automatic removals
STRUCT_TRAIN_RFE10_CUR = BASE_DIR / "git_structural_train_rfe_top10_curated.csv"
STRUCT_TEST_RFE10_CUR  = BASE_DIR / "git_structural_test_rfe_top10_curated.csv"
GRAPH_TRAIN_RFE10_CUR  = BASE_DIR / "git_graph_train_rfe_top10_curated.csv"
GRAPH_TEST_RFE10_CUR   = BASE_DIR / "git_graph_test_rfe_top10_curated.csv"
COMB_TRAIN_RFE10_CUR   = BASE_DIR / "git_combine_train_rfe_top10_curated.csv"
COMB_TEST_RFE10_CUR    = BASE_DIR / "git_combine_test_rfe_top10_curated.csv"

# Persisted models/metadata
STRUCT_SCALER_FILE = BASE_DIR / "git_structural_robust_scaler.joblib"
GRAPH_SCALER_FILE  = BASE_DIR / "git_graph_robust_scaler.joblib"
COMB_SCALER_FILE   = BASE_DIR / "git_combine_robust_scaler.joblib"
STRUCT_RF_FILE     = BASE_DIR / "git_structural_rf_importance.joblib"
GRAPH_RF_FILE      = BASE_DIR / "git_graph_rf_importance.joblib"
COMB_RF_FILE       = BASE_DIR / "git_combine_rf_importance.joblib"
STRUCT_RFE_FILE    = BASE_DIR / "git_structural_rfe_selector.joblib"
GRAPH_RFE_FILE     = BASE_DIR / "git_graph_rfe_selector.joblib"
COMB_RFE_FILE      = BASE_DIR / "git_combine_rfe_selector.joblib"

# Manifests and reporting artifacts
PIPELINE_MANIFEST = BASE_DIR / "pipeline_manifest.json"          # locks the split
SCALING_MANIFEST  = BASE_DIR / "pipeline_scaling_manifest.json"  # records scaling inputs
CORR_MANIFEST     = BASE_DIR / "pipeline_corr_manifest.json"     # correlation kept/dropped
FS_REPORT_JSON    = BASE_DIR / "pipeline_feature_selection_report.json"

BEST_MODELS_XLSX = BASE_DIR / "git_best_models.xlsx"

# Plot outputs (e.g., correlation heatmaps)
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)  # Ensure plot directory exists (used for corr heatmaps and ROC curves)


# =========================
# Global settings
# =========================
REQUIRED_COLUMNS = {"Subject_ID", "Label"}
PROTECTED_COLUMNS: List[str] = ["Subject_ID", "Label"]  # carried through; not transformed as features
OUTPUT_INDEX = False
VERBOSE = True

# Split settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cleaning policy (strictly remove duplicate, constant, infinite, or missing-value feature columns)
# (no extra knobs; strict removal of dup/constant/inf/NAs in feature columns)

# Outlier policy
ROW_OUTLIER_FRACTION = 0.70   # row flagged if >=70% numeric features beyond Tukey bounds (TRAIN-learned)
MAX_DROP_FRACTION_TRAIN = 0.10  # only drop if <10% of TRAIN rows would be removed

# Diagnostics directory (ensures path exists)
DIAG_DIR = BASE_DIR / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)

# Correlation pruning policy
CORR_THRESHOLD = 0.8         # absolute Pearson > threshold -> consider pair for drop
SAVE_HEATMAPS = True

# RobustScaler parameters
ROBUST_WITH_CENTERING = True
ROBUST_WITH_SCALING = True
ROBUST_QUANTILE_RANGE = (25.0, 75.0)
ROBUST_UNIT_VARIANCE = False

# Feature selection settings
# Multiple-testing control (Welch t-test + BH-FDR)
FDR_ENABLE = True          # set False to use raw p-values with ALPHA
# Univariate Welch t-test screening policy
MIN_TTEST_PASS = 5                         # target minimum #features to pass (do NOT force if signal is too low)
FDR_Q= None
FDR_Q_GRID = (0.05,0.10, 0.15, 0.20, 0.25,0.3,0.35)      # pre-specified, small grid for adaptive choice
ALPHA = 0.05
MIN_GROUP_N = 10
MIN_CLASS_UNIQUES = 2
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42
RF_MAX_FEATURES = "sqrt"
RFE_TOP_K = 10
RFE_ESTIMATOR = LogisticRegression(solver="liblinear", max_iter=200, random_state=42,  class_weight="balanced")

# Post-selection feature curation settings
CURATION_MODE = "ask"   # "auto" (drop from blocklist) or "ask" (prompt user per dataset)
UNRELATED_FEATURES = {
    "4th-Ventricle",
    "BrainSegVol-to-eTIV",
    "Left-Lateral-Ventricle",
    "Optic-Chiasm",
}
# Names must exactly match column names (case-sensitive); applied after RFE-10.


# =========================
# Functions
# =========================
def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

def _read_csv_with_checks(path: Path, require_cols: bool = False) -> pd.DataFrame:
    if VERBOSE:
        print(f"Loading: {path}")
    df = pd.read_csv(path)
    if require_cols:
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing required: {sorted(missing)}")
    if "Subject_ID" in df.columns:
        df["Subject_ID"] = df["Subject_ID"].astype(str).str.strip()
    return df

def _check_no_duplicate_ids(df: pd.DataFrame, name: str) -> None:
    if "Subject_ID" not in df.columns:
        return
    dup = df["Subject_ID"][df["Subject_ID"].duplicated(keep=False)]
    if not dup.empty:
        sample = dup.unique()[:20]
        raise ValueError(f"Duplicate Subject_ID in {name}: {len(dup)} duplicate rows. Example IDs: {list(sample)}")

def _assert_perfect_id_match(struct_df: pd.DataFrame, graph_df: pd.DataFrame) -> None:
    struct_ids = set(struct_df["Subject_ID"]); graph_ids  = set(graph_df["Subject_ID"])
    only_in_struct = sorted(struct_ids - graph_ids); only_in_graph  = sorted(graph_ids  - struct_ids)
    if only_in_struct or only_in_graph:
        raise RuntimeError(
            "Unmatched Subject_ID rows detected.\n"
            + (f"- Only in structural ({len(only_in_struct)}): {only_in_struct[:25]}\n" if only_in_struct else "")
            + (f"- Only in graph ({len(only_in_graph)}): {only_in_graph[:25]}\n" if only_in_graph else "")
        )

def _coerce_label_for_compare(s: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    return numeric if numeric.notna().all() else s.astype(str).str.strip()

def _get_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in PROTECTED_COLUMNS]].copy()

def _get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = _get_feature_block(df).apply(pd.to_numeric, errors="coerce")
    return feats.select_dtypes(include=[np.number])

def _rejoin(protected: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([protected.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

def build_and_write_combined(struct_path: Path, graph_path: Path, out_path: Path) -> pd.DataFrame:
    _ensure_file_exists(struct_path); _ensure_file_exists(graph_path)
    struct_df = _read_csv_with_checks(struct_path, require_cols=True)
    graph_df  = _read_csv_with_checks(graph_path, require_cols=True)
    _check_no_duplicate_ids(struct_df, struct_path.name); _check_no_duplicate_ids(graph_df, graph_path.name)
    _assert_perfect_id_match(struct_df, graph_df)
    merged = pd.merge(struct_df, graph_df, on="Subject_ID", suffixes=("_struct", "_graph"),
                      how="inner", validate="one_to_one")
    a = _coerce_label_for_compare(merged["Label_struct"]); b = _coerce_label_for_compare(merged["Label_graph"])
    mismask = a != b
    if mismask.any():
        bad = merged.loc[mismask, ["Subject_ID", "Label_struct", "Label_graph"]].head(25).to_string(index=False)
        raise RuntimeError(f"Label mismatch in combined.\nFirst few:\n{bad}")
    merged = merged.rename(columns={"Label_struct": "Label"}).drop(columns=["Label_graph"], errors="ignore")
    if VERBOSE: print(f"Writing combined: {out_path}")
    merged.to_csv(out_path, index=OUTPUT_INDEX)
    return merged

def report_duplicate_headers(df: pd.DataFrame, name: str) -> None:
    counts = pd.Series(df.columns, dtype="object").value_counts()
    dups = counts[counts > 1]
    if dups.empty:
        print(f"[{name}] No duplicate headers.")
    else:
        print(f"[{name}] Duplicate headers:")
        for col, n in dups.items():
            idxs = [i for i, c in enumerate(df.columns) if c == col]
            print(f"  - '{col}' x{n} at positions {idxs}")

# ---- Cleaning helpers (features-only) ----
def _remove_duplicate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, list]:
    if df.shape[1] <= 1: return df, 0, []
    hashes = {col: pd.util.hash_pandas_object(df[col], index=False).sum() for col in df.columns}
    seen = {}; drop_cols = []
    for col, h in hashes.items():
        if h in seen and df[col].equals(df[seen[h]]): drop_cols.append(col)
        else: seen[h] = col
    return df.drop(columns=drop_cols), len(drop_cols), drop_cols

def _remove_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, list]:
    nun = df.nunique(dropna=False); drop_cols = nun[nun <= 1].index.tolist()
    return df.drop(columns=drop_cols), len(drop_cols), drop_cols

def _remove_infinite_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, list]:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0: return df, 0, []
    inf_mask = np.isinf(df[num_cols].to_numpy()); cols_with_inf = set(num_cols[np.any(inf_mask, axis=0)])
    return df.drop(columns=list(cols_with_inf)), len(cols_with_inf), sorted(cols_with_inf)

def _remove_missing_value_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, list]:
    miss_mask = df.isna().any(axis=0); drop_cols = miss_mask[miss_mask].index.tolist()
    return df.drop(columns=drop_cols), len(drop_cols), drop_cols

def clean_dataset(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, Dict[str, int], int, int]:
    before_cols = df.shape[1]
    prot = df[PROTECTED_COLUMNS].copy()
    feats = _get_feature_block(df)
    feats, n_dupes, _ = _remove_duplicate_columns(feats)
    feats, n_const, _ = _remove_constant_columns(feats)
    feats, n_inf,   _ = _remove_infinite_columns(feats)
    feats, n_na,    _ = _remove_missing_value_columns(feats)
    cleaned = _rejoin(prot, feats)
    after_cols = cleaned.shape[1]
    counts = dict(duplicate_cols_removed=n_dupes,
                  constant_cols_removed=n_const,
                  infinite_cols_removed=n_inf,
                  missing_value_cols_removed=n_na)
    if VERBOSE:
        print(f"[{name}] clean: {before_cols} -> {after_cols} (removed {before_cols - after_cols})")
        for k, v in counts.items(): print(f"  {k}: {v}")
    return cleaned, counts, before_cols, after_cols

# ---- Split locking ----
def _file_sha256(path: Path, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""): h.update(chunk)
    return h.hexdigest()

def _manifest_load(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def _manifest_save(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def _assert_stage_files_exist(label: str, paths: List[Path]) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing: raise RuntimeError(f"[{label}] Missing prior outputs: {missing}")

def _register_split_once(clean_path: Path, name: str, out_train: Path, out_test: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = _manifest_load(PIPELINE_MANIFEST); key = f"split:{name}"
    src_hash = _file_sha256(clean_path)
    both = out_train.exists() and out_test.exists(); xor = out_train.exists() ^ out_test.exists()
    if xor: raise RuntimeError(f"[{name}] Inconsistent split: one of train/test exists.")
    if both:
        entry = m.get(key)
        if not entry or entry.get("sha256") != src_hash or entry.get("clean_path") != clean_path.name:
            raise RuntimeError(f"[{name}] Split exists but manifest mismatch/absent; refusing to proceed.")
        if VERBOSE: print(f"[{name}] Using locked split.")
        return pd.read_csv(out_train), pd.read_csv(out_test)
    df = pd.read_csv(clean_path); y = df["Label"]
    cls = y.value_counts()
    if (cls < 2).any(): raise ValueError(f"[{name}] Stratified split impossible; classes <2: {cls[cls<2].to_dict()}")
    tr_idx, te_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    train_df = df.loc[tr_idx].reset_index(drop=True); test_df = df.loc[te_idx].reset_index(drop=True)
    train_df.to_csv(out_train, index=OUTPUT_INDEX); test_df.to_csv(out_test, index=OUTPUT_INDEX)
    m[key] = {"clean_path": clean_path.name, "sha256": src_hash,
              "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
              "train_rows": len(train_df), "test_rows": len(test_df),
              "created_at": dt.datetime.now().isoformat(timespec="seconds")}
    _manifest_save(PIPELINE_MANIFEST, m)
    if VERBOSE: print(f"[{name}] Split created & locked. train={len(train_df)} test={len(test_df)}")
    return train_df, test_df

# ---- Outlier helpers (TRAIN-learned Tukey bounds) ----
def _tukey_bounds_from_train(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute per-feature Tukey (IQR) bounds on TRAIN only.

    Returns
    -------
    dict
        Mapping {feature -> (lower, upper)} where bounds are Q1 - 1.5*IQR and Q3 + 1.5*IQR.
        Non-finite or zero-IQR features receive (-inf, inf).

    Notes
    -----
    These bounds are later used to flag out-of-range values in TRAIN and TEST.
    """
    feats = _get_numeric_features(train_df)
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in feats.columns:
        s = feats[col].dropna()
        if s.empty:
            bounds[col] = (-np.inf, np.inf)
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            bounds[col] = (-np.inf, np.inf)
        else:
            bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    if VERBOSE:
        print(f"Tukey bounds learned for {len(bounds)} numeric feature(s).")
    return bounds

def _flag_outlier_features(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.Series:
    """
    Count, for each row, how many numeric features fall outside the TRAIN-learned Tukey bounds.

    Parameters
    ----------
    df : pd.DataFrame
    bounds : dict
        Output of `_tukey_bounds_from_train`.

    Returns
    -------
    pd.Series
        Per-row integer counts of out-of-bounds features.
    """
    feats = _get_numeric_features(df)
    if feats.shape[1] == 0:
        return pd.Series(0, index=df.index)

    outlier_mask = pd.DataFrame(False, index=feats.index, columns=feats.columns)
    for col in feats.columns:
        lo, up = bounds.get(col, (-np.inf, np.inf))
        v = feats[col]
        outlier_mask[col] = ((v < lo) | (v > up)).fillna(False)

    return outlier_mask.sum(axis=1)

# Doc: Return a boolean mask flagging rows with >= ROW_OUTLIER_FRACTION of numeric features outside bounds.
def _apply_bounds_flag_rows(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.Series:
    feats = _get_numeric_features(df)
    if feats.shape[1] == 0: return pd.Series(False, index=df.index)
    O = pd.DataFrame(False, index=feats.index, columns=feats.columns)
    for col in feats.columns:
        lo, up = bounds.get(col, (-np.inf, np.inf)); v = feats[col]
        O[col] = ((v < lo) | (v > up)).fillna(False)
    per_row_counts = O.sum(axis=1)
    threshold = int(np.ceil(ROW_OUTLIER_FRACTION * feats.shape[1]))
    return per_row_counts >= threshold

def remove_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, name: str,
                    out_train: Path, out_test: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outlier filtering using TRAIN-learned Tukey bounds and a row-level fraction rule.

    Procedure
    ---------
    1) Learn per-feature Tukey bounds on TRAIN only.
    2) Compute, for TRAIN and TEST, per-row counts of features outside bounds.
    3) Flag rows with counts >= ROW_OUTLIER_FRACTION * (# numeric features).
    4) Drop rows in BOTH splits only if the proportion flagged in TRAIN is < MAX_DROP_FRACTION_TRAIN.
    5) Persist diagnostics (feature-level bounds, per-row counts/flags).

    All thresholds and decisions are derived from TRAIN to avoid leakage.
    """
    # 1) Learn bounds on TRAIN only
    bounds = _tukey_bounds_from_train(train_df)

    # 2) Save feature-level bounds diagnostics
    feats_train = _get_numeric_features(train_df)
    q1 = feats_train.quantile(0.25)
    q3 = feats_train.quantile(0.75)
    iqr = q3 - q1
    lower = (q1 - 1.5 * iqr).where((iqr != 0) & iqr.notna(), -np.inf).fillna(-np.inf)
    upper = (q3 + 1.5 * iqr).where((iqr != 0) & iqr.notna(),  np.inf).fillna( np.inf)
    bounds_df = pd.DataFrame({"lower": lower, "upper": upper, "IQR": iqr})
    bpath = DIAG_DIR / f"{name.lower()}_outlier_feature_bounds.csv"
    bounds_df.to_csv(bpath)
    if VERBOSE:
        print(f"[{name}] Saved feature bounds -> {bpath.name}")

    # 3) Row-level outlier counts on TRAIN and TEST
    tr_counts = _flag_outlier_features(train_df, bounds)
    te_counts = _flag_outlier_features(test_df, bounds)

    nfeat = _get_numeric_features(train_df).shape[1]
    row_threshold = ROW_OUTLIER_FRACTION * max(1, nfeat)  # no ceil (matches the reference script)
    tr_flag = tr_counts >= row_threshold
    te_flag = te_counts >= row_threshold

    # 4) Diagnostics: per-row counts (train/test)
    diag = pd.DataFrame({
        "split": (["TRAIN"] * len(train_df)) + (["TEST"] * len(test_df)),
        "row_idx_within_split": list(range(len(train_df))) + list(range(len(test_df))),
        "Subject_ID": pd.concat([train_df["Subject_ID"], test_df["Subject_ID"]], ignore_index=True),
        "Label": pd.concat([train_df["Label"], test_df["Label"]], ignore_index=True),
        "outlier_feature_count": pd.concat([tr_counts, te_counts], ignore_index=True).astype(int),
        "is_flagged_row": pd.concat([tr_flag, te_flag], ignore_index=True).astype(bool)
    })
    dpath = DIAG_DIR / f"{name.lower()}_outlier_row_counts.csv"
    diag.to_csv(dpath, index=False)
    if VERBOSE:
        print(f"[{name}] Saved row-level outlier diagnostics -> {dpath.name}")

    # 5) Policy: only drop if TRAIN-flagged < MAX_DROP_FRACTION_TRAIN * |TRAIN|
    n_tr_flag = int(tr_flag.sum())
    drop_allowed = (n_tr_flag < MAX_DROP_FRACTION_TRAIN * len(train_df))

    print(f"[{name}] Numeric features: {nfeat}")
    print(f"[{name}] Row outlier threshold: {ROW_OUTLIER_FRACTION:.2f} x {nfeat} = {row_threshold:.2f}")
    print(f"[{name}] Flagged rows — TRAIN: {n_tr_flag} / {len(train_df)} | TEST: {int(te_flag.sum())} / {len(test_df)}")
    print(f"[{name}] Drop policy: drop if TRAIN-flagged < {MAX_DROP_FRACTION_TRAIN:.0%} of TRAIN "
          f"({n_tr_flag} < {MAX_DROP_FRACTION_TRAIN * len(train_df):.0f}) -> drop_allowed={drop_allowed}")

    if drop_allowed and (n_tr_flag > 0 or int(te_flag.sum()) > 0):
        tr_keep = train_df.loc[~tr_flag].reset_index(drop=True)
        te_keep = test_df.loc[~te_flag].reset_index(drop=True)
        print(f"[{name}] Dropped rows. TRAIN kept={len(tr_keep)} | TEST kept={len(te_keep)}")
    else:
        tr_keep = train_df.reset_index(drop=True)
        te_keep = test_df.reset_index(drop=True)
        if n_tr_flag == 0 and int(te_flag.sum()) == 0:
            print(f"[{name}] No rows dropped (none flagged).")
        else:
            print(f"[{name}] No rows dropped (train proportion >= threshold).")

    tr_keep.to_csv(out_train, index=OUTPUT_INDEX)
    te_keep.to_csv(out_test, index=OUTPUT_INDEX)
    print(f"[{name}] Saved after outlier step -> {out_train.name} ({len(tr_keep)}), {out_test.name} ({len(te_keep)})")
    print("-" * 60)
    return tr_keep, te_keep

# ---- Correlation pruning (TRAIN-learned; applied to TEST) ----
def _heatmap(matrix: pd.DataFrame, title: str, outfile: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def corr_prune_train_apply_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
    out_train: Path,
    out_test: Path,
    threshold: float = CORR_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Correlation pruning learned on TRAIN and mirrored to TEST.

    Rules
    -----
    - Hard fail if any NaNs are present in numeric features of TRAIN/TEST at this stage.
    - Remove TRAIN-constant features before computing correlations (mirror the removal to TEST).
    - Build an absolute pairwise correlation matrix on TRAIN; for any pair > threshold,
      drop the feature with the smaller absolute correlation with the (numeric) label.
    - Rejoin protected columns and persist manifests and optional heatmaps.

    Returns
    -------
    (tr_out, te_out, kept_features, dropped_features)
    """
    # ---- Extract numeric features
    Xtr_raw = _get_numeric_features(train_df)
    Xte_raw = _get_numeric_features(test_df)
    ytr = pd.to_numeric(train_df["Label"], errors="coerce")

    # ---- Strict NaN checks (hard error)
    tr_nan_counts = Xtr_raw.isna().sum()
    te_nan_counts = Xte_raw.isna().sum()
    n_tr_nan = int(tr_nan_counts.sum())
    n_te_nan = int(te_nan_counts.sum())
    if n_tr_nan > 0 or n_te_nan > 0:
        def _fmt_nan_report(counts: pd.Series, top_k: int = 15) -> str:
            bad = counts[counts > 0].sort_values(ascending=False)
            head = ", ".join([f"{c}({int(v)})" for c, v in bad.head(top_k).items()])
            more = "" if len(bad) <= top_k else f", ... (+{len(bad)-top_k} more)"
            return head + more if not bad.empty else "(none)"
        raise RuntimeError(
            f"[{name}] NaNs detected in numeric features after prior cleaning.\n"
            f"- TRAIN total NaNs: {n_tr_nan} | cols: {_fmt_nan_report(tr_nan_counts)}\n"
            f"- TEST  total NaNs: {n_te_nan} | cols: {_fmt_nan_report(te_nan_counts)}\n"
            f"Please investigate earlier stages; this step expects no NaNs."
        )

    # ---- Remove TRAIN-constant features (std==0) BEFORE any correlation; mirror drop to TEST
    stds = Xtr_raw.std(axis=0, ddof=0)
    train_constant_cols = stds[stds == 0].index.tolist()
    if train_constant_cols:
        if VERBOSE:
            preview = ", ".join(train_constant_cols[:20])
            tail = "" if len(train_constant_cols) <= 20 else f", ... (+{len(train_constant_cols)-20} more)"
            print(f"[{name}] Removing {len(train_constant_cols)} TRAIN-constant feature(s): {preview}{tail}")
        Xtr = Xtr_raw.drop(columns=train_constant_cols, errors="ignore")
        Xte = Xte_raw.drop(columns=[c for c in train_constant_cols if c in Xte_raw.columns], errors="ignore")
    else:
        if VERBOSE:
            print(f"[{name}] No TRAIN-constant numeric features found.")
        Xtr, Xte = Xtr_raw, Xte_raw

    # Safety: ensure we still have columns to correlate
    if Xtr.shape[1] == 0:
        # Rejoin and exit early — nothing to prune
        tr_out = _rejoin(train_df[PROTECTED_COLUMNS], Xtr)
        te_out = _rejoin(test_df[PROTECTED_COLUMNS], Xte)
        tr_out.to_csv(out_train, index=OUTPUT_INDEX)
        te_out.to_csv(out_test, index=OUTPUT_INDEX)

        man = _manifest_load(CORR_MANIFEST)
        man[f"corr:{name}"] = {
            "threshold": threshold,
            "dropped": [],
            "kept": [],
            "train_constant_dropped": train_constant_cols,
            "train_out": out_train.name,
            "test_out": out_test.name,
        }
        _manifest_save(CORR_MANIFEST, man)
        print(f"[{name}] Corr-prune skipped (no numeric features after constant removal).")
        return tr_out, te_out, [], []

    # ---- (Optional) Heatmaps BEFORE pruning but AFTER constant removal (avoids divide-by-zero warnings)
    if SAVE_HEATMAPS and Xtr.shape[1] > 1:
        _heatmap(Xtr.corr().fillna(0.0), f"{name} TRAIN corr (pre-prune)", PLOTS_DIR / f"{name.lower()}_train_corr_before.png")
    if SAVE_HEATMAPS and Xte.shape[1] > 1:
        _heatmap(Xte.corr().fillna(0.0), f"{name} TEST corr (pre-prune)", PLOTS_DIR / f"{name.lower()}_test_corr_before.png")

    # ---- Label correlation & pairwise correlation on TRAIN
    if ytr.isna().any():
        raise RuntimeError(f"[{name}] Label column contains NaNs after coercion — cannot compute correlations.")
    label_corr = Xtr.corrwith(ytr.astype(float)).abs().fillna(0.0)
    corr_train = Xtr.corr().abs()

    # ---- Greedy prune by correlation threshold, keep the more label-correlated of each pair
    cols = list(corr_train.columns)
    drop_set: Set[str] = set()
    for i in range(len(cols)):
        for j in range(i):
            cij = corr_train.iloc[i, j]
            if pd.notna(cij) and cij > threshold:
                fi, fj = cols[i], cols[j]
                ci = float(label_corr.get(fi, 0.0))
                cj = float(label_corr.get(fj, 0.0))
                drop = fi if ci < cj else fj
                drop_set.add(drop)

    kept_features = [c for c in cols if c not in drop_set]
    Xtr_pruned = Xtr[kept_features].copy()
    Xte_pruned = Xte[kept_features].copy()

    # ---- Rejoin protected cols
    tr_out = _rejoin(train_df[PROTECTED_COLUMNS], Xtr_pruned)
    te_out = _rejoin(test_df[PROTECTED_COLUMNS], Xte_pruned)

    # ---- Heatmaps AFTER pruning
    if SAVE_HEATMAPS and Xtr_pruned.shape[1] > 1:
        _heatmap(Xtr_pruned.corr().fillna(0.0), f"{name} TRAIN corr (post-prune)", PLOTS_DIR / f"{name.lower()}_train_corr_after.png")
    if SAVE_HEATMAPS and Xte_pruned.shape[1] > 1:
        _heatmap(Xte_pruned.corr().fillna(0.0), f"{name} TEST corr (post-prune)", PLOTS_DIR / f"{name.lower()}_test_corr_after.png")

    # ---- Save outputs
    tr_out.to_csv(out_train, index=OUTPUT_INDEX)
    te_out.to_csv(out_test, index=OUTPUT_INDEX)

    # ---- Manifest: include constants removed
    man = _manifest_load(CORR_MANIFEST)
    man[f"corr:{name}"] = {
        "threshold": threshold,
        "dropped": sorted(drop_set),
        "kept": kept_features,
        "train_constant_dropped": train_constant_cols,
        "train_out": out_train.name,
        "test_out": out_test.name,
    }
    _manifest_save(CORR_MANIFEST, man)

    print(f"[{name}] Corr-prune: dropped {len(drop_set)} / {len(cols)} features (thr={threshold}). Kept: {len(kept_features)}")
    if train_constant_cols:
        print(f"[{name}] Additionally removed {len(train_constant_cols)} TRAIN-constant feature(s) before pruning.")
    return tr_out, te_out, kept_features, sorted(drop_set)


# ---- RobustScaler (fit on TRAIN after correlation pruning) ----
# Doc: Return numeric feature block; fill any residual NaNs with 0 before scaling (with warning).
def _numeric_features_only(df: pd.DataFrame) -> pd.DataFrame:
    feats_num = _get_feature_block(df).apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    if feats_num.isna().any().any():
        n = int(feats_num.isna().sum().sum())
        if n>0 and VERBOSE: print(f"Warning: filling {n} NaNs with 0 before scaling.")
        feats_num = feats_num.fillna(0)
    return feats_num

# Doc: Fit RobustScaler on TRAIN numeric features and return (scaler, column_order).
def _fit_scaler_on_train(train_df: pd.DataFrame) -> Tuple[RobustScaler, List[str]]:
    feats = _numeric_features_only(train_df); cols = list(feats.columns)
    scaler = RobustScaler(with_centering=ROBUST_WITH_CENTERING,
                          with_scaling=ROBUST_WITH_SCALING,
                          quantile_range=ROBUST_QUANTILE_RANGE,
                          unit_variance=ROBUST_UNIT_VARIANCE)
    scaler.fit(feats.values)
    if VERBOSE: print(f"RobustScaler fit on {len(cols)} numeric feature(s).")
    return scaler, cols

# Doc: Apply a previously fitted scaler to df; enforces exact column match and rejoins protected columns.
def _apply_scaler(df: pd.DataFrame, scaler: RobustScaler, cols: List[str]) -> pd.DataFrame:
    protected = df[PROTECTED_COLUMNS].copy()
    feats = _numeric_features_only(df)
    missing = [c for c in cols if c not in feats.columns]
    extra   = [c for c in feats.columns if c not in cols]
    if missing or extra:
        raise RuntimeError(f"Scaler feature mismatch.\nMissing: {missing}\nExtra: {extra}")
    scaled = pd.DataFrame(scaler.transform(feats[cols].values), columns=cols, index=df.index)
    passthrough = [c for c in _get_feature_block(df).columns if c not in cols]
    if passthrough:
        scaled = pd.concat([scaled, df[passthrough].reset_index(drop=True)], axis=1)
        scaled = scaled[_get_feature_block(df).columns]
    return _rejoin(protected, scaled)

# Doc: Fit RobustScaler on TRAIN, transform TRAIN/TEST, persist scaled CSVs, and record a scaling manifest.
def scale_and_save(train_path: Path, test_path: Path,
                   out_train_path: Path, out_test_path: Path,
                   scaler_path: Path, dataset_name: str) -> None:
    for p in [train_path, test_path]:
        if not p.exists(): raise RuntimeError(f"[{dataset_name}] Missing input for scaling: {p.name}")
    tr = pd.read_csv(train_path); te = pd.read_csv(test_path)
    scaler, cols = _fit_scaler_on_train(tr)
    tr_s = _apply_scaler(tr, scaler, cols); te_s = _apply_scaler(te, scaler, cols)
    tr_s.to_csv(out_train_path, index=OUTPUT_INDEX); te_s.to_csv(out_test_path, index=OUTPUT_INDEX)
    joblib.dump({"scaler": scaler, "columns": cols}, scaler_path)
    man = _manifest_load(SCALING_MANIFEST)
    man[f"robust:{dataset_name}"] = {
        "train_in": train_path.name, "test_in": test_path.name,
        "train_out": out_train_path.name, "test_out": out_test_path.name,
        "scaler_file": scaler_path.name, "n_cols_scaled": len(cols),
        "quantile_range": list(ROBUST_QUANTILE_RANGE),
        "with_centering": ROBUST_WITH_CENTERING, "with_scaling": ROBUST_WITH_SCALING,
        "unit_variance": ROBUST_UNIT_VARIANCE
    }
    _manifest_save(SCALING_MANIFEST, man)
    print(f"[{dataset_name}] Robust-scaled saved. Train={out_train_path.name} Test={out_test_path.name}")

# ---- Feature selection (TRAIN only; apply to TEST) ----
def _numeric_features_scaled(df: pd.DataFrame) -> pd.DataFrame:
    return _get_feature_block(df).select_dtypes(include=[np.number])

def _split_groups(train_df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    y = train_df["Label"]
    return y[y == 0].index, y[y == 1].index

def _ttest_screen(train_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Univariate screening on TRAIN using Welch's t-test with BH-FDR control.

    Behavior
    --------
    - Skips low-support or degenerate features (per-class counts/uniques).
    - If FDR_ENABLE:
        * If FDR_Q is set (float), use that q.
        * Else select the smallest q in FDR_Q_GRID yielding at least MIN_TTEST_PASS features
          (otherwise use the largest q and accept fewer). Reports the chosen q and expected false discoveries.
    - If FDR_ENABLE is False, use raw p-values <= ALPHA.

    Returns
    -------
    kept_features : list[str]
    results_df : pd.DataFrame
        Sorted by raw p-value with an added 'p_adj_BH' column.

    References
    ----------
    Benjamini & Hochberg (1995), J. R. Stat. Soc. B.
    """
    # ----- prepare data
    X = _numeric_features_scaled(train_df)
    y = train_df["Label"].astype(int)
    idx0 = y[y == 0].index
    idx1 = y[y == 1].index

    if len(idx0) < MIN_GROUP_N or len(idx1) < MIN_GROUP_N:
        print(f"Warning: small classes (n0={len(idx0)}, n1={len(idx1)}); low-support features will be skipped.")

    features, pvals, n0s, n1s = [], [], [], []
    for col in X.columns:
        x0 = X.loc[idx0, col].dropna()
        x1 = X.loc[idx1, col].dropna()
        # support / degeneracy checks (avoid 0-variance within class)
        if len(x0) < MIN_GROUP_N or len(x1) < MIN_GROUP_N or x0.nunique() < MIN_CLASS_UNIQUES or x1.nunique() < MIN_CLASS_UNIQUES:
            continue
        _, p = ttest_ind(x0, x1, equal_var=False)  # Welch t-test
        features.append(col); pvals.append(float(p)); n0s.append(len(x0)); n1s.append(len(x1))

    # Assemble base results table
    res = pd.DataFrame({
        "feature": features,
        "p_value": pvals,
        "n_class0": n0s,
        "n_class1": n1s
    }).sort_values("p_value").reset_index(drop=True)

    kept: List[str] = []

    if not FDR_ENABLE:
        # Raw p-value mode
        kept_mask = res["p_value"] <= ALPHA
        kept = res.loc[kept_mask, "feature"].tolist()
        print(f"Welch t-test (raw p <= {ALPHA}): kept {len(kept)} / {len(res)}")
        return kept, res

    # ----- FDR (BH) mode
    if len(res) == 0:
        print("Welch t-test + BH-FDR: no features had sufficient support; kept 0 / 0.")
        res["p_adj_BH"] = []
        return [], res

    # Compute BH adjusted p-values once (independent of q threshold)
    m = len(res)
    order = np.argsort(res["p_value"].to_numpy())
    ranks = np.arange(1, m + 1)
    p_sorted = res.loc[order, "p_value"].to_numpy()
    p_adj_sorted = np.minimum.accumulate((p_sorted[::-1] * m / ranks[::-1]))[::-1]
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = np.minimum(p_adj_sorted, 1.0)
    res["p_adj_BH"] = p_adj

    # Select policy:
    chosen_q = None
    grid = None

    # Case 1: legacy single-q mode (FDR_Q is a float)
    if isinstance(FDR_Q, (int, float)):
        chosen_q = float(FDR_Q)
        kept_mask = res["p_adj_BH"] <= chosen_q
        kept = res.loc[kept_mask, "feature"].tolist()
        print(f"Welch t-test + BH-FDR@{chosen_q}: kept {len(kept)} / {len(res)} "
              f"(E[false disc] <= {chosen_q * max(len(kept), 0):.2f})")
        return kept, res

    # Case 2: adaptive small-grid policy (use global FDR_Q_GRID)
    # Choose smallest q in the grid yielding at least MIN_TTEST_PASS kept features; else use largest q.
    if 'FDR_Q_GRID' in globals() and isinstance(FDR_Q_GRID, (list, tuple)) and len(FDR_Q_GRID) > 0:
        grid = sorted([float(q) for q in FDR_Q_GRID])
        # try in ascending order to find first q with enough features
        for q in grid:
            r = int((res["p_adj_BH"] <= q).sum())
            if r >= MIN_TTEST_PASS:
                chosen_q = q
                break
        if chosen_q is None:
            chosen_q = grid[-1]  # accept fewer than MIN_TTEST_PASS; do NOT force
        kept_mask = res["p_adj_BH"] <= chosen_q
        kept = res.loc[kept_mask, "feature"].tolist()
        print(
            f"Welch t-test + BH-FDR adaptive: q_grid={grid}, chosen_q={chosen_q} -> "
            f"kept {len(kept)} / {len(res)} (target >= {MIN_TTEST_PASS}; "
            f"E[false disc] <= {chosen_q * max(len(kept), 0):.2f})"
        )
        # (optional) annotate the DataFrame with the choice for downstream reporting
        res.attrs["chosen_q"] = chosen_q
        res.attrs["q_grid"] = grid
        res.attrs["min_pass"] = MIN_TTEST_PASS
        return kept, res

    # Fallback (shouldn't happen): use default FDR_Q if grid not configured
    chosen_q = 0.10
    kept_mask = res["p_adj_BH"] <= chosen_q
    kept = res.loc[kept_mask, "feature"].tolist()
    print(f"Welch t-test + BH-FDR@{chosen_q} (fallback): kept {len(kept)} / {len(res)} "
          f"(E[false disc] <= {chosen_q * max(len(kept), 0):.2f})")
    return kept, res

def _rf_top_k(train_df: pd.DataFrame, features: List[str], k: int) -> Tuple[List[str], RandomForestClassifier]:
    if not features: return [], None
    X = _numeric_features_scaled(train_df)[features].copy()
    y = train_df["Label"].astype(int)
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE,
                                n_jobs=-1, max_features=RF_MAX_FEATURES, class_weight="balanced")
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feats = list(importances.index[:min(k, len(importances))])
    return top_feats, rf

def _rfe_top_k(train_df: pd.DataFrame, features: List[str], k: int) -> Tuple[List[str], RFE]:
    if not features: return [], None
    X = _numeric_features_scaled(train_df)[features].copy()
    y = train_df["Label"].astype(int)
    rfe = RFE(estimator=RFE_ESTIMATOR, n_features_to_select=min(k, X.shape[1]), step=1)
    rfe.fit(X, y)
    support = pd.Series(rfe.support_, index=X.columns)
    final_feats = list(support[support].index)
    return final_feats, rfe

def _save_selected(train_df: pd.DataFrame, test_df: pd.DataFrame, selected: List[str],
                   out_train: Path, out_test: Path) -> None:
    cols = PROTECTED_COLUMNS + selected
    train_df[cols].to_csv(out_train, index=False)
    test_df[cols].to_csv(out_test, index=False)

def _report(dataset: str, stage: str, feats: List[str]) -> None:
    print(f"[{dataset}] {stage} features ({len(feats)}):")
    print("  " + (", ".join(feats) if feats else "(none)"))
    print("-"*60)

def _run_feature_selection_for_dataset(train_path: Path, test_path: Path,
                                       out_ttrain: Path, out_ttest: Path,
                                       out_rftrain: Path, out_rftest: Path,
                                       out_rf_model: Path,
                                       out_rfetrain: Path, out_rfetest: Path,
                                       out_rfe_model: Path,
                                       dataset_name: str,
                                       fs_report: Dict[str, dict]) -> None:
    if not train_path.exists() or not test_path.exists():
        raise RuntimeError(f"[{dataset_name}] Missing scaled inputs for feature selection.")
    train_df = pd.read_csv(train_path); test_df = pd.read_csv(test_path)

    # 1) Welch t-tests (TRAIN only)
    t_keep, t_df = _ttest_screen(train_df)
    _save_selected(train_df, test_df, t_keep, out_ttrain, out_ttest)
    _report(dataset_name, "T-test pass", t_keep)

    # 2) RF importance -> top 20 (TRAIN only, from t-test-passed)
    rf_keep, rf_model = _rf_top_k(train_df, t_keep, k=20)
    _save_selected(train_df, test_df, rf_keep, out_rftrain, out_rftest)
    if rf_model is not None: joblib.dump(rf_model, out_rf_model)
    _report(dataset_name, "RF top-20", rf_keep)

    # 3) RFE -> top 10 (TRAIN only, from RF-20)
    rfe_keep, rfe_model = _rfe_top_k(train_df, rf_keep, k=RFE_TOP_K)
    _save_selected(train_df, test_df, rfe_keep, out_rfetrain, out_rfetest)
    if rfe_model is not None: joblib.dump(rfe_model, out_rfe_model)
    _report(dataset_name, "RFE top-10", rfe_keep)

    fs_report[dataset_name] = {
        "ttest_kept": t_keep,
        "rf_top20": rf_keep,
        "rfe_top10": rfe_keep,
        "ttest_results_preview": t_df.head(30).to_dict(orient="list")
    }

# --- NEW: Post-selection curation helpers ---
def _curation_candidate_columns(df: pd.DataFrame) -> List[str]:
    """Return just the *feature* columns (no Subject_ID/Label)."""
    return [c for c in df.columns if c not in PROTECTED_COLUMNS]

def _drop_features(df: pd.DataFrame, to_drop: Set[str]) -> pd.DataFrame:
    """Drop features (case-sensitive, exact match) if present; keep Subject_ID/Label."""
    safe_drop = [c for c in to_drop if c in df.columns]
    if safe_drop:
        if VERBOSE: print(f"Dropping {len(safe_drop)} feature(s): {', '.join(sorted(safe_drop))}")
        df = df.drop(columns=safe_drop, errors="ignore")
    else:
        if VERBOSE: print("No curated features found to drop.")
    return df

def _pretty_feature_menu(cols: List[str]) -> str:
    if not cols:
        return "(none)"
    width = len(str(len(cols)))
    lines = [f"  {str(i+1).rjust(width)}. {c}" for i, c in enumerate(cols)]
    return "\n".join(lines)

def curate_post_fs(train_path: Path, test_path: Path,
                   out_train: Path, out_test: Path,
                   dataset_name: str,
                   mode: str = CURATION_MODE,
                   auto_blocklist: Set[str] = UNRELATED_FEATURES) -> Dict[str, List[str]]:
    """
    Post-RFE curation of selected features.

    Modes
    -----
    - "auto": drop any features present in the blocklist (exact, case-sensitive matches).
    - "ask": prompt once for indices/names to drop; same decision applied to TRAIN/TEST.

    Returns
    -------
    dict
        {"kept": [...], "dropped": [...]}  (also saves curated CSVs)
    """
    _ensure_file_exists(train_path); _ensure_file_exists(test_path)
    tr = pd.read_csv(train_path); te = pd.read_csv(test_path)

    # Candidate features to curate (exclude protected)
    feat_cols = [c for c in tr.columns if c not in PROTECTED_COLUMNS]

    # Early exit: nothing to curate
    if not feat_cols:
        tr.to_csv(out_train, index=False)
        te.to_csv(out_test, index=False)
        print(f"[{dataset_name}] Final features (0): (none) -> saved {out_train.name}, {out_test.name}")
        print("-"*60)
        return {"kept": [], "dropped": []}

    # Decide drop set
    to_drop: Set[str] = set()

    if mode.lower() == "auto":
        to_drop = set(f for f in auto_blocklist if f in feat_cols)
    else:
        # Build and show a numbered menu (once)
        width = len(str(len(feat_cols)))
        menu_lines = [f"  {str(i+1).rjust(width)}. {c}" for i, c in enumerate(feat_cols)]
        menu = "\n".join(menu_lines)
        try:
            raw = input(
                f"\n[{dataset_name}] Select features to DROP:\n{menu}\n\n"
                "Enter numbers (e.g., 1,3,7) or exact names (comma-separated).\n"
                "Type 'auto' to drop only the predefined blocklist, or press Enter to keep all.\n> "
            ).strip()
        except Exception:
            raw = "auto"  # non-interactive fallback

        if not raw:
            to_drop = set()
        elif raw.lower() == "auto":
            to_drop = set(f for f in auto_blocklist if f in feat_cols)
        else:
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            picked: Set[str] = set()
            for t in tokens:
                if t.isdigit():
                    i = int(t) - 1
                    if 0 <= i < len(feat_cols):
                        picked.add(feat_cols[i])
                elif t in feat_cols:
                    picked.add(t)
            to_drop = {f for f in picked if f in feat_cols}

    # Compute kept once; apply same drop to both splits (no duplicate prints)
    kept = [c for c in feat_cols if c not in to_drop]
    if to_drop:
        tr = tr.drop(columns=[c for c in to_drop if c in tr.columns], errors="ignore")
        te = te.drop(columns=[c for c in to_drop if c in te.columns], errors="ignore")

    # Save and print **only** the concise summary
    tr.to_csv(out_train, index=False)
    te.to_csv(out_test, index=False)
    print(f"[{dataset_name}] Final features ({len(kept)}): {', '.join(kept) if kept else '(none)'} -> saved {out_train.name}, {out_test.name}")
    print("-"*60)

    return {"kept": kept, "dropped": sorted(to_drop)}

# =========================
# Running it
# =========================
# 1) Build combined (strict ID & label checks)
combined_df = build_and_write_combined(STRUCT_FILE, GRAPH_FILE, COMBINED_FILE)

# 2) Load raws & header check
struct_df   = _read_csv_with_checks(STRUCT_FILE)
graph_df    = _read_csv_with_checks(GRAPH_FILE)
combined_df = _read_csv_with_checks(COMBINED_FILE)
report_duplicate_headers(struct_df, "STRUCTURAL (raw)")
report_duplicate_headers(graph_df,  "GRAPH (raw)")
report_duplicate_headers(combined_df, "COMBINED (raw)")

# 3) Clean features (keep Subject_ID & Label untouched)
struct_clean, _, _, _ = clean_dataset(struct_df, "STRUCTURAL")
graph_clean,  _, _, _ = clean_dataset(graph_df,  "GRAPH")
comb_clean,   _, _, _ = clean_dataset(combined_df, "COMBINED")
struct_clean.to_csv(OUT_STRUCT_CLEAN, index=OUTPUT_INDEX)
graph_clean.to_csv(OUT_GRAPH_CLEAN,  index=OUTPUT_INDEX)
comb_clean.to_csv(OUT_COMB_CLEAN,    index=OUTPUT_INDEX)

# 4) One-time stratified split (locked)
for p in [OUT_STRUCT_CLEAN, OUT_GRAPH_CLEAN, OUT_COMB_CLEAN]:
    _ensure_file_exists(p)
struct_train, struct_test = _register_split_once(OUT_STRUCT_CLEAN, "STRUCTURAL", STRUCT_TRAIN, STRUCT_TEST)
graph_train,  graph_test  = _register_split_once(OUT_GRAPH_CLEAN,  "GRAPH",      GRAPH_TRAIN,  GRAPH_TEST)
comb_train,   comb_test   = _register_split_once(OUT_COMB_CLEAN,   "COMBINED",   COMB_TRAIN,   COMB_TEST)

# 5) Outlier removal (TRAIN-learned Tukey; apply to both)
for p in [STRUCT_TRAIN, STRUCT_TEST, GRAPH_TRAIN, GRAPH_TEST, COMB_TRAIN, COMB_TEST]:
    _ensure_file_exists(p)
struct_train_olrm, struct_test_olrm = remove_outliers(struct_train, struct_test, "STRUCTURAL", STRUCT_TRAIN_OLRM, STRUCT_TEST_OLRM)
graph_train_olrm,  graph_test_olrm  = remove_outliers(graph_train,  graph_test,  "GRAPH",      GRAPH_TRAIN_OLRM,  GRAPH_TEST_OLRM)
comb_train_olrm,   comb_test_olrm   = remove_outliers(comb_train,   comb_test,   "COMBINED",   COMB_TRAIN_OLRM,   COMB_TEST_OLRM)

# 6) Correlation-pruning (>|0.8|) on TRAIN, applied to TEST
for p in [STRUCT_TRAIN_OLRM, STRUCT_TEST_OLRM, GRAPH_TRAIN_OLRM, GRAPH_TEST_OLRM, COMB_TRAIN_OLRM, COMB_TEST_OLRM]:
    _ensure_file_exists(p)
struct_train_corr, struct_test_corr, _, _ = corr_prune_train_apply_test(
    pd.read_csv(STRUCT_TRAIN_OLRM), pd.read_csv(STRUCT_TEST_OLRM),
    "STRUCTURAL", STRUCT_TRAIN_CORR, STRUCT_TEST_CORR, threshold=CORR_THRESHOLD
)
graph_train_corr, graph_test_corr, _, _ = corr_prune_train_apply_test(
    pd.read_csv(GRAPH_TRAIN_OLRM), pd.read_csv(GRAPH_TEST_OLRM),
    "GRAPH", GRAPH_TRAIN_CORR, GRAPH_TEST_CORR, threshold=CORR_THRESHOLD
)
comb_train_corr, comb_test_corr, _, _ = corr_prune_train_apply_test(
    pd.read_csv(COMB_TRAIN_OLRM), pd.read_csv(COMB_TEST_OLRM),
    "COMBINED", COMB_TRAIN_CORR, COMB_TEST_CORR, threshold=CORR_THRESHOLD
)

# 7) RobustScaler (fit on TRAIN after correlation pruning; apply to TEST)
scale_and_save(STRUCT_TRAIN_CORR, STRUCT_TEST_CORR, STRUCT_TRAIN_SCALED, STRUCT_TEST_SCALED, STRUCT_SCALER_FILE, "STRUCTURAL")
scale_and_save(GRAPH_TRAIN_CORR,  GRAPH_TEST_CORR,  GRAPH_TRAIN_SCALED,  GRAPH_TEST_SCALED,  GRAPH_SCALER_FILE,  "GRAPH")
scale_and_save(COMB_TRAIN_CORR,   COMB_TEST_CORR,   COMB_TRAIN_SCALED,   COMB_TEST_SCALED,   COMB_SCALER_FILE,   "COMBINED")

# 8) Feature selection (TRAIN only; apply to TEST) — Welch t-test -> RF top-20 -> RFE top-10
fs_report: Dict[str, dict] = {}
_run_feature_selection_for_dataset(
    STRUCT_TRAIN_SCALED, STRUCT_TEST_SCALED,
    STRUCT_TRAIN_TPASS, STRUCT_TEST_TPASS,
    STRUCT_TRAIN_RF20, STRUCT_TEST_RF20, STRUCT_RF_FILE,
    STRUCT_TRAIN_RFE10, STRUCT_TEST_RFE10, STRUCT_RFE_FILE,
    "STRUCTURAL", fs_report
)
_run_feature_selection_for_dataset(
    GRAPH_TRAIN_SCALED, GRAPH_TEST_SCALED,
    GRAPH_TRAIN_TPASS, GRAPH_TEST_TPASS,
    GRAPH_TRAIN_RF20, GRAPH_TEST_RF20, GRAPH_RF_FILE,
    GRAPH_TRAIN_RFE10, GRAPH_TEST_RFE10, GRAPH_RFE_FILE,
    "GRAPH", fs_report
)
_run_feature_selection_for_dataset(
    COMB_TRAIN_SCALED, COMB_TEST_SCALED,
    COMB_TRAIN_TPASS, COMB_TEST_TPASS,
    COMB_TRAIN_RF20, COMB_TEST_RF20, COMB_RF_FILE,
    COMB_TRAIN_RFE10, COMB_TEST_RFE10, COMB_RFE_FILE,
    "COMBINED", fs_report
)

# Save feature-selection report
with open(FS_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(fs_report, f, indent=2)

print("Feature selection completed. Report saved to:", FS_REPORT_JSON.name)

# 9) Post-selection curation (interactive "ask" or automatic "auto")
curate_post_fs(STRUCT_TRAIN_RFE10, STRUCT_TEST_RFE10, STRUCT_TRAIN_RFE10_CUR, STRUCT_TEST_RFE10_CUR, "STRUCTURAL")
curate_post_fs(GRAPH_TRAIN_RFE10,  GRAPH_TEST_RFE10,  GRAPH_TRAIN_RFE10_CUR,  GRAPH_TEST_RFE10_CUR,  "GRAPH")
curate_post_fs(COMB_TRAIN_RFE10,   COMB_TEST_RFE10,   COMB_TRAIN_RFE10_CUR,   COMB_TEST_RFE10_CUR,   "COMBINED")

# 10) Model training:
#     - Hyperparameter optimization via Bayesian search (fallback: randomized) with repeated stratified CV
#     - Oversampling inside training folds only to target a 1:2 minority:majority ratio
#     - No nested CV; decision threshold tuned via independent repeated CV on TRAIN (OOF scores)
from pathlib import Path as _Path
import json as _json
import warnings
from datetime import datetime as _dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as _plt
import joblib, json

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef,
    brier_score_loss, roc_curve
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.svm import SVC as _SVC
from sklearn.ensemble import RandomForestClassifier as _RF, HistGradientBoostingClassifier as _HGB

from imblearn.pipeline import Pipeline as _ImbPipeline
from imblearn.over_sampling import RandomOverSampler as _ROS

from sklearn.linear_model import LogisticRegression as _LR
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.calibration import CalibratedClassifierCV as _Cal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
from sklearn.naive_bayes import GaussianNB as _GNB
from sklearn.neighbors import KNeighborsClassifier as _KNN
from sklearn.ensemble import ExtraTreesClassifier as _ET
from imblearn.ensemble import BalancedRandomForestClassifier as _BRF, EasyEnsembleClassifier as _EEC

# Optional: tree boosters (only used if installed)
_HAVE_XGB = _HAVE_LGBM = _HAVE_CAT = False
try:
    from xgboost import XGBClassifier as _XGB
    _HAVE_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier as _LGBM
    _HAVE_LGBM = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier as _CAT
    _HAVE_CAT = True
except Exception:
    pass

# Silence noisy warnings in tiny samples / bootstrap resamples
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")

# Enforce Bayesian optimization only — hard fail if scikit-optimize is missing
REQUIRE_BAYES = True
_USE_BAYES = True
try:
    from skopt import BayesSearchCV as _BayesSearchCV
    from skopt.space import Real as _Real, Integer as _Integer, Categorical as _Categorical
except Exception as e:
    raise ImportError(
        "Bayesian optimization is required (REQUIRE_BAYES=True) but scikit-optimize ('skopt') "
        "is not available. Install it: pip install scikit-optimize"
    ) from e


def _scores_from_model(model, X):
    """Return 1D scores; prefer predict_proba[:,1], else decision_function; else None."""
    s = None
    if hasattr(model, "predict_proba"):
        try:
            s = model.predict_proba(X)[:, 1]
        except Exception:
            s = None
    if s is None and hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X)
        except Exception:
            s = None
    return None if s is None else np.asarray(s).ravel()


def _metrics_report(y_true, y_pred, y_scores):
    proba = None
    # If scores look like probabilities in [0,1], we'll compute brier
    if y_scores is not None:
        ys = np.asarray(y_scores).ravel()
        proba = ys if (np.isfinite(ys).all() and ys.min() >= 0 and ys.max() <= 1) else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_sensitivity_TPR": sensitivity,
        "specificity_TNR": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if (y_scores is not None) and (np.unique(y_true).size == 2):
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_scores)
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc_avg_precision"] = average_precision_score(y_true, y_scores)
        except Exception:
            out["pr_auc_avg_precision"] = None
        out["brier_score"] = brier_score_loss(y_true, proba, pos_label=1) if proba is not None else None
    else:
        out["roc_auc"] = None
        out["pr_auc_avg_precision"] = None
        out["brier_score"] = None
    return out


# ---- CV + sampling config (use separate CVs for tuning vs threshold)
_CV_N_SPLITS = 5
_CV_N_REPEATS = 5  # 5x5 = 25 validations for more stable CV mean

# CV used inside the hyperparameter search
_cv_tune = RepeatedStratifiedKFold(
    n_splits=_CV_N_SPLITS,
    n_repeats=_CV_N_REPEATS,
    random_state=RANDOM_STATE
)

# Independent CV (different seed) used only for decision-threshold selection
_cv_thresh = RepeatedStratifiedKFold(
    n_splits=_CV_N_SPLITS,
    n_repeats=_CV_N_REPEATS,
    random_state=RANDOM_STATE + 137  # any different seed is fine
)

_PRIMARY_SCORING = "balanced_accuracy"
_N_ITER_BAYES = 30 if _USE_BAYES else 200
_N_JOBS = -1

# Oversample to reach minority:majority = 1:2 inside training folds only
_ROS_RATIO = 0.5
_ros = _ROS(sampling_strategy=_ROS_RATIO, random_state=RANDOM_STATE)


def _count_and_print_class_balance(y_train, dataset_name):
    cls, cnts = np.unique(y_train, return_counts=True)
    d = dict(zip(cls, cnts))
    n0, n1 = int(d.get(0, 0)), int(d.get(1, 0))
    if VERBOSE:
        print(f"[{dataset_name}] TRAIN class counts before oversample ⇒ 0: {n0}, 1: {n1} "
              f"(minority->majority target ratio = 1:2)")


# ---------------------------
# Threshold selection (OOF)
# ---------------------------
def _cv_pick_threshold(estimator, X, y, cv, grid_size=201, strategy="ba"):
    """
    Choose a single decision threshold using OOF scores from a repeated stratified CV.

    Strategy
    --------
    - Train on each CV train fold and score the corresponding val fold.
    - Concatenate OOF scores once; scan a threshold grid to maximize the chosen metric
      ('ba' balanced accuracy, 'youden', or 'f1').

    Returns
    -------
    (best_threshold, [threshold], [metric_at_best_threshold])
    """
    # ensure pandas df/series indexable by iloc
    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    y = np.asarray(y)

    oof = np.full(shape=y.shape[0], fill_value=np.nan, dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[tr_idx], y[tr_idx])
        s = _scores_from_model(est, X.iloc[va_idx])
        if s is None:
            s = est.predict(X.iloc[va_idx]).astype(float)
        oof[va_idx] = s

    m = np.isfinite(oof)
    sx, yx = oof[m], y[m]
    smin, smax = float(np.nanmin(sx)), float(np.nanmax(sx))
    thr_grid = np.linspace(0, 1, grid_size) if (smin >= 0 and smax <= 1) else np.quantile(sx,
                                                                                          np.linspace(0, 1, grid_size))

    best_metric, best_thr = -np.inf, 0.5
    for t in thr_grid:
        yp = (sx >= t).astype(int)
        if strategy == "ba":
            metric = balanced_accuracy_score(yx, yp)
        elif strategy == "youden":
            tn, fp, fn, tp = confusion_matrix(yx, yp, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            metric = tpr + tnr - 1.0
        else:  # "f1"
            metric = f1_score(yx, yp, zero_division=0)
        if metric > best_metric:
            best_metric, best_thr = metric, float(t)

    return float(best_thr), [float(best_thr)], [float(best_metric)]


# ---------------------------
# Stratified bootstrap CIs
# ---------------------------
def _bootstrap_ci_scores(y_true, scores, threshold, n_boot=2000, seed=RANDOM_STATE):
    """
    Stratified percentile bootstrap 95% CIs for test metrics (fixed threshold).

    Metrics
    -------
    - Balanced accuracy at the supplied threshold
    - ROC AUC
    - PR AUC (average precision)

    Sampling preserves class counts (stratified) within each bootstrap draw.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    ba_vals, roc_vals, pr_vals = [], [], []
    for _ in range(n_boot):
        boot_pos = pos_idx[rng.integers(0, n_pos, size=n_pos)]
        boot_neg = neg_idx[rng.integers(0, n_neg, size=n_neg)]
        idx = np.concatenate([boot_pos, boot_neg])

        yt = y_true[idx]
        st = scores[idx]
        yp = (st >= threshold).astype(int)

        ba_vals.append(balanced_accuracy_score(yt, yp))
        roc_vals.append(roc_auc_score(yt, st))
        pr_vals.append(average_precision_score(yt, st))

    def _ci(arr):
        arr = np.asarray(arr, float)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"ci95_low": float(lo), "ci95_high": float(hi), "n_boot_valid": int(arr.size)}

    return {
        "balanced_accuracy": _ci(ba_vals),
        "roc_auc": _ci(roc_vals),
        "pr_auc_avg_precision": _ci(pr_vals),
    }


# ---------------------------
# Model spaces (you can widen/narrow bounds here)
# ---------------------------
def _model_spaces(dataset_name: str):
    """
    Define model families, search spaces, and whether to use ROS inside CV.

    Notes
    -----
    - For imbalance-aware models, ROS is disabled to avoid double-compensation.
    - Some spaces differ by dataset (STRUCTURAL/GRAPH/COMBINED) to control capacity.
    - If scikit-optimize is unavailable, falls back to RandomizedSearchCV with comparable ranges.

    Returns
    -------
    dict
        key -> (estimator, search_space, use_ros)
    """
    ds = dataset_name.upper()
    is_combined = (ds == "COMBINED")
    is_struct   = (ds == "STRUCTURAL")
    is_graph    = (ds == "GRAPH")

    # Decide when to use oversampling (ROS) vs class_weight. We typically avoid using both.
    def _use_ros_default(model_name: str) -> bool:
        # KNN never oversamples. For GRAPH we prefer class_weight (or built-in balancing) for margin models.
        if model_name == "knn":
            return False
        if is_graph and model_name in {"logreg", "enet", "svc", "linsvm", "rf", "extratrees", "hgb"}:
            return False
        return True

    # Apply class_weight='balanced' only on GRAPH (margin models); else None.
    def _cw_for_graph():
        return ("balanced" if is_graph else None)

    if _USE_BAYES:
        # ======================
        # Bayesian optimization
        # ======================
        # --- HistGradientBoosting (capacity regularized on STRUCT/GRAPH; early stopping on COMBINED)
        if is_combined:
            hgb_est = _HGB(
                random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, max_iter=1000
            )
            hgb_space = {
                "clf__learning_rate": _Real(1e-2, 1e-1, prior="log-uniform"),
                "clf__max_depth": _Integer(2, 6),
                "clf__max_leaf_nodes": _Integer(7, 31),
                "clf__l2_regularization": _Real(1e-3, 1.0, prior="log-uniform"),
                "clf__min_samples_leaf": _Integer(2, 6),
            }
        elif is_struct or is_graph:
            hgb_est = _HGB(random_state=RANDOM_STATE)
            hgb_space = {
                "clf__learning_rate": _Real(1e-2, 1e-1, prior="log-uniform"),
                "clf__max_depth": _Integer(2, 4),
                "clf__max_leaf_nodes": _Integer(7, 31),
                "clf__l2_regularization": _Real(1e-3, 3e+1, prior="log-uniform"),
                "clf__min_samples_leaf": _Integer(3, 8),
            }
        else:
            hgb_est = _HGB(random_state=RANDOM_STATE)
            hgb_space = {
                "clf__learning_rate": _Real(1e-3, 3e-1, prior="log-uniform"),
                "clf__max_depth": _Integer(2, 12),
                "clf__max_leaf_nodes": _Integer(7, 63),
                "clf__l2_regularization": _Real(1e-4, 10.0, prior="log-uniform"),
            }

        # --- RandomForest / ExtraTrees (no max_features=None; shallower on STRUCT)
        if is_combined or is_graph:
            rf_space = {
                "clf__n_estimators": _Integer(150, 400),
                "clf__max_depth": _Integer(3, 10),
                "clf__min_samples_split": _Integer(2, 20),
                "clf__min_samples_leaf": _Integer(2, 6),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }
            et_space = {
                "clf__n_estimators": _Integer(150, 400),
                "clf__max_depth": _Integer(3, 10),
                "clf__min_samples_leaf": _Integer(2, 6),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }
        elif is_struct:
            rf_space = {
                "clf__n_estimators": _Integer(150, 400),
                "clf__max_depth": _Integer(2, 6),
                "clf__min_samples_split": _Integer(2, 20),
                "clf__min_samples_leaf": _Integer(2, 6),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }
            et_space = {
                "clf__n_estimators": _Integer(150, 400),
                "clf__max_depth": _Integer(2, 6),
                "clf__min_samples_leaf": _Integer(2, 6),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }
        else:
            rf_space = {
                "clf__n_estimators": _Integer(100, 800),
                "clf__max_depth": _Integer(2, 20),
                "clf__min_samples_split": _Integer(2, 20),
                "clf__min_samples_leaf": _Integer(1, 10),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }
            et_space = {
                "clf__n_estimators": _Integer(100, 800),
                "clf__max_depth": _Integer(2, 20),
                "clf__min_samples_leaf": _Integer(1, 10),
                "clf__max_features": _Categorical(["sqrt", "log2"]),
            }

        # --- Linear / Elastic-net / RBF-SVC
        if is_combined:
            logreg_space = {"clf__C": _Real(1e-3, 1e-1, prior="log-uniform"),
                            "clf__penalty": _Categorical(["l1", "l2"])}
            enet_space  = {"clf__C": _Real(1e-3, 1e-1, prior="log-uniform"),
                           "clf__l1_ratio": _Real(0.0, 0.4)}
            svc_space   = {"clf__C": _Real(1e-2, 1e+1, prior="log-uniform"),
                           "clf__gamma": _Real(1e-4, 1e-2, prior="log-uniform"),
                           "clf__kernel": _Categorical(["rbf"])}
        elif is_struct:
            logreg_space = {"clf__C": _Real(1e-2, 1e+1, prior="log-uniform"),
                            "clf__penalty": _Categorical(["l1", "l2"])}
            enet_space  = {"clf__C": _Real(1e-1, 1e+1, prior="log-uniform"),
                           "clf__l1_ratio": _Real(0.2, 0.8)}
            svc_space   = {"clf__C": _Real(1e-1, 1e+1, prior="log-uniform"),
                           "clf__gamma": _Real(1e-3, 1e-1, prior="log-uniform"),
                           "clf__kernel": _Categorical(["rbf"])}
        else:
            # GRAPH – keep SVC near-linear and rely on class_weight instead of ROS
            logreg_space = {"clf__C": _Real(1e-4, 1e-1, prior="log-uniform"),
                            "clf__penalty": _Categorical(["l1", "l2"])}
            enet_space  = {"clf__C": _Real(1e-4, 1e-1, prior="log-uniform"),
                           "clf__l1_ratio": _Real(0.0, 0.6)}
            svc_space   = {"clf__C": _Real(1e-2, 1e+1, prior="log-uniform"),
                           "clf__gamma": _Real(1e-4, 1e-3, prior="log-uniform"),
                           "clf__kernel": _Categorical(["rbf"])}

        spaces = {
            # Linear-ish
            "logreg": (_LR(max_iter=5000, solver="liblinear", class_weight=_cw_for_graph()),
                       logreg_space, _use_ros_default("logreg")),
            "enet":   (_LR(solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=10000,
                           class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                       enet_space,  _use_ros_default("enet")),
            "svc":    (_SVC(probability=True, class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                       svc_space,   _use_ros_default("svc")),
            "linsvm": (_Cal(_LinearSVC(C=1.0, class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                            cv=5, method="sigmoid"),
                       {"clf__estimator__C": _Real(1e-3, 1e+2, prior="log-uniform")},
                       _use_ros_default("linsvm")),

            # Trees / ensembles
            "rf":         (_RF(random_state=RANDOM_STATE, n_jobs=-1, class_weight=_cw_for_graph()),
                           rf_space, _use_ros_default("rf")),
            "extratrees": (_ET(random_state=RANDOM_STATE, n_jobs=-1),
                           et_space, _use_ros_default("extratrees")),
            "hgb":        (hgb_est, hgb_space, _use_ros_default("hgb")),

            # Small-n friendly
            "lda": (_LDA(solver="lsqr", shrinkage=None),
                    {"clf__shrinkage": _Categorical(["auto"])},
                    _use_ros_default("lda")),
            "gnb": (_GNB(),
                    {"clf__var_smoothing": _Real(1e-12, 1e-7, prior="log-uniform")},
                    _use_ros_default("gnb")),
            "knn": (_KNN(),
                    {"clf__n_neighbors": _Integer(11, 35),
                     "clf__weights": _Categorical(["uniform"]),
                     "clf__p": _Categorical([2])},
                    False),  # never ROS for KNN

            # Imbalance-aware (already handle class imbalance; we keep ROS off)
            "brf": (_BRF(random_state=RANDOM_STATE, n_jobs=-1),
                    ({"clf__n_estimators": _Integer(150, 400),
                      "clf__max_depth": _Integer(3, 10),
                      "clf__min_samples_split": _Integer(2, 20),
                      "clf__min_samples_leaf": _Integer(2, 6),
                      "clf__max_features": _Categorical(["sqrt", "log2"])}
                     if (is_combined or is_graph) else
                     {"clf__n_estimators": _Integer(100, 800),
                      "clf__max_depth": _Integer(2, 20),
                      "clf__min_samples_split": _Integer(2, 20),
                      "clf__min_samples_leaf": _Integer(1, 10),
                      "clf__max_features": _Categorical(["sqrt", "log2"])}),
                    False),
            "easy_ensemble": (_EEC(random_state=RANDOM_STATE, n_estimators=10),
                              {"clf__n_estimators": _Integer(5, 40)},
                              False),
        }

        # Optional boosters (kept without ROS)
        if _HAVE_XGB:
            spaces["xgb"] = (
                _XGB(random_state=RANDOM_STATE, tree_method="hist",
                     eval_metric="logloss", n_estimators=200, verbosity=0),
                {"clf__learning_rate": _Real(1e-3, 3e-1, prior="log-uniform"),
                 "clf__max_depth": _Integer(2, 6),
                 "clf__min_child_weight": _Real(1.0, 8.0, prior="log-uniform"),
                 "clf__subsample": _Real(0.6, 1.0),
                 "clf__colsample_bytree": _Real(0.6, 1.0),
                 "clf__reg_alpha": _Real(1e-8, 1e-1, prior="log-uniform"),
                 "clf__reg_lambda": _Real(1e-3, 10.0, prior="log-uniform"),
                 "clf__gamma": _Real(0.0, 5.0),
                 "clf__scale_pos_weight": _Real(1.0, 4.0)},
                False
            )
        if _HAVE_LGBM:
            spaces["lgbm"] = (
                _LGBM(random_state=RANDOM_STATE, n_estimators=200),
                {"clf__learning_rate": _Real(1e-3, 3e-1, prior="log-uniform"),
                 "clf__max_depth": _Integer(-1, 8),
                 "clf__num_leaves": _Integer(7, 63),
                 "clf__min_child_samples": _Integer(5, 50),
                 "clf__subsample": _Real(0.6, 1.0),
                 "clf__colsample_bytree": _Real(0.6, 1.0),
                 "clf__reg_alpha": _Real(1e-8, 1e-1, prior="log-uniform"),
                 "clf__reg_lambda": _Real(1e-3, 10.0, prior="log-uniform"),
                 "clf__scale_pos_weight": _Real(1.0, 4.0)},
                False
            )
        if _HAVE_CAT:
            spaces["cat"] = (
                _CAT(random_seed=RANDOM_STATE, verbose=0, loss_function="Logloss",
                     depth=6, iterations=300, learning_rate=0.05),
                {"clf__depth": _Integer(3, 8),
                 "clf__learning_rate": _Real(1e-3, 3e-1, prior="log-uniform"),
                 "clf__l2_leaf_reg": _Real(1.0, 10.0, prior="log-uniform"),
                 "clf__border_count": _Integer(32, 255),
                 "clf__scale_pos_weight": _Real(1.0, 4.0)},
                False
            )
        return spaces

    # ==============================
    # RandomizedSearchCV fallback
    # ==============================
    from scipy.stats import loguniform as _loguniform, randint as _randint, uniform as _uniform

    if is_combined:
        hgb_est = _HGB(random_state=RANDOM_STATE, early_stopping=True,
                       validation_fraction=0.2, n_iter_no_change=20, max_iter=1000)
        hgb_space = {
            "clf__learning_rate": _loguniform(1e-2, 1e-1),
            "clf__max_depth": _randint(2, 7),
            "clf__max_leaf_nodes": _randint(7, 32),
            "clf__l2_regularization": _loguniform(1e-3, 1.0),
            "clf__min_samples_leaf": _randint(2, 7),
        }
    elif is_struct or is_graph:
        hgb_est = _HGB(random_state=RANDOM_STATE)
        hgb_space = {
            "clf__learning_rate": _loguniform(1e-2, 1e-1),
            "clf__max_depth": _randint(2, 5),
            "clf__max_leaf_nodes": _randint(7, 32),
            "clf__l2_regularization": _loguniform(1e-3, 3e+1),
            "clf__min_samples_leaf": _randint(3, 9),
        }
    else:
        hgb_est = _HGB(random_state=RANDOM_STATE)
        hgb_space = {
            "clf__learning_rate": _loguniform(1e-3, 3e-1),
            "clf__max_depth": _randint(2, 13),
            "clf__max_leaf_nodes": _randint(7, 64),
            "clf__l2_regularization": _loguniform(1e-4, 10.0),
        }

    if is_combined or is_graph:
        rf_space = {
            "clf__n_estimators": _randint(150, 401),
            "clf__max_depth": _randint(3, 11),
            "clf__min_samples_split": _randint(2, 21),
            "clf__min_samples_leaf": _randint(2, 7),
            "clf__max_features": ["sqrt", "log2"],
        }
        et_space = {
            "clf__n_estimators": _randint(150, 401),
            "clf__max_depth": _randint(3, 11),
            "clf__min_samples_leaf": _randint(2, 7),
            "clf__max_features": ["sqrt", "log2"],
        }
    elif is_struct:
        rf_space = {
            "clf__n_estimators": _randint(150, 401),
            "clf__max_depth": _randint(2, 7),
            "clf__min_samples_split": _randint(2, 21),
            "clf__min_samples_leaf": _randint(2, 7),
            "clf__max_features": ["sqrt", "log2"],
        }
        et_space = {
            "clf__n_estimators": _randint(150, 401),
            "clf__max_depth": _randint(2, 7),
            "clf__min_samples_leaf": _randint(2, 7),
            "clf__max_features": ["sqrt", "log2"],
        }
    else:
        rf_space = {
            "clf__n_estimators": _randint(100, 801),
            "clf__max_depth": _randint(2, 21),
            "clf__min_samples_split": _randint(2, 21),
            "clf__min_samples_leaf": _randint(1, 11),
            "clf__max_features": ["sqrt", "log2"],
        }
        et_space = {
            "clf__n_estimators": _randint(100, 801),
            "clf__max_depth": _randint(2, 21),
            "clf__min_samples_leaf": _randint(1, 11),
            "clf__max_features": ["sqrt", "log2"],
        }

    if is_combined:
        logreg_space = {"clf__C": _loguniform(1e-3, 1e-1), "clf__penalty": ["l1", "l2"]}
        enet_space  = {"clf__C": _loguniform(1e-3, 1e-1), "clf__l1_ratio": _uniform(0.0, 0.4)}
        svc_space   = {"clf__C": _loguniform(1e-2, 1e+1), "clf__gamma": _loguniform(1e-4, 1e-2), "clf__kernel": ["rbf"]}
    elif is_struct:
        logreg_space = {"clf__C": _loguniform(1e-2, 1e+1), "clf__penalty": ["l1", "l2"]}
        enet_space  = {"clf__C": _loguniform(1e-1, 1e+1), "clf__l1_ratio": _uniform(0.2, 0.6)}  # 0.2..0.8
        svc_space   = {"clf__C": _loguniform(1e-1, 1e+1), "clf__gamma": _loguniform(1e-3, 1e-1), "clf__kernel": ["rbf"]}
    else:
        # GRAPH
        logreg_space = {"clf__C": _loguniform(1e-4, 1e-1), "clf__penalty": ["l1", "l2"]}
        enet_space  = {"clf__C": _loguniform(1e-4, 1e-1), "clf__l1_ratio": _uniform(0.0, 0.6)}
        svc_space   = {"clf__C": _loguniform(1e-2, 1e+1), "clf__gamma": _loguniform(1e-4, 1e-3), "clf__kernel": ["rbf"]}

    spaces = {
        "logreg": (_LR(max_iter=5000, solver="liblinear", class_weight=_cw_for_graph()),
                   logreg_space, _use_ros_default("logreg")),
        "enet":   (_LR(solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=10000,
                       class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                   enet_space,  _use_ros_default("enet")),
        "svc":    (_SVC(probability=True, class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                   svc_space,   _use_ros_default("svc")),
        "linsvm": (_Cal(_LinearSVC(C=1.0, class_weight=_cw_for_graph(), random_state=RANDOM_STATE),
                        cv=5, method="sigmoid"),
                   {"clf__base_estimator__C": _loguniform(1e-3, 1e+2)},
                   _use_ros_default("linsvm")),

        "rf":         (_RF(random_state=RANDOM_STATE, n_jobs=-1, class_weight=_cw_for_graph()),
                       rf_space, _use_ros_default("rf")),
        "extratrees": (_ET(random_state=RANDOM_STATE, n_jobs=-1),
                       et_space, _use_ros_default("extratrees")),
        "hgb":        (hgb_est, hgb_space, _use_ros_default("hgb")),

        "lda": (_LDA(solver="lsqr", shrinkage=None),
                {"clf__shrinkage": ["auto"]},
                _use_ros_default("lda")),
        "gnb": (_GNB(),
                {"clf__var_smoothing": _loguniform(1e-12, 1e-7)},
                _use_ros_default("gnb")),
        "knn": (_KNN(),
                {"clf__n_neighbors": _randint(11, 36),
                 "clf__weights": ["uniform"],
                 "clf__p": [2]},
                False),
        "brf": (_BRF(random_state=RANDOM_STATE, n_jobs=-1),
                ({"clf__n_estimators": _randint(150, 401),
                  "clf__max_depth": _randint(3, 11),
                  "clf__min_samples_split": _randint(2, 21),
                  "clf__min_samples_leaf": _randint(2, 7),
                  "clf__max_features": ["sqrt", "log2"]}
                 if (is_combined or is_graph) else
                 {"clf__n_estimators": _randint(100, 801),
                  "clf__max_depth": _randint(2, 21),
                  "clf__min_samples_split": _randint(2, 21),
                  "clf__min_samples_leaf": _randint(1, 11),
                  "clf__max_features": ["sqrt", "log2"]}),
                False),
        "easy_ensemble": (_EEC(random_state=RANDOM_STATE, n_estimators=10),
                          {"clf__n_estimators": _randint(5, 41)},
                          False),
    }

    if _HAVE_XGB:
        spaces["xgb"] = (
            _XGB(random_state=RANDOM_STATE, tree_method="hist", eval_metric="logloss",
                 n_estimators=200, verbosity=0),
            {"clf__learning_rate": _loguniform(1e-3, 3e-1),
             "clf__max_depth": _randint(2, 7),
             "clf__min_child_weight": _loguniform(1.0, 8.0),
             "clf__subsample": _uniform(0.6, 0.4),
             "clf__colsample_bytree": _uniform(0.6, 0.4),
             "clf__reg_alpha": _loguniform(1e-8, 1e-1),
             "clf__reg_lambda": _loguniform(1e-3, 10.0),
             "clf__gamma": _uniform(0.0, 5.0),
             "clf__scale_pos_weight": _uniform(1.0, 3.0)},
            False
        )
    if _HAVE_LGBM:
        spaces["lgbm"] = (
            _LGBM(random_state=RANDOM_STATE, n_estimators=200),
            {"clf__learning_rate": _loguniform(1e-3, 3e-1),
             "clf__max_depth": [-1, 2, 3, 4, 5, 6, 7, 8],
             "clf__num_leaves": _randint(7, 64),
             "clf__min_child_samples": _randint(5, 51),
             "clf__subsample": _uniform(0.6, 0.4),
             "clf__colsample_bytree": _uniform(0.6, 0.4),
             "clf__reg_alpha": _loguniform(1e-8, 1e-1),
             "clf__reg_lambda": _loguniform(1e-3, 10.0),
             "clf__scale_pos_weight": _uniform(1.0, 3.0)},
            False
        )
    if _HAVE_CAT:
        spaces["cat"] = (
            _CAT(random_seed=RANDOM_STATE, verbose=0, loss_function="Logloss",
                 depth=6, iterations=300, learning_rate=0.05),
            {"clf__depth": _randint(3, 9),
             "clf__learning_rate": _loguniform(1e-3, 3e-1),
             "clf__l2_leaf_reg": _loguniform(1.0, 10.0),
             "clf__border_count": _randint(32, 256),
             "clf__scale_pos_weight": _uniform(1.0, 3.0)},
            False
        )
    return spaces




def _train_eval_dataset(dataset_name: str, train_path: _Path, test_path: _Path):
    # Doc: Train/tune per-model pipelines on a dataset, tune a global decision threshold, evaluate on TEST, and persist reports.
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=PROTECTED_COLUMNS).copy()
    y_train = train_df["Label"].astype(int).values
    X_test = test_df.drop(columns=PROTECTED_COLUMNS).copy()
    y_test = test_df["Label"].astype(int).values

    _count_and_print_class_balance(y_train, dataset_name)

    spaces = _model_spaces(dataset_name)
    results = []
    best_overall = None

    model_dir = BASE_DIR / f"models_{dataset_name.lower()}"
    model_dir.mkdir(exist_ok=True)

    # old: for key, (estimator, search_space) in spaces.items():
    for key, spec in spaces.items():
        if len(spec) == 3:
            estimator, search_space, use_ros = spec
        else:
            estimator, search_space = spec
            use_ros = True  # default: keep your ROS step


        steps = [("clf", estimator)] if not use_ros else [("ros", _ros), ("clf", estimator)]
        pipe = _ImbPipeline(steps=steps)

        if _USE_BAYES:
            search = _BayesSearchCV(
                estimator=pipe,
                search_spaces=search_space,
                n_iter=_N_ITER_BAYES,
                cv=_cv_tune,
                scoring=_PRIMARY_SCORING,
                n_jobs=_N_JOBS,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=0
            )
        else:
            search = _BayesSearchCV(
                estimator=pipe,
                param_distributions=search_space,
                n_iter=_N_ITER_BAYES,
                cv=_cv_tune,
                scoring=_PRIMARY_SCORING,
                n_jobs=_N_JOBS,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=0
            )

        if VERBOSE:
            print(f"\n[{dataset_name}] === Optimizing {key} ({'Bayesian' if _USE_BAYES else 'Randomized'}) ===")

        # Hyperparameter search with oversampling inside CV folds only
        search.fit(X_train, y_train)

        # CV summary: mean from best_score_, std across split scores for best index
        cv_mean = float(search.best_score_)
        best_idx = int(search.best_index_)
        split_cols = [c for c in search.cv_results_.keys() if c.startswith("split") and c.endswith("_test_score")]
        split_scores = [float(search.cv_results_[c][best_idx]) for c in split_cols]
        cv_std = float(np.std(split_scores)) if len(split_scores) > 1 else 0.0
        cv_total_folds = int(len(split_scores))  # e.g., 25

        if VERBOSE:
            print(f"[{dataset_name}] {key} best CV balanced_accuracy = {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"[{dataset_name}] {key} best params: {search.best_params_}")

        best_model = search.best_estimator_

        # ===== CV-based decision threshold (OOF on training data) =====
        thr, thr_list, metric_list = _cv_pick_threshold(best_model, X_train, y_train, _cv_thresh, grid_size=201, strategy="ba")
        if VERBOSE:
            print(f"[{dataset_name}] {key} CV-tuned decision threshold = {thr:.4f} (OOF optimal)")

        # TRAIN metrics (using tuned threshold)
        y_score_tr = _scores_from_model(best_model, X_train)
        y_pred_tr = (y_score_tr >= thr).astype(int) if y_score_tr is not None else best_model.predict(X_train)
        metrics_train = _metrics_report(y_train, y_pred_tr, y_score_tr)

        # TEST metrics (using tuned threshold)
        y_score_te = _scores_from_model(best_model, X_test)
        y_pred_te = (y_score_te >= thr).astype(int) if y_score_te is not None else best_model.predict(X_test)
        metrics_test = _metrics_report(y_test, y_pred_te, y_score_te)

        # 95% bootstrap CIs (stratified) on TEST
        test_cis = None
        if (y_score_te is not None) and (np.unique(y_test).size == 2):
            test_cis = _bootstrap_ci_scores(y_test, y_score_te, thr, n_boot=2000, seed=RANDOM_STATE)

        metrics_cv = {
            "cv_balanced_accuracy_mean": cv_mean,
            "cv_balanced_accuracy_std":  cv_std,
            "cv_n_splits":               _CV_N_SPLITS,
            "cv_n_repeats":              _CV_N_REPEATS,
            "cv_total_validations":      _CV_N_SPLITS * _CV_N_REPEATS,
            "primary_metric":            _PRIMARY_SCORING,
        }

        # ROC plot (test)
        out_png = None
        if metrics_test.get("roc_auc") is not None and y_score_te is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score_te)
            _plt.figure(figsize=(5, 4))
            _plt.plot(fpr, tpr, label=f"{key} (AUC={metrics_test['roc_auc']:.3f})")
            _plt.plot([0, 1], [0, 1], linestyle="--")
            _plt.xlabel("FPR");
            _plt.ylabel("TPR");
            _plt.title(f"{dataset_name} ROC - {key}")
            _plt.legend(loc="lower right")
            out_png = model_dir / f"roc_{key}.png"
            _plt.tight_layout();
            _plt.savefig(out_png);
            _plt.close()

        # save model
        model_path = model_dir / f"best_{key}.joblib"
        joblib.dump(best_model, model_path)

        row = {
            "dataset": dataset_name,
            "model_key": key,
            "best_params": search.best_params_,
            "decision_threshold": float(thr),
            "cv_summary": metrics_cv,
            "train_metrics": metrics_train,
            "test_metrics": metrics_test,
            "test_ci95": test_cis,
            "n_features_used": int(X_train.shape[1]),
            "model_path": str(model_path),
            "roc_plot": str(out_png) if out_png else None
        }
        results.append(row)

        if (best_overall is None) or (cv_mean > best_overall["cv_summary"]["cv_balanced_accuracy_mean"]):
            best_overall = row

    # save per-dataset report (includes train+test metrics for each algorithm's best)
    report_path = BASE_DIR / f"git_{dataset_name.lower()}_ml_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "primary_metric": _PRIMARY_SCORING,
            "cv_n_splits": _CV_N_SPLITS,
            "cv_n_repeats": _CV_N_REPEATS,
            "cv_total_folds": _CV_N_SPLITS * _CV_N_REPEATS,
            "oversampling_ratio_minority_to_majority": "1:2 (sampling_strategy=0.5)",
            "results": results,
            "best_by_cv_balanced_accuracy": best_overall
        }, f, indent=2)
    print(f"[{dataset_name}] ML report saved -> {report_path.name}")
    return best_overall


def _print_best_summary(best):
    print(f"\n=== BEST MODEL FOR {best['dataset']} (by CV balanced accuracy) ===")
    print("Model:", best["model_key"])
    print("Best params:", json.dumps(best["best_params"], indent=2))
    print(f"Decision threshold (CV-tuned): {best.get('decision_threshold')}")
    print("CV (train):", json.dumps(best["cv_summary"], indent=2))
    print("TRAIN metrics:", json.dumps(best["train_metrics"], indent=2))
    print("TEST  metrics:", json.dumps(best["test_metrics"], indent=2))
    if best.get("test_ci95") is not None:
        print("TEST  95% CIs:", json.dumps(best["test_ci95"], indent=2))
    print("Model path:", best["model_path"])
    if best.get("roc_plot"):
        print("ROC PNG:", best["roc_plot"])


# ---- Excel summary (single canonical implementation: train+test + gaps) ----
def _save_best_models_xlsx(best_models, out_path):
    rows = []
    for b in best_models:
        trm = b.get("train_metrics", {}) or {}
        tem = b.get("test_metrics", {}) or {}
        cv  = b.get("cv_summary", {}) or {}
        cm  = tem.get("confusion_matrix", {}) or {}

        train_bal = trm.get("balanced_accuracy", np.nan)
        test_bal  = tem.get("balanced_accuracy", np.nan)
        train_auc = trm.get("roc_auc", np.nan)
        test_auc  = tem.get("roc_auc", np.nan)
        train_ap  = trm.get("pr_auc_avg_precision", np.nan)
        test_ap   = tem.get("pr_auc_avg_precision", np.nan)

        rows.append({
            "Dataset": b["dataset"],
            "Model": b["model_key"],
            "Decision threshold": b.get("decision_threshold"),
            "CV bal.acc (mean±SD)": f"{cv.get('cv_balanced_accuracy_mean', np.nan):.3f}±{cv.get('cv_balanced_accuracy_std', 0.0):.3f}",
            "CV folds": cv.get("cv_n_splits"),
            # TRAIN
            "Train bal.acc": train_bal,
            "Train ROC-AUC": train_auc,
            "Train PR-AUC": train_ap,
            "Train Brier": trm.get("brier_score", np.nan),
            # TEST
            "Test bal.acc": test_bal,
            "Test ROC-AUC": test_auc,
            "Test PR-AUC": test_ap,
            "Test Brier": tem.get("brier_score", np.nan),
            # GAPS (train - test)
            "Gap bal.acc": (np.nan if (pd.isna(train_bal) or pd.isna(test_bal)) else (train_bal - test_bal)),
            "Gap ROC-AUC": (np.nan if (pd.isna(train_auc) or pd.isna(test_auc)) else (train_auc - test_auc)),
            "Gap PR-AUC":  (np.nan if (pd.isna(train_ap)  or pd.isna(test_ap))  else (train_ap  - test_ap)),
            # Confusion matrix + sizes
            "TN": cm.get("tn"), "FP": cm.get("fp"), "FN": cm.get("fn"), "TP": cm.get("tp"),
            "N_train": trm.get("n"), "N_test": tem.get("n"),
            # Misc
            "Features used": b.get("n_features_used"),
            "Params": json.dumps(b.get("best_params", {})),
            "Model path": b.get("model_path"),
            "ROC PNG": b.get("roc_plot"),
        })

    df = pd.DataFrame(rows)

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_excel(path, engine):
        with pd.ExcelWriter(path, engine=engine) as writer:
            sheet = "BestModels"
            df.to_excel(writer, sheet_name=sheet, index=False)

            if engine == "xlsxwriter":
                wb = writer.book
                ws = writer.sheets[sheet]
                nrows, ncols = df.shape

                # Turn range into an Excel table
                ws.add_table(0, 0, nrows, ncols - 1, {
                    "style": "Table Style Light 9",
                    "columns": [{"header": col} for col in df.columns]
                })

                num_fmt = wb.add_format({"num_format": "0.000"})
                int_fmt = wb.add_format({"num_format": "0"})
                wrap    = wb.add_format({"text_wrap": True})

                col_idx = {c: i for i, c in enumerate(df.columns)}
                # widths
                for c, w in {
                    "Dataset": 12, "Model": 14, "Decision threshold": 12,
                    "CV bal.acc (mean±SD)": 18, "Params": 42,
                    "Model path": 36, "ROC PNG": 24
                }.items():
                    if c in col_idx: ws.set_column(col_idx[c], col_idx[c], w, wrap if c in ("Params", "Model path", "ROC PNG") else None)

                # numeric 3 d.p.
                for c in [
                    "Train bal.acc","Train ROC-AUC","Train PR-AUC","Train Brier",
                    "Test bal.acc","Test ROC-AUC","Test PR-AUC","Test Brier",
                    "Gap bal.acc","Gap ROC-AUC","Gap PR-AUC"
                ]:
                    if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 12, num_fmt)

                # integers
                for c in ["CV folds","TN","FP","FN","TP","N_train","N_test","Features used"]:
                    if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 10, int_fmt)

    try:
        _write_excel(out_path, "xlsxwriter")
        print(f"Best models Excel saved → {out_path.name}")
    except PermissionError:
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        alt = out_path.with_name(out_path.stem + f"_{ts}.xlsx")
        try:
            _write_excel(alt, "xlsxwriter")
            print(f"Best models Excel was locked; wrote to → {alt.name}")
        except Exception:
            _write_excel(alt, "openpyxl")
            print(f"Best models Excel was locked; wrote (openpyxl) to → {alt.name}")
    except Exception:
        _write_excel(out_path, "openpyxl")
        print(f"Best models Excel saved (openpyxl) → {out_path.name}")



# ---- run training on the curated (final) FS outputs — NO re-splitting
best_struct = _train_eval_dataset("STRUCTURAL", STRUCT_TRAIN_RFE10_CUR, STRUCT_TEST_RFE10_CUR)
best_graph = _train_eval_dataset("GRAPH", GRAPH_TRAIN_RFE10_CUR, GRAPH_TEST_RFE10_CUR)
best_comb = _train_eval_dataset("COMBINED", COMB_TRAIN_RFE10_CUR, COMB_TEST_RFE10_CUR)

# 11) Print best per dataset (no single overall winner)
print("\n--- BEST PER DATASET (by CV balanced accuracy) ---")
for b in [best_struct, best_graph, best_comb]:
    _print_best_summary(b)

# 12) Save the three best models to a journal-ready Excel table
_save_best_models_xlsx(
    [best_struct, best_graph, best_comb],
    BEST_MODELS_XLSX if 'BEST_MODELS_XLSX' in globals() else (BASE_DIR / "git_best_models.xlsx")
)
