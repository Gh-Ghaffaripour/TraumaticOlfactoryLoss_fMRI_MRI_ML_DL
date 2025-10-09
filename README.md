<!--
# TraumaticOlfactoryLoss_fMRI_MRI_ML_DL

**Repository for analysis of MRI and fMRI data to detect traumatic olfactory loss using Machine Learning and Deep Learning methods.**

---

## Team Information
- **Supervisor:** Dr. Abolhasan Rezaeyan
- **Lead Developer:** Ghazale Ghaffaripour
- **Team Members:**
  - Ghazale Ghaffaripour
  - Fatemeh Arezoomand
  - Samin Afzoonkar
  - Arman Khanjani


---

## Project Overview

This repository contains Python scripts designed for processing and analyzing imaging data to predict traumatic olfactory loss using machine learning (ML) and deep learning (DL) techniques. The workflow involves data merging, cleaning, outlier detection, feature selection, normalization, and machine learning modeling.

## Repository Structure and Scripts

### Data Preparation and Cleaning

1. **Data Merging** ([`Scripts/01_data_merging.py`](Scripts/01_data_merging.py)):
   - Merges structural MRI data and graph data by normalizing subject identifiers and saves combined dataset.

2. **Data Cleaning** ([`Scripts/02_data_cleaning.py`](Scripts/02_data_cleaning.py)):
   - Removes missing values, duplicates, infinite values, constant columns, and columns with significant missing data (>50%).

3. **Outlier Detection** ([`Scripts/03_outlier_detection.py`](Scripts/03_outlier_detection.py)):
   - Detects and removes subjects identified as outliers using the Interquartile Range (IQR) method.

### Normalization & Feature Selection

4. **Correlated Feature Removal** ([`Scripts/04_correlated_feature_removal.py`](Scripts/04_correlated_feature_removal.py)):
   - Identifies and removes highly correlated features (correlation threshold > 0.8).

5. **Normalization** ([`Scripts/05_normalization.py`](Scripts/05_normalization.py)):
   - Normalizes numerical features using Z-score standardization.

6. **Feature Selection** ([`Scripts/06_feature_selection.py`](Scripts/06_feature_selection.py)):
   - Selects significant features using statistical tests (T-tests), Random Forest feature importance, and Recursive Feature Elimination (RFE).

### Machine Learning Pipeline

7. **Machine Learning and Evaluation** ([`Scripts/07_machine_learning.py`](Scripts/07_machine_learning.py)):
   - Implements a complete ML pipeline:
     - Hyperparameter tuning using Bayesian optimization.
     - Cross-validation and evaluation metrics (accuracy, F1-score, precision, recall, specificity, ROC-AUC).
     - Model interpretability using SHAP, ROC curves, confusion matrices, and feature importance plots.

## Output and Analysis
All processed datasets, model outputs, and evaluation results are saved systematically in:
```
D:\image_group_data\team44\Analysis
```

## Usage
To replicate the analysis pipeline:
1. Run scripts sequentially from `01_data_merging.py` to `07_machine_learning.py`.
2. Verify intermediate results and plots generated at each step for data validation.
3. Refer to output CSVs and visualizations for model interpretation and selection.

## Required Libraries
Install the required libraries using:
```bash
pip install pandas numpy scipy sklearn statsmodels matplotlib seaborn shap imblearn xgboost lightgbm catboost scikit-optimize
```

---

## Project Overview

### Traumatic Olfactory Loss Project
This project leverages supervised Machine Learning and Deep Learning methods to classify and understand traumatic olfactory loss using neuroimaging (fMRI, MRI) data.

### Goals
- Merge, clean, and preprocess MRI and fMRI data.
- Select features predictive of traumatic olfactory loss.
- Build accurate and interpretable classification models.

## Contributions (Ongoing)
This project is a collaborative effort by **IDS Team 44**. The primary **code development** has been led by **Ghazale Ghaffaripour**, with contributions from **Samin Afzoonkar**. Other team members have been involved in **data preprocessing, analysis, and research article preparation**. 

### **Code Contributions (Confirmed So Far):**
- **Ghazale Ghaffaripour** – Lead Developer, Core ML/DL Implementation, Data Processing ,Preprocessing (CONN, FreeSurfer), Research Article Writing, Documentation 
- **Samin Afzoonkar** – Outlier Detection & Correlation Removal, Preprocessing (FreeSurfer) , Data Analysis, Research Article Writing, Documentation  

### **Additional Contributions (To Be Finalized with Article Completion):**
- **Fatemeh Arezoomand** – Preprocessing (CONN, FreeSurfer), Research Article Writing, Documentation   
- **Mahsa Bahrami** – Research Article Writing, Documentation    
- **Arman Khanjani** –  Preprocessing (CONN, FreeSurfer), Research Article Writing, Documentation  

Final contributions will be updated upon the completion of the research article.

---

© 2024 Team44.  
Code is licensed under the MIT License.  
Research papers and datasets may have different licensing—please refer to their respective terms.

-->



<h1>TraumaticOlfactoryLoss_fMRI_MRI_ML</h1>
<p><strong>Repository for analysis of MRI and fMRI data to detect traumatic olfactory loss using Machine Learning methods.</strong><br>
<em>(This repo now uses a single, end-to-end Python pipeline instead of separate step scripts.)</em></p>
<hr>

<h2>Team Information</h2>
<ul>
  <li><strong>Supervisor:</strong> Dr. Abolhasan Rezaeyan</li>
  <li><strong>Lead Developer:</strong> Ghazale Ghaffaripour</li>
  <li><strong>Team Members:</strong>
    <ul>
      <li>Ghazale Ghaffaripour</li>
      <li>Fatemeh Arezoomand</li>
      <li>Samin Afzoonkar</li>
      <li>Arman Khanjani</li>
    </ul>
  </li>
</ul>

<hr>
<h2>Project Overview</h2>
<p>This repository provides a <strong>reproducible, one-shot pipeline</strong> that:</p>
<ol>
  <li>validates and merges imaging tables,</li>
  <li>cleans features,</li>
  <li>locks a stratified train/test split,</li>
  <li>filters outliers (learned on TRAIN only),</li>
  <li>prunes highly correlated features (learned on TRAIN, mirrored to TEST),</li>
  <li>scales robustly (fit on TRAIN, applied to TEST),</li>
  <li>selects features (Welch t-test + BH-FDR → RF top-20 → RFE top-10),</li>
  <li>allows post-selection curation, and</li>
  <li>trains &amp; evaluates multiple ML models with Bayesian HPO, class-imbalance handling, CV-tuned decision thresholds, bootstrap CIs, and consolidated reporting (JSON, PNG, Excel).</li>
</ol>

<h3>Inputs (CSV)</h3>
<ul>
  <li><strong>structural_clean_subID.csv</strong></li>
  <li><strong>graph_clean_subID.csv</strong></li>
</ul>
<p>Each must include <code>Subject_ID</code> and <code>Label</code> columns. <code>Subject_ID</code> sets must match <strong>1:1</strong> between files and <strong>labels must agree</strong> across modalities (the pipeline enforces this).</p>
<p><em>Default base directory and filenames are controlled at the top of the script via <code>BASE_DIR</code> and file constants.</em></p>

<hr>
<h2>How to Run</h2>
<ol>
  <li><strong>Set paths</strong><br>
    Open the pipeline script (e.g., <code>pipeline_v1.py</code>) and adjust:
    <ul>
      <li><code>BASE_DIR</code> (default in code points to a Windows path)</li>
      <li><code>GRAPH_FILE</code>, <code>STRUCT_FILE</code> (input CSVs)</li>
      <li>Optional knobs (see <strong>Key Parameters</strong> below)</li>
    </ul>
  </li>
  <li><strong>Install dependencies</strong>
    <pre><code>pip install pandas numpy scipy scikit-learn imbalanced-learn scikit-optimize matplotlib joblib xlsxwriter openpyxl
# Optional (enables extra model families if present):
pip install xgboost lightgbm catboost
</code></pre>
    <p><strong>Note:</strong> Bayesian search via <code>scikit-optimize</code> is required; the pipeline hard-fails if <code>skopt</code> is missing.</p>
  </li>
  <li><strong>Run</strong>
    <pre><code>python pipeline_v1.py
</code></pre>
    <p>(Rename the file in the command if your script has a different name.)</p>
  </li>
</ol>

<hr>
<h2>Pipeline at a Glance (What the script does)</h2>
<ol>
  <li><strong>Combine data (strict checks)</strong>
    <ul>
      <li>Normalizes <code>Subject_ID</code>, enforces <strong>one-to-one</strong> ID match, and verifies <strong>label consistency</strong> across structural &amp; graph CSVs.</li>
      <li>Writes combined CSV.</li>
    </ul>
  </li>
  <li><strong>Feature cleaning (no leakage)</strong>
    <ul>
      <li>Drops <strong>duplicate columns</strong>, <strong>constant columns</strong>, <strong>columns with inf</strong>, and <strong>columns with any missing values</strong> (features only; <code>Subject_ID</code> &amp; <code>Label</code> are preserved).</li>
    </ul>
  </li>
  <li><strong>One-time stratified split (locked)</strong>
    <ul>
      <li>Creates a <strong>single</strong> stratified train/test split and <strong>locks</strong> it with a SHA-256 manifest. Future runs reuse the exact split.</li>
    </ul>
  </li>
  <li><strong>Outlier filtering (TRAIN-learned, row-wise rule)</strong>
    <ul>
      <li>Learns <strong>per-feature Tukey bounds</strong> on TRAIN only.</li>
      <li>Flags rows whose fraction of numeric features outside bounds ≥ <code>ROW_OUTLIER_FRACTION</code>.</li>
      <li><strong>Drops rows</strong> in both TRAIN/TEST <strong>only if</strong> the proportion flagged in TRAIN is below <code>MAX_DROP_FRACTION_TRAIN</code>.</li>
      <li>Saves <strong>feature-level bounds</strong> and <strong>row diagnostics</strong>.</li>
    </ul>
  </li>
  <li><strong>Correlation pruning (TRAIN-learned)</strong>
    <ul>
      <li>Removes TRAIN-constant numeric features, then prunes highly correlated pairs (|r| &gt; <code>CORR_THRESHOLD</code>), <strong>keeping the feature more correlated with the label</strong>.</li>
      <li>Mirrors kept/dropped features to TEST. Optional <strong>heatmaps</strong> saved before/after.</li>
    </ul>
  </li>
  <li><strong>Robust scaling (TRAIN fit → apply to TEST)</strong>
    <ul>
      <li>Fits <code>RobustScaler</code> on TRAIN numeric features (with configurable quantile range and centering/scaling), applies to TEST.</li>
      <li>Writes a <strong>scaling manifest</strong> with hyperparameters and column order.</li>
    </ul>
  </li>
  <li><strong>Feature selection (TRAIN only; applied to TEST)</strong>
    <ul>
      <li><strong>Welch’s t-test</strong> with <strong>BH-FDR</strong> control: either a fixed <code>FDR_Q</code> or <strong>adaptive</strong> selection over <code>FDR_Q_GRID</code> to target at least <code>MIN_TTEST_PASS</code> features (without forcing if signal is weak).</li>
      <li><strong>Random Forest</strong> importance → <strong>top-20</strong>.</li>
      <li><strong>RFE</strong> with logistic regression → <strong>top-10</strong>.</li>
      <li>Saves a compact <strong>feature-selection report</strong> (<code>pipeline_feature_selection_report.json</code>).</li>
    </ul>
  </li>
  <li><strong>Post-selection curation (final feature set)</strong>
    <ul>
      <li>Mode <code>"ask"</code> (interactive prompt once) or <code>"auto"</code> (drop names in a predefined <strong>blocklist</strong>).</li>
      <li>Applies <strong>the same decision</strong> to TRAIN &amp; TEST and persists <strong>curated</strong> CSVs.</li>
    </ul>
  </li>
  <li><strong>Modeling &amp; evaluation</strong>
    <ul>
      <li><strong>Model families</strong> searched via <strong>Bayesian optimization</strong> (primary metric: <code>balanced_accuracy</code>), using <strong>RepeatedStratifiedKFold (5×5)</strong> inside tuning.</li>
      <li><strong>Random oversampling</strong> to <strong>1:2</strong> (minority:majority) <strong>inside CV folds only</strong> for most models; disabled where class weighting or ensemble balance is already built in.</li>
      <li><strong>Decision threshold</strong> selected via <strong>out-of-fold</strong> CV scores to maximize <strong>balanced accuracy</strong> on TRAIN.</li>
      <li>Final <strong>TEST evaluation</strong>: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, <strong>ROC-AUC</strong>, <strong>PR-AUC</strong>, <strong>Brier score</strong> (when probabilities are available).</li>
      <li><strong>Stratified bootstrap CIs</strong> (n=2000) for TEST balanced accuracy, ROC-AUC, PR-AUC.</li>
      <li><strong>ROC curves</strong> saved per model.</li>
      <li>Per-dataset JSON report + best model persisted to disk.</li>
    </ul>
  </li>
  <li><strong>Summary outputs</strong>
    <ul>
      <li>Prints the <strong>best model per dataset</strong> (STRUCTURAL, GRAPH, COMBINED).</li>
      <li>Writes a compact <strong>Excel</strong> (<code>git_best_models.xlsx</code>) summarizing CV means/SDs, train/test metrics, gaps, confusion matrices, thresholds, and file paths.</li>
    </ul>
  </li>
</ol>

<hr>
<h2>Key Parameters (tunable in the script)</h2>
<ul>
  <li><strong>Split:</strong> <code>TEST_SIZE=0.2</code>, <code>RANDOM_STATE=42</code></li>
  <li><strong>Outliers:</strong> <code>ROW_OUTLIER_FRACTION=0.70</code>, <code>MAX_DROP_FRACTION_TRAIN=0.10</code></li>
  <li><strong>Correlation pruning:</strong> <code>CORR_THRESHOLD=0.8</code>, <code>SAVE_HEATMAPS=True</code></li>
  <li><strong>Scaling (RobustScaler):</strong> <code>ROBUST_QUANTILE_RANGE=(25.0, 75.0)</code>, <code>ROBUST_WITH_CENTERING=True</code>, <code>ROBUST_WITH_SCALING=True</code></li>
  <li><strong>Feature selection (Welch + BH-FDR):</strong> <code>FDR_ENABLE=True</code>, <code>FDR_Q</code> (optional fixed), <code>FDR_Q_GRID=(0.05,0.10,0.15,0.20,0.25,0.30,0.35)</code>, <code>MIN_TTEST_PASS=5</code></li>
  <li><strong>RFE:</strong> <code>RFE_TOP_K=10</code>, estimator = logistic regression (<code>liblinear</code>, <code>class_weight="balanced"</code>)</li>
  <li><strong>Curation:</strong> <code>CURATION_MODE="ask"</code> or <code>"auto"</code>, <code>UNRELATED_FEATURES</code> = {…}</li>
  <li><strong>Modeling:</strong> primary scoring = <code>balanced_accuracy</code>; CV = 5 folds × 5 repeats; ROS ratio = <code>0.5</code> (→ 1:2), Bayesian search iterations = 30</li>
  <li><strong>Bootstrap CIs (test):</strong> <code>n_boot=2000</code></li>
</ul>

<hr>
<h2>Generated Artifacts (by default under <code>BASE_DIR</code>)</h2>
<ul>
  <li><strong>Manifests:</strong> <code>pipeline_manifest.json</code> (split lock), <code>pipeline_scaling_manifest.json</code>, <code>pipeline_corr_manifest.json</code></li>
  <li><strong>Diagnostics:</strong> <code>diagnostics/</code> — outlier bounds and row-flag reports</li>
  <li><strong>Plots:</strong> <code>plots/</code> — correlation heatmaps (pre/post), ROC curves per model</li>
  <li><strong>Cleaned &amp; Derived CSVs (per dataset: STRUCTURAL, GRAPH, COMBINED):</strong>
    <ul>
      <li><code>git_*_cleaned.csv</code></li>
      <li>Split: <code>git_*_train.csv</code>, <code>git_*_test.csv</code> (locked)</li>
      <li>Outliers removed: <code>git_*_train_outliers_removed.csv</code>, <code>git_*_test_outliers_removed.csv</code></li>
      <li>Corr-pruned: <code>git_*_train_corr_pruned.csv</code>, <code>git_*_test_corr_pruned.csv</code></li>
      <li>Robust-scaled: <code>git_*_train_robust_scaled.csv</code>, <code>git_*_test_robust_scaled.csv</code></li>
      <li>FS outputs: <code>git_*_train_ttest_pass.csv</code> → <code>git_*_train_rf_top20.csv</code> → <code>git_*_train_rfe_top10.csv</code> (and matching TEST files)</li>
      <li><strong>Curated final:</strong> <code>git_*_train_rfe_top10_curated.csv</code>, <code>git_*_test_rfe_top10_curated.csv</code></li>
    </ul>
  </li>
  <li><strong>Feature selection report:</strong> <code>pipeline_feature_selection_report.json</code></li>
  <li><strong>Models &amp; reports:</strong> <code>models_structural/</code>, <code>models_graph/</code>, <code>models_combined/</code> (best model + ROC PNGs); <code>git_structural_ml_report.json</code>, <code>git_graph_ml_report.json</code>, <code>git_combined_ml_report.json</code></li>
  <li><strong>Excel summary:</strong> <code>git_best_models.xlsx</code></li>
</ul>

<hr>
<h2>Usage Notes &amp; Troubleshooting</h2>
<ul>
  <li><strong>scikit-optimize is required.</strong> If missing, the script raises an ImportError with installation instructions.</li>
  <li><strong>NaN guardrails:</strong> Correlation pruning <strong>hard-fails</strong> if NaNs appear in numeric features at that stage—investigate earlier cleaning if this triggers.</li>
  <li><strong>Duplicate IDs / label mismatches:</strong> The combine step raises a clear error with the first few offending rows.</li>
  <li><strong>Outlier policy:</strong> Rows are only dropped if TRAIN’s flagged proportion is below the configured maximum; otherwise nothing is dropped (to avoid drastic sample loss).</li>
</ul>

<hr>
<h2>Required Libraries</h2>
<p>Minimal set:</p>
<pre><code>pip install pandas numpy scipy scikit-learn imbalanced-learn scikit-optimize matplotlib joblib xlsxwriter openpyxl
</code></pre>
<p>Optional (enables extra model families if present):</p>
<pre><code>pip install xgboost lightgbm catboost
</code></pre>
<p><strong>Removed from old README:</strong> <code>seaborn</code>, <code>shap</code> (not used by the current pipeline).</p>

<hr>
<h2>Contributions (Ongoing)</h2>
<p>This project is a collaborative effort by <strong>IDS Team 44</strong>. The primary <strong>code development</strong> has been led by <strong>Ghazale Ghaffaripour</strong>, with contributions from <strong>Samin Afzoonkar</strong>. Other team members have been involved in <strong>data preprocessing, analysis, and research article preparation</strong>.</p>

<h3>Code Contributions (Confirmed So Far)</h3>
<ul>
  <li><strong>Ghazale Ghaffaripour</strong> – Lead Developer; Core ML Implementation; Data Processing/Preprocessing (CONN, FreeSurfer); Research Article Writing; Documentation</li>
  <li><strong>Samin Afzoonkar</strong> – Outlier Detection &amp; Correlation Removal; Preprocessing (FreeSurfer); Data Analysis; Research Article Writing; Documentation</li>
</ul>

<h3>Additional Contributions (To Be Finalized with Article)</h3>
<ul>
  <li><strong>Fatemeh Arezoomand</strong> – Preprocessing (CONN, FreeSurfer); Research Article Writing; Documentation</li>
  <li><strong>Mahsa Bahrami</strong> – Research Article Writing; Documentation</li>
  <li><strong>Arman Khanjani</strong> – Preprocessing (CONN, FreeSurfer); Research Article Writing; Documentation</li>
</ul>
<p>Final contributions will be updated upon completion of the research article.</p>

<hr>
<p>© 2024 Team44.<br>
Code is licensed under the MIT License.<br>
Research papers and datasets may have different licensing—please refer to their respective terms.</p>
