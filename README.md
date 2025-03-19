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
  - Mahsa Bahrami
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

