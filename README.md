# 👶 CardioCare: Cardiotocogram (CTG) Fetal Health Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end clinical machine learning system and interactive bedside Streamlit web application for real-time **Cardiotocography (CTG)** multi-class fetal state classification (Normal, Suspect, Pathological).

---

## 📋 Table of Contents
1. [Clinical Problem and Overview](#-clinical-problem-and-overview)
2. [Dataset and Physiological Indicators](#-dataset-and-physiological-indicators)
3. [Jupyter Notebook Workflow](#-jupyter-notebook-workflow)
4. [Methodology and Pipeline Architecture](#-methodology-and-pipeline-architecture)
5. [Benchmark Results](#-benchmark-results)
6. [Best Model Performance (LightGBM)](#-best-model-performance-lightgbm)
7. [Top Clinical Risk Factors](#-top-clinical-risk-factors)
8. [Interactive Streamlit Application](#-interactive-streamlit-application)
9. [Project Structure](#-project-structure)
10. [Installation and Quickstart](#-installation-and-quickstart)

---

## 🩺 Clinical Problem and Overview

Cardiotocography (CTG) is the continuous electronic recording of the **Fetal Heart Rate (FHR)** alongside **Uterine Contractions (UC)** during the third trimester of pregnancy and intrapartum labor.

### The Obstetric Stakes
During labor, uterine contractions temporarily compress intra-myometrial blood vessels, transiently reducing uteroplacental blood flow. A healthy fetus with intact autonomic autoregulatory mechanisms easily tolerates this transient stress. However, a compromised fetus may rapidly develop **intrauterine hypoxia (oxygen deprivation)** and systemic metabolic acidosis. Failure or delay in recognizing acute fetal distress can cause:
- **Hypoxic-Ischemic Encephalopathy (HIE)**
- **Permanent neurological damage (Cerebral Palsy)**
- **Intrauterine or intrapartum fetal demise**

### Multi-Class Clinical States
Obstetric consensus guidelines classify CTG patterns into three triage categories:
* **Class 1 (Normal)**: Reassuring baseline (110–160 bpm), frequent accelerations, preserved short/long-term variability, and no pathological decelerations. Routine intrapartum monitoring.
* **Class 2 (Suspect)**: Equivocal tracings with reduced variability or non-severe decelerations. Mandates active surveillance, maternal repositioning, oxygenation, and fetal scalp blood lactate sampling.
* **Class 3 (Pathological)**: Severe tracing abnormalities (prolonged/recurrent late decelerations, severely abnormal variability, or sinusoidal pattern). Indicates acute acidemia and mandates emergency operative delivery (Cesarean section).

---

## 📊 Dataset and Physiological Indicators

The cohort comprises **2,126 cardiotocograms** from the official UCI Machine Learning CTG repository with **zero missing values**.

```
Class 1 (Normal):       1,655 patients (77.8%)  ████████████████
Class 2 (Suspect):        295 patients (13.9%)  ███
Class 3 (Pathological):   176 patients ( 8.3%)  ██
Total Patients:         2,126 recordings
```

### Key Diagnostic Variables
| Variable | Clinical Significance | Normal Target Range |
| :--- | :--- | :--- |
| `baseline value` | Baseline fetal heart rate in beats per minute | 110 – 160 bpm |
| `accelerations` | Transient increases in FHR ($\ge 15$ bpm for $\ge 15$ s) | Frequent with fetal movement |
| `uterine_contractions` | Frequency of uterine contractions per second | 3 – 5 per 10 minutes during active labor |
| `abnormal_short_term_variability` (ASTV) | Percentage of time with abnormal short-term beat-to-beat variability | $< 50\%$ |
| `mean_value_of_short_term_variability` (MSTV) | Mean short-term FHR variability | Reassuring when $> 1.0$ |
| `percentage_of_time_with_abnormal_long_term_variability` (ALTV) | Percentage of time with abnormal long-term variability | $< 15\%$ |
| `mean_value_of_long_term_variability` (MLTV) | Mean long-term FHR variability | Reassuring when elevated |
| `prolongued_decelerations` | Severe FHR decelerations lasting $> 2$ minutes | $0.0$ (Critical emergency if present) |
| `histogram_variance` | Spread and variability of the FHR distribution | Narrow and stable |

---

## 📓 Jupyter Notebook Workflow

The repository includes a single, self-contained, fully executed Jupyter notebook: [`fetal_health_classification.ipynb`](fetal_health_classification.ipynb).

The notebook is organized into 7 sequential steps:

1. **Data Loading and Exploratory Data Analysis (EDA)**:
   - Verification of dataset dimensions (2,126 rows $\times$ 22 columns) and confirmation of 0 missing values.
   - Target class frequency analysis and visualization (bar chart and pie chart).
   - Summary statistics and boxplot distributions of critical clinical indicators across the three fetal states.
2. **Feature Matrix Preparation and Stratified Train-Test Split**:
   - Extraction of 21 morphological and histogram CTG features.
   - 75% train (1,594 records) / 25% test (532 records) stratified split (`random_state=42`), preserving class ratios.
3. **Class Imbalance Handling via SMOTE**:
   - Application of Synthetic Minority Over-sampling Technique (SMOTE) strictly to the training fold:
     ```python
     smote = SMOTE(random_state=42)
     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
     ```
   - Balanced training size expanded to 3,723 instances (exactly 1,241 per class) with zero data leakage into the test set.
4. **Model Training and 5-Fold Stratified GridSearchCV Tuning**:
   - Prefix-free hyperparameter grids passed directly to `GridSearchCV` without dictionary prefixes.
   - 5-Fold Stratified Cross-Validation optimizing for `f1_macro` across 3 competitive ensemble architectures:
     - **Random Forest Classifier**: tuned over `n_estimators`, `max_depth`, and `min_samples_leaf`.
     - **XGBoost Classifier**: tuned over `n_estimators`, `max_depth`, and `learning_rate` (using 0-indexed targets).
     - **LightGBM Classifier**: tuned over `n_estimators`, `max_depth`, `learning_rate`, and `num_leaves`.
5. **Models Evaluation**:
   - Comparative test holdout evaluation across all 3 models on 532 unseen patients.
   - **Normalized Confusion Matrix for the best model only (LightGBM)**, clearly highlighting true vs. predicted rates.
6. **Clinical Feature Importance Rankings**:
   - Extraction and visualization of Gini feature importances from Random Forest.
   - Clinical interpretation of the top diagnostic variables (ALTV, ASTV, histogram mean, prolonged decelerations).
7. **Export Production Model Artifact Bundle**:
   - Serialization of trained estimators, evaluation scores, best hyperparameters, feature rankings, and summary statistics to [`fetal_health_bundle.joblib`](fetal_health_bundle.joblib).

---

## 🔬 Methodology and Pipeline Architecture

```
[Raw CTG Data: 2,126 recordings]
               │
               ▼
[Stratified 75/25 Train-Test Split]
        ├── Train: 1,594 records (preserved class balance)
        └── Test:    532 records (unseen holdout)
               │
               ▼
[SMOTE Resampling on Training Set Only]
        └── 3,723 balanced training samples (1,241 per class)
               │
               ▼
[5-Fold Stratified GridSearchCV (Macro F1 Objective)]
        ├── Random Forest (n_estimators, max_depth, min_samples_leaf)
        ├── XGBoost (n_estimators, max_depth, learning_rate)
        └── LightGBM (n_estimators, max_depth, learning_rate, num_leaves)
               │
               ▼
[Holdout Test Evaluation (532 Patients)]
        └── Best Model Confusion Matrix (LightGBM), Macro F1, Class 3 Recall
               │
               ▼
[Serialized Production Bundle -> Streamlit Clinical App]
```

### Optimal Hyperparameters Found via GridSearch
* **LightGBM**: `{'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 150, 'num_leaves': 31}`
* **XGBoost**: `{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 150}`
* **Random Forest**: `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}`

---

## 🏆 Benchmark Results

Evaluated on the **unseen holdout test set of 532 patients** (Normal: 414, Suspect: 74, Pathological: 44):

| Model Architecture | 5-Fold CV Macro F1 | Holdout Accuracy | Macro F1 | Pathological Sensitivity (Recall) | Suspect Sensitivity | Normal Sensitivity | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** 🥇 | **0.9868** | **94.17%** | **0.8895** | **95.45% (42/44)** | 71.62% | **98.07%** | **0.9397** |
| **Random Forest** 🥈 | 0.9812 | 93.05% | 0.8822 | 90.91% (40/44) | **79.73%** | 95.65% | 0.9311 |
| **XGBoost** 🥉 | 0.9820 | 93.05% | 0.8700 | 90.91% (40/44) | 71.62% | 97.10% | 0.9290 |

---

## 🎯 Best Model Performance (LightGBM)

**LightGBM** achieved superior performance across both cross-validation and the unseen holdout test set:
- **Test Accuracy**: **94.17%** (501 / 532 patients correctly classified).
- **Emergency Recall (Pathological)**: **95.45%** (correctly identified 42 out of 44 critical fetal distress cases).
- **5-Fold CV Macro F1**: **0.9868**.

### Normalized Confusion Matrix (LightGBM)
```
                  Predicted Normal    Predicted Suspect    Predicted Pathological
True Normal            98.1% (406)          1.7% (7)              0.2% (1)
True Suspect           25.7% (19)          71.6% (53)             2.7% (2)
True Pathological       4.5% (2)            0.0% (0)             95.5% (42)
```

> **Clinical Significance**: LightGBM minimizes False Negatives for the critical Pathological class to only 4.5% (2 out of 44), ensuring that fetuses in acute intrapartum distress are flagged for immediate obstetric evaluation and intervention.

---

## 🔍 Top Clinical Risk Factors

Analysis of Random Forest Gini feature importances reveals the dominant physiological predictors:

| Rank | CTG Diagnostic Feature | Gini Importance | Clinical Rationale |
| :---: | :--- | :---: | :--- |
| **1** | `percentage_of_time_with_abnormal_long_term_variability` (ALTV) | **13.37%** | Prolonged suppression of long-term variability indicates autonomic depression and developing fetal acidosis. |
| **2** | `abnormal_short_term_variability` (ASTV) | **11.99%** | Beat-to-beat variability reflects instantaneous vagal tone. High ASTV is one of the earliest signs of fetal hypoxia. |
| **3** | `histogram_mean` | **9.14%** | Shifts in the mean heart rate capture underlying baseline bradycardia or compensatory tachycardia. |
| **4** | `prolongued_decelerations` | **8.90%** | Decelerations lasting $> 2$ minutes signify profound cord compression or acute placental hypoperfusion. |
| **5** | `mean_value_of_short_term_variability` (MSTV) | **7.39%** | Quantitative measure of beat-to-beat fluctuations; low values reflect non-reassuring physiological state. |
| **6** | `histogram_median` | **6.93%** | Median FHR reflects central tendency without distortion from transient deceleration nadirs. |
| **7** | `accelerations` | **6.89%** | Periodic FHR accelerations provide robust confirmation of intact central nervous system oxygenation. |
| **8** | `mean_value_of_long_term_variability` (MLTV) | **6.60%** | Captures slower rhythmic autonomic cycles over minutes. |

---

## 🖥️ Interactive Streamlit Application

The bedside clinical decision support interface features:

* **Bedside CTG Evaluation**:
  * 3 quick-load clinical presets:
    1. *Normal Fetal Status* (High accelerations, normal baseline)
    2. *Suspect Fetal Status* (Reduced variability, moderate contractions)
    3. *Pathological Fetal Distress* (Prolonged decelerations, severe abnormal variability)
  * Balanced 4-column numerical inputs for physiological markers.
  * Right-aligned key vital indicators (ASTV, Prolonged Decelerations, FHR Baseline) with real-time normal/warning alerts.
  * Real-time triage recommendation and probability distribution table.
* **Model Benchmarks and Optimal Parameters**:
  * Side-by-side performance comparison across all tuned models.
  * Collapsible inspection of optimal hyperparameter values found by 5-fold grid search.
* **Feature Importance Rankings**:
  * Interactive horizontal bar chart ranking all 21 diagnostic CTG features.
* **Explore Clinical Cohort Data**:
  * Full interactive data table exploring the complete 2,126-patient dataset.

---

## 📁 Project Structure

```
fetal-health-classification/
├── app.py                             # Interactive Streamlit Web Application
├── fetal_health_classification.ipynb  # Self-contained Jupyter Notebook (EDA -> Tuning -> Evaluation)
├── fetal_health.csv                   # Cleaned UCI Cardiotocography Dataset (2,126 rows)
├── fetal_health_bundle.joblib         # Serialized models, parameters, and metadata
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Standard git ignore rules
└── README.md                          # Comprehensive clinical and technical documentation
```

---

## 🚀 Installation and Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Rabie62/fetal-health-classification.git
cd fetal-health-classification
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook fetal_health_classification.ipynb
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive application.
