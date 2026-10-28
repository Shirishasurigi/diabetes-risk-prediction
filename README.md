# DiaGuard AI — Diabetes Risk Prediction Using Supervised Learning

An end-to-end clinical machine learning system for diabetes risk prediction using the authentic **Kaggle Pima Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases). 

---

## 🌟 Project Overview

This project implements a complete, medical-grade supervised learning pipeline:
1. **Dataset Ingestion**: Kaggle Pima Indians Diabetes Database (`diabetes.csv`, 768 patient records, 8 clinical features + binary outcome).
2. **Exploratory Data Analysis (EDA)**: Missing zero inspection, distribution modeling, outlier detection, and correlation analysis.
3. **Data Preprocessing & Imputation**: Handling biologically implausible zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI using median imputation fitted strictly on the training set to prevent data leakage.
4. **Supervised Algorithms Benchmarked**:
   - Random Forest Classifier (Top Performing: **82.07% ROC-AUC**, **75.32% Accuracy**, **75.93% Sensitivity**)
   - Gradient Boosting Classifier (**82.04% ROC-AUC**)
   - Support Vector Machine (RBF Kernel, **79.70% ROC-AUC**)
   - Logistic Regression (**81.26% ROC-AUC**)
   - Decision Tree Classifier (**80.16% ROC-AUC**, **76.62% Accuracy**)
   - Multi-Layer Perceptron Neural Network (**81.76% ROC-AUC**)
   - K-Nearest Neighbors (**79.69% ROC-AUC**)
5. **Interactive Glassmorphic Web Application**: Real-time risk probability gauge, clinical driver explanation, dynamic presets, EDA tab switcher, and model evaluation charts.
6. **Command-Line Interface (CLI)**: Interactive single-patient mode and high-throughput batch CSV processing.

---

## 📁 Repository Structure

```
diabetes-risk-prediction/
│
├── diabetes.csv                    # Kaggle Pima Indians Diabetes Dataset
├── data/
│   ├── diabetes.csv                # Data directory copy
│   └── eda_summary.json            # Machine-readable statistical summary
│
├── plots/                          # High-resolution generated analytical charts
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_matrix.png
│   ├── 04_glucose_bmi_scatter.png
│   ├── 05_boxplots_by_outcome.png
│   ├── 06_model_comparison_metrics.png
│   ├── 07_roc_curves.png
│   ├── 08_confusion_matrices.png
│   └── 09_feature_importances.png
│
├── models/
│   ├── best_model.joblib           # Bundled best model, imputer, scaler & metadata
│   ├── all_models.joblib           # Dictionary of all 7 trained models
│   ├── scaler.joblib               # StandardScaler object
│   ├── imputer.joblib              # SimpleImputer object
│   └── metrics_summary.json        # Quantitative benchmark evaluation
│
├── templates/
│   └── index.html                  # Responsive modern web UI
├── static/
│   ├── css/style.css               # Dark-mode glassmorphic styling
│   ├── js/app.js                   # Interactive client logic & live API integration
│   └── plots/                      # Web-accessible plot assets
│
├── eda.py                          # Exploratory Data Analysis script
├── train.py                        # Supervised Learning training & benchmarking pipeline
├── predict.py                      # CLI & batch prediction engine
├── app.py                          # Flask web server & REST API
├── requirements.txt                # Python environment specifications
└── README.md                       # Complete technical & clinical documentation
```

---

## 📊 Dataset Description & Clinical Features

Source: [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Feature | Description | Range / Units | Imputed Zeros |
|---|---|---|---|
| **Pregnancies** | Number of times pregnant | 0 - 17 | Valid 0 (No) |
| **Glucose** | 2-hour oral glucose tolerance test | 44 - 199 mg/dL | 5 (0.65%) |
| **BloodPressure** | Diastolic blood pressure | 24 - 122 mm Hg | 35 (4.56%) |
| **SkinThickness** | Triceps skin fold thickness | 7 - 99 mm | 227 (29.56%) |
| **Insulin** | 2-Hour serum insulin | 14 - 846 µU/mL | 374 (48.70%) |
| **BMI** | Body mass index (Weight in kg / Height in m²) | 18.2 - 67.1 kg/m² | 11 (1.43%) |
| **DiabetesPedigreeFunction** | Genetic hereditary pedigree score | 0.078 - 2.42 | Valid continuous |
| **Age** | Chronological age | 21 - 81 years | Valid continuous |
| **Outcome** (Target) | Diabetes diagnosis indicator | 0 = Negative, 1 = Positive | Target |

---

## 🔬 Supervised Learning Benchmark Results

Evaluated on unseen holdout test set (20%, N=154) with 5-Fold Stratified Cross-Validation on training data:

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | 5-Fold CV AUC |
|---|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | **75.32%** | **62.12%** | **75.93%** | **0.7500** | **0.6833** | **0.8207** | **0.8371 ± 0.024** |
| **Decision Tree** | 76.62% | 65.00% | 72.22% | 0.7900 | 0.6842 | 0.8016 | 0.7616 ± 0.033 |
| **Gradient Boosting** | 72.73% | 62.00% | 57.41% | 0.8100 | 0.5962 | 0.8204 | 0.8196 ± 0.022 |
| **Neural Network (MLP)** | 71.43% | 60.42% | 53.70% | 0.8100 | 0.5686 | 0.8176 | 0.8486 ± 0.014 |
| **Logistic Regression** | 69.48% | 57.45% | 50.00% | 0.8000 | 0.5347 | 0.8126 | 0.8430 ± 0.019 |
| **Support Vector Machine** | 74.68% | 66.67% | 55.56% | 0.8500 | 0.6061 | 0.7970 | 0.8332 ± 0.022 |
| **K-Nearest Neighbors** | 74.03% | 63.46% | 61.11% | 0.8100 | 0.6226 | 0.7969 | 0.8115 ± 0.035 |

> **Key Clinical Takeaway:** In medical diagnostic screening, **Recall (Sensitivity)** is of paramount importance to minimize false negatives (failing to diagnose a diabetic individual). The **Random Forest Classifier** achieved the highest Recall (**75.93%**) and highest ROC-AUC (**0.8207**).

---

## 🚀 Getting Started

### 1. Environment Setup
```powershell
cd C:\Users\Shirisha\.gemini\antigravity-ide\scratch\diabetes-risk-prediction
.\.venv\Scripts\activate
```

### 2. Run Exploratory Data Analysis (EDA)
Generates full statistical summaries and saves all 5 analytical charts:
```powershell
python eda.py
```

### 3. Run Model Training & Evaluation
Trains all 7 supervised models, evaluates test metrics, and exports serialized models:
```powershell
python train.py
```

### 4. Run CLI Predictions
**Interactive Mode:**
```powershell
python predict.py
```

**Direct Flag Prediction:**
```powershell
python predict.py --glucose 165 --bmi 34.5 --age 45 --pregnancies 3
```

**Batch Processing:**
```powershell
python predict.py --batch data/diabetes.csv
```

### 5. Launch the Web Application
```powershell
python app.py
```
Open your browser at **`http://127.0.0.1:5000`** to access the interactive dashboard.

---

## 🌐 REST API Usage

### `POST /api/predict`
**Request Payload:**
```json
{
  "model": "Random Forest",
  "Pregnancies": 4,
  "Glucose": 160.0,
  "BloodPressure": 82.0,
  "SkinThickness": 30.0,
  "Insulin": 150.0,
  "BMI": 34.2,
  "DiabetesPedigreeFunction": 0.58,
  "Age": 46
}
```

**Response Payload:**
```json
{
  "success": true,
  "model_used": "Random Forest",
  "prediction": 1,
  "diagnosis": "Diabetic Likelihood",
  "probability_percent": 81.4,
  "risk_category": "High Risk",
  "risk_color": "#ef4444",
  "guidance": "Significant clinical markers indicate elevated diabetes risk or insulin resistance. Diagnostic confirmation (Fasting Plasma Glucose & HbA1c testing) with a physician is recommended.",
  "contributing_factors": [
    {"factor": "Elevated Glucose", "impact": "High", "desc": "160.0 mg/dL is above normal postprandial threshold (>140 mg/dL)"},
    {"factor": "Obesity Index (BMI)", "impact": "High", "desc": "BMI 34.2 indicates Class I+ obesity, compounding insulin resistance"}
  ]
}
```
