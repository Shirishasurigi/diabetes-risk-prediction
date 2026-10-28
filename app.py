"""
Flask Web Application & REST API for Diabetes Risk Assessment
Supervised Learning Project with Interactive Clinical Dashboard
Author: Antigravity AI
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ALL_MODELS_PATH = os.path.join(BASE_DIR, "models", "all_models.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics_summary.json")
EDA_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "eda_summary.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load artifacts into memory
bundle = joblib.load(BUNDLE_PATH) if os.path.exists(BUNDLE_PATH) else None
all_models = joblib.load(ALL_MODELS_PATH) if os.path.exists(ALL_MODELS_PATH) else {}
metrics_data = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

eda_data = {}
if os.path.exists(EDA_SUMMARY_PATH):
    with open(EDA_SUMMARY_PATH, "r") as f:
        eda_data = json.load(f)

@app.route("/")
def home():
    # Load dataset sample for the UI table
    sample_records = []
    if os.path.exists(DATA_PATH):
        df_sample = pd.read_csv(DATA_PATH).head(8)
        sample_records = df_sample.to_dict(orient="records")

    models_list = list(all_models.keys()) if all_models else ["Random Forest"]
    default_model = bundle.get("best_model_name", "Random Forest") if bundle else "Random Forest"
    
    return render_template(
        "index.html",
        models=models_list,
        default_model=default_model,
        metrics=metrics_data.get("models", {}),
        eda=eda_data,
        sample_records=sample_records
    )

@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        model_name = data.get("model", bundle["best_model_name"] if bundle else "Random Forest")
        
        features = bundle["features"]
        imputer = bundle["imputer"]
        scaler = bundle["scaler"]
        
        clf = all_models.get(model_name, bundle["model"])

        # Extract values
        values = []
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for f in features:
            v = data.get(f)
            val = float(v) if (v is not None and str(v).strip() != "") else 0.0
            if f in zero_cols and val <= 0:
                val = np.nan
            values.append(val)

        X_raw = pd.DataFrame([values], columns=features)
        X_imp = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imp)

        pred = int(clf.predict(X_scaled)[0])
        prob = float(clf.predict_proba(X_scaled)[0][1]) if hasattr(clf, "predict_proba") else (1.0 if pred == 1 else 0.0)

        # Categorize Risk
        if prob < 0.30:
            risk_category = "Low Risk"
            risk_color = "#10b981"
            guidance = "Clinical metrics are predominantly within standard physiological ranges. Maintain an active lifestyle and schedule standard routine annual health checkups."
        elif prob < 0.60:
            risk_category = "Moderate Risk"
            risk_color = "#f59e0b"
            guidance = "Several markers suggest metabolic strain or pre-diabetic tendencies. Prioritize dietary refinements (low glycemic index), 150 min/week aerobic exercise, and glucose monitoring."
        else:
            risk_category = "High Risk"
            risk_color = "#ef4444"
            guidance = "Significant clinical markers indicate elevated diabetes risk or insulin resistance. Diagnostic confirmation (Fasting Plasma Glucose & HbA1c testing) with a medical physician is recommended."

        # Compute factor contributions
        contributions = []
        # Key thresholds
        glucose_val = float(data.get("Glucose", 0))
        bmi_val = float(data.get("BMI", 0))
        age_val = float(data.get("Age", 0))
        insulin_val = float(data.get("Insulin", 0))
        bp_val = float(data.get("BloodPressure", 0))
        pedigree_val = float(data.get("DiabetesPedigreeFunction", 0))

        if glucose_val >= 140:
            contributions.append({"factor": "Elevated Glucose", "impact": "High", "desc": f"{glucose_val} mg/dL is above normal postprandial threshold (>140 mg/dL)"})
        elif glucose_val > 110:
            contributions.append({"factor": "Borderline Glucose", "impact": "Moderate", "desc": f"{glucose_val} mg/dL is in the borderline pre-diabetic range"})

        if bmi_val >= 30:
            contributions.append({"factor": "Obesity Index (BMI)", "impact": "High", "desc": f"BMI {bmi_val} indicates Class I+ obesity, compounding insulin resistance"})
        elif bmi_val >= 25:
            contributions.append({"factor": "Overweight BMI", "impact": "Moderate", "desc": f"BMI {bmi_val} is above optimal range (18.5 - 24.9)"})

        if age_val >= 45:
            contributions.append({"factor": "Age Risk Bracket", "impact": "Moderate", "desc": f"Age {int(age_val)} presents naturally higher physiological susceptibility"})

        if pedigree_val >= 0.60:
            contributions.append({"factor": "Genetic Predisposition", "impact": "Moderate", "desc": f"Pedigree score {pedigree_val} reflects notable hereditary history"})

        if not contributions:
            contributions.append({"factor": "Balanced Indicators", "impact": "Positive", "desc": "Current physiological indicators are well within healthy boundaries"})

        response_payload = {
            "success": True,
            "model_used": model_name,
            "prediction": pred,
            "diagnosis": "Diabetic Likelihood" if pred == 1 else "Non-Diabetic Range",
            "probability": round(prob, 4),
            "probability_percent": round(prob * 100, 1),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "guidance": guidance,
            "contributing_factors": contributions,
            "imputed_values": {f: round(float(X_imp[0][i]), 1) for i, f in enumerate(features)}
        }
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/download/dataset")
def download_dataset():
    return send_from_directory(os.path.join(BASE_DIR, "data"), "diabetes.csv", as_attachment=True)

@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics_data)

if __name__ == "__main__":
    print("[*] Starting Diabetes Risk Prediction Web Application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
