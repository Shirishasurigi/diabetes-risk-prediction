"""
Command Line Prediction Tool for Diabetes Risk Assessment
Uses Trained Supervised Learning Models
Author: Antigravity AI
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ALL_MODELS_PATH = os.path.join(BASE_DIR, "models", "all_models.joblib")

def load_artifacts():
    if not os.path.exists(BUNDLE_PATH):
        print(f"[!] Model bundle not found at {BUNDLE_PATH}. Please run train.py first.")
        sys.exit(1)
    bundle = joblib.load(BUNDLE_PATH)
    all_models = joblib.load(ALL_MODELS_PATH) if os.path.exists(ALL_MODELS_PATH) else {bundle["best_model_name"]: bundle["model"]}
    return bundle, all_models

def predict_single(data_dict, model_name=None):
    bundle, all_models = load_artifacts()
    features = bundle["features"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    
    selected_name = model_name if (model_name and model_name in all_models) else bundle["best_model_name"]
    clf = all_models[selected_name]

    # Convert to array according to feature order
    values = []
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for f in features:
        val = float(data_dict.get(f, 0.0))
        if f in zero_cols and val <= 0:
            val = np.nan
        values.append(val)
        
    X_raw = pd.DataFrame([values], columns=features)
    X_imp = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp)

    # Prediction
    pred = int(clf.predict(X_scaled)[0])
    prob = float(clf.predict_proba(X_scaled)[0][1]) if hasattr(clf, "predict_proba") else (1.0 if pred == 1 else 0.0)

    # Risk category determination
    if prob < 0.30:
        risk_level = "LOW RISK"
        color_code = "\033[92m" # Green
        clinical_note = "Normal physiological range. Maintain regular healthy lifestyle and annual screening."
    elif prob < 0.60:
        risk_level = "MODERATE RISK"
        color_code = "\033[93m" # Yellow
        clinical_note = "Elevated risk indicators observed. Dietary counseling, regular aerobic exercise, and glycemic monitoring advised."
    else:
        risk_level = "HIGH RISK"
        color_code = "\033[91m" # Red
        clinical_note = "High likelihood of diabetes or severe insulin resistance. Medical evaluation (HbA1c & Fasting Glucose confirmation) strongly recommended."

    reset_code = "\033[0m"

    return {
        "model_used": selected_name,
        "prediction": pred,
        "diagnosis": "Diabetic / High Risk" if pred == 1 else "Non-Diabetic / Normal",
        "probability_diabetic": round(prob, 4),
        "probability_percent": f"{prob * 100:.2f}%",
        "risk_level": risk_level,
        "color_code": color_code,
        "reset_code": reset_code,
        "clinical_note": clinical_note,
        "imputed_features": {f: round(float(X_imp[0][i]), 2) for i, f in enumerate(features)}
    }

def interactive_mode():
    print("=" * 65)
    print("       DIABETES RISK PREDICTION - INTERACTIVE CLINICAL CLI")
    print("=" * 65)
    print("Enter patient clinical parameters (press Enter for defaults):")
    
    defaults = {
        'Pregnancies': 2,
        'Glucose': 120.0,
        'BloodPressure': 70.0,
        'SkinThickness': 25.0,
        'Insulin': 85.0,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.35,
        'Age': 32
    }

    patient = {}
    for feat, default_val in defaults.items():
        user_input = input(f" - {feat} [Default: {default_val}]: ").strip()
        patient[feat] = float(user_input) if user_input else default_val

    result = predict_single(patient)
    print("\n" + "-" * 65)
    print("                      ASSESSMENT REPORT")
    print("-" * 65)
    print(f"Model Engine        : {result['model_used']}")
    print(f"Outcome Diagnosis   : {result['diagnosis']}")
    print(f"Diabetes Probability: {result['probability_percent']}")
    print(f"Risk Stratification : {result['color_code']}{result['risk_level']}{result['reset_code']}")
    print(f"Recommendation      : {result['clinical_note']}")
    print("-" * 65)

def batch_predict(csv_path):
    print(f"[*] Loading batch file: {csv_path}")
    df_batch = pd.read_csv(csv_path)
    bundle, all_models = load_artifacts()
    features = bundle["features"]
    
    results = []
    for _, row in df_batch.iterrows():
        p_res = predict_single(row.to_dict())
        results.append({
            "Probability": p_res["probability_percent"],
            "Diagnosis": p_res["diagnosis"],
            "Risk_Level": p_res["risk_level"]
        })
    res_df = pd.concat([df_batch, pd.DataFrame(results)], axis=1)
    out_path = os.path.join(BASE_DIR, "data", "batch_predictions.csv")
    res_df.to_csv(out_path, index=False)
    print(f"[+] Processed {len(res_df)} records. Saved results to {out_path}")
    print(res_df.head(10).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict diabetes risk using trained ML models")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch inference")
    parser.add_argument("--glucose", type=float, help="Plasma glucose concentration")
    parser.add_argument("--bmi", type=float, help="Body mass index")
    parser.add_argument("--age", type=int, help="Patient age")
    parser.add_argument("--insulin", type=float, help="Serum insulin")
    parser.add_argument("--bp", type=float, help="Diastolic Blood Pressure")
    parser.add_argument("--pregnancies", type=int, default=0, help="Number of pregnancies")
    parser.add_argument("--pedigree", type=float, default=0.47, help="Diabetes pedigree function")
    parser.add_argument("--skin", type=float, default=20.0, help="Skinfold thickness")
    parser.add_argument("--model", type=str, default="Random Forest", help="Algorithm name")

    args = parser.parse_args()

    if args.batch:
        batch_predict(args.batch)
    elif args.glucose is not None and args.bmi is not None and args.age is not None:
        patient = {
            'Pregnancies': args.pregnancies,
            'Glucose': args.glucose,
            'BloodPressure': args.bp if args.bp is not None else 70.0,
            'SkinThickness': args.skin,
            'Insulin': args.insulin if args.insulin is not None else 80.0,
            'BMI': args.bmi,
            'DiabetesPedigreeFunction': args.pedigree,
            'Age': args.age
        }
        res = predict_single(patient, model_name=args.model)
        print(f"\n[Result] Diagnosis: {res['diagnosis']} | Risk: {res['risk_level']} | Probability: {res['probability_percent']}")
    else:
        interactive_mode()
