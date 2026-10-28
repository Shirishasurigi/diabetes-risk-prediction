"""
Supervised Learning Pipeline for Diabetes Risk Prediction
Dataset: Kaggle Pima Indians Diabetes Database
Author: Antigravity AI
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STATIC_PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def save_dual_plot(fig, filename):
    p1 = os.path.join(PLOTS_DIR, filename)
    p2 = os.path.join(STATIC_PLOTS_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved visualization: {filename}")

def train_and_evaluate():
    print("=" * 70)
    print("  SUPERVISED LEARNING MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    target = 'Outcome'

    X = df[features].copy()
    y = df[target].copy()

    # 2. Impute Biologically Invalid Zeros
    # Biological features where 0 is invalid
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        X[col] = X[col].replace(0, np.nan)

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    print(f"[*] Test Set Class Balance: Non-Diabetic={sum(y_test==0)}, Diabetic={sum(y_test==1)}")

    # 4. Impute with Median (fit on train, transform on test to prevent data leakage)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 5. Standard Scaling (fit on train, transform on test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 6. Define Supervised Learning Classifiers with optimized parameters
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=4, min_samples_leaf=2,
            random_state=42, class_weight='balanced'
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=42
        ),
        "Support Vector Machine": SVC(
            C=1.2, kernel='rbf', gamma='scale', probability=True, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=9, weights='distance', metric='minkowski'
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=6, min_samples_leaf=3, random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', max_iter=600,
            alpha=0.01, random_state=42, early_stopping=True
        )
    }

    # 7. Model Training, 5-Fold Cross Validation, and Test Evaluation
    results = {}
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        print(f"\n[>] Training {name}...")
        
        # 5-fold cross validation score on training set
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_stratified, scoring='roc_auc')
        cv_auc_mean = float(np.mean(cv_scores))
        cv_auc_std = float(np.std(cv_scores))

        # Fit model on entire training set
        clf.fit(X_train_scaled, y_train)

        # Predictions on unseen test set
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_scaled)

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        brier = brier_score_loss(y_test, y_prob)

        results[name] = {
            "model": clf,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_auc_mean": round(cv_auc_mean, 4),
            "cv_auc_std": round(cv_auc_std, 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": cm.tolist(),
            "y_prob": y_prob.tolist(),
            "y_pred": y_pred.tolist()
        }

        print(f"    - Accuracy   : {acc*100:.2f}%")
        print(f"    - Precision  : {prec*100:.2f}%")
        print(f"    - Recall     : {rec*100:.2f}%")
        print(f"    - F1-Score   : {f1:.4f}")
        print(f"    - ROC-AUC    : {auc:.4f} (5-Fold CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f})")

    # 8. Comparison Table
    summary_list = []
    for name, r in results.items():
        summary_list.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "Specificity": r["specificity"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"],
            "CV-AUC (5-Fold)": f"{r['cv_auc_mean']} ± {r['cv_auc_std']}"
        })
    summary_df = pd.DataFrame(summary_list).sort_values(by="ROC-AUC", ascending=False)
    print("\n" + "=" * 80)
    print("                    MODEL PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Identify Best Model by ROC-AUC and F1-Score
    best_model_name = summary_df.iloc[0]["Model"]
    best_entry = results[best_model_name]
    print(f"\n[*] Top Performing Model: {best_model_name} (ROC-AUC: {best_entry['roc_auc']:.4f}, Accuracy: {best_entry['accuracy']*100:.2f}%)")

    # ================= VISUALIZATION 6: Model Comparison Bar Chart =================
    fig, ax = plt.subplots(figsize=(13, 7))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())
    
    x = np.arange(len(model_names))
    width = 0.16
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
    
    for i, (metric, label, col) in enumerate(zip(metrics_to_plot, labels_to_plot, colors)):
        vals = [results[m][metric] for m in model_names]
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=col, edgecolor='#1e293b', linewidth=0.6)

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Supervised Learning Models Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=18, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_dual_plot(fig, '06_model_comparison_metrics.png')

    # ================= VISUALIZATION 7: Multi-Model ROC Curves =================
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(models))
    for (name, r), col in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{name} (AUC = {r["roc_auc"]:.3f})')
        
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=9.5)
    plt.tight_layout()
    save_dual_plot(fig, '07_roc_curves.png')

    # ================= VISUALIZATION 8: Confusion Matrices Grid =================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        cm_data = np.array(r["confusion_matrix"])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Non-Diabetic', 'Diabetic'],
                    yticklabels=['Non-Diabetic', 'Diabetic'],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {r['accuracy']*100:.1f}% | AUC: {r['roc_auc']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
    # Hide the 8th subplot
    axes[7].axis('off')
    plt.suptitle('Confusion Matrices on Holdout Test Set (N = 154)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_dual_plot(fig, '08_confusion_matrices.png')

    # ================= VISUALIZATION 9: Feature Importance / Odds Ratio =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest Feature Importance
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    ax1.barh([features[i] for i in indices], [importances[i] for i in indices], color='#0ea5e9', edgecolor='#0369a1')
    ax1.set_title('Feature Importance (Random Forest Ensemble)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Gini Impurity Importance', fontsize=10)
    for i, v in enumerate([importances[idx] for idx in indices]):
        ax1.text(v + 0.005, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=9)

    # Logistic Regression Standardized Coefficients
    lr_model = results["Logistic Regression"]["model"]
    coefs = lr_model.coef_[0]
    indices_lr = np.argsort(coefs)
    colors_lr = ['#ef4444' if c > 0 else '#10b981' for c in [coefs[i] for i in indices_lr]]
    ax2.barh([features[i] for i in indices_lr], [coefs[i] for i in indices_lr], color=colors_lr, edgecolor='#334155')
    ax2.set_title('Standardized Coefficients (Logistic Regression)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Log-Odds Coefficient (Positive = Higher Risk)', fontsize=10)
    ax2.axvline(0, color='#64748b', linestyle='--')
    for i, v in enumerate([coefs[idx] for idx in indices_lr]):
        ax2.text(v + (0.03 if v >= 0 else -0.15), i, f"{v:+.2f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_dual_plot(fig, '09_feature_importances.png')

    # 9. Save Best Model and Serialization Artifacts
    best_bundle = {
        "best_model_name": best_model_name,
        "model": results[best_model_name]["model"],
        "scaler": scaler,
        "imputer": imputer,
        "features": features,
        "metrics": {
            k: {m: v for m, v in val.items() if m not in ["model", "y_prob", "y_pred"]}
            for k, val in results.items()
        },
        "zero_medians": {
            col: float(imputer.statistics_[features.index(col)]) for col in zero_cols
        }
    }
    bundle_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_bundle, bundle_path)
    print(f"\n[+] Serialized best model bundle to: {bundle_path}")

    # Also save individual scaler and all models for multi-model web selector
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))
    
    # Save all models dictionary
    all_models_dict = {name: r["model"] for name, r in results.items()}
    joblib.dump(all_models_dict, os.path.join(MODELS_DIR, "all_models.joblib"))

    # Save metrics JSON
    metrics_json_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    clean_metrics = {
        "best_model": best_model_name,
        "models": {
            k: {m: v for m, v in val.items() if m not in ["model", "y_prob", "y_pred"]}
            for k, val in results.items()
        }
    }
    with open(metrics_json_path, "w") as f:
        json.dump(clean_metrics, f, indent=4)
    print(f"[+] Saved metrics JSON to: {metrics_json_path}")

    print("\n" + "=" * 70)
    print("  MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate()
