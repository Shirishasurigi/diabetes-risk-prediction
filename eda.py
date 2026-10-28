"""
Exploratory Data Analysis (EDA) for Diabetes Risk Prediction
Dataset: Kaggle Pima Indians Diabetes Database
Author: Antigravity AI
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STATIC_PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

def save_dual_plot(fig, filename):
    p1 = os.path.join(PLOTS_DIR, filename)
    p2 = os.path.join(STATIC_PLOTS_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved visualization: {filename}")

def run_eda():
    print("=" * 60)
    print("  DIABETES RISK PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"\n[*] Loaded dataset successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[*] First 5 records:")
    print(df.head().to_string())
    
    print("\n[*] Data Types and Non-Null Counts:")
    print(df.info())
    
    # 2. Descriptive Statistics
    print("\n[*] Statistical Summary:")
    stats_df = df.describe().T
    print(stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_string())
    
    # 3. Biologically implausible zero check
    # In Pima Indians dataset, 0 in Glucose, BloodPressure, SkinThickness, Insulin, BMI represents missing measurement.
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_stats = {}
    print("\n[*] Analysis of Missing Values Coded as Zeros:")
    for col in zero_columns:
        zero_cnt = (df[col] == 0).sum()
        zero_pct = (zero_cnt / len(df)) * 100
        zero_stats[col] = {'count': int(zero_cnt), 'percentage': round(zero_pct, 2)}
        print(f"  - {col:<15}: {zero_cnt:3d} zeros ({zero_pct:5.2f}%)")

    # 4. Target Class Balance
    outcome_counts = df['Outcome'].value_counts()
    outcome_pct = df['Outcome'].value_counts(normalize=True) * 100
    print(f"\n[*] Target Distribution (Outcome):")
    print(f"  - Non-Diabetic (0): {outcome_counts[0]} ({outcome_pct[0]:.2f}%)")
    print(f"  - Diabetic (1)    : {outcome_counts[1]} ({outcome_pct[1]:.2f}%)")

    # 5. Correlation with Outcome
    correlations = df.corr()['Outcome'].sort_values(ascending=False)
    print("\n[*] Pearson Correlation with Outcome:")
    for feat, corr_val in correlations.items():
        if feat != 'Outcome':
            print(f"  - {feat:<24}: {corr_val:+.4f}")

    # ================= VISUALIZATION 1: Target Distribution =================
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#10b981', '#ef4444']
    bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], outcome_counts.values, color=colors, width=0.5, edgecolor='#334155', linewidth=1.2)
    ax.set_title('Distribution of Target Variable (Outcome)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Patients', fontsize=11)
    ax.set_ylim(0, max(outcome_counts.values) * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 10, f'{int(h)} ({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1e293b')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    save_dual_plot(fig, '01_target_distribution.png')

    # ================= VISUALIZATION 2: Feature Distributions by Outcome =================
    features = [c for c in df.columns if c != 'Outcome']
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        ax = axes[i]
        sns.histplot(data=df, x=col, hue='Outcome', kde=True, ax=ax,
                     palette={0: '#0ea5e9', 1: '#f43f5e'}, bins=25, alpha=0.45,
                     edgecolor='white', linewidth=0.5)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        legend = ax.get_legend()
        if legend:
            legend.set_title('Diagnosis')
            labels = ['Non-Diabetic (0)', 'Diabetic (1)']
            for t, l in zip(legend.texts, labels):
                t.set_text(l)
    
    plt.tight_layout()
    save_dual_plot(fig, '02_feature_distributions.png')

    # ================= VISUALIZATION 3: Correlation Heatmap =================
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.7, vmin=-.3, center=0,
                square=True, linewidths=1, annot=True, fmt='.2f',
                cbar_kws={"shrink": .8, "label": "Pearson Correlation"}, ax=ax)
    ax.set_title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_dual_plot(fig, '03_correlation_matrix.png')

    # ================= VISUALIZATION 4: Glucose vs BMI Scatter Plot =================
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df[df['Outcome'] == 0]['BMI'], df[df['Outcome'] == 0]['Glucose'],
                         c='#10b981', alpha=0.65, edgecolors='none', s=45, label='Non-Diabetic (0)')
    scatter2 = ax.scatter(df[df['Outcome'] == 1]['BMI'], df[df['Outcome'] == 1]['Glucose'],
                          c='#ef4444', alpha=0.75, edgecolors='k', linewidth=0.3, s=55, label='Diabetic (1)')
    ax.axhline(140, color='#b91c1c', linestyle='--', linewidth=1.5, label='Impaired Glucose (>140 mg/dL)')
    ax.axvline(30, color='#d97706', linestyle=':', linewidth=1.5, label='Obesity Threshold (BMI > 30)')
    ax.set_title('Clinical Risk Landscape: Glucose vs BMI', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Body Mass Index (BMI)', fontsize=11)
    ax.set_ylabel('2-Hour Plasma Glucose (mg/dL)', fontsize=11)
    ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    save_dual_plot(fig, '04_glucose_bmi_scatter.png')

    # ================= VISUALIZATION 5: Boxplots by Outcome =================
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(features):
        ax = axes[i]
        sns.boxplot(x='Outcome', y=col, data=df, ax=ax, palette=['#38bdf8', '#fb7185'], width=0.45)
        ax.set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'], fontsize=10)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
    plt.suptitle('Clinical Indicator Variations by Outcome (Boxplots & Outliers)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_dual_plot(fig, '05_boxplots_by_outcome.png')

    # 6. Save JSON Summary
    eda_summary = {
        "dataset_name": "Pima Indians Diabetes Database (Kaggle)",
        "sample_count": int(df.shape[0]),
        "feature_count": int(df.shape[1] - 1),
        "target_column": "Outcome",
        "class_distribution": {
            "non_diabetic": int(outcome_counts[0]),
            "diabetic": int(outcome_counts[1]),
            "diabetic_ratio": round(float(outcome_pct[1]), 2)
        },
        "missing_zeros_summary": zero_stats,
        "correlations_with_outcome": {k: round(float(v), 4) for k, v in correlations.items()},
        "feature_means_by_outcome": {
            col: {
                "non_diabetic": round(float(df[df['Outcome'] == 0][col].mean()), 2),
                "diabetic": round(float(df[df['Outcome'] == 1][col].mean()), 2)
            }
            for col in features
        }
    }
    
    summary_path = os.path.join(BASE_DIR, "data", "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=4)
    print(f"\n[+] EDA summary JSON written to: {summary_path}")
    print("\n" + "=" * 60)
    print("  EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
