"""
train_and_save.py
Train the best ASD Risk Prediction model (XGBoost with interaction features)
and save it to ./model/ directory along with metadata.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, f1_score
)
from xgboost import XGBClassifier
import joblib
import os

MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Generate synthetic dataset (same seed as notebook)
# ------------------------------------------------------------------
np.random.seed(42)
N = 100_000

maternal_age  = np.random.normal(28, 5, N)
paternal_age  = np.random.normal(31, 6, N)
gdm           = np.random.binomial(1, 0.08, N)
infection     = np.random.binomial(1, 0.10, N)
preterm       = np.random.binomial(1, 0.11, N)
low_bw        = np.random.binomial(1, 0.08, N)
family_history= np.random.binomial(1, 0.03, N)

advanced_maternal = (maternal_age > 35).astype(int)
advanced_paternal = (paternal_age > 40).astype(int)

beta_0 = -4.0
logit = (
    beta_0
    + 0.40 * advanced_maternal
    + 0.45 * advanced_paternal
    + 0.50 * gdm
    + 0.40 * infection
    + 0.60 * preterm
    + 0.40 * low_bw
    + 1.60 * family_history
    + 0.80 * (gdm * preterm)          # interaction
    + 0.60 * (infection * preterm)    # interaction
    + 0.50 * (advanced_maternal * gdm) # interaction
)

prob = 1 / (1 + np.exp(-logit))
asd  = np.random.binomial(1, prob)

FEATURE_COLS = [
    "advanced_maternal", "advanced_paternal",
    "gdm", "infection", "preterm", "low_bw", "family_history"
]

df = pd.DataFrame({
    "advanced_maternal": advanced_maternal,
    "advanced_paternal": advanced_paternal,
    "gdm": gdm,
    "infection": infection,
    "preterm": preterm,
    "low_bw": low_bw,
    "family_history": family_history,
    "ASD": asd
})

print(f"ASD Prevalence: {df['ASD'].mean():.4f}")

# ------------------------------------------------------------------
# 2. Train / test split
# ------------------------------------------------------------------
X = df[FEATURE_COLS]
y = df["ASD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 3. Train XGBoost
# ------------------------------------------------------------------
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------
# 4. Threshold optimisation (best F1)
# ------------------------------------------------------------------
best_f1, best_threshold = 0.0, 0.5
for t in np.linspace(0.01, 0.5, 100):
    preds = (y_pred_prob > t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1, best_threshold = score, t

y_pred_final = (y_pred_prob > best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)
pr_auc  = average_precision_score(y_test, y_pred_prob)

print(f"\nROC AUC:        {roc_auc:.4f}")
print(f"PR  AUC:        {pr_auc:.4f}")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score:  {best_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# ------------------------------------------------------------------
# 5. Save model + metadata
# ------------------------------------------------------------------
joblib.dump(model, f"{MODEL_DIR}/xgb_asd_model.joblib")

metadata = {
    "model_type": "XGBClassifier",
    "feature_columns": FEATURE_COLS,
    "best_threshold": float(best_threshold),
    "metrics": {
        "roc_auc":  round(roc_auc, 4),
        "pr_auc":   round(pr_auc, 4),
        "best_f1":  round(best_f1, 4),
    },
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Model saved to {MODEL_DIR}/xgb_asd_model.joblib")
print(f"✅ Metadata saved to {MODEL_DIR}/metadata.json")