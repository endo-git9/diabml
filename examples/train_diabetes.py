# examples/train_diabetes.py
"""
Diabetes Ensemble Model Trainer & Evaluator (Final Version)
===========================================================
- Modular, reliable, and fully reproducible pipeline.
- Trains, calibrates, ensembles, and evaluates model.

import os
import sys
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

# --- Import project modules ---
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from core.data import load_dataset, train_test_calib_split, resample_and_scale
from core.model import train_base_models, calibrate_models
from core.evaluate import compute_metrics
from utils.io_utils import ensure_dir, save_json

sns.set_theme(style="whitegrid", context="talk")

# ============================================================
# CONFIGURATION
# ============================================================
CFG_PATH = os.path.join(ROOT, "config.yaml")
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError("Missing config.yaml")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

OUT_DIR = os.path.join(ROOT, "results")
ensure_dir(OUT_DIR)

# filenames
FN = {
    "roc": "roc.png",
    "pr": "precision_recall.png",
    "cm": "confusion_matrix.png",
    "cal": "calibration_curve.png",
    "prob": "prob_distribution.png",
    "shap_sum": "figure_shap_summary.png",
    "ensemble": "ensemble_model.pkl",
    "metrics": "final_metrics.json"
}

# ============================================================
# LOG HELPER
# ============================================================
def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now():%H:%M:%S} - {msg}")

def safe_save_figure(fig, path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        log(f"Saved → {os.path.basename(path)}", "OK")
    except Exception as e:
        log(f"Failed saving {path}: {e}", "WARN")
    finally:
        plt.close(fig)

def ensure_binary(y):
    return (np.asarray(y).ravel() != 0).astype(int)

# ============================================================
# LOAD DATA
# ============================================================
log("Loading dataset...")
df = load_dataset(cfg["dataset_path"])
target = cfg.get("target_col", "Outcome")
X = df.drop(columns=[target])
y = df[target]

# Split train/calib/test
X_train_sub, X_calib, X_test, y_train_sub, y_calib, y_test = train_test_calib_split(
    df, test_size=cfg.get("test_size", 0.3),
    calib_size=cfg.get("calibration_split", 0.2),
    random_state=cfg.get("random_seed", 42)
)

log("Scaling and resampling...")
X_train_scaled, y_res, X_test_scaled, scaler = resample_and_scale(
    X_train_sub, y_train_sub, X_test, random_state=cfg.get("random_seed", 42)
)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ============================================================
# TRAIN MODELS
# ============================================================
log("Training base models...")
svm, xgb, logreg = train_base_models(X_train_scaled, y_res)

log("Calibrating models...")
X_calib_scaled = scaler.transform(X_calib)
cal_svm, cal_xgb, cal_log = calibrate_models(svm, xgb, logreg, X_calib_scaled, y_calib)

# ============================================================
# ENSEMBLE
# ============================================================
weights = cfg.get("ensemble_weights", [0.33, 0.33, 0.34])
svm_p = cal_svm.predict_proba(X_test_scaled)[:, 1]
xgb_p = cal_xgb.predict_proba(X_test_scaled)[:, 1]
log_p = cal_log.predict_proba(X_test_scaled)[:, 1]
stack_p = np.average([svm_p, xgb_p, log_p], axis=0, weights=weights)

# ============================================================
# METRICS
# ============================================================
TH = cfg.get("threshold", 0.5)
y_bin = ensure_binary(y_test)
metrics = compute_metrics(y_bin, stack_p, threshold=TH)
metrics.update({
    "roc_auc": roc_auc_score(y_bin, stack_p),
    "brier": brier_score_loss(y_bin, stack_p),
    "pr_auc": average_precision_score(y_bin, stack_p)
})
save_json(metrics, os.path.join(OUT_DIR, FN["metrics"]))
log("Saved metrics JSON", "OK")

joblib.dump({"svm": cal_svm, "xgb": cal_xgb, "log": cal_log, "scaler": scaler},
            os.path.join(OUT_DIR, FN["ensemble"]))
log("Saved ensemble_model.pkl", "OK")

# ============================================================
# VISUALIZATIONS
# ============================================================
log("Generating plots...")
# ROC
fpr, tpr, _ = roc_curve(y_bin, stack_p)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
ax.plot([0,1],[0,1],"--",color="gray")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
ax.legend()
safe_save_figure(fig, os.path.join(OUT_DIR, FN["roc"]))

# PR
prec, rec, _ = precision_recall_curve(y_bin, stack_p)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(rec, prec, label=f"AP={metrics['pr_auc']:.3f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
ax.legend()
safe_save_figure(fig, os.path.join(OUT_DIR, FN["pr"]))

# Confusion Matrix
cm = confusion_matrix(y_bin, (stack_p >= TH).astype(int))
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
safe_save_figure(fig, os.path.join(OUT_DIR, FN["cm"]))

# Calibration
prob_true, prob_pred = calibration_curve(y_bin, stack_p, n_bins=10)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(prob_pred, prob_true, "o-", label="Calibration")
ax.plot([0,1],[0,1],"--",color="gray")
ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives", title="Calibration Curve")
ax.legend()
safe_save_figure(fig, os.path.join(OUT_DIR, FN["cal"]))

# Probability Distribution
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(stack_p[y_bin==0], label="Non-Diabetic", kde=True, stat="density", ax=ax)
sns.histplot(stack_p[y_bin==1], label="Diabetic", kde=True, stat="density", ax=ax)
ax.set(xlabel="Predicted Probability", ylabel="Density", title="Probability Distribution by Class")
ax.legend()
safe_save_figure(fig, os.path.join(OUT_DIR, FN["prob"]))

# ============================================================
# SHAP INTERPRETABILITY
# ============================================================
def generate_shap(xgb_model, X_df, out_dir):
    """Generate SHAP summary + local force plot."""
    import shap
    import matplotlib.pyplot as plt

    try:
        X_sample = X_df.sample(min(150, len(X_df)), random_state=42)
        explainer = shap.Explainer(xgb_model, X_sample)
        shap_values = explainer(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Global SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figure_shap_summary.png"), dpi=300)
        plt.close()
        log("Saved SHAP summary", "OK")

        idx = min(5, len(X_sample)-1)
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx].values if hasattr(shap_values[idx], "values") else shap_values[idx],
            X_sample.iloc[idx, :],
            matplotlib=True, show=False
        )
        plt.title(f"Local SHAP Force Plot – Patient #{idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"figure_shap_force_patient_{idx}.png"), dpi=300)
        plt.close()
        log("Saved SHAP force plot", "OK")
    except Exception as e:
        log(f"SHAP generation failed: {e}", "WARN")

log("Generating SHAP plots...")
generate_shap(xgb, X_test_scaled_df, OUT_DIR)

log("Training complete.", "OK")
log(f"Accuracy={metrics['accuracy']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}", "OK")
log(f"Results saved to: {OUT_DIR}", "OK")
