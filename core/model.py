"""
core/model.py
=========================
Final Fix:
- Compatible dengan scikit-learn >= 1.5 dan XGBoost >= 2.0
- Pastikan XGBClassifierWrapper terdeteksi sebagai classifier oleh sklearn
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


# ============================================================
# WRAPPER FIX (Explicit classifier identity)
# ============================================================
class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper untuk XGBClassifier agar dikenali sklearn sebagai classifier,
    bukan regressor, saat digunakan dengan CalibratedClassifierCV.
    """
    _estimator_type = "classifier"  # ✅ penting agar sklearn mengenalinya

    def __init__(self, model):
        self.model = model
        # copy atribut fitted bila model sudah dilatih
        if hasattr(model, "classes_"):
            self.classes_ = model.classes_
        if hasattr(model, "n_features_in_"):
            self.n_features_in_ = model.n_features_in_

    def fit(self, X, y):
        """Wrapper tidak melakukan training lagi, hanya memastikan atribut ada."""
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        else:
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
        self.n_features_in_ = getattr(self.model, "n_features_in_", X.shape[1])
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return self.model.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        return self.model.predict_proba(X)

    def score(self, X, y):
        check_is_fitted(self, "classes_")
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ============================================================
# TRAIN BASE MODELS
# ============================================================
def train_base_models(X_train, y_train):
    print("[INFO] Training SVM (RBF kernel)...")
    svm = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)

    print("[INFO] Training XGBoost Classifier...")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    print("[INFO] Training Logistic Regression...")
    log = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=42)
    log.fit(X_train, y_train)

    print("[OK] Base models trained successfully.")
    return svm, xgb, log


# ============================================================
# CALIBRATION
# ============================================================
def calibrate_models(svm, xgb, log, X_calib, y_calib):
    print("[INFO] Calibrating base models...")
    try:
        # Bungkus XGBoost agar dikenali sklearn sebagai classifier
        xgb_wrapped = XGBClassifierWrapper(xgb)

        cal_svm = CalibratedClassifierCV(svm, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        cal_xgb = CalibratedClassifierCV(xgb_wrapped, method="isotonic", cv="prefit").fit(X_calib, y_calib)
        cal_log = CalibratedClassifierCV(log, method="sigmoid", cv="prefit").fit(X_calib, y_calib)

        print("[OK] Models calibrated successfully.")
        return cal_svm, cal_xgb, cal_log
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        raise


# ============================================================
# SAVE ENSEMBLE
# ============================================================
def save_ensemble(models_dict, out_path: str):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(models_dict, out_path)
        print(f"[OK] Ensemble saved to {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save ensemble: {e}")
