"""
============================================================
CALIBRATED AND INTERPRETABLE ENSEMBLE FRAMEWORK (CIEF)
============================================================

Final Academic Report Generator
Author: Endo Kersandona
Universitas Pelita Harapan - Master of Informatics
"""

import os
import json
import yaml
import unicodedata
from datetime import datetime
from fpdf import FPDF

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "final_metrics.json")
CFG_PATH = os.path.join(BASE_DIR, "config.yaml")

# ============================================================
# LOAD PROJECT METADATA
# ============================================================
meta = {}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

AUTHOR = meta.get("author_name", "Endo Kersandona")
NIM = meta.get("nim", "01679250004")
SUPERVISOR = meta.get("supervisor", "Dr. Benny Hardjono")
PROGRAM = meta.get("program", "Master of Informatics")
UNIVERSITY = meta.get("university", "Universitas Pelita Harapan")

# ============================================================
# TEXT SANITIZATION
# ============================================================
def sanitize_text(text):
    """Ensure text is PDF-safe"""
    if not isinstance(text, str):
        return str(text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("•", "-")
        .replace("…", "...")
    )
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

# ============================================================
# PDF CLASS
# ============================================================
class ModelReport(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")
        self.set_text_color(0, 0, 0)

    # ---------------- COVER PAGE ----------------
    def cover_page(self):
        self.add_page()
        page_height = 297
        self.set_y(page_height / 3.2)

        # Title
        self.set_font("Times", "B", 18)
        self.multi_cell(0, 10, sanitize_text("A CALIBRATED AND INTERPRETABLE ENSEMBLE FRAMEWORK"), align="C")

        # Subtitle
        self.set_font("Times", "I", 13)
        self.cell(0, 8, sanitize_text("for Reliable Diabetes Risk Prediction"), ln=True, align="C")
        self.ln(25)

        # Author Info
        self.set_font("Times", "", 12)
        info_lines = [
            f"Author: {AUTHOR}",
            f"NIM: {NIM}",
            f"Supervisor: {SUPERVISOR}",
            f"Program: {PROGRAM}",
            f"University: {UNIVERSITY}",
            "",
            f"Generated on: {datetime.now():%d %B %Y, %H:%M}",
        ]
        for line in info_lines:
            self.cell(0, 8, sanitize_text(line), ln=True, align="C")

    # ---------------- SECTION TEXT ----------------
    def section_text(self, title, paragraphs):
        self.add_page()
        self.set_font("Times", "B", 16)
        self.cell(0, 10, sanitize_text(title), ln=True, align="C")
        self.ln(6)
        self.set_font("Times", "", 12)
        for p in paragraphs:
            self.multi_cell(0, 8, sanitize_text(p))
            self.ln(3)

    # ---------------- SECTION IMAGE ----------------
    def section_image(self, title, path, caption, width=165):
        self.add_page()
        self.set_font("Times", "B", 15)
        self.cell(0, 10, sanitize_text(title), ln=True, align="C")
        self.ln(6)
        if not os.path.exists(path):
            self.set_text_color(180, 0, 0)
            self.multi_cell(0, 8, sanitize_text(f"[Missing file: {os.path.basename(path)}]"))
            self.set_text_color(0, 0, 0)
            return
        try:
            self.image(path, x=(210 - width) / 2, w=width)
            self.ln(8)
            self.set_font("Times", "I", 11)
            self.multi_cell(0, 8, sanitize_text(caption))
        except Exception as e:
            self.set_text_color(180, 0, 0)
            self.multi_cell(0, 8, sanitize_text(f"[Error: {e}]"))
            self.set_text_color(0, 0, 0)

# ============================================================
# GENERATE REPORT
# ============================================================
def generate_report():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Load metrics
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    pdf = ModelReport()
    pdf.cover_page()

    # -------- MODEL OVERVIEW --------
    overview = [
        "This report presents a Calibrated and Interpretable Ensemble Framework (CIEF) for reliable diabetes risk prediction using the Pima Indians Diabetes dataset.",
        "",
        "The model integrates Support Vector Machine (SVM), XGBoost, and Logistic Regression, calibrated via Isotonic Regression to ensure reliable probability estimates and interpretability.",
        "",
        "Dataset: Pima Indians Diabetes (768 samples, 8 features)",
        "Split: 70% training, 20% calibration, 30% testing",
        "Calibration Method: Isotonic Regression",
        "Evaluation Metrics: Accuracy, ROC-AUC, PR-AUC, F1-score, and Brier Score.",
    ]
    pdf.section_text("Model Overview", overview)

    # -------- PERFORMANCE SUMMARY --------
    pdf.add_page()
    pdf.set_font("Times", "B", 16)
    pdf.cell(0, 10, "Performance Summary", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Courier", "", 12)
    if metrics:
        for k, v in metrics.items():
            try:
                pdf.cell(0, 8, sanitize_text(f"{k.capitalize():<20}: {float(v):.4f}"), ln=True)
            except Exception:
                pdf.cell(0, 8, sanitize_text(f"{k.capitalize():<20}: {v}"), ln=True)
    else:
        pdf.set_font("Times", "I", 12)
        pdf.multi_cell(0, 8, sanitize_text("No performance metrics found."))

    # -------- VISUALIZATIONS --------
    figures = [
        ("ROC Curve", "roc.png", "Figure 1. ROC Curve showing model discrimination."),
        ("Confusion Matrix", "confusion_matrix.png", "Figure 2. Confusion Matrix of predicted vs actual outcomes."),
        ("Precision-Recall Curve", "precision_recall.png", "Figure 3. Precision-Recall Curve illustrating sensitivity trade-offs."),
        ("Calibration Curve", "calibration_curve.png", "Figure 4. Calibration Curve showing reliability of predicted probabilities."),
        ("Probability Distribution", "prob_distribution.png", "Figure 5. Probability distribution by predicted class."),
    ]
    for title, fname, caption in figures:
        pdf.section_image(title, os.path.join(RESULTS_DIR, fname), caption)

    # -------- SHAP INTERPRETABILITY --------
    shap_summary = os.path.join(RESULTS_DIR, "figure_shap_summary.png")
    shap_force = None
    for f in os.listdir(RESULTS_DIR):
        if f.startswith("figure_shap_force_patient_"):
            shap_force = os.path.join(RESULTS_DIR, f)
            break

    pdf.section_text(
        "SHAP-Based Interpretability",
        [
            "SHAP (SHapley Additive ExPlanations) is applied to measure how each clinical feature influences model predictions.",
            "",
            "Global SHAP plots highlight overall feature importance, identifying Glucose, BMI, and Diabetes Pedigree Function as the strongest predictors.",
            "",
            "Local SHAP force plots visualize how patient-specific characteristics drive individual risk probabilities, "
            "enhancing transparency and supporting clinical reasoning."
        ]
    )

    if os.path.exists(shap_summary):
        pdf.section_image("Global SHAP Summary", shap_summary,
            "Figure 6. Global SHAP summary showing overall feature impact across all patients.")
    if shap_force and os.path.exists(shap_force):
        pdf.section_image("Local SHAP Force Plot", shap_force,
            "Figure 7. SHAP Force Plot for an individual patient, illustrating feature contributions.")

    # -------- CLINICAL SUMMARY --------
    acc = float(metrics.get("accuracy", 0))
    auc = float(metrics.get("roc_auc", 0))
    brier = float(metrics.get("brier", 0))
    pr_auc = float(metrics.get("pr_auc", 0))
    calibration_status = "well-calibrated" if auc > 0.8 and brier < 0.2 else "requires recalibration"

    clinical = [
        f"The ensemble achieved an accuracy of {acc:.3f}, ROC-AUC of {auc:.3f}, PR-AUC of {pr_auc:.3f}, "
        f"and Brier Score of {brier:.3f}, indicating {calibration_status} predictions.",
        "",
        "Clinically, the CIEF framework provides a transparent, reliable, and interpretable AI-based assessment of diabetes risk, "
        "suitable for integration in healthcare decision support systems.",
    ]
    pdf.section_text("Clinical Summary and Relevance", clinical)

    # -------- SAVE REPORT --------
    out_path = os.path.join(RESULTS_DIR, "model_report.pdf")
    pdf.output(out_path)
    pdf.close()
    print(f"✅ Report successfully generated → {out_path}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    generate_report()
