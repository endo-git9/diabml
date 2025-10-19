# core/interpret.py
import pandas as pd
import matplotlib.pyplot as plt
import shap

def save_shap_summary(xgb_model, X_df, out_path):
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_df)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
