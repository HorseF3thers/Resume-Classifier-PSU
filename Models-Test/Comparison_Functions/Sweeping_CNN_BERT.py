#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
DANGER CELL
Hyperparameter sweep:
Runs CNN and BERT multiple times with different max_len and num_epochs
to compare performance and training time.

Pulled from the final "DANGER CELL" in Comparison_CNN_BERT.py so it can be
run in a fresh kernel on its own.

Mitch Readinger
CMPSC 445
Final Project
Resume Classifier
"""


# In[ ]:


import os
import gc
import pandas as pd

from CNN_for_comparison import run_cnn
from BERT_for_comparison import run_bert


# === CONFIG ===
# Team: Update this path if your CSV lives somewhere else
CSV_PATH = r"C:\Users\mshar\Desktop\School Fall 2025\CMPSC 445\Final Project\models\data\Resume.csv"

# Base params (matching the comparison file defaults)
CNN_BASE_PARAMS = {
    "csv_path": CSV_PATH,
    "batch_size": 64,
}

BERT_BASE_PARAMS = {
    "csv_path": CSV_PATH,
    "batch_size": 8,
}

# Swept parameters
SWEEP_MAX_LENS   = [128, 256]
SWEEP_NUM_EPOCHS = [5, 10]


def run_hyperparameter_sweep():
    """
    Run the CNN/BERT hyperparameter sweep and return a summary DataFrame.
    """
    # Sanity check on CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    experiments = []

    for max_len in SWEEP_MAX_LENS:
        for num_epochs in SWEEP_NUM_EPOCHS:
            # --- CNN ---
            print(f"\n=== Running CNN (max_len={max_len}, num_epochs={num_epochs}) ===")
            cnn_res = run_cnn(
                csv_path=CNN_BASE_PARAMS["csv_path"],
                max_len=max_len,
                num_epochs=num_epochs,
                batch_size=CNN_BASE_PARAMS["batch_size"],
            )

            # Keep only the metrics we need for the table to reduce memory usage
            experiments.append({
                "model_name": cnn_res["model_name"],
                "max_len": cnn_res["max_len"],
                "num_epochs": cnn_res["num_epochs"],
                "accuracy": cnn_res["accuracy"],
                "f1": cnn_res["f1"],
                "total_training_time": cnn_res["total_training_time"],
            })

            gc.collect()

            # --- BERT ---
            print(f"\n=== Running BERT (max_len={max_len}, num_epochs={num_epochs}) ===")
            bert_res = run_bert(
                csv_path=BERT_BASE_PARAMS["csv_path"],
                max_len=max_len,
                num_epochs=num_epochs,
                batch_size=BERT_BASE_PARAMS["batch_size"],
            )

            experiments.append({
                "model_name": bert_res["model_name"],
                "max_len": bert_res["max_len"],
                "num_epochs": bert_res["num_epochs"],
                "accuracy": bert_res["accuracy"],
                "f1": bert_res["f1"],
                "total_training_time": bert_res["total_training_time"],
            })

            gc.collect()

    # Build a summary DataFrame of all experiment runs
    rows = []
    for r in experiments:
        rows.append({
            "model": r["model_name"],
            "max_len": r["max_len"],
            "num_epochs": r["num_epochs"],
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            # Include training time since that's part of the comparison story
            "total_training_time_sec": r["total_training_time"],
        })

    exp_results_df = pd.DataFrame(rows)
    return exp_results_df


if __name__ == "__main__":
    print("=== Hyperparameter Sweep: CNN vs BERT ===")
    results = run_hyperparameter_sweep()

    # Print in console
    print("\n=== Experiment Summary ===")
    # Sort for readability: by model, then max_len, then num_epochs
    results_sorted = results.sort_values(
        by=["model", "max_len", "num_epochs"],
        ascending=[True, True, True]
    )
    print(results_sorted.to_string(index=False))

    # Save to CSV in the same directory
    out_path = "hyperparameter_sweep_results.csv"
    results_sorted.to_csv(out_path, index=False)
    print(f"\nResults saved to {os.path.abspath(out_path)}")


# In[ ]:




