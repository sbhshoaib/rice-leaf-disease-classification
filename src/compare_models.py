"""
Run this AFTER all 5 models have been trained and you have gathered all
five outputs/<model_name>/metrics.json files together in one place
(e.g. combined into a single Kaggle dataset, or a local outputs/ folder
containing one sub-folder per model).

Usage:
    python compare_models.py --outputs-dir /kaggle/working/outputs
"""
import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from config import AVAILABLE_MODELS


def load_all_metrics(outputs_dir):
    rows = []
    for model_name in AVAILABLE_MODELS:
        path = os.path.join(outputs_dir, model_name, "metrics.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping {model_name}")
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "model": model_name,
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
        })
    if not rows:
        raise FileNotFoundError(
            f"No metrics.json files found under {outputs_dir}. "
            "Train the models first, then place all outputs/<model>/ "
            "folders together before running this script."
        )
    return pd.DataFrame(rows)


def plot_comparison(df, outputs_dir):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(10, 6))
    df.set_index("model")[metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Paddy Leaf Disease Classification")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "comparison_chart.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=None)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    if outputs_dir is None:
        from config import OUTPUT_DIR
        outputs_dir = OUTPUT_DIR

    df = load_all_metrics(outputs_dir)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(outputs_dir, "comparison_report.csv")
    df.to_csv(csv_path, index=False)
    plot_comparison(df, outputs_dir)

    print(df.to_string(index=False))
    print(f"\nBest model by macro F1: {df.iloc[0]['model']}")
    print(f"Saved comparison table to {csv_path}")
    print(f"Saved comparison chart to {os.path.join(outputs_dir, 'comparison_chart.png')}")


if __name__ == "__main__":
    main()
