"""
All evaluation / reporting utilities used after a model finishes training:
  - classification metrics (accuracy, precision, recall, F1)
  - confusion matrix
  - ROC curves (one-vs-rest, multiclass)
  - training curves
  - prediction example grid
  - Grad-CAM grids for correct and incorrect predictions
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import cv2

from gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer


def _collect_predictions(model, test_ds):
    y_true, y_prob = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        y_prob.append(probs)
        y_true.append(batch_y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def evaluate_and_save(model, test_ds, class_names, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    y_true, y_pred, y_prob = _collect_predictions(model, test_ds)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": report,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{model_name}] Accuracy={acc:.4f}  Precision={precision:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}")

    _plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name)
    _plot_roc_curves(y_true, y_prob, class_names, output_dir, model_name)

    return metrics, y_true, y_pred, y_prob


def _plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_roc_curves(y_true, y_prob, class_names, output_dir, model_name):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_training_curves(history, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist.get("accuracy", []), label="train")
    axes[0].plot(hist.get("val_accuracy", []), label="val")
    axes[0].set_title(f"Accuracy — {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist.get("loss", []), label="train")
    axes[1].plot(hist.get("val_loss", []), label="val")
    axes[1].set_title(f"Loss — {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f, indent=2)


def _load_raw_image(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img


def save_prediction_examples(model, test_files, class_names, preprocess_fn,
                              img_size, output_dir, n=16):
    os.makedirs(output_dir, exist_ok=True)
    filepaths, labels = test_files
    idxs = np.random.choice(len(filepaths), size=min(n, len(filepaths)), replace=False)

    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    plt.figure(figsize=(cols * 3, rows * 3.3))

    for plot_i, i in enumerate(idxs):
        raw = _load_raw_image(filepaths[i], img_size)
        batch = preprocess_fn(tf.expand_dims(tf.cast(raw, tf.float32), 0))
        probs = model.predict(batch, verbose=0)[0]
        pred = np.argmax(probs)
        true = labels[i]

        plt.subplot(rows, cols, plot_i + 1)
        plt.imshow(raw)
        color = "green" if pred == true else "red"
        plt.title(f"T:{class_names[true]}\nP:{class_names[pred]} ({probs[pred]*100:.0f}%)",
                   fontsize=8, color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_examples.png"), dpi=150)
    plt.close()


def save_gradcam_grid(model, test_files, class_names, preprocess_fn, img_size,
                       output_dir, correct=True, n=8):
    os.makedirs(output_dir, exist_ok=True)
    filepaths, labels = test_files
    last_conv = find_last_conv_layer(model)

    found = []
    order = np.random.permutation(len(filepaths))
    for i in order:
        raw = _load_raw_image(filepaths[i], img_size)
        batch = preprocess_fn(tf.expand_dims(tf.cast(raw, tf.float32), 0))
        probs = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(probs))
        true = labels[i]
        is_correct = (pred == true)
        if is_correct == correct:
            found.append((raw, batch, pred, true, probs[pred]))
        if len(found) >= n:
            break

    if not found:
        print(f"No {'correct' if correct else 'incorrect'} examples found for Grad-CAM grid.")
        return

    cols = 4
    rows = int(np.ceil(len(found) / cols))
    plt.figure(figsize=(cols * 3, rows * 3.3))

    for plot_i, (raw, batch, pred, true, conf) in enumerate(found):
        heatmap, _ = make_gradcam_heatmap(batch, model, last_conv, pred_index=pred)
        overlay = overlay_heatmap(raw.astype(np.uint8), heatmap)

        plt.subplot(rows, cols, plot_i + 1)
        plt.imshow(overlay)
        title_color = "green" if correct else "red"
        plt.title(f"T:{class_names[true]}\nP:{class_names[pred]} ({conf*100:.0f}%)",
                   fontsize=8, color=title_color)
        plt.axis("off")

    tag = "correct" if correct else "incorrect"
    plt.suptitle(f"Grad-CAM — {tag} predictions", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"gradcam_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()
