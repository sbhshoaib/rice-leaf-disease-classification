"""
Train + evaluate ONE model end-to-end.

Usage (run one model at a time, serially, e.g. once per Kaggle session):
    python train.py --model efficientnet_b0
    python train.py --model resnet50
    python train.py --model densenet121
    python train.py --model mobilenet_v3_large
    python train.py --model convnext_tiny

All outputs (metrics.json, confusion_matrix.png, roc_curves.png,
training_curves.png, prediction_examples.png, gradcam_correct.png,
gradcam_incorrect.png, and the saved .keras model) are written to
/kaggle/working/outputs/<model_name>/
"""
import os
import sys
import argparse
import random

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OUTPUT_DIR, IMG_SIZE, SEED, HEAD_EPOCHS, HEAD_LR,
    FINE_TUNE_EPOCHS, FINE_TUNE_LR, FINE_TUNE_UNFREEZE_LAYERS,
    EARLY_STOPPING_PATIENCE, AVAILABLE_MODELS,
)
from data_utils import build_datasets
from model_factory import build_model, unfreeze_top_layers, get_preprocess_fn
from eval_utils import (
    evaluate_and_save, plot_training_curves,
    save_prediction_examples, save_gradcam_grid,
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--data-dir", default=None,
                         help="Override auto-detected dataset root")
    parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    args = parser.parse_args()

    set_seed()

    model_name = args.model
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    preprocess_fn = get_preprocess_fn(model_name)
    train_ds, val_ds, test_ds, class_names, test_files = build_datasets(
        preprocess_fn, data_dir=args.data_dir
    )
    num_classes = len(class_names)

    model, base = build_model(model_name, num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7,
        ),
    ]

    # ---------------- Phase 1: train the classification head ----------------
    print("\n=== Phase 1: training head (backbone frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.head_epochs, callbacks=callbacks,
    )

    # ---------------- Phase 2: fine-tune top of the backbone ----------------
    print("\n=== Phase 2: fine-tuning top backbone layers ===")
    unfreeze_top_layers(base, FINE_TUNE_UNFREEZE_LAYERS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.fine_tune_epochs, callbacks=callbacks,
    )

    # Merge history from both phases for a single training-curve plot
    combined = {}
    for k in history1.history:
        combined[k] = history1.history[k] + history2.history.get(k, [])

    class _H:
        pass
    merged_history = _H()
    merged_history.history = combined
    plot_training_curves(merged_history, out_dir, model_name)

    # ---------------- Evaluation ----------------
    print("\n=== Evaluating on held-out test set ===")
    evaluate_and_save(model, test_ds, class_names, out_dir, model_name)

    print("\n=== Saving prediction examples ===")
    save_prediction_examples(model, test_files, class_names, preprocess_fn,
                              IMG_SIZE, out_dir, n=16)

    print("\n=== Saving Grad-CAM grids ===")
    save_gradcam_grid(model, test_files, class_names, preprocess_fn, IMG_SIZE,
                       out_dir, correct=True, n=8)
    save_gradcam_grid(model, test_files, class_names, preprocess_fn, IMG_SIZE,
                       out_dir, correct=False, n=8)

    # ---------------- Save the model ----------------
    model_path = os.path.join(out_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"\nSaved model to {model_path}")
    print(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
