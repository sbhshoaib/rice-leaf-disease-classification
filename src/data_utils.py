"""
Dataset discovery + stratified split + tf.data pipeline builders.

Kaggle dataset folder layouts vary (sometimes there's an extra nested
folder, sometimes train/val/test are pre-split). find_dataset_root()
walks the input directory and returns the first folder whose immediate
subfolders each contain image files — i.e. a folder-per-class layout.
"""
import os
import glob
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    KAGGLE_INPUT_DIR, DEFAULT_DATASET_HINT, LOCAL_DATASET_DIR,
    IMG_SIZE, BATCH_SIZE, SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def _dir_has_images(d):
    for f in os.listdir(d):
        if f.lower().endswith(tuple(e.lower() for e in IMAGE_EXTS)):
            return True
    return False


def find_dataset_root(base_dir=None):
    """Return the folder containing one sub-folder per class."""
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir)
    if os.path.isdir(KAGGLE_INPUT_DIR):
        for name in os.listdir(KAGGLE_INPUT_DIR):
            if DEFAULT_DATASET_HINT in name.lower():
                candidates.append(os.path.join(KAGGLE_INPUT_DIR, name))
        # fallback: search every dataset added to the notebook
        candidates.append(KAGGLE_INPUT_DIR)
    candidates.append(LOCAL_DATASET_DIR)

    for root in candidates:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if not dirnames:
                continue
            subdirs = [os.path.join(dirpath, d) for d in dirnames]
            image_subdirs = [d for d in subdirs if os.path.isdir(d) and _dir_has_images(d)]
            # A valid "folder per class" root has >=2 class folders,
            # each directly containing images.
            if len(image_subdirs) >= 2:
                return dirpath
    raise FileNotFoundError(
        "Could not auto-locate a folder-per-class image dataset under "
        f"{KAGGLE_INPUT_DIR} or {LOCAL_DATASET_DIR}. Pass --data-dir explicitly."
    )


def list_files_and_labels(root):
    class_names = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    filepaths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root, cls)
        for ext in IMAGE_EXTS:
            for fp in glob.glob(os.path.join(cls_dir, f"*{ext}")):
                filepaths.append(fp)
                labels.append(idx)
    if not filepaths:
        raise FileNotFoundError(f"No images found under {root}")
    return filepaths, labels, class_names


def stratified_split(filepaths, labels, seed=SEED):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        filepaths, labels, train_size=TRAIN_SPLIT,
        stratify=labels, random_state=seed,
    )
    rel_val = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=rel_val,
        stratify=y_tmp, random_state=seed,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _make_dataset(filepaths, labels, preprocess_fn, augment=False, shuffle=False):
    path_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, label

    ds = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def _augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            return img, label
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(2000, len(filepaths)), seed=SEED)

    def _preprocess(img, label):
        return preprocess_fn(img), label

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(preprocess_fn, data_dir=None):
    """Returns train_ds, val_ds, test_ds, class_names, and the raw test
    file list (needed later for prediction-example / Grad-CAM images)."""
    root = find_dataset_root(data_dir)
    filepaths, labels, class_names = list_files_and_labels(root)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(filepaths, labels)

    print(f"Dataset root: {root}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/Val/Test sizes: {len(Xtr)}/{len(Xva)}/{len(Xte)}")

    train_ds = _make_dataset(Xtr, ytr, preprocess_fn, augment=True, shuffle=True)
    val_ds = _make_dataset(Xva, yva, preprocess_fn, augment=False, shuffle=False)
    test_ds = _make_dataset(Xte, yte, preprocess_fn, augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, class_names, (Xte, yte)
