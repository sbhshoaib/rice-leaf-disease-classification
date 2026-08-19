"""
Central configuration for the paddy leaf disease classification project.
Edit the constants below if your paths differ.
"""
import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Where Kaggle mounts added datasets. If you added the dataset under a
# different slug, change this, or just let auto-detection in
# data_utils.find_dataset_root() locate it under /kaggle/input.
KAGGLE_INPUT_DIR = "/kaggle/input"

# Preferred default (works if running on Kaggle with the dataset added).
DEFAULT_DATASET_HINT = "rice-disease-dataset"

# Local fallback (e.g. running outside Kaggle for testing).
LOCAL_DATASET_DIR = os.path.join(os.getcwd(), "data")

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.getcwd(), "working")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")

# ---------------------------------------------------------------------
# Data / training hyperparameters
# ---------------------------------------------------------------------
IMG_SIZE = (224, 224)          # (height, width) — same for all 5 models
BATCH_SIZE = 32
SEED = 42

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Phase 1: train only the new classification head (backbone frozen)
HEAD_EPOCHS = 10
HEAD_LR = 1e-3

# Phase 2: fine-tune the top of the backbone at a low learning rate
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_LAYERS = 30   # unfreeze last N layers of the backbone

EARLY_STOPPING_PATIENCE = 6

# ---------------------------------------------------------------------
# Model registry — keys are the --model CLI argument values
# ---------------------------------------------------------------------
AVAILABLE_MODELS = [
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "convnext_tiny",
]
