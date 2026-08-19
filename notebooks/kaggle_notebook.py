# =====================================================================
# Copy each "# --- CELL n ---" block into its own cell in a Kaggle
# Notebook. Run one CNN model per Kaggle session (edit MODEL_NAME).
# =====================================================================

# --- CELL 1: clone the repo -------------------------------------------------
# !rm -rf rice-disease-classification
# !git clone https://github.com/<YOUR_USERNAME>/rice-disease-classification.git
# %cd rice-disease-classification
# !pip install -q -r requirements.txt

# --- CELL 2: sanity-check the dataset was added ----------------------------
# import os
# print(os.listdir("/kaggle/input"))

# --- CELL 3: train ONE model (change MODEL_NAME each run) ------------------
# MODEL_NAME = "efficientnet_b0"
# # options: efficientnet_b0 | resnet50 | densenet121 | mobilenet_v3_large | convnext_tiny
# !python src/train.py --model {MODEL_NAME}

# --- CELL 4: quick look at outputs ------------------------------------------
# import os
# out_dir = f"/kaggle/working/outputs/{MODEL_NAME}"
# print(os.listdir(out_dir))

# --- CELL 5 (OPTIONAL, only after all 5 models are trained and their
#             outputs/<model>/ folders are together, e.g. via a combined
#             Kaggle "outputs" dataset you build from your 5 runs) --------
# !python src/compare_models.py --outputs-dir /kaggle/working/outputs
