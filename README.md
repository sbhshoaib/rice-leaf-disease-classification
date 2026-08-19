# Paddy Leaf Disease Classification — Comparative CNN Study

Classifies paddy (rice) leaf diseases from images and compares five CNN
architectures trained on the same dataset:

1. EfficientNetB0
2. ResNet50
3. DenseNet121
4. MobileNetV3-Large
5. ConvNeXt-Tiny

Dataset: https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset

## Repository layout

```
rice-disease-classification/
├── requirements.txt
├── src/
│   ├── config.py          # paths & hyperparameters
│   ├── data_utils.py      # dataset discovery, splitting, tf.data pipelines
│   ├── model_factory.py   # builds each of the 5 backbones
│   ├── gradcam.py         # generic Grad-CAM (works for all 5 architectures)
│   ├── eval_utils.py      # metrics, confusion matrix, ROC, plots
│   ├── train.py           # train + evaluate ONE model end-to-end
│   └── compare_models.py  # combine metrics.json from all 5 runs
├── notebooks/
│   └── kaggle_notebook.py # paste-into-Kaggle-cells version of the pipeline
└── outputs/                # created at runtime, one sub-folder per model
```

## How to use (short version)

1. Push this repo to GitHub.
2. On Kaggle: New Notebook → Add Data → search `rice-disease-dataset`
   (by anshulm257) → Add.
3. In the first notebook cell, clone this repo (see step-by-step guide
   given in chat).
4. Run `python src/train.py --model efficientnet_b0` (then resnet50,
   densenet121, mobilenet_v3_large, convnext_tiny — one per Kaggle run).
5. Download each `outputs/<model>/` folder (Kaggle → Output tab).
6. After all 5 are trained, run `src/compare_models.py` once you have all
   five `outputs/<model>/metrics.json` files together, to get the final
   comparison table/chart.

Full step-by-step instructions were provided in chat.
