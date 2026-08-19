"""
Builds one of the 5 CNN backbones with a shared classification head, so
that the comparison between architectures is fair (same head, same
input size, same training regime — only the backbone differs).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50, DenseNet121, MobileNetV3Large, ConvNeXtTiny,
)
from tensorflow.keras.applications import (
    efficientnet, resnet, densenet, mobilenet_v3, convnext,
)

from config import IMG_SIZE

BACKBONES = {
    "efficientnet_b0": (EfficientNetB0, efficientnet.preprocess_input),
    "resnet50": (ResNet50, resnet.preprocess_input),
    "densenet121": (DenseNet121, densenet.preprocess_input),
    "mobilenet_v3_large": (MobileNetV3Large, mobilenet_v3.preprocess_input),
    "convnext_tiny": (ConvNeXtTiny, convnext.preprocess_input),
}


def get_preprocess_fn(model_name):
    if model_name not in BACKBONES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(BACKBONES)}")
    return BACKBONES[model_name][1]


def build_model(model_name, num_classes):
    if model_name not in BACKBONES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(BACKBONES)}")

    backbone_cls, _ = BACKBONES[model_name]
    input_shape = (*IMG_SIZE, 3)

    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = False  # phase 1: frozen backbone

    x = base.output
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name=f"{model_name}_classifier")
    return model, base


def unfreeze_top_layers(base_model, n_layers):
    """Unfreeze the last n_layers of the backbone for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    # keep BatchNorm layers frozen even inside the unfrozen block —
    # important for small fine-tuning datasets
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
