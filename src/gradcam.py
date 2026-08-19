"""
Generic Grad-CAM that works for any of the 5 backbones by automatically
locating the last layer whose output is a 4-D feature map (B, H, W, C).
"""
import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except AttributeError:
            continue
        if isinstance(shape, list):
            shape = shape[0]
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4-D feature map layer for Grad-CAM.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """img_array: preprocessed image batch of shape (1, H, W, 3)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img_uint8, heatmap, alpha=0.4):
    """original_img_uint8: HxWx3 uint8 RGB image."""
    heatmap = cv2.resize(heatmap, (original_img_uint8.shape[1], original_img_uint8.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(heatmap_color * alpha + original_img_uint8 * (1 - alpha))
    return overlaid
