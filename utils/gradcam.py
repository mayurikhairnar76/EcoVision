"""Grad-CAM: highlights the regions of the last conv layer that most increased the predicted class score."""
import cv2
import numpy as np
import tensorflow as tf


def make_gradcam(model, batch, class_idx, layer_name="last_conv"):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))            # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_heatmap(pil_img, cam, alpha=0.45):
    img = np.asarray(pil_img.convert("RGB"))
    heat = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]   # BGR -> RGB
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
