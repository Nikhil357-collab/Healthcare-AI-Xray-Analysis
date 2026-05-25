import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(
    "../models/best_model.keras"
)

def generate_heatmap(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224,224))

    image = image / 255.0

    image_array = np.expand_dims(image, axis=0)

    last_conv_layer = model.get_layer(
        "conv5_block16_concat"
    )

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_array)

        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    heatmap = tf.reduce_mean(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(heatmap[0], 0)

    heatmap /= np.max(heatmap)

    plt.matshow(heatmap)

    plt.title("Grad-CAM Heatmap")

    plt.show()