# -*- coding: utf-8 -*-

import tensorflow as tf
import tf2onnx

# Load your Keras model
model = tf.keras.models.load_model('license_plate_recognition.keras')
model.output_names=['output']

# Define input signature matching your model's input shape
input_signature = [tf.TensorSpec((None, 256, 65, 1), tf.float32, name="input_plate")]

# Convert using tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="license_plate_recognition.onnx"
)
