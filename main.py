from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from PIL import Image

# Define a custom loss class to handle unexpected keyword arguments
class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, ignore_class=None, **kwargs):
        super().__init__(from_logits=from_logits)

# Define custom objects
custom_objects = {'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy}

# Load model.h5 with custom objects
model = keras.models.load_model("currency_sense_model2.h5", custom_objects=custom_objects)
# Labels contained within model.h5
label = ['10000', '5000', '50000']

app = Flask(__name__)

# Function to make predictions on input images
def predict_label(img):
    i = np.asarray(img) / 255.0
    # Ensure input size matches the model's expected input size
    i = i.reshape(1, 224, 224, 3)
    pred = model.predict(i)
    result = label[np.argmax(pred)]
    return result

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Service is running"})

@app.route("/predict", methods=["POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224), Image.NEAREST)
        pred_img = predict_label(img)
        return jsonify({"prediction": pred_img})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
