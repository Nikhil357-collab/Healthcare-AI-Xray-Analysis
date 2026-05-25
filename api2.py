from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(title="AI Medical Image Analysis API")

# Load model
model = tf.keras.models.load_model(
    "models/multidisease_model.keras"
)

# Class labels
class_names = [
    "COVID",
    "NORMAL",
    "PNEUMONIA",
    "TB"
]

@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = np.array(image)

    image = cv2.resize(image, (224,224))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return {
        "prediction": class_names[predicted_class],
        "confidence": confidence
    }