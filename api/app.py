from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Keras Functional error fix karne ke liye
import tf_keras 

app = FastAPI()

# Model load logic
MODEL_PATH = os.path.join(os.getcwd(), "amd_model.keras")

# Error handling ke saath model load karein
try:
    # tf_keras ka use karke purane models load hote hain
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Loading via tf_keras... Error was: {e}")
    model = tf_keras.models.load_model(MODEL_PATH, compile=False)

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Image processing
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    prediction = model.predict(img_array)
    pred_value = float(prediction[0][0])

    if pred_value >= 0.5:
        result = f"Nromal Chances {pred_value:.2f}"
    else:
        result = f"Age Related Macular Degeneration Chances ({1 - pred_value:.2f})"

    return result

