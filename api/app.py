from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Keras Functional error fix karne ke liye (Agar zaroorat ho)
try:
    import tf_keras
except ImportError:
    tf_keras = None

app = FastAPI()

# 1. CORSMiddleware ka import aur setup theek kiya
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load logic
MODEL_PATH = os.path.join(os.getcwd(), "amd_model.keras")

# 2. Model Loading Logic
try:
    if tf_keras:
        model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Defaulting to standard tf.keras if tf_keras fails
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

    # 3. Logic and Syntax Fix (Chance calculation)
    # Note: "Nromal" typo theek karke "Normal" kiya gaya hai
    if pred_value >= 0.5:
        result = "Negative"
        chance = pred_value * 100
    else:
        result = "Positive"
        chance = (1 - pred_value) * 100

    # 4. Result Formatting
    return {
        "Detected": result,
        "Chance": f"{chance:.2f}%"
    }

