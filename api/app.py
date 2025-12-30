import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Model load karne se pehle ye check karein
MODEL_PATH = "amd_model.keras"

try:
    # Compile=False memory bachane mein madad karta hai
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Prediction
    prediction = model.predict(img_array)
    pred_value = float(prediction[0][0])

    if pred_value >= 0.5:
        return f"AMD Detected (Confidence: {pred_value:.2f})"
    else:
        return f"Normal (Confidence: {1 - pred_value:.2f})"

# 2. Gradio Interface setup
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AMD Detection System",
    description="Upload a Retinal Fundus image to check for AMD."
)

# 3. Render Port Binding (Ye sabse zaroori hai)
if __name__ == "__main__":
    # Render $PORT environment variable provide karta hai
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}...")
    interface.launch(server_name="0.0.0.0", server_port=port)
