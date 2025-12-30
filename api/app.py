import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Model Load karein
MODEL_PATH = "amd_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 2. Prediction Function
def predict(image):
    if image is None:
        return "Please upload an image."
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_value = float(prediction[0][0])

    if pred_value >= 0.5:
        return f"AMD Detected ({pred_value:.2f})"
    else:
        return f"Normal ({1 - pred_value:.2f})"

# 3. Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AMD Detection",
    description="Upload a retinal fundus image to check for AMD."
)

# 4. Render ke liye Launch settings
if __name__ == "__main__":
    # Render automatically PORT environment variable provide karta hai
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
