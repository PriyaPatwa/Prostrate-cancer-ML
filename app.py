import cv2
import numpy as np
import gradio as gr
from functools import lru_cache
import tensorflow as tf

MODEL_PATH = "model.tflite"
CLASS_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}


def sanitize_image(image: np.ndarray) -> np.ndarray:
    """Convert input image to a single-channel grayscale numpy array."""
    if image is None:
        raise ValueError("No image was provided.")

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if image.ndim == 2:
        return image

    raise ValueError(f"Unsupported image shape: {image.shape}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize, normalize, and add batch/channel dims for TFLite inference."""
    gray = sanitize_image(image)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))


@lru_cache(maxsize=1)
def load_tflite_model(model_path: str = MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def predict_risk(image: np.ndarray):
    processed = preprocess_image(image)
    interpreter, input_details, output_details = load_tflite_model()

    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])
    probs = np.squeeze(raw_output).astype(np.float32)

    if probs.ndim == 0:
        probs = np.array([float(probs)], dtype=np.float32)

    if probs.sum() > 0:
        probs = probs / np.sum(probs)

    predicted_index = int(np.argmax(probs))
    predicted_label = CLASS_LABELS.get(predicted_index, str(predicted_index))

    probability_map = {
        "Low Risk": float(probs[0]) if len(probs) > 0 else 0.0,
        "Medium Risk": float(probs[1]) if len(probs) > 1 else 0.0,
        "High Risk": float(probs[2]) if len(probs) > 2 else 0.0,
    }

    return predicted_label, probability_map


def build_interface():
    with gr.Blocks(title="Prostate Cancer Risk Prediction") as demo:
        gr.Markdown("""
        # Prostate Cancer Risk Prediction

        Upload a grayscale prostate MRI image (PNG or RGB) and the app will return a 3-class risk prediction:
        - **Low Risk**
        - **Medium Risk**
        - **High Risk**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Upload MRI Image", image_mode="RGB")
                predict_button = gr.Button("Predict Risk")

            with gr.Column(scale=1):
                prediction_output = gr.Textbox(label="Predicted Risk Level", interactive=False)
                probability_output = gr.Label(label="Prediction Probabilities")

        predict_button.click(
            fn=predict_risk,
            inputs=image_input,
            outputs=[prediction_output, probability_output],
        )

        gr.Markdown("""
        ### Notes
        - The model uses a TensorFlow Lite interpreter for fast, lightweight inference.
        - Images are resized to **128×128**, converted to grayscale, and normalized to **[0, 1]**.
        - This app is optimized for deployment on Hugging Face Spaces.
        """)

    return demo


def main():
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
