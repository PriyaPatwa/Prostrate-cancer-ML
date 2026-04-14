import tensorflow as tf
from tensorflow.keras.models import load_model


def convert_h5_to_tflite(keras_model_path: str = "prostate_cancer_model.h5", tflite_model_path: str = "model.tflite") -> None:
    """Convert a saved Keras H5 model to TensorFlow Lite format."""
    model = load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Convert to TFLite. On Spaces, this is fast and keeps the model lightweight.
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Converted '{keras_model_path}' to '{tflite_model_path}'.")


if __name__ == "__main__":
    convert_h5_to_tflite()
