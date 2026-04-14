import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
IMAGE_SIZE = 128
DATA_DIR = Path(".")
METADATA_FILE = "metadata.xlsx"
IMAGES_DIR = Path("processed_images")
IMAGES_DIR.mkdir(exist_ok=True)

def load_and_preprocess_metadata(metadata_file):
    """STEP 1: Load and preprocess metadata"""
    print("Loading metadata...")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    df = pd.read_excel(metadata_file)

    # Set "TCIA ID" as index if it exists
    if "TCIA ID" in df.columns:
        df = df.set_index("TCIA ID")
    else:
        df = df.set_index(df.columns[0])

    # Transpose so rows become patients
    df = df.T

    # Reset index and create PatientID column
    df = df.reset_index()
    df = df.rename(columns={"index": "PatientID"})

    print(f"Loaded metadata for {len(df)} patients")
    return df

def create_labels_from_nlp(df):
    """STEP 2: Create labels using NLP rules"""
    print("Creating labels using NLP rules...")

    if "Path report biopsy" not in df.columns:
        raise ValueError("Column 'Path report biopsy' not found in metadata")

    labels = []
    valid_indices = []

    for idx, row in df.iterrows():
        text = str(row["Path report biopsy"]).lower()
        if "adenocarcinoma" in text or "gleason" in text:
            labels.append(1)  # cancer
            valid_indices.append(idx)
        elif "no tumor identified" in text:
            labels.append(0)  # non-cancer
            valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("No valid labels could be extracted from biopsy reports")

    df_filtered = df.loc[valid_indices].copy()
    df_filtered["label"] = labels

    print(f"Created labels for {len(df_filtered)} patients ({sum(labels)} cancer, {len(labels) - sum(labels)} non-cancer)")
    return df_filtered

def convert_dicom_to_png(data_dir, images_dir):
    """STEP 3: Convert DICOM images to PNG"""
    print("Converting DICOM images to PNG...")

    dicom_files = []
    for ext in ['*.dcm', '*.dicom']:
        dicom_files.extend(data_dir.rglob(ext))

    print(f"Found {len(dicom_files)} DICOM files")

    patient_images = {}

    for dicom_path in dicom_files:
        try:
            path_parts = dicom_path.parts
            patient_id = None
            for part in path_parts:
                if "ProstateDx" in part:
                    patient_id = part
                    break

            if patient_id is None:
                continue

            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            image = image * slope + intercept

            if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
                image = image.max() - image

            center = getattr(ds, "WindowCenter", None)
            width = getattr(ds, "WindowWidth", None)
            if center is not None and width is not None:
                if isinstance(center, list):
                    center = center[0]
                if isinstance(width, list):
                    width = width[0]
                low = center - width / 2.0
                high = center + width / 2.0
                image = np.clip(image, low, high)
            else:
                low, high = np.percentile(image, [1.0, 99.0])
                image = np.clip(image, low, high)

            if high > low:
                image = (image - low) / (high - low) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

            image = cv2.equalizeHist(image)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            if patient_id not in patient_images:
                patient_images[patient_id] = 0

            index = patient_images[patient_id]
            png_filename = f"{patient_id}_{index}.png"
            png_path = images_dir / png_filename
            cv2.imwrite(str(png_path), image)

            patient_images[patient_id] += 1

        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
            continue

    print(f"Converted images for {len(patient_images)} patients")
    return patient_images

def map_images_to_labels(images_dir, df):
    """STEP 4: Map images to labels"""
    print("Mapping images to labels...")

    X = []
    y = []
    patient_ids = []

    for png_file in images_dir.glob("*.png"):
        filename = png_file.stem
        patient_id = "_".join(filename.split("_")[:-1])

        if patient_id in df["PatientID"].values:
            label = df[df["PatientID"] == patient_id]["label"].iloc[0]

            image = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                X.append(image)
                y.append(label)
                patient_ids.append(patient_id)

    if not X:
        raise ValueError("No images could be mapped to labels")

    X = np.array(X)
    y = np.array(y)

    print(f"Mapped {len(X)} images to labels")
    return X, y, patient_ids

def preprocess_data(X, y):
    """STEP 5: Preprocess data"""
    print("Preprocessing data...")

    if len(X) < 2:
        raise ValueError("Need at least 2 samples for train/test split")

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes for classification")

    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test

def build_cnn_model():
    """STEP 6: Build CNN model"""
    print("Building CNN model...")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """STEP 7: Train and evaluate"""
    print("Training model...")

    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

    print("Evaluating model...")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save the trained model
    model.save('prostate_cancer_model.h5')
    print("Model saved as 'prostate_cancer_model.h5'")

    return history, accuracy, cm

def main():
    # Step 1: Load and preprocess metadata
    df = load_and_preprocess_metadata(METADATA_FILE)

    # Step 2: Create labels
    df_labeled = create_labels_from_nlp(df)

    # Step 3: Convert DICOM to PNG
    patient_images = convert_dicom_to_png(DATA_DIR, IMAGES_DIR)

    # Step 4: Map images to labels
    X, y, patient_ids = map_images_to_labels(IMAGES_DIR, df_labeled)

    # Step 5: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 6: Build model
    model = build_cnn_model()

    # Step 7: Train and evaluate
    history, accuracy, cm = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    # Save the trained model
    model.save("prostate_cancer_model.h5")
    print("Model saved as 'prostate_cancer_model.h5'")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()