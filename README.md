# Prostate Cancer Risk Assessment Pipeline

A complete machine learning pipeline for automated prostate cancer severity classification from MRI DICOM images using deep learning and NLP-based Gleason score extraction.

## 🚀 Features

- **Complete Pipeline**: End-to-end processing from raw medical data to CNN predictions
- **Medical Imaging**: Proper DICOM processing with windowing, rescale, and enhancement
- **NLP Risk Assessment**: Automated Gleason score extraction and 3-class risk classification
- **CNN Model**: TensorFlow/Keras convolutional neural network for multi-class classification
- **Interactive Demo**: Streamlit web app for visualization and testing

## 📊 Pipeline Overview

1. **Data Loading**: Excel metadata with patient information and biopsy reports
2. **NLP Risk Classification**: Extract Gleason scores and assign risk levels (Low/Medium/High)
3. **DICOM Processing**: Convert medical images with proper preprocessing
4. **Data Preparation**: Image-label mapping and stratified train/test splitting
5. **CNN Training**: Deep learning model with 3-class classification and class weighting
6. **Web Demo**: Interactive Streamlit application for risk assessment

## 🏥 Medical Data Processing

- **DICOM Handling**: Proper rescale slope/intercept application
- **Windowing**: Medical image window center/width optimization
- **Enhancement**: Histogram equalization for better contrast
- **Standardization**: 128×128 pixel resizing for CNN input

## 🎯 Model Performance

**UPDATED: Fixed Data Leakage with Patient-wise Splitting**

- **Test Accuracy**: 58.69% (realistic evaluation - no data leakage)
- **Previous (Leaked)**: ~98% (artificially high due to patient overlap)
- **Architecture**: 3 Conv2D layers + MaxPooling + Dense (128) + Softmax (3 classes)
- **Dataset**: 18,598 images from 53 patients
- **Patient-wise Split**: 42 train patients, 11 test patients (no overlap)
- **Classes**: 3-class risk assessment (Low/Medium/High risk)

### Class Performance (Patient-wise Split):
- **Low Risk (0)**: 18% precision, 8% recall (316 test samples)
- **Medium Risk (1)**: 38% precision, 22% recall (1,102 test samples)  
- **High Risk (2)**: 65% precision, 82% recall (2,419 test samples)

### Data Leakage Fix:
- **Problem**: Images from same patients in both train/test sets
- **Solution**: Patient-wise splitting ensures no patient overlap
- **Impact**: Accuracy dropped from ~98% to 58.69% (honest evaluation)

## 🖥️ Interactive Web App

The Streamlit web application provides:

- **Overview**: Pipeline explanation and key metrics
- **Data Analysis**: Statistics and risk class distribution
- **Model Performance**: Multi-class accuracy metrics and confusion matrix
- **Sample Images**: Processed medical images gallery
- **Risk Assessment**: Upload images for automated 3-class risk prediction

### Risk Categories

- **Low Risk (0)**: Gleason score ≤6 or benign/no tumor
- **Medium Risk (1)**: Gleason score =7
- **High Risk (2)**: Gleason score ≥8 or aggressive cancer

### Running the Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training pipeline (optional - model already trained)
python create_better_model.py

# Test the model accuracy
python test_model_accuracy.py

# Create sample folders for testing
python copy_test_samples.py

# Launch the web app
streamlit run app.py
```

The app will be available at: http://localhost:8501

## ☁️ Hosting the App

This repo is built as a Streamlit web app and can be hosted locally or in the cloud.

### Local hosting
1. Activate your Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` in your browser.

### Streamlit Community Cloud
Best for quick deployment with no server management.

1. Push the repo to GitHub.
2. Create a Streamlit Community Cloud app.
3. Connect the GitHub repository.
4. Set the app file to `app.py`.
5. Deploy.

> Important: `prostate_cancer_model.h5` (37MB) is tracked with Git LFS. `metadata.xlsx`, and `processed_images/` must be available to the deployed app. If these files are too large for GitHub, use Git LFS or external storage and update the app accordingly.

### Docker or cloud VM hosting
For a custom server or Docker deployment:

1. Copy the repo to the server.
2. Install Python and create a virtual environment.
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Streamlit with a public host address:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
5. Open the external server URL on port `8501`.

### Hosting notes
- The app requires the trained model file (tracked with Git LFS) and metadata to run.
- The `processed_images/` folder (30,903 files, ~1-2GB) is excluded from Git due to size limits. For full functionality:
  - Create a ZIP: `Compress-Archive -Path processed_images -DestinationPath processed_images.zip` (PowerShell) or `zip -r processed_images.zip processed_images` (bash).
  - Upload `processed_images.zip` to GitHub Releases (create a release on the repo).
  - Users can download and extract it to `processed_images/` folder.
  - Alternatively, for demo purposes, the app can use sample folders if available.
- If you do not want to store large files in Git, add them to `.gitignore` or use Git LFS.
- For production, make sure the deployed environment has enough memory for TensorFlow and the model.

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.15+
- Streamlit 1.28+
- Medical imaging libraries (pydicom, opencv)
- Data processing (pandas, numpy, scikit-learn)

## 📁 Project Structure

```
prostate_diagnosis/
├── app.py                    # Streamlit web application
├── create_better_model.py    # Main training script for 3-class model
├── test_model_accuracy.py    # Model validation script
├── copy_test_samples.py      # Create sample folders for testing
├── metadata.xlsx            # Patient metadata and biopsy reports
├── requirements.txt         # Python dependencies
├── processed_images/        # All converted PNG images (30,903 files)
├── low_risk_samples/        # Sample low-risk images for testing
├── medium_risk_samples/     # Sample medium-risk images for testing
├── high_risk_samples/       # Sample high-risk images for testing
├── prostate_cancer_model.h5 # Trained 3-class CNN model
└── README.md               # This file
```

## 🔬 Technical Details

### Data Statistics
- **Total Patients**: 54
- **Labeled Patients**: 53 (2 low risk, 16 medium risk, 35 high risk)
- **DICOM Images**: 30,903
- **Training Samples**: 14,761
- **Test Samples**: 3,837
- **Risk Classes**: 3 (Low/Medium/High)

### Model Architecture
```
Input (128×128×1 grayscale)
├── Conv2D (32 filters, 3×3) + ReLU
├── MaxPooling2D (2×2)
├── Conv2D (64 filters, 3×3) + ReLU
├── MaxPooling2D (2×2)
├── Conv2D (128 filters, 3×3) + ReLU
├── MaxPooling2D (2×2)
├── Flatten
├── Dense (128) + ReLU
├── Dropout (0.5)
├── Dense (3) + Softmax
└── Multi-class Output
```

### Confusion Matrix (Test Set)
```
[[  24   24  268]
 [  44  244  814]
 [  69  366 1984]]
```

## 🎓 Usage for Education/Demo

This project demonstrates:
- Medical image processing techniques
- NLP for healthcare text analysis
- Deep learning for medical diagnosis
- End-to-end ML pipeline development
- Interactive data science applications

Perfect for:
- Medical AI education
- Portfolio demonstrations
- Research presentations
- Healthcare technology showcases

## ⚠️ Medical Disclaimer

This is a demonstration project for educational purposes. The model and pipeline are not intended for clinical use or medical diagnosis. Always consult qualified medical professionals for healthcare decisions.

## 📄 License

Educational and demonstration purposes only.

String labels such as `benign`, `normal`, `negative`, `malignant`, `cancer`, and `positive` are also accepted.

## Train

Quick smoke run:

```powershell
py train.py --data-dir . --epochs 1 --batch-size 8 --image-size 128 --max-slices-per-patient 8
```

Fuller run:

```powershell
py train.py --data-dir . --epochs 10 --batch-size 16 --image-size 160 --max-slices-per-patient 64
```

With CSV labels:

```powershell
py train.py --data-dir . --labels-csv labels.csv --patient-col patient_id --label-col label
```

## Critical Notebook Bugs Fixed

- Empty dataset crashes: the code now validates that DICOM files and labels exist before training starts.
- Incorrect label extraction: labels come from CSV when provided; the folder-name fallback is explicit and printed.
- Memory overflow: DICOM images stream from disk with `tf.data` instead of being appended to huge Python lists.
- Bad DICOM preprocessing: slices are decoded with `pydicom`, windowed/clipped, resized with OpenCV, normalized to `[0, 1]`, and shaped as `(H, W, 1)`.
- Data leakage: train/validation split is done by patient folder, not by individual slice.
- Shape mismatch: the model input shape is controlled by one config value and checked in the data loader.

