# ----- Core Libraries -----
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# ----- Configuration Section -----
# Paths
tumor_train_flair = 'D:\Research\Research\classification1-20220830T050713Z-001\classification2\\tumor'
nontumor_train_flair = 'D:\Research\Research\classification1-20220830T050713Z-001\classification2\\no tumor'
model_path = "SIENNA pre-trained.h5"

# Image processing parameters
image_size = (240, 240)
interpolation_method = cv2.INTER_CUBIC

# ----- Function Definitions -----
def PREMO(image, clip_limit=1.5, gamma=3.0):
    """
    Pixel Redistribution Enhancement, Masking, Optimization (PREMO) for MRI intensity equalization.
    Args:
        image: numpy array
        clip_limit: pixel threshold to clip
        gamma: gamma correction parameter
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    #otsu's thresholding
    thresh_val, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    #noise removal using morhological operations
    mask = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)
    #cumulative distribution function (CDF)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    #mapping intensities to CDF
    mapping = np.interp(np.arange(256), np.arange(256), cdf)
    out = np.interp(gray.flatten(), np.arange(256), mapping).reshape(gray.shape)
    out = np.clip(out, 0, 255)
    if clip_limit > 0:
        hist, _ = np.histogram(out.flatten(), bins=256, range=(0, 255))
        excess = np.sum(hist) - clip_limit * gray.size
        if excess > 0:
            limit_val = np.argmax(hist.cumsum() > excess)
            out = np.clip(out, 0, limit_val)
    # Gamma Correction
    out = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255
    out = np.power(out / 255.0, gamma) * 255.0
    #binary mask
    out = np.where(mask == 0, gray, out)
    #intensity range 0-255
    out = np.clip(out, 0, 255)
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return out

def load_models():
    """ Load the pre-trained models. """
    try:
        print(model_path)
        models = {
            "Model 1": tf.keras.models.load_model(model_path)
        }
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def process_patient(patient_id, tumor_train_flair, nontumor_train_flair, models):
    """
    Process images for a given patient, predict using the models, and calculate metrics.
    Args:
        patient_id: Identifier for the patient.
        tumor_train_flair: Path to tumor images.
        nontumor_train_flair: Path to non-tumor images.
        models: Dictionary of loaded models.
    """
    x_test, y_test, z_test = [], [], []
    # Process tumor images
    for i in os.listdir(tumor_train_flair):
        if patient_id in i:
            labeltumour = 1 if 'GBM' in i else 2 if 'MET' in i else 0
            for j in os.listdir(os.path.join(tumor_train_flair, i)):
                if j in ['Ax-Flair', 'Series-AxFLAIR']:
                    for k in os.listdir(os.path.join(tumor_train_flair, i, j)):
                        if k != 'Zaxis':
                            image_path = os.path.join(tumor_train_flair, i, j, k)
                            img = cv2.imread(image_path)
                            img_resized = cv2.resize(img, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
                            intensity_eq = PREMO(img_resized)
                            img_processed = intensity_eq / 255
                            x_test.append(img_processed)
                            y_test.append(labeltumour)
                            z_test.append(os.path.splitext(k)[0])
    # Process non-tumor images
    for i in os.listdir(nontumor_train_flair):
        if patient_id in i:
            for j in os.listdir(os.path.join(nontumor_train_flair, i)):
                if j in ['Ax-Flair', 'Series-AxFLAIR']:
                    for k in os.listdir(os.path.join(nontumor_train_flair, i, j)):
                        if k != 'Zaxis':
                            image_path = os.path.join(nontumor_train_flair, i, j, k)
                            img = cv2.imread(image_path)
                            img_resized = cv2.resize(img, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
                            intensity_eq = PREMO(img_resized)
                            img_processed = intensity_eq / 255
                            x_test.append(img_processed)
                            y_test.append(0)
                            z_test.append(os.path.splitext(k)[0])


    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # Check if x_test is empty
    if x_test.size == 0:
        print(f"No images found for {patient_id}. Skipping.")
        return None
    y_test = to_categorical(y_test, num_classes=3)
    true_classes = np.argmax(y_test, axis=1)

    patient_results = {}
    for model_name, model in models.items():
        predictions = model.predict(x_test)
        predicted_classes = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        accuracy = accuracy_score(true_classes, predicted_classes)
        f1 = f1_score(true_classes, predicted_classes, average='weighted')
        patient_results[model_name] = {
            "Confusion Matrix": conf_matrix,
            "Accuracy": accuracy,
            "F1 Score": f1
        }
    return patient_results


# ----- Main Execution -----
if __name__ == "__main__":
    # Load models
    models = load_models()
    if models is None:
        print("Failed to load models. Exiting.")
        exit(1)

    # Process each patient
    all_results = {}
    for patient_id in range(1, 11):
        patient_name = f"Patient-GBM{patient_id}"
        all_results[patient_name] = process_patient(patient_name, tumor_train_flair, nontumor_train_flair, models)

    # Convert results to a DataFrame for display
    results_df = pd.DataFrame(all_results)
    # Set display options
    pd.set_option('display.max_colwidth', None) 
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Display the DataFrame
    print(results_df)
