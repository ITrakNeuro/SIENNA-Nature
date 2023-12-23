
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (
    Input, InputLayer, Dense, Flatten, Conv2D, Activation,
    BatchNormalization, MaxPooling2D, AveragePooling2D, Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, cv2, random, pywt, scipy
from skimage import io, exposure
import shutil, pathlib
from sklearn.utils import shuffle
from keras.utils import to_categorical
import imutils
from os import listdir
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd

import tensorflow as tf
import keras
import numpy as np
import matplotlib
import pandas as pd
import cv2
import skimage
import sklearn
import imutils
import imblearn

print(f"Tensorflow=={tf.__version__}")
print(f"Keras=={keras.__version__}")
print(f"NumPy=={np.__version__}")
print(f"Matplotlib=={matplotlib.__version__}")
print(f"Pandas=={pd.__version__}")
print(f"OpenCV=={cv2.__version__}")
print(f"Scikit-image=={skimage.__version__}")
print(f"Scikit-learn=={sklearn.__version__}")
print(f"Imutils=={imutils.__version__}")
print(f"Imbalanced-learn=={imblearn.__version__}")

from google.colab import drive
drive.mount('/content/drive')

#path to tumor slices
tumor_train_flair = '/content/drive/MyDrive/classification1/tumor/'
#path to non-tumor slices
nontumor_train_flair = '/content/drive/MyDrive/classification1/no tumor/'

def PREMO(image, clip_limit=1.5, gamma=3.0):
    """
    Pixel Redistribution Enhancement, Masking, Optimization(PREMO) for MRI intensity equalization
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

def process_patient(patient_id, tumor_train_flair, nontumor_train_flair, models):
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
# Load models
models = {
    "Model 1": load_model("/content/SIENNA pre-trained.h5")
}

all_results = {}
for patient_id in range(1, 11):
    patient_name = f"Patient-GBM{patient_id}"
    all_results[patient_name] = process_patient(patient_name, tumor_train_flair, nontumor_train_flair, models)

# Convert results to a DataFrame for display
results_df = pd.DataFrame(all_results)
# Set display options
pd.set_option('display.max_colwidth', None)  # or use a large number instead of None
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display the DataFrame
print(results_df)
