import gdown
import pickle
import os
import numpy as np
import cv2
import streamlit as st
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# File IDs from Google Drive
model_file_id = "1poefBrc3_Nem0IuvN_6y0gvnVcECfSZC"  # Model file ID
scaler_file_id = "1cF2DhAn9C4EpGcJ6SoMsBJFNHc-kHT15"  # Scaler file ID

# File paths
model_path = "model.h5"
scaler_path = "scaler.pkl"

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# Download files if they don’t exist locally
if not os.path.exists(model_path):
    st.write("Downloading model file...")
    download_file_from_google_drive(model_file_id, model_path)
if not os.path.exists(scaler_path):
    st.write("Downloading scaler file...")
    download_file_from_google_drive(scaler_file_id, scaler_path)

# Load the model and scaler
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Train Wheel Defect Classification")

# Upload an image file for classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image with Canny Edge Detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 150, 250)

    flattened_image = edges.flatten().reshape(1, -1)

    # Apply PCA if the model was trained with PCA (adjust n_components to match model)
    pca = PCA(n_components=255)  # Reducing to 255 features
    flattened_image_pca = pca.fit_transform(flattened_image)

    # Apply the scaler to normalize the features
    scaled_features = scaler.transform(flattened_image_pca)

    # Predict using the model
    prediction = model.predict(scaled_features)

    # Display the prediction result
    if prediction == 1:
        st.write("Prediction: Defective")
    else:
        st.write("Prediction: Not Defective")
