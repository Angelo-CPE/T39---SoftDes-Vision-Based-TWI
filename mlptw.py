import gdown
import pickle
import os
import numpy as np
import cv2
import streamlit as st
from sklearn import svm
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

# Directory containing test images
test_dir = 'test/'
valid_path = 'valid'  # Base directory to save classified images
flawed_dir = os.path.join(valid_path, 'flawed')
not_flawed_dir = os.path.join(valid_path, 'not_flawed')

# Create output directories for Flawed and Not Flawed if they don't exist
os.makedirs(flawed_dir, exist_ok=True)
os.makedirs(not_flawed_dir, exist_ok=True)

# Loop through each test image
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Read the test image
        image_path = os.path.join(test_dir, filename)
        image = cv2.imread(image_path)

        # Apply Canny edge detection
        edges = cv2.Canny(image, 100, 200)

        # Resize the image to 256x256 pixels
        edges_resized = cv2.resize(edges, (256, 256))

        # Flatten the image for feature extraction
        features = edges_resized.flatten().reshape(1, -1)

        # Scale the features
        features = scaler.transform(features)

        # Predict using the SVM model
        prediction = svm_model.predict(features)

        # Determine the result text and save directory
        if prediction[0] == -1:  # Assuming -1 means 'Not Flawed'
            result_text = "Not Flawed"
            save_dir = not_flawed_dir
        else:  # Assuming anything else means 'Flawed'
            result_text = "Flawed"
            save_dir = flawed_dir

        # Show the image with the result
        cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Test Image", image)
        cv2.waitKey(0)

        # Generate a sequential file name
        file_count = len(os.listdir(save_dir)) + 1
        save_path = os.path.join(save_dir, f"{result_text.lower().replace(' ', '_')}{file_count}.jpg")
        
        # Save the image to the appropriate folder
        cv2.imwrite(save_path, image)
        print(f"Image saved to {save_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()
