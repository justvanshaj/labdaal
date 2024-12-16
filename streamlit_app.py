import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to detect impurities in an image
def detect_impurities(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to segment impurities (this is a simple example, adjust as needed)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (impurities)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the total number of pixels and number of impurity pixels
    total_pixels = gray.size
    impurity_pixels = 0
    
    for contour in contours:
        # Count the pixels of each impurity (contour area)
        impurity_pixels += cv2.contourArea(contour)
    
    # Calculate impurity percentage
    impurity_percentage = (impurity_pixels / total_pixels) * 100
    
    return impurity_percentage, thresh

# Streamlit app
st.title("Daal Impurity Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image of daal", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image using PIL (Python Imaging Library)
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert image from RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect impurities
    impurity_percentage, processed_image = detect_impurities(image_bgr)
    
    # Display the processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    
    # Display the impurity percentage
    st.write(f"Percentage of impurities: {impurity_percentage:.2f}%")
