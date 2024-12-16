import cv2
import streamlit as st
import numpy as np

# Function to detect impurities
def detect_impurities(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    impurity_count = len(contours)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image, impurity_count

# Streamlit App
st.title("Real-Time Daal Impurity Detection")

# Start video capture
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)  # Capture from default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access camera.")
            break

        # Detect impurities in each frame
        processed_frame, impurity_count = detect_impurities(frame)

        # Show live frame with impurity detection
        FRAME_WINDOW.image(processed_frame, channels="BGR")
        st.write(f"Number of impurities detected: {impurity_count}")

    cap.release()
