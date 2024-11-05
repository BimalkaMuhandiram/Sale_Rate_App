import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# Set the app title
st.title("Machine Learning Application")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

# Sidebar for user inputs
st.sidebar.header("User Inputs")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
uploaded_audio = st.sidebar.file_uploader("Upload an Audio File", type=["mp3", "wav"])
uploaded_video = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "mov"])

# Progress and status updates
with st.sidebar:
    st.write("App Status:")
    status = st.empty()

# Helper function to preprocess and predict image
def predict_image(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    status.text("Making Prediction...")
    preds = model.predict(img_array)
    return decode_predictions(preds, top=3)[0]

# Main container for displaying the app
with st.container():
    st.write("### Upload and Classify an Image")

    # Display uploaded image
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        img = Image.open(uploaded_image)

        # Input widgets and predict button
        with st.container():
            st.write("#### Choose Confidence Threshold")
            threshold = st.slider("Threshold", 0.0, 1.0, 0.2)
            user_name = st.text_input("Enter Your Name", placeholder="Type your name here")

            if st.button("Classify Image"):
                # Show loading progress
                status.text(f"Hello {user_name}, processing image...")
                progress_bar = st.progress(0)
                
                # Simulate progress for visualization
                for i in range(10):
                    progress_bar.progress(i + 1)
                    time.sleep(0.1)

                # Make predictions
                predictions = predict_image(img, model)

                # Filter predictions based on threshold
                filtered_preds = [(name, prob) for (_, name, prob) in predictions if prob >= threshold]
                
                # Display predictions
                st.write("### Prediction Results")
                if filtered_preds:
                    for label, score in filtered_preds:
                        st.write(f"**{label}:** {score*100:.2f}%")
                else:
                    st.write("No predictions met the confidence threshold.")

                # Clear the progress bar and update status
                progress_bar.empty()
                status.text("Prediction Complete")

    else:
        st.info("Awaiting Image Upload...")

    # Display uploaded audio
    if uploaded_audio is not None:
        st.write("### Uploaded Audio File")
        st.audio(uploaded_audio, format="audio/mp3")
    
    # Display uploaded video
    if uploaded_video is not None:
        st.write("### Uploaded Video File")
        st.video(uploaded_video)

# Display model performance graph (simulated for demonstration)
st.write("### Model Prediction Confidence Visualization")

# Example graph (bar chart) of prediction confidence
if uploaded_image is not None:
    labels, scores = zip(*[(name, prob * 100) for (_, name, prob) in predict_image(img, model)])
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top Predictions with Confidence Scores")
    st.pyplot(fig)
