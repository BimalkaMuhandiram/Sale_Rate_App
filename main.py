import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# Set up the main title and a brief description
st.title("Machine Learning Application 🎉")
st.write(
    """
    Welcome! This app lets you upload an image, audio, or video file and get insights from an image classifier.
    You can adjust the confidence threshold, and we'll show you the top predictions with a confidence chart.
    """
)

# Load and cache the pre-trained model to prevent reloading
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# Sidebar for file uploads and user inputs
st.sidebar.header("Upload Your Files Here 📂")
uploaded_image = st.sidebar.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
uploaded_audio = st.sidebar.file_uploader("Upload an Audio File (MP3, WAV)", type=["mp3", "wav"])
uploaded_video = st.sidebar.file_uploader("Upload a Video File (MP4, MOV)", type=["mp4", "mov"])

# Sidebar for additional settings
st.sidebar.header("Settings ⚙️")
user_name = st.sidebar.text_input("Your Name", placeholder="Enter your name")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2)

# Display status updates in the sidebar
status_display = st.sidebar.empty()

# Image Prediction Function
def predict_image(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    status_display.text("Analyzing the image... 📊")
    preds = model.predict(img_array)
    return decode_predictions(preds, top=3)[0]

# Main content: Image upload and prediction section
st.header("1. Image Classification 🖼️")

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Your Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_image)

    # Button for predicting image
    if st.button("Classify Image"):
        progress_bar = st.progress(0)
        
        # Simulate loading time for user experience
        for percent in range(10):
            progress_bar.progress((percent + 1) * 10)
            time.sleep(0.1)

        # Predict and filter results
        predictions = predict_image(img, model)
        filtered_preds = [(label, score) for (_, label, score) in predictions if score >= threshold]

        # Display predictions
        st.subheader("Top Predictions 🎯")
        if filtered_preds:
            for label, score in filtered_preds:
                st.write(f"- **{label.capitalize()}**: {score*100:.2f}% confidence")
        else:
            st.write("No predictions met the confidence threshold.")
        
        # Update status and remove progress bar
        progress_bar.empty()
        status_display.text("Prediction Complete! ✅")

        # Show confidence chart
        st.subheader("Prediction Confidence Chart 📊")
        labels, scores = zip(*[(label, score * 100) for label, score in filtered_preds])
        fig, ax = plt.subplots()
        ax.barh(labels, scores, color='skyblue')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Confidence Levels of Top Predictions")
        st.pyplot(fig)

else:
    st.info("Upload an image to begin classification.")

# Audio file section
st.header("2. Audio Player 🎵")
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    st.write("Enjoy your uploaded audio file above! 🎧")

# Video file section
st.header("3. Video Player 🎥")
if uploaded_video is not None:
    st.video(uploaded_video)
    st.write("Here's your uploaded video file! 🎬")

# Closing message
st.write("---")
st.write("Thank you for using this app! We hope you had fun exploring different media and predictions. 😊")
