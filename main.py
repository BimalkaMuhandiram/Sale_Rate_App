import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import joblib

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    elif model_name == "VGG16":
        model = tf.keras.applications.VGG16(weights="imagenet")
    elif model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")
    else:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

st.title("Enhanced Image Classification App")
st.sidebar.title("Upload and Enhance your image")

# Sidebar for model selection
model_name = st.sidebar.selectbox("Select a pre-trained model", ("MobileNetV2", "VGG16", "RandomForestRegressor"))

model = load_model(model_name)

# Sidebar to upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image enhancement options
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Image enhancement sliders
    st.sidebar.subheader("Image Enhancements")
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)

    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    st.image(image, caption="Enhanced Image", use_column_width=True)

    # Preprocess the image for model prediction
    st.write("Classifying...")
    with st.spinner("Processing..."):
        img = image.resize((64, 64))  # Resize image to match training dimensions (modify as needed)
        img_array = np.array(img) / 255.0  # Normalize the image
        img_flat = img_array.flatten().reshape(1, -1)  # Flatten the image for the regression model

        if model_name in ["MobileNetV2", "VGG16"]:
            # For classification models, we need specific preprocessing
            if model_name == "VGG16":
                img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            else:
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Make a prediction for classification models
            preds = model.predict(np.expand_dims(img_array, axis=0))
            if model_name == "VGG16":
                decoded_preds = tf.keras.applications.vgg16.decode_predictions(preds, top=5)[0]
            else:
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

            # Display the classification results
            st.write("Top predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                st.write(f"{i+1}. {label}: {score * 100:.2f}%")

            # Progress Bar
            st.progress(100)

        elif model_name == "RandomForestRegressor":
            # Make prediction with Random Forest
            prediction = model.predict(img_flat)
            st.write(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction

        # Visualization for prediction results can be added if necessary
