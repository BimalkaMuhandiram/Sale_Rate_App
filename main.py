import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import joblib

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")
    # Additional model loading can be added here as needed
    return model

st.title("Enhanced Image Classification App")
st.sidebar.title("Upload and Enhance your image")

# Sidebar for model selection
model_name = st.sidebar.selectbox("Select a pre-trained model", ("RandomForestRegressor",))

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
        img = image.resize((64, 64))  # Resize to 64x64 pixels (change as per your training parameters)
        img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
        img_flat = img_array.flatten().reshape(1, -1)  # Flatten to 1D array

        # Debug: Print the shape of the input
        st.write(f"Input shape for prediction: {img_flat.shape}")

        # Make prediction with Random Forest
        try:
            prediction = model.predict(img_flat)
            st.write(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction
        except ValueError as e:
            st.error(f"Prediction error: {e}")
            st.write("Ensure that the input shape matches the model's expected shape.")

