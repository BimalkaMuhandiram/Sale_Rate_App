import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import joblib
from skimage.feature import greycomatrix, greycoprops

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")
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
        # Convert to array and normalize
        img_array = np.array(image) / 255.0  # Normalize the image

        # Feature Extraction
        features = []

        # 1. Mean of RGB channels
        mean_rgb = img_array.mean(axis=(0, 1))  # Mean for R, G, B
        features.extend(mean_rgb)

        # 2. Standard deviation of RGB channels
        std_rgb = img_array.std(axis=(0, 1))  # Std for R, G, B
        features.extend(std_rgb)

        # 3. Add other features as necessary to reach 17 features
        # For instance, histograms or other statistics:
        hist_red, _ = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))
        hist_green, _ = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))
        hist_blue, _ = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))

        features.extend(hist_red)
        features.extend(hist_green)
        features.extend(hist_blue)

        # Ensure the number of features matches what your model expects
        if len(features) != 17:
            st.error(f"Extracted {len(features)} features, but expected 17. Please check the feature extraction logic.")
        else:
            features = np.array(features).reshape(1, -1)  # Reshape for model input

            # Debug: Print the shape of the input
            st.write(f"Input shape for prediction: {features.shape}")

            # Make prediction with Random Forest
            try:
                prediction = model.predict(features)
                st.write(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction
            except ValueError as e:
                st.error(f"Prediction error: {e}")
                st.write("Ensure that the input shape matches the model's expected shape.")
