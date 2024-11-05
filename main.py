import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import joblib

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")
        return model
    else:
        st.error("Unsupported model type selected.")
        return None

# Image enhancement function
def enhance_image(image, brightness, contrast):
    """Enhances the image based on brightness and contrast values."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    return image

# Feature extraction function
def extract_features(image):
    """Extracts features from the image for prediction."""
    img_array = np.array(image) / 255.0  # Normalize the image
    features = []

    # Mean and Standard deviation of RGB channels
    mean_rgb = img_array.mean(axis=(0, 1))
    std_rgb = img_array.std(axis=(0, 1))
    features.extend(mean_rgb)
    features.extend(std_rgb)

    # Histogram of RGB channels
    hist_red, _ = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))
    hist_green, _ = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))
    hist_blue, _ = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))
    features.extend(hist_red)
    features.extend(hist_green)
    features.extend(hist_blue)

    return np.array(features).reshape(1, -1)

# Main app function
def main():
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
        enhanced_image = enhance_image(image, brightness, contrast)
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

        # Preprocess the image for model prediction
        st.write("Classifying...")
        with st.spinner("Processing..."):
            features = extract_features(enhanced_image)

            # Debug: Print the shape of the input
            st.write(f"Input shape for prediction: {features.shape}")

            # Make prediction with Random Forest
            try:
                prediction = model.predict(features)
                st.success(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction
            except ValueError as e:
                st.error(f"Prediction error: {e}")
                st.write("Ensure that the input shape matches the model's expected shape.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
