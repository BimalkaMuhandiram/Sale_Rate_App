import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")
    return model

# Function to extract features from the image
def extract_features(image):
    """Extracts features from the image for prediction."""
    img_array = np.array(image) / 255.0  # Normalize the image
    features = []

    # Mean of RGB channels
    mean_rgb = img_array.mean(axis=(0, 1))  # Mean for R, G, B
    features.extend(mean_rgb)  # 3 features

    # Standard deviation of RGB channels
    std_rgb = img_array.std(axis=(0, 1))  # Std for R, G, B
    features.extend(std_rgb)  # 3 features

    # Extract histogram features
    hist_red, _ = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))
    hist_green, _ = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))
    hist_blue, _ = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))

    # Use the first 4 histogram bins from each channel
    features.extend(hist_red[:4])  # First 4 bins from Red
    features.extend(hist_green[:4])  # First 4 bins from Green
    features.extend(hist_blue[:4])  # First 4 bins from Blue

    return np.array(features).reshape(1, -1)

# Streamlit App Layout
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
        features = extract_features(image)  # Extract features from the enhanced image

        # Debug: Print the shape of the input
        st.write(f"Input shape for prediction: {features.shape}")

        # Make prediction with Random Forest
        try:
            prediction = model.predict(features)
            st.write(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction
            
            # Visualization: Bar Chart for Features
            labels = ['Mean Red', 'Mean Green', 'Mean Blue',
                      'Std Red', 'Std Green', 'Std Blue'] + \
                     [f'Hist Red {i+1}' for i in range(4)] + \
                     [f'Hist Green {i+1}' for i in range(4)] + \
                     [f'Hist Blue {i+1}' for i in range(4)]
            feature_values = features.flatten()

            # Bar chart for feature values
            fig, ax = plt.subplots()
            ax.barh(labels, feature_values, color='skyblue')
            ax.set_xlabel('Feature Value')
            ax.set_title('Extracted Features from Image')
            st.pyplot(fig)

            # Histogram of RGB channel values
            fig, ax = plt.subplots()
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                ax.hist(np.array(image)[:, :, i].flatten(), bins=32, color=color, alpha=0.5, label=f'{color.capitalize()} Channel')
            ax.set_title('RGB Histogram')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"Prediction error: {e}")
            st.write("Ensure that the input shape matches the model's expected shape.")
