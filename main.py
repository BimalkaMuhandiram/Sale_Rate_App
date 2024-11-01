import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import io
import base64

# Set the page layout
st.set_page_config(layout="wide")
st.title("Prediction Dashboard")

# Sidebar header
st.sidebar.header("Image and Prediction Dashboard")

# Allow user to select a date
selected_date = st.sidebar.date_input("Select a date", pd.to_datetime("today"))
year = selected_date.year

# Allow user to upload wallpaper image
uploaded_wallpaper = st.sidebar.file_uploader("Upload a wallpaper image", type=["jpg", "jpeg", "png"])

# Allow user to select a color theme
color_theme = st.sidebar.selectbox("Select a color theme", ['Blues', 'Reds', 'Greens'])

# Set background image if uploaded
if uploaded_wallpaper is not None:
    image = Image.open(uploaded_wallpaper)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    background_image = f'url(data:image/png;base64,{img_str})'
else:
    background_image = ''

# Apply selected color theme to the app's main color scheme using CSS
if color_theme == 'Blues':
    primary_color = "#1f77b4"
    secondary_color = "#aec7e8"
elif color_theme == 'Reds':
    primary_color = "#d62728"
    secondary_color = "#ff9896"
elif color_theme == 'Greens':
    primary_color = "#2ca02c"
    secondary_color = "#98df8a"

# Inject custom CSS for dynamic color themes and background image
st.markdown(f"""
    <style>
    .main {{
        background-color: {secondary_color};
        background-image: {background_image};
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: black;  /* Default text color for better visibility */
    }}
    div.stButton > button:first-child {{
        background-color: {primary_color};
        color: white;
    }}
    div.stButton > button:hover {{
        background-color: #00ff00;
        color: red;
    }}
    header {{
        background-color: {primary_color};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    elif model_name == "VGG16":
        model = tf.keras.applications.VGG16(weights="imagenet")
    else:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

# Sidebar for model selection
model_name = st.sidebar.selectbox("Select a pre-trained model", ("MobileNetV2", "VGG16"))
model = load_model(model_name)

# Sidebar to upload image for classification
uploaded_file = st.sidebar.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

# Image enhancement options
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB format
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
        img = image.resize((224, 224))  # Resizing image to 224x224 for the models
        img = np.array(img) / 255.0      # Normalize the image
        img = np.expand_dims(img, axis=0)

        # Model-specific preprocessing
        if model_name == "VGG16":
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Make a prediction
        preds = model.predict(img)

        # Sidebar to select number of top-N predictions
        top_n = st.sidebar.slider("Select top N predictions to display", 1, 10, 5)
        if model_name == "VGG16":
            decoded_preds = tf.keras.applications.vgg16.decode_predictions(preds, top=top_n)[0]
        else:
            decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top_n)[0]

        # Display the results
        st.write("Top predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
            st.write(f"{i+1}. {label}: {score * 100:.2f}%")

        # Visualizations
        labels = [label for (_, label, _) in decoded_preds]
        scores = [score for (_, _, score) in decoded_preds]

        # Bar chart of the top-N predictions
        fig, ax = plt.subplots()
        ax.barh(labels, scores, color='blue')
        ax.set_xlabel('Confidence Score')
        ax.set_title(f'Top-{top_n} Predictions (Bar Chart)')
        st.pyplot(fig)

        # Pie chart for better visualization of scores
        fig, ax = plt.subplots()
        ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        st.pyplot(fig)

        # Line Chart: Prediction scores vs. Labels
        fig, ax = plt.subplots()
        ax.plot(labels, scores, marker='o', linestyle='-', color='orange')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Confidence Scores')
        ax.set_title(f'Top-{top_n} Predictions (Line Chart)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Histogram of confidence scores
        fig, ax = plt.subplots()
        ax.hist(scores, bins=5, color='purple', alpha=0.7)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top-{top_n} Predictions (Histogram)')
        st.pyplot(fig)

        # Scatter Plot: Confidence distribution
        fig, ax = plt.subplots()
        ax.scatter(labels, scores, color='red', s=100)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Confidence Scores')
        ax.set_title(f'Top-{top_n} Predictions (Scatter Plot)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

st.subheader(f'Loan Eligibility Prediction for the Year {year}')
# Additional functionality for loan eligibility prediction can be added here
