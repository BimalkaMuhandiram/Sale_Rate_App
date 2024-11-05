import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import joblib  # For loading the RandomForestRegressor model

# Load the pre-trained model based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    elif model_name == "VGG16":
        model = tf.keras.applications.VGG16(weights="imagenet")
    elif model_name == "RandomForestRegressor":
        model = joblib.load("random_forest_regressor_model.pkl")  # Load the RandomForestRegressor model
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
        img = image.resize((224, 224))  # Resizing image to 224x224 for the models
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        if model_name in ["MobileNetV2", "VGG16"]:
            # Model-specific preprocessing
            if model_name == "VGG16":
                img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            else:
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Make a prediction
            preds = model.predict(img_array)

            # Decoding predictions based on the selected model
            if model_name == "VGG16":
                decoded_preds = tf.keras.applications.vgg16.decode_predictions(preds, top=5)[0]
            else:
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

            # Display the results
            st.write("Top predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                st.write(f"{i+1}. {label}: {score * 100:.2f}%")

            # Progress Bar
            st.progress(100)

            # Visualization (Bar Chart, Pie Chart, etc.) for image classification predictions
            labels = [label for (_, label, _) in decoded_preds]
            scores = [score for (_, _, score) in decoded_preds]

            # Bar chart of predictions
            fig, ax = plt.subplots()
            ax.barh(labels, scores, color='blue')
            ax.set_xlabel('Confidence Score')
            ax.set_title('Top-5 Predictions (Bar Chart)')
            st.pyplot(fig)

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        elif model_name == "RandomForestRegressor":
            # Flatten the image array and prepare for Random Forest regression
            img_flat = img_array.flatten().reshape(1, -1)  # Flatten the image for regression model

            # Make prediction with Random Forest
            prediction = model.predict(img_flat)
            st.write(f"Predicted value: {prediction[0]:.2f}")  # Display the prediction

            # Optionally, provide a visualization if relevant for regression predictions
            st.line_chart([0, prediction[0]], use_container_width=True)

        # Sidebar to select number of top-N predictions for classification models
        if model_name in ["MobileNetV2", "VGG16"]:
            top_n = st.sidebar.slider("Select top N predictions to display", 1, 10, 5)

            # Decoding the top-N predictions
            if model_name == "VGG16":
                decoded_preds = tf.keras.applications.vgg16.decode_predictions(preds, top=top_n)[0]
            else:
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top_n)[0]

            # Display the top-N predictions in text
            st.write(f"Top-{top_n} predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                st.write(f"{i+1}. {label}: {score * 100:.2f}%")

            # Visualization for top-N predictions (Bar Chart, Line Chart, etc.)
            labels = [label for (_, label, _) in decoded_preds]
            scores = [score for (_, _, score) in decoded_preds]

            # Bar chart of the top-N predictions
            fig, ax = plt.subplots()
            ax.barh(labels, scores, color='green')
            ax.set_xlabel('Confidence Score')
            ax.set_title(f'Top-{top_n} Predictions (Bar Chart)')
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
