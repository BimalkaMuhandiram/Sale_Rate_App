import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_regressor_model.pkl")

model = load_model()

# App Title
st.title("Sales Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    feature1 = st.sidebar.slider("Feature 1", 0, 100, 50) 
    feature2 = st.sidebar.slider("Feature 2", 0, 100, 50)  
    data = {'Feature 1': feature1,
            'Feature 2': feature2}
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Progress bar for prediction
if st.button("Predict"):
    with st.spinner("Making prediction..."):
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]:.2f}")

# Media Upload Section
st.header("Upload Your Media")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility='collapsed')
uploaded_csv = st.file_uploader("Upload your CSV file...", type=["csv"], label_visibility='collapsed')

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)  # Open the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")  # Show error if image cannot be opened

if uploaded_csv is not None:
    try:
        data = pd.read_csv(uploaded_csv)  # Read the uploaded CSV file
        st.write("Data from CSV:")
        st.dataframe(data)  # Display the dataframe in the app

        # Example of displaying a graph based on the CSV data
        if 'YourColumn' in data.columns:  # Adjust this according to your CSV
            plt.figure(figsize=(10, 5))
            sns.histplot(data['YourColumn'], bins=30)  # Adjust 'YourColumn' as needed
            st.pyplot(plt)
        else:
            st.warning("Column 'YourColumn' not found in the uploaded CSV.")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")  # Show error if CSV cannot be read
