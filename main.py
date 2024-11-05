import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image  # Import the Image class from PIL

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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility='collapsed')

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)  # Open the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")  # Show error if image cannot be opened

# Data visualization section
st.header("Data Visualization")
try:
    data = pd.read_csv('your_dataset.csv')  # Adjust with actual dataset path
    st.write(data.head())
    
    # Example of displaying a graph
    plt.figure(figsize=(10, 5))
    sns.histplot(data['YourColumn'], bins=30)  # Adjust 'YourColumn' as needed
    st.pyplot(plt)
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the correct file path.")

# Footer
st.write("Created by [Your Name]")
