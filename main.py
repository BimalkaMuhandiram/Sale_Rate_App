import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# App Title
st.title("Sales Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
def user_input_features():
    feature1 = st.sidebar.slider("Feature 1", 0, 100, 50)  # Adjust range as needed
    feature2 = st.sidebar.slider("Feature 2", 0, 100, 50)  # Adjust range as needed
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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Data visualization section
st.header("Data Visualization")
data = pd.read_csv('your_dataset.csv')  # Adjust with actual dataset path
st.write(data.head())

# Example of displaying a graph
plt.figure(figsize=(10, 5))
sns.histplot(data['YourColumn'], bins=30)
st.pyplot(plt)

# Additional features and graphs as necessary

# Footer
st.write("Created by [Your Name]")
