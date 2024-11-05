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
    return joblib.load("model.pkl")

model = load_model()

# App Title
st.title("Sales Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    feature1 = st.sidebar.slider("Feature 1", 0, 100, 50)  # Adjust the feature ranges as needed
    feature2 = st.sidebar.slider("Feature 2", 0, 100, 50)  # Adjust the feature ranges as needed
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

        # Display basic statistics about the data
        st.write("Basic Statistics:")
        st.write(data.describe())

        # Example: Display sales by category
        st.subheader("Sales by Category")
        sales_by_category = data.groupby('category')['sales'].sum().reset_index()
        st.bar_chart(sales_by_category.set_index('category'))

        # Example: Display sales over time
        st.subheader("Sales Over Time")
        data['order_date'] = pd.to_datetime(data['order_date'])  # Ensure the date column is in datetime format
        sales_over_time = data.groupby(data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
        sales_over_time['order_date'] = sales_over_time['order_date'].dt.to_timestamp()  # Convert back to timestamp for plotting
        st.line_chart(sales_over_time.set_index('order_date')['sales'])

    except Exception as e:
        st.error(f"Error loading CSV: {e}")  # Show error if CSV cannot be read

# Footer
st.write("Created by [Your Name]")
