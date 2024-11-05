import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time

# Load the pre-trained Random Forest model
@st.cache_resource
def load_model():
    # Make sure to have your model saved as 'model.pkl' in the same directory
    return joblib.load("model.pkl")

model = load_model()

# Set up the main title and a brief description
st.title("Random Forest Regression Application 📊")
st.write(
    """
    Welcome to the Random Forest Regression app! This app allows you to upload a CSV file of data and make predictions using a trained Random Forest model.
    You can input parameters manually or upload a dataset to predict outcomes.
    """
)

# Sidebar for file uploads
st.sidebar.header("Upload Your Data 📂")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for predictions", type="csv")

# User inputs for manual prediction
st.sidebar.header("Input Parameters for Prediction ⚙️")
input_params = {}

# Assuming the model requires the following features, customize according to your model's input features
input_params['feature1'] = st.sidebar.number_input("Feature 1", min_value=0.0)
input_params['feature2'] = st.sidebar.number_input("Feature 2", min_value=0.0)
input_params['feature3'] = st.sidebar.number_input("Feature 3", min_value=0.0)
input_params['feature4'] = st.sidebar.number_input("Feature 4", min_value=0.0)

# Button for manual prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[input_params['feature1'], input_params['feature2'], input_params['feature3'], input_params['feature4']]])
    prediction = model.predict(input_data)

    # Display the prediction
    st.subheader("Predicted Output 🔮")
    st.write(f"The predicted value is: **{prediction[0]:.2f}**")

# Display uploaded file results
if uploaded_file is not None:
    st.header("Uploaded Data Preview 📈")
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Button to make predictions on the uploaded data
    if st.button("Predict on Uploaded Data"):
        if data.shape[1] < 5:
            st.error("Uploaded data must have at least 4 features for prediction.")
        else:
            # Ensure only the required features are selected for predictions
            features = data.iloc[:, :4]  # Adjust according to your model's input features
            predictions = model.predict(features)

            # Display predictions
            data['Predicted Value'] = predictions
            st.subheader("Predictions for Uploaded Data 📊")
            st.write(data)

            # Visualize predictions
            fig, ax = plt.subplots()
            ax.scatter(data.index, data.iloc[:, -1], label='Actual Values', color='blue')
            ax.scatter(data.index, predictions, label='Predicted Values', color='red')
            ax.set_title("Actual vs Predicted Values")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            ax.legend()
            st.pyplot(fig)

else:
    st.info("Upload a CSV file to make predictions.")

# Closing message
st.write("---")
st.write("Thank you for using this app! We hope you found it useful for making predictions. 😊")
