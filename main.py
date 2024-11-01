import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the model
model = joblib.load('random_forest_regressor_model.pkl')

# Preprocessing functions
def preprocess_data(input_data):
    # Standardize column names
    input_data.columns = input_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\\w\\s]', '')

    # Encoding nominal and ordinal columns
    ordinal_columns = ['ship_mode', 'segment', 'region']
    nominal_columns = ['order_id', 'order_date', 'ship_date', 'customer_id', 'customer_name', 'city', 'state', 'product_id', 'sub-category', 'product_name']

    le = LabelEncoder()
    for col in ordinal_columns:
        if col in input_data.columns:
            input_data[col] = le.fit_transform(input_data[col])
    
    input_data = pd.get_dummies(input_data, columns=nominal_columns, drop_first=True)

    # Fill missing values
    input_data.fillna(input_data.mean(), inplace=True)

    return input_data

# Streamlit app
st.title("Sales Rate Prediction App")

# User inputs
st.header("Input Features")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    preprocessed_data = preprocess_data(user_data)
    
    # Predict sales rate
    if st.button("Predict"):
        predictions = model.predict(preprocessed_data)
        user_data['Predicted Sales Rate'] = predictions
        st.write("Prediction Results:")
        st.write(user_data[['order_id', 'Predicted Sales Rate']])  # Customize as needed
else:
    st.write("Please upload a CSV file to make predictions.")
