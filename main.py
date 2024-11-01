import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

# Preprocessing function
def preprocess_data(input_data):
    # Normalize column names to lowercase
    input_data.columns = input_data.columns.str.lower()

    # Convert dates to datetime objects
    input_data['order_date'] = pd.to_datetime(input_data['order_date'])
    input_data['ship_date'] = pd.to_datetime(input_data['ship_date'])

    # Encode categorical columns
    categorical_columns = ['ship_mode', 'segment', 'region', 'category', 'sub-category']
    le = LabelEncoder()
    
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna("Unknown")  # Fill missing values
            input_data[col] = le.fit_transform(input_data[col])  # Label encode categorical variables

    # Handle missing numeric values
    numeric_cols = input_data.select_dtypes(include=['float64', 'int64']).columns
    input_data[numeric_cols] = input_data[numeric_cols].fillna(0)  # Fill numeric missing values with 0

    return input_data

# Streamlit app
st.title("Sales Prediction App")

# Input fields for features
order_date = st.date_input("Order Date")
ship_date = st.date_input("Ship Date")
ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
city = st.text_input("City")
state = st.text_input("State")
postal_code = st.number_input("Postal Code", min_value=0)
region = st.selectbox("Region", ["East", "West", "Central", "South"])
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
sub_category = st.text_input("Sub-Category")
product_id = st.text_input("Product ID")
product_name = st.text_input("Product Name")

# When the user clicks "Predict", preprocess and make a prediction
if st.button("Predict"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'order_date': [order_date],
        'ship_date': [ship_date],
        'ship_mode': [ship_mode],
        'segment': [segment],
        'city': [city],
        'state': [state],
        'postal_code': [postal_code],
        'region': [region],
        'category': [category],
        'sub-category': [sub_category],
        'product_id': [product_id],
        'product_name': [product_name]
    })

    # Preprocess input data
    preprocessed_data = preprocess_data(input_data)
    
    # Make a prediction
    try:
        prediction = model.predict(preprocessed_data)
        # Display the prediction result
        st.write(f"Predicted Sales: ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
