import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the model
model = joblib.load('random_forest_regressor_model.pkl')

# Expected columns for the model, excluding 'Sales' as it is the prediction target
expected_columns = ['Order Date', 'Ship Date', 'Ship Mode', 'Segment', 'Country', 'City', 'State', 
                    'Postal Code', 'Region', 'Category', 'Sub-Category']

# Updated preprocessing function
def preprocess_data(input_data):
    # Select only the columns that the model was trained on
    input_data = input_data[expected_columns]

    # Encoding categorical columns
    categorical_columns = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    le = LabelEncoder()
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = le.fit_transform(input_data[col])

    # Handle missing values
    # Fill numeric columns with 0 or the mean
    numeric_cols = input_data.select_dtypes(include=['float64', 'int64']).columns
    input_data[numeric_cols] = input_data[numeric_cols].fillna(0)

    # Fill categorical columns with 'Unknown'
    categorical_cols = input_data.select_dtypes(include=['object']).columns
    input_data[categorical_cols] = input_data[categorical_cols].fillna("Unknown")

    return input_data

# Streamlit app
st.title("Sales Prediction App")

st.header("Input Features")
# Create input fields for each feature
order_id = st.text_input("Order ID")
order_date = st.date_input("Order Date")
ship_date = st.date_input("Ship Date")
ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
customer_id = st.text_input("Customer ID")
customer_name = st.text_input("Customer Name")
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
country = st.text_input("Country", value="United States")
city = st.text_input("City")
state = st.text_input("State")
postal_code = st.number_input("Postal Code", min_value=0)
region = st.selectbox("Region", ["East", "West", "Central", "South"])
product_id = st.text_input("Product ID")
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
sub_category = st.text_input("Sub-Category")
product_name = st.text_input("Product Name")

# When the user clicks "Predict", preprocess and make a prediction
if st.button("Predict"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Order Date': [order_date],
        'Ship Date': [ship_date],
        'Ship Mode': [ship_mode],
        'Segment': [segment],
        'Country': [country],
        'City': [city],
        'State': [state],
        'Postal Code': [postal_code],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [sub_category]
    })

    # Preprocess input data
    preprocessed_data = preprocess_data(input_data)
    
    # Predict sales
    try:
        prediction = model.predict(preprocessed_data)
        # Display the prediction result
        st.write(f"Predicted Sales: ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
