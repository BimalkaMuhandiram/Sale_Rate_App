import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the model
model = joblib.load('random_forest_regressor_model.pkl')

# Preprocessing function
def preprocess_data(input_data):
    # Encoding categorical columns
    ordinal_columns = ['Ship Mode', 'Segment', 'Region']
    le = LabelEncoder()
    for col in ordinal_columns:
        if col in input_data.columns:
            input_data[col] = le.fit_transform(input_data[col])

    # Convert dates to datetime format (assuming model accepts date-related features)
    input_data['Order Date'] = pd.to_datetime(input_data['Order Date'], errors='coerce')
    input_data['Ship Date'] = pd.to_datetime(input_data['Ship Date'], errors='coerce')

    # Convert dates to numerical features (days since a reference date, e.g., 1970-01-01)
    input_data['Order Date'] = (input_data['Order Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
    input_data['Ship Date'] = (input_data['Ship Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

    # Fill any remaining missing values
    input_data.fillna(0, inplace=True)

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
country = st.text_input("Country", value="United States")  # Defaulted to United States if needed
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
        'Order ID': [order_id],
        'Order Date': [order_date],
        'Ship Date': [ship_date],
        'Ship Mode': [ship_mode],
        'Customer ID': [customer_id],
        'Customer Name': [customer_name],
        'Segment': [segment],
        'Country': [country],
        'City': [city],
        'State': [state],
        'Postal Code': [postal_code],
        'Region': [region],
        'Product ID': [product_id],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Product Name': [product_name]
    })

    # Preprocess input data
    preprocessed_data = preprocess_data(input_data)

    # Predict sales
    prediction = model.predict(preprocessed_data)
    
    # Display the prediction result
    st.write(f"Predicted Sales: ${prediction[0]:,.2f}")
