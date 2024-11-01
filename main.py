import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the model
model = joblib.load('random_forest_regressor_model.pkl')

# Expected columns in the model
expected_columns = ['Order Date', 'Ship Date', 'Ship Mode', 'Segment', 'Country', 'City', 'State', 
                    'Postal Code', 'Region', 'Category', 'Sub-Category', 'Sales']

# Preprocessing function
def preprocess_data(input_data):
    # Select only the columns that the model was trained on
    input_data = input_data[expected_columns]

    # Encoding categorical columns
    categorical_columns = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    le = LabelEncoder()
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = le.fit_transform(input_data[col])
    
    # Fill missing values if any
    input_data.fillna(input_data.mean(), inplace=True)
    return input_data

# Streamlit app
st.title("Sales Rate Prediction App")

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
sales = st.number_input("Sales", min_value=0.0)

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
        'Sub-Category': [sub_category],
        'Sales': [sales]
    })

    # Preprocess input data
    preprocessed_data = preprocess_data(input_data)
    
    # Predict sales rate
    try:
        prediction = model.predict(preprocessed_data)
        # Display the prediction result
        st.write(f"Predicted Sales Rate: {prediction[0]}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
