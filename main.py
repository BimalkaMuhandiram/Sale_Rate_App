# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Sales Prediction App", page_icon=":shopping_bags:", layout="wide")

# Title of the app
st.title("Sales Prediction")

# Function to load machine learning toolkit
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    with open(relative_path, "rb") as file:
        return pickle.load(file)

# Function to load dataset
@st.cache()
def load_data(relative_path):
    train_data = pd.read_csv(relative_path, index_col=0)
    train_data["ship_date"] = pd.to_datetime(train_data["ship_date"]).dt.date
    train_data["order_date"] = pd.to_datetime(train_data["order_date"]).dt.date
    return train_data

# Function to get date features
def get_date_features(df, date):
    df['date'] = pd.to_datetime(df[date])
    date_features = {
        'month': df['date'].dt.month,
        'day_of_month': df['date'].dt.day,
        'day_of_year': df['date'].dt.dayofyear,
        'week_of_year': df['date'].dt.isocalendar().week,
        'day_of_week': df['date'].dt.dayofweek,
        'year': df['date'].dt.year,
        'is_weekend': np.where(df['date'].dt.dayofweek > 4, 1, 0),
        'is_month_start': df['date'].dt.is_month_start.astype(int),
        'is_month_end': df['date'].dt.is_month_end.astype(int),
        'quarter': df['date'].dt.quarter,
        'is_quarter_start': df['date'].dt.is_quarter_start.astype(int),
        'is_quarter_end': df['date'].dt.is_quarter_end.astype(int),
        'is_year_start': df['date'].dt.is_year_start.astype(int),
        'is_year_end': df['date'].dt.is_year_end.astype(int)
    }
    return pd.DataFrame(date_features)

# Load dataset and machine learning toolkit
train_data = load_data("train.csv")
loaded_toolkit = load_ml_toolkit("/Users/emmanythedon/Documents/PostBAP_ASSESSMENT/ML_items")

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state["results"] = []

# Load ML model components
ml_model = loaded_toolkit["model"]
encode = loaded_toolkit["encoder"]

# Display header and image
st.sidebar.header("Sales Prediction for Corporation Favorita")
image = Image.open("/Users/emmanythedon/Documents/SALES FORECASTING/sales.jpg")
st.image(image, width=500)

# Sidebar for column information
if st.sidebar.checkbox("Click here to know more about your columns"):
    st.sidebar.markdown("""
        - **row_id**: Unique identifier for each row.
        - **order_id**: Unique identifier for each order.
        - **order_date**: Date when the order was placed.
        - **ship_date**: Date when the order was shipped.
        - **ship_mode**: The mode of shipping.
        - **customer_id**: Unique identifier for the customer.
        - **customer_name**: Name of the customer.
        - **segment**: Market segment of the customer.
        - **country**: Country where the order was placed.
        - **city**: City where the order was placed.
        - **state**: State where the order was placed.
        - **postal_code**: Postal code of the delivery address.
        - **region**: Region where the order was placed.
        - **product_id**: Unique identifier for the product.
        - **category**: Category of the product.
        - **sub-category**: Sub-category of the product.
        - **product_name**: Name of the product.
        - **sales**: Total sales value for the order.
    """)

# Preview the dataset
if st.checkbox("Preview the dataset"):
    st.write(train_data.head())

# Input form for predictions
with st.form(key="input_form", clear_on_submit=True):
    order_date = st.date_input("Order Date:", min_value=train_data["order_date"].min())
    ship_date = st.date_input("Ship Date:", min_value=train_data["ship_date"].min())
    ship_mode = st.selectbox("Ship Mode:", options=train_data['ship_mode'].unique())
    customer_id = st.selectbox("Customer ID:", options=train_data['customer_id'].unique())
    customer_name = st.text_input("Customer Name:")
    segment = st.selectbox("Segment:", options=train_data['segment'].unique())
    country = st.selectbox("Country:", options=train_data['country'].unique())
    city = st.selectbox("City:", options=train_data['city'].unique())
    state = st.selectbox("State:", options=train_data['state'].unique())
    postal_code = st.text_input("Postal Code:")
    region = st.selectbox("Region:", options=train_data['region'].unique())
    product_id = st.selectbox("Product ID:", options=train_data['product_id'].unique())
    category = st.selectbox("Category:", options=train_data['category'].unique())
    sub_category = st.selectbox("Sub-category:", options=train_data['sub-category'].unique())

    submitted = st.form_submit_button("Submit")

    if submitted:
        # Prepare input data
        input_dict = {
            "order_date": [order_date],
            "ship_date": [ship_date],
            "ship_mode": [ship_mode],
            "customer_id": [customer_id],
            "customer_name": [customer_name],
            "segment": [segment],
            "country": [country],
            "city": [city],
            "state": [state],
            "postal_code": [postal_code],
            "region": [region],
            "product_id": [product_id],
            "category": [category],
            "sub-category": [sub_category],
        }
        input_data = pd.DataFrame.from_dict(input_dict)

        # Process date features
        df_processed = pd.concat([get_date_features(input_data, "order_date"), 
                                   get_date_features(input_data, "ship_date")], axis=1)

        # Encode categorical variables
        encoded_categoricals = encode.transform(input_data[categoricals])
        df_processed = df_processed.join(pd.DataFrame(encoded_categoricals, columns=encode.get_feature_names_out().tolist()))

        # Make predictions
        predicted_sales = ml_model.predict(df_processed)

        # Display results
        st.success(f"**Predicted sales**: USD {predicted_sales[0]:.2f}")

        # Store results for review
        st.session_state["results"].append(input_data)
        previous_results = pd.concat(st.session_state["results"])

        # Expander for previous predictions
        with st.expander("Review Previous Predictions"):
            st.write(previous_results)
